"""r14 (2026-10-13)：雷達候選池每日歸檔 (chip/predict/radar_archive.py → site/data/radar_pool.jsonl) 單元測試。不連網、不需要 LightGBM。

    python tests/test_radar_archive.py

(a) conc10 / conc10_stk 數學 (合成 MI_INDEX 列：零成交 / 缺值 / ETF / 帶字母代號 / |pct|>11% / 列數不足)；
(b) pool_block 形狀 (只列評到分的、srank 空號、欄位順序與 cols 一致、四捨五入、整數壓縮、pick 旗標、th_q92 有才有、可 JSON)；
(c) scan() 回傳 pool (假 market_snapshot / twii_hist / valuation / score_one)：conc10 用全部 MI 列不只宇宙；pool_block 壞掉時 scan 其他輸出完全一樣；
(d) build()：pool 不進 out['scan'] / LOCAL，交給 out['_pools'] (live / backfill 標記)，serialize.clean 會丟掉 _pools；
(e) 歸檔：line_from 驗證、同日去重 (有列 > 0 列、live > backfill、asof 新者勝)、CAP、carry (Pages ∪ 備份 ∪ 本機、冪等、Pages 失敗保留)、縮水保護、壞行容錯、緊湊格式；
(f) publish() 掛鉤：fast 只帶回、沒掃描日不追加、fetch / append 例外不外拋、大小估計；export_static 原始碼有掛鉤且包在 try 內。
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip.predict import radar_archive as RA  # noqa: E402
from chip.predict import treasure_live as L  # noqa: E402
from chip.serialize import clean  # noqa: E402


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _mi(n: int = 30, extra: dict | None = None) -> dict:
    """n 檔 4 碼普通股，成交金額 1..n 億、pct 1%。"""
    rows = {str(1100 + i): {"name": f"s{i}", "close": 10.0 + i, "value": (i + 1) * 1e8, "pct": 1.0} for i in range(n)}
    rows.update(extra or {})
    return rows


# ------------------------------------------------------------------ (a) conc10
def test_conc10_math():
    c = RA.conc(_mi(30))
    top, tot = sum(range(21, 31)), sum(range(1, 31))          # 前 10 大 = 21..30 億、合計 465 億
    assert c["conc10"] == round(top / tot, 5) == 0.54839 and c["conc10_stk"] == c["conc10"]
    assert c["mkt_n"] == 30 and c["mkt_n_stk"] == 30 and c["mkt_value"] == 465
    # 零成交 / 缺值 / 負值 / NaN / 非數字 → 去掉，結果不變
    junk = {"1901": {"value": 0, "pct": 0.0}, "1902": {"value": float("nan"), "pct": 0.0}, "1903": {"value": None, "pct": 0.0},
            "1904": {"value": -5e8, "pct": 0.0}, "1905": {"value": "x", "pct": 0.0}, "1906": None}
    c2 = RA.conc(_mi(30, junk))
    assert c2 == c, c2


def test_conc10_all_vs_stocks():
    # ETF (00 開頭)、帶字母 (特別股 / 槓桿 ETF / 受益證券)、|pct| > 11% 的股日：conc10 算、conc10_stk 不算
    big = {"0050": {"value": 50e8, "pct": 1.0}, "00631L": {"value": 40e8, "pct": 2.0}, "2881A": {"value": 35e8, "pct": 0.5},
           "01001T": {"value": 33e8, "pct": 0.0}, "6999": {"value": 31e8, "pct": 12.5}, "6998": {"value": 0.5e8, "pct": -11.0}}
    c = RA.conc(_mi(30, big))
    allv = sorted([float(i) for i in range(1, 31)] + [50, 40, 35, 33, 31, 0.5], reverse=True)
    assert c["conc10"] == round(sum(allv[:10]) / sum(allv), 5) and c["mkt_n"] == 36
    stk = sorted([float(i) for i in range(1, 31)] + [0.5], reverse=True)          # 6998 pct −11.0 剛好邊界 → 算
    assert c["conc10_stk"] == round(sum(stk[:10]) / sum(stk), 5) and c["mkt_n_stk"] == 31
    assert c["conc10"] == 0.50267 and c["conc10_stk"] == 0.54780          # 329 / 654.5、255 / 465.5
    assert c["mkt_value"] == round(sum(allv), 1)


def test_conc10_too_few_rows():
    c = RA.conc(_mi(19))
    assert c["conc10"] is None and c["conc10_stk"] is None and c["mkt_n"] == 19 and c["mkt_value"] == 190
    # 股票口徑不足 20 但全部口徑夠
    c = RA.conc(_mi(15, {f"00{900 + i}": {"value": 1e8, "pct": 0.0} for i in range(10)}))
    assert c["conc10"] is not None and c["conc10_stk"] is None and c["mkt_n_stk"] == 15
    assert RA.conc({}) == {"conc10": None, "conc10_stk": None, "mkt_value": None, "mkt_n": 0, "mkt_n_stk": 0}
    assert RA.conc(None)["mkt_n"] == 0


# ------------------------------------------------------------------ (b) pool_block
TM = {"trained_at": "2026-10-11 08:51:40", "th_A": 0.5038, "th_Aplus": 0.6246, "gate": {"m_bias20_lt": 0.0},
      "surge": {"th_top10": 0.3983, "def": {"topk": 3}}}


def _pool():
    pool = [{"code": "2330", "close": 1045.0, "pct": 1.234, "value": 312.456e8, "pbr": 6.789, "per": 25.111, "power": 40.5061},
            {"code": "2489", "close": 23.45, "pct": 9.6, "value": 0.5349e8, "pbr": float("nan"), "per": 0.0, "power": 39.0},
            {"code": "3008", "close": 2100.0, "pct": -2.0, "value": 50e8, "pbr": None, "per": float("nan"), "power": 38.123},
            {"code": "1101", "close": 30.0, "pct": 0.0, "value": 10e8, "pbr": 0.9, "per": 12.0, "power": 37.0}]
    res = {"2330": {"p": 0.61234, "ps": 0.41239, "tier": "A", "m_bias20": -1.0, "_x": {"a": 1}},
           "2489": {"p": 0.3, "ps": None, "tier": "B", "m_bias20": -1.0},
           "1101": {"p": 0.70004, "ps": 0.1, "tier": "A+", "m_bias20": -1.0}}       # 3008 沒評到分 → 不列、srank 3 空號
    tre = [{"code": "1101"}, {"code": "2330"}]
    sur = [{"code": "2330"}, {"code": "2489"}]
    day = {"mkt_close": 26001.5, "mkt_bias20": -1.234, "mkt_pct": 0.56, "pool_n": 4, "scored_n": 3, "errors": 1, "treasure": ["ignored"]}
    return pool, res, tre, sur, day


def test_pool_block_shape():
    pool, res, tre, sur, day = _pool()
    b = RA.pool_block("2026-10-12", pool, res, tre, sur, TM, _mi(30), day)
    assert b["cols"] == RA.COLS and RA.COLS[:8] == ["code", "srank", "p", "ps", "pct", "close", "value", "tier"]
    assert all(len(r) == len(b["cols"]) for r in b["rows"])
    assert [r[0] for r in b["rows"]] == ["2330", "2489", "1101"] and [r[1] for r in b["rows"]] == [1, 2, 4]
    r = dict(zip(b["cols"], b["rows"][0]))
    assert r == {"code": "2330", "srank": 1, "p": 0.6123, "ps": 0.4124, "pct": 1.23, "close": 1045, "value": 312.46, "tier": "A",
                 "pbr": 6.79, "per": 25.11, "power": 40.51, "pick": 3}, r
    assert isinstance(r["close"], int)          # 整數值壓成 int
    r2 = dict(zip(b["cols"], b["rows"][1]))
    assert r2["ps"] is None and r2["pbr"] is None and r2["per"] == 0 and r2["value"] == 0.53 and r2["pick"] == RA.PICK_SURGE and r2["close"] == 23.45
    r4 = dict(zip(b["cols"], b["rows"][2]))
    assert r4["pick"] == RA.PICK_TREASURE and r4["tier"] == "A+" and r4["p"] == 0.7
    for k, v in {"date": "2026-10-12", "model_ver": "2026-10-11 08:51:40", "th_A": 0.5038, "th_Aplus": 0.6246, "gate_lt": 0.0, "th_top10": 0.3983,
                 "surge_k": 3, "mkt_close": 26001.5, "mkt_bias20": -1.234, "mkt_pct": 0.56, "pool_n": 4, "scored_n": 3, "errors": 1,
                 "conc10": RA.conc(_mi(30))["conc10"], "mkt_n": 30}.items():
        assert b[k] == v, (k, b[k], v)
    assert "th_q92" not in b and "treasure" not in b and "_x" not in json.dumps(b)
    b2 = RA.pool_block("2026-10-12", pool, res, tre, sur, {**TM, "th_q92": 0.5555}, _mi(30), day)
    assert b2["th_q92"] == 0.5555
    json.loads(json.dumps(b, allow_nan=False))      # 沒有 NaN、可發布
    e = RA.pool_block("2026-10-12", [], {}, None, None, None, None, None)
    assert e["rows"] == [] and e["conc10"] is None and e["surge_k"] == 3 and e["model_ver"] is None


# ------------------------------------------------------------------ (c) scan() 回傳 pool
SCAN_TM = {"features": ["f"], "init": 0.0, "trees": [{"r": 0, "n": [[0, 0.0, -1, -2, 1]], "v": [0.0, 1.0]}], "th_A": 0.5, "th_Aplus": 0.64,
           "gate": {"m_bias20_lt": 0.0}, "universe": ["2489", "3008", "2330", "1101", "2603"], "trained_at": "2026-10-11 08:00:00",
           "surge": {"th_top10": 0.39, "def": {"topk": 2}}}
D = "2026-10-12"
DATES = ["2026-09-%02d" % d for d in range(1, 31)] + ["2026-10-%02d" % d for d in range(1, 13)]
P = {"2489": (0.7, "A+", 0.45), "3008": (0.55, "A", 0.2), "2330": (0.3, "B", 0.5), "1101": (0.2, "B", None)}   # 2603 → score_one None


def _scan_rows():
    rows = {c: {"name": "n" + c, "close": 100.0 + i, "open": 99.0, "high": 101.0, "low": 98.0, "volume": 1000, "value": 5e9 - i * 1e8, "pct": 1.0 + i * 0.1,
                "amp": 2.0, "per": 10.0} for i, c in enumerate(["2489", "3008", "2330", "1101", "2603"])}
    rows.update({str(4000 + i): {"name": "x", "close": 10.0, "open": 10.0, "high": 10.0, "low": 10.0, "volume": 1, "value": (i + 1) * 1e7, "pct": 0.5,
                                 "amp": 0.0, "per": 0.0} for i in range(40)})                                  # 不在宇宙
    rows["0050"] = {"name": "etf", "close": 150.0, "open": 150.0, "high": 150.0, "low": 150.0, "volume": 1, "value": 9e9, "pct": 0.3, "amp": 0.0, "per": 0.0}
    return rows


def _fake_score(code, r, d, mkf, tm):
    if code not in P:
        return None
    p, tier, ps = P[code]
    return {"p": p, "tier": tier, "ps": ps, "m_bias20": -2.0, "_x": {"f": 1.0}}


@contextlib.contextmanager
def _scan_env():
    rows = _scan_rows()
    with patched(L, "market_snapshot", lambda date=None: (D, {k: dict(v) for k, v in rows.items()})), patched(L, "twii_hist", lambda: {d: 1000.0 for d in DATES}), \
            patched(L, "valuation", lambda: {"2330": {"per": 20.0, "pbr": 5.0}}), patched(L, "score_one", _fake_score):
        yield rows


def test_scan_returns_pool():
    with _scan_env() as rows:
        sc = L.scan(SCAN_TM, D)
        with patched(RA, "pool_block", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))):
            sc_bad = L.scan(SCAN_TM, D)
    pb = sc["pool"]
    assert pb["date"] == D and pb["model_ver"] == SCAN_TM["trained_at"] and pb["th_A"] == 0.5 and pb["surge_k"] == 2 and pb["th_top10"] == 0.39
    assert pb["pool_n"] == sc["pool_n"] == 5 and pb["scored_n"] == sc["scored_n"] == 4 and pb["mkt_bias20"] == sc["mkt_bias20"] and pb["mkt_pct"] == sc["mkt_pct"]
    R = {r[0]: dict(zip(pb["cols"], r)) for r in pb["rows"]}
    assert set(R) == {"2489", "3008", "2330", "1101"}, R          # 2603 沒評到分、宇宙外與 ETF 不在候選池
    order = [r["code"] for r in L.screen(L.pool_rows(rows, SCAN_TM), {"2330": {"per": 20.0, "pbr": 5.0}}, sc["mkt_pct"]) if not r["code"].startswith("00")]
    assert {c: x["srank"] for c, x in R.items()} == {c: i + 1 for i, c in enumerate(order) if c in P}
    assert order[0] == "2603" and sorted(x["srank"] for x in R.values()) == [2, 3, 4, 5]   # 2603 power 最高但沒評到分 → 名次 1 空號
    assert R["2489"]["tier"] == "A+" and R["2489"]["p"] == 0.7 and R["1101"]["ps"] is None and R["2330"]["pbr"] == 5
    s_codes = {r["code"] for r in sc["surge"]}
    t_codes = {r["code"] for r in sc["treasure"]}
    assert all(bool(x["pick"] & RA.PICK_SURGE) == (c in s_codes) and bool(x["pick"] & RA.PICK_TREASURE) == (c in t_codes) for c, x in R.items())
    assert s_codes == {"2330", "2489"}
    # conc10：全部 MI 列 (含宇宙外 40 檔與 ETF 0050)，不是只有宇宙
    assert pb["conc10"] == RA.conc(rows)["conc10"] and pb["mkt_n"] == 46 and pb["mkt_n_stk"] == 45
    # pool_block 壞掉 → 沒有 pool，其餘輸出完全一樣 (不改既有行為)
    assert "pool" not in sc_bad
    assert json.dumps({k: v for k, v in sc.items() if k != "pool"}, sort_keys=True) == json.dumps(sc_bad, sort_keys=True)
    assert set(sc_bad) == {"date", "pool_n", "scored_n", "errors", "treasure", "surge", "pool_rule", "uni_n", "mkt_close", "mkt_bias20", "mkt_pct"}
    json.loads(json.dumps(sc["pool"], allow_nan=False))      # pool 本身沒有 NaN (scan 的 treasure[].pbr 可能是 NaN，發布時 serialize.clean 處理，不在此測)


# ------------------------------------------------------------------ (d) build()
def test_build_hands_pool_off():
    tm = dict(SCAN_TM)
    pb = {"date": D, "rows": [["2330", 1, 0.5, None, 1.0, 10, 1.0, "B", None, None, 1.0, 0]], "cols": RA.COLS, "conc10": 0.3}

    def fake_scan(tm_, date=None):
        d = date or D
        return {"date": d, "pool_n": 1, "scored_n": 1, "errors": 0, "treasure": [], "surge": [], "mkt_close": 1000.0, "mkt_bias20": -1.0, "mkt_pct": 0.1,
                "pool": {**pb, "date": d}}

    with tempfile.TemporaryDirectory() as td:
        loc = Path(td) / "treasure_live.json"
        today = dt.datetime.now(L.config.TZ).date()
        b1, b2, d_live = (today - dt.timedelta(days=3)).isoformat(), (today - dt.timedelta(days=2)).isoformat(), today.isoformat()   # backfill 視窗 = 昨天以前
        with patched(L, "LOCAL", loc), patched(L, "load_prev", lambda: {"ledger": {"treasure": [], "surge": []}, "scans": [{"date": b1}]}), \
                patched(L, "twii_hist", lambda: {d: 1000.0 for d in [b1, b2, d_live]}), patched(L, "scan", fake_scan), \
                patched(L.M, "load_json", lambda name: tm if name == "treasure_model" else None):
            out = L.build(date=D)
            assert "pool" not in out["scan"] and out["scan"]["date"] == D
            assert [p["src"] for p in out["_pools"]] == ["live"] and out["_pools"][0]["rows"] == pb["rows"]
            saved = json.loads(loc.read_text(encoding="utf-8"))
            assert "_pools" not in saved and "pool" not in saved["scan"]
            assert "_pools" not in clean(out) and "pool" not in clean(out)["scan"]
            # backfill：已掃過 (b1) 跳過、b2 補掃 → src=backfill；live 在最後
            out2 = L.build(date=d_live, backfill_days=10)
            assert [(p["date"], p["src"]) for p in out2["_pools"]] == [(b2, "backfill"), (d_live, "live")], out2["_pools"]
            assert "_pools" not in json.loads(loc.read_text(encoding="utf-8")) and all("pool" not in x for x in out2["scans"])
            # 沒有掃描日 → 沒有 pool
            with patched(L, "scan", lambda tm_, date=None: {"date": None, "error": "no market snapshot"}):
                out3 = L.build(date=D)
            assert out3["_pools"] == []


# ------------------------------------------------------------------ (e) 歸檔
def _line(ts: str, n: int = 2, src: str = "live", asof: str = "2026-10-12 15:50:00", **kw) -> dict:
    rows = [["2330", i + 1, 0.5, 0.2, 1.0, 100, 10.0, "B", 1.0, 10.0, 30.0, 0] for i in range(n)]
    return RA.line_from({"date": ts, "rows": rows, "cols": RA.COLS, "conc10": 0.31, **kw}, src=src, asof=asof)


def test_line_from():
    assert RA.line_from(None) is None and RA.line_from({"rows": []}) is None and RA.line_from({"date": "2026/10/12", "rows": []}) is None
    assert RA.line_from({"date": "2026-10-12", "rows": None}) is None
    ln = _line("2026-10-12", 3, th_A=0.5)
    assert ln["ts"] == "2026-10-12" and ln["mode"] == "scan" and ln["v"] == 1 and ln["src"] == "live" and ln["n"] == 3 and ln["cols"] == RA.COLS
    assert ln["th_A"] == 0.5 and ln["conc10"] == 0.31 and "date" not in ln and list(ln)[:5] == ["ts", "mode", "v", "src", "asof"]
    assert RA.line_from({"date": "2026-10-12 15:40:00", "rows": [], "src": "backfill"})["ts"] == "2026-10-12"
    assert RA.line_from({"date": "2026-10-12", "rows": [], "src": "backfill"})["src"] == "backfill"
    assert len(RA.line_from({"date": "2026-10-12", "rows": []})["asof"]) == 19


def test_append_dedupe_and_priority():
    with tempfile.TemporaryDirectory() as td:
        p, c = Path(td) / "radar_pool.jsonl", Path(td) / "cache" / "radar_pool.jsonl"
        assert RA.append(p, None, cache=c) == 0 and not p.exists() and not c.exists()
        assert RA.append(p, _line("2026-10-12", 2, asof="2026-10-12 15:50:00"), cache=c) == 1
        assert RA.append(p, _line("2026-10-12", 3, asof="2026-10-12 21:50:00"), cache=c) == 1          # 同日：後寫的勝
        got = RA.PL.load(p)
        assert len(got) == 1 and got[0]["n"] == 3 and RA.PL.load(c) == got
        RA.append(p, _line("2026-10-12", 5, src="backfill", asof="2026-10-13 09:00:00"), cache=c)     # live 不被 backfill 蓋
        RA.append(p, _line("2026-10-12", 0, asof="2026-10-13 10:00:00"), cache=c)                    # 有列不被 0 列蓋
        got = RA.PL.load(p)
        assert len(got) == 1 and got[0]["n"] == 3 and got[0]["src"] == "live"
        RA.append(p, [_line("2026-10-08", 1, src="backfill"), _line("2026-10-09", 0)], cache=c)      # 多行；依 ts 排序
        assert [x["ts"] for x in RA.PL.load(p)] == ["2026-10-08", "2026-10-09", "2026-10-12"]
        RA.append(p, _line("2026-10-08", 2, asof="2026-10-14 08:00:00"), cache=c)                    # backfill 之後同日 live → live
        RA.append(p, _line("2026-10-09", 4, src="backfill"), cache=c)                                 # 0 列 live 被有列 backfill 取代
        got = {x["ts"]: x for x in RA.PL.load(p)}
        assert (got["2026-10-08"]["src"], got["2026-10-08"]["n"]) == ("live", 2) and (got["2026-10-09"]["src"], got["2026-10-09"]["n"]) == ("backfill", 4)
        # 週日重訓後週一退回重掃上週五：事後的新模型 (model_ver 日期 > ts) 不蓋掉當時那一行；同模型再掃 → 新的勝
        RA.append(p, _line("2026-10-09", 6, asof="2026-10-09 15:50:00", model_ver="2026-10-04 08:00:00"), cache=c)
        RA.append(p, _line("2026-10-09", 7, asof="2026-10-12 15:50:00", model_ver="2026-10-11 08:00:00"), cache=c)
        got = {x["ts"]: x for x in RA.PL.load(p)}
        assert (got["2026-10-09"]["n"], got["2026-10-09"]["model_ver"]) == (6, "2026-10-04 08:00:00")
        RA.append(p, _line("2026-10-09", 8, asof="2026-10-09 21:50:00", model_ver="2026-10-04 08:00:00"), cache=c)
        assert {x["ts"]: x["n"] for x in RA.PL.load(p)}["2026-10-09"] == 8
        # 緊湊格式 (沒有 ", " / ": ")、一行一個掃描日、UTF-8
        txt = p.read_text(encoding="utf-8")
        assert txt.count("\n") == 3 and ", " not in txt and '": ' not in txt
        # cache=False → 不碰備份
        RA.append(p, _line("2026-10-13"), cache=False)
        assert len(RA.PL.load(p)) == 4 and len(RA.PL.load(c)) == 3


def test_cap():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.jsonl"
        for d in range(1, 8):
            RA.append(p, _line(f"2026-10-{d:02d}"), cap=5, cache=False)
        assert [x["ts"] for x in RA.PL.load(p)] == [f"2026-10-{d:02d}" for d in range(3, 8)]
        assert RA.CAP == 1250 and RA.merge([_line(f"2026-10-{d:02d}") for d in range(1, 8)], cap=3)[0]["ts"] == "2026-10-05"


def test_carry_over():
    pub = [_line("2026-10-08"), _line("2026-10-09")]
    with tempfile.TemporaryDirectory() as td:
        p, c = Path(td) / "site" / "radar_pool.jsonl", Path(td) / "cache" / "radar_pool.jsonl"
        assert RA.carry(p, ("http://pages/x", "http://raw/x"), fetch=lambda u: [], cache=c) == 0 and not p.exists() and not c.exists()   # 首次：不建檔
        seen = []
        assert RA.carry(p, ("http://pages/x", "http://raw/x"), fetch=lambda u: seen.append(u) or pub, cache=c) == 2
        assert seen == [("http://pages/x", "http://raw/x")] and len(RA.PL.load(c)) == 2
        assert RA.carry(p, "http://x", fetch=lambda u: pub, cache=c) == 2                    # 冪等
        RA.append(p, _line("2026-10-12"), cache=c)
        # 新 runner：site/data 空、Pages 抓不到 → 由備份帶回 (不洗掉歷史)
        p2 = Path(td) / "site2" / "radar_pool.jsonl"
        assert RA.carry(p2, "http://x", fetch=lambda u: [], cache=c) == 3 and len(RA.PL.load(p2)) == 3
        # Pages 有較新的同日行 (asof 較新) → 取 Pages 的
        newer = _line("2026-10-12", 7, asof="2026-10-12 22:00:00")
        assert RA.carry(p2, "http://x", fetch=lambda u: [newer], cache=c) == 3
        assert {x["ts"]: x["n"] for x in RA.PL.load(p2)}["2026-10-12"] == 7 and {x["ts"]: x["n"] for x in RA.PL.load(c)}["2026-10-12"] == 7
        # 沒有 urls → 只用備份 / 本機
        assert RA.carry(p2, (), fetch=lambda u: (_ for _ in ()).throw(AssertionError("should not fetch")), cache=c) == 3


def test_shrink_protection_and_tolerant_load():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.jsonl"
        rows = [_line(f"2026-10-{d:02d}") for d in range(1, 6)]
        assert RA._write(p, rows) is True
        before = p.read_text(encoding="utf-8")
        assert RA._write(p, []) is False and RA._write(p, rows[:2]) is False and p.read_text(encoding="utf-8") == before   # 不縮水
        assert RA._write(p, rows[-3:], cap=3) is True and len(RA.PL.load(p)) == 3                                        # CAP 截掉的可以
        assert not RA._write(Path(td) / "new.jsonl", []) and not (Path(td) / "new.jsonl").exists()                       # 0 行不建空檔
        # 壞行 (截斷 / 非 JSON) 跳過，其餘保留；append 後壞行消失
        q = Path(td) / "bad.jsonl"
        q.write_text(RA._dumps(rows[0]) + "\n" + RA._dumps(rows[1])[:40] + "\nnot json\n" + RA._dumps(rows[2]) + "\n", encoding="utf-8")
        assert [x["ts"] for x in RA.PL.load(q)] == ["2026-10-01", "2026-10-03"]
        assert RA.append(q, _line("2026-10-04"), cache=False) == 3
        assert all(ln.startswith("{") and ln.endswith("}") for ln in q.read_text(encoding="utf-8").splitlines())
        # 讀不到 / 不是 dict 的行、ts 壞掉的行 → merge 丟掉
        assert [x["ts"] for x in RA.merge([{"ts": "bad"}, {"x": 1}, "str", None, rows[0]])] == ["2026-10-01"]


# ------------------------------------------------------------------ (f) publish() 掛鉤
def _pool_dict(d: str, n: int = 3, src: str = "live") -> dict:
    return {"date": d, "src": src, "rows": [["2330", i + 1, 0.5, 0.2, 1.0, 100, 10.0, "B", 1.0, 10.0, 30.0, 0] for i in range(n)], "cols": RA.COLS, "conc10": 0.3}


def test_publish_hook():
    pub = [_line("2026-10-08"), _line("2026-10-09")]
    with tempfile.TemporaryDirectory() as td:
        p, c = Path(td) / "site" / "radar_pool.jsonl", Path(td) / "cache" / "radar_pool.jsonl"
        # fast：只帶回，pools 不寫
        rep = RA.publish(p, [_pool_dict("2026-10-12")], fast=True, urls=("u",), fetch=lambda u: pub, cache=c)
        assert rep["carried"] == 2 and rep["lines"] == 2 and rep["added"] == [] and rep["error"] is None and rep["bytes"] > 0
        assert "fast" in RA.report_txt(rep, fast=True)
        # 完整：live + backfill
        rep = RA.publish(p, [_pool_dict("2026-10-10", 2, "backfill"), _pool_dict("2026-10-12")], fast=False, urls=("u",), fetch=lambda u: pub, cache=c,
                         asof="2026-10-12 15:50:00")
        assert rep["carried"] == 2 and rep["lines"] == 4 and rep["added"] == [{"ts": "2026-10-10", "src": "backfill", "n": 2}, {"ts": "2026-10-12", "src": "live", "n": 3}]
        assert rep["line_bytes"] == len(RA._dumps(RA.line_from(_pool_dict("2026-10-12"), asof="2026-10-12 15:50:00")).encode()) + 1
        assert rep["bytes"] == p.stat().st_size and rep["est_cap_bytes"] == round(rep["bytes"] / 4 * RA.CAP)
        txt = RA.report_txt(rep)
        assert txt.startswith("radar_pool: carried 2 → 4 lines") and "MB" in txt and "2026-10-12 live 3" in txt
        # 單一 dict、None、沒有日期 → 不追加
        assert RA.publish(p, None, urls=(), cache=c)["added"] == [] and len(RA.PL.load(p)) == 4
        assert RA.publish(p, {"date": None, "rows": []}, urls=(), cache=c)["added"] == []
        assert RA.publish(p, _pool_dict("2026-10-13"), urls=(), cache=c)["lines"] == 5
        # fetch 例外 → 不外拋，仍可追加
        rep = RA.publish(p, _pool_dict("2026-10-14"), urls=("u",), fetch=lambda u: (_ for _ in ()).throw(OSError("net down")), cache=c)
        assert rep["error"] and "carry" in rep["error"] and rep["lines"] == 6 and "錯誤" in RA.report_txt(rep)
        # append 例外 → 不外拋、檔案不動
        before = p.read_text(encoding="utf-8")
        with patched(RA, "append", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk"))):
            rep = RA.publish(p, _pool_dict("2026-10-15"), urls=(), cache=c)
        assert "append" in rep["error"] and p.read_text(encoding="utf-8") == before
        # 首次 (什麼都沒有) + fast → 不建檔
        q = Path(td) / "empty" / "radar_pool.jsonl"
        rep = RA.publish(q, None, fast=True, urls=("u",), fetch=lambda u: [], cache=Path(td) / "empty" / "c.jsonl")
        assert rep["lines"] == 0 and rep["bytes"] == 0 and not q.exists()


def test_export_static_hook_guarded():
    src = (ROOT / "tools" / "export_static.py").read_text(encoding="utf-8")
    i_tl, i_ra, i_pit = src.index("_tl.build("), src.index("_ra.publish("), src.index("from chip import pit_ledger as _pit")
    assert i_tl < i_ra < i_pit, "掛在 treasure_live 之後、pit_ledger 之前"
    assert '_pools = _tlo.pop("_pools", None)' in src and src.index('_tlo.pop("_pools"') < src.index('dump("treasure_live", _tlo)')
    blk = src[src.rindex("try:", 0, i_ra): src.index("# 時點帳本", i_ra)]
    assert "except Exception" in blk and "radar_pool failed" in blk and "fast=args.fast" in blk and "_ra.RAW_URL" in blk and "gov8.PAGES_URL" in blk
    assert "_tlo, _pools = None, None" in src
    assert RA.RAW_URL.endswith("/gh-pages/data/radar_pool.jsonl") and RA.NAME == "radar_pool.jsonl" and RA.CACHE_PATH.name == RA.NAME


if __name__ == "__main__":
    fails = 0
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            n += 1
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'} ({n} tests)")
    sys.exit(1 if fails else 0)
