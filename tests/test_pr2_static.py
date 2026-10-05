"""pr2 P4/P5 靜態輸出測試：資料關係圖 schema 驗證 (tools/relations_map.py) 與時點帳本 (chip/pit_ledger.py)。不連網 (HTTP 以假 session 取代)。

    python tests/test_pr2_static.py
"""
from __future__ import annotations

import contextlib
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import relations_map as RM  # noqa: E402

from chip import pit_ledger as PIT  # noqa: E402


class _Resp:
    def __init__(self, status: int, body: str = ""):
        self.status_code = status; self.ok = 200 <= status < 300
        self.content = body.encode("utf-8"); self.text = body.encode("utf-8").decode("latin-1")   # 模擬 text/plain 無 charset 的 requests.text (中文會變亂碼)


class _Session:
    """假 requests.Session：url → _Resp | Exception；記錄呼叫順序。"""
    def __init__(self, table: dict):
        self.table = table; self.calls: list[str] = []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        r = self.table.get(url)
        if r is None:
            return _Resp(404)
        if isinstance(r, Exception):
            raise r
        return r


@contextlib.contextmanager
def fake_session(table: dict):
    s = _Session(table)
    old = PIT.session
    PIT.session = lambda: s
    try:
        yield s
    finally:
        PIT.session = old


def test_relations_map_installed_valid():
    doc = RM.load(RM.DEST)
    assert RM.validate(doc) == []
    assert doc["schema"] == RM.SCHEMA and doc["asof"] and doc["installed_at"] and doc["source_file"] == "relations_map_app.json"
    assert doc["summary"]["n_edges"] == len(doc["edges"]) == 81 and len(doc["nodes"]) == 28 and len(doc["rejected"]) == 13
    assert all(e["tradable"] is False for e in doc["edges"]) and doc["summary"]["model_changes_adopted"] == 0
    assert [k for k, _ in sorted(doc["layers"].items(), key=lambda kv: kv[1]["order"])][:2] == ["overnight", "day_ahead"]
    ids = {e["id"] for e in doc["edges"]}
    assert {"E01", "E02", "E03", "E37", "E52", "E59"} <= ids
    e03 = next(e for e in doc["edges"] if e["id"] == "E03")
    assert e03["grade"] == "C" and e03["grade_researcher"] == "A" and e03["ic"] == 0.092      # 驗證者降級保留在 JSON


def test_relations_map_validator_catches_problems():
    doc = RM.load(RM.DEST)
    d = copy.deepcopy(doc); d["edges"][0]["tradable"] = True
    assert any("tradable" in p for p in RM.validate(d))
    d = copy.deepcopy(doc); d["edges"][1]["grade"] = "F"
    assert any("grade" in p for p in RM.validate(d))
    d = copy.deepcopy(doc); d["edges"][2]["source"] = "NOPE"
    assert any("source/target" in p for p in RM.validate(d))
    d = copy.deepcopy(doc); d["edges"][3]["label"] = "買進 0050"
    assert any("advice" in p for p in RM.validate(d))
    d = copy.deepcopy(doc); d["summary"]["n_edges"] = 5
    assert any("n_edges" in p for p in RM.validate(d))
    d = copy.deepcopy(doc); d["edges"].pop()
    assert RM.validate(d)   # by_grade / n_edges 不符
    d = copy.deepcopy(doc); d["layers"]["overnight"]["order"] = 2
    assert any("orders" in p for p in RM.validate(d))
    assert RM.validate({"schema": "x"}) and RM.validate([]) == ["not a dict"]
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.json"; bad.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        try:
            RM.install(bad, Path(td) / "out.json")
            raise AssertionError("should raise")
        except ValueError as e:
            assert "invalid" in str(e)
        assert not (Path(td) / "out.json").exists()
        out = RM.install(RM.DEST, Path(td) / "ok.json", asof="2026-10-05")
        assert out["asof"] == "2026-10-05" and (Path(td) / "ok.json").exists()


def _fc():
    return {"date": "2026-10-02", "close": 49500.0, "trained_at": "2026-10-04", "trend7": {"state": "flat"},
            "next_days": [{"n": 1, "date": "2026-10-06", "variant": "night", "call": "偏多", "call_strength": "強", "p_up": 0.85, "conf_tier": "高", "level": 49775, "buy_at": 49189, "sell_at": 50298, "stop": 48962, "target": 50466,
                           "range_mode": "night", "range_sigma_src": "txo_iv", "recent_flag": None, "recent_hit": 0.84, "recent_up_rate": 0.55, "drivers": {"x": 1}},
                          {"n": 2, "date": "2026-10-07", "variant": "night", "call": "中性", "p_up": 0.56, "level": 49800}],
            "horizons": {"5": {"call": "中性", "call_model": "偏多", "p_up": 0.6, "call_five": "偏多", "variant": "daily"}, "20": {"p_up": 0.5}},
            "verdict": {"verdict": "偏多", "call": "偏多", "call_action": "偏多", "bucket": "分歧", "bucket_role": "info", "net": -3, "agree": 1, "disagree": 4, "conf_tier": "高", "votes": [1, 2]},
            "five": {"call": "偏多", "net": 3}, "learn": {"flags": {"degrade": False, "platt": False, "touch_factor": False}}}


def _tl():
    return {"model": {"trained_at": "2026-10-04 08:36:18"}, "scan": {"date": "2026-10-02", "treasure": [{"code": "1111", "tier": "A+", "p": 0.7}, {"code": "2222", "tier": "B", "p": 0.45}, {"code": "3333", "tier": "B+", "p": 0.52}],
                                                                        "surge": [{"code": "9999", "ps": 0.41}]}}


def test_pit_line_and_append_cap():
    line = PIT.line_from(_fc(), _tl(), {"phase": "closed"}, mode="full", now="2026-10-02 15:50:00")
    assert line["ts"] == "2026-10-02 15:50:00" and line["mode"] == "full" and line["date"] == "2026-10-02" and line["phase"] == "closed"
    nd = line["next_days"]
    assert len(nd) == 2 and nd[0]["call"] == "偏多" and nd[0]["conf_tier"] == "高" and nd[0]["buy_at"] == 49189 and "drivers" not in nd[0]
    assert line["horizons"]["5"] == {"call": "中性", "call_model": "偏多", "p_up": 0.6, "call_five": "偏多", "variant": "daily"} and line["horizons"]["20"] == {"p_up": 0.5}
    assert line["verdict"]["bucket"] == "分歧" and line["verdict"]["bucket_role"] == "info" and "votes" not in line["verdict"]
    assert line["five"] == {"call": "偏多", "net": 3} and line["treasure"]["AA"] == [{"code": "1111", "tier": "A+", "p": 0.7}] and line["treasure"]["n_B"] == 2 and line["treasure"]["model_ver"] == "2026-10-04 08:36:18"
    assert line["learn_flags"]["platt"] is False
    assert len(json.dumps(line, ensure_ascii=False)) < 4000
    assert PIT.line_from({"error": "x"}) is None and PIT.line_from({"date": "2026-10-02"}) is None
    assert PIT.line_from(_fc(), None, None)["treasure"] is None
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pit_ledger.jsonl"; c = Path(td) / "cache" / "pit_ledger.jsonl"
        assert PIT.append(p, None, cache=c) == 0 and not p.exists() and not c.exists()
        assert PIT.append(p, line, cache=c) == 1 and PIT.append(p, line, cache=c) == 1          # 同 ts+mode 覆蓋
        assert PIT.load(c) == PIT.load(p) == [line]                                              # 備份同步
        for i in range(5):
            PIT.append(p, dict(line, ts=f"2026-10-0{i + 3} 15:50:00"), cap=3, cache=c)
        rows = PIT.load(p)
        assert len(rows) == 3 and [r["ts"][:10] for r in rows] == ["2026-10-05", "2026-10-06", "2026-10-07"] and PIT.load(c) == rows
        # fast 行 (mode 不同) 與 full 行可共存
        PIT.append(p, dict(line, ts="2026-10-07 15:50:00", mode="fast"), cache=c)
        assert len(PIT.load(p)) == 4
        # cache=False → 不碰備份 (離線 / 測試)
        PIT.append(p, dict(line, ts="2026-10-08 15:50:00"), cache=False)
        assert len(PIT.load(p)) == 5 and len(PIT.load(c)) == 4
    assert PIT.CACHE_PATH.name == "pit_ledger.jsonl" and PIT.CACHE_PATH.parent.name == "cache"


def test_pit_numpy_scalars_become_json_numbers():
    """稽核檔內數值必須是 JSON number/bool：np.int64 不是 int、np.bool_ 不是 bool，_get/line_from/write 都要過 serialize.clean。"""
    fc = _fc()
    fc["close"] = np.float64(49500.5)
    fc["next_days"][0].update({"n": np.int64(1), "level": np.int64(49775), "p_up": np.float64(0.85), "buy_at": np.int64(49189)})
    fc["horizons"]["5"]["p_up"] = np.float64(0.6)
    fc["verdict"].update({"net": np.int64(-3), "agree": np.int64(1), "disagree": np.int64(4)})
    fc["five"]["net"] = np.int64(3); fc["intraday"] = np.bool_(False)
    tl = _tl(); tl["scan"]["treasure"][0]["p"] = np.float64(0.7)
    line = PIT.line_from(fc, tl, None, now="2026-10-02 15:50:00")
    nd = line["next_days"][0]
    assert nd["n"] == 1 and type(nd["n"]) is int and type(nd["level"]) is int and type(nd["buy_at"]) is int and type(nd["p_up"]) is float
    assert type(line["verdict"]["net"]) is int and line["verdict"]["net"] == -3 and type(line["verdict"]["agree"]) is int
    assert type(line["five"]["net"]) is int and type(line["close"]) is float and type(line["intraday"]) is bool and type(line["treasure"]["AA"][0]["p"]) is float
    assert PIT._get({"a": np.int64(5)}, "a") == 5 and PIT._get({"a": np.bool_(True)}, "a") is True and PIT._get({"a": np.float64(1.5)}, "a") == 1.5
    assert PIT._get({"a": float("nan")}, "a") is None and PIT._get({"a": {"x": 1}}, "a") == "{'x': 1}"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pit_ledger.jsonl"
        PIT.append(p, line, cache=False)
        raw = p.read_text(encoding="utf-8")
        assert '"n": 1,' in raw and '"net": -3' in raw and '"n": "1"' not in raw
        back = PIT.load(p)[0]
        assert type(back["next_days"][0]["n"]) is int and type(back["verdict"]["net"]) is int
        # write() 本身也會 clean：直接塞 numpy 進 rows
        PIT.write(p, [{"ts": "2026-10-03 15:50:00", "mode": "full", "v": np.int64(7), "b": np.bool_(True)}])
        assert PIT.load(p) == [{"ts": "2026-10-03 15:50:00", "mode": "full", "v": 7, "b": True}]


def test_pit_parse_lines_tolerant():
    good = {"ts": "2026-10-01 15:50:00", "mode": "full", "date": "2026-10-01", "x": "中文"}
    text = json.dumps(good, ensure_ascii=False) + "\n{not json\n\n[1,2]\n" + json.dumps(dict(good, ts="2026-10-02 15:50:00")) + "\n{\"ts\": \"2026-10-03 15:50:00\", \"mode\": \"fu"   # 最後一行被截斷
    rows = PIT.parse_lines(text)
    assert [r["ts"][:10] for r in rows] == ["2026-10-01", "2026-10-02"] and rows[0]["x"] == "中文"
    assert PIT.parse_lines("") == [] and PIT.parse_lines(None) == []
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pit_ledger.jsonl"; p.write_text(text, encoding="utf-8")
        assert PIT.load(p) == rows                                                             # load 走同一個 helper
    with fake_session({"http://pages/x.jsonl": _Resp(200, text)}):
        assert PIT.fetch_published("http://pages/x.jsonl") == rows                            # 遠端一行壞掉不丟整份、不用 r.text 所以中文不亂碼


def test_pit_fetch_published_session_and_fallback():
    pub = [{"ts": "2026-09-30 15:50:00", "mode": "full", "date": "2026-09-30", "x": "中文"}]
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pub)
    # Pages 404 → raw gh-pages 備援
    with fake_session({"http://raw/x.jsonl": _Resp(200, body)}) as s:
        assert PIT.fetch_published(("http://pages/x.jsonl", "http://raw/x.jsonl")) == pub and s.calls == ["http://pages/x.jsonl", "http://raw/x.jsonl"]
    # Pages 有內容 → 不去備援
    with fake_session({"http://pages/x.jsonl": _Resp(200, body), "http://raw/x.jsonl": _Resp(200, "")}) as s:
        assert PIT.fetch_published(["http://pages/x.jsonl", "http://raw/x.jsonl"]) == pub and s.calls == ["http://pages/x.jsonl"]
    # 連線例外 / 5xx / 空檔 / 全壞行 → []，例外被吞
    with fake_session({"http://pages/x.jsonl": ConnectionError("boom"), "http://raw/x.jsonl": _Resp(503, "x")}) as s:
        assert PIT.fetch_published(("http://pages/x.jsonl", "http://raw/x.jsonl"), timeout=1) == [] and len(s.calls) == 2
    with fake_session({"http://pages/x.jsonl": _Resp(200, "{bad\n{also bad")}):
        assert PIT.fetch_published("http://pages/x.jsonl") == []
    with fake_session({}):
        assert PIT.fetch_published("http://pages/x.jsonl") == [] and PIT.fetch_published(()) == []


def test_pit_carry_over():
    line = PIT.line_from(_fc(), _tl(), None, now="2026-10-02 15:50:00")
    pub = [dict(line, ts="2026-09-30 15:50:00"), dict(line, ts="2026-10-01 15:50:00")]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pit_ledger.jsonl"; c = Path(td) / "cache" / "pit_ledger.jsonl"
        assert PIT.carry_over(p, "http://x", fetch=lambda url: [], cache=c) == 0 and not p.exists() and not c.exists()   # 無來源不建檔
        PIT.append(p, line, cache=c)
        assert PIT.carry_over(p, "http://x", fetch=lambda url: pub, cache=c) == 3
        rows = PIT.load(p)
        assert [r["ts"][:10] for r in rows] == ["2026-09-30", "2026-10-01", "2026-10-02"] and PIT.load(c) == rows      # 兩處同步
        assert PIT.carry_over(p, "http://x", fetch=lambda url: pub, cache=c) == 3                                       # 冪等
        assert PIT.carry_over(p, "http://x", fetch=lambda url: [], cache=c) == 3 and len(PIT.load(p)) == 3            # Pages 失敗 → 保留本機/備份
        # 真實 fetch_published 經由 (假) session，接受多個 url
        with fake_session({"http://raw/x.jsonl": _Resp(200, "".join(json.dumps(dict(line, ts="2026-09-29 15:50:00"), ensure_ascii=False) + "\n"))}):
            assert PIT.carry_over(p, ("http://pages/x.jsonl", "http://raw/x.jsonl"), cache=c) == 4
        assert [r["ts"][:10] for r in PIT.load(p)] == ["2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]


def test_pit_fresh_runner_failed_fetch_keeps_cache():
    """Actions 情境：site/data 每次都是空的 (gitignore、不在 cache)，只有 data/cache 進 actions/cache。Pages 一次抓不到 (timeout/5xx/404) 不得洗掉歷史。"""
    pages = [{"ts": f"2026-09-{d:02d} 15:40:00", "mode": "full", "date": f"2026-09-{d:02d}"} for d in range(1, 31)]
    line = {"ts": "2026-10-05 15:40:00", "mode": "full", "date": "2026-10-05"}
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "cache" / "pit_ledger.jsonl"
        # run 1 (full，Pages 正常)：site/data 空 → 帶回 30 行、追加 1 行、備份同步
        p1 = Path(td) / "run1" / "pit_ledger.jsonl"
        assert PIT.carry_over(p1, "http://x", fetch=lambda u: pages, cache=c) == 30
        assert PIT.append(p1, line, cache=c) == 31 and len(PIT.load(c)) == 31
        # run 2 (fast，新 runner、Pages 抓取失敗)：靠備份仍有 31 行、檔案有建 (發布不會少掉檔)
        p2 = Path(td) / "run2" / "pit_ledger.jsonl"
        assert PIT.carry_over(p2, "http://x", fetch=lambda u: [], cache=c) == 31 and p2.exists() and len(PIT.load(p2)) == 31
        # run 3 (full，新 runner、Pages 抓取失敗)：不會變成只有今天 1 行
        p3 = Path(td) / "run3" / "pit_ledger.jsonl"
        PIT.carry_over(p3, "http://x", fetch=lambda u: [], cache=c)
        assert PIT.append(p3, dict(line, ts="2026-10-06 15:40:00"), cache=c) == 32 and len(PIT.load(c)) == 32
        # run 4：Pages 回來但只有被截短的版本 (例如 run 2 之前發布的 1 行) → 聯集，不縮水
        p4 = Path(td) / "run4" / "pit_ledger.jsonl"
        assert PIT.carry_over(p4, "http://x", fetch=lambda u: [line], cache=c) == 32
        # 縮水保護：write() 不以 0 行覆蓋
        assert PIT.write(p4, []) is False and len(PIT.load(p4)) == 32
        assert PIT.write(Path(td) / "none.jsonl", []) is False and not (Path(td) / "none.jsonl").exists()
        assert PIT.write(p4, [line]) is True and len(PIT.load(p4)) == 1                      # 非空覆蓋照常 (cap 行為)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
