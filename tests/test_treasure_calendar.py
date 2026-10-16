"""BE-1 (2026-10-13)：雷達帳本 / 評分的日 K 對齊證交所交易日曆 (chip/predict/treasure_live.py TradeCal / align_bars) 單元測試。不連網。

    python tests/test_treasure_calendar.py

背景：Yahoo 日 K 的日子和證交所不一致 — 2026-10-12 (真實交易日) 個股收盤全是 null → 那根被丟掉；2026-07-10 (非交易日) 卻有一根 (幽靈日)。
帳本「第幾天 / 第 21 日 / 停損日」與特徵 (>11% 跳動濾網) 都把 Yahoo 的列當交易日 → 位移一天、整批股票被誤判「疑似減資」。

(a) FMTQIK 交易日：抓不到 / stat 不是 OK → 丟例外且不寫快取 (#87 cached() 會把 None 存起來)；已結束月份用 ':final' 長期鍵、本月短 TTL；
(b) TradeCal：逐月懶載入、本月只到已知最後交易日、FMTQIK 失敗 → 後備日曆 (只補洞不刪列)、未來月份未知；
(c) 10-12 型缺日：盤後表補上 → 第 21 日 / 結案日 / 停損日落在 10-12 (2303 / 2308 型)；配股縮放過的歷史依比例換算；停牌 / 長段缺口不補；
(d) 07-10 型幽靈日：去掉 → fin21Date 是第 21 個交易日；
(e) score_one：跨洞的 +9.8% / +10.0% 兩天 (6834 型) 不再被當成 >11% 跳動略過；
(f) 一次性重算 (calv)：標記與日曆不一致的已結案列 → 存 cal_orig、重設重算；一致的只標 calv；無法驗證 (盤後表抓不到) 不動、下次再查；
(g) 懶載入：空帳本的 build() / 假評分的 scan() 完全不抓日曆；盤後表查詢有上限。
"""
from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip import config  # noqa: E402
from chip import http as H  # noqa: E402
from chip.predict import treasure_live as TL  # noqa: E402

TODAY = "2026-10-13"
HOL = {"2026-07-10", "2026-09-25", "2026-09-28", "2026-10-09"}       # 07-10 = Yahoo 幽靈日 (證交所休市)；其餘為假日
TW = [d for d in pd.bdate_range("2026-03-02", "2026-11-30").strftime("%Y-%m-%d") if d not in HOL]
TW_NOW = [d for d in TW if d <= TODAY]
MK = {d: 1000.0 for d in TW_NOW}                                    # 大盤持平 → rel = cur


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def fake_fmtqik(calls=None, fail=()):
    """假 FMTQIK：本月只到 TODAY；fail 內的月份 ('YYYYMM') 丟例外。"""
    def f(ym, final):
        if calls is not None:
            calls.append((ym, final))
        if ym in fail:
            raise LookupError(f"FMTQIK {ym} down")
        ds = [d for d in TW_NOW if d[:7] == f"{ym[:4]}-{ym[4:]}"]
        if not ds:
            raise LookupError(f"FMTQIK {ym}: no data")
        return ds
    return f


def fake_snapshot(mi: dict, calls=None):
    """假 market_snapshot：有那天 → (那天, 列)；沒有 → (None, {}) (= 抓不到 / 沒公布)。"""
    def f(date=None):
        if calls is not None:
            calls.append(date)
        return (date, mi[date]) if date in mi else (None, {})
    return f


def bar(d, c, lo=None, hi=None, v=1000):
    return {"date": d, "open": c, "high": hi if hi is not None else c, "low": lo if lo is not None else c, "close": c, "volume": v}


def cal(**kw):
    return TL.TradeCal(today=TODAY, **kw)


# ------------------------------------------------------------------ (a) FMTQIK
def test_fmtqik_raises_and_never_caches_failure():
    rows = [["115/10/01", "1", "1", "1", "49,000.00", "10"], ["115/10/02", "1", "1", "1", "49,100.00", "10"], ["115/10/12", "1", "1", "1", "49,301.96", "10"]]
    state = {"j": {"stat": "很抱歉，沒有符合條件的資料!"}}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return state["j"]

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            assert url.endswith("/afterTrading/FMTQIK") and params["date"].endswith("01"), (url, params)
            return R()

    with tempfile.TemporaryDirectory() as td, patched(config, "CACHE_DIR", Path(td)), patched(TL, "session", lambda: S()):
        for bad in ({"stat": "很抱歉，沒有符合條件的資料!"}, {"stat": "OK", "data": []}, {"stat": "OK", "data": [["115/09/30", "1"]]}, ["not", "dict"]):
            state["j"] = bad
            try:
                TL._fmtqik_days("202610", final=False)
                assert False, ("should raise", bad)
            except Exception as e:  # noqa: BLE001
                assert not isinstance(e, AssertionError), e
            assert not H._cache_path("twse:fmtqik_days:202610").exists(), "失敗不可寫快取 (#87)"
        state["j"] = {"stat": "OK", "data": rows}
        assert TL._fmtqik_days("202610", final=False) == ["2026-10-01", "2026-10-02", "2026-10-12"]
        p_now, p_fin = H._cache_path("twse:fmtqik_days:202610"), H._cache_path("twse:fmtqik_days:202610:final")
        assert p_now.exists() and not p_fin.exists()
        assert json.loads(p_now.read_text(encoding="utf-8"))["data"] == ["2026-10-01", "2026-10-02", "2026-10-12"]
        TL._fmtqik_days("202610", final=True)
        assert p_fin.exists(), "已結束月份用 ':final' 鍵 (月中抓的本月清單不會被當成整月)"


def test_fmtqik_ttl_keys():
    seen = []

    def fake_cached(key, ttl, loader, allow_stale=True):
        seen.append((key, ttl))
        return ["2026-09-01"]

    with patched(TL, "cached", fake_cached):
        TL._fmtqik_days("202609", final=True)
        TL._fmtqik_days("202610", final=False)
    assert seen == [("twse:fmtqik_days:202609:final", TL.CAL_FINAL_TTL), ("twse:fmtqik_days:202610", config.TTL_INTRADAY)], seen
    assert TL.CAL_FINAL_TTL >= 365 * 86400


# ------------------------------------------------------------------ (b) TradeCal
def test_tradecal_lazy_months_and_fallback():
    calls = []
    with patched(TL, "_fmtqik_days", fake_fmtqik(calls, fail={"202607"})):
        c = cal(extra=lambda: set(TW_NOW) | {"2026-07-10"})    # 後備 = ^TWII ∪ 掃描日 (假設 ^TWII 也有幽靈日)
        assert calls == [], "建立時不抓"
        assert c.known("2026-10-12") == "exact" and c.is_day("2026-10-12") and not c.is_day("2026-10-09")
        assert c.known("2026-10-13") == "exact" and c.known("2026-10-14") is None, "本月只到已知最後交易日"
        assert c.known("2026-11-02") is None and ("202611", False) not in calls, "未來月份不抓"
        assert ("202610", False) in calls
        assert c.known("2026-09-30") == "exact" and ("202609", True) in calls, "已結束月份 final=True"
        n = len(calls)
        c.known("2026-09-01"); c.is_day("2026-09-02")
        assert len(calls) == n, "同月不重抓"
        assert c.month("2026-07")[0] == "fallback" and c.known("2026-07-10") == "fallback"
        assert not c.exact("2026-07-01", "2026-08-05") and c.exact("2026-08-03", "2026-10-12") and not c.exact("2026-09-01", "2026-10-14")
        assert c.days("2026-10-07", "2026-10-13") == ["2026-10-07", "2026-10-08", "2026-10-12", "2026-10-13"]
    with patched(TL, "_fmtqik_days", fake_fmtqik(fail={"202607"})):
        c2 = cal()
        assert c2.month("2026-07") == (None, set()) and c2.known("2026-07-06") is None, "FMTQIK 失敗且沒有後備 → 未知 (不處理)"


# ------------------------------------------------------------------ (c) 10-12 型缺日
def _hole_bars(code_path: dict, start="2026-07-01", hole=("2026-10-12",)):
    """TW 日 (到 TODAY)，缺 hole；code_path = {date: (close, low)}，其餘 100。"""
    return [bar(d, *code_path.get(d, (100.0, 100.0))) for d in TW_NOW if d >= start and d not in hole]


MI_1012 = {"2026-10-12": {"2303": {"open": 101.0, "high": 106.5, "low": 100.5, "close": 106.2, "volume": 23.5e6},
                          "2308": {"open": 95.0, "high": 95.5, "low": 90.49, "close": 92.0, "volume": 1e6}},
           "2026-10-08": {"2303": {"close": 101.0}, "2308": {"close": 96.0}}}


def _after(D, n):
    return [d for d in TW_NOW if d > D][:n]


def test_fill_null_day_lands_day21_and_stop_on_1012():
    """2303 (09-08 推薦)：第 21 個交易日是 10-12 (+6.2%)；Yahoo 缺 10-12 時舊算法拿 10-13 (+3.28%) 當第 21 日 → 未命中。
    2308 (10-06 推薦)：10-12 盤中 −9.51% 是第 3 天的停損；舊算法記成 10-13 的 −13.66%。"""
    assert _after("2026-09-08", 21)[-1] == "2026-10-12" and _after("2026-10-06", 3)[-1] == "2026-10-12"
    p03 = {d: (101.0, 100.0) for d in _after("2026-09-08", 20)}; p03["2026-10-13"] = (103.28, 102.0)
    p08 = {"2026-10-07": (98.0, 97.0), "2026-10-08": (96.0, 95.0), "2026-10-13": (88.0, 86.34)}
    B = {"2303": _hole_bars(p03), "2308": _hole_bars(p08)}

    def rows():
        return {"ledger": {"treasure": [{"code": "2303", "name": "聯電", "date": "2026-09-08", "entry": 100.0, "mktEntry": 1000.0, "tier": "A", "status": "追蹤"},
                                        {"code": "2308", "name": "台達電", "date": "2026-10-06", "entry": 100.0, "mktEntry": 1000.0, "tier": "A", "status": "追蹤"}],
                           "surge": []}}
    snap = []
    with patched(TL, "bars", lambda code, years="2y": B[code]), patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot(MI_1012, snap)):
        old = rows(); TL.evaluate(old, MK)                      # 舊行為 (不給日曆)
        new = rows(); TL.evaluate(new, MK, cal=cal())
    o = {x["code"]: x for x in old["ledger"]["treasure"]}; n = {x["code"]: x for x in new["ledger"]["treasure"]}
    assert o["2303"]["fin21Date"] == "2026-10-13" and o["2303"]["status"] == "未命中", o["2303"]
    a = n["2303"]
    assert a["days"] == 21 and a["fin21Date"] == a["evalAt"] == a["lastDate"] == "2026-10-12" and abs(a["fin21"] - 6.2) < 1e-6, a
    assert a["status"] == "命中" and a["hitDay"] == 21 and a["hitDate"] == "2026-10-12" and a["calv"] == 1, a
    assert o["2308"]["stopDate"] == "2026-10-13" and abs(o["2308"]["stopVal"] + 13.66) < 1e-6
    s = n["2308"]
    assert s["status"] == "未命中" and s["stopDay"] == 3 and s["stopDate"] == s["evalAt"] == "2026-10-12" and abs(s["stopVal"] + 9.51) < 1e-6 and abs(s["cur"] + 8.0) < 1e-6, s
    assert "第 3 天 (10-12) 盤中跌破" in s["reason"], s["reason"]
    assert set(snap) == {"2026-10-12", "2026-10-08"}, ("只查缺日與對照基準那天，每天一次", snap)
    # bars() 快取裡的原 list 沒被改 (補的列只在對齊後的新 list)
    assert all(r["date"] != "2026-10-12" for r in B["2303"])


def test_fill_scaled_suspended_and_long_gap():
    """Yahoo 歷史已因配股縮放 (Yahoo = 盤後表 × 0.5) → 補的這根同比例；該股不在盤後表 (停牌) → 維持缺口；連續缺 > FILL_MAX_RUN 天 → 不補也不逐日查。"""
    days = [d for d in TW_NOW if "2026-09-01" <= d <= TODAY]
    b = [bar(d, 50.0, 49.0, 51.0, 2000) for d in days if d != "2026-10-12"]
    mi = {"2026-10-12": {"9001": {"open": 101.0, "high": 104.0, "low": 99.0, "close": 102.0, "volume": 3000.0}}, "2026-10-08": {"9001": {"close": 100.0}}}
    snap = []
    with patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot(mi, snap)):
        c = cal()
        got = TL.align_bars(b, c, "9001", lo="2026-09-01")
        f = next(r for r in got if r["date"] == "2026-10-12")
        assert f["fill"] == "mi" and abs(f["close"] - 51.0) < 1e-9 and abs(f["high"] - 52.0) < 1e-9 and abs(f["low"] - 49.5) < 1e-9 and abs(f["volume"] - 6000.0) < 1e-9, f
        assert [r["date"] for r in got] == days, "日期 = 證交所日曆"
        g2 = TL.align_bars(b, c, "9002", lo="2026-09-01")       # 9002 不在 10-12 盤後表 → 缺口保留
        assert "2026-10-12" not in {r["date"] for r in g2} and len(g2) == len(b)
        # 10-01 ~ 10-07 (5 天) 整段缺 = 停牌 → 不補、不為它查盤後表
        snap.clear()
        long_gap = [r for r in b if not ("2026-10-01" <= r["date"] <= "2026-10-07")]
        g3 = TL.align_bars(long_gap, c, "9001", lo="2026-09-01")
        assert not any("2026-10-01" <= r["date"] <= "2026-10-07" for r in g3) and any(r["date"] == "2026-10-12" for r in g3)
        assert not any("2026-10-01" <= d <= "2026-10-07" for d in snap), snap
        # cal = None → 原樣 (舊行為)；cal_bars 包 bars()
        assert TL.align_bars(b, None, "9001") is b
        with patched(TL, "bars", lambda code, years="2y": b):
            assert [r["date"] for r in TL.cal_bars("9001", c, lo="2026-09-01")] == days


def test_unknown_tail_kept_and_fallback_fill_only():
    """本月 FMTQIK 還沒有今天 → 今天那根 (日曆未知) 不刪；FMTQIK 抓不到的月份 → 後備日曆只補洞、幽靈日不刪 (^TWII 自己也會缺日)。"""
    upto_1012 = lambda ym, final: [d for d in fake_fmtqik()(ym, final) if d <= "2026-10-12"]   # noqa: E731
    b = [bar(d, 100.0) for d in TW_NOW if d >= "2026-10-01"]
    with patched(TL, "_fmtqik_days", upto_1012), patched(TL, "market_snapshot", fake_snapshot({})):
        got = TL.align_bars(b, cal(), "1101", lo="2026-10-01")
    assert got[-1]["date"] == "2026-10-13", "日曆未知的最新一根保留"
    jul = [bar(d, 100.0) for d in TW_NOW if "2026-07-01" <= d <= "2026-07-31" and d != "2026-07-15"] + [bar("2026-07-10", 100.0)]
    mi = {"2026-07-15": {"1101": {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}}, "2026-07-14": {"1101": {"close": 100.0}}}
    with patched(TL, "_fmtqik_days", fake_fmtqik(fail={"202607"})), patched(TL, "market_snapshot", fake_snapshot(mi)):
        c = cal(extra=set(TW_NOW))
        got = TL.align_bars(jul, c, "1101", lo="2026-07-01")
    ds = [r["date"] for r in got]
    assert "2026-07-10" in ds and "2026-07-15" in ds and ds == sorted(ds), "後備月份：只補洞、不刪列"


# ------------------------------------------------------------------ (d) 07-10 型幽靈日
def _phantom_bars():
    out = []
    for d in sorted(set(TW_NOW) | {"2026-07-10"}):
        if d < "2026-06-01":
            continue
        k = len([x for x in TW_NOW if "2026-07-06" < x <= d])
        out.append(bar(d, 100.0 + 0.1 * k, 99.5 + 0.1 * k))
    return out


def test_phantom_day_dropped_fin21_is_21st_trading_day():
    B = _phantom_bars()
    D = "2026-07-06"
    d21 = _after(D, 21)[-1]
    assert d21 == "2026-08-05"

    def led():
        return {"ledger": {"treasure": [{"code": "2527", "name": "宏璟", "date": D, "entry": 100.0, "mktEntry": 1000.0, "tier": "A", "status": "追蹤"}], "surge": []}}
    with patched(TL, "bars", lambda code, years="2y": B), patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot({})):
        old = led(); TL.evaluate(old, MK, upto="2026-08-31")
        new = led(); TL.evaluate(new, MK, upto="2026-08-31", cal=cal())
    o, n = old["ledger"]["treasure"][0], new["ledger"]["treasure"][0]
    assert o["fin21Date"] == o["evalAt"] == "2026-08-04", ("舊算法：幽靈日算一天 → 第 20 個交易日當第 21 日", o)
    assert n["fin21Date"] == n["evalAt"] == d21 and n["days"] == 21 and abs(n["fin21"] - 2.1) < 1e-6 and "calv" not in n, n   # upto (回填) 不做日曆核對


# ------------------------------------------------------------------ (e) score_one 跨洞
def test_score_one_hole_no_false_split_skip():
    """6834 型：10-12 +9.8%、10-13 +10.0%；Yahoo 缺 10-12 → 10-08→10-13 兩天 +20.8% 被當成 >11% 跳動 (疑似減資) 略過 ~60 個交易日。"""
    D = "2026-10-13"
    hist = [bar(d, 50.0, 49.5, 50.5) for d in TW_NOW if d <= "2026-10-08"]
    mi = {"2026-10-12": {"6834": {"open": 51.0, "high": 54.9, "low": 50.8, "close": 54.9, "volume": 5000.0}}, "2026-10-08": {"6834": {"close": 50.0}}}
    row = {"open": 55.0, "high": 60.39, "low": 54.9, "close": 60.39, "volume": 6000, "value": 6e5}
    ds = [d for d in TW_NOW]
    mkf = pd.DataFrame({"date": ds, "m_close": 1000.0, "m_ret1": 0.0, "m_bias20": -1.0, "m_ret20": 0.0})
    tm = {"features": ["ret20", "m_bias20"], "trees": [], "init": 0.0, "th_A": 0.5, "gate": {"m_bias20_lt": 0.0}}
    calls = []
    with patched(TL, "bars", lambda code, years="2y": hist), patched(TL, "_fmtqik_days", fake_fmtqik(calls)), patched(TL, "market_snapshot", fake_snapshot(mi)):
        assert TL.score_one("6834", row, D, mkf, tm) is None, "舊行為：兩天跳動 > 11% → 略過"
        got = TL.score_one("6834", row, D, mkf, tm, cal=cal(mi={D: {"6834": row}}))
    assert got is not None and abs(got["_x"]["ret20"] - (60.39 / 50.0 - 1) * 100) < 1e-6, got
    lo = (pd.Timestamp(D) - pd.Timedelta(days=TL.CAL_LOOKBACK_DAYS)).strftime("%Y%m")
    assert min(ym for ym, _ in calls) == lo, ("只對齊 D 前 CAL_LOOKBACK_DAYS 天", calls)


# ------------------------------------------------------------------ (f) 一次性重算 (calv)
def test_calv_rewalks_inconsistent_closed_rows_once():
    B = _phantom_bars()
    D = "2026-07-06"
    good = "2026-08-10"          # 窗口內沒有幽靈日 / 缺日 → 一致
    with patched(TL, "bars", lambda code, years="2y": B), patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot({})):
        prev = {"ledger": {"treasure": [{"code": "2527", "name": "宏璟", "date": D, "entry": 100.0, "mktEntry": 1000.0, "tier": "A", "status": "追蹤"},
                                        {"code": "2104", "name": "國際中橡", "date": good, "entry": B[[r["date"] for r in B].index(good)]["close"], "mktEntry": 1000.0, "tier": "B", "status": "追蹤"}],
                           "surge": [{"code": "2527", "name": "宏璟", "date": D, "entry": 100.0, "ps": 0.4, "status": "追蹤", "days": 0}]}}
        TL.evaluate(prev, MK, upto="2026-09-30")                  # 舊算法結案 (幽靈日算一天)
        t, g, s = prev["ledger"]["treasure"][0], prev["ledger"]["treasure"][1], prev["ledger"]["surge"][0]
        assert t["evalAt"] == "2026-08-04" and s["evalAt"] == "2026-08-03" and s["status"] == "未飆" and "calv" not in t, (t, s)
        g_before = {k: g[k] for k in ("status", "cur", "days", "evalAt", "fin21", "fin21Date")}
        t_status = t["status"]
        n = TL.evaluate(prev, MK, cal=cal())                      # 有日曆 (不帶 upto)：不一致的重算一次
        assert n == 2
        assert t["cal_fix"] == 1 and t["cal_orig"]["evalAt"] == "2026-08-04" and t["cal_orig"]["status"] == t_status and t["calv"] == 1, t
        assert t["evalAt"] == t["fin21Date"] == "2026-08-05" and t["days"] == 21 and t["status"] != "追蹤", t
        assert s["cal_fix"] == 1 and s["evalAt"] == _after(D, 20)[-1] == "2026-08-04" and s["days"] == 20 and s["calv"] == 1, s
        assert g["calv"] == 1 and "cal_orig" not in g and "cal_fix" not in g and {k: g[k] for k in g_before} == g_before, g
        snap = json.dumps(prev, sort_keys=True, ensure_ascii=False)
        assert TL.evaluate(prev, MK, cal=cal()) == 0, "calv=1 之後不再進 todo"
        assert json.dumps(prev, sort_keys=True, ensure_ascii=False) == snap


def test_calv_unverifiable_waits_for_mi():
    """已在 10-13 結案 (少算 10-12) 的列：10-12 盤後表抓不到 → 無法驗證、不動、不標 calv；盤後表回來 → 重算 (第 21 日 10-12 命中)。"""
    p03 = {d: (101.0, 100.0) for d in _after("2026-09-08", 20)}; p03["2026-10-13"] = (103.28, 102.0)
    B = _hole_bars(p03)
    prev = {"ledger": {"treasure": [{"code": "2303", "name": "聯電", "date": "2026-09-08", "entry": 100.0, "mktEntry": 1000.0, "tier": "A", "status": "追蹤"}], "surge": []}}
    with patched(TL, "bars", lambda code, years="2y": B), patched(TL, "_fmtqik_days", fake_fmtqik()):
        TL.evaluate(prev, MK)
        x = prev["ledger"]["treasure"][0]
        assert x["status"] == "未命中" and x["evalAt"] == "2026-10-13"
        before = json.dumps(x, sort_keys=True, ensure_ascii=False)
        with patched(TL, "market_snapshot", fake_snapshot({})):
            TL.evaluate(prev, MK, cal=cal())
        assert json.dumps(x, sort_keys=True, ensure_ascii=False) == before and "calv" not in x, x
        with patched(TL, "market_snapshot", fake_snapshot(MI_1012)):
            TL.evaluate(prev, MK, cal=cal())
    assert x["status"] == "命中" and x["evalAt"] == "2026-10-12" and x["cal_orig"]["status"] == "未命中" and x["calv"] == 1, x


def test_calv_not_reset_under_upto_or_lagging_bars():
    """審查修正：一致的已結案列 — 回填逐日模擬 evaluate(upto=較早日期) 把日 K 截在結案標記之前 → 不核對、不重設；
    日 K 落後 (Yahoo 最後一天早於結案標記) → 無法驗證 (None)、不重設、不標 calv。"""
    B = _phantom_bars()
    D = "2026-08-10"             # 窗口內沒有幽靈日 / 缺日 → 一致
    with patched(TL, "bars", lambda code, years="2y": B), patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot({})):
        prev = {"ledger": {"treasure": [{"code": "2104", "name": "國際中橡", "date": D, "entry": B[[r["date"] for r in B].index(D)]["close"], "mktEntry": 1000.0, "tier": "B", "status": "追蹤"}],
                           "surge": []}}
        TL.evaluate(prev, MK, cal=cal())
        x = prev["ledger"]["treasure"][0]
        assert x["status"] != "追蹤" and x["fin21Date"] == _after(D, 21)[-1] and x["calv"] == 1, x
        x.pop("calv")
        before = json.dumps(x, sort_keys=True, ensure_ascii=False)
        TL.evaluate(prev, MK, upto="2026-08-20", cal=cal())           # 回填：日 K 截到 08-20 (< fin21Date)
        assert json.dumps(x, sort_keys=True, ensure_ascii=False) == before, ("upto 不該重設已結案列", x)
        c = cal()
        lag = [r for r in B if r["date"] <= "2026-08-25"]
        assert TL._cal_check(x, TL.align_bars(lag, c, "2104", lo=D), c) is None, "日 K 落後 → 無法驗證"
    with patched(TL, "bars", lambda code, years="2y": lag), patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot({})):
        TL.evaluate(prev, MK, cal=cal())                              # 不帶 upto，但 Yahoo 日 K 只到 08-25
    assert json.dumps(x, sort_keys=True, ensure_ascii=False) == before and "calv" not in x and "cal_fix" not in x, x


# ------------------------------------------------------------------ (g) 懶載入 / 上限
def test_lazy_no_calendar_fetch_without_rows():
    calls = []
    import datetime as dt
    D = dt.datetime.now(config.TZ).date().isoformat()
    tm = {"trees": [{}], "trained_at": "2026-01-01 00:00:00", "th_A": 0.5, "th_Aplus": 0.6, "features": ["ret20"]}
    with tempfile.TemporaryDirectory() as td, patched(TL, "_fmtqik_days", fake_fmtqik(calls)), patched(TL, "LOCAL", Path(td) / "tl.json"), \
            patched(TL, "load_prev", lambda: {"ledger": {"treasure": [], "surge": []}, "scans": []}), patched(TL, "twii_hist", lambda: {}), \
            patched(TL, "scan", lambda tm_, date=None: {"date": D, "treasure": [], "surge": [], "mkt_close": 1000.0}), patched(TL.M, "load_json", lambda name: tm):
        TL.build(date=D)
    assert calls == [], "空帳本：不抓日曆"
    seen = {}

    def fake_score(code, r, d, mkf, tm_, cal=None):
        seen["cal"] = cal
        return None
    rows = {"2330": {"name": "台積電", "close": 100.0, "open": 99.0, "high": 101.0, "low": 98.0, "volume": 1000, "value": 5e9, "pct": 1.0, "amp": 2.0}}
    with patched(TL, "_fmtqik_days", fake_fmtqik(calls)), patched(TL, "market_snapshot", lambda date=None: ("2026-10-12", rows)), \
            patched(TL, "twii_hist", lambda: {d: 1000.0 for d in TW_NOW}), patched(TL, "valuation", lambda: {}), patched(TL, "score_one", fake_score):
        TL.scan({"features": ["ret20"], "universe": ["2330"]}, "2026-10-12")
    assert isinstance(seen["cal"], TL.TradeCal) and seen["cal"].mi("2026-10-12") is rows and calls == [], "scan 帶懶載入的日曆，假評分不抓"


def test_mi_fetch_budget():
    snap = []
    with patched(TL, "market_snapshot", fake_snapshot({}, snap)), patched(TL, "MI_FETCH_MAX", 3):
        c = cal(mi={"2026-10-13": {"1": {}}})
        for d in ("2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"):
            c.mi(d)
        assert c.mi("2026-10-13") == {"1": {}} and c.mi("2026-10-01") == {}
    assert snap == ["2026-10-01", "2026-10-02", "2026-10-05"], snap


def test_trailing_missing_day_filled_up_to_market_date():
    """10-16：Yahoo 還沒出最新交易日 (10-13) 的 K 棒、大盤 mk 已有 10-13 → 用盤後表補結尾那天，追蹤中列「第幾天 / 至今」不停在前一日。
    1303 型：10-08 推薦，10-09 休市 → 10-12 第 1 天、10-13 第 2 天。"""
    B = {"1303": [bar(d, 100.0) for d in TW_NOW if "2026-09-01" <= d <= "2026-10-12"]}   # Yahoo 只到 10-12
    MI = {"2026-10-13": {"1303": {"open": 101.0, "high": 113.0, "low": 100.5, "close": 112.0, "volume": 2e6}},
          "2026-10-12": {"1303": {"close": 100.0}}}

    def led():
        return {"ledger": {"treasure": [], "surge": [{"code": "1303", "name": "南亞", "date": "2026-10-08", "entry": 100.0, "mktEntry": 1000.0, "status": "追蹤"}]}}
    with patched(TL, "bars", lambda code, years="2y": B[code]), patched(TL, "_fmtqik_days", fake_fmtqik()), patched(TL, "market_snapshot", fake_snapshot(MI)):
        new = led(); TL.evaluate(new, MK, cal=cal())
        old = led(); TL.evaluate(old, {d: v for d, v in MK.items() if d <= "2026-10-12"}, cal=cal())   # 大盤也只到 10-12 → 不補 (不超前)
    s = new["ledger"]["surge"][0]
    assert s["days"] == 2 and s["lastDate"] == "2026-10-13" and abs(s["cur"] - 12.0) < 1e-6, s
    o = old["ledger"]["surge"][0]
    assert o["days"] == 1 and o["lastDate"] == "2026-10-12", o


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
