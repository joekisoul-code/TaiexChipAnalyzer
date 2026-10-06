"""挖寶雷達雲端帳本 (pr2 P1) 單元測試：合成日 K、不連網。

    python tests/test_treasure_live_ledger.py
    python -m pytest tests/test_treasure_live_ledger.py

(a) record 寫入 model_ver / th_A / th_Aplus / mkt_bias20 / scan_src；(b) 停損列 21 日後有 fin21、cur 不變；(c) stats 的 by_source 與 by_tier[*].role、B 級無 drift；
(d) alerts 文案不含「命中機率 / 進場 / 持有」；(e) 空帳本 / 全追蹤不崩；(f) backfill 標 backfill=True 與 scan_src=backfill。
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import treasure as T  # noqa: E402
from chip.predict import treasure_live as TL  # noqa: E402

TM = {"trained_at": "2026-10-04 08:36:18", "th_A": 0.5062, "th_Aplus": 0.6362, "gate": {"m_bias20_lt": 0.0},
      "oos": {"tiers": {"A": {"hit": 0.548, "fin": 9.2}, "A+": {"hit": 0.638, "fin": 11.16}, "A-": {"hit": 0.471, "fin": 7.52}, "B+": {"hit": 0.421, "fin": 5.22}, "B": {"hit": 0.387, "fin": 2.86}}},
      "surge": {"oos": {"app_hit": 0.329, "app_fin": 5.65}, "th_top10": 0.3977}}
D0 = "2026-09-01"
DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", "2026-12-31")]
AFTER = [d for d in DATES if d > D0]
MK = {d: 1000.0 for d in DATES}    # 大盤持平 → rel = cur


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _bars(path: list[tuple[float, float]]) -> list[dict]:
    """path = D0 之後每日 (close%, low%)；D0 (含) 之前持平 100。"""
    out = []
    for d in DATES:
        if d <= D0:
            out.append({"date": d, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000})
            continue
        i = AFTER.index(d)
        c, lo = path[i] if i < len(path) else path[-1]
        out.append({"date": d, "open": 100 * (1 + c / 100), "high": 100 * (1 + max(c, 0) / 100), "low": 100 * (1 + lo / 100), "close": 100 * (1 + c / 100), "volume": 1000})
    return out


# 1111 A+：第 3 日達 +6%、第 21 日收 +8% → 命中 (fin21 = cur = 8)
P_HIT = [(2, 1), (4, 3), (7, 6)] + [(7.5, 7)] * 17 + [(8, 7.5)] * 10
# 2222 B：第 3 日盤中 −9% (收 −7%) → 停損未命中、cur = −7；之後反彈到第 21 日 +10 → fin21 = +10
P_STOP = [(-2, -2.5), (-5, -5.5), (-7, -9)] + [(0, -1)] * 17 + [(10, 9)] * 10
# 3333 A：第 10 日達 +6%、第 21 日 +3 → 命中
P_A = [(1, 0)] * 9 + [(6.5, 5)] + [(5, 4)] * 10 + [(3, 2)] * 10
BARS = {"1111": _bars(P_HIT), "2222": _bars(P_STOP), "3333": _bars(P_A), "9999": _bars([(25, 0)] * 30)}
SC = {"date": D0, "pool_n": 80, "mkt_close": 1000.0, "mkt_bias20": -1.25, "mkt_pct": -0.4,
      "treasure": [{"code": "1111", "name": "甲", "close": 100.0, "pct": 2.0, "value": 1e8, "pbr": 1.0, "p": 0.70, "tier": "A+", "ps": 0.2, "m_bias20": -1.25},
                   {"code": "3333", "name": "丙", "close": 100.0, "pct": 1.0, "value": 1e8, "pbr": 1.0, "p": 0.55, "tier": "A", "ps": 0.1, "m_bias20": -1.25},
                   {"code": "2222", "name": "乙", "close": 100.0, "pct": 3.0, "value": 1e8, "pbr": 1.0, "p": 0.45, "tier": "B", "ps": 0.3, "m_bias20": -1.25}],
      "surge": [{"code": "9999", "name": "丁", "close": 100.0, "pct": 5.0, "ps": 0.41}]}


def _fresh():
    return {"ledger": {"treasure": [], "surge": []}, "scans": []}


def _fake_bars(code, years="2y"):
    return BARS.get(code, [])


def _run_days(prev, k):
    for d in AFTER[:k]:
        TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=d)


def test_record_meta_fields():
    prev = TL.record(_fresh(), SC, TM)
    tre = prev["ledger"]["treasure"]
    assert [x["code"] for x in tre] == ["1111", "3333", "2222"]
    for x in tre:
        assert x["model_ver"] == TM["trained_at"] and x["th_A"] == 0.5062 and x["th_Aplus"] == 0.6362
        assert x["mkt_bias20"] == -1.25 and x["scan_src"] == "live" and x["status"] == "追蹤" and "backfill" not in x
    s = prev["ledger"]["surge"][0]
    assert s["model_ver"] == TM["trained_at"] and s["scan_src"] == "live" and s["th_top10"] == 0.3977
    assert prev["scans"][0]["n_A"] == 2 and prev["scans"][0]["mkt_bias20"] == -1.25
    # 舊呼叫方式 (不帶 tm) 仍可用
    p2 = TL.record(_fresh(), SC)
    assert p2["ledger"]["treasure"][0]["model_ver"] is None and p2["ledger"]["treasure"][0]["scan_src"] == "live"


def test_stop_row_gets_fin21_cur_unchanged():
    prev = TL.record(_fresh(), SC, TM)
    with patched(TL, "bars", _fake_bars):
        _run_days(prev, 3)
        by = {x["code"]: x for x in prev["ledger"]["treasure"]}
        b = by["2222"]
        assert b["status"] == "未命中" and "跌破" in b["reason"] and b["days"] == 3 and abs(b["cur"] - (-7.0)) < 1e-6 and b.get("fin21") is None and b["trail"] == b["cur"]
        assert by["1111"]["status"] == "追蹤" and by["3333"]["status"] == "追蹤"
        snap_b = dict(b)
        _run_days(prev, 21)
        by = {x["code"]: x for x in prev["ledger"]["treasure"]}
        b = by["2222"]
        assert abs(b["fin21"] - 10.0) < 1e-6 and b["fin21Date"] == AFTER[20], b
        for k in ("status", "cur", "days", "peak", "trough", "rel", "trail", "evalAt", "reason"):
            assert b[k] == snap_b[k], (k, b[k], snap_b[k])          # 結案值不被第 21 日重算改寫
        assert "fin21_na" not in b
        a = by["1111"]
        assert a["status"] == "命中" and abs(a["fin21"] - a["cur"]) < 1e-6 and abs(a["cur"] - 8.0) < 1e-6 and a["days"] == 21
        assert by["3333"]["status"] == "命中" and abs(by["3333"]["fin21"] - 3.0) < 1e-6
        # 第 22 日再對帳：結案列不再在 todo (fin21 已有)，統計不變
        n = TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=AFTER[21])
        assert n == 0, n
        # 舊格式結案列 (無 fin21、第 21 日結案) 的 fin21 取 cur；早停損的舊列為 None
        assert TL._fin21_of({"status": "回落", "days": 21, "cur": -1.5}) == -1.5 and TL._fin21_of({"status": "未命中", "days": 5, "cur": -9.0}) is None


def test_fin21_na_when_bars_end_early():
    prev = TL.record(_fresh(), SC, TM)
    short = {"2222": [r for r in BARS["2222"] if r["date"] <= AFTER[5]] + [{"date": AFTER[50], "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1}]}   # 第 6 日後停牌、很久以後一根
    with patched(TL, "bars", lambda code, years="2y": short.get(code, BARS.get(code, []))):
        _run_days(prev, 3)
        TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=AFTER[50])
    b = {x["code"]: x for x in prev["ledger"]["treasure"]}["2222"]
    assert b["status"] == "未命中" and b.get("fin21") is None and b.get("fin21_na") is True


def test_fin21_na_when_bars_empty():
    """review2 #20：bars() 回 [] (下市/改代號) 的結案列，超過 MAX_FIN21_WAIT_DAYS 後標 fin21_na，不再每次重抓。"""
    prev = TL.record(_fresh(), SC, TM)
    with patched(TL, "bars", _fake_bars):
        _run_days(prev, 3)
    b = {x["code"]: x for x in prev["ledger"]["treasure"]}["2222"]
    assert b["status"] == "未命中" and b.get("fin21") is None and not b.get("fin21_na")
    calls = []
    def empty(code, years="2y"):
        calls.append(code)
        return [] if code == "2222" else BARS.get(code, [])
    with patched(TL, "bars", empty):
        TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=AFTER[10])          # 距 D0 < 60 日曆日：仍等待
        assert b.get("fin21_na") is None and "2222" in calls
        late = (pd.Timestamp(D0) + pd.Timedelta(days=TL.MAX_FIN21_WAIT_DAYS + 1)).strftime("%Y-%m-%d")
        TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=late)
        assert b.get("fin21_na") is True and b.get("fin21") is None
        snap_b = dict(b)
        calls.clear()
        TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=late)
        assert "2222" not in calls, calls                                   # 不再進 todo
        assert b == snap_b
        # 追蹤中的列 bars 為空不標 fin21_na
        p2 = TL.record(_fresh(), SC, TM)
        with patched(TL, "bars", lambda code, years="2y": []):
            TL.evaluate(p2, MK, max_fetch=10 ** 6, upto=late)
        assert all(x["status"] == "追蹤" and "fin21_na" not in x for x in p2["ledger"]["treasure"])


def test_stats_sources_roles_and_drift():
    prev = TL.record(_fresh(), SC, TM)
    with patched(TL, "bars", _fake_bars):
        _run_days(prev, 21)
    st = TL.stats(prev, TM)
    ts = st["treasure"]
    assert ts["all"]["n"] == 3 and ts["all"]["hit"] == 2 and ts["all"]["n_published"] == 3 and ts["all"]["n_backfill"] == 0
    assert ts["by_source"]["published"]["n"] == 3 and ts["by_source"]["backfill"] is None
    assert ts["signal"]["n"] == 2 and ts["signal"]["rate"] == 1.0 and ts["signal"]["expect"] == [0.55, 0.57] and ts["signal"]["role"] == "signal" and ts["signal"]["drift"] is False
    bt = ts["by_tier"]
    assert bt["A+"]["role"] == "signal" and bt["A"]["role"] == "signal" and bt["B"]["role"] == "descriptive"
    assert bt["A+"]["expect"] == [0.55, 0.57] and "expect" not in bt["B"]
    assert bt["B"]["drift"] is False and "描述級" in bt["B"]["note"] and bt["B"]["rate"] == 0.0
    assert abs(bt["B"]["avg_fin"] - (-7.0)) < 1e-6 and abs(bt["B"]["avg_fin21"] - 10.0) < 1e-6 and bt["B"]["n_fin21"] == 1   # 結案日報酬 vs 21 日報酬分開
    assert bt["A+"]["bt_hit"] == 0.638 and bt["A"]["bt_hit"] == 0.471                                              # 舊鍵保留
    assert ts["roles"] == {"A+": "signal", "A": "signal", "B+": "descriptive", "B": "descriptive"} and ts["expect_AAplus"] == [0.55, 0.57]
    assert ts["by_month"]["2026-09"]["n_A"] == 2 and ts["by_month"]["2026-09"]["avg_fin21"] == 7.0
    # 真實發布 A 級 20 筆命中 20% → 漂移 (只看 published)；同樣 20 筆若全是回填 → 不漂移；B 級 100 筆 5% → 永不漂移
    def rows(tier, n, hits, backfill):
        return [{"code": f"{i:04d}", "date": "2026-07-01", "tier": tier, "status": "命中" if i < hits else "未命中", "cur": 1.0, "rel": 0.0, "trail": 1.0, "days": 21, **({"backfill": True} if backfill else {})} for i in range(n)]
    st2 = TL.stats({"ledger": {"treasure": rows("A", 20, 4, False) + rows("B", 100, 5, False), "surge": []}, "scans": []}, TM)
    assert st2["treasure"]["by_tier"]["A"]["drift"] is True and st2["treasure"]["by_tier"]["A"]["published"]["n"] == 20 and st2["treasure"]["signal"]["drift"] is True
    assert st2["treasure"]["by_tier"]["B"]["drift"] is False and st2["treasure"]["by_tier"]["B"]["below_bt"] is True
    st3 = TL.stats({"ledger": {"treasure": rows("A", 20, 4, True), "surge": []}, "scans": []}, TM)
    assert st3["treasure"]["by_tier"]["A"]["drift"] is False and st3["treasure"]["by_tier"]["A"]["published"] is None and st3["treasure"]["by_source"]["backfill"]["n"] == 20
    # 漂移警報只來自訊號級真實發布
    al2 = TL.alerts({"date": D0, "treasure": [], "surge": []}, st2)
    assert len(al2) == 1 and al2[0]["kind"] == "drift" and "A 級真實發布" in al2[0]["msg"] and "B" not in al2[0]["msg"].split("級")[0]
    assert TL.alerts({"date": D0, "treasure": [], "surge": []}, st3) == []


def test_alert_wording():
    st = TL.stats(TL.record(_fresh(), SC, TM), TM)
    al = TL.alerts(SC, st | {"surge": {**(st.get("surge") or {}), "th_top10": 0.3977}})
    tre = [a for a in al if a["kind"] == "treasure"]
    assert len(tre) == 2 and all(a["tier"] in ("A+", "A") for a in tre)
    for a in tre:
        m = a["msg"]
        assert "模型分 0." in m and "55~57%" in m and "非買賣建議" in m
        for bad in ("命中機率", "進場", "持有", "買進", "賣出"):
            assert bad not in m, (bad, m)
    assert any(a["kind"] == "surge" for a in al)
    assert not any(a["kind"] == "drift" for a in al)


def test_empty_and_tracking_only():
    st = TL.stats(_fresh(), TM)
    assert st["treasure"]["all"] is None and st["treasure"]["signal"] is None and st["treasure"]["by_source"] == {"published": None, "backfill": None} and st["treasure"]["by_tier"] == {}
    assert TL.alerts({"date": None, "treasure": [], "surge": []}, st) == []
    prev = TL.record(_fresh(), SC, TM)      # 全部追蹤中、尚未對帳
    st = TL.stats(prev, TM)
    assert st["treasure"]["all"] is None and st["treasure"]["tracking"] == 3 and st["treasure"]["signal"] is None
    out = TL.alerts(SC, st | {"surge": {**(st.get("surge") or {}), "th_top10": 0.3977}})
    assert len(out) == 3


def test_backfill_marks_source():
    with patched(TL, "scan", lambda tm, date=None: dict(SC, date=date)), patched(TL, "bars", _fake_bars):
        prev = TL.backfill(_fresh(), TM, [D0], MK)
    for x in prev["ledger"]["treasure"] + prev["ledger"]["surge"]:
        assert x["backfill"] is True and x["scan_src"] == "backfill" and x["model_ver"] == TM["trained_at"]
    st = TL.stats(prev, TM)
    assert st["treasure"]["all"] is None   # 只有 D0 一天的 K → 尚未結案


def test_h_constant_matches_backtest():
    assert T.H == 21 and TL.EXPECT_AAPLUS == (0.55, 0.57)


def test_hit_day_and_reason():
    """10-06：結案列記第幾天達標 / 跌破 (挖寶、飆股)，原因寫出天數與日期；同日兩邊都選到互相標記。"""
    sc = dict(SC); sc["surge"] = SC["surge"] + [{"code": "2222", "name": "乙", "close": 100.0, "pct": 3.0, "ps": 0.33}]
    with patched(TL, "bars", _fake_bars):
        prev = TL.record(_fresh(), sc, TM)
        _run_days(prev, 22)
    t = {x["code"]: x for x in prev["ledger"]["treasure"]}; s = {x["code"]: x for x in prev["ledger"]["surge"]}
    # 大盤持平 → 第 2 天 +4% 已贏大盤 4pt (相對達標)，第 3 天收盤 +7% 才是漲幅達標；記第一次
    assert t["1111"]["status"] == "命中" and t["1111"]["hitDay"] == 2 and t["1111"]["hitDate"] == AFTER[1] and t["1111"]["hitBy"] == "相對大盤", t["1111"]
    assert t["1111"]["reason"] == "第 2 天 (" + AFTER[1][5:] + ") 贏大盤 4pt 達標；結案 +8.0% (峰值 +8.0%)", t["1111"]["reason"]
    assert t["2222"]["status"] == "未命中" and t["2222"]["stopDay"] == 3 and "盤中跌破 -8%" in t["2222"]["reason"] and t["2222"]["chk"] == 1, t["2222"]
    assert t["3333"]["hitDay"] == 10 and t["3333"]["peakDay"] == 10
    assert s["9999"]["status"] == "飆" and s["9999"]["hitDay"] == 1 and "先到 +20%" in s["9999"]["reason"], s["9999"]
    assert t["2222"].get("sPick") is True and "sPick" not in t["1111"], "同日也是飆股前 3 → sPick"
    assert s["2222"]["tier"] == "B" and s["2222"]["p"] == 0.45 and "tier" not in s["9999"], "飆股列帶挖寶等級"


def test_closed_rows_annotated_once():
    """舊結案列 (沒有 chk) 補一次「第幾天」，狀態與數值不動。"""
    with patched(TL, "bars", _fake_bars):
        prev = TL.record(_fresh(), SC, TM)
        _run_days(prev, 22)
        for x in prev["ledger"]["treasure"] + prev["ledger"]["surge"]:
            for k in ("hitDay", "hitDate", "hitVal", "hitBy", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate", "chk"):
                x.pop(k, None)
        before = {x["code"]: (x["status"], x["cur"], x["days"]) for x in prev["ledger"]["treasure"]}
        TL.evaluate(prev, MK, max_fetch=10 ** 6, upto=AFTER[25])
    t = {x["code"]: x for x in prev["ledger"]["treasure"]}
    assert all(x.get("chk") == 1 for x in prev["ledger"]["treasure"] + prev["ledger"]["surge"])
    assert t["1111"]["hitDay"] == 2 and t["2222"]["stopDay"] == 3
    assert {c: (x["status"], x["cur"], x["days"]) for c, x in t.items()} == before


def test_market_snapshot_does_not_cache_unpublished_day():
    """10-06：TWSE 當天還沒公布 → 不寫快取 (原本快取 null 30 天，該日永遠掃不到)，往前找到有資料的日子。"""
    writes = []

    def fake_cached(key, ttl, loader, allow_stale=True):
        data = loader()            # 未公布 → 丟例外，不會寫入
        writes.append(key)
        return data

    rows = [["1101", "台泥", "1,000", "1", "100", "30.00", "30.50", "29.90", "30.20", "+", "0.20", "", "", "", "", "10.5"]] * 600
    fields = ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差", "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比"]

    class R:
        def __init__(self, j): self.j = j
        def raise_for_status(self): pass
        def json(self): return self.j

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            return R({"stat": "很抱歉，沒有符合條件的資料!"} if params["date"] == "20261006" else {"tables": [{"fields": fields, "data": rows}]})

    with patched(TL, "cached", fake_cached), patched(TL, "session", lambda: S()):
        D, rows_out = TL.market_snapshot("2026-10-06")
    assert D == "2026-10-05" and "1101" in rows_out, D
    assert writes == ["twse:mi_index_all2:20261005"], writes


def test_close_alerts_recent_signal_rows():
    """10-06：最近結案的 A/A+ 與飆股發「結案」提醒 (B 級不發)，文案寫結果與原因、不含交易指令。"""
    with patched(TL, "bars", _fake_bars):
        prev = TL.record(_fresh(), SC, TM)
        _run_days(prev, 2)
        early = {(a["radar"], a["code"]) for a in TL.close_alerts(prev)}
        assert early == {("s", "9999")}, early          # 飆股第 1 天就先到 +20% → 先發
        _run_days(prev, 22)
    al = TL.close_alerts(prev)
    codes = {(a["radar"], a["code"]) for a in al}
    assert ("t", "1111") in codes and ("t", "3333") in codes, codes
    assert ("s", "9999") not in codes, "超過 4 天前結案的不再提醒"
    assert ("t", "2222") not in codes, "B 級不發結案提醒"
    a1 = next(a for a in al if a["code"] == "1111")
    assert a1["kind"] == "close" and a1["status"] == "命中" and "✓ 命中" in a1["msg"] and "第 2 天" in a1["msg"] and "非買賣建議" in a1["msg"], a1
    for a in al:
        for w in ("進場", "持有", "買進", "賣出", "加碼", "減碼"):
            assert w not in a["msg"], (w, a["msg"])
    assert all(a["date"] >= "2026-09-01" for a in al)
    sc2 = dict(SC); out = TL.alerts(sc2, {}, prev)
    assert any(a["kind"] == "close" for a in out)


def test_pool_rows_restricts_to_training_universe():
    """10-06：候選池只用模型訓練過的股票；舊模型檔沒有 universe 時維持全市場。"""
    rows = {"2330": {"value": 1e10}, "1101": {"value": 1e9}, "9999": {"value": 2e8}}
    assert set(TL.pool_rows(rows, {"universe": ["2330", "1101"]})) == {"2330", "1101"}
    assert set(TL.pool_rows(rows, {})) == {"2330", "1101", "9999"}
    assert TL.RADAR_BT["pool_rule"]["after"]["S3"]["win21"] > TL.RADAR_BT["pool_rule"]["before"]["S3"]["win21"]
    assert TL.RADAR_BT["pool"]["surge"] == 0.16, "原本「一般股票」基準率不可被覆蓋"


def _tiny_model():
    """兩棵樹：特徵 0 > 5 加分、特徵 1 < 0 加分；其餘特徵不影響。"""
    feats = ["ret20", "dd_hi60", "vola20"]
    trees = [{"r": 0, "n": [[0, 5.0, -1, -2, 1]], "v": [0.0, 1.0]}, {"r": 0, "n": [[1, 0.0, -1, -2, 1]], "v": [0.6, 0.0]}]
    return {"features": feats, "init": -0.5, "trees": trees}


def test_explain_ranks_and_formats():
    tm = _tiny_model()
    x = {"ret20": 12.0, "dd_hi60": -20.0, "vola20": 3.0}
    med = {"ret20": 2.0, "dd_hi60": 5.0, "vola20": 3.0}
    w = TL.explain(tm, x, med)
    assert [z["f"] for z in w] == ["ret20", "dd_hi60"], w          # 換成中位數掉分：ret20 (1.0) > dd_hi60 (0.6)；vola20 等於中位不列
    assert w[0]["d"] > w[1]["d"] > 0
    assert w[0]["txt"] == "近 20 日漲幅 +12.0% (今日候選中位 +2.0%)", w[0]["txt"]
    assert "A 級的必要條件" in TL.gate_txt("A", -1.23) and "A+" in TL.gate_txt("A+", -1.0)


def test_alert_carries_why():
    sc = {"date": "2026-09-14", "treasure": [{"code": "2489", "name": "瑞軒", "tier": "A", "p": 0.56, "why": ["近 60 日漲幅 -22.5% (今日候選中位 +2.0%)"], "why_gate": "大盤在月線下"}], "surge": []}
    a = TL.alerts(sc, {})[0]
    assert a["name"] == "瑞軒" and a["why"] and "原因：近 60 日漲幅" in a["msg"] and "非買賣建議" in a["msg"]


def test_score_one_skips_unadjusted_split():
    """近 60 日有單日 >11% 跳動 (減資/分割未還原) → 不評分。"""
    D = "2026-09-14"; days = pd.bdate_range(end=D, periods=80).strftime("%Y-%m-%d").tolist()
    closes = [100.0] * 60 + [5.0] * 20                               # 一天 −95%
    bars = [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1000} for d, c in zip(days, closes)]
    old = TL.bars
    try:
        TL.bars = lambda code, years="2y": bars
        row = {"open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 1000, "value": 5000}
        assert TL.score_one("6949", row, D, pd.DataFrame(), {"features": ["ret20"]}) is None
    finally:
        TL.bars = old


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
