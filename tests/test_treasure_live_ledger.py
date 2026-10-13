"""挖寶雷達雲端帳本 (pr2 P1) 單元測試：合成日 K、不連網。

    python tests/test_treasure_live_ledger.py
    python -m pytest tests/test_treasure_live_ledger.py

(a) record 寫入 model_ver / th_A / th_Aplus / mkt_bias20 / scan_src；(b) 停損列 21 日後有 fin21、cur 不變；(c) stats 的 by_source 與 by_tier[*].role、B 級無 drift；
(d) alerts 文案不含「命中機率 / 進場 / 持有」；(e) 空帳本 / 全追蹤不崩；(f) backfill 標 backfill=True 與 scan_src=backfill。
r6 (10-07)：(g) A/A+ 記錄規則同回測 (30 日曆日內未記過就記，不受前 20 / MAX_ACTIVE；每日 6 檔共用)，B/B+ 舊規則不變；
(h) 常數為訓練窗與正式相同 (2020 起) 的版本、警報 A+ / A 分開寫；(i) 帶到 App 的文字不含禁用詞。
BE-2 (2026-10-13)：(j) 同一推薦日再記一次 (15:40 + 21:45 / 重掃) 仍 ≤ 6 檔、飆股 ≤ topk；(k) is_late 規則 (模型晚於推薦日 / 隔天 09:00 後)；
(l) build() 遲到的執行：已發布的日子不再記、不蓋 scans[D]、不發新推薦提醒；未發布的記成 late + backfill；(m) tag_late 修正既有列、真實發布統計排除 late。
BE-3：(n) 缺當天 ^TWII 不再把大盤當 0% (相對大盤只在兩個收盤都有時算)、追蹤中的列清掉上次的 hit 標記、第 21 日缺大盤收盤不結案 (之後有更晚的 K 才照常結案)；
(o) build() 把 scan 以盤後表補上的加權收盤帶進 evaluate 的大盤序列。
其餘審查項 (2026-10-13)：(p) BE-4 新列記 in_uni；(q) N3 飆股 fin20 (提早出場仍追到第 20 日)；(r) BE-7 stats.surge 真實發布 / 回填與失準；
(s) N4 飆股提醒寫「飆股分」不寫機率；(t) N8/N9/BE-5 常數期間；(u) BE-6 先掃當天再回填；(v) BE-9 帳本截斷保留 A/A+。日曆對齊 (BE-1) 見 tests/test_treasure_calendar.py。
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import sys
import tempfile
import types
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
# r6：App 文字禁用詞 (含後端 JSON 帶進 UI 的字串)
BANNED = ("買進", "賣出", "加碼", "減碼", "放空", "做多", "做空", "建議買", "建議賣", "抄底", "逃頂", "縮小部位", "進場", "持有", "持股水位", "續抱", "試單")
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


def _cand(code, tier, p, pct=1.0):
    return {"code": code, "name": "測" + code[-2:], "close": 100.0, "pct": pct, "value": 1e8, "pbr": 1.0, "p": p, "tier": tier, "ps": 0.1, "m_bias20": -1.25}


def _active_rows(n, date="2026-08-20", tier="B"):
    return [{"code": f"8{i:03d}", "name": "舊", "date": date, "tier": tier, "status": "追蹤", "entry": 100.0, "p": 0.4, "days": 5} for i in range(n)]


def test_record_a_rule_ignores_top20_and_max_active():
    """r6 (radarshadow F6)：A/A+ 用回測規則 — 追蹤中已滿 40 檔、或排在第 20 名之後，A/A+ 仍記；B/B+ 維持舊規則 (被擋)。"""
    prev = _fresh(); prev["ledger"]["treasure"] = _active_rows(TL.MAX_ACTIVE)
    tre = [_cand(f"{1000 + i}", "B", 0.40 - i * 0.001) for i in range(22)] + [_cand("2001", "A", 0.39), _cand("2002", "A+", 0.385)]   # A 排第 23、24 名 (測順序無關)
    sc = dict(SC, treasure=tre, surge=[])
    out = TL.record(prev, sc, TM)
    new = [x for x in out["ledger"]["treasure"] if x["date"] == D0]
    assert [x["code"] for x in new] == ["2001", "2002"], [x["code"] for x in new]
    assert all(x["rec_rule"] == TL.REC_RULE_A == "A30" for x in new)
    # 同樣候選、但追蹤中只有 0 檔：B 只看前 20、A 照記；每日 6 檔上限共用
    out2 = TL.record(_fresh(), sc, TM)
    new2 = out2["ledger"]["treasure"]
    assert len(new2) == TL.PER_DAY == 6 and [x["code"] for x in new2] == [f"{1000 + i}" for i in range(6)], [x["code"] for x in new2]
    assert all(x["rec_rule"] == TL.REC_RULE_B == "B20" for x in new2)
    # A 依 p 排在前面 (正式 scan 已依 p 排序) → 先記 A，B 補到 6 檔
    sc3 = dict(SC, treasure=[_cand("2001", "A+", 0.70), _cand("2002", "A", 0.60)] + [_cand(f"{1000 + i}", "B", 0.40 - i * 0.001) for i in range(30)], surge=[])
    new3 = TL.record(_fresh(), sc3, TM)["ledger"]["treasure"]
    assert [x["code"] for x in new3] == ["2001", "2002", "1000", "1001", "1002", "1003"] and [x["rec_rule"] for x in new3] == ["A30", "A30"] + ["B20"] * 4
    # 10 檔 A：每日上限 6 檔仍適用 (回測 cap 6)，依 p 取前 6
    sc4 = dict(SC, treasure=[_cand(f"{3000 + i}", "A", 0.70 - i * 0.01) for i in range(10)], surge=[])
    p4 = _fresh(); p4["ledger"]["treasure"] = _active_rows(TL.MAX_ACTIVE + 5)
    new4 = [x for x in TL.record(p4, sc4, TM)["ledger"]["treasure"] if x["date"] == D0]
    assert [x["code"] for x in new4] == [f"{3000 + i}" for i in range(6)]


def test_record_a_cooldown_30_calendar_days():
    """A/A+：同檔 30 個日曆日內記過 (任何等級、不論狀態) 才跳過；舊規則的「未命中 60 日不重複」「追蹤中不重複」只管 B/B+。"""
    def led(code, date, tier, status, evalAt=None):
        return {"code": code, "name": "x", "date": date, "tier": tier, "status": status, "entry": 100.0, "p": 0.5, "days": 21, **({"evalAt": evalAt} if evalAt else {})}
    d29 = (pd.Timestamp(D0) - pd.Timedelta(days=29)).strftime("%Y-%m-%d")
    d30 = (pd.Timestamp(D0) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    d40 = (pd.Timestamp(D0) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    prev = _fresh()
    prev["ledger"]["treasure"] = [led("4001", d29, "B", "未命中", d29),          # 29 天前記過 (B 級) → A 跳過
                                  led("4002", d30, "A", "命中", d29),            # 剛好 30 天 → A 可再記
                                  led("4003", d40, "A", "未命中", d29),          # 40 天前、29 天前未命中 → A 可記 (舊規則會擋 60 日)
                                  led("4004", d40, "B", "追蹤"),                 # 40 天前、仍追蹤中 → A 可記 (回測沒有「追蹤中」限制)
                                  led("5001", d40, "A", "未命中", d29),          # B 級候選：未命中 60 日內 → 舊規則擋
                                  led("5002", d40, "B", "追蹤"),                 # B 級候選：追蹤中 → 舊規則擋
                                  {"code": "4005", "date": "壞日期", "tier": "A", "status": "命中"}]   # 日期壞掉的舊列：略過不崩
    sc = dict(SC, treasure=[_cand("4001", "A+", 0.80), _cand("4002", "A", 0.75), _cand("4003", "A", 0.70), _cand("4004", "A", 0.65), _cand("4005", "A", 0.62),
                            _cand("5001", "B", 0.45), _cand("5002", "B", 0.44), _cand("5003", "B", 0.43)], surge=[])
    new = [x for x in TL.record(prev, sc, TM)["ledger"]["treasure"] if x["date"] == D0]
    assert [x["code"] for x in new] == ["4002", "4003", "4004", "4005", "5003"], [x["code"] for x in new]
    assert {x["code"]: x["rec_rule"] for x in new} == {"4002": "A30", "4003": "A30", "4004": "A30", "4005": "A30", "5003": "B20"}
    # 同一天再跑一次：不重複
    again = TL.record(prev, sc, TM)
    assert len([x for x in again["ledger"]["treasure"] if x["date"] == D0]) == 5
    # 隔天：昨天剛記的 A 在 30 日內 → 不再記
    nxt = AFTER[0]
    sc2 = dict(sc, date=nxt)
    new2 = [x for x in TL.record(again, sc2, TM)["ledger"]["treasure"] if x["date"] == nxt]
    assert all(x["code"] not in ("4002", "4003", "4004", "4005") for x in new2), [x["code"] for x in new2]
    assert TL._recent_codes([{"code": "1", "date": D0}], "壞日期", 30) == set()
    # 回填較早的日子：之後已記的列 (30 日內) 也算 → 不擠在一起
    later = (pd.Timestamp(D0) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    assert TL._recent_codes([{"code": "9", "date": later}], D0, 30) == {"9"}


def test_record_b_only_day_unchanged():
    """沒有 A/A+ 的日子：結果與舊規則完全相同 (前 20、追蹤上限 40、每日 6 檔)。"""
    prev = _fresh(); prev["ledger"]["treasure"] = _active_rows(TL.MAX_ACTIVE - 2)
    sc = dict(SC, treasure=[_cand(f"{6000 + i}", "B+" if i % 2 else "B", 0.6 - i * 0.01) for i in range(30)], surge=[])
    new = [x for x in TL.record(prev, sc, TM)["ledger"]["treasure"] if x["date"] == D0]
    assert [x["code"] for x in new] == ["6000", "6001"] and all(x["rec_rule"] == "B20" for x in new)


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
    assert ts["signal"]["n"] == 2 and ts["signal"]["rate"] == 1.0 and ts["signal"]["expect"] == list(TL.EXPECT_AAPLUS) and ts["signal"]["role"] == "signal" and ts["signal"]["drift"] is False
    # r6：依記錄規則分開 (新列 A30；沒有 rec_rule 的舊列算 legacy)
    assert ts["signal"]["by_rule"]["A30"]["n"] == 2 and ts["signal"]["by_rule"]["legacy"] is None
    for x in prev["ledger"]["treasure"]:
        x.pop("rec_rule", None)
    ts_old = TL.stats(prev, TM)["treasure"]
    assert ts_old["signal"]["by_rule"] == {"A30": None, "legacy": ts["signal"]["by_rule"]["A30"]}
    bt = ts["by_tier"]
    assert bt["A+"]["role"] == "signal" and bt["A"]["role"] == "signal" and bt["B"]["role"] == "descriptive"
    assert bt["A+"]["expect"] == list(TL.EXPECT_AAPLUS) and "expect" not in bt["B"]
    assert bt["B"]["drift"] is False and "描述級" in bt["B"]["note"] and bt["B"]["rate"] == 0.0
    assert abs(bt["B"]["avg_fin"] - (-7.0)) < 1e-6 and abs(bt["B"]["avg_fin21"] - 10.0) < 1e-6 and bt["B"]["n_fin21"] == 1   # 結案日報酬 vs 21 日報酬分開
    assert bt["A+"]["bt_hit"] == TL.RADAR_TIER["A+"]["hit"] and bt["A"]["bt_hit"] == TL.RADAR_TIER["A"]["hit"]       # 10-07 時點正確回測 (鍵名不變)
    assert st["surge"]["bt_hit"] == TL.RADAR_BT["S3"]["surge"]
    assert ts["roles"] == {"A+": "signal", "A": "signal", "B+": "descriptive", "B": "descriptive"} and ts["expect_AAplus"] == list(TL.EXPECT_AAPLUS)
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
    lo, hi = round(TL.EXPECT_AAPLUS[0] * 100), round(TL.EXPECT_AAPLUS[1] * 100)
    for a in tre:
        m = a["msg"]
        assert "模型分 0." in m and f"{lo}~{hi}%" in m and "非買賣建議" in m
        # r6：A+ 與 A 的歷史命中分開寫 (RADAR_TIER)
        assert f"A+ 約 {round(TL.RADAR_TIER['A+']['hit'] * 100)}%" in m and f"、A 約 {round(TL.RADAR_TIER['A']['hit'] * 100)}%" in m, m
        for bad in ("命中機率",) + BANNED:
            assert bad not in m, (bad, m)
    # RADAR_TIER 缺值 (舊常數) 不崩：退回 A+ 用 RADAR_BT、A 略過；全缺 → 只寫合計區間
    with patched(TL, "RADAR_TIER", {}):
        t = TL.tier_expect_txt()
        assert "A+ 約" in t and "、A 約" not in t and f"{lo}~{hi}%" in t, t
        with patched(TL, "RADAR_BT", {}):
            assert TL.tier_expect_txt() == f"A/A+ 命中約 {lo}~{hi}%"
    assert any(a["kind"] == "surge" for a in al)
    assert not any(a["kind"] == "drift" for a in al)


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _strings(k); yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)


def test_ui_strings_no_banned_words():
    """r6：雲端 JSON 會帶到 App 的文字 (model.radar_bt、stats 說明、各種警報) 不含禁用詞。"""
    prev = TL.record(_fresh(), SC, TM)
    with patched(TL, "bars", _fake_bars):
        _run_days(prev, 22)
    st = TL.stats(prev, TM)
    weak = {"ledger": {"treasure": [{"code": "1111", "name": "甲", "date": "2026-09-01", "tier": "A", "status": "追蹤", "days": 3, "cur": -6.2}], "surge": []}}
    texts = list(_strings(TL.RADAR_BT)) + list(_strings(st)) + [TL.NOTE_TIERS, TL.tier_expect_txt()]
    texts += [a["msg"] for a in TL.alerts(SC, st | {"surge": {**(st.get("surge") or {}), "th_top10": 0.3977}}, prev) + TL.weak_alerts(weak)]
    for t in texts:
        for w in BANNED:
            assert w not in t, (w, t)


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
    assert T.H == 21 and TL.EXPECT_AAPLUS == (0.49, 0.54)   # r6 10-07：時點正確、訓練窗與正式相同 (2020 起)
    # 預期區間 = RADAR_BT A∪A+ 10 組命中範圍；A+ 高於 A∪A+ 高於單獨 A (RADAR_TIER 的 A 不含 A+)
    lo, hi = TL.RADAR_BT["A"]["hit_rng"]
    assert abs(TL.EXPECT_AAPLUS[0] - lo) < 0.006 and abs(TL.EXPECT_AAPLUS[1] - hi) < 0.006
    assert TL.RADAR_TIER["A+"]["hit"] > TL.RADAR_BT["A"]["hit"] > TL.RADAR_TIER["A"]["hit"]
    assert TL.RADAR_TIER["A+"]["n"] + TL.RADAR_TIER["A"]["n"] == TL.RADAR_BT["A"]["n"]
    assert TL.RADAR_BT["method"]["train_start"] == "2020-01-01" and TL.RADAR_BT["method"]["old"]["A"] == {"win21": 0.717, "hit": 0.563}
    assert TL.RADAR_BT["method"]["old"]["S3"] == {"win21": 0.554, "surge": 0.33}
    assert "2020" in TL.RADAR_BT["note"] and "2020" in TL.RADAR_DAY0["note"]


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
    PR = TL.RADAR_BT["pool_rule"]
    assert PR["after"]["S3"]["win21"] > PR["before"]["S3"]["win21"]
    # r6：同 2020 訓練窗重算；after = RADAR_BT 本身 (同一批 10 組)，差值附 3 個月區塊 bootstrap 範圍
    assert "2020" in PR["window"] and PR["after"]["A"]["win21"] == TL.RADAR_BT["A"]["win21"] and PR["within_noise"] in (True, False)
    assert all(len(v["ci95"]) == 2 and v["ci95"][0] <= v["d"] <= v["ci95"][1] for v in PR["diff"].values())
    assert TL.RADAR_BT["pool"]["n"] > 10000 and 0.1 < TL.RADAR_BT["pool"]["surge"] < 0.3, "「一般股票」基準率不可被覆蓋"
    assert TL.RADAR_BT["method"]["pit"] and TL.RADAR_BT["A"]["win21"] < TL.RADAR_BT["method"]["old"]["A"]["win21"], "10-07 誠實回測 (時點正確宇宙) 應低於舊的後見之明版"


def test_path_prob_and_weak_alert():
    """10-06：推薦後第 k 天漲跌 → 歷史命中機率查表；第 3 天 ≤ −5% 的 A/飆股發「轉弱」提醒 (B 不發)。"""
    p3 = TL.path_prob("A", 3, -6.0); assert p3["k"] == 3 and p3["bin"] == "−8~−5%" and p3["hit"] <= 0.1, p3
    assert TL.path_prob("A", 4, 6.0)["k"] == 3 and TL.path_prob("A", 4, 6.0)["bin"] == "≥+5%"
    assert TL.path_prob("S3", 0, 1.0) is None and TL.path_prob("S3", 12, -9)["k"] == 10
    assert TL.path_prob("A", 3, 2.0)["bin"] == "−2~+2%" and TL.path_prob("A", 3, 5.0)["bin"] == "+2~+5%" and TL.path_prob("A", 3, -8.0)["bin"] == "≤−8%"
    prev = {"ledger": {"treasure": [{"code": "1111", "name": "甲", "date": "2026-09-01", "tier": "A", "status": "追蹤", "days": 3, "cur": -6.2, "lastDate": "2026-09-04"},
                                    {"code": "2222", "name": "乙", "date": "2026-09-01", "tier": "B", "status": "追蹤", "days": 3, "cur": -9.0},
                                    {"code": "3333", "name": "丙", "date": "2026-09-01", "tier": "A", "status": "追蹤", "days": 4, "cur": -9.0}],
                       "surge": [{"code": "9999", "name": "丁", "date": "2026-09-01", "status": "追蹤", "days": 3, "cur": -8.5}]}}
    al = TL.weak_alerts(prev); codes = [a["code"] for a in al]
    assert codes == ["1111", "9999"], codes
    assert "第 3 天 -6.2%" in al[0]["msg"] and "非買賣建議" in al[0]["msg"] and al[0]["kind"] == "weak"
    for a in al:
        for w in ("進場", "持有", "買進", "賣出", "加碼", "減碼"):
            assert w not in a["msg"], (w, a["msg"])
    assert TL.RADAR_BT["path"]["A"]["3"] and TL.RADAR_BT["fail"]["early"]["A"]["exit"] < TL.RADAR_BT["fail"]["early"]["A"]["hold"]
    # 10-07：飆股表只用「仍在追蹤」的推薦 (已飆 +20% 結案的不算) → 第 15 天還漲 ≥5% 的飆機率不可能像舊表 64% 那麼高
    s15 = TL.path_prob("S3", 16, 6.0); assert s15["k"] == 15 and s15["hit"] < 0.4 and s15.get("nd", 0) >= s15["n"], s15
    d0 = TL.RADAR_BT["day0"]; assert len(d0["hit"]) == len(d0["edges"]) + 1 and d0["hit"] == sorted(d0["hit"]), d0
    assert TL.day0_prob(0.5, 0.5) == d0["hit"][0] and TL.day0_prob(0.95, 0.5) == d0["hit"][-1] and TL.day0_prob(None, 0.5) is None


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



def _days(start, n):
    out, d = [], pd.Timestamp(start)
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
    return out


def test_px_basis_anchor_and_repair():
    """r7：Yahoo 事後因配股/減資把歷史回溯縮放 (推薦日收盤 90 vs 帳本 entry 100) → 以同一序列推薦日收盤當基準；
    已結案的錯誤列一次性重算 (原結果存 orig)，停損日語意同每日對帳；差 <1% 不動。"""
    ds = _days("2026-07-01", 25); D = ds[0]
    # Yahoo 縮放後：推薦日 90，第 2 天盤中 84 (相對 90 = -6.7%，未破 -8%；相對原始 100 = -16% 假停損)，第 5 天收 96 (+6.7% 達標)，之後持平 97
    closes = [90, 88, 87, 89, 92, 96] + [97] * 19
    lows = [c - 1 for c in closes]; lows[2] = 84
    b = [{"date": d, "open": c, "high": c + 1, "low": l, "close": c, "volume": 1000} for d, c, l in zip(ds, closes, lows)]
    mk = {d: 1000.0 for d in ds}
    x = {"code": "9001", "name": "測", "date": D, "entry": 100.0, "tier": "A", "status": "未命中", "cur": -16.0, "fin21": -3.0, "days": 2, "trail": -16.0, "chk": 1,
         "reason": "第 2 天盤中跌破 -8%"}
    prev = {"ledger": {"treasure": [x], "surge": []}}
    with patched(TL, "bars", lambda code, years="2y": b):
        TL.evaluate(prev, mk, max_fetch=10, upto=ds[-1])
    assert x["px_adj"] == 0.9 and x["px_fix"] == 1 and x["pxk"] == 1 and x["orig"]["status"] == "未命中", x
    assert x["status"] == "命中" and x["hitDay"] == 5 and abs(x["fin21"] - (97 / 90 - 1) * 100) < 0.01, x
    assert x["entry"] == 100.0, "畫面顯示的推薦價不改"
    # 一致的列：只做檢查、不重算 (結果維持原樣)
    y = {"code": "9002", "date": D, "entry": 90.2, "tier": "A", "status": "回落", "cur": 1.0, "fin21": 7.5, "days": 21, "trail": 1.0, "chk": 1}
    prev2 = {"ledger": {"treasure": [y], "surge": []}}
    with patched(TL, "bars", lambda code, years="2y": b):
        TL.evaluate(prev2, mk, max_fetch=10, upto=ds[-1])
    assert y["pxk"] == 1 and "px_adj" not in y and y["status"] == "回落" and y["cur"] == 1.0 and "orig" not in y, y
    # 停損日語意：晚對帳也以停損當天結案 (cur = 停損日收盤，days = 停損日序)
    b2 = [dict(r) for r in b]; b2[3]["low"] = 70; b2[3]["close"] = 80
    z = {"code": "9003", "date": D, "entry": 90.0, "tier": "A", "status": "追蹤"}
    with patched(TL, "bars", lambda code, years="2y": b2):
        TL.evaluate({"ledger": {"treasure": [z], "surge": []}}, mk, max_fetch=10, upto=ds[-1])
    assert z["status"] == "未命中" and z["stopDay"] == 3 and z["days"] == 3 and abs(z["cur"] - (80 / 90 - 1) * 100) < 0.01, z


def test_px_basis_surge_fake_surge_fixed():
    """r7：減資回溯放大 (Yahoo 推薦日 133.3 vs 帳本 100) → 用原始 entry 會出現假飆股；改用同序列基準後不是飆股。"""
    ds = _days("2026-07-01", 22); D = ds[0]
    closes = [133.3] + [135.0] * 21
    b = [{"date": d, "open": c, "high": c + 2, "low": c - 2, "close": c, "volume": 1000} for d, c in zip(ds, closes)]
    x = {"code": "9004", "date": D, "entry": 100.0, "status": "飆", "cur": 35.0, "days": 1, "trail": 35.0, "chk": 1}
    with patched(TL, "bars", lambda code, years="2y": b):
        TL.evaluate({"ledger": {"treasure": [], "surge": [x]}}, {d: 1000.0 for d in ds}, max_fetch=10, upto=ds[-1])
    assert x["status"] == "未飆" and x["px_fix"] == 1 and x["orig"]["status"] == "飆", x


def test_score_one_rescaled_history_no_fake_jump():
    """r7：回填日之後有配股/減資 → Yahoo 歷史已縮放，MI 原始價換到同一基準，不再出現 D 日假跳動 (原本 >11% 會被當壞資料略過、5~10% 會扭曲特徵)。"""
    ds = _days("2026-03-01", 130); D = ds[-1]
    rows = [{"date": d, "open": 50.0 + i * 0.01, "high": 50.5 + i * 0.01, "low": 49.5 + i * 0.01, "close": 50.0 + i * 0.01, "volume": 1000 + i} for i, d in enumerate(ds)]
    mkf = pd.DataFrame({"date": ds, "m_close": 1000.0, "m_ret1": 0.0, "m_bias20": -1.0, "m_ret20": 0.0})
    row = {"open": 100.0, "high": 101.0, "low": 99.0, "close": rows[-1]["close"] * 2, "volume": 5000, "value": 5e8}   # MI 原始價 = Yahoo 縮放後的 2 倍 (1:2 減資前)
    with patched(TL, "bars", lambda code, years="2y": rows):
        got = TL.score_one("9005", row, D, mkf, {"features": ["ret20", "m_bias20"], "trees": [], "init": 0.0, "th_A": 0.5, "gate": {"m_bias20_lt": 0.0}})
    assert got is not None and abs(got["_x"]["ret20"]) < 5, got


def test_exdiv_archive_parse_and_snapshot():
    """r7：除權息預告表 (TWT48U) 每日快照：民國日期轉換、待公告 → None、同日覆蓋、抓不到不寫。"""
    import json as _j
    import tempfile
    from chip.predict import exdiv_archive as XD
    j = {"stat": "OK", "fields": ["除權除息日期", "股票代號", "名稱", "除權息", "無償配股率", "現金增資配股率", "現金增資認購價", "現金股利", "詳細資料"],
         "data": [["115年10月15日", "1463", "強盛新", "息", "0.00000000", "0.00000000", "0.00000000", "0.12730000", "x"],
                  ["115年10月20日", "1711", "永光", "權", "0.00000000", "0.09128214", "尚未公告", "0.00000000", "x"]]}
    rows = XD.parse(j)
    assert rows[0] == ["2026-10-15", "1463", "強盛新", "息", 0.0, 0.0, 0.0, 0.1273] and rows[1][6] is None, rows
    for bad in ({"stat": "OK", "data": []}, {"stat": "很抱歉"}, None):
        try:
            XD.parse(bad); assert False, "should raise"
        except LookupError:
            pass
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "exdiv.jsonl"
        assert XD.snapshot(path, cache=False, rows=rows, asof="2026-10-07") == 1
        assert XD.snapshot(path, cache=False, rows=rows[:1], asof="2026-10-07") == 1, "同日覆蓋"
        assert XD.snapshot(path, cache=False, rows=rows, asof="2026-10-08") == 2
        old = XD.fetch
        try:
            XD.fetch = lambda: (_ for _ in ()).throw(LookupError("down"))
            assert XD.snapshot(path, cache=False, asof="2026-10-09") == 2, "抓不到 → 不寫"
        finally:
            XD.fetch = old
        lines = [_j.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        assert [ln["ts"] for ln in lines] == ["2026-10-07", "2026-10-08"] and lines[0]["n"] == 1


# ------------------------------------------------------------------ BE-2 (2026-10-13)：遲到的執行不把推薦記成真實發布
def test_record_same_date_caps_across_calls():
    """15:40 + 21:45 兩次準時發布 (或重掃) 同一推薦日：每日 6 檔上限以該日已有的真實發布列起算；飆股每日最多 topk 檔。"""
    sc1 = dict(SC, treasure=[_cand(f"{3000 + i}", "A", 0.70 - i * 0.01) for i in range(6)], surge=[{"code": f"900{i}", "name": "飆", "close": 100.0, "pct": 3.0, "ps": 0.4} for i in range(1, 4)])
    sc2 = dict(SC, treasure=[_cand(f"{3100 + i}", "A", 0.80 - i * 0.01) for i in range(10)], surge=[{"code": f"910{i}", "name": "飆", "close": 100.0, "pct": 3.0, "ps": 0.5} for i in range(1, 4)])
    prev = TL.record(_fresh(), sc1, TM)
    prev = TL.record(prev, sc2, TM)
    tre = [x for x in prev["ledger"]["treasure"] if x["date"] == D0]
    assert len(tre) == TL.PER_DAY == 6 and {x["code"] for x in tre} == {f"{3000 + i}" for i in range(6)}, [x["code"] for x in tre]
    sur = [x for x in prev["ledger"]["surge"] if x["date"] == D0]
    assert [x["code"] for x in sur] == ["9001", "9002", "9003"], [x["code"] for x in sur]
    # 第一次只記到 4 檔 → 第二次最多補 2 檔；topk = 2 的模型 → 每日 2 檔
    tm2 = dict(TM, surge={**TM["surge"], "def": {"topk": 2}})
    p2 = TL.record(_fresh(), dict(sc1, treasure=sc1["treasure"][:4]), tm2)
    p2 = TL.record(p2, sc2, tm2)
    assert len([x for x in p2["ledger"]["treasure"] if x["date"] == D0]) == 6 and len([x for x in p2["ledger"]["surge"] if x["date"] == D0]) == 2
    # 回填列不佔真實發布的名額 (同 refined_fix：起算只數非回填)
    p3 = _fresh(); p3["ledger"]["treasure"] = [dict(_cand(f"{5000 + i}", "B", 0.3), date=D0, status="追蹤", backfill=True) for i in range(6)]
    assert len([x for x in TL.record(p3, sc1, TM)["ledger"]["treasure"] if x["date"] == D0 and not x.get("backfill")]) == 6


def test_is_late_rule():
    D = "2026-10-08"
    assert not TL.is_late(D, "2026-10-08 21:45:00", "2026-10-04 08:36:18")
    assert not TL.is_late(D, "2026-10-09 08:59:59") and TL.is_late(D, "2026-10-09 09:00:00"), "隔天 09:00 (台北) 起算遲到"
    assert TL.is_late(D, "2026-10-08 15:40:00", "2026-10-11 08:51:40"), "模型在推薦日之後才訓練 (週日重訓重掃上週五)"
    assert not TL.is_late(D, None, "2026-10-08 06:00:00") and not TL.is_late(D) and not TL.is_late("壞日期", "2030-01-01 00:00:00", "2030-01-01")
    assert TL.is_late(D, dt.datetime(2026, 10, 12, 15, 40, tzinfo=TL.config.TZ)) and not TL.is_late(D, dt.datetime(2026, 10, 8, 15, 40))


def _fake_dt(now):
    """假的 datetime 模組：只有 datetime.now() 回傳固定時刻 (台北)，其餘同 datetime。"""
    class _DT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(now.year, now.month, now.day, now.hour, now.minute, now.second, tzinfo=tz)
    ns = types.SimpleNamespace(**{k: getattr(dt, k) for k in dir(dt) if not k.startswith("__")})
    ns.datetime = _DT
    return ns


def _build(prev, sc, tm, mk=None, scan_fn=None, backfill_days=0, now=None):
    """scan_fn：依日期回傳掃描 (預設固定回傳 sc)；now：假的「現在」(BE-6 測試用)。"""
    tm = {"trees": [{}], **tm}
    net = []

    def no_net(*a, **k):                    # TradeCal 會吞掉例外 → 記下呼叫、最後檢查 (日 K 為空 → 不該用到日曆 / 盤後表)
        net.append(a)
        raise LookupError("不應連網")
    with tempfile.TemporaryDirectory() as td, patched(TL, "LOCAL", Path(td) / "tl.json"), patched(TL, "load_prev", lambda: prev), \
            patched(TL, "twii_hist", lambda: dict(mk or {})), patched(TL, "scan", scan_fn or (lambda tm_, date=None: dict(sc))), \
            patched(TL.M, "load_json", lambda name: tm if name == "treasure_model" else None), patched(TL, "bars", lambda code, years="2y": []), \
            patched(TL, "_fmtqik_days", no_net), patched(TL, "market_snapshot", no_net), patched(TL, "dt", _fake_dt(now) if now else dt):
        out = TL.build(date=sc["date"], backfill_days=backfill_days)
    assert net == [], net
    return out


def test_build_late_runs_not_recorded_as_published():
    today = dt.datetime.now(TL.config.TZ).date()
    D, D3 = today.isoformat(), (today - dt.timedelta(days=3)).isoformat()
    old_tm, new_tm = dict(TM, trained_at=(today - dt.timedelta(days=5)).isoformat() + " 08:00:00"), dict(TM, trained_at=(today + dt.timedelta(days=1)).isoformat() + " 08:00:00")
    # 準時 → live
    out = _build(_fresh(), dict(SC, date=D), old_tm)
    assert "late" not in out["scan"] and {x["scan_src"] for x in out["ledger"]["treasure"]} == {"live"} and not any(x.get("backfill") for x in out["ledger"]["treasure"] + out["ledger"]["surge"])
    assert any(a["kind"] == "treasure" for a in out["alerts"]) and any(a["kind"] == "surge" for a in out["alerts"])
    # 已發布的日子、事後的新模型重掃 → 不記、不蓋 scans[D]、不發新推薦提醒
    pub = TL.record(_fresh(), dict(SC, date=D, treasure=SC["treasure"][:1], surge=[]), old_tm)
    scan_before = json.dumps(pub["scans"], sort_keys=True, ensure_ascii=False)
    out = _build(pub, dict(SC, date=D), new_tm)
    assert out["scan"]["late"] is True and [x["code"] for x in out["ledger"]["treasure"]] == ["1111"] and out["ledger"]["surge"] == []
    assert json.dumps(out["scans"], sort_keys=True, ensure_ascii=False) == scan_before, "scans[D] 維持當時發布的內容"
    assert not any(a["kind"] in ("treasure", "surge") for a in out["alerts"]), out["alerts"]
    # 隔天 09:00 後才跑 (平日休市 / MI_INDEX 未出) 且那天已發布 → 同樣不記
    pub3 = TL.record(_fresh(), dict(SC, date=D3, treasure=SC["treasure"][:1], surge=[]), old_tm)
    out = _build(pub3, dict(SC, date=D3), old_tm)
    assert out["scan"]["late"] is True and len(out["ledger"]["treasure"]) == 1 and out["ledger"]["surge"] == []
    # 那天沒發布過 → 記成 late + backfill (App 與 stats 都當回填)，之後同日再遲到重掃也不再追加
    out = _build(_fresh(), dict(SC, date=D3), old_tm)
    rows = out["ledger"]["treasure"] + out["ledger"]["surge"]
    assert len(rows) == 4 and all(x["backfill"] is True and x["scan_src"] == "late" for x in rows), rows
    assert D3 in {s["date"] for s in out["scans"]} and not any(a["kind"] in ("treasure", "surge") for a in out["alerts"])
    again = _build({"ledger": out["ledger"], "scans": out["scans"]}, dict(SC, date=D3, treasure=[_cand("7777", "A+", 0.9)]), new_tm)
    assert len(again["ledger"]["treasure"]) == 3


def test_tag_late_migration_and_published_stats():
    def row(code, date, mv, **kw):
        return {"code": code, "name": "x", "date": date, "tier": "A", "status": "命中", "cur": 5.0, "rel": 1.0, "trail": 5.0, "days": 21, "scan_src": "live",
                **({"model_ver": mv} if mv else {}), **kw}
    tre = [row("4989", "2026-10-02", "2026-10-04 08:36:18"), row("8261", "2026-10-02", None), row("2330", "2026-10-08", "2026-10-04 08:36:18"),
           row("1101", "2026-07-01", "2026-10-04 08:36:18", backfill=True, scan_src="backfill"), row("2002", "2026-10-07", "2026-10-07 08:00:00")]
    sur = [{"code": "2383", "date": "2026-10-08", "model_ver": "2026-10-11 08:51:40", "status": "追蹤", "scan_src": "live"},
           {"code": "1303", "date": "2026-10-08", "model_ver": "2026-10-04 08:36:18", "status": "追蹤", "scan_src": "live"}]
    prev = {"ledger": {"treasure": tre, "surge": sur}, "scans": []}
    assert TL.tag_late(prev) == 3
    late = {(x["code"], x["date"]) for x in tre + sur if x.get("scan_src") == "late"}
    assert late == {("4989", "2026-10-02"), ("8261", "2026-10-02"), ("2383", "2026-10-08")}, late
    assert all(x["backfill"] is True for x in tre + sur if x.get("scan_src") == "late")
    assert tre[3]["scan_src"] == "backfill" and "backfill" not in tre[2] and "backfill" not in tre[4]
    assert TL.tag_late(prev) == 0, "冪等"
    st = TL.stats(prev, TM)["treasure"]
    assert st["by_source"]["published"]["n"] == 2 and st["signal"]["published"]["n"] == 2 and st["by_tier"]["A"]["published"]["n"] == 2
    # 只標了 scan_src='late' 沒標 backfill 的列 (保險)：真實發布統計仍排除
    next(x for x in tre if x["code"] == "2330")["scan_src"] = "late"           # (stats 會就地排序帳本 → 依代號找)
    assert TL.stats(prev, TM)["treasure"]["by_source"]["published"]["n"] == 1


# ------------------------------------------------------------------ BE-3：缺當天 ^TWII 不把大盤當 0%
MK_UP = {d: (1000.0 if d <= D0 else 1025.0) for d in DATES}     # 推薦後大盤 +2.5%


def _row(code="7777"):
    return {"code": code, "name": "測", "date": D0, "entry": 100.0, "mktEntry": 1000.0, "tier": "A", "status": "追蹤", "p": 0.6}


def test_stale_hit_keys_cleared_and_no_fake_rel_hit():
    """第 3 天收 +4.6%、大盤 +2.5% (贏 2.1pt)：缺第 3 天 ^TWII 時不算相對大盤 (原本當 0% → 假達標)；上次殘留的 hit 標記被清掉。"""
    b = _bars([(1, 0.5), (2, 1.5), (4.6, 4)] + [(3, 2.5)] * 30)
    x = _row()
    mk_miss = {k: v for k, v in MK_UP.items() if k != AFTER[2]}
    with patched(TL, "bars", lambda code, years="2y": b):
        TL.evaluate({"ledger": {"treasure": [x], "surge": []}}, mk_miss, upto=AFTER[2])
        assert "hitDay" not in x and x["status"] == "追蹤" and x["rel"] is None and abs(x["relMax"] - (2 - 2.5)) < 1e-6, x
        x.update(hitDay=3, hitDate=AFTER[2], hitVal=4.6, hitBy="相對大盤")       # 舊版缺 ^TWII 那次留下的假達標
        TL.evaluate({"ledger": {"treasure": [x], "surge": []}}, MK_UP, upto=AFTER[3])
    assert not any(k in x for k in ("hitDay", "hitDate", "hitVal", "hitBy")) and abs(x["relMax"] - 2.1) < 1e-6 and x["peakDay"] == 3, x
    # 完全沒有大盤 (推薦日也沒有) → relMax / rel 為 None，不是 −1e9
    y = dict(_row("7778")); y.pop("mktEntry")
    with patched(TL, "bars", lambda code, years="2y": b):
        TL.evaluate({"ledger": {"treasure": [y], "surge": []}}, {}, upto=AFTER[3])
    assert y["relMax"] is None and y["rel"] is None and "hitDay" not in y, y


def test_no_close_when_twii_missing_day21():
    """第 21 日收 +4.5%、大盤 +2.5% (只贏 2pt)：缺第 21 日 ^TWII → 先不結案 (原本當 0% → 假命中)；有了 → 未命中；
    一直缺但已有更晚的 K (大盤永久缺那天) → 照常結案、不卡住。"""
    b = _bars([(1, 0.5)] * 20 + [(4.5, 4)] * 10)
    mk_miss = {k: v for k, v in MK_UP.items() if k != AFTER[20]}
    x = _row()
    with patched(TL, "bars", lambda code, years="2y": b):
        led = {"ledger": {"treasure": [x], "surge": []}}
        TL.evaluate(led, mk_miss, upto=AFTER[20])
        assert x["status"] == "追蹤" and x["days"] == 21, x
        TL.evaluate(led, MK_UP, upto=AFTER[20])
        assert x["status"] == "未命中" and x["evalAt"] == AFTER[20] and abs(x["cur"] - 4.5) < 1e-6 and abs(x["rel"] - 2.0) < 1e-6, x
        z = _row("7779")
        TL.evaluate({"ledger": {"treasure": [z], "surge": []}}, mk_miss, upto=AFTER[21])
    assert z["status"] == "未命中" and z["evalAt"] == AFTER[20] and z["days"] == 21 and "hitDay" not in z, z


def test_build_passes_scan_mkt_close_into_evaluate():
    today = dt.datetime.now(TL.config.TZ).date().isoformat()
    got = {}

    def fake_eval(prev, mk, max_fetch=200, upto=None, cal=None):
        got.update(mk=dict(mk), cal=cal)
        return 0
    with patched(TL, "evaluate", fake_eval):
        out = _build(_fresh(), dict(SC, date=today, mkt_close=49301.96), dict(TM, trained_at="2026-01-01 00:00:00"), mk={"2026-10-08": 48475.74})
    assert got["mk"].get(today) == 49301.96 and got["mk"]["2026-10-08"] == 48475.74 and isinstance(got["cal"], TL.TradeCal), got
    assert "mk" not in out["scan"] and out["scan"]["mkt_close"] == 49301.96, "大盤序列不放進發布的 scan"

# ------------------------------------------------------------------ 2026-10-13 其餘雷達審查項
def test_record_stamps_in_uni():
    """BE-4：新列 (挖寶 / 飆股) 記掃描當時是否在模型宇宙內；模型檔沒有 universe → 不寫此欄 (App 退回目前宇宙)。"""
    tm = dict(TM, universe=["1111", "9999", "5555"])
    prev = TL.record(_fresh(), SC, tm)
    got = {x["code"]: x.get("in_uni") for x in prev["ledger"]["treasure"]}
    assert got == {"1111": True, "3333": False, "2222": False}, got
    assert prev["ledger"]["surge"][0]["in_uni"] is True
    # 之後宇宙換了 (下週重訓) 也不改已記的列
    prev = TL.record(prev, dict(SC, date="2026-10-01", treasure=[_cand("5555", "B", 0.3)], surge=[]), dict(TM, universe=["5555"]))
    assert {x["code"]: x.get("in_uni") for x in prev["ledger"]["treasure"]} == {"1111": True, "3333": False, "2222": False, "5555": True}
    p2 = TL.record(_fresh(), SC, TM)
    assert not any("in_uni" in x for x in p2["ledger"]["treasure"] + p2["ledger"]["surge"])


# 8888 飆股：第 1 天盤中 +25% → 飆；第 2 天收 +10% (自最高 125 回落 8% 以上) → trail 第 2 天出場；第 20 天收 +15%
P_SURGE_TRAIL = [(25, 0), (10, 5)] + [(5, 4)] * 17 + [(15, 14)] + [(0, -1)] * 10


def test_surge_fin20_after_early_exit():
    """N3：飆 / trail 提早出場的飆股列仍追到第 20 個交易日補 fin20 (同挖寶 fin21)；補上後不再重算。"""
    b = _bars(P_SURGE_TRAIL)
    prev = _fresh(); prev["ledger"]["surge"] = [{"code": "8888", "name": "戊", "date": D0, "entry": 100.0, "ps": 0.41, "status": "追蹤", "days": 0}]
    x = prev["ledger"]["surge"][0]
    with patched(TL, "bars", lambda code, years="2y": b):
        _run_days(prev, 2)
        assert x["status"] == "飆" and x["hitDay"] == 1 and x["trail"] == 10.0 and x["trailDate"] == AFTER[1] and x.get("fin20") is None, x
        assert any(t[1] is x and t[2] for t in TL._todo(prev)), "trail 已出場、還沒第 20 日 → 仍要重算 (舊版到這裡就停)"
        _run_days(prev, 19)
        assert x.get("fin20") is None
        _run_days(prev, 25)
    assert x["fin20"] == 15.0 and x["fin20Date"] == AFTER[19] and x["status"] == "飆" and x["trail"] == 10.0 and x["hitDay"] == 1, x
    assert not any(t[1] is x for t in TL._todo(prev)), "fin20 補上後不再處理"
    st = TL.stats(prev, TM)["surge"]
    assert st["n_fin20"] == 1 and st["win20"] == 1.0 and st["avg_fin20"] == 15.0, st
    # 舊結案列 (第 20 日結案、沒 fin20) 會補一次 fin20；取不到日 K 超過 60 個日曆日 → fin20_na，不再重試
    old = {"code": "8889", "name": "己", "date": D0, "entry": 100.0, "ps": 0.4, "status": "未飆", "days": 20, "trail": 3.0, "chk": 1, "pxk": 1}
    p2 = {"ledger": {"treasure": [], "surge": [old]}}
    assert any(t[1] is old and t[2] for t in TL._todo(p2))
    with patched(TL, "bars", lambda code, years="2y": []):
        TL.evaluate(p2, MK, upto=AFTER[30])
        assert "fin20_na" not in old
        TL.evaluate(p2, MK, upto=(pd.Timestamp(D0) + pd.Timedelta(days=61)).strftime("%Y-%m-%d"))
    assert old["fin20_na"] is True and not any(t[1] is old for t in TL._todo(p2))
    old2 = dict(old, code="8890"); old2.pop("fin20_na", None)
    with patched(TL, "bars", lambda code, years="2y": _bars([(3, 2)] * 30)):
        TL.evaluate({"ledger": {"treasure": [], "surge": [old2]}}, MK, upto=AFTER[25])
    assert old2["fin20"] == 3.0 and old2["status"] == "未飆", old2
    # 兩份帳本合併：同狀態但另一份已補 fin20 → 取有 fin20 的
    a = {"code": "8888", "date": D0, "status": "飆"}
    bb = dict(a, fin20=15.0)

    class _R:
        ok, text = True, json.dumps({"ledger": {"treasure": [], "surge": [a]}, "scans": []})

        def json(self):
            return json.loads(self.text)

    class _S:
        def get(self, *a_, **k_):
            return _R()
    with tempfile.TemporaryDirectory() as td:
        Path(td, "tl.json").write_text(json.dumps({"ledger": {"treasure": [], "surge": [bb]}, "scans": []}), encoding="utf-8")
        with patched(TL, "LOCAL", Path(td) / "tl.json"), patched(TL, "session", lambda: _S()):
            assert TL.load_prev()["ledger"]["surge"][0].get("fin20") == 15.0


def _sur_rows(n, hits, backfill, late=False, base=8000):
    return [{"code": f"{base + i}", "date": "2026-07-01", "status": "飆" if i < hits else "未飆", "cur": 1.0, "days": 20,
             **({"backfill": True} if backfill else {}), **({"scan_src": "late", "backfill": True} if late else {})} for i in range(n)]


def test_surge_stats_published_split_and_drift():
    """BE-7：stats.surge 分真實發布 / 回填；失準只看真實發布結案 ≥20 筆 (回填不觸發也不稀釋)；遲到列不算真實發布。"""
    st = TL.stats({"ledger": {"treasure": [], "surge": _sur_rows(20, 0, True)}, "scans": []}, TM)["surge"]
    assert st["drift"] is False and st["published"] is None and st["n_published"] == 0 and st["n_backfill"] == 20, st
    st = TL.stats({"ledger": {"treasure": [], "surge": _sur_rows(19, 0, False)}, "scans": []}, TM)["surge"]
    assert st["drift"] is False and st["published"] == {"done": 19, "hits": 0, "rate": 0.0}, "不足 20 筆不判定"
    # 回填 30 筆全飆 + 真實發布 20 筆全沒飆：舊規則 (全部 60%) 不會發現，新規則判失準
    rows = _sur_rows(30, 30, True) + _sur_rows(20, 0, False, base=8100)
    st = TL.stats({"ledger": {"treasure": [], "surge": rows}, "scans": []}, TM)["surge"]
    assert st["rate"] == 0.6 and st["drift"] is True and st["n_published"] == 20 and st["n_backfill"] == 30 and st["published"] == {"done": 20, "hits": 0, "rate": 0.0}, st
    al = [a for a in TL.alerts({"date": D0, "treasure": [], "surge": []}, {"treasure": {}, "surge": st}) if a["kind"] == "drift"]
    assert len(al) == 1 and "真實發布飆股率 0%" in al[0]["msg"] and "20 筆結案" in al[0]["msg"], al
    # 回填 20 筆全沒飆 + 真實發布 20 筆全飆：舊規則 (全部 50%) 沒事、新規則也沒事；回填全沒飆不會誤觸
    st = TL.stats({"ledger": {"treasure": [], "surge": _sur_rows(20, 0, True) + _sur_rows(20, 20, False, base=8100)}, "scans": []}, TM)["surge"]
    assert st["drift"] is False and st["published"]["rate"] == 1.0
    # 遲到重掃列 (late) 算回填
    st = TL.stats({"ledger": {"treasure": [], "surge": _sur_rows(20, 0, False, late=True)}, "scans": []}, TM)["surge"]
    assert st["n_published"] == 0 and st["n_backfill"] == 20 and st["published"] is None and st["drift"] is False
    assert TL.stats(_fresh(), TM)["surge"]["published"] is None


def test_surge_alert_says_score_not_probability():
    """N4：飆股提醒寫「飆股分 0.xx」(模型原始分數)，不再寫「機率」；附前 3 名回測實際飆股率。"""
    st = TL.stats(TL.record(_fresh(), SC, TM), TM)
    al = [a for a in TL.alerts(SC, st | {"surge": {**st["surge"], "th_top10": 0.3977}}) if a["kind"] == "surge"]
    assert len(al) == 1
    m = al[0]["msg"]
    assert m == f"🚀 飆股雷達 (09-01)：9999 丁 飆股分 0.41 (前 3 名回測 20 日內先漲 20% 約 {round(TL.RADAR_BT['S3']['surge'] * 100)}%；高風險)", m
    assert re.fullmatch(r"🚀 飆股雷達 \(\d\d-\d\d\)：\S+ \S+ 飆股分 0\.\d\d \(前 3 名回測 20 日內先漲 20% 約 \d+%；高風險\)", m), m
    for bad in ("機率",) + BANNED:
        assert bad not in m, (bad, m)


def test_radar_bt_periods_and_same_window():
    """N8 / BE-5 / N9：同期回測有推薦日界線 (from ≤ to)；分數組 / 路徑表標自己的期間 (2021~)，不是 radar_bt.period。"""
    sm = TL.RADAR_BT["same"]
    assert sm["from"] == "2026-07-01" and sm["to"] == "2026-09-03" and sm["from"] <= sm["to"] and re.fullmatch(r"\d{4}-\d\d-\d\d", sm["to"])
    assert TL.RADAR_DAY0["period"] == "2021-01~2026-09" and TL.RADAR_BT["day0"]["period"] == "2021-01~2026-09"
    assert TL.RADAR_BT["path_period"] == "2021-01~2026-09 (冷啟動路徑 2022-01~)" and TL.RADAR_BT["path_period"] != TL.RADAR_BT["period"]
    assert set(TL.RADAR_BT["path"]) == {"A", "S3"}, "path_period 不放進 kind 為鍵的路徑表"
    assert TL.RADAR_BT["basis"]["research_date"] == "2026-10-07" and TL.RADAR_BT["basis"]["train_start"] == TL.RADAR_BT["method"]["train_start"]
    json.dumps(TL.RADAR_BT, ensure_ascii=False)


def test_build_scans_today_before_backfill():
    """BE-6：GitHub 排程把第一次 full run 拖到午夜後 (D+1 00:59) 時，D 仍記成真實發布 (先掃 D 再回填、D 不進回填)；更早的缺日照常回填。"""
    D = D0                                       # 2026-09-01 (二)；現在 = 09-02 00:59
    tm = dict(TM, trained_at="2026-08-30 08:00:00")
    seen = []

    def scan_fn(tm_, date=None):
        d = date or D
        seen.append(d)
        if d == D:
            return dict(SC, date=D)
        k = DATES.index(d)
        return dict(SC, date=d, treasure=[_cand(f"{6000 + k}", "A", 0.6)], surge=[{"code": f"{7000 + k}", "name": "飆", "close": 100.0, "pct": 3.0, "ps": 0.4}])
    out = _build(_fresh(), dict(SC, date=D), tm, mk=MK, scan_fn=scan_fn, backfill_days=5, now=dt.datetime(2026, 9, 2, 0, 59))
    assert seen[0] == D and seen.count(D) == 1, seen
    rows = out["ledger"]["treasure"] + out["ledger"]["surge"]
    today = [x for x in rows if x["date"] == D]
    assert len(today) == 4 and all(x["scan_src"] == "live" and "backfill" not in x for x in today), today
    older = [x for x in rows if x["date"] != D]
    assert {x["date"] for x in older} == {"2026-08-27", "2026-08-28", "2026-08-31"} and all(x["backfill"] is True and x["scan_src"] == "backfill" for x in older), older
    assert [s["date"] for s in out["scans"]] == ["2026-08-27", "2026-08-28", "2026-08-31", D] and "late" not in out["scan"]
    assert any(a["kind"] == "treasure" for a in out["alerts"]), "準時 (隔天 09:00 前) → 照發新推薦提醒"
    # 同樣情況但已過隔天 09:00 → D 記成 late (回填)，也不被一般回填搶先
    out = _build(_fresh(), dict(SC, date=D), tm, mk=MK, scan_fn=scan_fn, backfill_days=5, now=dt.datetime(2026, 9, 2, 9, 30))
    today = [x for x in out["ledger"]["treasure"] + out["ledger"]["surge"] if x["date"] == D]
    assert len(today) == 4 and all(x["scan_src"] == "late" and x["backfill"] is True for x in today) and out["scan"]["late"] is True


def test_ledger_trim_keeps_signal_rows():
    """BE-9：帳本超過 LEDGER_MAX 時只刪最舊的描述級 (B/B+)，A/A+ 永不刪；有刪就記在 prev["trimmed"] (不再無聲截斷)。"""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-01", periods=TL.LEDGER_MAX)]
    old_a = {"code": "0001", "name": "老A", "date": "2019-12-31", "tier": "A", "status": "命中", "cur": 8.0, "days": 21}
    bs = [{"code": f"B{i:04d}", "name": "b", "date": d, "tier": "B", "status": "未命中", "evalAt": "2020-01-01", "cur": -1.0, "days": 21} for i, d in enumerate(days)]
    prev = _fresh(); prev["ledger"]["treasure"] = [old_a] + bs
    prev = TL.record(prev, SC, TM)                  # 新增 1111 (A+)、3333 (A)、2222 (B)
    tre = prev["ledger"]["treasure"]
    codes = {x["code"] for x in tre}
    assert len(tre) == TL.LEDGER_MAX and {"0001", "1111", "3333", "2222"} <= codes, len(tre)
    assert prev["trimmed"] == {"treasure": 4} and not ({"B0000", "B0001", "B0002", "B0003"} & codes) and "B0004" in codes
    assert [x["date"] for x in tre] == sorted(x["date"] for x in tre)
    # 沒超過 → 不刪、不標
    p2 = TL.record(_fresh(), SC, TM)
    assert "trimmed" not in p2 and len(p2["ledger"]["treasure"]) == 3
    # A/A+ 本身就超過上限 → 全留 (描述級全刪)
    led = {"treasure": [dict(old_a, code=f"A{i}") for i in range(TL.LEDGER_MAX + 2)] + bs[:3], "surge": []}
    assert TL.trim_ledger(led) == {"treasure": 3} and len(led["treasure"]) == TL.LEDGER_MAX + 2


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
