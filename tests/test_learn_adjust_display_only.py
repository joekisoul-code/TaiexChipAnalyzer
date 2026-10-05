"""線上自學 (pr2 P2/P5) 單元測試：合成帳本、不連網。

    python tests/test_learn_adjust_display_only.py
    python -m pytest tests/test_learn_adjust_display_only.py

(a) 60 筆夜盤叫牌 ewm 命中 ~0.75、model_hit 0.86 → adjust_forecast 後 call 不變、無 call_degraded、recent_flag=below、learn_note 含「不改叫牌」；
(b) 80 筆有斜率的 p_up → 無 p_up_adj、adjust.platt is None、platt_diag 有值；(c) 60 筆觸及率 35% → touch_factor()==1.0、sigma_factor==1.0 (raw 1.32)；
(d) records_from_forecast / annotate_ledger 帶 call_model / conf_tier / bucket；(e) 個股 by_h 的 degrade 行為與現行相同 (迴歸)；(f) 旗標開啟時舊路徑仍在。
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import learn as L  # noqa: E402

DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-05", "2026-09-30")]


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _mkt_rows(n, h, variant, hit_fn, call_hit, p_fn=lambda i: 0.8, backfill_n=0, touch=None, call="偏多"):
    rows = []
    for i in range(n):
        hit = hit_fn(i)
        p = p_fn(i)
        r = {"kind": "mkt", "sid": "TAIEX", "as_of": DATES[i], "target": DATES[i + h] if h <= 3 else None, "h": h, "mode": "night" if variant == "night" else "close", "phase": "closed", "live": False,
             "base": 1000.0, "p_up": p, "base_hit": 0.55, "call": call, "strength": "", "call_hit": call_hit, "variant": variant, "level": 1001.0, "buy_at": 990.0, "sell_at": 1010.0, "stop": 985.0, "target_px": 1015.0,
             "range_mode": "base", "trend7": "flat",
             "realized": {"date": DATES[i + h], "ret": 0.6 if hit else -0.6, "rel": None, "up": bool(hit) if call == "偏多" else (not hit), "hit": bool(hit) if call != "中性" else None,
                          "brier": round((p - (1.0 if hit else 0.0)) ** 2, 4)}}
        if touch is not None:
            r["realized"].update({"path_low": 985.0, "path_high": 1012.0, "buy_touch": touch(i), "sell_touch": touch(i), "stop_hit": False, "target_hit": False})
        if i < backfill_n:
            r["backfill"] = True; r["bf_ver"] = L.BACKFILL_VER
        rows.append(r)
    return rows


def test_a_degrade_display_only():
    rows = _mkt_rows(60, 1, "night", lambda i: i % 4 != 3, 0.86, backfill_n=40)     # 45/60 = 0.75 命中 vs 長期 0.86 → 差 −0.11
    summ = L.summarize(rows)
    g = summ["market"]["by_h"]["1"]
    assert g["n_calls"] == 60 and g["n_published"] == 20 and g["n_backfill"] == 40 and abs(g["up_rate"] - 0.75) < 1e-9
    adj = g["adjust"]
    assert adj["degrade"] is False and adj["recent_flag"] == "below" and adj["gap"] is not None and adj["gap"] < -0.05
    assert adj["platt"] is None and isinstance(adj["platt_diag"], list) and len(adj["platt_diag"]) == 3
    assert "不改叫牌" in adj["note"] and "暫改中性" not in adj["note"]
    gv = g["by_variant"]["night"]
    assert gv["adjust"]["recent_flag"] == "below" and gv["adjust"]["degrade"] is False
    fc = {"date": "2026-10-02", "close": 1000.0, "next_days": [{"n": 1, "date": "2026-10-06", "variant": "night", "call": "偏多", "call_strength": "強", "p_up": 0.85, "base_hit": 0.55, "level": 1005}],
          "horizons": {}}
    out = L.adjust_forecast(fc, summ)
    x = out["next_days"][0]
    assert x["call"] == "偏多" and x["call_strength"] == "強" and "call_degraded" not in x and "call_model" not in x
    assert "p_up_adj" not in x and x["p_up"] == 0.85
    assert x["recent_flag"] == "below" and abs(x["recent_hit"] - g["hit_ewm"]) < 1e-9 and x["recent_n"] == 60 and abs(x["recent_up_rate"] - 0.75) < 1e-9
    assert "不改叫牌" in x["learn_note"] and "同期上漲率 75%" in x["learn_note"] and "近期實際命中" in x["learn_note"]
    assert out["learn"]["flags"] == {"degrade": False, "platt": False, "touch_factor": False} and "停用" in out["learn"]["note"]
    # 高於長期 → above，同樣不改叫牌
    rows2 = _mkt_rows(60, 1, "night", lambda i: i % 10 != 9, 0.80)   # 0.9 vs 0.80
    adj2 = L.summarize(rows2)["market"]["by_h"]["1"]["adjust"]
    assert adj2["recent_flag"] == "above" and adj2["degrade"] is False and "不改叫牌" in adj2["note"]


def test_b_platt_diag_only():
    rows = _mkt_rows(80, 2, "base", lambda i: (i * 7) % 10 < 7 if (0.3 + 0.5 * i / 79) > 0.55 else (i * 7) % 10 < 3, 0.60, p_fn=lambda i: round(0.3 + 0.5 * i / 79, 3))
    summ = L.summarize(rows)
    g = summ["market"]["by_h"]["2"]
    assert g["n_eval"] == 80 and g["adjust"]["platt"] is None and g["adjust"]["platt_diag"] is not None and g["adjust"]["platt_diag"][2] == 1.0
    fc = {"date": "2026-10-02", "close": 1000.0, "next_days": [{"n": 2, "date": "2026-10-07", "variant": "base", "call": "偏多", "call_strength": "", "p_up": 0.62, "base_hit": 0.55}],
          "horizons": {"5": {"call": "偏多", "p_up": 0.6, "base_hit": 0.55, "variant": "daily"}}}
    out = L.adjust_forecast(fc, summ)
    assert "p_up_adj" not in out["next_days"][0] and out["next_days"][0]["recent_up_rate"] is not None
    assert "p_up_adj" not in out["horizons"]["5"]


def test_c_touch_factor_fixed():
    rows = _mkt_rows(60, 1, "base", lambda i: i % 2 == 0, 0.60, touch=lambda i: i % 20 < 7)   # 35% 觸及
    summ = L.summarize(rows)
    t = summ["market"]["touch"]["1"]
    assert t["n"] == 60 and abs(t["buy_touch"] - 0.35) < 1e-9 and t["sigma_factor"] == 1.0 and t["enabled"] is False
    assert abs(t["sigma_factor_raw"] - 1.323) < 1e-3 and "乘數停用 ×1.00" in t["note"] and "1.32" in t["note"]
    assert L.touch_factor(summ, 1) == 1.0 and L.touch_factor({"market": {"touch": {"1": {"sigma_factor": 1.3}}}}, 1) == 1.0
    fc = {"date": "2026-10-02", "close": 1000.0, "next_days": [{"n": 1, "date": "2026-10-06", "variant": "base", "call": "偏多", "p_up": 0.6}], "horizons": {}}
    out = L.adjust_forecast(fc, summ)
    assert out["learn"]["touch_sigma_factor"]["1"] == 1.0
    with patched(L, "TOUCH_FACTOR_ENABLED", True):      # 旗標開啟 → 舊路徑 (診斷值即乘數)
        t2 = L._touch_stats(rows)
        assert abs(t2["sigma_factor"] - 1.323) < 1e-3 and "水準乘數 ×1.32" in t2["note"]
        assert abs(L.touch_factor({"market": {"touch": {"1": t2}}}, 1) - 1.323) < 1e-3


def test_d_records_and_annotate():
    fc = {"date": "2026-10-02", "close": 1000.0, "trend7": {"state": "flat"},
          "next_days": [{"n": 1, "date": "2026-10-06", "variant": "night", "call": "偏多", "call_strength": "強", "p_up": 0.85, "base_hit": 0.55, "conf_tier": "高", "level": 1005, "buy_at": 990, "sell_at": 1010, "stop": 985, "target": 1015},
                        {"n": 2, "date": "2026-10-07", "variant": "night", "call": "中性", "call_strength": "", "p_up": 0.56, "base_hit": 0.55, "level": 1006}],
          "horizons": {5: {"call": "中性", "call_model": "偏多", "p_up": 0.6, "base_hit": 0.55, "variant": "daily"}, 10: {"p_up": 0.5, "base_hit": 0.55}},
          "verdict": {"bucket": "高共識", "call": "偏多"}}
    rows = L.records_from_forecast(fc, {"phase": "closed"})
    by = {r["h"]: r for r in rows}
    assert by[1]["call_model"] == "偏多" and by[1]["conf_tier"] == "高" and by[1]["verdict_bucket"] == "高共識" and by[1]["p_up_adj"] is None and by[1]["recent_flag"] is None
    assert by[2]["verdict_bucket"] is None and by[2]["call_model"] == "中性" and by[2]["conf_tier"] is None
    assert by[5]["call_model"] == "偏多" and by[5]["call"] == "中性" and by[5]["verdict_bucket"] is None and by[10]["call_model"] == "中性"
    assert all("bucket" not in r for r in rows)   # 判斷總結桶取名 verdict_bucket；bucket 保留給 kind=gap 列的跳空幅度桶 (records_from_precheck)
    gap_rows = L.records_from_precheck({"date": "2026-10-02", "close": 1000.0, "next_days": fc["next_days"],
                                        "precheck": {"gap": {"est": 0.6, "source": "美期", "bucket": "開高 (0.5~1.5%)", "stats": {"p_hold": 0.6, "p_fill": 0.3}}}})
    assert gap_rows and gap_rows[0]["kind"] == "gap" and gap_rows[0]["bucket"] == "開高 (0.5~1.5%)" and "verdict_bucket" not in gap_rows[0]
    # 發布流程：run() 記帳時尚無 verdict / recent_flag → 之後 annotate_ledger 補上，call 不動、已對帳列不碰、已有值不覆蓋
    fc0 = {k: v for k, v in fc.items() if k != "verdict"}
    fc0["next_days"][0].pop("conf_tier")
    rows0 = L.records_from_forecast(fc0, {"phase": "closed"})
    assert rows0[0]["verdict_bucket"] is None and rows0[0]["conf_tier"] is None
    rows0[1]["realized"] = {"hit": True}
    out = {"ledger": rows0 + gap_rows}
    fc["next_days"][0]["conf_tier"] = "高"; fc["next_days"][0]["recent_flag"] = "below"; fc["next_days"][1]["recent_flag"] = "above"
    n = L.annotate_ledger(out, fc, {"phase": "closed"})
    assert n == 1 and rows0[0]["verdict_bucket"] == "高共識" and rows0[0]["conf_tier"] == "高" and rows0[0]["recent_flag"] == "below" and rows0[0]["call"] == "偏多"
    assert "verdict_bucket" in rows0[1] and rows0[1]["verdict_bucket"] is None and rows0[1].get("recent_flag") is None   # 已對帳列不改
    assert gap_rows[0]["bucket"] == "開高 (0.5~1.5%)" and "verdict_bucket" not in gap_rows[0]                          # gap 列不被 annotate 碰
    assert L.annotate_ledger({"ledger": []}, fc, None) == 0 and L.annotate_ledger(out, {"error": "x"}, None) == 0
    # review2 #21：resave=True 時補完欄位後重寫 sqlite 當日快照 (內容含補上的欄位)；預設不寫、沒補到列也不寫
    saved = []
    orig = L.store.save_snapshot
    L.store.save_snapshot = lambda d, src, payload: saved.append((src, json.loads(json.dumps(payload, ensure_ascii=False, default=str))))
    try:
        rows1 = L.records_from_forecast(fc0, {"phase": "closed"})
        out1 = {"ledger": rows1}
        assert L.annotate_ledger(out1, fc, {"phase": "closed"}) >= 1 and saved == []
        rows2 = L.records_from_forecast(fc0, {"phase": "closed"})
        out2 = {"ledger": rows2}
        assert L.annotate_ledger(out2, fc, {"phase": "closed"}, resave=True) >= 1 and len(saved) == 1
        src, payload = saved[0]
        assert src == "pred_ledger" and payload["n"] == len(rows2)
        r0 = payload["rows"][0]
        assert r0["verdict_bucket"] == "高共識" and r0["conf_tier"] == "高" and r0["recent_flag"] == "below"
        assert L.annotate_ledger(out2, fc, {"phase": "closed"}, resave=True) == 0 and len(saved) == 1   # 已補過 → 不再寫
    finally:
        L.store.save_snapshot = orig
    src_exp = (ROOT / "tools" / "export_static.py").read_text(encoding="utf-8")
    assert "annotate_ledger(learn_out, fc, snap, resave=True)" in src_exp   # 發布流程補完後重寫快照


def test_e_stock_degrade_regression():
    rows = []
    for i in range(30):
        hit = i % 5 < 2   # 12/30 = 0.4 vs call_hit 0.62 → 差 −0.22 → 現行規則降級
        rows.append({"kind": "stk", "sid": "2330", "as_of": DATES[i], "target": None, "h": 5, "mode": "close", "phase": "closed", "live": False, "base": 500.0, "p_up": 0.6, "base_hit": 0.5,
                     "call": "偏多", "strength": "", "call_hit": 0.62, "variant": "stock", "realized": {"date": DATES[i + 5], "ret": 1.0, "rel": 0.5 if hit else -0.5, "up": hit, "hit": hit, "brier": 0.2}})
    summ = L.summarize(rows)
    adj = summ["stocks"]["2330"]["by_h"]["5"]["adjust"]
    assert adj["degrade"] is True and "暫改中性" in adj["note"] and adj["recent_flag"] == "below" and adj["platt"] is not None   # 個股維持現行 (export_static 讀 adjust.degrade)
    # 同一組列若當市場組看 (預設旗標) → 不降級
    g = L._group_stats(rows)
    assert g["adjust"]["degrade"] is False and g["adjust"]["recent_flag"] == "below" and g["adjust"]["platt"] is None and g["adjust"]["platt_diag"] is not None


def test_f_flags_restore_old_path():
    rows = _mkt_rows(60, 1, "base", lambda i: i % 4 != 3, 0.86)
    with patched(L, "DEGRADE_ENABLED", True), patched(L, "PLATT_ENABLED", True):
        summ = L.summarize(rows)
        g = summ["market"]["by_h"]["1"]
        assert g["adjust"]["degrade"] is True and g["adjust"]["platt"] is not None
        fc = {"date": "2026-10-02", "close": 1000.0, "next_days": [{"n": 1, "date": "2026-10-06", "variant": "base", "call": "偏多", "call_strength": "", "p_up": 0.8}], "horizons": {}}
        x = L.adjust_forecast(fc, summ)["next_days"][0]
        assert x["call"] == "中性" and x["call_degraded"] is True and x["call_model"] == "偏多" and "p_up_adj" in x
    out = L.summarize(rows)
    assert out["market"]["by_h"]["1"]["adjust"]["degrade"] is False
    assert L.DEGRADE_ENABLED is False and L.PLATT_ENABLED is False and L.TOUCH_FACTOR_ENABLED is False


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
