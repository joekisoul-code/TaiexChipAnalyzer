"""10-18 F1/F2：夜盤跳空只算一次、收盤分位 close_q 與 level 同口徑。

F1：短線 v2 夜盤變體 (variant='night') 的 hist_mean/q20/q80 已含 night_chg_pct → refine_short_term 不再乘 (1+gap)，標 gap_in_model=True；
    非夜盤變體 (base/日模型) 照舊乘跳空，close_q_* 也跟著乘 (F2c)。
F2b：小時模型覆寫 nd[0] 時 (range_mode 為真)，close_q_lo/hi 改為 13:30 目標的 level_lo/level_hi。

    python tests/test_forecast_gap_once.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import market_forecast as MF  # noqa: E402

BASE = 47111.0
GAP_BETA = 0.48
NIGHT = -2.42


def _patch():
    MF.night_gap_beta = lambda: (GAP_BETA, 0.6, 700)
    MF._ST_nf = lambda snap: True


def _fc(nd):
    hz = {h: {"pred": 0.5, "bin": 2, "drivers": {"positive": [], "negative": []}, "p_up": 0.5, "base_hit": 0.55, "hist_mean": 0.48, "q20": -2.0, "q80": 3.0, "variant": "night" if h == 5 else None} for h in (5, 10, 20)}
    return {"date": "2026-10-16", "close": BASE, "next_days": nd, "horizons": hz}


def _snap():
    return {"phase": "pre", "tx_night": {"change_pct": NIGHT, "last": 46140}}


def _row(n, variant, hm, q20, q80, range_mode=None):
    x = {"n": n, "label": f"第{n}天", "date": f"2026-10-{18 + n}", "p_up": 0.3, "base_hit": 0.55, "variant": variant,
         "hist_mean": hm, "q20": q20, "q80": q80,
         "level": round(BASE * (1 + hm / 100)), "level_lo": round(BASE * (1 + q20 / 100)), "level_hi": round(BASE * (1 + q80 / 100)), "source": "短線 v2"}
    if range_mode:
        x["range_mode"] = range_mode
        x["close_q_lo"], x["close_q_hi"] = x["level_lo"], x["level_hi"]
        x["level_lo"], x["level_hi"] = round(BASE * 0.97), round(BASE * 1.02)     # 路徑型 (不應被改)
    return x


def test_night_variant_not_double_counted():
    _patch()
    out = MF.refine_short_term(_fc([_row(1, "night", -1.0, -2.2, 0.3), _row(2, "night", -1.0, -2.2, 0.3, range_mode="path")]), None, _snap(), None)
    for x in out["next_days"]:
        assert x["level"] == round(BASE * (1 - 0.01)), x
        assert x["gap_in_model"] is True and x["gap_adj"] == round(GAP_BETA * NIGHT, 2), x
        assert "level_raw" not in x
    x2 = out["next_days"][1]
    assert (x2["close_q_lo"], x2["close_q_hi"]) == (round(BASE * (1 - 0.022)), round(BASE * 1.003)), x2
    assert x2["close_q_lo"] <= x2["level"] <= x2["close_q_hi"], x2
    assert x2["level_lo"] == round(BASE * 0.97)        # 路徑型不動
    assert out["horizons"][5]["gap_in_model"] is True


def test_base_variant_still_applies_gap_and_close_q_follows():
    _patch()
    gap = GAP_BETA * NIGHT / 100
    out = MF.refine_short_term(_fc([_row(1, "base", -1.0, -2.2, 0.3, range_mode="path")]), None, _snap(), None)
    x = out["next_days"][0]
    assert x["level"] == round(BASE * (1 + gap) * (1 - 0.01)), x
    assert x["gap_in_model"] is False and x["level_raw"] == round(BASE * 0.99)
    assert x["close_q_lo"] == round(BASE * (1 + gap) * (1 - 0.022)), x
    assert x["close_q_hi"] == round(BASE * (1 + gap) * 1.003), x
    assert x["close_q_lo"] <= x["level"] <= x["close_q_hi"], x
    assert x["level_lo"] == round(BASE * 0.97)


def test_hourly_override_updates_close_q():
    _patch()
    nd = [_row(1, "night", -1.0, -2.2, 0.3, range_mode="path")]
    hourly = {"live": False, "day": nd[0]["date"], "targets": {"13:30": {"p_up": 0.2, "base_hit": 0.57, "pred": -1.2, "hist_mean": -0.8, "q20": -1.65, "q80": 0.05,
                                                                          "level": 46734.0, "level_lo": 46334.4, "level_hi": 47134.6}}}
    out = MF.refine_short_term(_fc(nd), hourly, _snap(), None)
    x = out["next_days"][0]
    assert x["level"] == 46734.0
    assert (x["close_q_lo"], x["close_q_hi"]) == (46334, 47135), x
    assert x["close_q_lo"] <= x["level"] <= x["close_q_hi"]
    assert x["level_lo"] == round(BASE * 0.97)        # 路徑型 level_lo 不被小時模型覆寫
    assert x["gap_in_model"] is True


def test_no_night_no_gap_keys():
    MF.night_gap_beta = lambda: (GAP_BETA, 0.6, 700)
    MF._ST_nf = lambda snap: False
    out = MF.refine_short_term(_fc([_row(1, "base", -1.0, -2.2, 0.3)]), None, _snap(), None)
    x = out["next_days"][0]
    assert "gap_adj" not in x and "gap_in_model" not in x and x["level"] == round(BASE * 0.99)


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
