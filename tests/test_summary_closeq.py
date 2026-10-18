"""10-18 market_forecast.summarize 收盤區間：r2m 起 next_days 的 level_lo/level_hi 被 range_levels 覆寫為路徑型「約兩成機率觸及」下緣/上緣，
收盤 20~80% 分位改存 close_q_lo/close_q_hi → 文字摘要「預估收盤 X (區間 lo~hi)」有 close_q_* 就用它，缺才退回 level_lo/level_hi，且 lo ≤ level ≤ hi。

    python tests/test_summary_closeq.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import market_forecast as MF  # noqa: E402


def _hz():
    d = {"pred": 0.5, "bin": 2, "p_up": 0.6, "base_hit": 0.55, "hist_mean": 0.4, "q20": -2.0, "q80": 3.0,
         "drivers": {"positive": [], "negative": []}}
    return {5: dict(d), 10: dict(d), 20: dict(d)}


def _nd(**kw):
    x = {"label": "隔天", "date": "2026-10-19", "p_up": 0.52, "base_hit": 0.55, "level": 46701.0, "level_lo": 45584.0, "level_hi": 46688.0}
    x.update(kw)
    return x


def _band(txt):
    m = re.search(r"預估收盤 ([\d,]+) \(區間 ([\d,]+)~([\d,]+)\)", txt)
    assert m, txt
    return tuple(int(v.replace(",", "")) for v in m.groups())


def test_uses_close_q_when_present():
    lv, lo, hi = _band(MF.summarize({"horizons": _hz(), "next_days": [_nd(close_q_lo=46050.0, close_q_hi=47320.0)]}))
    assert (lv, lo, hi) == (46701, 46050, 47320), (lv, lo, hi)
    assert lo <= lv <= hi


def test_fallback_level_lo_hi_unchanged():
    nd = _nd(level_lo=46100.0, level_hi=47200.0)
    assert _band(MF.summarize({"horizons": _hz(), "next_days": [nd]})) == (46701, 46100, 47200)
    nd2 = _nd(level_lo=46100.0, level_hi=47200.0, close_q_lo=None, close_q_hi=None)   # range_levels 從缺值複製來的 None → 視為缺
    assert _band(MF.summarize({"horizons": _hz(), "next_days": [nd2]})) == (46701, 46100, 47200)


def test_clamped_when_level_outside():
    lv, lo, hi = _band(MF.summarize({"horizons": _hz(), "next_days": [_nd(level=47500.0, close_q_lo=46050.0, close_q_hi=47320.0)]}))
    assert (lo, hi) == (46050, 47500) and lo <= lv <= hi


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
