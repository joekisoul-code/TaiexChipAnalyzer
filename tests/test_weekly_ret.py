"""10-18 chips-1：週別表「指數週漲跌」改用 前週最後收盤 → 本週最後收盤，與同列週淨買賣 / 融資週變 (每日值加總，含週一) 同一時間窗。

    python tests/test_weekly_ret.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.analysis import gov8 as G, margin_dyn as MD  # noqa: E402


def _h():
    # week1 2026-09-28~10-02 收 100..110；week2 10-05(一) 120 ... 10-09(五) 115
    dates = pd.bdate_range("2026-09-28", periods=10)
    close = [100, 102, 105, 108, 110, 120, 118, 117, 116, 115]
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close, "gov8_net": [10] * 10, "chg1": [5] * 10})


def test_gov8_weekly_uses_prev_week_close():
    w = G._weekly(_h())
    assert len(w) == 2 and w[1]["week"] == "2026-10-05"
    assert w[1]["ret"] == round((115 / 110 - 1) * 100, 2), w        # +4.55%，不是 115/120-1 = -4.17%
    assert w[1]["net"] == 50 and w[1]["days"] == 5
    assert w[0]["ret"] == round((110 / 100 - 1) * 100, 2)            # 最舊一週沒有前收 → 退回週內首日收盤


def test_margin_weekly_uses_prev_week_close():
    w = MD._weekly(_h())
    assert w[1]["ret"] == round((115 / 110 - 1) * 100, 2), w
    assert w[1]["chg"] == 25


def test_ret_basis_keys():
    """10-18 審查：App 只認 ret_basis / weekly_ret_basis = 'prev_close' 才改用新口徑標示 + 顯示順勢/逆勢欄。"""
    for w in (G._weekly(_h()), MD._weekly(_h())):
        assert w[1]["ret_basis"] == "prev_close" and w[0]["ret_basis"] == "first_close", w
    assert G.weekly_ret_basis(G._weekly(_h())) == "mixed"            # 含最舊一週退回首日收 → 不冒稱全為前週收
    h = pd.concat([pd.DataFrame({"date": ["2026-09-25"], "close": [99], "gov8_net": [0], "chg1": [0]}), _h()], ignore_index=True)
    w = G._weekly(h)[-2:]                                             # 前面多一週 → 這兩週都有前收
    assert all(x["ret_basis"] == "prev_close" for x in w) and G.weekly_ret_basis(w) == "prev_close"
    assert G.weekly_ret_basis([]) is None


def test_market_level_weekly_ret_basis_in_views():
    import numpy as np
    sys.path.insert(0, str(ROOT / "tests"))
    import test_margin_dyn as TM
    v, _ = MD.market_view(TM._hist(end="2026-10-13"), None)
    assert v["weekly_ret_basis"] == "prev_close" and all(w["ret_basis"] == "prev_close" for w in v["weekly"])
    n = 300
    rng = np.random.default_rng(0)
    h = pd.DataFrame({"date": pd.bdate_range("2025-06-02", periods=n).strftime("%Y-%m-%d"),
                      "close": 20000 * np.cumprod(1 + rng.normal(0, .01, n)), "gov8_net": rng.normal(0, 20, n)})
    v, _ = G.market_view(h, None)
    assert v["weekly_ret_basis"] == "prev_close" and all(w["ret_basis"] == "prev_close" for w in v["weekly"])


def test_single_day_week_has_ret():
    h = _h().iloc[:6]                                                 # week2 只有 10-05 一天
    assert G._weekly(h)[1]["ret"] == round((120 / 110 - 1) * 100, 2)


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
