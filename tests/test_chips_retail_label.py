"""10-18 stocks-1：holders.retail20 在集保來源 (股權分散 / TDCC open data) 其實是 <10 張 (級距 1~3) →
retail_lots 標門檻、notes 文案跟著寫 <10 張；玩股網 lessThan20 路徑仍為 <20 張。鍵名 retail20 不變 (舊 App 相容)。

    python tests/test_chips_retail_label.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.analysis import chips as C  # noqa: E402


def _conc(n=5, foreign=None):
    return pd.DataFrame({"date": pd.date_range("2026-09-01", periods=n, freq="W").strftime("%Y-%m-%d"), "close": 30.0,
                         "big400": [60.0 + i for i in range(n)], "big1000": 50.0, "retail20": 23.17, "foreign_hold": foreign})


def test_tdcc_source_is_lt10():
    h = C.annotate_holders(C.holder_trend(_conc()), "集保 open data (散戶=<10 張，歷史需每週累積)")
    assert h["retail_lots"] == 10 and h["retail20"] == 23.17
    assert h["foreign_hold"] is None and "外資" in h["foreign_hold_why"]
    assert h["big400_chg12w"] is None and "13 週" in h["big400_chg12w_why"]
    h2 = C.annotate_holders(C.holder_trend(_conc()), "集保股權分散 (散戶=<10 張)")
    assert h2["retail_lots"] == 10


def test_wantgoo_source_is_lt20():
    h = C.annotate_holders(C.holder_trend(_conc(14, foreign=70.0)), None)
    assert h["retail_lots"] == 20 and h["holders_source"] == "玩股網大戶籌碼"
    assert "foreign_hold_why" not in h and "big400_chg12w_why" not in h and h["big400_chg12w"] is not None


def test_note_uses_retail_lots():
    import inspect
    src = inspect.getsource(C.assess)
    assert "散戶 (<{holders.get('retail_lots', 20)} 張)" in src and "散戶 (<20 張)" not in src
    assert C.annotate_holders({}, "x") == {}


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
