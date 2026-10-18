"""10-18 chips-4：desk opt 區塊加 pcr_oi_official (期交所官方 PutCallOIRatio%，同日)，與自算 pcr_oi 並列。

    python tests/test_desk_pcr_official.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.analysis import desk as D  # noqa: E402
from chip.sources import taifex as TF  # noqa: E402


def _pcr():
    return pd.DataFrame({"date": ["2026-10-15", "2026-10-16"], "pcr_oi": [95.1, 88.456], "pcr_vol": [90.0, 91.0]})


def test_official_same_day():
    orig = TF.put_call_ratio
    TF.put_call_ratio = _pcr
    try:
        o = D.opt_block({"date": "2026-10-16", "pcr_oi": 72.34}, [], "2026-10-16", {})
        assert o["pcr_oi"] == 72.34 and o["pcr_oi_official"] == 88.46, o
        o = D.opt_block({"date": "2026-10-17", "pcr_oi": 72.34}, [], "2026-10-17", {})   # 官方無同日 → None，不拿前一日冒充
        assert o["pcr_oi_official"] is None
    finally:
        TF.put_call_ratio = orig


def test_official_fetch_failure_is_none():
    orig = TF.put_call_ratio
    def boom():
        raise RuntimeError("net")
    TF.put_call_ratio = boom
    try:
        o = D.opt_block({"date": "2026-10-16", "pcr_oi": 72.34}, [], "2026-10-16", {})
        assert o["pcr_oi_official"] is None and o["pcr_oi"] == 72.34
    finally:
        TF.put_call_ratio = orig


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
