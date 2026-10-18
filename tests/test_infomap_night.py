"""10-18 F3：預測邏輯總表 (infomap.build) 今日讀數補上夜盤台指期。

build_matrix 以「下一列日期」對夜盤 → 最後一列 (今日) 的 night_chg_pct 永遠 NaN → 原本被跳過，多空票少一張。
夜盤已收 (phase pre/closed) 時改用 snapshot tx_night.change_pct；夜盤進行中 (phase night) 不補。

    python tests/test_infomap_night.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import infomap as IM  # noqa: E402

N = 300


def _frame(scored, night):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=N, freq="D").astype(str),
                      "night_chg_pct": rng.normal(0, 1, N), "ret5": rng.normal(0, 1, N)})
    d.loc[d.index[-1], "night_chg_pct"] = np.nan        # 今日夜盤 NaN (build_matrix 的對齊方式)
    d.loc[d.index[-1], "ret5"] = 5.0                    # 高檔
    return d


def _stats():
    v = lambda f, name, diff: {"f": f, "name": name, "group": "g", "diff": diff, "up_hi": 0.83, "up_lo": 0.20, "cons": 1.0, "dir": "正向"}
    return {"trained_at": "x", "groups": {}, "horizons": {str(h): {"n_tested": 83, "n_valid": 2, "text": "t",
                                                                    "valid": [v("night_chg_pct", "前晚夜盤台指期%", 0.6), v("ret5", "5 日報酬", 0.1)]} for h in IM.HORIZONS}}


def _patch():
    IM._frame = _frame
    IM.M.load_json = lambda name: _stats()


def test_night_filled_when_final():
    _patch()
    out = IM.build(None, None, {"phase": "pre", "tx_night": {"change_pct": -2.4}})
    h1 = out["h"]["1"]
    fs = {it["f"]: it for it in h1["items"]}
    assert "night_chg_pct" in fs, h1
    assert fs["night_chg_pct"]["value"] == -2.4 and fs["night_chg_pct"]["tier"] == "低檔" and fs["night_chg_pct"]["sig"] == -1
    assert (h1["bull"], h1["bear"]) == (1, 1), h1


def test_not_filled_when_session_running_or_missing():
    _patch()
    for snap in (None, {"phase": "night", "tx_night": {"change_pct": -2.4}}, {"phase": "pre"}):
        h1 = IM.build(None, None, snap)["h"]["1"]
        assert [it["f"] for it in h1["items"]] == ["ret5"], (snap, h1)
        assert (h1["bull"], h1["bear"]) == (1, 0)


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
