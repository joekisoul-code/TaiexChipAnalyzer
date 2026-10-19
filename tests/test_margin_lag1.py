"""10-19 融資研究 margin2/A：模型矩陣的融資券一律用 D−1 (features.margin_lagged / MARGIN_LAG=1)。

- 下午 (D 日融資未公布、最後一列 NaN) 與 21:45 (已公布) 的最後一列模型特徵完全相同 → 一整天叫牌口徑一致
- 融資衍生特徵 = lag0 版前一列的值；非融資特徵不變；MARGIN_LAG=0 時與舊行為相同

    python tests/test_margin_lag1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.analysis import market  # noqa: E402
from chip.predict import features as F  # noqa: E402

MARGIN_FEATS = ["margin_pct20", "margin_div20", "margin_chg5_pct", "short_chg5_z", "short_ratio", "smart_core", "f_margin", "f_short"]


def synth(n=420, seed=7):
    r = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    c = 18000 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    o = c * (1 + r.normal(0, 0.003, n))
    df = pd.DataFrame({"date": dates, "open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                       "amount": r.uniform(2e11, 5e11, n), "volume": r.uniform(4e9, 8e9, n),
                       "foreign": r.normal(0, 80, n), "trust": r.normal(0, 20, n), "dealer": r.normal(0, 30, n),
                       "fut_foreign_net_oi": np.cumsum(r.normal(0, 2000, n)) - 20000,
                       "margin_amt": 2500 + np.cumsum(r.normal(0, 15, n)), "margin_lots": 7e6 + np.cumsum(r.normal(0, 3e4, n)),
                       "short_lots": 2e5 + np.cumsum(r.normal(0, 4e3, n))})
    df["total"] = df["foreign"] + df["trust"] + df["dealer"]
    s = market.score_frame(market.add_features(df))
    return s.loc[:, ~s.columns.duplicated(keep="last")]


def last_feats(scored, cols):
    m = F.market_matrix(scored)
    return m[cols].iloc[-1].astype(float)


def main():
    ev = synth()
    pm = ev.copy()
    for c in F.MARGIN_RAW:
        pm.loc[pm.index[-1], c] = np.nan            # 下午：D 日融資券尚未公布
    pm = market.score_frame(market.add_features(pm)); pm = pm.loc[:, ~pm.columns.duplicated(keep="last")]
    cols = [c for c in F.MARKET_FEATURES if c in F.market_matrix(ev)]
    a, b = last_feats(ev, cols), last_feats(pm, cols)
    assert np.allclose(a.fillna(-9e9), b.fillna(-9e9)), "下午與晚上最後一列模型特徵應完全相同：" + str((a - b)[(a - b).abs() > 1e-9].to_dict())
    assert a[[c for c in MARGIN_FEATS if c in a]].notna().all(), "下午版融資特徵不是 NaN (用 D−1)"
    # 舊行為 (lag0) 下午版會不同 → 證明這個測試抓得到問題
    F.MARGIN_LAG = 0
    try:
        a0, b0 = last_feats(ev, cols), last_feats(pm, cols)
        assert not np.allclose(a0.fillna(-9e9), b0.fillna(-9e9)), "lag0 下午/晚上應不同"
        m0 = F.market_matrix(ev)
    finally:
        F.MARGIN_LAG = 1
    m1 = F.market_matrix(ev)
    # 純融資衍生 (不混其他輸入) = lag0 前一列
    for c in ["margin_pct20", "margin_chg5_pct", "short_ratio"]:
        x, y = m1[c].iloc[-50:].to_numpy(), m0[c].shift(1).iloc[-50:].to_numpy()
        assert np.allclose(x, y, equal_nan=True), c
    changed = [c for c in cols if not np.allclose(m1[c].astype(float).fillna(-9e9), m0[c].astype(float).fillna(-9e9))]
    assert set(changed) <= set(MARGIN_FEATS) | {"composite", "composite_smooth", "composite_chg5"}, "只有融資衍生欄改變：" + str(changed)
    assert {"margin_chg5_pct", "smart_core", "f_margin"} <= set(changed)
    assert m1["close"].equals(m0["close"]) and m1["date"].equals(m0["date"])
    # 冪等：沒有融資欄 → 原樣
    nm = ev.drop(columns=list(F.MARGIN_RAW))
    assert F.margin_lagged(nm) is nm
    print("test_margin_lag1: 5 checks passed")


if __name__ == "__main__":
    main()
