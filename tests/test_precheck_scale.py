"""2026-11-01 precheck 跳空表分位依波動縮放 (SQ_BETA = 0.5)：

- train：每格多存 sq (分位 ÷ 前一日 σ^0.5)
- scale_cell：發布時 SQ_KEYS 換成 sq × 今日 σ^0.5，原值留 raw_q；沒有 sq / σ 無效 → 原樣
- build：盤中實際開盤 → stats 的 intra_p20 等 = sq × σ_now^0.5 (σ_now = sigma_series 最後一筆)；text 用縮放後的數字

    python tests/test_precheck_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import model as M  # noqa: E402
from chip.predict import precheck as PC  # noqa: E402
from chip.predict import range_levels as RL  # noqa: E402
from chip.predict import short_term as ST  # noqa: E402


def frame(n=900, seed=1):
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.6, 1.8)            # 前半低波動、後半高波動
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol / 100)))
    prev = np.r_[c[0], c[:-1]]
    o = prev * (1 + rng.normal(0, vol / 200))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 200)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 200)))
    return pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n).strftime("%Y-%m-%d"), "open": o, "high": h, "low": lo, "close": c})


def main():
    n = 0
    # 1. _sq：σ 全為 4 → sq = 原分位 / 2
    gg = pd.DataFrame({"intra": np.linspace(-2, 2, 50), "dip": np.linspace(-3, 0, 50), "pop": np.linspace(0, 3, 50), "hold": [True, False] * 25, "s": 4.0})
    sq = PC._sq(gg)
    assert abs(sq["intra_p20"] - gg["intra"].quantile(0.2) / 2) < 1e-3 and abs(sq["dip_hold_q50"] - gg["dip"][gg["hold"]].median() / 2) < 1e-3, sq
    assert PC._sq(gg.drop(columns="s")) is None and PC._sq(gg.head(10)) is None
    n += 2

    # 2. scale_cell
    cell = {"intra_p20": -0.9, "intra_p80": 0.8, "dip_hold_q20": None, "sq": {"intra_p20": -0.5, "intra_p80": 0.4, "dip_hold_q20": -0.3}}
    c2 = PC.scale_cell(cell, 9.0)
    assert c2["intra_p20"] == -1.5 and c2["intra_p80"] == 1.2 and c2["raw_q"]["intra_p20"] == -0.9 and c2["q_scaled"]["beta"] == 0.5, c2
    assert c2["dip_hold_q20"] is None, "原值缺 (樣本不足) → 不硬補"
    assert cell["intra_p20"] == -0.9, "不改原 dict"
    assert PC.scale_cell(cell, None) is cell and PC.scale_cell(cell, float("nan")) is cell and PC.scale_cell({"intra_p20": 1}, 2.0) == {"intra_p20": 1}
    n += 3

    # 3. train → 每格 sq；4. build (盤中實際開盤) → stats 為縮放後
    sc = frame()
    orig_nh, orig_load = ST._night_hist, M.load_json
    try:
        ST._night_hist = lambda *a, **k: None
        st = PC.train(sc, write=False, verbose=False)
        cells = st["gap"]["cells"]
        assert cells and all("sq" in c for c in cells.values() if c["n"] >= 20), [k for k, c in cells.items() if "sq" not in c]
        M.load_json = lambda name, *a, **k: st if name == "precheck" else orig_load(name, *a, **k)
        prev = float(sc["close"].iloc[-1])
        out = PC.build(sc, {"taiex": {"open": prev * 1.008, "prev": prev}}, {"intraday": True})
        gp = out["gap"]; s_now = float(RL.sigma_series(sc).iloc[-1])
        raw = cells[f"{int(np.searchsorted(PC.GAP_EDGES, 0.8, side='right'))}|{'bull' if prev >= sc['close'].rolling(20).mean().iloc[-1] else 'bear'}"]
        assert gp["stats"]["intra_p20"] == round(raw["sq"]["intra_p20"] * s_now ** 0.5, 2) and gp["stats"]["raw_q"]["intra_p20"] == raw["intra_p20"], (gp["stats"], raw)
        assert f"{gp['stats']['intra_p20']:+.2f}~" in gp["text"], gp["text"]
        n += 3
    finally:
        ST._night_hist, M.load_json = orig_nh, orig_load
    print(f"test_precheck_scale: {n} checks passed")


if __name__ == "__main__":
    main()
