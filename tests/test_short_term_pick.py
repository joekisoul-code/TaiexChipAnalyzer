"""10-19 短線選模改「叫牌超額」(edge_per100) + 現行模型遲滯 (KEEP_BAND)。

    python tests/test_short_term_pick.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import short_term as ST  # noqa: E402


def oos(up_hit, dn_hit, n=3000, base=0.55, seed=0, ic_noise=1.0):
    """合成 OOS：pred 前 30% 的實際上漲率 = up_hit、後 30% 的下跌率 = dn_hit，中間依 base；年份 2014~ 平均分。"""
    r = np.random.default_rng(seed)
    pred = np.arange(n, dtype=float); r.shuffle(pred)
    q = pd.Series(pred).rank(pct=True).to_numpy()
    p_up = np.where(q >= 0.7, up_hit, np.where(q <= 0.3, 1 - dn_hit, base))
    actual = np.where(r.random(n) < p_up, 1.0, -1.0) * r.uniform(0.1, 2, n)
    return pd.DataFrame({"date": [f"{2014 + i * 13 // n}-01-{1 + i % 28:02d}" for i in range(n)], "year": [2014 + i * 13 // n for i in range(n)],
                         "pred": pred + r.normal(0, ic_noise, n), "actual": actual})


def main():
    both = oos(0.60, 0.52, seed=1)      # 兩邊都啟用、命中普通 (覆蓋 60%)
    one = oos(0.64, 0.40, seed=2)       # 只有偏多啟用、命中較高 (覆蓋 30%)
    rb, ro = ST._report(both), ST._report(one)
    assert rb["tiers"]["call_cov"] > 0.55 and ro["tiers"]["call_cov"] < 0.35, (rb["tiers"]["call_cov"], ro["tiers"]["call_cov"])
    assert ro["tiers"]["call_hit"] > rb["tiers"]["call_hit"], "前提：單邊命中率較高"
    # edge_per100 = 每 100 日多叫對幾次；手算對照
    t = rb["tiers"]; exp = (t["up_n"] * (t["up_hit"] - t["base_up"]) + t["dn_n"] * (t["dn_hit"] - (1 - t["base_up"]))) / rb["n"] * 100
    assert abs(t["edge_per100"] - exp) < 0.01, (t["edge_per100"], exp)
    assert rb["tiers"]["edge_per100"] > ro["tiers"]["edge_per100"]
    k, reps = ST._pick({"both": both, "one": one})
    assert k == "both", "超額較大的兩邊模型勝出 (舊規則會選單邊高命中)"
    # 遲滯：現行模型在最佳 KEEP_BAND 內 → 保留；超出 → 換；現行不在候選 / 沒有 → 最佳
    near = oos(0.60, 0.52, seed=1, ic_noise=1.3)
    e_b, e_n = ST._report(both)["tiers"]["edge_per100"], ST._report(near)["tiers"]["edge_per100"]
    if e_n <= e_b and e_b - e_n <= ST.KEEP_BAND:
        assert ST._pick({"both": both, "near": near}, prev="near")[0] == "near"
    assert ST._pick({"both": both, "one": one}, prev="one")[0] == "both", "差距超過 KEEP_BAND 就換"
    assert ST._pick({"both": both, "one": one}, prev="gone")[0] == "both"
    assert ST._pick({"both": both, "one": one}, prev=None)[0] == "both"
    # 都沒啟用叫牌 (超額 0) → 次序 IC 決勝
    flat_hi, flat_lo = oos(0.55, 0.45, seed=3, ic_noise=0.5), oos(0.55, 0.45, seed=4, ic_noise=5000)
    r1, r2 = ST._report(flat_hi), ST._report(flat_lo)
    assert r1["tiers"]["edge_per100"] == 0 == r2["tiers"]["edge_per100"]
    assert ST._pick({"lo": flat_lo, "hi": flat_hi})[0] == ("hi" if (r1["rank_ic"] or 0) >= (r2["rank_ic"] or 0) else "lo")
    # 空候選略過
    assert ST._pick({"e": pd.DataFrame(), "both": both})[0] == "both"
    print("test_short_term_pick: 9 checks passed")


if __name__ == "__main__":
    main()
