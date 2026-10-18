"""10-18 chips-3：market.json 16 因子 / 檢查表的大盤融資維持率在玩股網缺時改用 TWSE 自算 (daily_metrics maint_twse，覆蓋 100%)。

    python tests/test_market_maint_twse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.analysis import market as MK  # noqa: E402

N = 80


def _df(wg_until=None):
    dates = pd.bdate_range("2026-06-01", periods=N).strftime("%Y-%m-%d")
    rng = np.random.default_rng(1)
    close = 40000 * np.cumprod(1 + rng.normal(0, 0.01, N))
    d = pd.DataFrame({"date": dates, "close": close, "open": close, "high": close * 1.01, "low": close * 0.99, "amount": 5000.0,
                      "foreign": rng.normal(0, 100, N), "trust": 10.0, "dealer": 5.0, "total": 15.0, "margin_amt": 3000.0, "margin_lots": 1e6,
                      "short_lots": 2e5, "gov8_net": 1.0, "pcr_oi": 100.0})
    d["maint_ratio"] = np.nan
    if wg_until is not None:
        d.loc[: wg_until, "maint_ratio"] = 175.0
    return d


def _metrics(d, days, cov=99.65):        # 真實 maint_cov 是市值覆蓋率 99.6~99.99，不會剛好 100
    return pd.DataFrame({"date": [d["date"].iloc[i] for i in days], "maint_twse": [183.4] * len(days), "maint_cov": [cov] * len(days)})


def test_fill_tail_only_same_or_prev_day():
    d = _df(wg_until=N - 6)
    m = _metrics(d, [N - 10, N - 4, N - 2])                 # N-10 在玩股網段內 → 不覆寫
    out = MK.fill_maint_twse(d, m)
    assert out["maint_ratio"].iloc[N - 10] == 175.0 and out["maint_twse_fill"].iloc[N - 10] == 0.0
    assert np.isnan(out["maint_ratio"].iloc[N - 5])          # 前一日也沒有 → 不補
    assert out["maint_ratio"].iloc[N - 4] == 183.4 and out["maint_ratio"].iloc[N - 3] == 183.4   # N-3 沿用前 1 日
    assert out["maint_ratio"].iloc[N - 1] == 183.4 and out["maint_twse_fill"].iloc[N - 1] == 1.0


def test_partial_coverage_not_used():
    d = _df()
    out = MK.fill_maint_twse(d, _metrics(d, [N - 1], cov=-90))
    assert out["maint_ratio"].isna().all()
    out = MK.fill_maint_twse(d, _metrics(d, [N - 1], cov=-80))
    assert out["maint_ratio"].isna().all() and (out["maint_twse_fill"] == 0.0).all()


def test_real_coverage_below_100_is_filled():
    """10-18 審查：真實 DB maint_cov 範圍 99.63~99.99、無一筆 == 100 → 舊的 == 100 過濾讓補值永遠不跑。"""
    d = _df()
    for cov in (99.65, 99.63, 99.99, 100):
        out = MK.fill_maint_twse(d, _metrics(d, [N - 1], cov=cov))
        assert out["maint_ratio"].iloc[N - 1] == 183.4 and out["maint_twse_fill"].iloc[N - 1] == 1.0, cov


def test_assess_factor_and_checklist_use_twse():
    d = MK.fill_maint_twse(_df(), _metrics(d0 := _df(), list(range(N - 30, N))))
    sc = MK.score_frame(MK.add_features(d))
    a = MK.assess(sc)
    f = next(x for x in a["factors"] if x.key == "maint")
    assert f.available and "TWSE自算" in f.value and "183.40" in f.value, f
    ck = dict(a["checklist"])
    assert ck["融資維持率安全 (≥ 165%)"] is True
    assert a["total"] == 9 - sum(1 for _, v in a["checklist"] if v is None) and ck["融資維持率安全 (≥ 165%)"] is not None
    # 無任何維持率 → 因子不可用、文案不再寫「需玩股網資料」
    a2 = MK.assess(MK.score_frame(MK.add_features(MK.fill_maint_twse(_df(), pd.DataFrame()))))
    f2 = next(x for x in a2["factors"] if x.key == "maint")
    assert not f2.available and "需玩股網" not in f2.comment
    assert dict(a2["checklist"])["融資維持率安全 (≥ 165%)"] is None and a2["total"] == a["total"] - 1


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
