"""大盤回落模型夜盤對齊 (r2m B1) 單元測試 (final_spec §3.1)：
1. 列 D 的夜盤 = FinMind date 為 next_td(D) 的那一晚 (D 收盤後)；
2. date==D 錯位合併時回歸守門要 raise；
3. build() 讀到沒有 night.aligned 旗標的舊 pullback.json → 夜盤變體 use_model=False (0a 停損)；
4. export_static.apply_pullback：IV 生效時 base 變體看 use_model_iv (X1)。

    python tests/test_pullback_night_align.py
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

from chip.predict import pullback as PB  # noqa: E402


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _synthetic(n=600, seed=3):
    """交易日序列 (工作日)；夜盤 night[date=D+1] 發生在 D 收盤後 → D+1 開盤跳空 ≈ 0.8 × 該夜盤。"""
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-01-04", periods=n)]
    night = rng.normal(0, 0.9, n)                      # night[i] = D_{i-1} 收盤後、D_i 開盤前
    close = np.empty(n); open_ = np.empty(n)
    c = 15000.0
    for i in range(n):
        o = c * (1 + (0.8 * night[i] + rng.normal(0, 0.15)) / 100) if i else c
        c = o * (1 + rng.normal(0, 0.6) / 100)
        open_[i], close[i] = o, c
    m = pd.DataFrame({"date": dates, "open": open_, "close": close})
    nh = pd.DataFrame({"date": dates[1:], "night_chg_pct": night[1:]})       # FinMind 語意：date = 夜盤準備的交易日
    nh.loc[5, "night_chg_pct"] = 12.0                                        # 極端值 → 剔除
    return m, nh


def test_night_merge_aligned():
    m, nh = _synthetic()
    mn, corr = PB._night_merge(m, nh)
    assert corr > 0.7, corr
    nm = dict(zip(nh["date"], nh["night_chg_pct"]))
    nxt = dict(zip(m["date"], m["date"].shift(-1)))
    for _, r in mn.iterrows():
        exp = nm.get(nxt[r["date"]]) if isinstance(nxt[r["date"]], str) else None
        if exp is None:
            assert np.isnan(r["night_chg_pct"])
        else:
            assert r["night_chg_pct"] == exp, (r["date"], r["night_chg_pct"], exp)
    assert np.isnan(mn["night_chg_pct"].iloc[-1])                           # 末列沒有 D+1 → NaN (保留，交給 LGB)
    bad_date = nh.loc[5, "date"]
    assert m.loc[m["date"].shift(-1) == bad_date, "date"].iloc[0] not in set(mn["date"])     # |夜盤| > 8 剔除


def test_guard_raises_on_date_eq_d_merge():
    m, nh = _synthetic()
    wrong = m.merge(nh, on="date", how="left")                                # 舊版 (B1)：date==D → 拿到 D 開盤前那一晚
    try:
        PB._align_guard(wrong)
    except ValueError as e:
        assert "守門" in str(e)
    else:
        raise AssertionError("date==D 合併應觸發守門")


def _fake_frame(scored):
    m = pd.DataFrame({"date": ["2026-09-22"], "close": [20000.0], "sigma_range": [1.2], "f1": [0.1], "ma5": [19900.0], "ma20": [19500.0],
                      "ma60": [19000.0], "prev_low": [19800.0], "lo20": [19000.0]})
    return m, ["f1"]


class _Q:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def _fake_load(name):
    feats = ["f1", "night_chg_pct"] if name.endswith("_night") else ["f1"]
    return {"buy": [_Q(-0.8)], "stop": [_Q(-1.2)], "features": feats}


def _build(st, night_ret):
    with patched(PB, "_frame", _fake_frame), patched(PB.M, "load_json", lambda name: st), patched(PB.M, "load", _fake_load):
        return PB.build(pd.DataFrame(), None, night_ret=night_ret)


def test_build_night_variant_needs_aligned_flag():
    k = {str(i): {"use_model": True, "improve_pinball": 0.1} for i in (1, 2, 3)}
    old = {"k": dict(k), "night": {"features": ["f1", "night_chg_pct"], "k": dict(k)}}          # 舊 json：沒有 aligned
    out = _build(old, 0.7)
    assert out["variant"] == "night"
    assert all(v["variant"] == "night" and v["use_model"] is False for v in out["k"].values())
    new = {"k": dict(k), "night": {"features": ["f1", "night_chg_pct"], "k": dict(k), "aligned": True}}
    out2 = _build(new, 0.7)
    assert all(v["use_model"] is True for v in out2["k"].values())                               # 旗標存在 → 依重訓結果
    out3 = _build(old, None)                                                                     # base 變體不受影響
    assert out3["variant"] == "base" and all(v["use_model"] is True and v["variant"] == "base" for v in out3["k"].values())
    kiv = {str(i): {"use_model": True, "use_model_iv": False, "improve_vs_iv": 0.01, "iv": {"n": 900}} for i in (1, 2, 3)}
    out4 = _build({"k": kiv, "night": {}}, None)
    assert out4["k"]["1"]["use_model_iv"] is False and out4["k"]["1"]["oos_iv"]["improve_vs_iv"] == 0.01


def test_export_apply_pullback_x1():
    import export_static as X
    def pb(use_iv):
        return {"k": {"1": {"variant": "base", "use_model": True, "use_model_iv": use_iv, "buy_model": 19700, "stop_model": 19500},
                      "2": {"variant": "base", "use_model": True, "buy_model": 19600, "stop_model": 19400}}}
    nd = [{"n": 1, "range_mode": "base", "range_sigma_src": "txo_iv", "buy_at": 19800, "stop": 19650, "level_lo": 19800},
          {"n": 2, "range_mode": "base", "range_sigma_src": "atr_ewma", "buy_at": 19750, "stop": 19550, "level_lo": 19750}]
    p = pb(False)
    X.apply_pullback(nd, p)
    assert nd[0]["buy_at"] == 19800 and nd[0]["stop"] == 19650 and "buy_src" not in nd[0]          # IV 生效：不覆寫
    assert nd[0]["buy_at_sigma"] == 19800 and nd[0]["buy_at_pullback"] == 19700 and nd[0]["stop_pullback"] == 19500
    assert p["k"]["1"]["use_model"] is False and p["k"]["1"]["use_model_atr"] is True
    assert nd[1]["buy_at"] == 19600 and nd[1]["buy_src"] == "model" and nd[1]["buy_at_sigma"] == 19750   # IV 缺 → ATR → 照舊覆寫
    nd2 = [{"n": 1, "range_mode": "base", "range_sigma_src": "txo_iv", "buy_at": 19800, "stop": 19650, "level_lo": 19800}]
    X.apply_pullback(nd2, pb(True))
    assert nd2[0]["buy_at"] == 19700 and nd2[0]["buy_src"] == "model"                               # use_model_iv 通過時仍可覆寫
    nd3 = [{"n": 1, "range_mode": "base_event", "range_sigma_src": "atr_ewma", "buy_at": 19800, "stop": 19650}]
    X.apply_pullback(nd3, pb(True))
    assert nd3[0]["buy_at"] == 19800 and "buy_at_sigma" not in nd3[0]                               # 休市後首日不覆寫
    nd4 = [{"n": 1, "range_mode": "night", "range_sigma_src": "txo_iv", "buy_at": 19800, "stop": 19650}]
    pbn = {"k": {"1": {"variant": "night", "use_model": False, "buy_model": 19700, "stop_model": 19500}}}
    X.apply_pullback(nd4, pbn)
    assert nd4[0]["buy_at"] == 19800 and "buy_at_pullback" not in nd4[0]                            # 夜盤變體 (B1 停用) 不覆寫


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
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
