"""2026-11-04 延續模型：13:30 整點之間區間縮放 (v_z / _cont_k) + 短目標 (現在 → 11:00 / 12:00 / 13:00) 延續表：

- _cont_table 存 v_z = ((q80−q20 of y/σ)/1.6832)²；_cont_k：mark 當下 = 1、到下一時點 = √(V(下一)/V(本))、中間依時間線性內插變異；13:00 後 / 舊表 → 1
- cont_tables：short[目標][時點] 表 + short_metrics；只建 CONT_SHORT 的 (時點, 目標)
- forecast 11:30：13:30 區間依 k 縮 (上漲率 / 平均不變)；短目標 model=cont、區間依 √(剩餘/全長) 縮；
  上漲率只在 11:00 → 13:00 (CONT_SHORT_CALL) 且離目標 ≥ 30 分鐘時給，其他格 p_up / base_hit = None (App 顯示 —)
- 12:45 → 13:00 只剩 15 分鐘 → 不給上漲率

    python tests/test_intraday_cont_short.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import intraday as I  # noqa: E402
from chip.predict import range_levels as RL  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print("  ✓", name)


def hist_rows(seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-06-01", "2026-10-30").strftime("%Y-%m-%d")
    sig = {d: float(s) for d, s in zip(days, rng.uniform(0.6, 2.0, len(days)))}
    vol = {"10:00": 0.5, "11:00": 0.4, "12:00": 0.3, "13:00": 0.2}   # 剩餘時間越短，y/σ 變異越小
    rows = []
    for d in days:
        for mk in I.CONT_MARKS:
            ro = rng.normal(0, 0.5) * sig[d]
            r = {"date": d, "mark": mk, "ret_open": ro, "to_13:30": (0.3 * ro + rng.normal(0, vol[mk])) * sig[d] if True else 0}
            for tm in ("11:00", "12:00", "13:00"):
                r[f"to_{tm}"] = (0.2 * ro + rng.normal(0, 0.2)) * sig[d] if tm > mk else np.nan
            rows.append(r)
    return pd.DataFrame(rows), sig


def main():
    H, sp = hist_rows()
    ct = I.cont_tables(H, sp)
    g = H[(H["mark"] == "10:00") & (H["date"] >= I.CONT_FROM)]
    yz = (g["to_13:30"] / g["date"].map(sp)).to_numpy()
    v = ((np.quantile(yz, 0.8) - np.quantile(yz, 0.2)) / 1.6832) ** 2
    M_ = ct["marks"]
    check("v_z = ((q80−q20)/1.6832)²，剩餘時間短的時點變異較小", abs(M_["10:00"]["v_z"] - round(v, 5)) < 1e-9 and M_["10:00"]["v_z"] > M_["11:00"]["v_z"] > M_["12:00"]["v_z"])
    k_end = I._cont_k(M_, "10:00", "11:00")
    check("_cont_k：整點當下 1、到下一時點 √(V下/V本)、中間變異線性內插；13:00 後與舊表 (無 v_z) → 1",
          I._cont_k(M_, "10:00", "10:00") == 1.0 and abs(k_end - np.sqrt(M_["11:00"]["v_z"] / M_["10:00"]["v_z"])) < 1e-12
          and abs(I._cont_k(M_, "10:00", "10:30") - np.sqrt((M_["10:00"]["v_z"] + M_["11:00"]["v_z"]) / 2 / M_["10:00"]["v_z"])) < 1e-12
          and I._cont_k(M_, "13:00", "13:20") == 1.0 and I._cont_k({"10:00": {}, "11:00": {}}, "10:00", "10:30") == 1.0)
    check("short 表：只建 CONT_SHORT 的 (時點, 目標)，各有走動式指標",
          {tm: sorted(d) for tm, d in ct["short"].items()} == {tm: sorted(m) for tm, m in I.CONT_SHORT.items()}
          and all(mk in ct["short_metrics"][tm] for tm, m in I.CONT_SHORT.items() for mk in m))

    # ---------------------------------------------------------------- forecast
    day = "2026-11-04"
    bars = [{"time": f"{h:02d}:{mi:02d}", "close": 100.0 + (h - 9) * 0.3 + mi * 0.005} for h in (9, 10, 11, 12, 13) for mi in range(60) if f"{h:02d}:{mi:02d}" <= "13:30"]
    px = {b["time"]: b["close"] for b in bars}

    class _Stub:
        def predict(self, X, pred_contrib=False):
            return np.zeros((len(X), X.shape[1] + 1)) if pred_contrib else np.full(len(X), 0.05)

    cal = {"edges": [-0.1, 0.0, 0.1, 0.2], "table": [{"bin": k, "n": 100, "hit": 0.5, "mean": 0.0, "p20": -0.3, "p80": 0.3} for k in range(5)], "base_hit": 0.52, "base_mean": 0.01}
    bundle = {"features": I.FEATURES, "trained_at": "x", "days": 1, "source": "twse", "models": {tm: [_Stub()] for tm in I.TARGET_MARKS},
              "calibration": {tm: {"intra": cal, "pre": cal} for tm in I.TARGET_MARKS}, "metrics": {tm: {} for tm in I.TARGET_MARKS},
              "early": {"marks": ["09:15", "09:30"], "models": {tm: [_Stub()] for tm in I.TARGET_MARKS}, "calibration": {tm: {"intra": cal} for tm in I.TARGET_MARKS}, "metrics": {}}, "cont": ct}
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
    scored = pd.DataFrame({"date": pd.bdate_range("2026-05-11", periods=120).strftime("%Y-%m-%d"), "open": c, "high": c * 1.01, "low": c * 0.99, "close": c})
    scored = scored[scored["date"] < day].reset_index(drop=True)
    sig = float(RL.sigma_series(scored).iloc[-1])
    saved = (I.M.load, I.market_matrix, I._daily_for_next, I.dt)

    def at(hm):
        class _F(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 11, 4, int(hm[:2]), int(hm[3:]), tzinfo=tz)
        I.dt = type("dtmod", (), {"datetime": _F, "date": dt.date, "timedelta": dt.timedelta})

    def fc_at(hm):
        at(hm)
        return I.forecast(scored, {"phase": "open", "intraday": {"bars": [b for b in bars if b["time"] <= hm]}, "taiex": {"prev": 99.0, "last": px[hm]}})

    I.M.load = lambda name: bundle
    I.market_matrix = lambda s: pd.DataFrame({"date": s["date"].values, "close": s["close"].values})
    I._daily_for_next = lambda mat, d: {f: 0.0 for f in I.FEATURES if f.startswith(("prev_", "g_")) or f in ("dow", "days_to_settle")}
    try:
        fc = fc_at("11:30")
        z = (px["11:30"] / px["09:05"] - 1) * 100 / sig
        T = M_["11:00"]; b = int(np.digitize([z], T["z_edges"])[0]); B = T["bins"][b]
        k = I._cont_k(M_, "11:00", "11:30"); hm = B["mean_z"] * sig
        t = fc["targets"]["13:30"]
        check("13:30 區間依 k 縮放 (k<1)，上漲率 / 平均不變",
              0 < k < 1 and t["band_k"] == round(k, 3) and t["p_up"] == B["p_up"] and t["pred"] == round(hm, 2)
              and t["q20"] == round(hm + (B["q20_z"] * sig - hm) * k, 2) and t["q80"] == round(hm + (B["q80_z"] * sig - hm) * k, 2))
        t13 = fc["targets"]["13:00"]; T2 = ct["short"]["13:00"]["11:00"]; b2 = int(np.digitize([z], T2["z_edges"])[0]); B2 = T2["bins"][b2]
        k2 = np.sqrt((13 * 60 - (11 * 60 + 30)) / 120); h2 = B2["mean_z"] * sig
        check("短目標 11:00→13:00：延續表、上漲率照給 (離目標 90 分)、區間依 √(剩餘/全長) 縮",
              t13["model"] == "cont" and t13["call_ok"] and t13["p_up"] == B2["p_up"] and t13["base_hit"] == T2["base_hit"]
              and t13["band_k"] == round(k2, 3) and t13["q80"] == round(h2 + (B2["q80_z"] * sig - h2) * k2, 2))
        t12 = fc["targets"]["12:00"]
        check("11:00→12:00：區間用延續表，但不給上漲率 (p_up / base_hit = None)",
              t12["model"] == "cont" and not t12["call_ok"] and t12["p_up"] is None and t12["base_hit"] is None and t12["p_up_table"] is not None)
        check("11:30 已過 11:00 目標 → 不出現", "11:00" not in fc["targets"])
        fc2 = fc_at("12:45")
        check("12:45：12:00→13:00 不給上漲率 (非叫牌格且只剩 15 分)", fc2["targets"]["13:00"]["p_up"] is None and fc2["targets"]["13:00"]["model"] == "cont")
        bundle["cont"] = {**ct, "short": {}}
        fc3 = fc_at("11:30")
        check("舊 cont (無 short) → 短目標仍 LightGBM", "model" not in fc3["targets"]["13:00"] and fc3["targets"]["13:00"]["p_up"] == 0.5)
    finally:
        I.M.load, I.market_matrix, I._daily_for_next, I.dt = saved
    print(f"\nintraday cont short: {ok} passed")


if __name__ == "__main__":
    main()
