"""2026-11-02 小時模型 10:00~13:00 → 13:30 改延續模型 (cont) + 10:xx 開盤價 bug + 對帳分模型：

- cont_tables：只用 CONT_MARKS、CONT_FROM 起的列；z = ret_open/σ_prev 的 5 分位切點、各區 y/σ 統計、base_hit；走動式叫牌統計
- _live_rec 10:30：open = 09:05 價 (11-02 前被換成一小時前 09:30 的價)；ret_1h 與訓練同口徑 (以 09:05 開盤計)
- forecast 10:30：13:30 目標 model='cont'、p_up = 該區上漲率、level = 現價 × (1 + mean_z×σ/100)；其他目標仍 LightGBM；
  σ 用前一交易日 (scored 含今日列時往前一筆)；09:10 (標 10:00 的近似) / 無 cont 表 → 原 LightGBM
- learn：09:00~09:14 標 10:00 的發布不記帳；紀錄帶 model；summarize 有 by_model

    python tests/test_intraday_cont.py
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
from chip.predict import learn as L  # noqa: E402
from chip.predict import range_levels as RL  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print("  ✓", name)


def hist_rows(seed=2):
    """合成 build_history 列：延續效應 y = 0.3·ret_open + 雜訊 (σ 單位)。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2026-10-30").strftime("%Y-%m-%d")
    sig = {d: float(s) for d, s in zip(days, rng.uniform(0.6, 2.0, len(days)))}
    rows = []
    for d in days:
        for mk in ["pre"] + I.CONT_MARKS:
            ro = np.nan if mk == "pre" else rng.normal(0, 0.5) * sig[d]
            y = (0.3 * ro if mk != "pre" else 0) + rng.normal(0, 0.4) * sig[d]
            rows.append({"date": d, "mark": mk, "ret_open": ro, "to_13:30": y})
    return pd.DataFrame(rows), sig


def main():
    H, sp = hist_rows()
    ct = I.cont_tables(H, sp)
    sel = H[H["mark"].isin(I.CONT_MARKS) & (H["date"] >= I.CONT_FROM)]
    T = ct["marks"]["11:00"]; g = sel[sel["mark"] == "11:00"]
    z = (g["ret_open"] / g["date"].map(sp)).to_numpy(); yz = (g["to_13:30"] / g["date"].map(sp)).to_numpy()
    e = np.quantile(z, [0.2, 0.4, 0.6, 0.8])
    check("cont_tables：只 10:00~13:00、2018~ 的列；z 切點 = 5 分位；base_hit = y>0 比例",
          set(ct["marks"]) == set(I.CONT_MARKS) and T["n"] == len(g) and I.CONT_FROM == "2018-01-01" and ct["from"] >= "2018-01-01"   # 走動式驗證的設定 (2016~ 版不同)
          and np.allclose(T["z_edges"], np.round(e, 4)) and T["base_hit"] == round(float((yz > 0).mean()), 3))
    b4 = z >= e[3]
    check("最高區 (z 最大) 上漲率 / 平均 = 該區 y/σ 統計；延續效應 → 最高區上漲率 > 最低區",
          T["bins"][4]["p_up"] == round(float((yz[b4] > 0).mean()), 3) and abs(T["bins"][4]["mean_z"] - round(float(yz[b4].mean()), 4)) < 1e-9
          and T["bins"][4]["p_up"] > T["bins"][0]["p_up"] + 0.1)
    m = ct["metrics"]
    check("走動式指標：各時間點與全體都有叫牌比例 / 命中 / 淨正確 / 區間覆蓋，測試月 ≥ 2019 (月前資料 ≥ 250 列)",
          all(k in m for k in I.CONT_MARKS + ["all"]) and m["all"]["call_hit"] > 55 and 50 < m["all"]["band_cov"] < 70 and m["all"]["from"] >= "2019-01")
    r = I._cont_apply(T, 99.0, 1.5)
    check("_cont_apply：z 超過最高切點 → 第 4 區；mean / q20 / q80 × σ", r["bin"] == 4 and abs(r["mean"] - T["bins"][4]["mean_z"] * 1.5) < 1e-12 and r["q80"] > r["q20"])

    # ---------------------------------------------------------------- _live_rec 10:30
    day = "2026-11-02"
    bars = [{"time": f"{h:02d}:{mi:02d}", "close": 100.0 + (h - 9) * 0.5 + mi * 0.01} for h in (9, 10) for mi in range(60) if f"{h:02d}:{mi:02d}" <= "10:30"]
    px = {b["time"]: b["close"] for b in bars}
    snap = {"phase": "open", "intraday": {"bars": bars}, "taiex": {"prev": 99.0, "last": px["10:30"]}}
    orig_dt = I.dt

    def at(hm):
        class _F(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 11, 2, int(hm[:2]), int(hm[3:]), tzinfo=tz)
        I.dt = type("dtmod", (), {"datetime": _F, "date": dt.date, "timedelta": dt.timedelta})

    try:
        at("10:30")
        rec, mk = I._live_rec(snap, early=True)
        check("_live_rec 10:30：mark 10:00、open = 09:05 價 (不再被換成 09:30 的價)",
              mk == "10:00" and rec["open"] == px["09:05"] == rec["open_0905"] and rec["open"] != px["09:30"])
        row = I.mark_rows(day, rec, {}, None, live_mark=mk)[0]
        check("10:30 的 gap_pct / ret_open 以 09:05 開盤計，ret_1h 與訓練同口徑 (= ret_open)",
              abs(row["gap_pct"] - (px["09:05"] / 99.0 - 1) * 100) < 1e-9 and abs(row["ret_open"] - (px["10:30"] / px["09:05"] - 1) * 100) < 1e-9
              and abs(row["ret_1h"] - row["ret_open"]) < 1e-12)

        # ------------------------------------------------------------ forecast
        class _Stub:
            def __init__(self, v):
                self.v = v

            def predict(self, X, pred_contrib=False):
                return np.zeros((len(X), X.shape[1] + 1)) if pred_contrib else np.full(len(X), self.v)

        cal = {"edges": [-0.1, 0.0, 0.1, 0.2], "table": [{"bin": k, "n": 100, "hit": 0.5, "mean": 0.0, "p20": -0.3, "p80": 0.3} for k in range(5)], "base_hit": 0.52, "base_mean": 0.01}
        bundle = {"features": I.FEATURES, "trained_at": "x", "days": 1, "source": "twse", "models": {tm: [_Stub(0.05)] for tm in I.TARGET_MARKS},
                  "calibration": {tm: {"intra": cal, "pre": cal} for tm in I.TARGET_MARKS}, "metrics": {tm: {} for tm in I.TARGET_MARKS},
                  "early": {"marks": ["09:15", "09:30"], "models": {tm: [_Stub(0.0)] for tm in I.TARGET_MARKS}, "calibration": {tm: {"intra": cal} for tm in I.TARGET_MARKS}, "metrics": {}},
                  "cont": ct}
        rng = np.random.default_rng(1)
        n = 120; c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        scored = pd.DataFrame({"date": pd.bdate_range("2026-05-11", periods=n).strftime("%Y-%m-%d"), "open": c, "high": c * 1.01, "low": c * 0.99, "close": c})
        scored = scored[scored["date"] < day].reset_index(drop=True)
        sig_prev = float(RL.sigma_series(scored).iloc[-1])
        saved = (I.M.load, I.market_matrix, I._daily_for_next)
        I.M.load = lambda name: bundle
        I.market_matrix = lambda s: pd.DataFrame({"date": s["date"].values, "close": s["close"].values})
        I._daily_for_next = lambda mat, d: {f: 0.0 for f in I.FEATURES if f.startswith(("prev_", "g_")) or f in ("dow", "days_to_settle")}
        try:
            fc = I.forecast(scored, snap)
            t = fc["targets"]["13:30"]
            ro = (px["10:30"] / px["09:05"] - 1) * 100; zz = ro / sig_prev; Tm = ct["marks"]["10:00"]
            bb = int(np.digitize([zz], Tm["z_edges"])[0])
            check("forecast 10:30：13:30 = 延續模型 (model=cont)，p_up / base_hit = 10:00 表該區，level = 現價 × (1 + mean_z·σ/100)",
                  t["model"] == "cont" and t["bin"] == bb and t["p_up"] == Tm["bins"][bb]["p_up"] and t["base_hit"] == Tm["base_hit"]
                  and t["level"] == round(px["10:30"] * (1 + Tm["bins"][bb]["mean_z"] * sig_prev / 100), 0) and fc["cont"]["sigma_prev"] == round(sig_prev, 3)
                  and t["lgbm"]["p_up"] == 0.5 and fc["asof_t"] == "10:30" and "延續模型" in fc["note"])
            check("其他目標 (11:00/12:00/13:00) 仍為 LightGBM 校準", all("model" not in fc["targets"][tm] and fc["targets"][tm]["p_up"] == 0.5 for tm in ("11:00", "12:00", "13:00")))
            today_row = pd.DataFrame({"date": [day], "open": [c[-1] * 1.05], "high": [c[-1] * 1.2], "low": [c[-1] * 0.8], "close": [c[-1] * 1.1]})
            fc2 = I.forecast(pd.concat([scored, today_row], ignore_index=True), snap)
            check("scored 已含今日列 → σ 仍用前一交易日", fc2["cont"]["sigma_prev"] == round(sig_prev, 3) and fc2["cont"]["sigma_date"] == scored["date"].iloc[-1])
            at("09:10")
            fc3 = I.forecast(scored, {**snap, "intraday": {"bars": [b for b in bars if b["time"] <= "09:10"]}})
            check("09:10 (標 10:00 的近似)：不用延續模型", fc3["mark"] == "10:00" and "model" not in fc3["targets"]["13:30"] and fc3["asof_t"] == "09:10")
            at("10:30")
            bundle.pop("cont")
            fc4 = I.forecast(scored, snap)
            check("舊 bundle 無 cont → 13:30 仍為 LightGBM", "model" not in fc4["targets"]["13:30"] and "cont" not in fc4)
        finally:
            I.M.load, I.market_matrix, I._daily_for_next = saved
    finally:
        I.dt = orig_dt

    # ---------------------------------------------------------------- learn
    hr = {"live": True, "day": day, "mark": "10:00", "price": 100.0, "asof_t": "09:10", "model": "main",
          "targets": {"13:30": {"p_up": 0.6, "base_hit": 0.5, "level": 101.0}}}
    check("learn：09:10 標 10:00 的發布不記帳 (會擋掉真正 10:00 的紀錄)", L.records_from_hourly(hr, None) == [])
    r1 = L.records_from_hourly({**hr, "asof_t": "10:30", "targets": {"13:30": {**hr["targets"]["13:30"], "model": "cont"}}}, None)
    r2 = L.records_from_hourly({**hr, "mark": "09:15", "asof_t": "09:20", "model": "early"}, None)
    r3 = L.records_from_hourly({k: v for k, v in hr.items() if k != "asof_t"}, None)
    check("紀錄帶 model：cont / early / 舊 (無 asof_t) → lgbm", r1[0]["model"] == "cont" and r1[0]["call"] == "偏多" and r2[0]["model"] == "early" and r3[0]["model"] == "lgbm")
    rows = []
    for i, (mdl, hit) in enumerate([("lgbm", True)] * 6 + [("cont", False)] * 6):
        rows.append({"kind": "hr", "sid": "TAIEX", "as_of": f"2026-10-{i + 1:02d}", "target": f"2026-10-{i + 1:02d}", "h": 0, "mode": "live:10:00", "mark": "10:00",
                     "p_up": 0.6, "base_hit": 0.5, "call": "偏多", "model": mdl if mdl != "lgbm" or i % 2 else None,
                     "realized": {"hit": hit, "up": hit, "brier": 0.2, "ret": 0.1}})
    hs = L.summarize(rows)["market"]["hourly"]
    check("summarize：hourly.by_model 分開 lgbm (含舊紀錄無 model) / cont", hs["by_model"]["lgbm"]["hit_all"] == 1.0 and hs["by_model"]["cont"]["hit_all"] == 0.0 and hs["all"]["n_calls"] == 12)
    print(f"\nintraday cont: {ok} passed")


if __name__ == "__main__":
    main()
