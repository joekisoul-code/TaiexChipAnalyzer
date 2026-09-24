"""個股回落模型 + 回落判斷 (2026-09-23)：跨股票共用 (pooled) 的「路徑最低點分位」與「回落止跌機率」模型，並輸出前端可離線套用的查表。

- 樣本：realtime.LARGE_CAPS (40 檔) + chips.WATCHLIST + 加權指數，2018~ 日 OHLCV (FinMind)；特徵只用價量 (前端對任何股票都能算) + 大盤當日/乖離。
- 目標：yLow_k = 未來 k 日路徑最低 / sigma (k=1,2,3)；hold3 = 未來 3 日最低不破今日低點 (−0.3%) = 「今日低點就是回落低點」。
- 模型：LightGBM quantile (0.20 買點 / 0.10 停損) 與 LGB 回歸 0/1 (止跌機率)，2021~ 逐年走動式 (pooled)；
  基準 = 訓練年份合併分位 (常數乘數) 與止跌基準率。
- 查表 (前端用)：dd_hi20 × 連漲跌 × K 棒位置 (clv) × 量能 4 維 36 格 → q20/q10 乘數、止跌率、n；也走動式驗證 (前幾年建表、當年套用)。
- 支撐止跌統計 (pooled)：5 日線/月線/季線/昨低/20 日低 的回測後止跌率。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
KS = (1, 2, 3)
Q_BUY, Q_STOP = 0.20, 0.10
START = "2018-01-01"
FIRST_TEST_YEAR = 2021
HOLD_TOL = 0.003
FEATS = ["sigma", "bias5", "bias20", "bias60", "dd_hi20", "lo20_dist", "days_since_hi20", "clv", "lower_wick", "upper_wick", "range_pct", "ret1", "ret5", "streak",
         "vol_ratio", "vola_ratio", "m_ret1", "m_bias20", "dow"]
NAMES = {"sigma": "波動 σ", "bias5": "5 日線乖離%", "bias20": "月線乖離%", "bias60": "季線乖離%", "dd_hi20": "距 20 日高點%", "lo20_dist": "距 20 日低點%", "days_since_hi20": "高點後天數",
         "clv": "收盤在當日振幅位置", "lower_wick": "下影線%", "upper_wick": "上影線%", "range_pct": "振幅%", "ret1": "今日漲跌%", "ret5": "5 日漲跌%", "streak": "連漲(跌)天數",
         "vol_ratio": "量能/20 日均", "vola_ratio": "波動/60 日均", "m_ret1": "大盤今日%", "m_bias20": "大盤月線乖離%", "dow": "星期"}
LGB_Q = dict(M.PARAMS, n_estimators=150, min_child_samples=200)
SEEDS = (1, 2, 3)
BUCKETS = {"dd_hi20": [-6.0, -2.0], "streak": [-2.5, -0.5], "clv": [0.35], "vol_ratio": [0.8]}   # 邊界 → 格 0..len
SUPPORTS = {"ma5": "5 日線", "ma20": "月線", "ma60": "季線", "prev_low": "昨日低點", "lo20": "20 日低點"}
NEAR, HOLD_S = 0.003, 0.005


def features_from_ohlc(df: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: date/open/high/low/close/volume (單一標的、日期升冪)。前端 JS 有同款實作 (prediction.js pullbackFeatures)。"""
    from . import range_levels as RL
    d = df.copy().reset_index(drop=True)
    for c_ in ("open", "high", "low", "close", "volume"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    d["sigma"] = RL.sigma_series(d).values
    d["ma5"], d["ma20"], d["ma60"] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    d["bias5"], d["bias20"], d["bias60"] = (c / d["ma5"] - 1) * 100, (c / d["ma20"] - 1) * 100, (c / d["ma60"] - 1) * 100
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    d["hi20"], d["lo20"] = hi20, lo20
    d["dd_hi20"], d["lo20_dist"] = (c / hi20 - 1) * 100, (c / lo20 - 1) * 100
    # 高點後天數：20 日窗內最高 high 距今幾天
    d["days_since_hi20"] = h.rolling(20).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    rng = (h - l).replace(0, np.nan)
    d["clv"] = ((c - l) / rng).clip(0, 1).fillna(0.5)
    d["lower_wick"] = (np.minimum(o, c) - l) / c * 100
    d["upper_wick"] = (h - np.maximum(o, c)) / c * 100
    d["range_pct"] = (h - l) / c * 100
    d["ret1"] = c.pct_change() * 100; d["ret5"] = c.pct_change(5) * 100
    sgn = np.sign(d["ret1"].fillna(0))
    st = np.zeros(len(d))
    for i in range(1, len(d)):
        st[i] = st[i - 1] + sgn[i] if sgn[i] != 0 and (st[i - 1] == 0 or np.sign(st[i - 1]) == sgn[i]) else sgn[i]
    d["streak"] = st
    d["vol_ratio"] = v / v.rolling(20).mean()
    d["vola_ratio"] = d["sigma"] / d["sigma"].rolling(60).mean()
    d["prev_low"] = l
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
    if mkt is not None:
        mm = mkt[["date", "m_ret1", "m_bias20"]]
        d = d.merge(mm, on="date", how="left")
    else:
        d["m_ret1"] = np.nan; d["m_bias20"] = np.nan
    for k in KS:
        pl = pd.concat([l.shift(-j) for j in range(1, k + 1)], axis=1).min(axis=1, skipna=False)
        d[f"pathLow{k}"] = (pl / c - 1) * 100
        d[f"yLow{k}"] = d[f"pathLow{k}"] / d["sigma"]
    lo3 = pd.concat([l.shift(-j) for j in (1, 2, 3)], axis=1).min(axis=1, skipna=False)
    d["hold3"] = (lo3 >= l * (1 - HOLD_TOL)).astype(float).where(lo3.notna())
    d["c3"] = c.shift(-3)
    return d


def _market_env() -> pd.DataFrame:
    from ..analysis import backtest
    m = backtest.load_long("2010-01-01")[["date", "open", "high", "low", "close", "volume"]].copy()
    m["date"] = m["date"].astype(str)
    c = pd.to_numeric(m["close"], errors="coerce")
    return pd.DataFrame({"date": m["date"], "m_ret1": c.pct_change() * 100, "m_bias20": (c / c.rolling(20).mean() - 1) * 100}), m


def build_panel(universe: list[str] | None = None) -> pd.DataFrame:
    from ..sources import finmind
    from ..analysis import chips
    from ..realtime import LARGE_CAPS
    env, midx = _market_env()
    uni = universe or sorted(set(LARGE_CAPS) | set(chips.WATCHLIST))
    frames = []
    for sid in uni:
        try:
            p = finmind.stock_price(sid, START)
            if p.empty or len(p) < 120:
                continue
            f = features_from_ohlc(p[["date", "open", "high", "low", "close", "volume"]], env); f["stock_id"] = sid
            frames.append(f)
        except Exception as e:  # noqa: BLE001
            log.warning("stock_pullback %s: %s", sid, e)
    mi = midx[midx["date"] >= START]
    f = features_from_ohlc(mi, env); f["stock_id"] = "TAIEX"; frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    for k in KS:   # 除權息/異常
        panel.loc[panel[f"pathLow{k}"] < -30, [f"pathLow{k}", f"yLow{k}"]] = np.nan
    panel["year"] = panel["date"].str[:4].astype(int)
    return panel.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _qfit(X, y, alpha):
    import lightgbm as lgb
    return [lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=s, **LGB_Q).fit(X, y) for s in SEEDS]


def _rfit(X, y):
    import lightgbm as lgb
    return [lgb.LGBMRegressor(random_state=s, **LGB_Q).fit(X, y) for s in SEEDS]


def _pred(ms, X):
    return np.mean([m_.predict(X) for m_ in ms], axis=0)


def _pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def cell_of(row) -> str:
    idx = []
    for f, edges in BUCKETS.items():
        v = row[f] if not isinstance(row, pd.DataFrame) else row[f].iloc[0]
        v = float(v) if v == v else (edges[0] + 0.0)   # NaN → 中間格
        idx.append(str(int(np.searchsorted(edges, v, side="right"))))
    return "|".join(idx)


def _table(d: pd.DataFrame) -> dict:
    cells = d.apply(cell_of, axis=1) if len(d) else pd.Series(dtype=str)
    out = {"buckets": BUCKETS, "cells": {}, "all": {}}
    def stats(g):
        r = {"n": int(len(g))}
        for k in KS:
            yy = g[f"yLow{k}"].dropna()
            r[f"q20_{k}"] = round(float(yy.quantile(Q_BUY)), 3) if len(yy) >= 30 else None
            r[f"q10_{k}"] = round(float(yy.quantile(Q_STOP)), 3) if len(yy) >= 30 else None
        hh = g["hold3"].dropna(); r["hold3"] = round(float(hh.mean()), 3) if len(hh) >= 30 else None
        return r
    out["all"] = stats(d)
    for c_, g in d.groupby(cells):
        out["cells"][c_] = stats(g)
    return out


def _apply_table(tbl: dict, d: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    cells = d.apply(cell_of, axis=1)
    q20 = np.array([((tbl["cells"].get(c_) or {}).get(f"q20_{k}") or tbl["all"][f"q20_{k}"]) for c_ in cells], float)
    hold = np.array([((tbl["cells"].get(c_) or {}).get("hold3") or tbl["all"]["hold3"]) for c_ in cells], float)
    return q20, hold


def _support_stats(panel: pd.DataFrame) -> dict:
    out = {}
    c = panel["close"]; lo3 = panel.groupby("stock_id")["low"].transform(lambda s: pd.concat([s.shift(-j) for j in (1, 2, 3)], axis=1).min(axis=1, skipna=False))
    c3 = panel["c3"]; r3 = (c3 / c - 1) * 100
    for key, name in SUPPORTS.items():
        s = panel[key]
        cand = (s < c) & (s > c * 0.97) & lo3.notna() & s.notna()
        tested = cand & (lo3 <= s * (1 + NEAR)); held = tested & (lo3 >= s * (1 - HOLD_S)) & (c3 > s)
        n_t = int(tested.sum())
        out[key] = {"name": name, "n_tested": n_t, "hold_rate": round(float(held[tested].mean()), 3) if n_t else None,
                    "bounce3_after_hold": round(float(r3[held].mean()), 2) if held.any() else None, "break3_after_fail": round(float(r3[tested & ~held].mean()), 2) if (tested & ~held).any() else None}
        for st_, msk in (("bull", c >= panel["ma20"]), ("bear", c < panel["ma20"])):
            t2 = tested & msk
            out[key][f"hold_{st_}"] = round(float(held[t2].mean()), 3) if t2.sum() >= 50 else None
    return out


def train(write: bool = True, verbose: bool = True) -> dict:
    panel = build_panel()
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "features": FEATS, "n_rows": int(len(panel)), "n_stocks": int(panel["stock_id"].nunique()),
           "k": {}, "hold3": {}, "table_oos": {}, "supports": _support_stats(panel)}
    yrs = panel["year"]
    # --- 分位模型 (走動式) ---
    for k in KS:
        d = panel.dropna(subset=[f"yLow{k}", "sigma"]); d = d[d["sigma"] > 0]
        X, y, yy_ = d[FEATS], d[f"yLow{k}"], d["year"]
        pm = pd.Series(np.nan, index=d.index); pc = pd.Series(np.nan, index=d.index); pt = pd.Series(np.nan, index=d.index)
        for yv in range(FIRST_TEST_YEAR, int(yy_.max()) + 1):
            tr, te = yy_ < yv, yy_ == yv
            if tr.sum() < 3000 or not te.any():
                continue
            pm[te] = _pred(_qfit(X[tr], y[tr], Q_BUY), X[te])
            pc[te] = float(y[tr].quantile(Q_BUY))
            pt[te], _ = _apply_table(_table(d[tr]), d[te], k)
        ev = pm.notna()
        yv_, sg = y[ev].values, d.loc[ev, "sigma"].values
        res = {"n_oos": int(ev.sum())}
        for name, q in (("model", pm[ev].values), ("const", pc[ev].values), ("table", pt[ev].values)):
            touch = yv_ <= q
            byy = pd.DataFrame({"t": touch, "y": yy_[ev].values}).groupby("y")["t"].mean()
            res[name] = {"pinball20": round(_pinball(yv_ * sg, q * sg, Q_BUY), 4), "touch20": round(float(touch.mean()), 3), "touch_yr_min": round(float(byy.min()), 3), "touch_yr_max": round(float(byy.max()), 3),
                         "mae_low": round(float(np.mean(np.abs(yv_ - q) * sg)), 3)}
        res["improve_model"] = round(1 - res["model"]["pinball20"] / res["const"]["pinball20"], 3)
        if k == 1:   # 逐股 OOS 改善 (驗證：是否所有股票都受益)
            bs = []
            for sid, gi in d[ev].assign(pm=pm[ev].values, pc=pc[ev].values).groupby("stock_id"):
                if len(gi) < 150:
                    continue
                yy2, sg2 = gi[f"yLow{k}"].values, gi["sigma"].values
                bs.append(round(1 - _pinball(yy2 * sg2, gi["pm"].values * sg2, Q_BUY) / _pinball(yy2 * sg2, gi["pc"].values * sg2, Q_BUY), 3))
            res["by_stock"] = {"n_stocks": len(bs), "improve_median": round(float(np.median(bs)), 3) if bs else None, "improve_min": round(float(min(bs)), 3) if bs else None, "n_negative": int(sum(1 for b_ in bs if b_ < 0))}
        res["improve_table"] = round(1 - res["table"]["pinball20"] / res["const"]["pinball20"], 3)
        res["use_model"] = bool(res["improve_model"] >= 0.02 and 0.15 <= res["model"]["touch20"] <= 0.26)
        out["k"][str(k)] = res
        if write:
            M.save(f"stock_pullback_k{k}", {"buy": _qfit(X, y, Q_BUY), "stop": _qfit(X, y, Q_STOP), "features": FEATS, "trained_at": out["trained_at"]})
        if verbose:
            print(f"  stock_pullback k{k}: pinball 模型 {res['model']['pinball20']} / 查表 {res['table']['pinball20']} / 常數 {res['const']['pinball20']} (改善 模型 {res['improve_model']}、查表 {res['improve_table']})；觸及率 模型 {res['model']['touch20']} 查表 {res['table']['touch20']} 常數 {res['const']['touch20']}；誤差 {res['model']['mae_low']} vs {res['const']['mae_low']}%")
    # --- 止跌機率 (走動式) ---
    d = panel.dropna(subset=["hold3"]); X, y, yy_ = d[FEATS], d["hold3"], d["year"]
    pm = pd.Series(np.nan, index=d.index); pt = pd.Series(np.nan, index=d.index); pb = pd.Series(np.nan, index=d.index)
    for yv in range(FIRST_TEST_YEAR, int(yy_.max()) + 1):
        tr, te = yy_ < yv, yy_ == yv
        if tr.sum() < 3000 or not te.any():
            continue
        pm[te] = np.clip(_pred(_rfit(X[tr], y[tr]), X[te]), 0, 1); pb[te] = float(y[tr].mean())
        _, pt[te] = _apply_table(_table(d[tr]), d[te], 1)
    ev = pm.notna(); yv_ = y[ev].values
    def tiers(p):
        hi, lo = np.quantile(p, 0.7), np.quantile(p, 0.3)
        return {"top30_hold": round(float(yv_[p >= hi].mean()), 3), "bot30_hold": round(float(yv_[p <= lo].mean()), 3), "brier": round(float(np.mean((p - yv_) ** 2)), 4)}
    out["hold3"] = {"n_oos": int(ev.sum()), "base": round(float(yv_.mean()), 3), "model": tiers(pm[ev].values), "table": tiers(pt[ev].values), "const_brier": round(float(np.mean((pb[ev].values - yv_) ** 2)), 4)}
    cal = pd.DataFrame({"p": pm[ev].values, "y": yv_, "year": yy_[ev].values}); cal["bin"] = pd.cut(cal["p"], [0, .3, .4, .5, .6, .7, 1.0])
    out["hold3"]["calibration"] = [{"bin": str(b_), "rate": round(float(g_["y"].mean()), 3), "n": int(len(g_))} for b_, g_ in cal.groupby("bin", observed=True)]
    out["hold3"]["by_year"] = {int(yv): {"top30": round(float(g_.loc[g_["p"] >= g_["p"].quantile(.7), "y"].mean()), 3), "bot30": round(float(g_.loc[g_["p"] <= g_["p"].quantile(.3), "y"].mean()), 3)} for yv, g_ in cal.groupby("year")}
    bs2 = []
    for sid, g_ in d[ev].assign(p=pm[ev].values).groupby("stock_id"):
        if len(g_) >= 150:
            bs2.append(round(float(g_.loc[g_["p"] >= g_["p"].quantile(.7), "hold3"].mean() - g_.loc[g_["p"] <= g_["p"].quantile(.3), "hold3"].mean()), 3))
    out["hold3"]["by_stock"] = {"n_stocks": len(bs2), "gap_median": round(float(np.median(bs2)), 3) if bs2 else None, "gap_min": round(float(min(bs2)), 3) if bs2 else None, "n_negative": int(sum(1 for b_ in bs2 if b_ < 0))}
    fb = _rfit(X, y)
    imp = np.mean([mm.feature_importances_ for mm in fb], axis=0); imp = imp / imp.sum()
    out["hold3"]["importance"] = sorted(({"f": f, "name": NAMES.get(f, f), "w": round(float(w), 3)} for f, w in zip(FEATS, imp) if w > 0.03), key=lambda x: -x["w"])
    if write:
        M.save("stock_pullback_hold3", {"models": fb, "features": FEATS, "trained_at": out["trained_at"]})
    if verbose:
        h3 = out["hold3"]; print(f"  止跌機率 hold3: 基準 {h3['base']}；模型 高 30% {h3['model']['top30_hold']} / 低 30% {h3['model']['bot30_hold']} (brier {h3['model']['brier']} vs 常數 {h3['const_brier']})；查表 {h3['table']['top30_hold']} / {h3['table']['bot30_hold']}；主要特徵 " + "、".join(f"{x['name']} {x['w']}" for x in h3["importance"][:5]))
        for key, s in out["supports"].items():
            print(f"  個股支撐 {s['name']}: 回測 {s['n_tested']} 止跌率 {s['hold_rate']} (多頭 {s.get('hold_bull')} / 空頭 {s.get('hold_bear')})，止跌後 3 日 {s['bounce3_after_hold']}%，跌破後 {s['break3_after_fail']}%")
    out["table"] = _table(panel.dropna(subset=["yLow1"]))   # 全樣本查表 (前端離線用)
    if write:
        M.save_json("stock_pullback", out)
    return out


def build(stock_id: str, price: pd.DataFrame | None = None, base_px: float | None = None) -> dict | None:
    """單一標的今日：模型買點/停損 (k=1..3)、止跌機率、回落脈絡、支撐 (pooled 止跌率)。price 需含 date/open/high/low/close/volume。"""
    st = M.load_json("stock_pullback")
    if not st:
        return None
    if price is None:
        from ..sources import finmind
        price = finmind.stock_price(stock_id, "2024-01-01")
    if price is None or price.empty or len(price) < 70:
        return None
    env, _ = _market_env()
    d = features_from_ohlc(price[["date", "open", "high", "low", "close", "volume"]], env)
    row = d.iloc[[-1]]
    close = float(row["close"].iloc[0]); px0 = float(base_px or close); sg = float(row["sigma"].iloc[0])
    if not (sg > 0):
        return None
    out = {"sid": stock_id, "date": str(row["date"].iloc[0]), "sigma": round(sg, 3), "k": {}, "stock": True,
           "context": {f: (round(float(row[f].iloc[0]), 2) if pd.notna(row[f].iloc[0]) else None) for f in ("dd_hi20", "days_since_hi20", "streak", "clv", "lower_wick", "vol_ratio", "bias20", "lo20_dist")}, "cell": cell_of(row)}
    for k in KS:
        b = M.load(f"stock_pullback_k{k}"); r = (st.get("k") or {}).get(str(k)) or {}
        if not b:
            continue
        q20 = float(_pred(b["buy"], row[b["features"]])[0]) * sg; q10 = min(float(_pred(b["stop"], row[b["features"]])[0]) * sg, q20)
        out["k"][str(k)] = {"buy_model": round(px0 * (1 + q20 / 100), 2), "stop_model": round(px0 * (1 + q10 / 100), 2), "low20_pct": round(q20, 2), "low10_pct": round(q10, 2),
                            "use_model": bool(r.get("use_model")), "oos": {"improve_pinball": r.get("improve_model"), "model": r.get("model"), "sigma": r.get("const"), "by_stock": r.get("by_stock")}}
    hb = M.load("stock_pullback_hold3")
    if hb:
        p = float(np.clip(_pred(hb["models"], row[hb["features"]])[0], 0, 1))
        h3 = st.get("hold3") or {}
        cal_ = next((c for c in (h3.get("calibration") or []) if _in_bin(c["bin"], p)), None)
        out["hold3"] = {"p": round(p, 3), "base": h3.get("base"), "top30": (h3.get("model") or {}).get("top30_hold"), "bot30": (h3.get("model") or {}).get("bot30_hold"),
                        "cal_rate": cal_["rate"] if cal_ else None, "cal_n": cal_["n"] if cal_ else None, "by_year": h3.get("by_year"), "by_stock": h3.get("by_stock"),
                        "label": "回落可能已到低點" if p >= (h3.get("base") or 0.5) + 0.08 else "回落可能未完" if p <= (h3.get("base") or 0.5) - 0.08 else "不明顯"}
    out["supports"] = []
    bull = bool(close >= float(row["ma20"].iloc[0])) if pd.notna(row["ma20"].iloc[0]) else None
    for key, name in SUPPORTS.items():
        s = float(row[key].iloc[0]) if pd.notna(row[key].iloc[0]) else None
        if s is None or not (s < px0 and s > px0 * 0.93):
            continue
        ss = (st.get("supports") or {}).get(key) or {}
        out["supports"].append({"key": key, "name": name, "level": round(s, 2), "dist_pct": round((s / px0 - 1) * 100, 2), "hold_rate": ss.get("hold_rate"),
                                "hold_regime": (ss.get("hold_bull") if bull else ss.get("hold_bear")) if bull is not None else None, "regime": "多頭" if bull else "空頭", "bounce3": ss.get("bounce3_after_hold"), "break3": ss.get("break3_after_fail"), "n": ss.get("n_tested")})
    out["supports"].sort(key=lambda x: -x["level"])
    return out


def _in_bin(b: str, p: float) -> bool:
    try:
        lo, hi = b.strip("()[]").split(",")
        return float(lo) < p <= float(hi)
    except Exception:  # noqa: BLE001
        return False


def client_table() -> dict | None:
    """給前端 (watchlist.json) 的離線查表：sigma 乘數 + 止跌率 + 支撐統計 + 驗證摘要。"""
    st = M.load_json("stock_pullback")
    if not st:
        return None
    return {"table": st.get("table"), "supports": st.get("supports"), "hold3_base": (st.get("hold3") or {}).get("base"), "oos": {"k": {k: {"table": v.get("table"), "const": v.get("const"), "improve_table": v.get("improve_table")} for k, v in (st.get("k") or {}).items()}, "hold3_table": (st.get("hold3") or {}).get("table")}, "trained_at": st.get("trained_at")}
