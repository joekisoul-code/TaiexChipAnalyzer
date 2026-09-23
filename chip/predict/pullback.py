"""回落進場點 (2026-09-23)：用歷史資料找「未來 k 日路徑最低點」落在哪裡，提高拉回買點 (buy_at) 的精準度。

三個層次：
1. 條件式分位模型：LightGBM quantile (α=0.20 買點 / 0.10 停損) 直接預測 pathLow_k / sigma (以波動正規化)，特徵 = 波動、乖離、距 20 日低/高、
   今日 K 棒位置 (clv / 下影線)、連漲跌、量能、外資、亞股同日、VIX、結算週期、星期… 走動式 (2014~ 逐年) 與現行「sigma × 固定乘數」比：
   pinball loss、觸及率校準 (目標 20%)、觸及後的超跌幅 (最低點離買點多遠)。只有模型 pinball 至少好 3% 才取代 buy_at，否則只提供參考。
2. 支撐止跌統計：價格回落到 5 日線 / 月線 / 季線 / 昨日低點 / 20 日低點 附近時，歷史上多常在那裡止跌 (hold) 並反彈；
   → 今日現價下方 3% 內的支撐清單，附「止跌率」與「止跌後 3 日平均反彈」。
3. 最低點時段：小時 K (Yahoo ^TWII 60m 730d) 統計當日最低價落在哪個時段，分「開低 / 開高」→ 拉回買點的時間窗。
4. 含夜盤變體 (2026-09-23 強化)：特徵 + 前晚夜盤台指期，訓練集含 2010~ (夜盤缺值由 LGB 處理)，2021~ 逐年走動式 vs range_levels 夜盤公式：
   pinball k1 0.304 vs 0.365 (改善 16%)、k2 0.421 vs 0.459 (8%)、k3 0.514 vs 0.545 (6%)，觸及率 20~22%。夜盤收後 buy_at/stop 改用此模型。
5. 盤中低點判斷 (小時 K 查表)：時間點 × 開盤跳空 (開低/平盤/開高) × 開盤後已回檔 (≥0.5σ / <0.5σ) → 「今日低點已出現」機率、最終低點 (相對現價) 兩成/五成分位。
   10:00 整體 58%、11:00 70%、12:00 77%、13:00 86%；開低且已回檔 ≥0.5σ 的日子在 10:00 只有 48%、最終低點 q20 −1.3%。
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
FIRST_TEST_YEAR = 2014
FEATS = ["sigma_range", "vola_ratio", "bias5", "bias20", "bias60", "lo20_dist", "hi20_dist", "clv", "lower_wick", "upper_wick", "range_pct", "ret1", "ret5", "streak",
         "vol_ratio", "amount_5d_ratio", "foreign_z5", "hsi_r0", "kospi_r0", "g_vix_level", "days_to_settle", "dow", "ma20_slope", "gap_open", "composite_smooth"]
SUPPORTS = {"ma5": "5 日線", "ma20": "月線", "ma60": "季線", "prev_low": "昨日低點", "lo20": "20 日低點"}
NEAR, HOLD_TOL = 0.003, 0.005   # 回落到支撐 ±0.3% 算「測試」；最低點不破支撐 0.5% 且 k 日後收在支撐上算「止跌」
LGB_Q = dict(M.PARAMS, n_estimators=150, min_child_samples=120)


def _frame(scored: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from . import range_levels as RL, short_term as ST
    m = ST.build_matrix(scored, None).copy()
    m["sigma_range"] = RL.sigma_series(m).values
    if "dow" not in m:
        m["dow"] = pd.to_datetime(m["date"]).dt.dayofweek
    tg = RL.path_targets(m)
    for k in KS:
        m[f"pathLow{k}"] = tg[f"pathLow{k}"].values
        m[f"yLow{k}"] = m[f"pathLow{k}"] / m["sigma_range"]
    c = m["close"].astype(float)
    m["ma5"], m["ma20"], m["ma60"] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    m["prev_low"] = m["low"].astype(float)
    m["lo20"] = m["low"].astype(float).rolling(20).min()
    feats = [f for f in FEATS if f in m.columns]
    return m, feats


def _pinball(y: np.ndarray, q: np.ndarray, a: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def _qfit(X, y, alpha):
    import lightgbm as lgb
    return [lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=s, **LGB_Q).fit(X, y) for s in M.SEEDS]


def _qpred(ms, X):
    return np.mean([m_.predict(X) for m_ in ms], axis=0)


def _support_stats(m: pd.DataFrame) -> dict:
    """支撐止跌率：t 日收盤在支撐之上且支撐在收盤下方 3% 內 → 未來 3 日內最低價是否回測 (±NEAR) → 回測日之後是否止跌。"""
    c = m["close"].astype(float); lo3 = pd.concat([m["low"].astype(float).shift(-j) for j in (1, 2, 3)], axis=1).min(axis=1, skipna=False)
    c3 = c.shift(-3); r3 = (c3 / c - 1) * 100
    out = {}
    for key, name in SUPPORTS.items():
        s = m[key].astype(float)
        cand = (s < c) & (s > c * 0.97) & lo3.notna()
        tested = cand & (lo3 <= s * (1 + NEAR))
        held = tested & (lo3 >= s * (1 - HOLD_TOL)) & (c3 > s)
        n_t = int(tested.sum())
        yr = m.loc[tested, "date"].str[:4]
        by_year = {}
        if n_t:
            for y, g in held[tested].groupby(yr):
                if len(g) >= 8:
                    by_year[y] = round(float(g.mean()), 2)
        out[key] = {"name": name, "n_candidate": int(cand.sum()), "n_tested": n_t, "test_rate": round(n_t / max(1, int(cand.sum())), 3),
                    "hold_rate": round(float(held[tested].mean()), 3) if n_t else None, "hold_yr_min": min(by_year.values()) if by_year else None,
                    "bounce3_after_hold": round(float(r3[held].mean()), 2) if held.any() else None, "break3_after_fail": round(float(r3[tested & ~held].mean()), 2) if (tested & ~held).any() else None}
        # 依多空狀態 (收盤在月線上/下)
        for st_, msk in (("bull", c >= m["ma20"]), ("bear", c < m["ma20"])):
            t2 = tested & msk
            out[key][f"hold_{st_}"] = round(float(held[t2].mean()), 3) if t2.sum() >= 30 else None
    return out


HL_MARKS = ("10:00", "11:00", "12:00", "13:00")


def _hour_low_table(m: pd.DataFrame) -> dict:
    """盤中低點判斷查表：時間點 × 跳空 (dn/flat/up) × 開盤後已回檔 (deep = 累積低點距開盤 ≤ -0.5σ) → 低點已出現機率、最終低點 (相對該時點價) q20/q50。"""
    try:
        from ..sources import yahoo
        marks = yahoo.daily_marks("730d")
    except Exception as e:  # noqa: BLE001
        log.warning("hour_low_table: %s", e)
        return {}
    sig = dict(zip(m["date"], m["sigma_range"]))
    rows = []
    for d_, rec in sorted(marks.items()):
        if not rec.get("prev_close") or "13:30" not in rec or "13:30" not in (rec.get("lo") or {}) or d_ not in sig:
            continue
        fin_low, op, pc, s = rec["lo"]["13:30"], rec["open"], rec["prev_close"], sig[d_]
        if not (s and s > 0):
            continue
        for mk in HL_MARKS:
            if mk not in rec or mk not in rec["lo"]:
                continue
            px, lo = rec[mk], rec["lo"][mk]
            gap = (op / pc - 1) * 100
            rows.append({"mark": mk, "g": "dn" if gap < -0.15 else "up" if gap > 0.15 else "flat", "deep": ((lo / op - 1) * 100 / s) <= -0.5, "done": lo <= fin_low * 1.0003, "fin": (fin_low / px - 1) * 100})
    if not rows:
        return {}
    h = pd.DataFrame(rows)
    out = {"n_days": int(len(h) / max(1, h["mark"].nunique())), "cells": {}, "marks": {}}
    for mk, g in h.groupby("mark"):
        out["marks"][mk] = {"n": int(len(g)), "p_done": round(float(g["done"].mean()), 3), "fin_q20": round(float(g["fin"].quantile(0.2)), 2), "fin_q50": round(float(g["fin"].median()), 2)}
        for (gg, dp), g3 in g.groupby(["g", "deep"]):
            if len(g3) >= 20:
                out["cells"][f"{mk}|{gg}|{'deep' if dp else 'shallow'}"] = {"n": int(len(g3)), "p_done": round(float(g3["done"].mean()), 3), "fin_q20": round(float(g3["fin"].quantile(0.2)), 2), "fin_q50": round(float(g3["fin"].median()), 2)}
    return out


def _hour_of_low() -> dict:
    """當日最低價落在哪個小時 K (Yahoo 60m，約 2 年)；分開低 / 開高。"""
    try:
        from ..sources import yahoo
        df = yahoo.taiex_hourly("730d")
    except Exception as e:  # noqa: BLE001
        log.warning("hour_of_low: %s", e)
        return {}
    if df is None or df.empty:
        return {}
    res = {"all": {}, "gap_down": {}, "gap_up": {}, "n": 0}
    prev_close = None
    cnt = {"all": {}, "gap_down": {}, "gap_up": {}}
    n = 0
    for date, g in df.groupby("date", sort=True):
        g = g.sort_values("time")
        if "09:00" not in set(g["time"]):
            prev_close = float(g["close"].iloc[-1]); continue
        i = g["low"].astype(float).idxmin(); t = str(g.loc[i, "time"])
        keys = ["all"]
        if prev_close:
            keys.append("gap_down" if float(g["open"].iloc[0]) < prev_close else "gap_up")
        for k in keys:
            cnt[k][t] = cnt[k].get(t, 0) + 1
        prev_close = float(g["close"].iloc[-1]); n += 1
    for k, c in cnt.items():
        tot = sum(c.values()) or 1
        res[k] = {t: round(v / tot, 3) for t, v in sorted(c.items())}
    res["n"] = n
    return res


def train(scored: pd.DataFrame, write: bool = True, verbose: bool = True) -> dict:
    from . import range_levels as RL
    m, feats = _frame(scored)
    m["year"] = m["date"].str[:4].astype(int)
    mult = RL.load_multipliers() or {}
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "features": feats, "k": {}, "supports": _support_stats(m), "hour_of_low": _hour_of_low()}
    for k in KS:
        d = m.dropna(subset=[f"yLow{k}", "sigma_range"]).reset_index(drop=True)
        X, y, yrs = d[feats], d[f"yLow{k}"], d["year"]
        pb = pd.Series(np.nan, index=d.index); ps = pd.Series(np.nan, index=d.index)
        for yv in sorted(yrs.unique()):
            if yv < FIRST_TEST_YEAR:
                continue
            tr = (yrs < yv)
            tr &= d.index < d.index[tr][-1] - k + 1 if tr.any() else tr
            if tr.sum() < 500:
                continue
            te = yrs == yv
            pb[te] = _qpred(_qfit(X[tr], y[tr], Q_BUY), X[te]); ps[te] = _qpred(_qfit(X[tr], y[tr], Q_STOP), X[te])
        ev = pb.notna()
        yy, sg = y[ev].values, d.loc[ev, "sigma_range"].values
        base_m = (mult.get("base") or {}).get(str(k)) or {}
        b20 = np.full(len(yy), float(base_m.get("low20", np.nan))); b10 = np.full(len(yy), float(base_m.get("low10", np.nan)))
        res = {"n_oos": int(ev.sum())}
        for name, q20, q10 in (("model", pb[ev].values, ps[ev].values), ("sigma", b20, b10)):
            if np.isnan(q20).all():
                continue
            touch20 = (yy <= q20); touch10 = (yy <= q10)
            over = (q20 - yy)[touch20] * sg[touch20]   # 觸及後再跌多少 (%)
            gap = (yy - q20)[~touch20] * sg[~touch20]  # 未觸及時最低點離買點多遠 (%)
            byy = pd.DataFrame({"t": touch20, "y": yrs[ev].values}).groupby("y")["t"].mean()
            res[name] = {"pinball20": round(_pinball(yy * sg, q20 * sg, Q_BUY), 4), "pinball10": round(_pinball(yy * sg, q10 * sg, Q_STOP), 4),
                         "touch20": round(float(touch20.mean()), 3), "touch10": round(float(touch10.mean()), 3), "touch20_yr_min": round(float(byy.min()), 3), "touch20_yr_max": round(float(byy.max()), 3),
                         "overshoot_mean": round(float(over.mean()), 3) if len(over) else None, "gap_mean": round(float(gap.mean()), 3) if len(gap) else None,
                         "mae_low": round(float(np.mean(np.abs(yy - q20) * sg)), 3)}
        if "model" in res and "sigma" in res:
            res["improve_pinball"] = round(1 - res["model"]["pinball20"] / res["sigma"]["pinball20"], 3)
            res["use_model"] = bool(res["improve_pinball"] >= 0.03 and 0.15 <= res["model"]["touch20"] <= 0.26)
        fb = _qfit(X, y, Q_BUY); fs = _qfit(X, y, Q_STOP)
        imp = np.mean([mm.feature_importances_ for mm in fb], axis=0); imp = imp / imp.sum()
        res["importance"] = sorted(({"f": f, "w": round(float(w), 3)} for f, w in zip(feats, imp) if w > 0.02), key=lambda x: -x["w"])
        out["k"][str(k)] = res
        M.save(f"pullback_k{k}", {"buy": fb, "stop": fs, "features": feats, "trained_at": out["trained_at"]}) if write else None
        if verbose:
            mo, sg_ = res.get("model", {}), res.get("sigma", {})
            print(f"  pullback k{k}: pinball20 模型 {mo.get('pinball20')} vs sigma {sg_.get('pinball20')} (改善 {res.get('improve_pinball')})；觸及率 模型 {mo.get('touch20')} (年 {mo.get('touch20_yr_min')}~{mo.get('touch20_yr_max')}) vs sigma {sg_.get('touch20')}；最低點誤差 {mo.get('mae_low')} vs {sg_.get('mae_low')}%；use_model={res.get('use_model')}")
    # 含夜盤變體：feats + night_chg_pct (訓練含無夜盤年份，缺值交給 LGB)，2021~ 逐年走動式 vs range_levels 夜盤公式
    try:
        from . import short_term as ST
        nh = ST._night_hist()
        mn = m.drop(columns=[c for c in m.columns if c == "night_chg_pct"]).merge(nh[["date", "night_chg_pct"]], on="date", how="left")
        mn = mn[(mn["night_chg_pct"].isna()) | (mn["night_chg_pct"].abs() <= 8)]
        featsN = feats + ["night_chg_pct"]
        out["night"] = {"features": featsN, "k": {}}
        for k in KS:
            d = mn.dropna(subset=[f"yLow{k}", "sigma_range"]); d = d[d["sigma_range"] > 0].reset_index(drop=True)
            te_all = d["night_chg_pct"].notna() & (d["year"] >= 2021)
            pb = pd.Series(np.nan, index=d.index); ps = pd.Series(np.nan, index=d.index)
            for yv in sorted(d.loc[te_all, "year"].unique()):
                tr, te = d["year"] < yv, te_all & (d["year"] == yv)
                pb[te] = _qpred(_qfit(d.loc[tr, featsN], d.loc[tr, f"yLow{k}"], Q_BUY), d.loc[te, featsN]); ps[te] = _qpred(_qfit(d.loc[tr, featsN], d.loc[tr, f"yLow{k}"], Q_STOP), d.loc[te, featsN])
            ev = pb.notna(); yy, sg, nv = d.loc[ev, f"yLow{k}"].values, d.loc[ev, "sigma_range"].values, d.loc[ev, "night_chg_pct"].values
            nm = (mult.get("night") or {}).get(str(k)) or {}
            base = ((nm.get("beta_low", 0) * nv + nm.get("low20", np.nan) * sg) / sg) if nm else np.full(len(yy), np.nan)
            r_ = {"n_oos": int(ev.sum())}
            for name, q in (("model", pb[ev].values), ("formula", base)):
                if np.isnan(q).all():
                    continue
                touch = yy <= q; byy = pd.DataFrame({"t": touch, "y": d.loc[ev, "year"].values}).groupby("y")["t"].mean()
                r_[name] = {"pinball20": round(_pinball(yy * sg, q * sg, Q_BUY), 4), "touch20": round(float(touch.mean()), 3), "touch20_yr_min": round(float(byy.min()), 3), "touch20_yr_max": round(float(byy.max()), 3), "mae_low": round(float(np.mean(np.abs(yy - q) * sg)), 3)}
            if "model" in r_ and "formula" in r_:
                r_["improve_pinball"] = round(1 - r_["model"]["pinball20"] / r_["formula"]["pinball20"], 3)
                r_["use_model"] = bool(r_["improve_pinball"] >= 0.03 and 0.15 <= r_["model"]["touch20"] <= 0.27)
            out["night"]["k"][str(k)] = r_
            if write:
                M.save(f"pullback_k{k}_night", {"buy": _qfit(d[featsN], d[f"yLow{k}"], Q_BUY), "stop": _qfit(d[featsN], d[f"yLow{k}"], Q_STOP), "features": featsN, "trained_at": out["trained_at"]})
            if verbose:
                mo, fo = r_.get("model", {}), r_.get("formula", {})
                print(f"  pullback(夜盤) k{k}: pinball 模型 {mo.get('pinball20')} vs 夜盤公式 {fo.get('pinball20')} (改善 {r_.get('improve_pinball')})；觸及 {mo.get('touch20')} vs {fo.get('touch20')}；誤差 {mo.get('mae_low')} vs {fo.get('mae_low')}%；use_model={r_.get('use_model')}")
    except Exception as e:  # noqa: BLE001
        log.warning("pullback night variant: %s", e)
    out["hour_low_table"] = _hour_low_table(m)
    if verbose:
        for key, s in out["supports"].items():
            print(f"  支撐 {s['name']}: 回測 {s['n_tested']} 次 止跌率 {s['hold_rate']} (多頭 {s.get('hold_bull')} / 空頭 {s.get('hold_bear')}，年最低 {s['hold_yr_min']})，止跌後 3 日 {s['bounce3_after_hold']}%，跌破後 3 日 {s['break3_after_fail']}%")
        print("  最低點時段:", out["hour_of_low"].get("all"), "開低:", out["hour_of_low"].get("gap_down"), "開高:", out["hour_of_low"].get("gap_up"))
    if write:
        M.save_json("pullback", out)
    return out


def build(scored: pd.DataFrame, base_px: float | None = None, night_ret: float | None = None) -> dict | None:
    """今日：模型買點/停損 (k=1..3；night_ret 給定且夜盤模型存在 → 含夜盤變體)、下方支撐清單 (含止跌率)、最低點時段、盤中低點查表。"""
    st = M.load_json("pullback")
    if not st:
        return None
    m, feats = _frame(scored)
    row = m.iloc[[-1]].copy()
    close = float(row["close"].iloc[0]); px0 = float(base_px or close); sg = float(row["sigma_range"].iloc[0])
    use_night = night_ret is not None and np.isfinite(float(night_ret)) and bool((st.get("night") or {}).get("k"))
    out = {"date": str(row["date"].iloc[0]), "sigma": round(sg, 3), "variant": "night" if use_night else "base", "night_ret": round(float(night_ret), 2) if use_night else None,
           "k": {}, "supports": [], "hour_of_low": st.get("hour_of_low") or {}, "hour_low_table": st.get("hour_low_table") or {}}
    if use_night:
        row["night_chg_pct"] = float(np.clip(float(night_ret), -8, 8))
    for k in KS:
        if use_night:
            b = M.load(f"pullback_k{k}_night"); r = ((st.get("night") or {}).get("k") or {}).get(str(k)) or {}
            if not b:
                b = M.load(f"pullback_k{k}"); r = (st.get("k") or {}).get(str(k)) or {}
        else:
            b = M.load(f"pullback_k{k}"); r = (st.get("k") or {}).get(str(k)) or {}
        if not b:
            continue
        q20 = float(_qpred(b["buy"], row[b["features"]])[0]) * sg; q10 = float(_qpred(b["stop"], row[b["features"]])[0]) * sg
        q10 = min(q10, q20)
        out["k"][str(k)] = {"buy_model": int(round(px0 * (1 + q20 / 100))), "stop_model": int(round(px0 * (1 + q10 / 100))), "low20_pct": round(q20, 2), "low10_pct": round(q10, 2),
                            "use_model": bool(r.get("use_model")), "variant": "night" if (use_night and "night_chg_pct" in (b.get("features") or [])) else "base",
                            "oos": {kk: r.get(kk) for kk in ("improve_pinball",)} | {"model": r.get("model"), "sigma": r.get("sigma") or r.get("formula")}}
    for key, name in SUPPORTS.items():
        s = float(row[key].iloc[0]) if key in row and pd.notna(row[key].iloc[0]) else None
        if s is None or not (s < px0 and s > px0 * 0.95):
            continue
        ss = (st.get("supports") or {}).get(key) or {}
        bull = close >= float(row["ma20"].iloc[0]) if pd.notna(row["ma20"].iloc[0]) else None
        out["supports"].append({"key": key, "name": name, "level": int(round(s)), "dist_pct": round((s / px0 - 1) * 100, 2), "hold_rate": ss.get("hold_rate"),
                                "hold_regime": (ss.get("hold_bull") if bull else ss.get("hold_bear")) if bull is not None else None, "regime": "多頭" if bull else "空頭",
                                "bounce3": ss.get("bounce3_after_hold"), "break3": ss.get("break3_after_fail"), "n": ss.get("n_tested")})
    out["supports"].sort(key=lambda x: -x["level"])
    return out
