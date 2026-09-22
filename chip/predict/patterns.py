"""歷史漲跌規律學習 (2026-09-22)：從 2010~ 逐日資料「找出邏輯」，只保留跨年度穩定的規律，並把「歷史相似走勢 (analog)」當成一個可驗證的預測器。

兩個部分：
A. 規律庫 (rule library)：約 60 條人看得懂的條件 (K 棒型態、連漲連跌、跳空、突破、乖離、量能、籌碼連續、外資期貨、國際盤同日、
   VIX、日曆、多空狀態組合)，對 1/2/3/5 日後的上漲率與平均報酬做逐年統計。規律不擬合任何參數，所以「逐年一致性」就是它的樣本外檢驗：
   要求 (1) n ≥ 60，(2) 逐年 (n≥8 的年份) 上漲率與基準同方向的比例 ≥ 65%，(3) 超額 t 值 |t| ≥ 2.5 (以 sqrt(h) 修正重疊)，(4) 近 3 年不反向。
   通過的規律才進「有效規律」；今日符合的規律列成「今日規律」，供前端顯示與 forecast 加註。
B. 相似走勢 (analog / k-NN)：把每天前 10 日的標準化報酬、振幅、位置 (乖離/20 日位置)、量能與外資 z 組成向量，只在該日之前的歷史找 K=80 個最相似的日子，
   統計它們之後 1/2/3/5 日的上漲率與平均 → knn_p{h}。逐年擴張視窗 (2014 起) 驗證：knn_p 是否比基準有預測力 (IC、頂/底分位命中)。
   驗證通過的視野才輸出 (forecast.json `analog`)，並可當 short_term 的候選特徵 (由 train() 的 _pick 決定是否採用)。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)
HORIZONS = (1, 2, 3, 5)
FIRST_TEST_YEAR = 2014
MIN_N, MIN_YEAR_N, YEAR_CONSIST, MIN_T = 60, 8, 0.65, 2.5
KNN_K, KNN_WINDOW = 80, 10
PATH = config.DATA_DIR / "models" / "patterns.json"


# ------------------------------------------------------------------ A. 規律庫
def _prep(m: pd.DataFrame) -> pd.DataFrame:
    d = m.copy()
    c, o, h, l = d["close"].astype(float), d["open"].astype(float), d["high"].astype(float), d["low"].astype(float)
    pc = c.shift(1)
    d["r1"] = (c / pc - 1) * 100
    for k in HORIZONS:
        d[f"fwd{k}"] = (c.shift(-k) / c - 1) * 100
    rng = (h - l).replace(0, np.nan)
    d["clv_"] = ((c - l) / rng).clip(0, 1)
    d["body_"] = (c - o) / rng
    d["uw_"] = (h - np.maximum(c, o)) / rng
    d["lw_"] = (np.minimum(c, o) - l) / rng
    d["gap_"] = (o / pc - 1) * 100
    d["gapfill_"] = np.where(d["gap_"] > 0, l <= pc, np.where(d["gap_"] < 0, h >= pc, False))
    d["atr_"] = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).rolling(14).mean() / c * 100
    d["range_ratio_"] = (h - l) / c * 100 / d["atr_"]
    sgn = np.sign(d["r1"].fillna(0))
    st = np.zeros(len(d));
    for i in range(1, len(d)):
        st[i] = st[i - 1] + sgn.iloc[i] if sgn.iloc[i] == np.sign(st[i - 1]) or st[i - 1] == 0 else sgn.iloc[i]
    d["streak_"] = st
    d["bull_"] = c >= d["ma60"].astype(float)
    d["above20_"] = c >= d["ma20"].astype(float)
    d["hi20_"] = c >= d["hi20"].astype(float)
    d["lo20_"] = c <= d["lo20"].astype(float)
    d["pos20_"] = (c - d["lo20"]) / (d["hi20"] - d["lo20"]).replace(0, np.nan)
    rsi_up = d["r1"].clip(lower=0).rolling(14).mean(); rsi_dn = (-d["r1"].clip(upper=0)).rolling(14).mean()
    d["rsi_"] = 100 - 100 / (1 + rsi_up / rsi_dn.replace(0, np.nan))
    d["ret3_"] = (c / c.shift(3) - 1) * 100
    d["dow_"] = pd.to_datetime(d["date"]).dt.weekday
    d["dom_"] = pd.to_datetime(d["date"]).dt.day
    d["vix_"] = d.get("g_vix_level")
    d["kospi0_"] = d.get("kospi_r0", d.get("g_kospi_r1"))
    d["sox1_"] = d.get("g_sox_r1")
    d["year"] = d["date"].astype(str).str[:4]
    return d


def rule_library(d: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    g = lambda k: d[k] if k in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    r1, r3, st = d["r1"], d["ret3_"], d["streak_"]
    R: dict[str, tuple[pd.Series, str]] = {
        # 價格型態
        "連漲 3 日以上": (st >= 3, "動能"), "連漲 5 日以上": (st >= 5, "動能"), "連跌 3 日以上": (st <= -3, "反轉"), "連跌 5 日以上": (st <= -5, "反轉"),
        "單日大漲 >2%": (r1 > 2, "動能"), "單日大跌 <-2%": (r1 < -2, "反轉"), "單日大跌 <-3%": (r1 < -3, "反轉"),
        "3 日急漲 >4%": (r3 > 4, "過熱"), "3 日急跌 <-4%": (r3 < -4, "超跌"),
        "長下影線 (下影 ≥50% 振幅)": (d["lw_"] >= 0.5, "承接"), "長上影線 (上影 ≥50%)": (d["uw_"] >= 0.5, "賣壓"),
        "十字線 (實體 <10%)": (d["body_"].abs() < 0.1, "猶豫"), "長紅 (實體 ≥70% 且漲 >1%)": ((d["body_"] >= 0.7) & (r1 > 1), "動能"), "長黑 (實體 ≤-70% 且跌 >1%)": ((d["body_"] <= -0.7) & (r1 < -1), "賣壓"),
        "收在最高 (CLV ≥0.9) 且漲": ((d["clv_"] >= 0.9) & (r1 > 0.5), "動能"), "收在最低 (CLV ≤0.1) 且跌": ((d["clv_"] <= 0.1) & (r1 < -0.5), "賣壓"),
        "跳空上漲 >1% 未回補": ((d["gap_"] > 1) & (~d["gapfill_"].astype(bool)), "動能"), "跳空上漲 >1% 已回補": ((d["gap_"] > 1) & d["gapfill_"].astype(bool), "假突破"),
        "跳空下跌 <-1% 未回補": ((d["gap_"] < -1) & (~d["gapfill_"].astype(bool)), "賣壓"), "跳空下跌 <-1% 已回補": ((d["gap_"] < -1) & d["gapfill_"].astype(bool), "承接"),
        "振幅 >1.8× ATR": (d["range_ratio_"] > 1.8, "波動"), "振幅 <0.5× ATR (窄幅)": (d["range_ratio_"] < 0.5, "盤整"),
        "創 20 日新高": (d["hi20_"], "突破"), "創 20 日新低": (d["lo20_"], "破底"), "創 20 日新高且量 >1.3×": (d["hi20_"] & (g("vol_ratio") > 1.3), "突破"),
        "月線正乖離 >4%": (d["bias20"] > 4, "過熱"), "月線負乖離 <-4%": (d["bias20"] < -4, "超跌"), "月線負乖離 <-6%": (d["bias20"] < -6, "超跌"),
        "RSI14 >75": (d["rsi_"] > 75, "過熱"), "RSI14 <25": (d["rsi_"] < 25, "超跌"),
        "跌破月線 (前日在上)": ((~d["above20_"]) & d["above20_"].shift(1).fillna(False).astype(bool), "轉弱"), "站回月線 (前日在下)": (d["above20_"] & (~d["above20_"].shift(1).fillna(True).astype(bool)), "轉強"),
        "季線之下大跌 <-2%": ((~d["bull_"]) & (r1 < -2), "空頭賣壓"), "季線之上大跌 <-2%": (d["bull_"] & (r1 < -2), "多頭回檔"),
        "季線之上連跌 3 日": (d["bull_"] & (st <= -3), "多頭回檔"), "季線之下連漲 3 日": ((~d["bull_"]) & (st >= 3), "空頭反彈"),
        # 量能
        "爆量 >1.8× 且漲": ((g("vol_ratio") > 1.8) & (r1 > 0), "量增"), "爆量 >1.8× 且跌": ((g("vol_ratio") > 1.8) & (r1 < 0), "量增賣壓"), "量縮 <0.6×": (g("vol_ratio") < 0.6, "量縮"),
        # 籌碼
        "外資連買 ≥3 日": (g("foreign_streak") >= 3, "籌碼"), "外資連買 ≥5 日": (g("foreign_streak") >= 5, "籌碼"), "外資連賣 ≥3 日": (g("foreign_streak") <= -3, "籌碼"), "外資連賣 ≥5 日": (g("foreign_streak") <= -5, "籌碼"),
        "外資單日買超 z>1.5": (g("foreign_z1") > 1.5, "籌碼"), "外資單日賣超 z<-1.5": (g("foreign_z1") < -1.5, "籌碼"),
        "外資賣超但指數漲": ((g("foreign") < 0) & (r1 > 0.5), "籌碼背離"), "外資買超但指數跌": ((g("foreign") > 0) & (r1 < -0.5), "籌碼背離"),
        "外資期貨淨部位低檔 (<25%)": (g("fut_foreign_pct") < 0.25, "期貨"), "外資期貨淨部位高檔 (>75%)": (g("fut_foreign_pct") > 0.75, "期貨"),
        "外資期貨 5 日增倉 z>1.5": (g("fut_foreign_chg5_z") > 1.5, "期貨"), "外資期貨 5 日減倉 z<-1.5": (g("fut_foreign_chg5_z") < -1.5, "期貨"),
        "融資 5 日增 >2%": (g("margin_chg5_pct") > 2, "散戶"), "融資 5 日減 >2%": (g("margin_chg5_pct") < -2, "散戶"),
        "投信連買 ≥3 日": (g("trust_streak") >= 3, "籌碼"), "投信連賣 ≥3 日": (g("trust_streak") <= -3, "籌碼"),
        # 國際盤 (同日已知)
        "KOSPI 同日 >1%": (d["kospi0_"] > 1, "亞股"), "KOSPI 同日 <-1%": (d["kospi0_"] < -1, "亞股"),
        "費半前晚 >2%": (d["sox1_"] > 2, "美股"), "費半前晚 <-2%": (d["sox1_"] < -2, "美股"),
        "VIX >25": (d["vix_"] > 25, "恐慌"), "VIX <13": (d["vix_"] < 13, "自滿"), "VIX >25 且收紅": ((d["vix_"] > 25) & (r1 > 0), "恐慌後"),
        # 日曆
        "週一": (d["dow_"] == 0, "日曆"), "週五": (d["dow_"] == 4, "日曆"), "月初 (1~3 日)": (d["dom_"] <= 3, "日曆"), "月底 (≥27 日)": (d["dom_"] >= 27, "日曆"),
        "結算週 (距結算 ≤4 日)": (g("settle_week") == 1, "日曆"),
        # 組合
        "多頭 + 外資連買 ≥3 + 月線乖離 <2%": (d["bull_"] & (g("foreign_streak") >= 3) & (d["bias20"] < 2), "組合"),
        "空頭 + 外資連賣 ≥3": ((~d["bull_"]) & (g("foreign_streak") <= -3), "組合"),
        "超跌 (乖離 <-4%) + 長下影線": ((d["bias20"] < -4) & (d["lw_"] >= 0.4), "組合"),
        "連跌 ≥3 + 外資買超": ((st <= -3) & (g("foreign") > 0), "組合"), "連漲 ≥3 + 外資賣超": ((st >= 3) & (g("foreign") < 0), "組合"),
        "KOSPI 同日 >1% + 台股漲 <0.3%": ((d["kospi0_"] > 1) & (r1 < 0.3), "落後補漲"), "費半前晚 >2% + 台股跳空未回補": ((d["sox1_"] > 2) & (d["gap_"] > 0.5) & (~d["gapfill_"].astype(bool)), "組合"),
    }
    return R


def _tstat(x: np.ndarray, base_mean: float, h: int) -> float:
    if len(x) < 5 or x.std() == 0:
        return 0.0
    return float((x.mean() - base_mean) / (x.std() / math.sqrt(len(x))) / math.sqrt(h))   # sqrt(h) 修正重疊視野


def evaluate_rules(d: pd.DataFrame) -> list[dict]:
    R = rule_library(d)
    years = sorted(d["year"].unique())
    last3 = years[-3:]
    out = []
    for name, (mask, cat) in R.items():
        m = mask.fillna(False).astype(bool)
        rec = {"name": name, "cat": cat, "n": int(m.sum()), "h": {}}
        if rec["n"] < MIN_N:
            continue
        best_ok = False
        for h in HORIZONS:
            col = f"fwd{h}"
            x = d.loc[m, col].dropna(); base = d[col].dropna()
            if len(x) < MIN_N:
                continue
            up, bup = float((x > 0).mean()), float((base > 0).mean())
            mean, bmean = float(x.mean()), float(base.mean())
            t = _tstat(x.values, bmean, h)
            sign = np.sign(mean - bmean)
            yrs = []
            for y in years:
                xy = d.loc[m & (d["year"] == y), col].dropna(); by = d.loc[d["year"] == y, col].dropna()
                if len(xy) >= MIN_YEAR_N and len(by):
                    yrs.append({"y": y, "n": len(xy), "up": round(float((xy > 0).mean()), 2), "mean": round(float(xy.mean()), 2), "same": bool(np.sign(xy.mean() - by.mean()) == sign)})
            cons = float(np.mean([z["same"] for z in yrs])) if yrs else 0.0
            recent = [z for z in yrs if z["y"] in last3]
            recent_ok = (not recent) or (float(np.mean([z["same"] for z in recent])) >= 0.5)
            ok = bool(cons >= YEAR_CONSIST and abs(t) >= MIN_T and len(yrs) >= 6 and recent_ok)
            best_ok |= ok
            rec["h"][str(h)] = {"n": len(x), "up": round(up, 3), "base_up": round(bup, 3), "mean": round(mean, 3), "base_mean": round(bmean, 3), "excess": round(mean - bmean, 3),
                                "t": round(t, 2), "years": len(yrs), "consist": round(cons, 2), "recent_ok": recent_ok, "valid": ok, "direction": "偏多" if sign > 0 else "偏空",
                                "by_year": yrs}
        rec["valid_any"] = best_ok
        rec["last_fired"] = str(d.loc[m, "date"].iloc[-1]) if m.any() else None
        out.append(rec)
    out.sort(key=lambda r: -max([abs(v["t"]) for v in r["h"].values() if v["valid"]] or [0]))
    return out


def active_today(d: pd.DataFrame, rules: list[dict]) -> list[dict]:
    R = rule_library(d)
    i = d.index[-1]
    act = []
    for r in rules:
        mask = R[r["name"]][0]
        if bool(mask.fillna(False).iloc[-1]):
            hv = {h: v for h, v in r["h"].items() if v["valid"]}
            act.append({"name": r["name"], "cat": r["cat"], "valid": bool(hv), "n": r["n"],
                        "h": {h: {k: v[k] for k in ("up", "base_up", "mean", "excess", "t", "consist", "direction", "valid")} for h, v in r["h"].items()}})
    act.sort(key=lambda a: (not a["valid"], -max([abs(v["t"]) for v in a["h"].values()] or [0])))
    return act


# ------------------------------------------------------------------ B. 相似走勢 (k-NN analog)
def _analog_vectors(d: pd.DataFrame) -> np.ndarray:
    c = d["close"].astype(float)
    r = (c.pct_change() * 100)
    vol = r.rolling(20).std().replace(0, np.nan)
    z = r / vol
    cols = [z.shift(k) for k in range(KNN_WINDOW)]          # 近 10 日標準化報酬 (最近在前)
    cols += [d["bias20"].astype(float) / 5, d["pos20_"].astype(float) * 2 - 1, (d["range_ratio_"].astype(float) - 1),
             (d.get("vol_ratio", pd.Series(1.0, index=d.index)).astype(float) - 1), d.get("foreign_z5", pd.Series(0.0, index=d.index)).astype(float) / 2]
    X = pd.concat(cols, axis=1).values.astype(float)
    return X


def knn_features(d: pd.DataFrame, k: int = KNN_K, min_hist: int = 750) -> pd.DataFrame:
    """逐日只用「該日之前」的歷史找最相似 k 日，輸出 knn_p{h} (之後 h 日上漲率) 與 knn_m{h} (平均報酬)。O(n²) 但 n≈4000、維度 15，數秒。"""
    X = _analog_vectors(d)
    ok = ~np.isnan(X).any(axis=1)
    F = {f"knn_p{h}": np.full(len(d), np.nan) for h in HORIZONS}
    F.update({f"knn_m{h}": np.full(len(d), np.nan) for h in HORIZONS})
    F["knn_dist"] = np.full(len(d), np.nan)
    fwd = {h: d[f"fwd{h}"].values.astype(float) for h in HORIZONS}
    idx_ok = np.where(ok)[0]
    Xn = X.copy()
    for i in idx_ok:
        if i < min_hist:
            continue
        past = idx_ok[idx_ok < i - max(HORIZONS)]          # 鄰居的未來報酬必須已知 (排除最後 5 日) → 無前視
        if len(past) < k * 3:
            continue
        dist = np.sqrt(((Xn[past] - Xn[i]) ** 2).sum(axis=1))
        nn = past[np.argpartition(dist, k)[:k]]
        w = 1.0 / (1.0 + dist[np.searchsorted(past, nn)])
        for h in HORIZONS:
            y = fwd[h][nn]; m = ~np.isnan(y)
            if m.sum() >= k // 2:
                F[f"knn_p{h}"][i] = float(np.average((y[m] > 0), weights=w[m]))
                F[f"knn_m{h}"][i] = float(np.average(y[m], weights=w[m]))
        F["knn_dist"][i] = float(np.sort(dist)[:k].mean())
    return pd.DataFrame(F, index=d.index)


def evaluate_knn(d: pd.DataFrame, kf: pd.DataFrame) -> dict:
    """逐年 (2014~) 檢驗：IC (knn_m vs fwd)、knn_p 頂/底 20% 的方向命中 vs 基準。純非參數，所以逐年就是 OOS。"""
    out = {}
    dd = pd.concat([d[["date", "year"] + [f"fwd{h}" for h in HORIZONS]], kf], axis=1)
    test = dd[dd["year"].astype(int) >= FIRST_TEST_YEAR]
    for h in HORIZONS:
        t = test.dropna(subset=[f"knn_p{h}", f"fwd{h}"])
        if len(t) < 200:
            continue
        ic = float(t[f"knn_m{h}"].rank().corr(t[f"fwd{h}"].rank()))
        yr = t.groupby("year").apply(lambda g: g[f"knn_m{h}"].rank().corr(g[f"fwd{h}"].rank()), include_groups=False)
        hi, lo = t[f"knn_p{h}"].quantile(0.8), t[f"knn_p{h}"].quantile(0.2)
        top, bot = t[t[f"knn_p{h}"] >= hi], t[t[f"knn_p{h}"] <= lo]
        base_up = float((t[f"fwd{h}"] > 0).mean())
        out[str(h)] = {"n": len(t), "ic": round(ic, 3), "ic_years_pos": f"{int((yr > 0).sum())}/{len(yr)}", "base_up": round(base_up, 3),
                       "top_up": round(float((top[f"fwd{h}"] > 0).mean()), 3), "top_mean": round(float(top[f"fwd{h}"].mean()), 3), "top_thr": round(float(hi), 3),
                       "bot_up": round(float((bot[f"fwd{h}"] > 0).mean()), 3), "bot_mean": round(float(bot[f"fwd{h}"].mean()), 3), "bot_thr": round(float(lo), 3),
                       "brier": round(float(((t[f"knn_p{h}"] - (t[f"fwd{h}"] > 0)) ** 2).mean()), 4), "brier_base": round(float(((base_up - (t[f"fwd{h}"] > 0)) ** 2).mean()), 4)}
        o = out[str(h)]
        o["valid"] = bool(o["ic"] >= 0.04 and int(o["ic_years_pos"].split("/")[0]) >= 0.65 * len(yr) and o["top_up"] - o["base_up"] >= 0.03)
    return out


def analog_today(d: pd.DataFrame, k: int = 30) -> dict:
    X = _analog_vectors(d)
    i = len(d) - 1
    if np.isnan(X[i]).any():
        return {}
    ok = np.where(~np.isnan(X).any(axis=1))[0]
    past = ok[ok < i - max(HORIZONS)]
    dist = np.sqrt(((X[past] - X[i]) ** 2).sum(axis=1))
    nn = past[np.argsort(dist)[:k]]
    rows = []
    for j in nn:
        rows.append({"date": str(d["date"].iloc[j]), **{f"fwd{h}": round(float(d[f"fwd{h}"].iloc[j]), 2) if pd.notna(d[f"fwd{h}"].iloc[j]) else None for h in HORIZONS}})
    stats = {}
    for h in HORIZONS:
        y = np.array([r[f"fwd{h}"] for r in rows if r[f"fwd{h}"] is not None])
        if len(y):
            stats[str(h)] = {"up": round(float((y > 0).mean()), 3), "mean": round(float(y.mean()), 3), "q20": round(float(np.quantile(y, 0.2)), 2), "q80": round(float(np.quantile(y, 0.8)), 2), "n": int(len(y))}
    return {"k": k, "neighbors": rows[:12], "stats": stats, "mean_dist": round(float(dist[np.argsort(dist)[:k]].mean()), 3)}


# ------------------------------------------------------------------ 主流程
def build(matrix: pd.DataFrame, write: bool = True) -> dict:
    d = _prep(matrix)
    rules = evaluate_rules(d)
    kf = knn_features(d)
    kres = evaluate_knn(d, kf)
    act = active_today(d, rules)
    an = analog_today(d)
    valid = [r for r in rules if r["valid_any"]]
    out = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "as_of": str(d["date"].iloc[-1]), "n_days": int(len(d)),
           "criteria": {"min_n": MIN_N, "year_consist": YEAR_CONSIST, "min_t": MIN_T, "recent_ok": "近 3 年不反向"},
           "n_rules": len(rules), "n_valid": len(valid),
           "rules": [{k: v for k, v in r.items() if k != "h"} | {"h": {h: {kk: vv for kk, vv in x.items() if kk != "by_year"} for h, x in r["h"].items()}} for r in rules],
           "by_year": {r["name"]: {h: x["by_year"] for h, x in r["h"].items() if x["valid"]} for r in valid},
           "today": act, "knn_eval": kres, "analog": an,
           "knn_today": {h: {"p": round(float(kf[f"knn_p{h}"].iloc[-1]), 3) if pd.notna(kf[f"knn_p{h}"].iloc[-1]) else None,
                             "m": round(float(kf[f"knn_m{h}"].iloc[-1]), 3) if pd.notna(kf[f"knn_m{h}"].iloc[-1]) else None} for h in HORIZONS}}
    if write:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def for_forecast(pat: dict) -> dict:
    """forecast.json 用的精簡版：今日符合的規律 (有效的排前面)、1 日規律分、相似走勢 (標明驗證無預測力則只作參考)。"""
    if not pat:
        return {}
    today = pat.get("today") or []
    valid_today = [a for a in today if a["valid"]]
    score, parts = 0.0, []
    for a in valid_today:
        v = a["h"].get("1")
        if v and v["valid"]:
            score += v["excess"]; parts.append(f"{a['name']} ({v['direction']} {v['up']:.0%} vs {v['base_up']:.0%})")
    kv = pat.get("knn_eval") or {}
    any_knn = any(v.get("valid") for v in kv.values())
    return {"as_of": pat.get("as_of"), "n_rules": pat.get("n_rules"), "n_valid": pat.get("n_valid"),
            "today": [{"name": a["name"], "cat": a["cat"], "valid": a["valid"], "n": a["n"], "h": a["h"]} for a in today[:14]],
            "score1": round(score, 3), "direction1": "偏多" if score > 0.1 else "偏空" if score < -0.1 else "中性",
            "note1": ("今日符合 %d 條驗證有效規律：%s" % (len(parts), "；".join(parts))) if parts else "今日無符合的驗證有效規律",
            "valid_rules": [{"name": r["name"], "cat": r["cat"], "n": r["n"], "h": {h: {k: x[k] for k in ("direction", "up", "base_up", "excess", "t", "consist", "years")} for h, x in r["h"].items() if x["valid"]}} for r in pat.get("rules", []) if r.get("valid_any")],
            "analog": {**(pat.get("analog") or {}), "validated": any_knn, "warn": "" if any_knn else "相似走勢法 2014~ 逐年驗證無預測力 (IC≈0)，僅供參考，不列入預測"},
            "knn_eval": {h: {k: v[k] for k in ("n", "ic", "ic_years_pos", "base_up", "top_up", "bot_up", "valid")} for h, v in kv.items()},
            "criteria": pat.get("criteria")}


def load() -> dict | None:
    return json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else None
