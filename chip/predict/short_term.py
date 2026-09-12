"""前五日預測模組 v2 (視野 1/2/3/5 日)：目標是提高「方向命中率」，而不是只做期望報酬排序。

作法 (全部 2014~ 逐年擴張視窗樣本外驗證，用 OOS 結果選模型與門檻，不用訓練集內表現)：
1. 短線專用特徵：只留與 1~5 日報酬有關的 (當日/前兩日漲跌、跳空、振幅、收盤位置、近 5 日上漲天數、乖離、量能、
   外資當日/5 日、外資期貨變化、結算週、前晚美股/費半/ADR/韓股、VIX)，去掉 20~60 日慢變數 → 少雜訊、少過擬合。
2. 兩種模型 + 集成：淺層 LightGBM 多種子 (非線性) 與 Ridge (線性、穩定)，各自標準化後平均 → 通常比單一模型更穩。
3. 夜盤變體：加入「前晚夜盤台指期漲跌」(2017-05 起有資料，隔日開盤跳空 r≈0.71) 另訓一組，收盤後~開盤前用它。
4. 命中率提升的關鍵是「有把握才叫方向」：以 OOS 預測分位切三檔 (前 30% 偏多 / 後 30% 偏空 / 中間中性)，
   只有該檔位 OOS 命中率高於基準 3 個百分點以上才啟用，並把該檔位的歷史命中率一起輸出，讓使用者知道這次叫牌的可信度。
5. 每個視野在 {lgb, ridge, ens} 中依 OOS「叫牌命中率」自動選最佳；夜盤變體另選。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from ..analysis import backtest
from ..sources import finmind
from . import model as M
from .features import FEATURE_NAMES, market_matrix

log = logging.getLogger(__name__)
HORIZONS = (1, 2, 3, 5)
FIRST_TEST_YEAR = 2014
FIRST_TEST_YEAR_NIGHT = 2020
ST_FEATURES = [
    "ret1", "ret1_l1", "ret1_l2", "ret5", "bias5", "bias20", "gap_open", "range_pct", "clv", "up5",
    "hi20_dist", "lo20_dist", "vola20", "vola_ratio", "vol_ratio", "amount_5d_ratio",
    "foreign_z1", "foreign_z5", "foreign_streak", "trust_z5", "dealer_z1",
    "fut_foreign_pct", "fut_foreign_chg1_z", "fut_foreign_chg5_z", "foreign_consistency", "margin_chg5_pct", "short_chg5_z",
    "f_fut_foreign", "f_reversion", "composite_smooth", "composite_chg5", "state_bull", "state_bear",
    "dow", "days_to_settle", "settle_week", "days_to_month_end",
    "g_vix_level", "g_vix_chg", "g_sox_r1", "g_sox_r5", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1",
    "g_kospi_r1", "g_kospi_r5", "g_usdtwd_r5", "g_dxy_r1", "g_us10y_r5", "g_vix_term",
]
NIGHT_FEATURE = "night_chg_pct"
NAMES = {**FEATURE_NAMES, "ret1_l1": "前 1 日漲跌%", "ret1_l2": "前 2 日漲跌%", "gap_open": "今日開盤跳空%", "range_pct": "今日振幅%",
         "clv": "收盤在當日區間位置", "up5": "近 5 日上漲天數", NIGHT_FEATURE: "前晚夜盤台指期%"}
LGB_PARAMS = dict(M.PARAMS, n_estimators=120, min_child_samples=150)
RIDGE_ALPHA = 30.0
TIER = 0.30          # 前/後 30% 才叫方向
MIN_EDGE = 0.03      # 檔位命中率需高於基準 3 個百分點才啟用


# ------------------------------------------------------------------ 特徵
def build_matrix(scored: pd.DataFrame, night: pd.DataFrame | None = None) -> pd.DataFrame:
    d = market_matrix(scored)
    c = d["close"].astype(float)
    prev = c.shift(1)
    d["ret1_l1"] = d["ret1"].shift(1)
    d["ret1_l2"] = d["ret1"].shift(2)
    o, h, l = d["open"].astype(float), d["high"].astype(float), d["low"].astype(float)
    d["gap_open"] = (o / prev - 1) * 100
    d["range_pct"] = (h - l) / prev * 100
    d["clv"] = np.where((h - l) > 0, ((c - l) - (h - c)) / (h - l).replace(0, np.nan), 0.0)
    d["up5"] = (d["ret1"] > 0).astype(int).rolling(5).sum()
    # 夜盤：FinMind after_market 的 date = 該夜盤準備的「隔一交易日」→ 列 D 要用 date == D 的下一列日期 的夜盤
    d[NIGHT_FEATURE] = np.nan
    if night is not None and not night.empty:
        nm = dict(zip(night["date"].astype(str), night["night_chg_pct"].astype(float)))
        nxt = d["date"].astype(str).shift(-1)
        d[NIGHT_FEATURE] = nxt.map(nm)
    for col in ST_FEATURES:
        if col not in d:
            d[col] = np.nan
    return d


def _night_hist() -> pd.DataFrame:
    try:
        return finmind.tx_night_history("2017-01-01")
    except Exception as e:  # noqa: BLE001
        log.warning("night history: %s", e)
        return pd.DataFrame()


# ------------------------------------------------------------------ 模型
class RidgeModel:
    """標準化 + 中位數補值 + Ridge；預測值除以訓練期預測標準差，方便與 LGB 平均。"""

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.linear_model import Ridge
        self.med = X.median().fillna(0.0)          # 整欄缺值 (早期無資料) 的中位數為 NaN → 補 0
        Xf = X.fillna(self.med).fillna(0.0)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1.0).fillna(1.0)
        Z = (Xf - self.mu) / self.sd
        self.m = Ridge(alpha=self.alpha).fit(Z.values, y.values)
        p = self.m.predict(Z.values)
        self.psd = float(np.std(p)) or 1.0
        self.coef = dict(zip(X.columns, self.m.coef_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = ((X.fillna(self.med).fillna(0.0) - self.mu) / self.sd).fillna(0.0)
        return self.m.predict(Z.values)

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X) / self.psd


class LgbModel:
    def __init__(self, params: dict | None = None):
        self.params = params or LGB_PARAMS

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.models = M.fit_ensemble(X, y, self.params)
        p = M.predict_ensemble(self.models, X)
        self.psd = float(np.std(p)) or 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return M.predict_ensemble(self.models, X)

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X) / self.psd


def _wf(d: pd.DataFrame, feats: list[str], target: str, h: int, first_year: int, make, min_train: int = 400) -> pd.DataFrame:
    d = d.dropna(subset=[target]).copy()
    d["year"] = d["date"].str[:4].astype(int)
    rows = []
    for y in sorted(d["year"].unique()):
        if y < first_year:
            continue
        test = d[d["year"] == y]
        train = d[d["year"] < y].iloc[:-h]
        if len(train) < min_train or test.empty:
            continue
        m = make().fit(train[feats], train[target])
        rows.append(pd.DataFrame({"date": test["date"].values, "year": y, "pred": m.predict_std(test[feats]), "actual": test[target].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ 命中率評估與叫牌門檻
def tier_stats(oos: pd.DataFrame) -> dict:
    """三檔叫牌：pred 前 30% 叫偏多、後 30% 叫偏空。回傳各檔 OOS 命中率、覆蓋率與是否啟用。"""
    if oos.empty:
        return {}
    lo, hi = float(oos["pred"].quantile(TIER)), float(oos["pred"].quantile(1 - TIER))
    up, dn, mid = oos[oos["pred"] >= hi], oos[oos["pred"] <= lo], oos[(oos["pred"] > lo) & (oos["pred"] < hi)]
    base_up = float((oos["actual"] > 0).mean())
    up_hit = float((up["actual"] > 0).mean()) if len(up) else np.nan
    dn_hit = float((dn["actual"] < 0).mean()) if len(dn) else np.nan
    up_on = bool(up_hit >= base_up + MIN_EDGE)
    dn_on = bool(dn_hit >= (1 - base_up) + MIN_EDGE)
    calls = (len(up) if up_on else 0) + (len(dn) if dn_on else 0)
    hits = (float((up["actual"] > 0).sum()) if up_on else 0) + (float((dn["actual"] < 0).sum()) if dn_on else 0)
    return {"edge_lo": round(lo, 4), "edge_hi": round(hi, 4), "base_up": round(base_up, 3),
            "up_hit": round(up_hit, 3), "up_n": int(len(up)), "up_on": up_on, "up_mean": round(float(up["actual"].mean()), 2) if len(up) else None,
            "dn_hit": round(dn_hit, 3), "dn_n": int(len(dn)), "dn_on": dn_on, "dn_mean": round(float(dn["actual"].mean()), 2) if len(dn) else None,
            "mid_up": round(float((mid["actual"] > 0).mean()), 3) if len(mid) else None,
            "call_hit": round(hits / calls, 3) if calls else None, "call_cov": round(calls / len(oos), 3),
            "by_year": _tier_by_year(oos, lo, hi, up_on, dn_on)}


def _tier_by_year(oos: pd.DataFrame, lo: float, hi: float, up_on: bool, dn_on: bool) -> list[dict]:
    out = []
    for y, g in oos.groupby("year"):
        up, dn = g[g["pred"] >= hi], g[g["pred"] <= lo]
        calls = (len(up) if up_on else 0) + (len(dn) if dn_on else 0)
        hits = (float((up["actual"] > 0).sum()) if up_on else 0) + (float((dn["actual"] < 0).sum()) if dn_on else 0)
        out.append({"year": int(y), "n": int(len(g)), "base_up": round(float((g["actual"] > 0).mean()), 3),
                    "call_hit": round(hits / calls, 3) if calls else None, "calls": int(calls)})
    return out


def _combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b[["date", "pred"]], on="date", suffixes=("", "_b"))
    m["pred"] = (m["pred"] + m["pred_b"]) / 2
    return m.drop(columns=["pred_b"])


def _report(oos: pd.DataFrame) -> dict:
    met = M.metrics(oos)
    ts = tier_stats(oos)
    return {"n": met.get("n"), "rank_ic": met.get("rank_ic"), "ic_positive_years": met.get("ic_positive_years"),
            "base_hit": met.get("base_hit"), "bin_hit": met.get("bin_hit"), "tiers": {k: v for k, v in ts.items() if k != "by_year"},
            "tier_by_year": ts.get("by_year"), "calibration": met.get("calibration")}


def _pick(cands: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    """依 OOS 叫牌命中率 (次序 IC) 選最佳。"""
    best, best_key, reps = None, None, {}
    for k, oos in cands.items():
        if oos.empty:
            continue
        rep = _report(oos)
        reps[k] = rep
        score = ((rep["tiers"].get("call_hit") or 0), rep.get("rank_ic") or 0)
        if best is None or score > best:
            best, best_key = score, k
    return best_key, reps


# ------------------------------------------------------------------ 訓練
def train(write: bool = True, verbose: bool = True) -> dict:
    scored = backtest.load_long()
    night = _night_hist()
    mat = build_matrix(scored, night)
    results = {}
    for h in HORIZONS:
        tgt = f"fwd{h}"
        # 基本變體 (2010~)
        oos_l = _wf(mat, ST_FEATURES, tgt, h, FIRST_TEST_YEAR, lambda: LgbModel())
        oos_r = _wf(mat, ST_FEATURES, tgt, h, FIRST_TEST_YEAR, lambda: RidgeModel())
        cands = {"lgb": oos_l, "ridge": oos_r, "ens": _combine(oos_l, oos_r) if not oos_l.empty and not oos_r.empty else pd.DataFrame()}
        key, reps = _pick(cands)
        # 夜盤變體 (2017-05~)
        mn = mat.dropna(subset=[NIGHT_FEATURE])
        feats_n = ST_FEATURES + [NIGHT_FEATURE]
        oos_ln = _wf(mn, feats_n, tgt, h, FIRST_TEST_YEAR_NIGHT, lambda: LgbModel(), min_train=400)
        oos_rn = _wf(mn, feats_n, tgt, h, FIRST_TEST_YEAR_NIGHT, lambda: RidgeModel(), min_train=400)
        cands_n = {"lgb": oos_ln, "ridge": oos_rn, "ens": _combine(oos_ln, oos_rn) if not oos_ln.empty and not oos_rn.empty else pd.DataFrame()}
        key_n, reps_n = _pick(cands_n)
        results[h] = {"base": {"chosen": key, "reports": reps}, "night": {"chosen": key_n, "reports": reps_n}}
        if verbose:
            r = reps.get(key, {}).get("tiers", {}); rn = reps_n.get(key_n, {}).get("tiers", {})
            print(f"h{h}: 基本 {key} IC {reps.get(key, {}).get('rank_ic')} 基準 {r.get('base_up')} 叫牌命中 {r.get('call_hit')} (覆蓋 {r.get('call_cov')}; 偏多 {r.get('up_hit')}{'✓' if r.get('up_on') else '✗'} 偏空 {r.get('dn_hit')}{'✓' if r.get('dn_on') else '✗'})"
                  f" | 夜盤 {key_n} IC {reps_n.get(key_n, {}).get('rank_ic')} 叫牌命中 {rn.get('call_hit')} (覆蓋 {rn.get('call_cov')}; 偏多 {rn.get('up_hit')} 偏空 {rn.get('dn_hit')})")
        if write:
            for variant, feats, dd, k, rr in (("base", ST_FEATURES, mat, key, reps), ("night", feats_n, mn, key_n, reps_n)):
                if not k:
                    continue
                d = dd.dropna(subset=[tgt])
                bundle = {"horizon": h, "variant": variant, "chosen": k, "features": feats, "trained_at": dt.datetime.now(config.TZ).isoformat(),
                          "train_end": str(d["date"].max()), "n_train": int(len(d)), "report": rr[k],
                          "lgb": LgbModel().fit(d[feats], d[tgt]) if k in ("lgb", "ens") else None,
                          "ridge": RidgeModel().fit(d[feats], d[tgt]) if k in ("ridge", "ens") else None}
                M.save(f"st_h{h}_{variant}", bundle)
    if write:
        M.save_json("short_term_metrics", {str(h): {v: {"chosen": results[h][v]["chosen"], **{kk: vv for kk, vv in (results[h][v]["reports"].get(results[h][v]["chosen"]) or {}).items() if kk != "calibration"}}
                                                    for v in ("base", "night")} for h in HORIZONS})
    return results


# ------------------------------------------------------------------ 預測
def _predict_bundle(b: dict, row: pd.DataFrame) -> tuple[float, dict]:
    X = row[b["features"]].astype(float)
    preds, drivers = [], {}
    if b.get("lgb") is not None:
        preds.append(float(b["lgb"].predict_std(X)[0]))
        drivers = M.explain(b["lgb"].models, X, NAMES)
    if b.get("ridge") is not None:
        preds.append(float(b["ridge"].predict_std(X)[0]))
        if not drivers:
            Z = ((X.fillna(b["ridge"].med).fillna(0.0) - b["ridge"].mu) / b["ridge"].sd).fillna(0.0).iloc[0]
            contrib = sorted(((f, float(Z[f] * b["ridge"].coef[f])) for f in b["features"]), key=lambda x: x[1])
            drivers = {"negative": [{"feature": f, "name": NAMES.get(f, f), "value": float(X.iloc[0][f]) if pd.notna(X.iloc[0][f]) else None, "contrib": c} for f, c in contrib if c < 0][:6],
                       "positive": [{"feature": f, "name": NAMES.get(f, f), "value": float(X.iloc[0][f]) if pd.notna(X.iloc[0][f]) else None, "contrib": c} for f, c in reversed(contrib) if c > 0][:6], "bias": 0.0}
    return float(np.mean(preds)), drivers


def forecast(scored: pd.DataFrame, snapshot: dict | None = None) -> dict:
    """回傳 {h: {pred, p_up, hist_mean, q20, q80, call, call_hit, base_hit, variant, drivers}}；模型未訓練回 {}。"""
    snap = snapshot or {}
    tn = snap.get("tx_night") or {}
    use_night = tn.get("change_pct") is not None and snap.get("phase") in ("night", "closed", "pre")
    mat = build_matrix(scored, None)
    row = mat.iloc[[-1]].copy()
    if use_night:
        row[NIGHT_FEATURE] = float(tn["change_pct"])
    out = {}
    for h in HORIZONS:
        b = M.load(f"st_h{h}_night") if use_night else None
        variant = "night" if b else "base"
        b = b or M.load(f"st_h{h}_base")
        if not b:
            continue
        pred, drivers = _predict_bundle(b, row)
        rep = b["report"]
        cal = M.apply_calibration(rep["calibration"], pred)
        t = rep["tiers"]
        call, call_hit = "中性", t.get("mid_up")
        if t.get("up_on") and pred >= t["edge_hi"]:
            call, call_hit = "偏多", t["up_hit"]
        elif t.get("dn_on") and pred <= t["edge_lo"]:
            call, call_hit = "偏空", t["dn_hit"]
        out[h] = {"pred_std": round(pred, 3), "p_up": cal["p_up"], "hist_mean": cal["hist_mean"], "q20": cal["q20"], "q80": cal["q80"], "bin": cal["bin"],
                  "base_hit": cal["base_hit"], "call": call, "call_hit": round(call_hit, 3) if call_hit is not None else None,
                  "tier_up_hit": t.get("up_hit"), "tier_dn_hit": t.get("dn_hit"), "call_cov": t.get("call_cov"),
                  "variant": variant, "model": b["chosen"], "rank_ic": rep.get("rank_ic"), "drivers": drivers,
                  "note": f"短線模型 v2 ({variant == 'night' and '含夜盤' or '不含夜盤'}‧{b['chosen']}‧OOS IC {rep.get('rank_ic')}‧叫牌命中 {t.get('call_hit')} 覆蓋 {t.get('call_cov')})"}
    return out


def load_metrics() -> dict | None:
    return M.load_json("short_term_metrics")
