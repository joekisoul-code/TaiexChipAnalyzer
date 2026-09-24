"""預測模型核心 (v2，依實驗結果重設計)。

實驗結論 (2014~2026 逐年樣本外)：
- 漲跌「方向分類」(分類器 / 邏輯迴歸 / 深層 GBM) 全部輸給「永遠看漲」基準，AUC≈0.45~0.52 → 不可用。
- 淺層 LightGBM (葉數 4、min_child 200、強正則) 多種子平均的「期望報酬排序」有微弱但跨年穩定的優勢：
  10 日 rank-IC 0.07 (10/13 年為正)，20 日 0.075 (11/13 年)，預測最強五分位實際報酬約為最弱的 2~3 倍。
因此：只用回歸排序 + 以樣本外分位數做「經驗校準」，把預測值對應到歷史上同分位的實際上漲機率與報酬區間，
不用模型自己輸出的機率 (過度自信)。
"""
from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)
MODEL_DIR = config.DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=4, min_child_samples=200, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0, verbose=-1)
SEEDS = (1, 2, 3, 4, 5)
N_BINS = 5


def fit_ensemble(X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> list:
    import lightgbm as lgb
    p = params or PARAMS
    return [lgb.LGBMRegressor(random_state=s, **p).fit(X, y) for s in SEEDS]


def predict_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def contributions(models: list, x: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict(x, pred_contrib=True)[0] for m in models], axis=0)


# ------------------------------------------------------------------ walk-forward
def walk_forward(df: pd.DataFrame, features: list[str], target: str, horizon: int, first_test_year: int,
                 min_train: int = 500, params: dict | None = None) -> pd.DataFrame:
    """逐年擴張視窗；測試年前 horizon 日剔除以避免標籤重疊。回傳 OOS DataFrame(date, year, pred, actual)。"""
    d = df.dropna(subset=[target]).copy()
    d["year"] = d["date"].str[:4].astype(int)
    rows = []
    dates = np.array(sorted(d["date"].unique()))
    for y in sorted(d["year"].unique()):
        if y < first_test_year:
            continue
        test = d[d["year"] == y]
        # 2026-09-24：以「交易日」剔除測試年前 horizon 日 (舊 iloc[:-horizon] 以列計，面板資料一天數十列 → 只剔除不到 1 天，標籤跨入測試年)
        i0 = int(np.searchsorted(dates, test["date"].min()))
        cut = dates[max(0, i0 - horizon)] if horizon else test["date"].min()
        train = d[d["date"] < cut]
        if len(train) < min_train or test.empty:
            continue
        models = fit_ensemble(train[features], train[target], params)
        rows.append(pd.DataFrame({"date": test["date"].values, "year": y, "pred": predict_ensemble(models, test[features]),
                                  "actual": test[target].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ 校準與指標
def calibrate(oos: pd.DataFrame) -> dict:
    """以 OOS 預測值的分位數切 N_BINS 區，記錄每區實際：上漲率、平均、p20/p80、樣本數。"""
    edges = oos["pred"].quantile(np.linspace(0, 1, N_BINS + 1)[1:-1]).tolist()
    bins = np.digitize(oos["pred"], edges)
    table = []
    for b in range(N_BINS):
        a = oos.loc[bins == b, "actual"]
        table.append({"bin": b, "n": int(len(a)), "hit": round(float((a > 0).mean()), 3) if len(a) else None,
                      "mean": round(float(a.mean()), 2) if len(a) else None,
                      "p20": round(float(a.quantile(0.2)), 2) if len(a) else None,
                      "p80": round(float(a.quantile(0.8)), 2) if len(a) else None,
                      "pred_mean": round(float(oos.loc[bins == b, "pred"].mean()), 2) if len(a) else None})
    return {"edges": [round(e, 4) for e in edges], "table": table, "base_hit": round(float((oos["actual"] > 0).mean()), 3),
            "base_mean": round(float(oos["actual"].mean()), 2)}


def apply_calibration(cal: dict, pred: float) -> dict:
    b = int(np.digitize([pred], cal["edges"])[0])
    row = cal["table"][b]
    return {"bin": b, "p_up": row["hit"], "hist_mean": row["mean"], "q20": row["p20"], "q80": row["p80"], "n": row["n"],
            "base_hit": cal["base_hit"], "base_mean": cal["base_mean"]}


def metrics(oos: pd.DataFrame) -> dict:
    if oos.empty:
        return {}
    ic = float(oos["pred"].rank().corr(oos["actual"].rank()))
    byy = oos.groupby("year").apply(lambda g: pd.Series({
        "n": len(g), "ic": g["pred"].rank().corr(g["actual"].rank()),
        "top_q_mean": g[g["pred"] >= g["pred"].quantile(0.8)]["actual"].mean(),
        "bot_q_mean": g[g["pred"] <= g["pred"].quantile(0.2)]["actual"].mean(),
        "base_mean": g["actual"].mean()}), include_groups=False).round(3)
    cal = calibrate(oos)
    return {"n": int(len(oos)), "rank_ic": round(ic, 3), "ic_year_mean": round(float(byy["ic"].mean()), 3),
            "ic_positive_years": f"{int((byy['ic'] > 0).sum())}/{len(byy)}",
            "base_hit": cal["base_hit"], "base_mean": cal["base_mean"],
            "bin_hit": [r["hit"] for r in cal["table"]], "bin_mean": [r["mean"] for r in cal["table"]],
            "spread_top_bottom": round((cal["table"][-1]["mean"] or 0) - (cal["table"][0]["mean"] or 0), 2),
            "by_year": byy.reset_index().to_dict("records"), "calibration": cal}


def explain(models: list, x: pd.DataFrame, names: dict, top: int = 6) -> dict:
    contrib = contributions(models, x)
    rows = [{"feature": f, "name": names.get(f, f), "value": float(x.iloc[0][f]) if pd.notna(x.iloc[0][f]) else None,
             "contrib": float(c)} for f, c in zip(list(x.columns), contrib[:-1])]
    rows.sort(key=lambda r: r["contrib"])
    return {"negative": [r for r in rows if r["contrib"] < 0][:top],
            "positive": [r for r in reversed(rows) if r["contrib"] > 0][:top], "bias": float(contrib[-1])}


# ------------------------------------------------------------------ 持久化
def save(name: str, payload: dict) -> None:
    with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(payload, f)


def load(name: str) -> dict | None:
    p = MODEL_DIR / f"{name}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_json(name: str, obj) -> None:
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def load_json(name: str):
    p = MODEL_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
