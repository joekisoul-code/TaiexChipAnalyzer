"""7 個交易日趨勢閘門 (trend7)：「若未來 7 個交易日趨勢向下，不推薦買點；若趨勢向上，不推薦賣點」。

研究結論 (2010~ 資料、2014~2026 逐年擴張視窗樣本外，見 research/trend7 與兩份驗證報告)：
- 傳統趨勢規則 (收盤 < 月線、月線斜率、20 日報酬、60 日位置) 幾乎不能預測 7 日方向 (AUC 0.50~0.52，趨勢向下時 7 日均報酬仍 +0.36%)。
- 兩個 down 成分：
  (1) regime 規則：市場狀態 = 空頭 (價 < 季線且月線 < 季線) 且籌碼平滑分 < 0 → 穩健的部分 (單獨 7 日 -0.58%/日，2024~2026 仍有效)，
      在 ATR 低點買進模擬中被壓掉的買點 7 日 -1.47%。信心標 'regime'。
  (2) model：全特徵淺層 LightGBM 分類器 P(fwd7<0) (AUC ~0.53；國際盤特徵是主要來源)。驗證者指出 2024/2025 逐年 AUC < 0.5、
      單獨 down 日均報酬只有 -0.02% → 門檻由 0.50 提高到 0.52 (每日邊際隨門檻單調改善)，且信心標 'model' 讓前端可以打折。
- up 閘門：籌碼平滑分 > 15 且 p_down < 0.45 (分類器低機率端沒有校準價值)；樣本外 7 日 +1.3~1.5%、下跌機率 ~29%，
  10 年中 0 年比其餘日差 (但 +1.46% 依賴同期設計的因子規則，視為偏樂觀)。
- 誠實提醒：down 狀態在 2017/2019/2023/2024 實際 7 日報酬為正 (閘門那幾年少賺)；down 相對 flat 只有 8/12 年較差；
  組合定義在同一批 OOS 上從 ~11 種候選挑出，t 值偏樂觀；覆蓋率隨 regime 漂移 (逐年 2%~59%)。
- 「7 日」= 7 個交易日 (約 9~11 個日曆日)。
- 時間特徵 days_to_month_end/quarter_end 已改為日曆定義 (features.busdays_to_period_end)，訓練與推論一致；本模組須在該修正後訓練。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from ..analysis import backtest
from . import model as M
from .features import FEATURE_NAMES, MARKET_FEATURES, market_matrix

log = logging.getLogger(__name__)
NAME = "trend7"
HORIZON = 7
HORIZON_NOTE = "7 個交易日"
FIRST_TEST_YEAR = 2014
P_DOWN = 0.52            # 分類器 P(fwd7<0) 門檻 → down (model)；驗證者建議由 0.50 提高
CS_UP = 15.0             # 籌碼平滑分 > 15 → up (不推薦賣點)
P_UP_MAX = 0.45          # up 另需 p_down < 0.45 (模型不偏空)
SEEDS = (1, 2, 3, 4, 5)
PARAMS = dict(M.PARAMS)  # 淺層 (葉 4、min_child 200、lambda 20)，與大盤模型相同
N_BINS = 10
STATE_TEXT = {"down": "偏下", "up": "偏上", "flat": "中性"}


# ------------------------------------------------------------------ 資料 / 模型
def _target(mat: pd.DataFrame) -> pd.DataFrame:
    c = mat["close"].astype(float)
    mat["fwd7"] = (c.shift(-HORIZON) / c - 1) * 100
    mat["down7"] = (mat["fwd7"] < 0).astype(float)
    mat.loc[mat["fwd7"].isna(), "down7"] = np.nan
    mat["year"] = mat["date"].astype(str).str[:4].astype(int)
    return mat


def _fit(X: pd.DataFrame, y: pd.Series) -> list:
    import lightgbm as lgb
    return [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(X, y.astype(int)) for s in SEEDS]


def _p_down(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def _regime_mask(state: pd.Series, cs: pd.Series) -> pd.Series:
    """down 的穩健成分：空頭狀態且籌碼平滑分 < 0。"""
    return (state.astype(str) == "空頭") & (cs < 0)


def _model_mask(p: pd.Series, thr: float = P_DOWN) -> pd.Series:
    return p > thr


def _up_mask(p: pd.Series, cs: pd.Series, cs_up: float = CS_UP, p_up_max: float = P_UP_MAX) -> pd.Series:
    return (cs > cs_up) & (p < p_up_max)


def walk_forward(mat: pd.DataFrame, first_year: int = FIRST_TEST_YEAR) -> pd.DataFrame:
    """逐年擴張視窗；訓練集只用測試年之前並剔除最後 7 列 (標籤重疊)。"""
    d = mat.dropna(subset=["down7"])
    rows = []
    for y in sorted(d["year"].unique()):
        if y < first_year:
            continue
        test, train = d[d["year"] == y], d[d["year"] < y].iloc[:-HORIZON]
        if len(train) < 500 or test.empty:
            continue
        models = _fit(train[MARKET_FEATURES].astype(float), train["down7"])
        rows.append(pd.DataFrame({"date": test["date"].values, "year": y, "p_down": _p_down(models, test[MARKET_FEATURES].astype(float)),
                                  "fwd7": test["fwd7"].values, "state": test["state"].values, "composite_smooth": test["composite_smooth"].values}))
    return pd.concat(rows, ignore_index=True)


def _calibrate(oos: pd.DataFrame) -> dict:
    edges = oos["p_down"].quantile(np.linspace(0, 1, N_BINS + 1)[1:-1]).tolist()
    bins = np.digitize(oos["p_down"], edges)
    table = []
    for b in range(N_BINS):
        a = oos.loc[bins == b, "fwd7"]
        table.append({"bin": b, "n": int(len(a)), "mean7": round(float(a.mean()), 2), "p_neg7": round(float((a < 0).mean()), 3),
                      "p_lo": round(float(oos.loc[bins == b, "p_down"].min()), 3), "p_hi": round(float(oos.loc[bins == b, "p_down"].max()), 3)})
    return {"edges": [round(e, 4) for e in edges], "table": table}


def _gate_stats(oos: pd.DataFrame) -> dict:
    """三態統計 + 兩個 down 成分分開統計 (regime 規則 / 僅模型)，以便前端打折或日後停用分類器。"""
    from sklearn.metrics import roc_auc_score
    y = (oos["fwd7"] < 0).astype(int)
    auc = float(roc_auc_score(y, oos["p_down"]))
    by_year = {int(yy): round(float(roc_auc_score((g["fwd7"] < 0).astype(int), g["p_down"])), 3) for yy, g in oos.groupby("year") if g["fwd7"].lt(0).nunique() > 1}
    rm = _regime_mask(oos["state"], oos["composite_smooth"])
    mm = _model_mask(oos["p_down"])
    dm = rm | mm
    um = _up_mask(oos["p_down"], oos["composite_smooth"]) & ~dm
    state3 = np.where(dm, "down", np.where(um, "up", "flat"))

    def s(m: pd.Series) -> dict:
        g = oos[m]
        return {"n": int(len(g)), "cov": round(float(m.mean()), 3), "mean7": round(float(g["fwd7"].mean()), 3) if len(g) else None,
                "p_neg7": round(float((g["fwd7"] < 0).mean()), 3) if len(g) else None}

    def years_worse(m: pd.Series, ref: pd.Series) -> str:
        n_ok = n_all = 0
        for _, g in oos.groupby("year"):
            a, b = g[m[g.index]], g[ref[g.index]]
            if len(a) >= 10 and len(b) >= 10:
                n_all += 1
                n_ok += int(a["fwd7"].mean() < b["fwd7"].mean())
        return f"{n_ok}/{n_all}"

    down_s, flat_s, up_s = pd.Series(state3 == "down", index=oos.index), pd.Series(state3 == "flat", index=oos.index), pd.Series(state3 == "up", index=oos.index)
    recent = [yy for yy, g in oos.groupby("year") if len(g) >= 200 and yy in by_year][-2:]     # 只看完整年度 (部分年度不算)
    return {"n": int(len(oos)), "auc": round(auc, 3), "auc_by_year": by_year, "auc_years_gt_50": f"{sum(v > 0.5 for v in by_year.values())}/{len(by_year)}",
            "auc_alert": bool(len(recent) == 2 and all(by_year[yy] < 0.5 for yy in recent)), "auc_alert_years": [int(yy) for yy in recent],
            "base": s(pd.Series(True, index=oos.index)), "down": s(down_s), "up": s(up_s), "flat": s(flat_s),
            "down_regime": s(rm), "down_model_only": s(mm & ~rm), "down_both": s(mm & rm),
            "down_vs_nondown_years": years_worse(down_s, ~down_s), "down_vs_flat_years": years_worse(down_s, flat_s),
            "up_vs_flat_years_better": years_worse(flat_s, up_s),
            "down_cov_by_year": {int(yy): round(float(dm[g.index].mean()), 2) for yy, g in oos.groupby("year")},
            "down_mean7_by_year": {int(yy): round(float(g.loc[dm[g.index], "fwd7"].mean()), 2) if dm[g.index].sum() >= 10 else None for yy, g in oos.groupby("year")}}


def format_metrics(m: dict) -> str:
    """給 cli.py train 印的摘要。"""
    if not m:
        return "  trend7：無指標"
    d, u, f, r, mo = m["down"], m["up"], m["flat"], m.get("down_regime", {}), m.get("down_model_only", {})
    lines = [f"  OOS n={m['n']} AUC {m['auc']} (逐年 >0.5：{m['auc_years_gt_50']}){'  ⚠ 最近兩年 AUC 皆 <0.5，分類器成分偏弱' if m.get('auc_alert') else ''}",
             f"  down 覆蓋 {d['cov']:.1%} (n={d['n']}) 7 日均 {d['mean7']:+.2f}% 下跌率 {d['p_neg7']:.0%}｜up 覆蓋 {u['cov']:.1%} (n={u['n']}) {u['mean7']:+.2f}% / {u['p_neg7']:.0%}"
             f"｜flat {f['mean7']:+.2f}% / {f['p_neg7']:.0%}｜基準 {m['base']['mean7']:+.2f}% / {m['base']['p_neg7']:.0%}",
             f"  down 成分：regime (空頭且平滑分<0) n={r.get('n')} {r.get('mean7')}%；僅模型 (p>{P_DOWN}) n={mo.get('n')} {mo.get('mean7')}%",
             f"  逐年：down<非down {m.get('down_vs_nondown_years')}，down<flat {m.get('down_vs_flat_years')}；up>flat {m.get('up_vs_flat_years_better')}",
             "  逐年 AUC：" + " ".join(f"{y}:{v:.2f}" for y, v in m["auc_by_year"].items()),
             "  逐年 down 覆蓋：" + " ".join(f"{y}:{v:.0%}" for y, v in m["down_cov_by_year"].items())]
    return "\n".join(lines)


def train(write: bool = True, verbose: bool = True) -> dict:
    """逐年 walk-forward (指標/校準) + 全樣本最終擬合；存 trend7.pkl / trend7_metrics.json / trend7_oos.csv。"""
    scored = backtest.load_long()
    mat = _target(market_matrix(scored))
    oos = walk_forward(mat)
    rep, cal = _gate_stats(oos), _calibrate(oos)
    if verbose:
        print(format_metrics(rep))
    d = mat.dropna(subset=["down7"])
    bundle = {"models": _fit(d[MARKET_FEATURES].astype(float), d["down7"]), "features": MARKET_FEATURES, "horizon": HORIZON,
              "thresholds": {"p_down": P_DOWN, "cs_up": CS_UP, "p_up_max": P_UP_MAX}, "trained_at": dt.datetime.now(config.TZ).isoformat(),
              "train_end": str(d["date"].max()), "n_train": int(len(d)), "n_features": len(MARKET_FEATURES), "metrics": rep, "calibration": cal,
              "time_feature_def": "busdays_to_period_end (calendar Mon-Fri)"}
    if write:
        M.save(NAME, bundle)
        M.save_json(f"{NAME}_metrics", {k: v for k, v in bundle.items() if k not in ("models",)})
        oos.to_csv(M.MODEL_DIR / f"{NAME}_oos.csv", index=False)
    return bundle


# ------------------------------------------------------------------ 即時閘門
def _empty(text: str) -> dict:
    return {"available": False, "state": "flat", "confidence": "", "p_down": None, "exp_ret7": None, "reasons": [], "text": text,
            "drivers_toward_down": [], "drivers_toward_up": [], "oos": {}, "horizon_note": HORIZON_NOTE}


def trend7_gate(scored: pd.DataFrame, bundle: dict | None = None) -> dict:
    """回傳 CONTRACT 物件：{available, state ('down'|'up'|'flat'), confidence ('regime'|'model'|''), p_down, exp_ret7 (%),
    reasons[], text, drivers_toward_down[], drivers_toward_up[], oos{down_n, down_mean7, down_pneg, up_n, up_mean7, up_pneg}, horizon_note}
    另附 date, hist_p_neg7, composite_smooth, market_state, thresholds, components{regime, model}。
    模型未訓練或特徵失敗時 available=False、state='flat'。以最後一個「已收盤」列計算 (盤中不重算)。"""
    try:
        bundle = bundle or M.load(NAME)
    except Exception as e:  # noqa: BLE001
        log.warning("trend7 load: %s", e)
        bundle = None
    if not bundle:
        return _empty("trend7 模型未訓練 (python cli.py train)")
    try:
        mat = market_matrix(scored.tail(400).reset_index(drop=True))     # 400 列足夠算 120 日 z 分數；時間特徵為日曆定義，截斷不影響
        row = mat.iloc[[-1]]
        X = row[bundle["features"]].astype(float)
        p = float(_p_down(bundle["models"], X)[0])
        cal = bundle["calibration"]
        b = int(np.digitize([p], cal["edges"])[0])
        crow = cal["table"][b]
        st = str(row["state"].iloc[0])
        cs_raw = row["composite_smooth"].iloc[0]
        cs = float(cs_raw) if pd.notna(cs_raw) else 0.0
        thr = bundle.get("thresholds", {"p_down": P_DOWN, "cs_up": CS_UP, "p_up_max": P_UP_MAX})
        regime = st == "空頭" and cs < 0
        model = p > thr["p_down"]
        reasons: list[str] = []
        if regime:
            reasons.append(f"regime：市場狀態空頭 (價在季線下且月線 < 季線) 且籌碼平滑分 {cs:+.0f} < 0")
        if model:
            reasons.append(f"model：LightGBM 7 日下跌機率 {p:.0%} > {thr['p_down']:.0%} (2024~2025 樣本外偏弱，權重較低)")
        if regime:
            state, conf = "down", "regime"
        elif model:
            state, conf = "down", "model"
        elif cs > thr["cs_up"] and p < thr["p_up_max"]:
            state, conf = "up", "regime"
            reasons = [f"regime：籌碼平滑分 {cs:+.0f} > {thr['cs_up']:.0f} 且模型下跌機率 {p:.0%} < {thr['p_up_max']:.0%}"]
        else:
            state, conf = "flat", ""
        m = bundle.get("metrics", {})
        hist = m.get(state, {}) or {}
        hm, hp = hist.get("mean7"), hist.get("p_neg7")
        hist_txt = f"歷史同狀態 {HORIZON_NOTE} 均報酬 {hm:+.2f}%、下跌機率 {hp:.0%} (2014~ 樣本外 n={hist.get('n')})" if hm is not None and hp is not None else ""
        text = {"down": f"未來 {HORIZON_NOTE} 趨勢偏下 [{conf}]：不推薦買點，僅賣點/觀望 ({'；'.join(reasons)})。{hist_txt}；2017/2019/2023/2024 同狀態實際為正，閘門會少賺。",
                "up": f"未來 {HORIZON_NOTE} 趨勢偏上：不推薦賣點 ({'；'.join(reasons)})。{hist_txt}",
                "flat": f"未來 {HORIZON_NOTE} 趨勢中性 (模型下跌機率 {p:.0%}、籌碼平滑分 {cs:+.0f}、狀態 {st})，買賣點照常顯示。"}[state]
        try:
            drv = M.explain(bundle["models"], X, FEATURE_NAMES)   # 分類器 pred_contrib 為 logit 貢獻：正 = 推向下跌
            d_down, d_up = [r["name"] for r in drv["positive"][:4]], [r["name"] for r in drv["negative"][:4]]
        except Exception as e:  # noqa: BLE001
            log.debug("trend7 explain: %s", e)
            d_down, d_up = [], []
        dn, up = m.get("down", {}) or {}, m.get("up", {}) or {}
        return {"available": True, "date": str(row["date"].iloc[0]), "state": state, "confidence": conf, "p_down": round(p, 3),
                "exp_ret7": crow["mean7"], "hist_p_neg7": crow["p_neg7"], "bin": b, "composite_smooth": round(cs, 1), "market_state": st,
                "components": {"regime": bool(regime), "model": bool(model)}, "reasons": reasons, "text": text, "thresholds": thr,
                "drivers_toward_down": d_down, "drivers_toward_up": d_up,
                "oos": {"down_n": dn.get("n"), "down_mean7": dn.get("mean7"), "down_pneg": None if dn.get("p_neg7") is None else round(dn["p_neg7"] * 100, 1),
                        "up_n": up.get("n"), "up_mean7": up.get("mean7"), "up_pneg": None if up.get("p_neg7") is None else round(up["p_neg7"] * 100, 1)},
                "horizon_note": HORIZON_NOTE}
    except Exception as e:  # noqa: BLE001
        log.warning("trend7_gate: %s", e)
        return _empty(f"trend7 計算失敗 ({e})")


def load_metrics() -> dict | None:
    return M.load_json(f"{NAME}_metrics")
