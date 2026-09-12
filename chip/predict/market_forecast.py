"""大盤走勢預測：訓練 (2010~ 長歷史)、逐年驗證、即時預測 (含盤中以現價推估)。"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .. import config
from ..analysis import backtest, market
from . import model as M
from ..sources import twse
from .features import FEATURE_NAMES, MARKET_FEATURES, as_if_close, market_matrix

log = logging.getLogger(__name__)
HORIZONS = (1, 2, 3, 5, 10, 20)
DAY_HORIZONS = (1, 2, 3)
FIRST_TEST_YEAR = 2014


def train(write: bool = True) -> dict:
    scored = backtest.load_long()
    mat = market_matrix(scored)
    results = {}
    for h in HORIZONS:
        oos = M.walk_forward(mat, MARKET_FEATURES, f"fwd{h}", h, FIRST_TEST_YEAR)
        met = M.metrics(oos)
        results[h] = met
        if write:
            d = mat.dropna(subset=[f"fwd{h}"])
            models = M.fit_ensemble(d[MARKET_FEATURES], d[f"fwd{h}"])
            M.save(f"market_h{h}", {"models": models, "features": MARKET_FEATURES, "horizon": h,
                                    "trained_at": dt.datetime.now(config.TZ).isoformat(), "train_end": str(d["date"].max()),
                                    "n_train": int(len(d)), "metrics": met, "calibration": met.get("calibration")})
            oos.to_csv(M.MODEL_DIR / f"market_h{h}_oos.csv", index=False)
    if write:
        M.save_json("market_metrics", {str(h): {k: v for k, v in m.items() if k != "calibration"} for h, m in results.items()})
    return results


def _predict_row(bundle: dict, row: pd.DataFrame) -> dict:
    x = row[bundle["features"]].astype(float)
    pred = float(M.predict_ensemble(bundle["models"], x)[0])
    cal = M.apply_calibration(bundle["calibration"], pred)
    return {"pred": round(pred, 2), **cal, "drivers": M.explain(bundle["models"], x, FEATURE_NAMES)}


def forecast(scored: pd.DataFrame, snapshot: dict | None = None) -> dict:
    bundles = {h: M.load(f"market_h{h}") for h in HORIZONS}
    if any(b is None for b in bundles.values()):
        return {"error": "尚未訓練模型，請執行 python cli.py train"}
    mat = market_matrix(scored)
    last = mat.iloc[[-1]]
    out = {"date": str(last["date"].iloc[0]), "close": float(last["close"].iloc[0]), "trained_at": bundles[10]["trained_at"],
           "train_end": bundles[10]["train_end"], "horizons": {h: _predict_row(b, last) for h, b in bundles.items()}, "intraday": None}
    idx = (snapshot or {}).get("taiex") or {}
    if snapshot and snapshot.get("phase") in ("open", "pre", "post") and idx.get("last") and not snapshot.get("same_day", False):
        try:
            d2 = market.score_frame(market.add_features(as_if_close(scored, float(idx["last"]), snapshot.get("amount_projected"))))
            m2 = market_matrix(d2).iloc[[-1]]
            out["intraday"] = {"price": float(idx["last"]), "chg_pct": idx.get("chg_pct"), "time": idx.get("time"),
                               "horizons": {h: _predict_row(b, m2) for h, b in bundles.items()}}
        except Exception as e:  # noqa: BLE001
            log.warning("intraday forecast failed: %s", e)
    out["metrics"] = {h: {k: v for k, v in b.get("metrics", {}).items() if k != "calibration"} for h, b in bundles.items()}
    # 隔天 / 後天 / 第三天 (跳過休市)：預測收盤水準 = 基準價 × (1 + 同分位歷史平均%)
    base_px, base_date = out["close"], out["date"]
    src = out["horizons"]
    if out["intraday"]:
        base_px, src = out["intraday"]["price"], out["intraday"]["horizons"]
        base_date = dt.date.today().isoformat()
    try:
        days = twse.next_trading_days(base_date, 3)
    except Exception:  # noqa: BLE001
        days = [f"T+{i}" for i in (1, 2, 3)]
    out["next_days"] = []
    for i, h in enumerate(DAY_HORIZONS):
        r = src[h]
        out["next_days"].append({"n": h, "date": days[i], "label": ["隔天", "後天", "第三天"][i], "p_up": r["p_up"], "base_hit": r["base_hit"],
                                 "pred": r["pred"], "hist_mean": r["hist_mean"], "q20": r["q20"], "q80": r["q80"], "bin": r["bin"],
                                 "level": round(base_px * (1 + (r["hist_mean"] or 0) / 100), 0),
                                 "level_lo": round(base_px * (1 + (r["q20"] or 0) / 100), 0), "level_hi": round(base_px * (1 + (r["q80"] or 0) / 100), 0),
                                 "drivers": r["drivers"]})
    out["summary"] = summarize(out)
    return out


def _tone(r: dict) -> str:
    diff = (r["p_up"] or 0) - r["base_hit"]
    return "偏多" if diff >= 0.04 else "偏空" if diff <= -0.04 else "中性"


def summarize(fc: dict) -> str:
    h10, h5, h20 = fc["horizons"][10], fc["horizons"][5], fc["horizons"][20]
    txt = ""
    for nd in fc.get("next_days", []):
        txt += f"{nd['label']} {nd['date']}：{_tone(nd)}，上漲率 {nd['p_up']:.0%} (基準 {nd['base_hit']:.0%})，預估收盤 {nd['level']:,.0f} (區間 {nd['level_lo']:,.0f}~{nd['level_hi']:,.0f})。"
    txt += (f" 未來 10 日：模型期望報酬 {h10['pred']:+.2f}%，落在歷史第 {h10['bin'] + 1}/5 分位；同分位過去實際上漲率 {h10['p_up']:.0%}"
           f"（基準 {h10['base_hit']:.0%}）、平均 {h10['hist_mean']:+.2f}%、20~80% 區間 {h10['q20']:+.2f}% ~ {h10['q80']:+.2f}% → {_tone(h10)}。"
           f" 5 日 {_tone(h5)} ({h5['p_up']:.0%})；20 日 {_tone(h20)} ({h20['p_up']:.0%}，平均 {h20['hist_mean']:+.2f}%)。")
    pos = "、".join(r["name"] for r in h10["drivers"]["positive"][:3])
    neg = "、".join(r["name"] for r in h10["drivers"]["negative"][:3])
    txt += f" 推升：{pos or '無'}；壓抑：{neg or '無'}。"
    if fc.get("intraday"):
        i = fc["intraday"]["horizons"][10]
        txt += f" 盤中以現價 {fc['intraday']['price']:,.0f} 推估：10 日 {_tone(i)}，上漲率 {i['p_up']:.0%}。"
    return txt
