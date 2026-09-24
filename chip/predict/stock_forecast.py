"""個股走勢預測：權值股樣本 (LARGE_CAPS) 2018~ 的籌碼/價量 + 大盤特徵 → 相對大盤超額報酬 (混合面板模型)。"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from ..analysis import backtest, stock as stock_an
from ..analysis.common import zscore
from ..realtime import LARGE_CAPS
from ..sources import finmind
from . import model as M
from .features import add_time_features, market_matrix

log = logging.getLogger(__name__)
START = "2018-01-01"
HORIZONS = (5, 10, 20)
FIRST_TEST_YEAR = 2021
MIN_HIT = 0.52
TIER_Q = 0.10   # 2026-09-24：叫牌 = OOS 預測前/後 10%。舊規則「5 分位命中 ≥ 基準+3pt」在 5/10/20 日從未叫多 (最高分位只 +2pt)
STOCK_FEATURES = [
    "s_foreign_z1", "s_foreign_z5", "s_foreign_z20", "s_trust_z5", "s_trust_z20", "s_dealer_z5", "s_foreign_streak", "s_trust_streak",
    "s_concentration", "s_margin_pct20", "s_margin_div20", "s_short_ratio", "s_holding_chg20", "s_holding_chg60",
    "s_ret1", "s_ret5", "s_ret20", "s_ret60", "s_bias20", "s_bias60", "s_vol_ratio", "s_vola20", "s_rs20", "s_rs60",
    "m_composite_smooth", "m_state_bull", "m_state_bear", "m_fut_foreign_pct", "m_bias20", "m_ret20", "m_foreign_z5",
    "dow", "days_to_settle", "days_to_month_end", "days_to_quarter_end",
]
STOCK_NAMES = {
    "s_foreign_z1": "外資當日 z", "s_foreign_z5": "外資 5 日 z", "s_foreign_z20": "外資 20 日 z", "s_trust_z5": "投信 5 日 z", "s_trust_z20": "投信 20 日 z",
    "s_dealer_z5": "自營商 5 日 z", "s_foreign_streak": "外資連買/賣天數", "s_trust_streak": "投信連買/賣天數", "s_concentration": "法人籌碼集中度",
    "s_margin_pct20": "融資 20 日變化%", "s_margin_div20": "融資-股價背離", "s_short_ratio": "券資比", "s_holding_chg20": "外資持股 20 日變化",
    "s_holding_chg60": "外資持股 60 日變化", "s_ret1": "當日漲跌", "s_ret5": "5 日漲跌", "s_ret20": "20 日漲跌", "s_ret60": "60 日漲跌",
    "s_bias20": "月線乖離", "s_bias60": "季線乖離", "s_vol_ratio": "量能/20 日均", "s_vola20": "20 日波動率", "s_rs20": "20 日相對大盤", "s_rs60": "60 日相對大盤",
    "m_composite_smooth": "大盤籌碼分", "m_state_bull": "大盤多頭", "m_state_bear": "大盤空頭", "m_fut_foreign_pct": "大盤外資期貨百分位",
    "m_bias20": "大盤月線乖離", "m_ret20": "大盤 20 日漲跌", "m_foreign_z5": "大盤外資 5 日 z",
    "dow": "星期", "days_to_settle": "距結算日", "days_to_month_end": "距月底", "days_to_quarter_end": "距季底",
}


def _market_env(start: str = START) -> pd.DataFrame:
    mat = market_matrix(backtest.load_long("2010-01-01"))
    env = pd.DataFrame({"date": mat["date"], "m_composite_smooth": mat["composite_smooth"], "m_state_bull": mat["state_bull"],
                        "m_state_bear": mat["state_bear"], "m_fut_foreign_pct": mat["fut_foreign_pct"], "m_bias20": mat["bias20"],
                        "m_ret20": mat["ret20"], "m_foreign_z5": mat["foreign_z5"], "m_close": mat["close"]})
    return env[env["date"] >= start].reset_index(drop=True)


def stock_frame(stock_id: str, start: str = START) -> pd.DataFrame:
    price = finmind.stock_price(stock_id, start)
    if price.empty:
        return price
    df = price
    for fn in (finmind.stock_institutional, finmind.stock_margin, finmind.stock_shareholding):
        try:
            part = fn(stock_id, start)
            if not part.empty:
                df = df.merge(part, on="date", how="left")
        except Exception as e:  # noqa: BLE001
            log.warning("%s %s: %s", stock_id, fn.__name__, e)
    d = stock_an.add_features(df.sort_values("date").reset_index(drop=True))
    c = d["close"].astype(float)
    out = pd.DataFrame({"date": d["date"], "stock_id": stock_id, "close": c})
    out["s_foreign_z1"] = zscore(d["foreign"], 60)
    out["s_foreign_z5"] = zscore(d["foreign_5d"], 60)
    out["s_foreign_z20"] = zscore(d["foreign_20d"], 120)
    out["s_trust_z5"] = zscore(d["trust_5d"], 60)
    out["s_trust_z20"] = zscore(d["trust_20d"], 120)
    out["s_dealer_z5"] = zscore(d["dealer_5d"], 60)
    out["s_foreign_streak"], out["s_trust_streak"] = d["foreign_streak"], d["trust_streak"]
    out["s_concentration"] = d["concentration"]
    out["s_margin_pct20"], out["s_margin_div20"], out["s_short_ratio"] = d["margin_pct20"], d["margin_div20"], d["short_ratio"]
    out["s_holding_chg20"] = d["holding_chg20"]
    out["s_holding_chg60"] = d["foreign_ratio"].diff(60)
    out["s_ret1"], out["s_ret5"], out["s_ret20"] = d["ret1"], d["ret5"], d["ret20"]
    out["s_ret60"] = c.pct_change(60) * 100
    out["s_bias20"] = d["bias20"]
    out["s_bias60"] = (c / d["ma60"] - 1) * 100
    out["s_vol_ratio"] = d["vol_ratio"]
    out["s_vola20"] = d["ret1"].rolling(20).std()
    return out


def _with_env(f: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    d = f.merge(env, on="date", how="left")
    d["s_rs20"] = d["s_ret20"] - d["m_ret20"]
    d["s_rs60"] = d["s_ret60"] - d.groupby("stock_id")["m_close"].transform(lambda s: s.pct_change(60) * 100)
    return add_time_features(d)


def build_panel(universe: list[str] | None = None) -> pd.DataFrame:
    env = _market_env()
    frames = []
    for sid in universe or LARGE_CAPS:
        try:
            f = stock_frame(sid)
            if not f.empty:
                frames.append(f)
        except Exception as e:  # noqa: BLE001
            log.warning("stock %s failed: %s", sid, e)
    panel = _with_env(pd.concat(frames, ignore_index=True), env)
    panel = panel[panel["close"] > 0].copy()
    for h in HORIZONS:
        fwd_s = panel.groupby("stock_id")["close"].transform(lambda s: (s.shift(-h) / s - 1) * 100)
        fwd_m = panel.groupby("stock_id")["m_close"].transform(lambda s: (s.shift(-h) / s - 1) * 100)
        panel[f"fwd{h}"], panel[f"xfwd{h}"] = fwd_s, fwd_s - fwd_m
        # 除權息/資料異常造成的極端值不進訓練
        panel.loc[panel[f"xfwd{h}"].abs() > 40, [f"fwd{h}", f"xfwd{h}"]] = np.nan
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _tiers(oos: pd.DataFrame) -> dict:
    """OOS 預測前/後 TIER_Q 的門檻與命中 (跑贏/落後大盤)；逐年最低一併記錄。"""
    if oos.empty:
        return {}
    hi, lo = float(oos["pred"].quantile(1 - TIER_Q)), float(oos["pred"].quantile(TIER_Q))
    up, dn = oos[oos["pred"] >= hi], oos[oos["pred"] <= lo]
    base = float((oos["actual"] > 0).mean())
    uy = up.groupby("year")["actual"].apply(lambda s: (s > 0).mean()); dy = dn.groupby("year")["actual"].apply(lambda s: (s < 0).mean())
    t = {"hi": round(hi, 4), "lo": round(lo, 4), "base_up": round(base, 3),
         "up_hit": round(float((up["actual"] > 0).mean()), 3), "up_mean": round(float(up["actual"].mean()), 2), "up_n": int(len(up)), "up_yr_min": round(float(uy.min()), 3),
         "dn_hit": round(float((dn["actual"] < 0).mean()), 3), "dn_mean": round(float(dn["actual"].mean()), 2), "dn_n": int(len(dn)), "dn_yr_min": round(float(dy.min()), 3)}
    t["up_on"] = bool(t["up_hit"] >= max(base + 0.03, MIN_HIT))   # 需同時高於基準 3pt 且 ≥52% (20 日跑贏 49.9% 雖高於基準 46.5%，顯示「跑贏 50%」無意義)
    t["dn_on"] = bool(t["dn_hit"] >= max((1 - base) + 0.03, MIN_HIT))
    return t


COMBO_HS = (5, 10)
COMBO_MKT_Q = 0.70   # 大盤模型分數前 30% = 大盤看多


def _combo(panel: pd.DataFrame, oos: pd.DataFrame, h: int, tiers: dict) -> dict:
    """股價偏漲 (2026-09-24)：個股「跑贏大盤」且大盤模型看多 → 個股股價本身上漲。兩邊都是走動式 OOS。
    研究 (門檻只用先前年份)：5 日上漲 66.0% (年低 58%)、平均 +4.2%；10 日 64.8% (年低 56%)、+5.6%；全體權值股 52~53%。
    反向「跑輸大盤 + 大盤看空」股價下跌只有 53~55% → 不做。"""
    from .features import MARKET_FEATURES, market_matrix
    mo = M.walk_forward(market_matrix(backtest.load_long("2010-01-01")), MARKET_FEATURES, f"fwd{h}", h, 2014).rename(columns={"pred": "mpred"})
    m_hi = float(mo["mpred"].quantile(COMBO_MKT_Q))
    d = panel.dropna(subset=[f"xfwd{h}"]); d = d[d["date"].str[:4].astype(int) >= FIRST_TEST_YEAR]
    o = oos.assign(abs=d[f"fwd{h}"].values).merge(mo[["date", "mpred"]], on="date", how="left")
    g = o[(o["pred"] >= tiers["hi"]) & (o["mpred"] >= m_hi)]
    if len(g) < 200:
        return {}
    by = (g["abs"] > 0).groupby(g["year"]).mean()
    return {"m_hi": round(m_hi, 4), "up_hit": round(float((g["abs"] > 0).mean()), 3), "up_mean": round(float(g["abs"].mean()), 2), "n": int(len(g)),
            "yr_min": round(float(by.min()), 3), "base_up": round(float((o["abs"] > 0).mean()), 3)}


def train(write: bool = True) -> dict:
    panel = build_panel()
    results = {}
    for h in HORIZONS:
        oos = M.walk_forward(panel, STOCK_FEATURES, f"xfwd{h}", h, FIRST_TEST_YEAR, min_train=3000)
        met = M.metrics(oos)
        met["tiers"] = _tiers(oos)
        if h in COMBO_HS and met["tiers"].get("up_on"):
            try:
                met["combo"] = _combo(panel, oos, h, met["tiers"])
            except Exception as e:  # noqa: BLE001
                log.warning("combo h%s: %s", h, e)
        results[h] = met
        if write:
            d = panel.dropna(subset=[f"xfwd{h}"])
            models = M.fit_ensemble(d[STOCK_FEATURES], d[f"xfwd{h}"])
            M.save(f"stock_h{h}", {"models": models, "features": STOCK_FEATURES, "horizon": h, "trained_at": dt.datetime.now(config.TZ).isoformat(),
                                   "train_end": str(d["date"].max()), "n_train": int(len(d)), "universe": sorted(panel["stock_id"].unique()),
                                   "metrics": met, "calibration": met.get("calibration")})
    if write:
        M.save_json("stock_metrics", {str(h): {k: v for k, v in m.items() if k != "calibration"} for h, m in results.items()})
    return results


def forecast(stock_id: str, market: dict | None = None) -> dict:
    """market：大盤 market_forecast 的 horizons (取 pred)，給定時計算「股價偏漲」組合訊號。"""
    bundles = {h: M.load(f"stock_h{h}") for h in HORIZONS}
    if any(b is None for b in bundles.values()):
        return {"error": "尚未訓練個股模型，請執行 python cli.py train --stock"}
    f = stock_frame(stock_id)
    if f.empty:
        return {"error": f"無 {stock_id} 資料"}
    d = _with_env(f, _market_env()).replace([np.inf, -np.inf], np.nan)
    last = d.iloc[[-1]]
    out = {"stock_id": stock_id, "date": str(last["date"].iloc[0]), "close": float(last["close"].iloc[0]), "horizons": {}}
    for h, b in bundles.items():
        x = last[b["features"]].astype(float)
        pred = float(M.predict_ensemble(b["models"], x)[0])
        t = (b.get("metrics") or {}).get("tiers") or {}
        call, call_hit = "中性", None
        if t.get("up_on") and pred >= t["hi"]:
            call, call_hit = "偏多", t["up_hit"]
        elif t.get("dn_on") and pred <= t["lo"]:
            call, call_hit = "偏空", t["dn_hit"]
        out["horizons"][h] = {"pred": round(pred, 2), **M.apply_calibration(b["calibration"], pred), "call": call, "call_hit": call_hit,
                              "drivers": M.explain(b["models"], x, STOCK_NAMES)}
        cb = (b.get("metrics") or {}).get("combo") or {}
        mr = (market or {}).get(h) or (market or {}).get(str(h)) or {}
        if cb and call == "偏多" and mr.get("pred") is not None:
            ok = float(mr["pred"]) >= cb["m_hi"]
            out["horizons"][h]["abs_call"] = "股價偏漲" if ok else None
            out["horizons"][h]["abs_hit"] = cb["up_hit"] if ok else None
            out["horizons"][h]["abs_mean"] = cb["up_mean"] if ok else None
    out["metrics"] = {h: {k: v for k, v in b.get("metrics", {}).items() if k != "calibration"} for h, b in bundles.items()}
    out["in_universe"] = stock_id in bundles[10].get("universe", [])
    h10 = out["horizons"][10]
    out["summary"] = (f"未來 10 日相對大盤：模型期望超額 {h10['pred']:+.2f}%，落在歷史第 {h10['bin'] + 1}/5 分位；"
                      f"同分位過去跑贏大盤機率 {h10['p_up']:.0%}（基準 {h10['base_hit']:.0%}），平均超額 {h10['hist_mean']:+.2f}%，"
                      f"區間 {h10['q20']:+.2f}% ~ {h10['q80']:+.2f}%。"
                      + ("" if out["in_universe"] else " 此股不在訓練樣本 (權值股) 內，屬外推，僅供參考。"))
    return out
