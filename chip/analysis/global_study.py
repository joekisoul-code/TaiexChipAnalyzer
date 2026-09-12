"""國際市場 × 台股歷史研究：哪些海外指標對台股「開盤跳空」「開盤後走勢」「未來 1/5/10/20 日」有預測力。

方法 (2007~2026，皆以台股交易日 D 為基準，特徵為 D 開盤前已知的資料)：
1. 同日效應：前一晚美股/費半/ADR 等的日報酬 vs 台股 D 日 (a) 跳空 open/prev_close (b) 開盤後 close/open (c) 全日 ret1
2. 預測力：各特徵對台股未來 1/5/10/20 日報酬的 rank-IC 與逐年一致性
3. 極端事件：VIX>30、費半單日 -3%/+3%、比特幣 5 日 -10%、油價 20 日 +10%、美元/台幣 5 日 +1%… 之後台股表現
4. 滾動相關：近 60 日台股與各市場的相關係數 (目前連動狀態)
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from ..sources import global_markets as gm
from . import backtest

log = logging.getLogger(__name__)
REPORT_PATH = config.DATA_DIR / "global_report.json"
HORIZONS = (1, 5, 10, 20)


def _ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank())) if m.sum() >= 30 else np.nan


def build() -> pd.DataFrame:
    scored = backtest.load_long("2007-01-01")
    d = scored[["date", "open", "close", "ret1", "ret5", "ret20", "state"]].copy()
    d["gap"] = (d["open"] / d["close"].shift(1) - 1) * 100
    d["intraday"] = (d["close"] / d["open"] - 1) * 100
    for h in HORIZONS:
        d[f"fwd{h}"] = (d["close"].shift(-h) / d["close"] - 1) * 100
    g = gm.aligned_features(d["date"])
    d = d.merge(g, on="date", how="left")
    d["year"] = d["date"].str[:4]
    return d


def same_day_effect(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (sym, name, region) in gm.SYMBOLS.items():
        col = f"g_{key}_r1"
        if col not in d:
            continue
        rows.append({"市場": name, "代碼": sym, "區域": region,
                     "vs 跳空 (open/前收)": round(_ic(d[col], d["gap"]), 3),
                     "vs 開盤後 (close/open)": round(_ic(d[col], d["intraday"]), 3),
                     "vs 全日": round(_ic(d[col], d["ret1"]), 3),
                     "Pearson 全日": round(float(d[[col, "ret1"]].dropna().corr().iloc[0, 1]), 3)})
    return pd.DataFrame(rows).sort_values("vs 全日", ascending=False)


def predictive_ic(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feats = [c for c in d.columns if c.startswith("g_")]
    yrs = sorted(y for y in d["year"].unique() if y >= "2010")
    for col in feats:
        rec = {"特徵": col}
        for h in HORIZONS:
            rec[f"IC_{h}d"] = round(_ic(d[col], d[f"fwd{h}"]), 3)
        by = [_ic(d[d["year"] == y][col], d[d["year"] == y]["fwd5"]) for y in yrs]
        by = [b for b in by if b == b]
        rec["5d 正IC年比"] = round(sum(1 for b in by if b > 0) / len(by), 2) if by else np.nan
        by10 = [_ic(d[d["year"] == y][col], d[d["year"] == y]["fwd10"]) for y in yrs]
        by10 = [b for b in by10 if b == b]
        rec["10d 正IC年比"] = round(sum(1 for b in by10 if b > 0) / len(by10), 2) if by10 else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["abs10"] = df["IC_10d"].abs()
    return df.sort_values("abs10", ascending=False).drop(columns="abs10")


def _events(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "VIX > 30 (恐慌)": d["g_vix_level"] > 30,
        "VIX < 13 (極度樂觀)": d["g_vix_level"] < 13,
        "VIX 單日飆升 > 20%": d["g_vix_chg"] > 20,
        "費半前晚 < -3%": d["g_sox_r1"] < -3,
        "費半前晚 > +3%": d["g_sox_r1"] > 3,
        "費半 5 日 < -7%": d["g_sox_r5"] < -7,
        "費半 20 日 > +15%": d["g_sox_r20"] > 15,
        "台積電 ADR 前晚 < -4%": d["g_tsm_adr_r1"] < -4,
        "台積電 ADR 前晚 > +4%": d["g_tsm_adr_r1"] > 4,
        "S&P500 前晚 < -2%": d["g_sp500_r1"] < -2,
        "S&P500 前晚 > +2%": d["g_sp500_r1"] > 2,
        "S&P500 距 20 日高 < -8% (修正)": d["g_sp500_hi20"] < -8,
        "比特幣 5 日 < -10%": d["g_btc_r5"] < -10,
        "比特幣 5 日 > +15%": d["g_btc_r5"] > 15,
        "油價 20 日 > +10%": d["g_oil_r20"] > 10,
        "油價 20 日 < -15%": d["g_oil_r20"] < -15,
        "黃金 5 日 > +4% (避險)": d["g_gold_r5"] > 4,
        "美元/台幣 5 日 > +1% (台幣急貶)": d["g_usdtwd_r5"] > 1,
        "美元/台幣 5 日 < -1% (台幣急升)": d["g_usdtwd_r5"] < -1,
        "美債殖利率 5 日 > +5%": d["g_us10y_r5"] > 5,
        "日經前日 < -2.5%": d["g_nikkei_r1"] < -2.5,
        "KOSPI 前日 < -2.5%": d["g_kospi_r1"] < -2.5,
        "KOSPI 5 日 > +5%": d["g_kospi_r5"] > 5,
        "銅 20 日 > +8%": d["g_copper_r5"].rolling(4).sum() > 8,
    }


def event_study(d: pd.DataFrame) -> pd.DataFrame:
    base = {h: d[f"fwd{h}"].mean() for h in HORIZONS}
    rows = [{"事件": "全體基準", "樣本數": int(d["fwd5"].notna().sum()), "跳空%": round(d["gap"].mean(), 2), "當日%": round(d["ret1"].mean(), 2),
             **{f"{h}日均報酬%": round(base[h], 2) for h in HORIZONS}, **{f"{h}日勝率%": round((d[f"fwd{h}"] > 0).mean() * 100, 1) for h in HORIZONS},
             "5日超額%": 0.0, "t值(5日)": 0.0}]
    for name, mask in _events(d).items():
        g = d[mask.fillna(False)]
        n = int(g["fwd5"].notna().sum())
        if n < 5:
            continue
        ex = g["fwd5"].mean() - base[5]
        sd = g["fwd5"].std()
        t = ex / (sd / np.sqrt(n)) if n > 1 and sd and sd == sd else np.nan
        rows.append({"事件": name, "樣本數": n, "跳空%": round(g["gap"].mean(), 2), "當日%": round(g["ret1"].mean(), 2),
                     **{f"{h}日均報酬%": round(g[f"fwd{h}"].mean(), 2) for h in HORIZONS},
                     **{f"{h}日勝率%": round((g[f"fwd{h}"] > 0).mean() * 100, 1) for h in HORIZONS},
                     "5日超額%": round(ex, 2), "t值(5日)": round(float(t), 2) if t == t else None})
    return pd.DataFrame(rows)


def rolling_corr(d: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """台股全日報酬 與 各市場前一日報酬 的滾動相關 (最近值 + 一年前值)。"""
    rows = []
    for key, (sym, name, region) in gm.SYMBOLS.items():
        col = f"g_{key}_r1"
        if col not in d:
            continue
        rc = d["ret1"].rolling(window).corr(d[col])
        rows.append({"市場": name, f"近{window}日相關": round(float(rc.iloc[-1]), 3) if pd.notna(rc.iloc[-1]) else None,
                     "一年前": round(float(rc.iloc[-250]), 3) if len(rc) > 250 and pd.notna(rc.iloc[-250]) else None,
                     "長期平均": round(float(rc.mean()), 3), "最新值": round(float(d[col].iloc[-1]), 2) if pd.notna(d[col].iloc[-1]) else None})
    return pd.DataFrame(rows).sort_values(f"近{window}日相關", ascending=False)


def run(write: bool = True) -> dict:
    d = build()
    out = {"frame": d, "same_day": same_day_effect(d), "predictive": predictive_ic(d), "events": event_study(d), "rolling": rolling_corr(d)}
    report = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M"), "start": d["date"].min(), "end": d["date"].max(), "rows": len(d),
              "same_day": out["same_day"].to_dict("records"), "predictive": out["predictive"].to_dict("records"),
              "events": out["events"].to_dict("records"), "rolling": out["rolling"].to_dict("records")}
    if write:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out["report"] = report
    return out


def load_report() -> dict | None:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
