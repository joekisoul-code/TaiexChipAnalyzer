"""跨市場研究：外匯／利率／波動率／原物料 對 台股、美股 (S&P500)、韓股 (KOSPI)、日股 (日經) 的影響。

對齊原則：特徵一律取「目標市場交易日 D 開盤前已知」的最後一筆 (merge_asof, 不含當日)。
- 台/韓/日股：前一晚美盤收盤的匯率/利率/商品/VIX 都已知 → lag 0。
- 美股：只能用前一美國交易日 → 等同 lag 1。
指標：同日效應 (特徵 r1 vs 目標當日報酬)、預測力 (rank-IC vs 未來 1/5/10/20 日)、逐年一致性、事件研究。
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
REPORT_PATH = config.DATA_DIR / "cross_market_report.json"
HORIZONS = (1, 5, 10, 20)
TARGETS = {"台股 TAIEX": "taiex", "美股 S&P500": "sp500", "韓股 KOSPI": "kospi", "日股 日經": "nikkei"}
GROUPS = {
    "外匯": ["usdtwd", "usdjpy", "usdkrw", "eurusd", "dxy"],
    "利率": ["us3m", "us5y", "us10y", "us30y"],
    "波動率": ["vix", "vxn", "vix3m", "skew"],
    "原物料": ["oil", "brent", "natgas", "gold", "silver", "platinum", "copper", "soybean", "corn", "bdry"],
}


def _ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank())) if m.sum() >= 30 else np.nan


def _features_for(dates: pd.Series, markets: dict) -> pd.DataFrame:
    """對齊到 dates (目標市場交易日) 的特徵：每個市場 r1/r5/r20、level (利率/波動率)，加派生：曲線、VIX 期限結構、銅金比。"""
    dts = pd.to_datetime(pd.Series(dates).astype(str))
    out: dict = {"date": dts.dt.strftime("%Y-%m-%d").values}
    lv = {}
    for key, df in markets.items():
        d = df.copy()
        d["dt"] = pd.to_datetime(d["date"])
        d = d.sort_values("dt").drop_duplicates("dt")
        d["r1"] = d["close"].pct_change() * 100
        d["r5"] = d["close"].pct_change(5) * 100
        d["r20"] = d["close"].pct_change(20) * 100
        d["d5"] = d["close"].diff(5)
        m = pd.merge_asof(pd.DataFrame({"dt": dts}).sort_values("dt"), d[["dt", "close", "r1", "r5", "r20", "d5"]], on="dt",
                          direction="backward", allow_exact_matches=False).set_index("dt").reindex(dts)
        lv[key] = m["close"].values
        for c in ("r1", "r5", "r20"):
            out[f"{key}_{c}"] = m[c].values
        if key in GROUPS["利率"] + GROUPS["波動率"]:
            out[f"{key}_level"] = m["close"].values
            out[f"{key}_d5"] = m["d5"].values
    if "us10y" in lv and "us3m" in lv:
        out["curve_10y_3m"] = lv["us10y"] - lv["us3m"]
    if "us10y" in lv and "us5y" in lv:
        out["curve_10y_5y"] = lv["us10y"] - lv["us5y"]
    if "vix" in lv and "vix3m" in lv:
        out["vix_term"] = lv["vix"] / lv["vix3m"]          # >1 = 近月恐慌 (backwardation)
    if "copper" in lv and "gold" in lv:
        cg = pd.Series(lv["copper"] / lv["gold"])
        out["copper_gold_r20"] = (cg / cg.shift(20) - 1).values * 100
    return pd.DataFrame(out)


def build() -> dict[str, pd.DataFrame]:
    markets = gm.all_markets()
    frames = {}
    for label, key in TARGETS.items():
        if key == "taiex":
            base = backtest.load_long("2007-01-01")[["date", "open", "close"]].copy()
        else:
            src = markets.get(key)
            if src is None:
                continue
            base = src[["date", "open", "close"]].copy()
        base = base.sort_values("date").reset_index(drop=True)
        base["ret1"] = base["close"].pct_change() * 100
        base["gap"] = (base["open"] / base["close"].shift(1) - 1) * 100
        for h in HORIZONS:
            base[f"fwd{h}"] = (base["close"].shift(-h) / base["close"] - 1) * 100
        feats = _features_for(base["date"], {k: v for k, v in markets.items() if k != key})
        d = base.merge(feats, on="date", how="left")
        d["year"] = d["date"].str[:4]
        frames[label] = d[d["date"] >= "2007-01-01"].reset_index(drop=True)
    return frames


def group_table(d: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    cols = [c for c in d.columns if any(c.startswith(k + "_") for k in GROUPS[group])]
    if group == "利率":
        cols += [c for c in ("curve_10y_3m", "curve_10y_5y") if c in d]
    if group == "波動率":
        cols += [c for c in ("vix_term",) if c in d]
    if group == "原物料":
        cols += [c for c in ("copper_gold_r20",) if c in d]
    yrs = sorted(y for y in d["year"].unique() if y >= "2010")
    for c in cols:
        rec = {"特徵": c, "同日相關": round(_ic(d[c], d["ret1"]), 3)}
        for h in HORIZONS:
            rec[f"IC_{h}d"] = round(_ic(d[c], d[f"fwd{h}"]), 3)
        by = [_ic(d[d["year"] == y][c], d[d["year"] == y]["fwd10"]) for y in yrs]
        by = [b for b in by if b == b]
        rec["10d 正IC年比"] = round(sum(b > 0 for b in by) / len(by), 2) if by else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_a"] = df["IC_10d"].abs()
    return df.sort_values("_a", ascending=False).drop(columns="_a").reset_index(drop=True)


def _events(d: pd.DataFrame) -> dict[str, pd.Series]:
    g = lambda c: d[c] if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    return {
        "台幣急貶 (美元/台幣 5 日 > +1%)": g("usdtwd_r5") > 1, "台幣急升 (5 日 < -1%)": g("usdtwd_r5") < -1,
        "韓元急貶 (美元/韓元 5 日 > +1.5%)": g("usdkrw_r5") > 1.5, "日圓急升 (美元/日圓 5 日 < -2%，套利平倉)": g("usdjpy_r5") < -2,
        "日圓急貶 (美元/日圓 5 日 > +2%)": g("usdjpy_r5") > 2, "美元指數 20 日 > +3%": g("dxy_r20") > 3, "美元指數 20 日 < -3%": g("dxy_r20") < -3,
        "美債 10Y 5 日升逾 20bp": g("us10y_d5") > 0.2, "美債 10Y 5 日降逾 20bp": g("us10y_d5") < -0.2,
        "殖利率曲線倒掛 (10Y-3M < 0)": g("curve_10y_3m") < 0, "VIX 期限倒掛 (VIX/VIX3M > 1)": g("vix_term") > 1,
        "SKEW > 150 (尾部風險定價高)": g("skew_level") > 150, "VXN > 30": g("vxn_level") > 30,
        "油價 20 日 > +10%": g("oil_r20") > 10, "油價 20 日 < -15%": g("oil_r20") < -15, "布蘭特單日 > +5%": g("brent_r1") > 5,
        "黃金 5 日 > +4%": g("gold_r5") > 4, "白銀 20 日 > +15%": g("silver_r20") > 15, "銅 20 日 > +8%": g("copper_r20") > 8, "銅 20 日 < -8%": g("copper_r20") < -8,
        "銅金比 20 日 > +8% (景氣偏強)": g("copper_gold_r20") > 8, "銅金比 20 日 < -8% (避險)": g("copper_gold_r20") < -8,
        "天然氣 20 日 > +25%": g("natgas_r20") > 25, "乾散貨 ETF 20 日 > +15%": g("bdry_r20") > 15, "黃豆 20 日 < -10%": g("soybean_r20") < -10,
    }


def event_table(d: pd.DataFrame) -> pd.DataFrame:
    base = {h: d[f"fwd{h}"].mean() for h in HORIZONS}
    rows = [{"事件": "全體基準", "樣本數": int(d["fwd5"].notna().sum()), "當日%": round(d["ret1"].mean(), 2),
             **{f"{h}日均報酬%": round(base[h], 2) for h in HORIZONS}, **{f"{h}日勝率%": round((d[f"fwd{h}"] > 0).mean() * 100, 1) for h in (5, 20)},
             "5日超額%": 0.0, "t值": 0.0}]
    for name, mask in _events(d).items():
        g = d[mask.fillna(False)]
        n = int(g["fwd5"].notna().sum())
        if n < 8:
            continue
        ex = g["fwd5"].mean() - base[5]
        sd = g["fwd5"].std()
        t = ex / (sd / np.sqrt(n)) if n > 1 and sd and sd == sd else np.nan
        rows.append({"事件": name, "樣本數": n, "當日%": round(g["ret1"].mean(), 2), **{f"{h}日均報酬%": round(g[f"fwd{h}"].mean(), 2) for h in HORIZONS},
                     **{f"{h}日勝率%": round((g[f"fwd{h}"] > 0).mean() * 100, 1) for h in (5, 20)}, "5日超額%": round(ex, 2), "t值": round(float(t), 2) if t == t else None})
    return pd.DataFrame(rows)


def summarize(results: dict) -> list[str]:
    """每個目標市場：各組最強 (|IC_10d| 且逐年一致) 的特徵與方向。"""
    lines = []
    for target, r in results.items():
        parts = []
        for grp, tbl in r["groups"].items():
            if tbl.empty:
                continue
            cand = tbl[(tbl["10d 正IC年比"] >= 0.7) | (tbl["10d 正IC年比"] <= 0.3)]
            top = (cand if not cand.empty else tbl).iloc[0]
            direction = "正向" if top["IC_10d"] > 0 else "反向"
            parts.append(f"{grp}：{top['特徵']} ({direction}，10 日 IC {top['IC_10d']:+.3f}，正 IC 年比 {top['10d 正IC年比']:.0%})")
        lines.append(f"{target} → " + "；".join(parts))
    return lines


def run(write: bool = True) -> dict:
    frames = build()
    results = {}
    for target, d in frames.items():
        results[target] = {"groups": {g: group_table(d, g) for g in GROUPS}, "events": event_table(d), "n": len(d),
                           "start": d["date"].min(), "end": d["date"].max()}
    summary = summarize(results)
    report = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M"), "summary": summary,
              "targets": {t: {"n": r["n"], "start": r["start"], "end": r["end"],
                              "groups": {g: tbl.to_dict("records") for g, tbl in r["groups"].items()},
                              "events": r["events"].to_dict("records")} for t, r in results.items()}}
    if write:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"results": results, "summary": summary, "report": report}


def load_report() -> dict | None:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
