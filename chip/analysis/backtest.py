"""長歷史回測與參數優化 (2010 起，FinMind 免費資料)。

- load_long()            2010~ 的加權指數/三大法人/融資融券 + 2018~ 期貨法人，套用同一套 add_features/score_frame
- factor_ic()            各因子 rank-IC (與未來 5/10/20 日報酬的排序相關)，依年份與市場狀態拆解
- event_study()          各種訊號/標籤發生後的未來報酬 vs 基準
- walk_forward()         逐年樣本外：預設權重 vs IC 權重 vs 等權重
- robust_weights()       依「跨年度方向一致性」給出保守的權重調整建議，寫入 data/tuned_weights.json 供模型讀取
- calibrate_thresholds() 以長期分位數校準門檻 (融資背離)

沒有長歷史的因子 (八大行庫、維持率、借券、PCR、大額交易人) 無法在此驗證，權重維持預設。
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from ..sources import finmind
from . import market
from .common import zscore

log = logging.getLogger(__name__)
TUNED_PATH = config.DATA_DIR / "tuned_weights.json"
REPORT_PATH = config.DATA_DIR / "backtest_report.json"
HORIZONS = (5, 10, 20)
TESTABLE = ["foreign", "trust", "dealer", "fut_foreign", "margin", "short", "volume", "trend", "reversion", "global", "fx_flow"]


# ------------------------------------------------------------------ 資料
def load_long(start: str = "2010-01-01") -> pd.DataFrame:
    price = finmind.taiex_price(start)
    inst = finmind.total_institutional(start)
    margin = finmind.total_margin(start)
    fut = finmind.tx_futures_institutional(start)
    df = price.merge(inst, on="date", how="left").merge(margin, on="date", how="left").merge(fut, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    try:
        from ..sources import global_markets as gm
        df = df.merge(gm.aligned_features(df["date"]), on="date", how="left")
    except Exception as e:  # noqa: BLE001
        log.warning("global features unavailable: %s", e)
    scored = market.score_frame(market.add_features(df))
    for h in HORIZONS:
        scored[f"fwd{h}"] = (scored["close"].shift(-h) / scored["close"] - 1) * 100
    scored["year"] = scored["date"].str[:4]
    return scored


def _ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 20:
        return np.nan
    return float(a[m].rank().corr(b[m].rank()))


# ------------------------------------------------------------------ 因子 IC
def factor_ic(scored: pd.DataFrame, horizon: int = 10) -> dict:
    """回傳 {'overall': DataFrame, 'by_year': DataFrame, 'by_state': DataFrame}"""
    fac = [f"f_{k}" for k in TESTABLE] + ["composite", "composite_smooth"]
    fwd = scored[f"fwd{horizon}"]
    overall = pd.DataFrame({
        "IC": [_ic(scored[c], fwd) for c in fac],
        "IC_5d": [_ic(scored[c], scored["fwd5"]) for c in fac],
        "IC_20d": [_ic(scored[c], scored["fwd20"]) for c in fac],
        "樣本數": [int(scored[c].notna().sum()) for c in fac],
    }, index=[market.NAMES.get(c[2:], c) for c in fac])
    by_year = pd.DataFrame({y: [_ic(g[c], g[f"fwd{horizon}"]) for c in fac] for y, g in scored.groupby("year")},
                           index=overall.index).round(3)
    yrs = by_year.loc[:, by_year.columns >= "2012"]
    overall["正IC年數比"] = ((yrs > 0).sum(axis=1) / yrs.notna().sum(axis=1)).round(2)
    overall["年IC平均"] = yrs.mean(axis=1).round(3)
    by_state = pd.DataFrame({s: [_ic(g[c], g[f"fwd{horizon}"]) for c in fac] for s, g in scored.groupby("state")},
                            index=overall.index).round(3)
    return {"overall": overall.round(3), "by_year": by_year, "by_state": by_state}


# ------------------------------------------------------------------ 事件研究
def _events(d: pd.DataFrame) -> dict[str, pd.Series]:
    z1 = zscore(d["foreign"], 60)
    m20, r20 = d["margin_pct20"], d["ret20"]
    return {
        "外資單日極端賣超 (z<-2.2) 且跌逾1.5%": (z1 < -2.2) & (d["ret1"] < -1.5),
        "外資單日極端買超 (z>2.2)": z1 > 2.2,
        "外資連賣 ≥5 日": d["foreign_streak"] <= -5,
        "外資連買 ≥5 日": d["foreign_streak"] >= 5,
        "投信連買 ≥5 日": d["trust_streak"] >= 5,
        "融資象限：散戶接刀": (m20 > 2) & (r20 < -2),
        "融資象限：追價過熱 (差>4%)": (m20 - r20 > 4),
        "融資象限：斷頭清洗": (m20 < -4) & (r20 < -3),
        "融資象限：籌碼沉澱 (漲而融資減)": (m20 - r20 < -1.5) & (r20 > 0),
        "外資期貨淨部位 一年最空 10%": d["fut_foreign_pct"] < 0.1,
        "外資期貨淨部位 一年最多 10%": d["fut_foreign_pct"] > 0.9,
        "外資現貨期貨 多方一致": d["foreign_consistency"] == 1,
        "外資現貨期貨 空方一致": d["foreign_consistency"] == -1,
        "月線負乖離 < -6%": d["bias20"] < -6,
        "月線負乖離 -6% ~ -3%": (d["bias20"] < -3) & (d["bias20"] >= -6),
        "月線正乖離 > +6%": d["bias20"] > 6,
        "月線正乖離 > +8%": d["bias20"] > 8,
        "爆量長黑": d["f_volume"] == -2,
        "價漲量增": d["f_volume"] == 1.5,
        "價跌量增": d["f_volume"] == -1.5,
        "量縮整理 (月線上)": (d["amount_5d_ratio"] < 0.8) & (d["ret5"].abs() < 2) & (d["close"] > d["ma20"]),
        "趨勢：多頭排列 (分≥1.5)": d["f_trend"] >= 1.5,
        "趨勢：空方格局 (分≤-1.5)": d["f_trend"] <= -1.5,
        "市場狀態：多頭": d["state"] == "多頭",
        "市場狀態：空頭": d["state"] == "空頭",
        "市場狀態：盤整": d["state"] == "盤整",
        "平滑綜合分 ≥ +20": d["composite_smooth"] >= 20,
        "平滑綜合分 ≤ -20": d["composite_smooth"] <= -20,
        "空頭狀態 + 平滑分 ≥ 25 + 站回月線": (d["state"] == "空頭") & (d["composite_smooth"] >= 25) & (d["close"] > d["ma20"]),
        "多頭狀態 + 平滑分 ≥ 20": (d["state"] == "多頭") & (d["composite_smooth"] >= 20),
        "多頭狀態 + 平滑分 ≤ -20": (d["state"] == "多頭") & (d["composite_smooth"] <= -20),
    }


def event_study(scored: pd.DataFrame, since: str = "2012-01-01") -> pd.DataFrame:
    d = scored[scored["date"] >= since]
    base = {h: d[f"fwd{h}"].mean() for h in HORIZONS}
    base_hit = {h: (d[f"fwd{h}"] > 0).mean() * 100 for h in HORIZONS}
    rows = [{"訊號": "全體基準", "樣本數": int(d["fwd10"].notna().sum()),
             **{f"{h}日均報酬%": round(base[h], 2) for h in HORIZONS}, **{f"{h}日勝率%": round(base_hit[h], 1) for h in HORIZONS},
             "10日超額%": 0.0, "t值(10日)": 0.0}]
    for name, mask in _events(d).items():
        g = d[mask.fillna(False)]
        n = int(g["fwd10"].notna().sum())
        if n == 0:
            continue
        ex = g["fwd10"].mean() - base[10]
        sd = g["fwd10"].std()
        t = ex / (sd / np.sqrt(n)) if n > 1 and sd and sd == sd else np.nan
        rows.append({"訊號": name, "樣本數": n,
                     **{f"{h}日均報酬%": round(g[f"fwd{h}"].mean(), 2) for h in HORIZONS},
                     **{f"{h}日勝率%": round((g[f"fwd{h}"] > 0).mean() * 100, 1) for h in HORIZONS},
                     "10日超額%": round(ex, 2), "t值(10日)": round(float(t), 2) if t == t else None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 逐年樣本外
def _combine(scored: pd.DataFrame, weights: dict) -> pd.Series:
    num = pd.Series(0.0, index=scored.index)
    den = pd.Series(0.0, index=scored.index)
    for k, w in weights.items():
        col = scored[f"f_{k}"]
        num += col.fillna(0) * w
        den += col.notna() * abs(w)
    return num / (2 * den.replace(0, np.nan)) * 100


def walk_forward(scored: pd.DataFrame, horizon: int = 10, first_test_year: int = 2015) -> pd.DataFrame:
    rows = []
    years = sorted(y for y in scored["year"].unique() if int(y) >= first_test_year)
    for y in years:
        train = scored[scored["year"] < y]
        test = scored[scored["year"] == y]
        ic = {k: _ic(train[f"f_{k}"], train[f"fwd{horizon}"]) for k in TESTABLE}
        ic_w = {k: float(np.clip((v if v == v else 0) * 40, -3, 3)) for k, v in ic.items()}   # IC 0.05 → 權重 2
        eq_w = {k: 1.0 for k in TESTABLE}
        def_w = {k: market.WEIGHTS[k] for k in TESTABLE}
        fwd = test[f"fwd{horizon}"]
        rows.append({"測試年": y, "樣本": int(fwd.notna().sum()),
                     "預設權重 IC": round(_ic(_combine(test, def_w), fwd), 3),
                     "IC權重 IC": round(_ic(_combine(test, ic_w), fwd), 3),
                     "等權重 IC": round(_ic(_combine(test, eq_w), fwd), 3),
                     "狀態自適應 IC": round(_ic(test["composite_smooth"], fwd), 3),
                     "年報酬%": round((test["close"].iloc[-1] / test["close"].iloc[0] - 1) * 100, 1)})
    df = pd.DataFrame(rows)
    if not df.empty:
        avg = {c: round(df[c].mean(), 3) for c in df.columns if c.endswith("IC")}
        df = pd.concat([df, pd.DataFrame([{"測試年": "平均", **avg}])], ignore_index=True)
    return df


# ------------------------------------------------------------------ 權重建議
def robust_weights(ic_tables: dict, horizon_label: str = "10日") -> dict:
    """依跨年度一致性給出保守調整：一致偏正 → ×1.25；年均 IC>0.02 → ×1；一致偏負 (反指標) → ×0.5；其餘 → ×0.75；無法驗證 → ×1。"""
    ov = ic_tables["overall"]
    out = {}
    for k in market.WEIGHTS:
        name = market.NAMES[k]
        if name not in ov.index or k not in TESTABLE:
            out[k] = {"multiplier": 1.0, "reason": "無長歷史，維持預設"}
            continue
        pos_ratio, mean_ic = ov.at[name, "正IC年數比"], ov.at[name, "年IC平均"]
        if pos_ratio >= 0.7 and mean_ic > 0.01:
            out[k] = {"multiplier": 1.25, "reason": f"跨年一致偏正 ({pos_ratio:.0%}, 年均 IC {mean_ic:+.3f})"}
        elif mean_ic > 0.02:
            out[k] = {"multiplier": 1.0, "reason": f"整體偏正但逐年不穩 ({pos_ratio:.0%}, 年均 IC {mean_ic:+.3f})，維持"}
        elif pos_ratio <= 0.3 and mean_ic < -0.01:
            out[k] = {"multiplier": 0.5, "reason": f"跨年一致偏負→歷史上為反指標 ({pos_ratio:.0%}, 年均 IC {mean_ic:+.3f})，降權"}
        else:
            out[k] = {"multiplier": 0.75, "reason": f"方向不穩定 ({pos_ratio:.0%}, 年均 IC {mean_ic:+.3f})，略降權"}
    return out


def calibrate_thresholds(scored: pd.DataFrame) -> dict:
    """融資背離 (融資20日% − 指數20日%) 的長期分位數，供調整象限門檻參考。"""
    div = scored["margin_div20"].dropna()
    q = div.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(2)
    return {"margin_div20_quantiles": {f"p{int(p * 100)}": float(v) for p, v in q.items()},
            "margin_div20_current_thresholds": {"追價過熱": 4, "偏浮動": 1.5, "沉澱": -1.5}}


# ------------------------------------------------------------------ 一鍵優化
def run(start: str = "2010-01-01", horizon: int = 10, write: bool = True) -> dict:
    scored = load_long(start)
    ic = factor_ic(scored, horizon)
    ev = event_study(scored)
    wf = walk_forward(scored, horizon)
    rw = robust_weights(ic)
    th = calibrate_thresholds(scored)
    report = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M"), "start": scored["date"].min(), "end": scored["date"].max(),
              "rows": len(scored), "horizon": horizon,
              "ic_overall": ic["overall"].reset_index().rename(columns={"index": "因子"}).to_dict("records"),
              "ic_by_year": ic["by_year"].reset_index().rename(columns={"index": "因子"}).to_dict("records"),
              "ic_by_state": ic["by_state"].reset_index().rename(columns={"index": "因子"}).to_dict("records"),
              "events": ev.to_dict("records"), "walk_forward": wf.to_dict("records"),
              "weights": rw, "thresholds": th}
    if write:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        TUNED_PATH.write_text(json.dumps({k: v["multiplier"] for k, v in rw.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"scored": scored, "ic": ic, "events": ev, "walk_forward": wf, "weights": rw, "thresholds": th, "report": report}


def load_report() -> dict | None:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return None
