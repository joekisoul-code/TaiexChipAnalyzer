"""事件模組驗收回測 (design.md §7)：只讀快取 (研究 panel 的 TAIEX OHLC = FinMind 快取、NYSE 實際交易日)，walk-forward。

驗收 A：√n_US 用在 chip/predict/range_levels.py 的 base k=1「路徑帶」(ATR14+EWMA sigma × 分位乘數)，2014+ 休市後首日。
  - 帶在「休市前最後交易日 t 收盤」發布，k=1 目標 = 休市後首日 t+1 的盤中最低 / 最高 相對 t 收盤。
  - 乘數 m_q walk-forward：只用 t 之前已實現的 ratio (target 在 t 收盤前已知 → j ≤ t−1)。
      prod  : 2010-01 起擴張視窗 (與 range_levels.fit_multipliers 同口徑：所有日子，含休市後)，至少 750 列
      norm750: 前 750 個「正常日」(無休市平日、恰 1 個美股交易日)
  - R0 = 不調整；R4 = level × √max(1, n_US)。
  - 條件一：R4 的 low20 / high80 觸及率都在 [0.12, 0.28]。
  - 條件二：四分位 pinball 合計相對 R0 的變化，事件群聚 bootstrap (每個休市事件一群，10,000 次) 95% CI 上限 < 0。
  主判定用 prod (與線上乘數同口徑)；norm750 只列參考。兩條件都過 → range_levels_path.enabled = true。
驗收 B (參考，不自動開啟)：連假前一日 ×0.87、年底最後 5 日 ×0.80，同法；另分 2010 前 / 後。

輸出：accept/out/accept_A_events.csv、accept_summary.json；stdout 摘要。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# 研究 panel (daily_panel.parquet / us_calendar.csv / tw_calendar.csv，由 FinMind/Yahoo 快取建成)；以環境變數 EVENT_PANEL 指定
PANEL = Path(os.getenv("EVENT_PANEL", str(HERE.parent / "panel")))
OUT = Path(os.getenv("EVENT_ACCEPT_OUT", str(HERE / "out")))
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "chip").exists() else Path("D:/TaiexChipAnalyzer")
sys.path.insert(0, str(REPO))
from chip.predict import events as E, range_levels as RL  # noqa: E402

QS = {"low10": 0.10, "low20": 0.20, "high80": 0.80, "high90": 0.90}
RNG = np.random.default_rng(20260927)
NBOOT = 10000


def pinball(y, q, tau):
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


# ------------------------------------------------------------------ data (cache only)
tx = pd.read_parquet(PANEL / "daily_panel.parquet", filters=[("market", "==", "taiex")])
tx["date"] = pd.to_datetime(tx["date"]).dt.strftime("%Y-%m-%d")
tx = tx[tx["date"] >= "2000-01-01"].sort_values("date").reset_index(drop=True)[["date", "open", "high", "low", "close"]]
usc = pd.read_csv(PANEL / "us_calendar.csv")
E.set_us_actual(usc.loc[~usc["is_future"], "date"].astype(str))
twc = pd.read_csv(PANEL / "tw_calendar.csv")
panel_nus = dict(zip(twc["date"].astype(str), twc["n_us_sessions_since_prev_td"]))

d = tx
dates = d["date"].tolist()
N = len(d)
sig = RL.sigma_series(d).values
tg = RL.path_targets(d)
lo1, hi1 = tg["pathLow1"].values, tg["pathHigh1"].values
rlo, rhi = lo1 / sig, hi1 / sig

# 每列 t：目標日 t+1 的休市 / n_US
cwd = np.zeros(N, dtype=int)
nus = np.full(N, np.nan)
mism = []
for t in range(N - 1):
    a, b = pd.Timestamp(dates[t]), pd.Timestamp(dates[t + 1])
    cwd[t] = sum(1 for x in pd.date_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1)) if x.weekday() < 5)
    nus[t] = E.n_us_between(dates[t], dates[t + 1])
    pn = panel_nus.get(dates[t + 1])
    if pn is not None and not np.isnan(pn) and int(pn) != int(nus[t]) and dates[t + 1] >= "2008-01-01":
        mism.append((dates[t + 1], int(pn), int(nus[t])))
normal = (cwd == 0) & (nus == 1)
ok = np.isfinite(rlo) & np.isfinite(rhi) & np.isfinite(sig)
print(f"n_US 與研究 panel 不一致的日子 (2008+): {len(mism)} {mism[:5]}")


def mult_prod(t: int) -> dict | None:
    j0 = next(i for i, x in enumerate(dates) if x >= "2010-01-01")
    idx = np.arange(j0, t)            # j ≤ t−1
    idx = idx[ok[idx]]
    if len(idx) < 750:
        return None
    return {q: float(np.quantile(rlo[idx] if q.startswith("low") else rhi[idx], tau)) for q, tau in QS.items()}


def mult_norm(t: int) -> dict | None:
    idx = np.arange(0, t)
    idx = idx[ok[idx] & normal[idx]][-750:]
    if len(idx) < 750:
        return None
    return {q: float(np.quantile(rlo[idx] if q.startswith("low") else rhi[idx], tau)) for q, tau in QS.items()}


def eval_rows(sel: np.ndarray, factor_fn, variant: str) -> pd.DataFrame:
    rows = []
    for t in np.where(sel)[0]:
        m = (mult_prod if variant == "prod" else mult_norm)(int(t))
        if m is None or not ok[t]:
            continue
        f = factor_fn(int(t))
        for rule, ff in (("R0", 1.0), ("R", f)):
            r = {"issue": dates[t], "session": dates[t + 1], "cwd": int(cwd[t]), "n_us": int(nus[t]), "rule": rule, "f": ff, "variant": variant}
            loss = 0.0
            for q, tau in QS.items():
                lv = m[q] * sig[t] * ff
                y = lo1[t] if q.startswith("low") else hi1[t]
                r["touch_" + q] = float(y <= lv) if q.startswith("low") else float(y >= lv)
                loss += pinball(y, lv, tau)
            r["pinball"] = loss
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    a, b = df[df.rule == "R0"].reset_index(drop=True), df[df.rule == "R"].reset_index(drop=True)
    n = len(a)
    if n == 0:
        return {"label": label, "n": 0}
    p0, p1 = a["pinball"].values, b["pinball"].values
    rel = (p1.sum() - p0.sum()) / p0.sum()
    boots = []
    for _ in range(NBOOT):
        ix = RNG.integers(0, n, n)
        boots.append((p1[ix].sum() - p0[ix].sum()) / p0[ix].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {"label": label, "n": n, "pinball_rel": round(float(rel), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
           "touch_R0": {q: round(float(a["touch_" + q].mean()), 3) for q in QS}, "touch_R": {q: round(float(b["touch_" + q].mean()), 3) for q in QS},
           "f_mean": round(float(b["f"].mean()), 3)}
    return out


res: dict = {"data": {"source": "research panel daily_panel.parquet (FinMind TAIEX 快取) + us_calendar.csv (NYSE 實際交易日)", "first": dates[0], "last": dates[-1],
                      "n_us_mismatch_vs_panel_2008plus": len(mism)}}
# ------------------------------------------------------------------ 驗收 A
post = np.array([cwd[t] >= 1 and dates[t + 1] >= "2014-01-01" if t < N - 1 else False for t in range(N)])
res["A"] = {}
for variant in ("prod", "norm750"):
    df = eval_rows(post, lambda t: math.sqrt(max(1.0, nus[t])), variant)
    df.to_csv(OUT / f"accept_A_events_{variant}.csv", index=False)
    s_all = summarize(df, "all post-closure 2014+")
    s_ge2 = summarize(df[df.n_us >= 2], "n_US>=2")
    c1 = all(0.12 <= s_all["touch_R"][q] <= 0.28 for q in ("low20", "high80"))
    c2 = s_all["ci95"][1] < 0
    res["A"][variant] = {"all": s_all, "n_us_ge2": s_ge2, "cond1_touch_in_[0.12,0.28]": c1, "cond2_ci_upper_lt_0": c2, "pass": bool(c1 and c2)}
# 正常日參考觸及率 (prod 乘數，2014+)
nm = np.array([normal[t] and dates[t + 1] >= "2014-01-01" if t < N - 1 else False for t in range(N)])
sub = np.where(nm)[0][::5]      # 每 5 天抽 1 天 (省時)
dfn = eval_rows(np.isin(np.arange(N), sub), lambda t: 1.0, "prod")
res["A"]["normal_day_ref_touch_R0"] = {q: round(float(dfn[dfn.rule == "R0"]["touch_" + q].mean()), 3) for q in QS} | {"n": int((dfn.rule == "R0").sum())}
res["A"]["pass"] = res["A"]["prod"]["pass"]

# ------------------------------------------------------------------ 驗收 B (參考)
nxt_closed = np.array([t + 2 < N and cwd[t + 1] >= 1 for t in range(N)])     # 目標日 t+1 是連假前一日
yr_last5 = np.zeros(N, dtype=bool)
yrs = pd.Series(dates).str[:4]
for y in yrs.unique():
    ix = np.where(yrs.values == y)[0]
    if len(ix) >= 5 and int(y) < int(dates[-1][:4]):
        yr_last5[ix[-5:] - 1] = True          # 發布日 = 目標日前一天
res["B"] = {}
for name, sel0, f in (("pre_holiday_0.87", nxt_closed, 0.87), ("yearend_last5_0.80", yr_last5, 0.80)):
    res["B"][name] = {}
    for period, (a_, b_) in (("2003-2009", ("2003-01-01", "2010-01-01")), ("2010+", ("2010-01-01", "2100-01-01"))):
        sel = np.array([sel0[t] and a_ <= dates[t + 1] < b_ if t < N - 1 else False for t in range(N)])
        df = eval_rows(sel, lambda t, f=f: f, "norm750")
        s = summarize(df, f"{name} {period}")
        s["cond1_touch_in_[0.12,0.28]"] = bool(s["n"]) and all(0.12 <= s["touch_R"][q] <= 0.28 for q in ("low20", "high80"))
        s["cond2_ci_upper_lt_0"] = bool(s["n"]) and s["ci95"][1] < 0
        res["B"][name][period] = s
    res["B"][name]["pass"] = all(v["cond1_touch_in_[0.12,0.28]"] and v["cond2_ci_upper_lt_0"] for v in res["B"][name].values() if isinstance(v, dict))

(OUT / "accept_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
