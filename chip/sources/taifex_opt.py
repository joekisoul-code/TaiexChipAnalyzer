"""期交所 OpenAPI：臺指選擇權 (TXO) 最新一日行情 → 固定天期 ATM 隱含波動 (iv5/iv21/iv42)、偏斜、OI 牆、最大痛點。

演算法與研究 (nd_taifex_options s01_features / s02_daily，2017~2026 回測用的同一套) 一致：
- 只用一般時段 (日盤) 列；結算價 ('-'/0 視為缺)。
- 到期日：月選 YYYYMM = 第三個週三；週選 YYYYMMWn = 第 n 個週三；YYYYMMFn = 第 n 個週五；遇休市順延到下一交易日。
  n = 資料日之後到到期日的交易日數 (跳過 TWSE 休市日)，T = n / 252；只用 1 ≤ n ≤ 90 的序列。
- 隱含遠期 F：買賣權平價 K + (C − P)·e^{rT}，取 |C − P| 最小的 3 個履約價中位數。
- IV：Black-76 對結算價反推 (二分法)，只用 |ln(K/F)| < 0.25；ATM = F 兩側最近的價外賣權/買權 IV 線性內插。
- 25/10 delta 賣權/買權 IV：價外選擇權依 delta 內插。
- 固定天期：n ≥ 2 的序列依 n 做「總變異數」線性內插 (iv5/iv21/iv42)；偏斜 (p25/c25) 直接線性內插。
- OI 牆 / 最大痛點：最近到期 (n ≥ 1) 序列的完整履約價；put 牆 = F 以下 Put OI 最大履約價，call 牆 = F 以上 Call OI 最大履約價。

研究結論 (見 desk 模組)：iv5 當 sigma 的 k 日高低點區間比 ATR/EWMA 準 (pinball −6%)；OI 牆/最大痛點的支撐壓力效果另有檢定。
"""
from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
import pandas as pd

from .. import config
from ..http import cached, num
from . import twse
from .taifex import _iso, _rows

log = logging.getLogger(__name__)
R = 0.01
OPT_KEYS = ["Date", "Contract", "ContractMonth(Week)", "StrikePrice", "CallPut", "Open", "High", "Low", "Close", "Volume",
            "SettlementPrice", "OpenInterest", "BestBid", "BestAsk", "HistoricalHigh", "HistoricalLow", "TradingHalt", "TradingSession"]


# ------------------------------------------------------------------ Black-76 (與研究 lib.py 相同)
def _ncdf(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def b76(F, K, T, sig, cp, r=R):
    F, K, T, sig, cp = map(lambda x: np.asarray(x, float), (F, K, T, sig, cp))
    sT = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT ** 2) / sT
    d2 = d1 - sT
    return np.exp(-r * T) * cp * (F * _ncdf(cp * d1) - K * _ncdf(cp * d2))


def b76_delta(F, K, T, sig, cp, r=R):
    F, K, T, sig, cp = map(lambda x: np.asarray(x, float), (F, K, T, sig, cp))
    sT = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT ** 2) / sT
    return np.exp(-r * T) * np.where(cp > 0, _ncdf(d1), _ncdf(d1) - 1)


def implied_vol(price, F, K, T, cp, r=R, lo=0.01, hi=3.0, it=60):
    price, F, K, T, cp = map(lambda x: np.asarray(x, float), (price, F, K, T, cp))
    intrinsic = np.exp(-r * T) * np.maximum(cp * (F - K), 0)
    ok = (price > intrinsic + 1e-9) & (T > 0) & (price < np.where(cp > 0, F, K))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(it):
        m = 0.5 * (a + b)
        up = b76(F, K, T, m, cp, r) > price
        b, a = np.where(up, m, b), np.where(up, a, m)
    iv = 0.5 * (a + b)
    iv[~ok | (iv > hi * 0.999) | (iv < lo * 1.001)] = np.nan
    return iv


# ------------------------------------------------------------------ 資料
def daily_report() -> pd.DataFrame:
    """TXO 一般時段最新一日：date, series, strike, cp(C/P), settle, oi, volume, close, bid, ask。"""
    rows = cached("taifex:opt_daily", config.TTL_INTRADAY, lambda: _rows("DailyMarketReportOpt", OPT_KEYS))
    out = []
    for r in rows:
        if (r.get("Contract") or "").strip() != "TXO" or (r.get("TradingSession") or "").strip() != "一般":
            continue
        cp = (r.get("CallPut") or "").strip()
        out.append({"date": _iso(r["Date"]), "series": (r.get("ContractMonth(Week)") or "").strip(), "strike": num(r.get("StrikePrice")),
                    "cp": "C" if cp in ("買權", "Call", "C") else "P", "settle": num(r.get("SettlementPrice")), "oi": num(r.get("OpenInterest")),
                    "volume": num(r.get("Volume")), "close": num(r.get("Close")), "bid": num(r.get("BestBid")), "ask": num(r.get("BestAsk"))})
    return pd.DataFrame(out)


def guess_expiry(code: str) -> dt.date:
    y, m = int(code[:4]), int(code[4:6])
    first = dt.date(y, m, 1)
    days = [first + dt.timedelta(days=i) for i in range(31) if (first + dt.timedelta(days=i)).month == m]
    weds = [d for d in days if d.weekday() == 2]
    fris = [d for d in days if d.weekday() == 4]
    if len(code) == 6:
        return weds[2]
    k = int(code[7]) - 1
    lst = fris if code[6] == "F" else weds
    return lst[min(k, len(lst) - 1)]


def _trading_days_after(date: str, n: int = 95) -> tuple[list[str], bool]:
    """(交易日列表, 是否退回只跳週末)。退回時剩餘天數會高估、IV 低估 (研究 s00：09-24 iv5 −17%) → 呼叫端要標示。"""
    try:
        twse.holidays(dt.date.fromisoformat(date).year)      # 當年度休市表必須可得 (抓不到會丟例外)
        return twse.next_trading_days(date, n), False
    except Exception as e:  # noqa: BLE001
        log.warning("twse calendar fallback (weekends only): %s", e)
        d, out = dt.date.fromisoformat(date), []
        while len(out) < n:
            d += dt.timedelta(days=1)
            if d.weekday() < 5:
                out.append(d.isoformat())
        return out, True


# ------------------------------------------------------------------ 特徵
def _series_stats(g: pd.DataFrame) -> dict:
    F = float(g.F.iloc[0])
    o = g[g.otm & g.iv.notna()]
    puts, calls = o[o.cp == "P"].sort_values("strike"), o[o.cp == "C"].sort_values("strike")
    out = {"series": g.series.iloc[0], "F": F, "n": int(g.n.iloc[0]), "kind": g.kind.iloc[0], "expiry": g.expiry.iloc[0]}
    if len(puts) and len(calls):
        kp, vp = puts.strike.iloc[-1], puts.iv.iloc[-1]
        kc, vc = calls.strike.iloc[0], calls.iv.iloc[0]
        out["atm"] = float(vp if kc == kp else vp + (vc - vp) * (F - kp) / (kc - kp))
    for tgt, name, side in ((-0.25, "p25", puts), (-0.10, "p10", puts), (0.25, "c25", calls), (0.10, "c10", calls)):
        s = side.dropna(subset=["delta"]).sort_values("delta")
        if len(s) >= 2 and s.delta.min() <= tgt <= s.delta.max():
            out[name] = float(np.interp(tgt, s.delta.values, s.iv.values))
    return out


def _walls(g: pd.DataFrame, F: float) -> dict:
    oi = g.pivot_table(index="strike", columns="cp", values="oi", aggfunc="sum", observed=True).fillna(0)
    if not ({"C", "P"} <= set(oi.columns)) or oi.values.sum() <= 0:
        return {}
    K, c, p = oi.index.values, oi["C"].values, oi["P"].values
    pain = [(c * np.maximum(k - K, 0)).sum() + (p * np.maximum(K - k, 0)).sum() for k in K]
    out = {"maxpain": float(K[int(np.argmin(pain))]), "oi_c": float(c.sum()), "oi_p": float(p.sum())}
    above, below = oi[oi.index >= F], oi[oi.index <= F]
    if len(above):
        out["call_wall"], out["call_wall_oi"] = float(above["C"].idxmax()), float(above["C"].max())
        top = above["C"].sort_values(ascending=False).head(3)
        out["call_top"] = [{"strike": float(k), "oi": float(v)} for k, v in top.items()]
    if len(below):
        out["put_wall"], out["put_wall_oi"] = float(below["P"].idxmax()), float(below["P"].max())
        top = below["P"].sort_values(ascending=False).head(3)
        out["put_top"] = [{"strike": float(k), "oi": float(v)} for k, v in top.items()]
    return out


def _cm(ss: pd.DataFrame, col: str, target: int, var: bool = True) -> float | None:
    g = ss.dropna(subset=[col])
    g = g[g.n >= 2].sort_values("n")
    if g.empty:
        return None
    n, v = g.n.values.astype(float), g[col].values.astype(float)
    if target <= n[0]:
        return float(v[0])
    if target >= n[-1]:
        return float(v[-1])
    j = int(np.searchsorted(n, target))
    n1, n2, v1, v2 = n[j - 1], n[j], v[j - 1], v[j]
    if n1 == n2:
        return float(v1)
    w = (n2 - target) / (n2 - n1)
    if var:
        return float(np.sqrt((w * v1 ** 2 * n1 + (1 - w) * v2 ** 2 * n2) / target))
    return float(w * v1 + (1 - w) * v2)


def features(df: pd.DataFrame | None = None) -> dict:
    """最新一日 TXO 特徵。失敗回 {}。"""
    df = daily_report() if df is None else df
    if df is None or df.empty:
        return {}
    date = str(df["date"].max())
    df = df[(df.date == date) & df.strike.notna()].copy()
    cal, cal_fb = _trading_days_after(date, 95)
    pos = {d: i + 1 for i, d in enumerate(cal)}

    def n_of(code: str) -> tuple[int | None, str]:
        e = guess_expiry(code).isoformat()
        if e not in pos:          # 到期日休市 → 順延到下一交易日
            later = [d for d in cal if d >= e]
            e = later[0] if later else e
        return pos.get(e), e
    meta = {s: n_of(s) for s in df.series.unique() if len(s) in (6, 8)}
    df = df[df.series.isin(meta)]
    df["n"] = df.series.map(lambda s: meta[s][0])
    df["expiry"] = df.series.map(lambda s: meta[s][1])
    df = df[df.n.notna() & (df.n >= 1) & (df.n <= 90)].copy()
    df["n"] = df.n.astype(int)
    df["T"] = df.n / 252.0
    df["kind"] = np.where(df.series.str.len() == 6, "M", np.where(df.series.str.contains("F"), "F", "W"))
    live = df[df.settle.notna() & (df.settle > 0)]
    pv = live.pivot_table(index=["series", "strike"], columns="cp", values="settle", observed=True).dropna().reset_index()
    if pv.empty:
        return {}
    pv = pv.merge(live[["series", "T"]].drop_duplicates(), on="series")
    pv["absd"] = (pv.C - pv.P).abs()
    pv["Fk"] = pv.strike + (pv.C - pv.P) * np.exp(R * pv["T"])
    fwd = pv.sort_values(["series", "absd"]).groupby("series").head(3).groupby("series")["Fk"].median().rename("F")
    live = live.join(fwd, on="series").dropna(subset=["F"])
    live = live[np.abs(np.log(live.strike / live.F)) < 0.25].copy()
    cpn = np.where(live.cp == "C", 1.0, -1.0)
    live["iv"] = implied_vol(live.settle.values, live.F.values, live.strike.values, live["T"].values, cpn)
    live["otm"] = ((live.cp == "C") & (live.strike >= live.F)) | ((live.cp == "P") & (live.strike <= live.F))
    live["delta"] = b76_delta(live.F.values, live.strike.values, live["T"].values, live.iv.fillna(0.2).values, cpn)
    ss = pd.DataFrame([_series_stats(g) for _, g in live.groupby("series")])
    if ss.empty:
        return {}
    out: dict = {"date": date, "generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M"), "calendar_fallback": cal_fb}
    for t in (5, 21, 42):
        out[f"iv{t}"] = _cm(ss, "atm", t)
    # 固定天期 k (交易日) 的 ATM IV：desk 區間用 (k 日路徑高低分位 = 乘數 × ivk/√252)
    out["ivk"] = {str(k): (round(v, 5) if (v := _cm(ss, "atm", k)) is not None else None) for k in (1, 2, 3, 5, 10, 20)}
    for c in ("p25", "c25", "p10", "c10"):
        out[f"{c}_21"] = _cm(ss, c, 21, var=False)
    if out.get("p25_21") is not None and out.get("c25_21") is not None:
        out["rr25"] = out["p25_21"] - out["c25_21"]
    if out.get("iv5") is not None and out.get("iv21") is not None:
        out["ts_5_21"] = out["iv5"] - out["iv21"]
    full = df.join(fwd, on="series").dropna(subset=["F"])

    def expiry_block(x) -> dict:
        g = full[full.series == x.series]
        b = {"series": x.series, "kind": x.kind, "n": int(x.n), "expiry": x.expiry, "F": round(float(x.F), 1),
             "atm": round(float(x.atm), 4) if pd.notna(x.get("atm")) else None}
        b.update(_walls(g, float(x.F)))
        return b
    near = ss[ss.n >= 1].sort_values("n")
    if len(near):
        out["near"] = expiry_block(near.iloc[0])
        out["F_near"] = out["near"]["F"]
        for k in ("maxpain", "call_wall", "put_wall"):
            if out["near"].get(k) is not None:
                out[{"maxpain": "mp_dist", "call_wall": "cwall_dist", "put_wall": "pwall_dist"}[k]] = (
                    (out["near"][k] - out["F_near"]) / out["F_near"] if k != "put_wall" else (out["F_near"] - out["near"][k]) / out["F_near"])
    mon = ss[(ss.kind == "M") & (ss.n >= 1)].sort_values("n")
    if len(mon):
        out["month"] = expiry_block(mon.iloc[0])
    oi_c, oi_p = full.loc[full.cp == "C", "oi"].sum(), full.loc[full.cp == "P", "oi"].sum()
    out["pcr_oi"] = round(float(oi_p / oi_c * 100), 2) if oi_c else None
    out["series"] = [{"series": r.series, "kind": r.kind, "n": int(r.n), "expiry": r.expiry, "F": round(float(r.F), 1),
                      "atm": round(float(r.atm), 4) if pd.notna(r.get("atm")) else None} for _, r in ss.sort_values("n").iterrows()]
    for k in ("iv5", "iv21", "iv42", "p25_21", "c25_21", "p10_21", "c10_21", "rr25", "ts_5_21", "mp_dist", "cwall_dist", "pwall_dist"):
        if out.get(k) is not None:
            out[k] = round(float(out[k]), 5)
    return out
