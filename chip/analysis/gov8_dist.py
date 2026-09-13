"""八大行庫 (官股) 價位別籌碼分布 (volume-at-price 式) 與「主力狀態」評估。

研究結論 (research/gov8dist, 2026-03-17~09-11, 124 個交易日)：
- 籌碼分布、成本線、支撐/供給區是「描述性資訊」：官股成本在哪、現價之下有多少買進量、哪些價位官股曾大量加碼/減碼。
- 買點規則 (低於成本且買超 / 在累積區且買超 …) 的回測報酬與「多頭中逢跌買」無法區分：官股買超與指數下跌日 94% 重疊，
  樣本 (極端多頭) 無法分離官股資訊；因此 current 只是「狀態描述」(官股逢跌加碼中 / 官股高檔調節 / 官股套牢)，不是交易訊號。
- RULE_EVIDENCE 是樣本內、訊號日 t 出現後「次日開盤」進場的報酬 (八大行庫資料盤後才公布，收盤進場不可執行)，僅供顯示證據強弱。

分桶：指數固定 250 點；個股 = 現價 × 1% 取 2 位有效數字 (每日重算)。
存量：簡單累計淨額 (不假設官股先賣舊部位)；FIFO 剩餘部位成本只作第二參考。
分割保護：FinMind 個股價未還原分割，|ret1| > 40% 之前的價格與籌碼全部截掉 (分割前後張數基礎不同，不能混用)。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from collections import deque

import numpy as np
import pandas as pd

from .. import config
from ..http import _cache_path
from ..sources import histock
from .common import streak

log = logging.getLogger(__name__)

DIST_WINDOWS = (20, 60, 120, None)          # None = 全部歷史
DEFAULT_ZONE_WINDOW = "60"
TOP_N = 3
SPLIT_JUMP = 0.40                            # |ret1| > 40% 視為分割/合併，之前的資料截掉
MARKET_BUCKET = 250.0
MARKET_TOL = 0.01
STOCK_TOL = 0.02
STOCK_BUCKET_PCT = 0.01
MIN_HIST = 20
LOOKBACK = 60
SCAN_ROWS = 250                              # 規則逐日計算只掃最後 N 列 (last_fired 用)，避免歷史累積後 O(n×window)


# ------------------------------------------------------------------ 分桶
def bucket_width(ref_price: float, pct: float | None = None, fixed: float | None = None) -> float:
    """個股：現價 × pct (預設 1%) 取 2 位有效數字 (2330 → 24, 00631L → 0.36)；指數：固定點數。"""
    if fixed:
        return float(fixed)
    w = float(ref_price) * float(pct or STOCK_BUCKET_PCT)
    if not np.isfinite(w) or w <= 0:
        return 1.0
    e = math.floor(math.log10(w))
    base = 10 ** (e - 1)
    return float(round(round(w / base) * base, max(0, 1 - e)))


def _level(px: pd.Series, width: float) -> pd.Series:
    return (np.floor(px.astype(float) / width) * width).round(6)


def _lv(v) -> float:
    """價位桶輸出：去掉浮點雜訊 (0.29 桶 → 26.39 而非 26.389999999999997)。"""
    return round(float(v), 6)


def _profile(t: pd.DataFrame, width: float) -> pd.DataFrame:
    """每個價位桶的買/賣/淨額與天數 (簡單累計 = 存量邏輯)。t 需有 close, flow。"""
    f = t["flow"].astype(float)
    g = pd.DataFrame({"level": _level(t["close"], width), "buy": f.clip(lower=0), "sell": f.clip(upper=0), "net": f, "days": 1})
    return g.groupby("level").sum().sort_index()


def _fifo(t: pd.DataFrame, width: float) -> dict:
    """FIFO：賣出先沖銷最早的買進；回傳剩餘存量 (依買進價位桶) 與均價。oversold = 賣量超過可追蹤買量 (歷史起點前的老部位)。"""
    lots: deque = deque()
    oversold = 0.0
    for px, f in zip(t["close"].astype(float), t["flow"].astype(float)):
        if f > 0:
            lots.append([px, f])
        elif f < 0:
            rem = -f
            while rem > 0 and lots:
                if lots[0][1] <= rem:
                    rem -= lots[0][1]
                    lots.popleft()
                else:
                    lots[0][1] -= rem
                    rem = 0.0
            oversold += rem
    inv, cost = pd.Series(dtype=float), None
    if lots:
        d = pd.DataFrame(list(lots), columns=["px", "amt"])
        inv = d.groupby(_level(d["px"], width))["amt"].sum().sort_index()
        cost = float((d["px"] * d["amt"]).sum() / d["amt"].sum())
    return {"inventory": round(float(inv.sum()), 2) if len(inv) else 0.0, "oversold": round(oversold, 2),
            "cost": round(cost, 2) if cost else None, "levels": [{"level": _lv(k), "amt": round(float(v), 2)} for k, v in inv.items()]}


def _rows(d: pd.DataFrame, asc: bool, top_n: int) -> list[dict]:
    d = d.sort_values("net", ascending=asc).head(top_n)
    return [{"level": _lv(k), "net": round(float(r["net"]), 2), "buy": round(float(r["buy"]), 2), "sell": round(float(r["sell"]), 2), "days": int(r["days"])}
            for k, r in d.iterrows()]


def _window_view(t: pd.DataFrame, width: float, px: float, top_n: int = TOP_N) -> dict:
    prof = _profile(t, width)
    f = t["flow"].astype(float)
    c = t["close"].astype(float)
    buy = f.clip(lower=0)
    cost = float((buy * c).sum() / buy.sum()) if buy.sum() > 0 else None
    net_sum = float(f.sum())
    cost_net = float((f * c).sum() / net_sum) if net_sum > 0 else None
    cur = _lv(np.floor(px / width) * width)
    below, above = prof[prof.index < cur], prof[prof.index > cur]
    buy_tot = float(prof["buy"].sum())
    net_pos, net_neg = prof[prof["net"] > 0], prof[prof["net"] < 0]
    top = prof.loc[prof["net"].abs().sort_values(ascending=False).index].head(top_n)
    return {
        "days": int(len(t)), "from": str(t["date"].iloc[0]), "to": str(t["date"].iloc[-1]),
        "cost": round(cost, 2) if cost else None, "cost_net": round(cost_net, 2) if cost_net else None,
        "vs_cost_pct": round((px / cost - 1) * 100, 2) if cost else None,
        "buy_total": round(buy_tot, 2), "sell_total": round(float(prof["sell"].sum()), 2), "net_total": round(net_sum, 2),
        "share_below_pct": round(float(below["buy"].sum() / buy_tot * 100), 1) if buy_tot > 0 else None,
        "share_above_pct": round(float(above["buy"].sum() / buy_tot * 100), 1) if buy_tot > 0 else None,
        "net_below": round(float(below["net"].sum()), 2), "net_above": round(float(above["net"].sum()), 2),
        "support": _rows(net_pos[net_pos.index <= cur], False, top_n),      # 現價以下 (含當前桶) 淨買最大 = 主力支撐候選
        "overhang": _rows(net_pos[net_pos.index > cur], False, top_n),      # 現價以上淨買區 = 主力套牢/防守區
        "supply": _rows(net_neg[net_neg.index >= cur], True, top_n),        # 現價以上 (含) 淨賣區 = 主力供給/調節區
        "dump_below": _rows(net_neg[net_neg.index < cur], True, top_n),     # 現價以下淨賣區 (主力曾減碼)
        "top_zones": [{"level": _lv(k), "net": round(float(r["net"]), 2), "days": int(r["days"])} for k, r in top.iterrows()],
        "rows": [{"level": _lv(k), "buy": round(float(r["buy"]), 2), "sell": round(float(r["sell"]), 2), "net": round(float(r["net"]), 2), "days": int(r["days"])}
                 for k, r in prof.iterrows()],
    }


def _cut_splits(df: pd.DataFrame) -> pd.DataFrame:
    """|ret1| > SPLIT_JUMP (分割/合併) 之前的資料截掉；價位分桶與成本不能混用分割前後的價格與張數。"""
    r = df["close"].astype(float).pct_change().abs()
    if (r > SPLIT_JUMP).any():
        df = df.loc[r[r > SPLIT_JUMP].index[-1]:]
    return df.reset_index(drop=True)


def _prep(df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    t = df.dropna(subset=[flow_col, "close"]).copy()
    t["date"] = t["date"].astype(str).str[:10]
    t = t.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    t = _cut_splits(t)
    t["flow"] = t[flow_col].astype(float)
    return t


def _distribution(df: pd.DataFrame, flow_col: str, width: float, windows=DIST_WINDOWS, unit: str = "") -> dict:
    t = _prep(df, flow_col)
    if t.empty:
        return {}
    px = float(t["close"].iloc[-1])
    out = {"as_of": str(t["date"].iloc[-1]), "price": px, "bucket": width, "unit": unit,
           "history_days": int(len(t)), "history_from": str(t["date"].iloc[0]), "windows": {}}
    for w in windows:
        if w is not None and len(t) < w:      # 只標示真正存在的視窗：62 天的歷史不會出現 "120"
            continue
        out["windows"]["all" if w is None else str(w)] = _window_view(t.tail(w) if w else t, width, px)
    out["fifo"] = _fifo(t, width)
    return out


def distribution(hist: pd.DataFrame, bucket: float = MARKET_BUCKET, windows=DIST_WINDOWS) -> dict:
    """全市場：hist = DataFrame[date, gov8_net (億), close (加權指數, FinMind)]。"""
    return _distribution(hist, "gov8_net", float(bucket), windows, unit="億")


def stock_distribution(flows: pd.DataFrame, bucket_pct: float = STOCK_BUCKET_PCT, windows=DIST_WINDOWS) -> dict:
    """個股：flows = DataFrame[date, gov8_lots (張), close]；桶寬 = 現價 × bucket_pct (每日以當天現價重算)。"""
    t = flows.dropna(subset=["gov8_lots", "close"]).sort_values("date")
    if t.empty:
        return {}
    return _distribution(flows, "gov8_lots", bucket_width(float(t["close"].iloc[-1]), pct=bucket_pct), windows, unit="張")


# ------------------------------------------------------------------ 規則 (point-in-time：第 t 日只用 ≤ t 的資料)
RULES = {
    "A_zone_buy": "價格落在主力累積區 (近{w}日淨買前{n}桶) ±{tol:.0f}% 且當日八大行庫買超",
    "A2_zone_str2": "價格落在主力累積區 ±{tol:.0f}% 且八大行庫連買≥2 日",
    "B_below_cost": "價格低於八大行庫 20 日成本 且當日買超 (護成本)",
    "B2_below_cost60": "價格低於八大行庫 60 日成本 且當日買超",
    "C_defend_zone": "護盤事件 (連買≥3 且 5 日跌) 且價格在累積區 ±{tol:.0f}%",
    "S_above_cost_sell": "[賣訊] 價格高於 20 日成本 且當日賣超",
    "S2_supply_sell": "[賣訊] 價格落在主力調節區 (淨賣前{n}桶) ±{tol:.0f}% 且當日賣超",
}
BUY_RULES = ("A_zone_buy", "A2_zone_str2", "B_below_cost", "B2_below_cost60", "C_defend_zone")
SELL_RULES = ("S_above_cost_sell", "S2_supply_sell")
# 樣本內證據 (TAIEX 2026-04-15~09-11, 105 個可評估日)：訊號日 t、次日開盤進場 (八大行庫資料盤後公布，收盤進場不可執行)。
# 不是預期報酬：官股買超與指數下跌日 94% 重疊，樣本無法區分官股資訊與「多頭中逢跌買」；B2/C 只有 1~5 個事件群，不顯示。
EVIDENCE_BASIS = "signal at t, entry next open; 2026-04~09 bull sample, n small"
EVIDENCE_CAVEAT = ("gov8 買超與下跌日 94% 重疊，樣本無法區分官股資訊與逢跌買；不含官股資訊的對照 (收盤<MA20 且下跌日) 報酬相當。"
                   "2010~2025 長期逢跌買對照每 20 日只多約 +0.3 個百分點，2011/2015/2020/2022 為負")
LONGRUN_DIPBUY_PP20 = 0.3
RULE_EVIDENCE = {
    "A_zone_buy": {"n": 22, "episodes": 8, "fwd5": 2.15, "fwd10": 2.91, "fwd20": 3.35,
                   "base": {"label": "官股買超但不在累積區", "fwd5": 2.10, "fwd10": 2.61, "fwd20": 4.03},
                   "note": "≈ 官股買超日基準，區位本身無加分"},
    "B_below_cost": {"n": 14, "episodes": 4, "neg_episodes": 1, "fwd5": 3.83, "fwd10": 3.86, "fwd20": 6.20,
                     "base": {"label": "同期任一日", "fwd5": 1.12, "fwd10": 2.19, "fwd20": 3.78},
                     "note": "14 日全是下跌日；4 個事件群 1 個為負，等於多頭逢跌買"},
    "S_above_cost_sell": {"n": 41, "episodes": 5, "fwd5": 0.40, "fwd10": 2.01, "fwd20": 4.07,
                          "base": {"label": "同期任一日", "fwd5": 1.12, "fwd10": 2.19, "fwd20": 3.78},
                          "note": "僅 5 日略弱於基準，20 日反而更強，多頭中不是賣點"},
}


def fwd_returns(df: pd.DataFrame, horizons=(5, 10, 20), entry: str = "next_open") -> pd.DataFrame:
    """回測用前向報酬 (%)：訊號在第 t 日出現，進場一律落後一天 (八大行庫資料盤後才公布)。
    entry='next_open' 用 open[t+1] (需 open 欄)，'next_close' 用 close[t+1]；報酬 = close[t+1+k] / entry - 1。"""
    if entry not in ("next_open", "next_close"):
        raise ValueError("entry must be 'next_open' or 'next_close'")
    c = df["close"].astype(float)
    e = df["open"].astype(float).shift(-1) if entry == "next_open" and "open" in df else c.shift(-1)
    out = pd.DataFrame({"date": df["date"]})
    for k in horizons:
        out[f"fwd{k}"] = (c.shift(-(k + 1)) / e - 1) * 100
    return out


def _near(px: float, levels: list, width: float, tol: float):
    """px 落在某桶中心 (level + width/2) 的 ±tol 內 → 回傳最近的桶，否則 None。"""
    best = None
    for lv in levels:
        c = lv + width / 2
        if abs(px / c - 1) <= tol and (best is None or abs(px - c) < abs(px - (best + width / 2))):
            best = lv
    return best


def signal_frame(df: pd.DataFrame, flow_col: str, width: float, zone_window: int = 60, tol: float = MARKET_TOL,
                 top_n: int = TOP_N, min_hist: int = MIN_HIST, scan: int | None = None) -> pd.DataFrame:
    """逐日規則布林欄 + cost20/cost60/zone/supply_zone，全部 point-in-time (第 i 日只看 ≤ i 的資料)。
    scan=N 只計算最後 N 列 (成本與區位仍用完整歷史的 tail)。"""
    t = _prep(df, flow_col)
    if t.empty:
        return pd.DataFrame()
    c = t["close"].astype(float)
    t["ret5"] = c.pct_change(5) * 100
    t["streak"] = streak(t["flow"])
    start = max(min_hist - 1, len(t) - scan) if scan else 0
    rows = []
    for i in range(start, len(t)):
        px = float(c.iloc[i])
        rec = {"date": t["date"].iloc[i], "close": px, "flow": float(t["flow"].iloc[i]), "streak": int(t["streak"].iloc[i]),
               "ret5": float(t["ret5"].iloc[i]) if pd.notna(t["ret5"].iloc[i]) else np.nan,
               "cost20": np.nan, "cost60": np.nan, "zone": np.nan, "supply_zone": np.nan, **{r: False for r in RULES}}
        if i + 1 >= min_hist:
            hist_i = t.iloc[: i + 1]
            w20, w60 = hist_i.tail(20), hist_i.tail(60)
            b20, b60 = w20["flow"].clip(lower=0), w60["flow"].clip(lower=0)
            if b20.sum() > 0:
                rec["cost20"] = float((b20 * w20["close"].astype(float)).sum() / b20.sum())
            if b60.sum() > 0:
                rec["cost60"] = float((b60 * w60["close"].astype(float)).sum() / b60.sum())
            prof = _profile(hist_i.tail(zone_window), width)
            acc = prof[prof["net"] > 0].sort_values("net", ascending=False).head(top_n)
            dist = prof[prof["net"] < 0].sort_values("net").head(top_n)
            zone, szone = _near(px, list(acc.index), width, tol), _near(px, list(dist.index), width, tol)
            if zone is not None:
                rec["zone"] = zone
            if szone is not None:
                rec["supply_zone"] = szone
            buy, sell = rec["flow"] > 0, rec["flow"] < 0
            rec["A_zone_buy"] = zone is not None and buy
            rec["A2_zone_str2"] = zone is not None and rec["streak"] >= 2
            rec["B_below_cost"] = bool(buy and pd.notna(rec["cost20"]) and px < rec["cost20"])
            rec["B2_below_cost60"] = bool(buy and pd.notna(rec["cost60"]) and px < rec["cost60"])
            rec["C_defend_zone"] = bool(rec["streak"] >= 3 and pd.notna(rec["ret5"]) and rec["ret5"] < 0 and zone is not None)
            rec["S_above_cost_sell"] = bool(sell and pd.notna(rec["cost20"]) and px > rec["cost20"])
            rec["S2_supply_sell"] = szone is not None and sell
        rows.append(rec)
    out = pd.DataFrame(rows)
    for r in RULES:
        out[r] = out[r].astype(bool)
    return out


def _fmt(v: float, unit: str) -> str:
    return f"{v:,.0f}" if unit == "億" or v >= 100 else f"{v:,.2f}"


def buy_points(df: pd.DataFrame, flow_col: str = "gov8_net", width: float = MARKET_BUCKET, tol: float = MARKET_TOL,
               zone_window: int = 60, lookback: int = LOOKBACK, unit: str = "億") -> dict:
    """{'points': 近 lookback 日觸發 [{date, level, close, rule, side, note}], 'current': 主力狀態 (描述，非交易訊號)}。
    全市場 flow_col='gov8_net', width=250, tol=1%；個股 flow_col='gov8_lots', width=bucket_width(px, 1%), tol=2%。"""
    sf = signal_frame(df, flow_col, width, zone_window=zone_window, tol=tol, scan=max(lookback, SCAN_ROWS))
    if sf.empty:
        return {"points": [], "current": {}}
    pts = []
    for _, r in sf.tail(lookback).iterrows():
        for rule in BUY_RULES + SELL_RULES:
            if not r[rule]:
                continue
            key = "supply_zone" if rule.startswith("S") else "zone"
            lvl = r[key] if pd.notna(r[key]) else r["close"]
            note = RULES[rule].format(w=zone_window, n=TOP_N, tol=tol * 100)
            if rule.startswith("B") and pd.notna(r["cost20"]):
                note += f"；成本 {_fmt(float(r['cost20']), unit)}，現價低 {(1 - r['close'] / r['cost20']) * 100:.1f}%"
            pts.append({"date": str(r["date"]), "level": _lv(lvl), "close": float(r["close"]), "rule": rule,
                        "side": "sell" if rule.startswith("S") else "buy", "note": note})
    last_fired = {rule: (str(sf.loc[sf[rule], "date"].iloc[-1]) if sf[rule].any() else None) for rule in RULES}
    last = sf.iloc[-1]
    fired = [k for k in BUY_RULES if last[k]]
    sells = [k for k in SELL_RULES if last[k]]
    px = float(last["close"])
    c20 = float(last["cost20"]) if pd.notna(last["cost20"]) else None
    c60 = float(last["cost60"]) if pd.notna(last["cost60"]) else None
    vs20 = round((px / c20 - 1) * 100, 2) if c20 else None
    vs60 = round((px / c60 - 1) * 100, 2) if c60 else None
    rules = []
    for k in fired + sells:
        note = RULES[k].format(w=zone_window, n=TOP_N, tol=tol * 100).replace("[賣訊] ", "")
        if k.startswith("B") and c20:
            note += f"；成本 {_fmt(c20, unit)}，現價低 {-vs20:.1f}%"
        rules.append({"rule": k, "side": "sell" if k.startswith("S") else "buy", "note": note, "last_fired": last_fired[k]})
    evidence = {k: {**RULE_EVIDENCE[k], "last_fired": last_fired[k]} for k in fired + sells if k in RULE_EVIDENCE}
    # 狀態文字：描述官股部位與現價的關係，不是買賣建議
    def _short(k: str) -> str:
        return RULES[k].format(w=zone_window, n=TOP_N, tol=tol * 100).replace("[賣訊] ", "").split(" 且")[0]

    if fired:
        state, text = "dip_buy", "官股逢跌加碼中 (" + "、".join(_short(k) for k in fired) + ")"
    elif sells:
        state, text = "trim", "官股高檔調節 (" + "、".join(_short(k) for k in sells) + ")"
    elif vs60 is not None and vs60 < -10:
        state, text = "trapped", f"官股套牢 {vs60:+.1f}% (現價低於 60 日成本 {_fmt(c60, unit)})"
    elif vs20 is not None and vs20 >= 0:
        state, text = "above_cost", f"現價高於官股 20 日成本 {_fmt(c20, unit)} ({vs20:+.1f}%)，官股部位獲利中"
    elif vs20 is not None:
        state, text = "below_cost", f"現價低於官股 20 日成本 {_fmt(c20, unit)} ({vs20:+.1f}%)，官股今日未加碼"
    else:
        state, text = "na", "官股成本資料不足"
    warn = f"官股套牢 {vs60:+.1f}%" if vs60 is not None and vs60 < -10 else None
    cur = {
        "date": str(last["date"]), "close": px, "flow": float(last["flow"]), "streak": int(last["streak"]),
        "cost20": round(c20, 2) if c20 else None, "cost60": round(c60, 2) if c60 else None,
        "vs_cost20_pct": vs20, "vs_cost60_pct": vs60,
        "in_zone": _lv(last["zone"]) if pd.notna(last["zone"]) else None,
        "in_supply_zone": _lv(last["supply_zone"]) if pd.notna(last["supply_zone"]) else None,
        "buy_rules": fired, "sell_rules": sells, "is_buy_point": bool(fired), "is_sell_point": bool(sells),
        "state": state, "text": text, "warn": warn, "rules": rules,
        "kind": "state", "disclaimer": "狀態描述，非交易訊號：" + EVIDENCE_CAVEAT.split("；")[0],
        "evidence": {"basis": EVIDENCE_BASIS, "caveat": EVIDENCE_CAVEAT, "longrun_dipbuy_pp20": LONGRUN_DIPBUY_PP20, "rules": evidence},
    }
    return {"points": pts, "current": cur}


# ------------------------------------------------------------------ 組合 (build 用)
def _costs(d: dict) -> dict:
    w = d.get("windows") or {}
    px = float(d["price"])

    def vs(c):
        return round((px / c - 1) * 100, 2) if c else None

    c = {"c20": (w.get("20") or {}).get("cost"), "c60": (w.get("60") or {}).get("cost"), "c120": (w.get("120") or {}).get("cost"),
         "call": (w.get("all") or {}).get("cost"), "fifo": (d.get("fifo") or {}).get("cost")}
    c["vs"] = {k: vs(v) for k, v in c.items()}
    return c


def _zones(d: dict) -> tuple[dict, str | None]:
    w = d.get("windows") or {}
    key = DEFAULT_ZONE_WINDOW if DEFAULT_ZONE_WINDOW in w else next((k for k in ("120", "all", "20") if k in w), None)
    if key is None:
        return {"support": [], "supply": [], "overhang": []}, None
    v = w[key]
    slim = lambda rows: [{"level": r["level"], "net": r["net"], "days": r["days"]} for r in rows]  # noqa: E731
    return {"support": slim(v["support"]), "supply": slim(v["supply"]), "overhang": slim(v["overhang"]), "dump_below": slim(v["dump_below"])}, key


def _assemble(d: dict, bp: dict) -> dict:
    zones, zkey = _zones(d)
    n = d.get("history_days") or 0
    cur = bp.get("current") or {}
    d.update({
        "costs": _costs(d), "zones": zones, "zones_window": zkey,
        "current": cur,
        "points": [{"date": p["date"], "level": p["level"], "close": p["close"], "rule": p["rule"], "side": p["side"], "note": p["note"]} for p in bp.get("points", [])],
        "rules": {k: {"label": v.format(w=60, n=TOP_N, tol=(MARKET_TOL if d.get("unit") == "億" else STOCK_TOL) * 100).replace("[賣訊] ", ""),
                      "side": "sell" if k.startswith("S") else "buy", "evidence": RULE_EVIDENCE.get(k)} for k, v in RULES.items()},
        "evidence_basis": EVIDENCE_BASIS, "evidence_caveat": EVIDENCE_CAVEAT,
        "note": (f"歷史僅 {n} 日，統計僅供參考" if n < 60 else ""),
        "sufficient": bool(n >= 60),
    })
    return d


def market_dist(h: pd.DataFrame) -> dict:
    """全市場：h = market_view() 的逐日框 (date, gov8_net, close=FinMind 加權指數)。"""
    if h is None or h.empty or "gov8_net" not in h or "close" not in h:
        return {}
    f = h[["date", "gov8_net", "close"]]
    d = distribution(f, bucket=MARKET_BUCKET)
    if not d:
        return {}
    return _assemble(d, buy_points(f, "gov8_net", MARKET_BUCKET, tol=MARKET_TOL, lookback=LOOKBACK, unit="億"))


def stock_dist(flows: pd.DataFrame, splits: list | None = None) -> dict:
    """個股：flows 需有 date, close 與 gov8_lots (或 gov8)。splits = chips.adjust_splits 的事件 [{date, factor}]：
    還原價不會觸發 |ret1| 截斷，但分割前後的張數基礎不同，仍要從最後一次分割日起算。"""
    if not isinstance(flows, pd.DataFrame) or flows.empty or "close" not in flows:
        return {}
    col = "gov8_lots" if "gov8_lots" in flows else "gov8" if "gov8" in flows else None
    if col is None:
        return {}
    f = flows[["date", col, "close"]].rename(columns={col: "gov8_lots"}).dropna().copy()
    f["date"] = f["date"].astype(str).str[:10]
    if splits:
        last = max(str(s.get("date"))[:10] for s in splits if s.get("date"))
        f = f[f["date"] >= last]
    if len(f) < MIN_HIST:
        return {}
    d = stock_distribution(f, bucket_pct=STOCK_BUCKET_PCT)
    if not d:
        return {}
    return _assemble(d, buy_points(f, "gov8_lots", d["bucket"], tol=STOCK_TOL, lookback=LOOKBACK, unit="張"))


def cached_stock_price(stock_id: str, max_age_days: int = 10) -> pd.DataFrame | None:
    """只讀磁碟快取 (不打 FinMind)：chips.load 的 key 是 start=today-400d、d=today，往回試最近幾天。找不到回 None。"""
    today = dt.date.today()
    for back in range(max_age_days + 1):
        d = today - dt.timedelta(days=back)
        start = (d - dt.timedelta(days=config.HISTORY_DAYS)).isoformat()
        key = f"finmind:data_id={stock_id}&dataset=TaiwanStockPrice&start_date={start}&d={d.isoformat()}"
        p = _cache_path(key)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8")).get("data") or []
        except Exception:  # noqa: BLE001
            continue
        if data:
            df = pd.DataFrame(data)
            if "date" in df and "close" in df:
                return df[["date", "close"]].copy()
    return None


def stock_dist_from_cache(stock_id: str) -> dict:
    """fast 模式：HiStock 個股八大行庫 (快取/網路) + FinMind 快取價 (未還原分割 → _cut_splits 截斷)。價快取缺就回 {}。"""
    px = cached_stock_price(stock_id)
    if px is None:
        log.debug("gov8 dist %s: price cache missing, skip", stock_id)
        return {}
    bk = histock.government_bank_stock(stock_id)
    if bk is None or bk.empty or "gov8_lots" not in bk:
        return {}
    bk = bk[["date", "gov8_lots"]].copy()
    bk["date"] = bk["date"].astype(str).str[:10]
    px["date"] = px["date"].astype(str).str[:10]
    return stock_dist(bk.merge(px, on="date", how="inner"))


def dist_section(h: pd.DataFrame, watch_res: dict | None, prev_wl: dict | None = None, ids: list[str] | None = None) -> tuple[dict, dict]:
    """build() 用。回傳 (market_dist, {sid: dist})。
    watch_res 有值 (完整模式)：用 chips 的 flows (已還原分割的 FinMind 收盤 + gov8 張數) 與 splits 事件。
    watch_res=None (fast 模式)：對 prev_wl 的股票 (或 ids) 用快取價重算；算不出來就沿用 prev 的 dist。"""
    market = {}
    try:
        market = market_dist(h)
    except Exception as e:  # noqa: BLE001
        log.warning("gov8 market distribution: %s", e)
    stocks = {}
    if watch_res is not None:
        for sid, a in watch_res.items():
            try:
                if not isinstance(a, dict) or "error" in a:
                    continue
                d = stock_dist(a.get("flows"), a.get("splits"))
                if d:
                    stocks[sid] = d
            except Exception as e:  # noqa: BLE001
                log.debug("gov8 stock distribution %s: %s", sid, e)
    else:
        for sid in list(ids or []) or list((prev_wl or {}).keys()):
            d = {}
            try:
                d = stock_dist_from_cache(sid)
            except Exception as e:  # noqa: BLE001
                log.debug("gov8 stock distribution (cache) %s: %s", sid, e)
            if not d:
                d = ((prev_wl or {}).get(sid) or {}).get("dist") or {}
            if d:
                stocks[sid] = d
    return market, stocks
