"""個股籌碼評分：法人、八大行庫、融資融券、借券、外資持股、量價、趨勢 + 大盤環境加權。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config
from ..sources import finmind, histock, twse
from .common import Factor, clip, composite, fmt, regime, streak, zscore

log = logging.getLogger(__name__)

WEIGHTS = {"foreign": 3.0, "trust": 2.0, "dealer": 0.5, "gov8": 1.5, "concentration": 2.0, "margin": 2.0,
           "short": 1.0, "sbl": 1.0, "holding": 1.0, "rs": 1.5, "volume": 1.5, "trend": 2.0}
NAMES = {"foreign": "外資買賣超", "trust": "投信買賣超", "dealer": "自營商買賣超", "gov8": "八大行庫買賣超",
         "concentration": "法人籌碼集中度 (20日淨買/成交量)", "margin": "融資量價象限", "short": "融券/券資比",
         "sbl": "借券賣出餘額", "holding": "外資持股比率趨勢", "rs": "相對大盤強弱 (20日)", "volume": "量價關係", "trend": "股價趨勢"}


def build_frame(stock_id: str) -> tuple[pd.DataFrame, dict]:
    meta: dict = {}
    price = finmind.stock_price(stock_id)
    if price.empty:
        raise ValueError(f"找不到 {stock_id} 的價格資料")
    df = price.copy()
    for name, fn in (("法人", lambda: finmind.stock_institutional(stock_id)),
                     ("融資融券", lambda: finmind.stock_margin(stock_id)),
                     ("外資持股", lambda: finmind.stock_shareholding(stock_id)),
                     ("八大行庫", lambda: histock.government_bank_stock(stock_id))):
        try:
            part = fn()
            if not part.empty:
                df = df.merge(part, on="date", how="left")
            meta[name] = {"status": "ok", "latest": str(part["date"].max()) if not part.empty else ""}
        except Exception as e:  # noqa: BLE001
            meta[name] = {"status": f"fail: {e}"}
    # 當日 TWSE 補丁 (FinMind 未更新時)
    try:
        t = twse.t86()
        row = t[t["code"] == stock_id]
        if not row.empty:
            r = row.iloc[0]
            if r["date"] not in set(df["date"]):
                df.loc[len(df)] = {"date": r["date"]}
            i = df.index[df["date"] == r["date"]][0]
            for src, dst in (("foreign_net", "foreign"), ("trust_net", "trust"), ("dealer_net", "dealer"), ("total_net", "total")):
                if dst not in df or pd.isna(df.at[i, dst]):
                    df.at[i, dst] = r[src]
        meta["TWSE T86"] = {"status": "ok", "latest": str(t["date"].iloc[0]) if not t.empty else ""}
    except Exception as e:  # noqa: BLE001
        meta["TWSE T86"] = {"status": f"fail: {e}"}
    try:
        s = twse.sbl_balance()
        srow = next((x for x in s["stocks"] if x["code"] == stock_id), None) if s else None
        meta["sbl_today"] = srow
    except Exception as e:  # noqa: BLE001
        meta["sbl_today"] = None
        meta["TWSE 借券"] = {"status": f"fail: {e}"}
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["close"].notna()].reset_index(drop=True)
    return add_features(df), meta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("foreign", "trust", "dealer", "total", "gov8_net", "gov8_lots", "margin_lots", "short_lots", "foreign_ratio", "volume_lots"):
        if c not in d:
            d[c] = np.nan
    if d["total"].isna().all():
        d["total"] = d[["foreign", "trust", "dealer"]].sum(axis=1, min_count=1)
    c = d["close"].astype(float)
    d["ret1"] = c.pct_change() * 100
    d["ret5"] = c.pct_change(5) * 100
    d["ret20"] = c.pct_change(20) * 100
    d["ma5"], d["ma20"], d["ma60"] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    d["bias20"] = (c / d["ma20"] - 1) * 100
    d["vol_ma20"] = d["volume_lots"].rolling(20).mean()
    d["vol_ratio"] = d["volume_lots"] / d["vol_ma20"]
    for who in ("foreign", "trust", "dealer", "total", "gov8_lots"):
        d[f"{who}_5d"] = d[who].rolling(5, min_periods=1).sum()
        d[f"{who}_20d"] = d[who].rolling(20, min_periods=1).sum()
    d["gov8_net_5d"] = d["gov8_net"].rolling(5, min_periods=1).sum()
    d["gov8_net_20d"] = d["gov8_net"].rolling(20, min_periods=1).sum()
    d["foreign_streak"] = streak(d["foreign"])
    d["trust_streak"] = streak(d["trust"])
    d["concentration"] = d["total_20d"] / d["volume_lots"].rolling(20).sum() * 100
    d["margin_pct20"] = d["margin_lots"].pct_change(20) * 100
    d["margin_div20"] = d["margin_pct20"] - d["ret20"]
    d["margin_chg5"] = d["margin_lots"].diff(5)
    d["short_chg5"] = d["short_lots"].diff(5)
    d["short_ratio"] = d["short_lots"] / d["margin_lots"].replace(0, np.nan) * 100
    d["holding_chg20"] = d["foreign_ratio"].diff(20)
    return d


def score_row(d: pd.DataFrame, sbl_today: dict | None, market_ret20: float | None = None) -> list[Factor]:
    r = d.iloc[-1]
    z = lambda col, w=60: zscore(d[col], w).iloc[-1]  # noqa: E731
    f: list[Factor] = []

    def add(key, score, value, comment, available=True, tags=None):
        f.append(Factor(key, NAMES[key], clip(score) if available else 0.0, WEIGHTS[key], value, comment, available, tags or []))

    st = int(r["foreign_streak"] or 0)
    z1, z5 = z("foreign"), z("foreign_5d")
    add("foreign", 0.5 * (0 if np.isnan(z1) else z1) + 0.5 * (0 if np.isnan(z5) else z5),
        f"今 {fmt(r['foreign'], 0)} 張｜5日 {fmt(r['foreign_5d'], 0)}｜20日 {fmt(r['foreign_20d'], 0)}｜連{'買' if st > 0 else '賣'} {abs(st)} 日",
        "外資連續買超，主力籌碼進駐" if st >= 3 else "外資連續賣超，籌碼流出" if st <= -3 else "外資動作反覆",
        available=not (np.isnan(z1) and np.isnan(z5)), tags=["外資連買"] if st >= 3 else ["外資連賣"] if st <= -3 else [])
    tst = int(r["trust_streak"] or 0)
    zt = z("trust_5d")
    both_buy = (r["foreign_5d"] or 0) > 0 and (r["trust_5d"] or 0) > 0 and st > -3 and tst > -3
    both_sell = (r["foreign_5d"] or 0) < 0 and (r["trust_5d"] or 0) < 0 and st < 3 and tst < 3
    ttags = ["投信認養"] if tst >= 3 else []
    ttags += ["外資投信雙買"] if both_buy else ["外資投信雙賣"] if both_sell else []
    add("trust", (0 if np.isnan(zt) else zt) + (0.5 if both_buy else -0.5 if both_sell else 0),
        f"今 {fmt(r['trust'], 0)} 張｜5日 {fmt(r['trust_5d'], 0)}｜20日 {fmt(r['trust_20d'], 0)}｜連{'買' if tst > 0 else '賣'} {abs(tst)} 日",
        ("投信認養中 (連續買超)" if tst >= 3 else "投信調節" if tst <= -3 else "投信中性")
        + ("；外資投信同步買超，籌碼最強組合" if both_buy else "；外資投信同步賣超，籌碼最弱組合" if both_sell else ""),
        available=not np.isnan(zt), tags=ttags)
    zd = z("dealer")
    add("dealer", zd, f"今 {fmt(r['dealer'], 0)} 張｜5日 {fmt(r['dealer_5d'], 0)}", "自營商偏多" if r["dealer_5d"] > 0 else "自營商偏空", available=not np.isnan(zd))
    zg = z("gov8_net_5d")
    g5 = r["gov8_net_5d"]
    bonus = 0.5 if (pd.notna(g5) and g5 > 0 and r["ret5"] < 0) else 0
    add("gov8", (0 if np.isnan(zg) else zg) + bonus,
        f"今 {fmt(r['gov8_net'], 0)} 萬｜5日 {fmt(g5, 0)} 萬｜20日 {fmt(r['gov8_net_20d'], 0)} 萬｜5日 {fmt(r['gov8_lots_5d'], 0)} 張",
        "官股逆勢買進 (護盤/低接)" if bonus else "官股買超" if (g5 or 0) > 0 else "官股賣超" if (g5 or 0) < 0 else "無官股動作",
        available=pd.notna(g5), tags=["官股護盤"] if bonus else [])
    cc = r["concentration"]
    add("concentration", np.select([cc > 15, cc > 5, cc > -5, cc > -15], [2, 1, 0, -1], -2) if pd.notna(cc) else 0,
        f"{fmt(cc, 1, '%')} (20日法人淨買 / 成交量)",
        "法人高度集中吸籌" if (cc or 0) > 15 else "法人溫和吸籌" if (cc or 0) > 5 else "法人倒貨" if (cc or 0) < -5 else "籌碼分散",
        available=pd.notna(cc))
    m20, r20 = r["margin_pct20"], r["ret20"]
    if pd.isna(m20) or pd.isna(r20):
        ms, mc, mt = 0, "資料不足", []
    elif m20 > 5 and r20 < -5:
        ms, mc, mt = -2, "股價下跌融資卻增加 → 散戶接刀，籌碼最差象限", ["散戶接刀"]
    elif m20 - r20 > 8:
        ms, mc, mt = -2, "融資追價過熱，散戶籌碼凌亂", ["融資過熱"]
    elif m20 - r20 > 3:
        ms, mc, mt = -1, "融資增速快於股價，籌碼略浮動", []
    elif m20 < -10 and r20 < -8:
        ms, mc, mt = (1 if r["ret5"] > 0 else 0), "融資大減且股價大跌 → 斷頭清洗" + ("，近 5 日止穩" if r["ret5"] > 0 else "，尚未止穩"), ["融資斷頭清洗"]
    elif m20 - r20 < -3 and r20 > 0:
        ms, mc, mt = 2, "股價上漲融資減少 → 籌碼沉澱，法人主導", []
    elif m20 - r20 < -3:
        ms, mc, mt = 1, "融資相對減少，籌碼趨於沉澱", []
    else:
        ms, mc, mt = 0, "融資與股價同步", []
    add("margin", ms, f"餘額 {fmt(r['margin_lots'], 0, ' 張', sign=False)}｜5日 {fmt(r['margin_chg5'], 0)}｜20日 {fmt(m20, 1)}% vs 股價 {fmt(r20, 1)}%",
        mc, available=not (pd.isna(m20) or pd.isna(r20)), tags=mt)
    sr, sc5 = r["short_ratio"], r["short_chg5"]
    sh = 0.0
    if pd.notna(sr):
        sh = 1 if (sr > 20 and r["ret5"] > 0) else 0.5 if sr > 10 else 0
        if pd.notna(sc5) and sc5 > 0 and r["ret5"] < 0:
            sh -= 0.5
    add("short", sh, f"融券 {fmt(r['short_lots'], 0, ' 張', sign=False)}｜5日 {fmt(sc5, 0)}｜券資比 {fmt(sr, 1, '%', sign=False)}",
        "券資比高且股價上漲 → 軋空題材" if sh >= 1 else "融券偏多，留意空方" if (sc5 or 0) > 0 else "融券影響小", available=pd.notna(sr))
    if sbl_today:
        chg = (sbl_today["sbl_today"] or 0) - (sbl_today["sbl_prev"] or 0)
        add("sbl", float(np.select([chg < -500, chg < 0, chg <= 0, chg < 500], [1, 0.5, 0, -0.5], -1.5)),
            f"借券賣出餘額 {fmt(sbl_today['sbl_today'], 0, ' 張', sign=False)}｜今變 {fmt(chg, 0)}",
            "借券賣出回補" if chg < 0 else "借券賣出增加 (法人放空)" if chg > 0 else "無變化")
    else:
        add("sbl", 0, "N/A", "無借券資料", available=False)
    hc = r["holding_chg20"]
    add("holding", np.select([hc > 1, hc > 0.2, hc > -0.2, hc > -1], [2, 1, 0, -1], -2) if pd.notna(hc) else 0,
        f"外資持股 {fmt(r['foreign_ratio'], 2, '%', sign=False)}｜20日 {fmt(hc, 2, '%')}",
        "外資持股比率持續上升" if (hc or 0) > 0.2 else "外資持股比率下降" if (hc or 0) < -0.2 else "外資持股持平", available=pd.notna(hc))
    if market_ret20 is not None and pd.notna(r["ret20"]):
        rs = r["ret20"] - market_ret20
        add("rs", float(np.select([rs > 5, rs > 2, rs > -2, rs > -5], [2, 1, 0, -1], -2)),
            f"個股 20 日 {fmt(r['ret20'], 1)}% vs 大盤 {fmt(market_ret20, 1)}% → 相對 {fmt(rs, 1)}%",
            "明顯強於大盤 (資金流入)" if rs > 5 else "略強於大盤" if rs > 2 else "明顯弱於大盤 (資金流出)" if rs < -5 else "略弱於大盤" if rs < -2 else "與大盤同步",
            tags=["強勢股"] if rs > 5 else ["弱勢股"] if rs < -5 else [])
    else:
        add("rs", 0, "N/A", "無大盤資料", available=False)
    r1, vr = r["ret1"], r["vol_ratio"]
    vs = 0.0
    if pd.notna(r1) and pd.notna(vr):
        vs = -2 if (vr > 2 and r1 < -2) else 1.5 if (r1 > 0.5 and vr > 1.2) else 0 if (r1 > 0.5) else 0.5 if (r1 < -0.5 and vr < 0.8) else -1.5 if (r1 < -0.5 and vr > 1.3) else 0
    add("volume", vs, f"漲跌 {fmt(r1, 2, '%')}｜量 {fmt(r['volume_lots'], 0, ' 張', sign=False)} ({fmt(vr, 2, 'x', sign=False)} 20日均)",
        {-2: "爆量長黑", 1.5: "價漲量增", 0.5: "價跌量縮", -1.5: "價跌量增"}.get(vs, "量價中性"), available=pd.notna(vr))
    c, ma20, ma60, b = r["close"], r["ma20"], r["ma60"], r["bias20"]
    ts = np.nan
    if pd.notna(ma60):
        ts = 2 if (c > ma20 and c > ma60 and ma20 > ma60) else 1.5 if (c > ma20 and c > ma60) else 1 if c > ma20 else -1 if c > ma60 else -2
        ts += -1 if b > 10 else 1 if b < -10 else 0
    add("trend", ts if pd.notna(ts) else 0, f"收 {fmt(c, 2, sign=False)}｜MA20 {fmt(ma20, 2, sign=False)}｜MA60 {fmt(ma60, 2, sign=False)}｜乖離20 {fmt(b, 2, '%')}",
        "多頭排列" if (ts or 0) >= 1.5 else "站上月線" if (ts or 0) >= 1 else "月線下整理" if (ts or 0) == -1 else "空方排列" if pd.notna(ts) else "資料不足",
        available=pd.notna(ts), tags=["乖離過大"] if (b or 0) > 10 else [])
    return f


def assess(stock_id: str, market_assessment: dict | None = None) -> dict:
    """market_assessment: 大盤 assess() 結果 (或至少含 regime / state / ret20)。"""
    d, meta = build_frame(stock_id)
    ma = market_assessment or {}
    factors = score_row(d, meta.get("sbl_today"), ma.get("ret20"))
    comp = composite(factors)
    mreg, mstate = ma.get("regime"), ma.get("state", "盤整")
    adj = {"強勢多方": 8, "偏多": 4, "中性": 0, "偏空": -8, "空方": -15}.get(mreg, 0) if mreg else 0.0
    final = round(max(-100, min(100, comp + adj)), 1)
    reg = regime(final)
    r = d.iloc[-1]
    quote = None
    try:
        quote = twse.stock_realtime(stock_id)
    except Exception:  # noqa: BLE001
        pass
    tags = sorted({t for f in factors for t in f.tags})
    trend_up = any(f.key == "trend" and f.available and f.score >= 1 for f in factors)
    strong_combo = "外資投信雙買" in tags or ("外資連買" in tags and "強勢股" in tags)
    weak_combo = "外資投信雙賣" in tags or "散戶接刀" in tags
    # 進場邏輯：個股籌碼 + 趨勢 + 大盤狀態
    if final >= 30 and trend_up and not weak_combo:
        action = "可進場（籌碼+趨勢同向）" + ("，法人共識強" if strong_combo else "")
    elif final >= 15 and trend_up:
        action = "可分批布局"
    elif final >= 15 and not trend_up:
        action = "籌碼轉佳但趨勢未翻多，等站上月線再進"
    elif final <= -25 or weak_combo:
        action = "不宜進場／減碼"
    elif "官股護盤" in tags or ("外資連賣" in tags and r["bias20"] < -10) or "融資斷頭清洗" in tags:
        action = "超跌觀察，小量試單"
    else:
        action = "觀望"
    if mstate == "空頭" and action.startswith("可"):
        action += "；大盤空頭，部位減半、只做強勢股"
    elif mreg in ("偏空", "空方") and action.startswith("可"):
        action += "；大盤偏空請縮小部位"
    name = quote["name"] if quote else stock_id
    return {"stock_id": stock_id, "name": name, "date": str(r["date"]), "close": float(r["close"]),
            "composite_raw": comp, "market_adj": adj, "composite": final, "regime": reg, "action": action,
            "factors": factors, "tags": tags, "quote": quote, "frame": d, "meta": meta}


def broker_report(stock_id: str, date: str) -> pd.DataFrame | None:
    """券商分點 (需 FinMind 贊助等級 token)。回傳買超/賣超前 15 名，無權限回 None。"""
    if not config.FINMIND_TOKEN:
        return None
    try:
        df = finmind.broker_daily(stock_id, date)
    except finmind.FinMindPermissionError:
        return None
    if df.empty:
        return df
    df["net"] = df["buy"] - df["sell"]
    g = df.groupby(["securities_trader_id", "securities_trader"], as_index=False)[["buy", "sell", "net"]].sum()
    return g.sort_values("net", ascending=False)
