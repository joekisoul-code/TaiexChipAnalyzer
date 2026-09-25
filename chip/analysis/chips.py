"""個股籌碼分布：各路資金 (外資/投信/自營/主力/八大行庫/融資) 的累積部位與平均成本、分價量 (籌碼密集區)、
大戶/散戶持股趨勢 (集保週資料)、券商買賣均價，並綜合成「籌碼分布判讀」。

籌碼在幾塊錢：
- 成本 = 期間內「淨買超日」的 Σ(淨買張 × 收盤價) / Σ 淨買張 (買方平均成本)；若期間淨部位為負則以淨賣日計算「出貨均價」。
- 分價量：把每日成交量平均分配到 [低, 高] 區間的價格箱，累積 N 日 → 密集區 (成本集中)、現價上方套牢量 / 下方獲利量。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..sources import finmind, histock, twse, wantgoo

log = logging.getLogger(__name__)
WATCHLIST = ["2330", "00631L", "00685L", "00981A", "00988A"]
WINDOWS = (5, 20, 60)


def _ms(v) -> str:
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(v / 1000, dt.UTC).date().isoformat()
    return str(v)[:10]


# ------------------------------------------------------------------ 資料組裝
# 已知分割 (分割日 → 倍數)：分割日當天的漲跌會讓「收盤比值四捨五入」猜錯倍數 (00631L 1拆22 當天 -4.4% → 比值 23.0)，
# 已知的以此為準 (以追蹤指數同期報酬反推驗證，2026-09-25)
KNOWN_SPLITS = {"0050": {"2025-06-18": 4.0}, "00631L": {"2026-03-31": 22.0}, "00663L": {"2025-06-11": 7.0}}


def adjust_splits(price: pd.DataFrame, stock_id: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """偵測分割/反分割 (相鄰收盤比值 >2.5 或 <0.4)，把之前的價格除以倍數、成交量乘以倍數 (FinMind 為未還原價)。
    stock_id 在 KNOWN_SPLITS 且日期相符時用已知倍數。"""
    if price.empty or len(price) < 3:
        return price, []
    p = price.copy().reset_index(drop=True)
    ratio = p["close"] / p["close"].shift(1)
    events = []
    for i in p.index[1:]:
        r = ratio[i]
        if pd.notna(r) and (r < 0.4 or r > 2.5):
            factor = round(1 / r) if r < 0.4 else 1 / round(r)     # 分割 1:N → factor = N；反分割 → 1/N
            factor = float(factor) if factor else 1.0
            factor = KNOWN_SPLITS.get(str(stock_id or ""), {}).get(str(p.at[i, "date"])[:10], factor)
            for c in ("open", "high", "low", "close"):
                p.loc[: i - 1, c] = p.loc[: i - 1, c] / factor
            for c in ("volume", "volume_lots"):
                if c in p:
                    p.loc[: i - 1, c] = p.loc[: i - 1, c] * factor
            events.append({"date": str(p.at[i, "date"]), "factor": factor})
    return p, events


def load(stock_id: str, wg: dict | None = None) -> dict:
    """回傳 {'price': DataFrame(date, open, high, low, close, volume_lots), 'flows': DataFrame(date, foreign, trust, dealer, main, gov8, margin_chg, short, sbl),
    'concentration': DataFrame(weekly), 'broker': list, 'margin': DataFrame, 'shareholding': DataFrame}"""
    price = finmind.stock_price(stock_id, (dt.date.today() - dt.timedelta(days=400)).isoformat())
    price, splits = adjust_splits(price, stock_id)
    out: dict = {"stock_id": stock_id, "price": price, "splits": splits}
    flows = price[["date", "close"]].copy() if not price.empty else pd.DataFrame(columns=["date", "close"])
    # 法人 (FinMind，含 ETF)
    try:
        inst = finmind.stock_institutional(stock_id, (dt.date.today() - dt.timedelta(days=400)).isoformat())
        if not inst.empty:
            flows = flows.merge(inst[["date", "foreign", "trust", "dealer"]], on="date", how="left")
    except Exception as e:  # noqa: BLE001
        log.warning("inst %s: %s", stock_id, e)
    # 八大行庫 (HiStock，張)
    try:
        g8 = histock.government_bank_stock(stock_id)
        if not g8.empty:
            flows = flows.merge(g8[["date", "gov8_lots", "gov8_net"]].rename(columns={"gov8_lots": "gov8", "gov8_net": "gov8_amt"}), on="date", how="left")
    except Exception as e:  # noqa: BLE001
        log.warning("gov8 %s: %s", stock_id, e)
    # 融資融券 (FinMind)
    try:
        mg = finmind.stock_margin(stock_id, (dt.date.today() - dt.timedelta(days=400)).isoformat())
        if not mg.empty:
            mg["margin_chg"] = mg["margin_lots"].diff()
            flows = flows.merge(mg[["date", "margin_lots", "margin_chg", "short_lots", "margin_buy"]], on="date", how="left")
    except Exception as e:  # noqa: BLE001
        log.warning("margin %s: %s", stock_id, e)
    # 玩股網：主力、大戶、券商均價、借券、融資維持率
    wg = wg or {}
    mt = wg.get("main_trend")
    if isinstance(mt, list) and mt:
        m = pd.DataFrame([{"date": _ms(r["date"]), "main": r.get("stockAgentDiff"), "main_power": r.get("stockAgentMainPower"),
                           "skp5": r.get("skp5"), "skp20": r.get("skp20")} for r in mt])
        flows = flows.merge(m, on="date", how="left")
    sbl = wg.get("sbl_hist")
    if isinstance(sbl, list) and sbl:
        s = pd.DataFrame([{"date": _ms(r["date"]), "sbl_bal": r.get("todayVolume"), "sbl_sell": r.get("sellOut")} for r in sbl])
        flows = flows.merge(s, on="date", how="left")
    mh = wg.get("margin_hist")
    if isinstance(mh, list) and mh:
        m2 = pd.DataFrame([{"date": _ms(r["date"]), "maint_ratio": (r.get("marginRatio") or 0) * 100} for r in mh])
        flows = flows.merge(m2, on="date", how="left")
    conc = wg.get("concentration")
    out["concentration"] = pd.DataFrame([{"date": _ms(r["date"]), "close": r.get("close"), "big400": r.get("moreThan400"), "big1000": r.get("moreThan1000"),
                                          "retail20": r.get("lessThan20"), "foreign_hold": r.get("rateOfForeignHolding"), "trust_hold": r.get("rateOfINGHolding"),
                                          "director": r.get("directorRatio")} for r in conc]).sort_values("date") if isinstance(conc, list) and conc else pd.DataFrame()
    br = wg.get("broker")
    out["broker"] = br.get("data") if isinstance(br, dict) and isinstance(br.get("data"), list) else (br if isinstance(br, list) else [])
    out["broker_date"] = br.get("newestDate", "")[:10] if isinstance(br, dict) else ""
    sh = wg.get("shareholding")
    out["shareholding"] = pd.DataFrame(sh) if isinstance(sh, list) and sh else pd.DataFrame()
    # ETF 等沒有大戶籌碼頁時，用集保股權分散 (15 級距) 推算：>400 張 = 級距 12~15，>1000 張 = 15，散戶 <10 張 = 1~3
    if out["concentration"].empty and not out["shareholding"].empty:
        s = out["shareholding"].copy()
        s["date"] = s["date"].map(_ms)
        piv = s.pivot_table(index="date", columns="index", values="ratio", aggfunc="last")
        rows = []
        for dte, r in piv.iterrows():
            rows.append({"date": dte, "close": np.nan, "big400": float(sum(r.get(i, 0) or 0 for i in (12, 13, 14, 15))),
                         "big1000": float(r.get(15, 0) or 0), "retail20": float(sum(r.get(i, 0) or 0 for i in (1, 2, 3))),
                         "foreign_hold": np.nan, "trust_hold": np.nan, "director": np.nan})
        out["concentration"] = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        cl = price.set_index("date")["close"] if not price.empty else pd.Series(dtype=float)
        out["concentration"]["close"] = out["concentration"]["date"].map(lambda d_: float(cl[cl.index <= d_].iloc[-1]) if (cl.index <= d_).any() else np.nan)
        out["concentration_source"] = "集保股權分散 (散戶=<10 張)"
    # 集保 open data 補大戶 (ETF 等玩股網無資料時)：最新一週 + SQLite 累積歷史
    if out["concentration"].empty:
        try:
            from ..sources import tdcc
            hs = tdcc.holder_summary(stock_id)
            hist = tdcc.holder_history(stock_id)
            if hs:
                rows = hist.to_dict("records") if not hist.empty else []
                if not any(r.get("date") == hs["date"] for r in rows):
                    rows.append({"date": hs["date"], "big400": hs["big400"], "big1000": hs["big1000"], "retail20": hs["retail10"]})
                c = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
                c["close"] = np.nan
                c["foreign_hold"] = np.nan
                out["concentration"] = c
                out["concentration_source"] = "集保 open data (散戶=<10 張，歷史需每週累積)"
        except Exception as e:  # noqa: BLE001
            log.warning("tdcc %s: %s", stock_id, e)
    # 價格資料異常防呆 (FinMind 偶有混入其他證券/拆分造成的極端值)
    if not price.empty:
        price = price.copy()
        med = price["close"].rolling(20, min_periods=5).median()
        bad = (price["close"] / med - 1).abs() > 0.5
        bad |= (price["high"] > price["close"] * 1.3) | (price["low"] < price["close"] * 0.7)
        if bad.any():
            price = price[~bad].reset_index(drop=True)
            out["price"] = price
            out["flows"] = out["flows"][out["flows"]["date"].isin(price["date"])].reset_index(drop=True) if "flows" in out else out.get("flows")
    out["flows"] = flows.sort_values("date").reset_index(drop=True)
    return out


# ------------------------------------------------------------------ 成本與分布
def cost_table(flows: pd.DataFrame, price: float | None) -> pd.DataFrame:
    """各路資金在 5/20/60 日的累積淨買張數、買方平均成本 (或出貨均價)、現價相對成本。"""
    rows = []
    who = {"foreign": "外資", "trust": "投信", "dealer": "自營商", "main": "主力 (券商分點)", "gov8": "八大行庫", "margin_chg": "融資 (散戶)"}
    for col, name in who.items():
        if col not in flows or flows[col].notna().sum() < 3:
            continue
        rec = {"資金": name}
        for w in WINDOWS:
            t = flows.dropna(subset=[col]).tail(w)
            net = float(t[col].sum())
            buys = t[t[col] > 0]
            sells = t[t[col] < 0]
            if net > 0 and not buys.empty:
                cost = float((buys[col] * buys["close"]).sum() / buys[col].sum())
                rec[f"{w}日淨買(張)"] = round(net)
                rec[f"{w}日成本"] = round(cost, 2)
                rec[f"{w}日現價vs成本%"] = round((price / cost - 1) * 100, 2) if price and cost else None
            elif net < 0 and not sells.empty:
                cost = float((sells[col] * sells["close"]).sum() / sells[col].sum())
                rec[f"{w}日淨買(張)"] = round(net)
                rec[f"{w}日成本"] = round(cost, 2)
                rec[f"{w}日現價vs成本%"] = round((price / cost - 1) * 100, 2) if price and cost else None
            else:
                rec[f"{w}日淨買(張)"] = round(net)
                rec[f"{w}日成本"] = None
                rec[f"{w}日現價vs成本%"] = None
        rows.append(rec)
    return pd.DataFrame(rows)


def volume_profile(price: pd.DataFrame, days: int = 60, bins: int = 30) -> dict:
    """分價量：回傳 {'bins': DataFrame(price_lo, price_hi, mid, volume, pct), 'poc': 密集區價, 'above_pct', 'below_pct', 'va_lo', 'va_hi' (70% 價值區)}"""
    p = price.dropna(subset=["high", "low", "volume_lots"]).tail(days)
    if p.empty:
        return {}
    lo, hi = float(p["low"].min()), float(p["high"].max())
    if hi <= lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for _, r in p.iterrows():
        l, h, v = float(r["low"]), float(r["high"]), float(r["volume_lots"])
        if h <= l:
            idx = min(bins - 1, max(0, int((l - lo) / (hi - lo) * bins)))
            vol[idx] += v
            continue
        # 均勻分配到與 [l,h] 重疊的箱
        for i in range(bins):
            ov = max(0.0, min(h, edges[i + 1]) - max(l, edges[i]))
            if ov > 0:
                vol[i] += v * ov / (h - l)
    total = vol.sum()
    df = pd.DataFrame({"price_lo": edges[:-1], "price_hi": edges[1:], "mid": (edges[:-1] + edges[1:]) / 2, "volume": vol, "pct": vol / total * 100})
    last = float(p["close"].iloc[-1])
    poc = float(df.loc[df["volume"].idxmax(), "mid"])
    above = float(df[df["mid"] > last]["pct"].sum())
    # 70% 價值區：從 POC 向兩側擴張
    order = df.sort_values("volume", ascending=False)
    cum, chosen = 0.0, []
    for i, r in order.iterrows():
        chosen.append(i)
        cum += r["pct"]
        if cum >= 70:
            break
    va_lo, va_hi = float(df.loc[chosen, "price_lo"].min()), float(df.loc[chosen, "price_hi"].max())
    return {"bins": df, "poc": round(poc, 2), "above_pct": round(above, 1), "below_pct": round(100 - above, 1), "va_lo": round(va_lo, 2), "va_hi": round(va_hi, 2),
            "last": last, "days": len(p)}


def holder_trend(conc: pd.DataFrame) -> dict:
    if conc.empty:
        return {}
    c = conc.dropna(subset=["big400"]).reset_index(drop=True)
    if c.empty:
        return {}
    last = c.iloc[-1]

    def chg(col, n):
        return round(float(last[col] - c[col].iloc[-1 - n]), 2) if len(c) > n and pd.notna(c[col].iloc[-1 - n]) else None
    return {"date": str(last["date"]), "big400": last["big400"], "big1000": last["big1000"], "retail20": last["retail20"],
            "foreign_hold": round(float(last["foreign_hold"]), 2) if pd.notna(last.get("foreign_hold")) else None,
            "big400_chg4w": chg("big400", 4), "big400_chg12w": chg("big400", 12), "retail_chg4w": chg("retail20", 4),
            "big1000_chg4w": chg("big1000", 4), "foreign_chg4w": chg("foreign_hold", 4)}


def broker_summary(broker: list, price: float | None) -> pd.DataFrame:
    if not broker:
        return pd.DataFrame()
    df = pd.DataFrame(broker)
    cols = {c: c for c in df.columns}
    rename = {"brokerName": "券商", "buyPriceAvg": "買進均價", "sellPriceAvg": "賣出均價", "buyQuantity": "買進張", "sellQuantity": "賣出張",
              "buyQuantities": "買進張", "sellQuantities": "賣出張", "diff": "買賣超", "netBuySell": "買賣超", "quantityDiff": "買賣超"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in cols})
    if "買賣超" not in df and {"買進張", "賣出張"} <= set(df.columns):
        df["買賣超"] = df["買進張"] - df["賣出張"]
    if price and "買進均價" in df:
        df["現價vs買進均價%"] = ((price / df["買進均價"] - 1) * 100).round(2)
    keep = [c for c in ("券商", "買賣超", "買進張", "賣出張", "買進均價", "賣出均價", "現價vs買進均價%") if c in df.columns]
    df = df[keep] if keep else df
    if "買賣超" in df:
        df = df.sort_values("買賣超", ascending=False)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ 綜合判讀
def assess(stock_id: str, wg: dict | None = None, quote: dict | None = None) -> dict:
    d = load(stock_id, wg)
    price_df, flows = d["price"], d["flows"]
    if price_df.empty:
        return {"stock_id": stock_id, "error": "無價格資料"}
    last_close = float(price_df["close"].iloc[-1])
    price = float(quote["last"]) if quote and quote.get("last") else last_close
    costs = cost_table(flows, price)
    vp = volume_profile(price_df, 60)
    vp120 = volume_profile(price_df, 120, 40)
    holders = holder_trend(d["concentration"])
    brokers = broker_summary(d["broker"], price)
    fl = flows.iloc[-1] if not flows.empty else {}

    notes, score = [], 0.0
    # 現價 vs 各路成本
    def cost_of(name, w=20):
        r = costs[costs["資金"] == name] if not costs.empty else pd.DataFrame()
        return (r.iloc[0].get(f"{w}日成本"), r.iloc[0].get(f"{w}日淨買(張)")) if not r.empty else (None, None)
    for name, w_, label in (("外資", 20, "外資"), ("投信", 20, "投信"), ("主力 (券商分點)", 20, "主力"), ("八大行庫", 20, "官股"), ("融資 (散戶)", 20, "融資")):
        c, net = cost_of(name, w_)
        if c and net:
            rel = (price / c - 1) * 100
            if net > 0:
                if rel >= 0:
                    notes.append(f"{label} 20 日淨買 {net:,.0f} 張、成本約 {c:,.1f}，現價高於成本 {rel:.1f}% (獲利中，籌碼穩)")
                    score += 0.5 if label != "融資" else -0.25
                else:
                    notes.append(f"{label} 20 日淨買 {net:,.0f} 張、成本約 {c:,.1f}，現價低於成本 {abs(rel):.1f}% (套牢，{'可能停損' if label == '融資' else '有護盤/加碼動機'})")
                    score += -0.5 if label == "融資" else 0.25
            else:
                notes.append(f"{label} 20 日淨賣 {abs(net):,.0f} 張、出貨均價約 {c:,.1f}，現價{'低於' if rel < 0 else '高於'}出貨價 {abs(rel):.1f}%")
                score += -0.5 if label != "融資" else 0.25
    if vp:
        pos = "上方套牢籌碼多" if vp["above_pct"] > 55 else "下方獲利籌碼多 (支撐強)" if vp["above_pct"] < 35 else "籌碼上下均衡"
        notes.append(f"60 日分價量密集區 {vp['poc']:,.1f}，價值區 {vp['va_lo']:,.1f}~{vp['va_hi']:,.1f}；現價上方 {vp['above_pct']:.0f}% / 下方 {vp['below_pct']:.0f}% → {pos}")
        score += -0.5 if vp["above_pct"] > 55 else 0.5 if vp["above_pct"] < 35 else 0
    if holders:
        t = f"大戶 (>400 張) 持股 {holders['big400']:.2f}%"
        if holders.get("big400_chg4w") is not None:
            t += f"，4 週 {holders['big400_chg4w']:+.2f}%、12 週 {holders['big400_chg12w']:+.2f}%" if holders.get("big400_chg12w") is not None else f"，4 週 {holders['big400_chg4w']:+.2f}%"
            score += 0.75 if holders["big400_chg4w"] > 0.3 else -0.75 if holders["big400_chg4w"] < -0.3 else 0
        t += f"；散戶 (<20 張) {holders['retail20']:.2f}%" + (f" (4 週 {holders['retail_chg4w']:+.2f}%)" if holders.get("retail_chg4w") is not None else "")
        notes.append(t)
    if "skp20" in flows and pd.notna(fl.get("skp20")):
        notes.append(f"主力籌碼集中度 5 日 {fl['skp5'] * 100:+.1f}% / 20 日 {fl['skp20'] * 100:+.1f}%")
        score += 0.5 if fl["skp20"] > 0.05 else -0.5 if fl["skp20"] < -0.05 else 0
    if "maint_ratio" in flows and pd.notna(fl.get("maint_ratio")):
        notes.append(f"個股融資維持率 {fl['maint_ratio']:.0f}%" + ("，偏低有追繳風險" if fl["maint_ratio"] < 150 else ""))
        score += -0.5 if fl["maint_ratio"] < 150 else 0
    if d.get("splits"):
        notes.append("價格已還原分割：" + "、".join(f"{s['date']} ×{s['factor']:g}" for s in d["splits"]) + "（分割前的融資/法人張數未換算，成本估算以分割後區間為準）")
    if "sbl_bal" in flows and pd.notna(fl.get("sbl_bal")):
        chg5 = flows["sbl_bal"].diff(5).iloc[-1] if flows["sbl_bal"].notna().sum() > 5 else np.nan
        notes.append(f"借券賣出餘額 {fl['sbl_bal']:,.0f} 張" + (f"，5 日 {chg5:+,.0f}" if pd.notna(chg5) else ""))
        score += -0.5 if pd.notna(chg5) and chg5 > 0 and chg5 > 0.02 * fl["sbl_bal"] else 0
    label = "籌碼偏多" if score >= 1.5 else "籌碼偏空" if score <= -1.5 else "籌碼中性"
    return {"stock_id": stock_id, "name": quote.get("name") if quote else stock_id, "price": price, "date": str(price_df["date"].iloc[-1]),
            "costs": costs, "profile60": vp, "profile120": vp120, "holders": holders, "brokers": brokers, "broker_date": d.get("broker_date"),
            "flows": flows, "concentration": d["concentration"], "notes": notes, "score": round(score, 2), "label": label, "quote": quote}


def assess_watchlist(ids: list[str] | None = None, use_wantgoo: bool = True) -> dict[str, dict]:
    ids = ids or WATCHLIST
    wg = wantgoo.fetch_stocks(ids) if use_wantgoo else {}
    out = {}
    for sid in ids:
        try:
            q = twse.stock_realtime(sid)
        except Exception:  # noqa: BLE001
            q = None
        try:
            out[sid] = assess(sid, wg.get(sid), q)
        except Exception as e:  # noqa: BLE001
            log.warning("chips %s: %s", sid, e)
            out[sid] = {"stock_id": sid, "error": str(e)}
    return out
