"""八大行庫 (公股銀行：合庫/土銀/台銀/台企銀/彰銀/第一金/兆豐銀/華南永昌) 籌碼進出監測。

資料：HiStock broker8.aspx (全市場每日買賣總額近半年 + 當日買賣超排行含各行庫明細)、broker.aspx?no= (個股各行庫每日金額/張數)。
HiStock 只給近半年 → 全市場序列另存 SQLite 累積，並與已發布的 GitHub Pages `data/gov8.json` 取聯集，讓歷史越跑越長。
排行每日存 SQLite snapshots，算「連續上榜天數」與「上榜期間累計金額」，抓出官股持續加碼/出貨的個股。

輸出 build()：
- market   全市場：今日/5/20/60 日累計 (億)、連買賣天數、5 日 z、今日在歷史的百分位、60 日與指數日漲跌相關 (負 = 逆勢護盤性格)、
           行為模式 (逆勢護盤/順勢加碼/高檔調節/順勢減碼/中性)、事件統計 (護盤/調節/極端買賣/跌日買/漲日賣/連買賣≥5 之後 1/2/5/10/20 日報酬 vs 基準)、
           護盤力道 -100..100 (單日 z + 5 日 z + 連續天數 + 模式)、存量 (自起點累計) 與存量位置、60 日行為統計 (買超天數比/跌買率/漲賣率/平均規模)、近 12 週週別統計
- history  逐日序列 (畫圖用)
- ranking  今日買超/賣超前 15 檔：金額 (億)、主買/主賣行庫、連續上榜天數、上榜期間累計；各行庫今日合計 (誰在主導)、各行庫近 5 日趨勢 (bank5，由每日快照累加)
- watchlist 追蹤清單個股：近 20 日逐日張數、5/20/60 日累計、各行庫 20 日張數、20 日成本與現價差
- dist     價位別籌碼分布 (market.dist / watchlist[sid].dist，見 gov8_dist.py)：各價位桶淨買賣、20/60/120/全期成本、支撐/供給/套牢區、
           FIFO 剩餘部位成本、主力狀態 (官股逢跌加碼中 / 高檔調節 / 套牢；描述性，非交易訊號)
- alerts   文字警示
"""
from __future__ import annotations

import datetime as dt
import logging
import os

import numpy as np
import pandas as pd

from .. import config, store
from ..http import session
from ..sources import histock
from . import gov8_dist
from .common import streak, zscore

log = logging.getLogger(__name__)
PAGES_URL = os.getenv("CHIP_PAGES_URL", "https://joekisoul-code.github.io/TaiexChipAnalyzer/")
BANKS = histock.BANKS
WAN_TO_YI = 1e-4  # 萬元 → 億
WATCH_IDS = ["2330", "00631L", "00685L", "00981A", "00988A"]   # 與 chips.WATCHLIST 相同 (fast 模式無 prev 時的備援)


# ------------------------------------------------------------------ 已發布資料 (Actions 跨次累積的保險)
def published() -> dict:
    try:
        r = session().get(PAGES_URL.rstrip("/") + "/data/gov8.json", timeout=20)
        if r.ok:
            return r.json() or {}
    except Exception as e:  # noqa: BLE001
        log.debug("published gov8.json unavailable: %s", e)
    return {}


# ------------------------------------------------------------------ 全市場序列
def market_history(prev: dict | None = None) -> pd.DataFrame:
    """date, gov8_net (億), close。HiStock 最新 (優先) ∪ SQLite 累積 ∪ 已發布 JSON。"""
    parts = []
    try:
        fresh = histock.government_banks_history()
        if not fresh.empty:
            store.upsert_frame(fresh, ["gov8_net"])
            parts.append(fresh[["date", "gov8_net", "close"]])
    except Exception as e:  # noqa: BLE001
        log.warning("HiStock 八大行庫歷史失敗: %s", e)
    try:
        stored = store.load_metrics(["gov8_net"])
        if not stored.empty and "gov8_net" in stored:
            parts.append(stored[["date", "gov8_net"]])
    except Exception as e:  # noqa: BLE001
        log.debug("store gov8: %s", e)
    hist = (prev or {}).get("history") or []
    if hist:
        p = pd.DataFrame(hist)
        if "gov8_net" in p:
            parts.append(p[["date", "gov8_net"]])
    if not parts:
        return pd.DataFrame(columns=["date", "gov8_net", "close"])
    df = pd.concat(parts, ignore_index=True).dropna(subset=["gov8_net"])
    df["date"] = df["date"].astype(str).str[:10]
    df = df.drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)
    return df


def _event_stats(h: pd.DataFrame, mask: pd.Series) -> dict | None:
    m = mask.fillna(False)
    g = h[m]
    n = int(g["fwd10"].notna().sum())
    if n == 0:
        return None
    out = {"n": n}
    for k in (1, 2, 5, 10, 20):
        if f"fwd{k}" not in g:
            continue
        v = g[f"fwd{k}"].dropna()
        out[f"fwd{k}"] = round(float(v.mean()), 2) if len(v) else None
        out[f"win{k}"] = round(float((v > 0).mean() * 100), 1) if len(v) else None
    out["last"] = str(g["date"].iloc[-1])
    return out


def market_view(hist: pd.DataFrame, scored: pd.DataFrame | None) -> tuple[dict, pd.DataFrame]:
    h = hist.dropna(subset=["gov8_net"]).copy()
    if h.empty:
        return {}, h
    if scored is not None and not scored.empty and "close" in scored:
        px = scored[["date", "close"]].copy()
        px["date"] = px["date"].astype(str)
        h = h.drop(columns=["close"], errors="ignore").merge(px, on="date", how="left")
    c = h["close"].astype(float)
    h["ret1"] = c.pct_change() * 100
    h["ret5"] = c.pct_change(5) * 100
    h["ret20"] = c.pct_change(20) * 100
    for k in (1, 2, 5, 10, 20):
        h[f"fwd{k}"] = (c.shift(-k) / c - 1) * 100
    g = h["gov8_net"].astype(float)
    h["cum5"] = g.rolling(5, min_periods=1).sum()
    h["cum20"] = g.rolling(20, min_periods=1).sum()
    h["cum60"] = g.rolling(60, min_periods=1).sum()
    h["cum120"] = g.rolling(120, min_periods=1).sum()
    h["ma5"] = g.rolling(5, min_periods=1).mean()
    h["ma20"] = g.rolling(20, min_periods=1).mean()
    h["inv"] = g.cumsum()                      # 存量：自歷史起點累計 (官股淨部位變化)
    h["streak"] = streak(g)
    h["z5"] = zscore(h["cum5"], 60)
    h["z1"] = zscore(g, 60)
    h["z20"] = zscore(h["cum20"], 120)
    last = h.iloc[-1]
    st = int(last["streak"])
    ret5 = float(last["ret5"]) if pd.notna(last["ret5"]) else 0.0
    cum5 = float(last["cum5"])
    net = float(last["gov8_net"])
    pct = float((g <= net).mean() * 100)
    tail = h.tail(60)
    corr60 = float(tail["gov8_net"].corr(tail["ret1"])) if tail["ret1"].notna().sum() > 20 else None
    if st >= 3 and ret5 < 0:
        mode, mtxt = "逆勢護盤", f"指數 5 日 {ret5:+.1f}% 但官股連買 {st} 日，國家隊進場跡象"
    elif cum5 > 30 and ret5 > 0:
        mode, mtxt = "順勢加碼", f"指數上漲官股 5 日仍買 {cum5:+.0f} 億，官股與市場同向"
    elif st <= -3 and ret5 > 0:
        mode, mtxt = "高檔調節", f"指數 5 日 {ret5:+.1f}% 官股連賣 {abs(st)} 日，高檔獲利了結"
    elif cum5 < -30 and ret5 < 0:
        mode, mtxt = "順勢減碼", f"指數下跌官股 5 日賣 {cum5:+.0f} 億，未護盤反而減碼 (偏空)"
    elif net > 0:
        mode, mtxt = "小幅買超", "官股買超但未成趨勢"
    elif net < 0:
        mode, mtxt = "小幅賣超", "官股賣超但未成趨勢"
    else:
        mode, mtxt = "中性", "官股動作不明顯"
    q90, q10 = float(g.quantile(0.9)), float(g.quantile(0.1))
    events = {
        "護盤 (連買≥3日且指數5日跌)": _event_stats(h, (h["streak"] >= 3) & (h["ret5"] < 0)),
        "調節 (連賣≥3日且指數5日漲)": _event_stats(h, (h["streak"] <= -3) & (h["ret5"] > 0)),
        f"極端買超 (單日 > {q90:.0f} 億, p90)": _event_stats(h, g >= q90),
        f"極端賣超 (單日 < {q10:.0f} 億, p10)": _event_stats(h, g <= q10),
        "5 日累計 z > 1.5": _event_stats(h, h["z5"] > 1.5),
        "5 日累計 z < -1.5": _event_stats(h, h["z5"] < -1.5),
        "跌日買超 (指數跌 >1% 且官股買 >50 億)": _event_stats(h, (h["ret1"] < -1) & (g > 50)),
        "漲日賣超 (指數漲 >1% 且官股賣 >50 億)": _event_stats(h, (h["ret1"] > 1) & (g < -50)),
        "連買 ≥5 日": _event_stats(h, h["streak"] >= 5),
        "連賣 ≥5 日": _event_stats(h, h["streak"] <= -5),
    }
    base = {}
    for k in (1, 2, 5, 10, 20):
        v = h[f"fwd{k}"].dropna()
        base[f"fwd{k}"] = round(float(v.mean()), 2) if len(v) else None
        base[f"win{k}"] = round(float((v > 0).mean() * 100), 1) if len(v) else None
    # 護盤後的表現是否優於基準 → 一句話結論 (樣本少時標註)
    p = events["護盤 (連買≥3日且指數5日跌)"]
    if p and p.get("fwd10") is not None and base.get("fwd10") is not None:
        diff = p["fwd10"] - base["fwd10"]
        verdict = (f"護盤後 10 日平均 {p['fwd10']:+.2f}% (基準 {base['fwd10']:+.2f}%，{'優於' if diff > 0 else '劣於'}基準 {abs(diff):.2f}%，n={p['n']}"
                   + ("，樣本少僅供參考" if p["n"] < 15 else "") + ")")
    else:
        verdict = "護盤事件樣本不足，持續累積中"
    # 60 日行為統計：買超天數比、平均買/賣規模、跌買率 (指數跌日官股買超的比例)、漲賣率
    t60 = h.tail(60)
    g60 = t60["gov8_net"].astype(float)
    r60 = t60["ret1"]
    down, up = t60[r60 < 0], t60[r60 > 0]
    behav = {
        "buy_days_pct": round(float((g60 > 0).mean() * 100), 0),
        "avg_buy": round(float(g60[g60 > 0].mean()), 1) if (g60 > 0).any() else None,
        "avg_sell": round(float(g60[g60 < 0].mean()), 1) if (g60 < 0).any() else None,
        "dip_buy_rate": round(float((down["gov8_net"] > 0).mean() * 100), 0) if len(down) else None,
        "rally_sell_rate": round(float((up["gov8_net"] < 0).mean() * 100), 0) if len(up) else None,
        "dip_days": int(len(down)), "up_days": int(len(up)),
        "net_on_down": round(float(down["gov8_net"].sum()), 0) if len(down) else None,
        "net_on_up": round(float(up["gov8_net"].sum()), 0) if len(up) else None,
    }
    # 護盤力道 -100..100：單日 z、5 日 z、連續天數、行為模式 綜合
    z1v = float(last["z1"]) if pd.notna(last["z1"]) else 0.0
    z5v = float(last["z5"]) if pd.notna(last["z5"]) else 0.0
    mode_pts = {"逆勢護盤": 25, "順勢加碼": 15, "小幅買超": 5, "中性": 0, "小幅賣超": -5, "高檔調節": -15, "順勢減碼": -25}[mode]
    power = max(-100, min(100, 30 * max(-2, min(2, z1v)) / 2 + 30 * max(-2, min(2, z5v)) / 2 + 20 * max(-5, min(5, st)) / 5 + mode_pts))
    power_text = ("強力護盤" if power >= 60 else "積極買進" if power >= 30 else "小幅偏買" if power >= 10 else
                  "強力出貨" if power <= -60 else "明顯賣出" if power <= -30 else "小幅偏賣" if power <= -10 else "觀望中性")
    # 存量位置：目前累計淨部位在歷史區間的位置 (100% = 歷史最高持有)
    inv = h["inv"].astype(float)
    inv_rng = float(inv.max() - inv.min())
    inv_pos = round(float((inv.iloc[-1] - inv.min()) / inv_rng * 100), 0) if inv_rng > 0 else None
    # 週別統計 (近 12 週)：官股週淨買賣 vs 指數週漲跌
    weekly = []
    try:
        wk = h.copy()
        wk["wk"] = pd.to_datetime(wk["date"]).dt.to_period("W").astype(str)
        for _k, grp in list(wk.groupby("wk"))[-12:]:
            cl = grp["close"].astype(float).dropna()
            weekly.append({"week": str(grp["date"].iloc[0])[:10], "net": round(float(grp["gov8_net"].sum()), 0), "days": int(len(grp)),
                           "ret": round(float(cl.iloc[-1] / cl.iloc[0] - 1) * 100, 2) if len(cl) >= 2 else None})
    except Exception as e:  # noqa: BLE001
        log.debug("gov8 weekly: %s", e)
    view = {
        "date": str(last["date"]), "net": round(net, 1), "cum5": round(cum5, 1), "cum20": round(float(last["cum20"]), 1), "cum60": round(float(last["cum60"]), 1),
        "cum120": round(float(last["cum120"]), 1), "z20": round(float(last["z20"]), 2) if pd.notna(last["z20"]) else None,
        "ma20": round(float(last["ma20"]), 1), "inv": round(float(inv.iloc[-1]), 0), "inv_pos": inv_pos, "inv_max": round(float(inv.max()), 0), "inv_min": round(float(inv.min()), 0),
        "behav": behav, "power": round(float(power), 0), "power_text": power_text, "weekly": weekly,
        "streak": st, "z5": round(float(last["z5"]), 2) if pd.notna(last["z5"]) else None, "z1": round(float(last["z1"]), 2) if pd.notna(last["z1"]) else None,
        "percentile": round(pct, 0), "corr60": round(corr60, 2) if corr60 is not None else None,
        "corr_text": ("逆勢操作 (跌買漲賣) 性格明顯" if corr60 is not None and corr60 < -0.2 else "偏順勢 (漲買跌賣)" if corr60 is not None and corr60 > 0.2 else "與漲跌無明顯關係") if corr60 is not None else "",
        "mode": mode, "mode_text": mtxt, "ret5": round(ret5, 2), "ret1": round(float(last["ret1"]), 2) if pd.notna(last["ret1"]) else None,
        "close": float(last["close"]) if pd.notna(last["close"]) else None,
        "history_days": int(len(h)), "history_from": str(h["date"].iloc[0]),
        "events": events, "baseline": base, "verdict": verdict,
        "max_buy": {"date": str(h.loc[g.idxmax(), "date"]), "net": round(float(g.max()), 1)},
        "max_sell": {"date": str(h.loc[g.idxmin(), "date"]), "net": round(float(g.min()), 1)},
    }
    return view, h


# ------------------------------------------------------------------ 排行 (含連續上榜)
def ranking() -> dict:
    rk = histock.government_banks_ranking()
    date = rk.get("date") or ""
    out = {"date": date, "buy": [], "sell": [], "banks": [], "n_days": 0}
    if not date:
        return out

    def rows(df: pd.DataFrame, side: str) -> list[dict]:
        if df is None or df.empty:
            return []
        d = df.copy()
        for b in BANKS:
            if b not in d:
                d[b] = np.nan
        bank_vals = d[BANKS].astype(float).fillna(0)
        d["top_bank"] = bank_vals.idxmax(axis=1) if side == "buy" else bank_vals.idxmin(axis=1)
        d["top_bank_amt"] = (bank_vals.max(axis=1) if side == "buy" else bank_vals.min(axis=1)) * WAN_TO_YI
        d["n_banks"] = ((bank_vals > 0) if side == "buy" else (bank_vals < 0)).sum(axis=1)   # 幾家行庫同向
        d["total_yi"] = d["total"].astype(float) * WAN_TO_YI
        return [{"code": str(r["code"]), "name": r["name"], "total": round(float(r["total_yi"]), 2), "top_bank": r["top_bank"],
                 "top_bank_amt": round(float(r["top_bank_amt"]), 2), "n_banks": int(r["n_banks"]),
                 "banks": {b: round(float(r[b]) * WAN_TO_YI, 2) for b in BANKS if pd.notna(r[b])}} for _, r in d.head(15).iterrows()]

    out["buy"], out["sell"] = rows(rk["buy"], "buy"), rows(rk["sell"], "sell")
    # 各行庫今日合計 (上榜 60 檔的加總，代表誰在主導)
    tot = {b: 0.0 for b in BANKS}
    for side in ("buy", "sell"):
        df = rk[side]
        if df is not None and not df.empty:
            for b in BANKS:
                if b in df:
                    tot[b] += float(pd.to_numeric(df[b], errors="coerce").fillna(0).sum()) * WAN_TO_YI
    out["banks"] = sorted([{"bank": b, "net": round(v, 2)} for b, v in tot.items()], key=lambda x: -x["net"])
    # 每日快照 → 連續上榜天數 / 上榜期間累計
    try:
        store.save_snapshot(date, "gov8_rank", {"buy": [{"code": r["code"], "total": r["total"]} for r in out["buy"]],
                                                "sell": [{"code": r["code"], "total": r["total"]} for r in out["sell"]],
                                                "banks": {b["bank"]: b["net"] for b in out["banks"]}})
        snaps = store.load_snapshots("gov8_rank", 10)   # 最新在前
    except Exception as e:  # noqa: BLE001
        log.debug("gov8 snapshots: %s", e)
        snaps = []
    out["n_days"] = len(snaps)
    for side in ("buy", "sell"):
        for r in out[side]:
            days, cum = 0, 0.0
            for _, payload in snaps:   # 從今天往回數，直到有一天沒上榜
                hit = next((x for x in payload.get(side, []) if x["code"] == r["code"]), None)
                if not hit:
                    break
                days += 1
                cum += float(hit["total"])
            r["days_on"], r["cum_on"] = days, round(cum, 2)
    # 各行庫近 5 日趨勢 (從每日快照的行庫合計累加)：誰在持續買、誰在持續賣
    bank5 = {b: {"net5": 0.0, "pos": 0, "n": 0} for b in BANKS}
    for _, payload in snaps[:5]:
        bk = payload.get("banks") or {}
        for b in BANKS:
            if b in bk:
                bank5[b]["net5"] += float(bk[b]); bank5[b]["n"] += 1
                if float(bk[b]) > 0:
                    bank5[b]["pos"] += 1
    out["bank5"] = sorted([{"bank": b, "net5": round(v["net5"], 2), "pos_days": v["pos"], "n": v["n"]} for b, v in bank5.items() if v["n"]], key=lambda x: -x["net5"])
    return out


# ------------------------------------------------------------------ 追蹤清單個股
def watchlist(res: dict | None) -> dict:
    out = {}
    for sid, a in (res or {}).items():
        if not isinstance(a, dict) or "error" in a or not isinstance(a.get("flows"), pd.DataFrame):
            continue
        fl = a["flows"]
        if "gov8" not in fl or fl["gov8"].notna().sum() < 3:
            continue
        f = fl.dropna(subset=["gov8"]).copy()
        g = f["gov8"].astype(float)
        rec = {"stock_id": sid, "name": a.get("name"), "price": a.get("price"), "date": str(f["date"].iloc[-1]),
               "today": round(float(g.iloc[-1])), "cum5": round(float(g.tail(5).sum())), "cum20": round(float(g.tail(20).sum())), "cum60": round(float(g.tail(60).sum())),
               "streak": int(streak(g).iloc[-1]),
               "daily": [{"date": str(r["date"]), "lots": round(float(r["gov8"])), "close": float(r["close"]) if pd.notna(r.get("close")) else None} for _, r in f.tail(20).iterrows()]}
        costs = a.get("costs")
        if isinstance(costs, pd.DataFrame) and not costs.empty:
            row = costs[costs["資金"] == "八大行庫"]
            if not row.empty:
                r0 = row.iloc[0]
                rec["cost20"] = r0.get("20日成本")
                rec["vs20"] = r0.get("20日現價vs成本%")
                rec["cost60"] = r0.get("60日成本")
                rec["vs60"] = r0.get("60日現價vs成本%")
        try:
            bk = histock.government_bank_stock(sid)
            if not bk.empty:
                t = bk.tail(20)
                rec["banks20"] = sorted([{"bank": b, "lots": round(float(pd.to_numeric(t[f"lots_{b}"], errors="coerce").fillna(0).sum()))}
                                         for b in BANKS if f"lots_{b}" in t], key=lambda x: -x["lots"])
        except Exception as e:  # noqa: BLE001
            log.debug("gov8 banks %s: %s", sid, e)
        out[sid] = rec
    return out


# ------------------------------------------------------------------ 警示
def alerts(view: dict, rank: dict, wl: dict) -> list[str]:
    al = []
    if view:
        if view["mode"] in ("逆勢護盤", "高檔調節", "順勢減碼"):
            al.append(f"八大行庫{view['mode']}：{view['mode_text']}")
        if view.get("z1") is not None and abs(view["z1"]) >= 2:
            al.append(f"八大行庫單日{'買超' if view['net'] > 0 else '賣超'} {abs(view['net']):.0f} 億，為 60 日 {view['z1']:+.1f} 個標準差 (歷史百分位 {view['percentile']:.0f}%)")
        if abs(view["streak"]) >= 5:
            al.append(f"八大行庫連{'買' if view['streak'] > 0 else '賣'} {abs(view['streak'])} 日")
        if abs(view.get("power") or 0) >= 60:
            al.append(f"八大行庫護盤力道 {view['power']:+.0f} ({view['power_text']})")
    for side, txt in (("buy", "買超"), ("sell", "賣超")):
        for r in rank.get(side, [])[:15]:
            if r.get("days_on", 0) >= 3:
                al.append(f"{r['code']} {r['name']} 八大行庫連續 {r['days_on']} 日{txt}上榜，累計 {abs(r['cum_on']):.1f} 億")
    for sid, r in wl.items():
        st = int(r.get("streak") or 0)
        if abs(st) >= 3 and r.get("cum20") is not None:
            al.append(f"{sid} {r.get('name') or ''} 八大行庫連{'買' if st > 0 else '賣'} {abs(st)} 日，20 日累計 {r['cum20']:+,} 張")
        w = ((r.get("dist") or {}).get("current") or {}).get("warn")
        if w:
            al.append(f"{sid} {r.get('name') or ''} {w} (現價低於八大行庫 60 日成本)")
    return al[:12]


def build(scored: pd.DataFrame | None = None, watch_res: dict | None = None, prev: dict | None = None) -> dict:
    """完整輸出。watch_res=None (fast 模式) 時追蹤清單沿用 prev (上次發布) 的內容。"""
    prev = prev if prev is not None else published()
    hist = market_history(prev)
    view, h = market_view(hist, scored)
    try:
        rank = ranking()
    except Exception as e:  # noqa: BLE001
        log.warning("gov8 ranking: %s", e)
        rank = prev.get("ranking") or {"date": "", "buy": [], "sell": [], "banks": []}
    prev_wl = prev.get("watchlist") or {}
    wl = watchlist(watch_res) if watch_res is not None else dict(prev_wl)
    # 價位別籌碼分布 + 主力狀態 (描述性)：完整模式用 chips 的 flows；fast 模式用快取價重算，算不出來沿用上次發布的 dist
    try:
        mdist, sdist = gov8_dist.dist_section(h, watch_res, prev_wl, ids=None if watch_res is not None else (list(prev_wl) or WATCH_IDS))
        if mdist:
            view["dist"] = mdist
        for sid, d in sdist.items():
            wl.setdefault(sid, {"stock_id": sid})["dist"] = d
    except Exception as e:  # noqa: BLE001
        log.warning("gov8 dist: %s", e)
    cols = [c for c in ("date", "gov8_net", "close", "ret1", "cum5", "cum20", "ma20", "inv", "streak", "z5") if c in h]
    return {
        "generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "market": view,
        "history": h[cols].tail(400).to_dict("records") if not h.empty else [],
        "ranking": rank,
        "watchlist": wl,
        "alerts": alerts(view, rank, wl),
        "banks": BANKS,
    }
