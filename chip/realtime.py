"""即時追蹤：盤中每幾秒可取得的資料與即時盤勢評分、警示。

資料源 (免費、免登入)：
- TWSE MIS  mis_ohlc_TSE.txt        加權指數 1 分鐘分時 + 當日累積成交量/金額
- TWSE MIS  getStockInfo.jsp        指數與多檔權值股即時報價 (含五檔委買賣量)
- TAIFEX MIS getQuoteList           臺指期日盤/夜盤即時報價、未平倉、現貨參考價 → 期現價差

籌碼 (法人/融資) 盤中不會更新，這裡補的是「盤勢」層：價、量、期貨、廣度、委買賣。
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from . import config, store
from .http import cached, get_text, num, session

log = logging.getLogger(__name__)

MIS = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
MIS_OHLC = "https://mis.twse.com.tw/stock/data/mis_ohlc_TSE.txt"
TAIFEX_Q = "https://mis.taifex.com.tw/futures/api/getQuoteList"
MIS_HEADERS = {"Referer": "https://mis.twse.com.tw/stock/index.jsp"}

# 盤中廣度 / 委買賣 用的權值股樣本 (上市市值前段，可自行調整)
LARGE_CAPS = ["2330", "2317", "2454", "2308", "2382", "2881", "2412", "2882", "2303", "2891",
              "3711", "2886", "2884", "1216", "2885", "3034", "2892", "5880", "2357", "2345",
              "3231", "2327", "2379", "3008", "2207", "2880", "2883", "2603", "2609", "2615",
              "3661", "2395", "6669", "3017", "2408", "2344", "3443", "2301", "1101", "1303"]
SESSION_OPEN, SESSION_CLOSE = dt.time(9, 0), dt.time(13, 30)
SESSION_MINUTES = 270


def now_tw() -> dt.datetime:
    return dt.datetime.now(config.TZ)


def session_phase(t: dt.datetime | None = None) -> str:
    """'pre' 08:30-09:00 / 'open' 09:00-13:30 / 'post' 13:30-15:00 / 'night' 15:00-05:00 / 'closed'"""
    t = t or now_tw()
    tm = t.time()
    if t.weekday() >= 5:
        return "closed"
    if dt.time(8, 30) <= tm < SESSION_OPEN:
        return "pre"
    if SESSION_OPEN <= tm < SESSION_CLOSE:
        return "open"
    if SESSION_CLOSE <= tm < dt.time(15, 0):
        return "post"
    if tm >= dt.time(15, 0) or tm < dt.time(5, 0):
        return "night"
    return "closed"


# ------------------------------------------------------------------ 指數分時
def taiex_intraday() -> dict:
    """1 分鐘分時 + 當日累積。回傳 {'date','bars': [{'time','close','vol'}], 'total_vol_lots', 'total_amount' (億)}"""
    def load():
        txt = get_text(MIS_OHLC, headers=MIS_HEADERS).strip()
        j = json.loads(txt)
        st = j.get("staticObj", {})
        bars = [{"time": f"{b['ts'][:2]}:{b['ts'][2:4]}", "close": num(b["c"]), "vol": num(b["s"])} for b in j.get("ohlcArray", []) if b.get("ts")]
        return {"date": (st.get("key") or "")[-8:], "bars": bars,
                "total_vol_lots": num(st.get("tv")), "total_amount": (num(st.get("tz")) or 0) / 1e8}
    return cached("rt:taiex_intraday", config.TTL_REALTIME, load)


# ------------------------------------------------------------------ 期貨
def futures_quotes(night: bool = False) -> dict | None:
    """臺指期近月即時報價。回傳 {'symbol','last','ref','change','change_pct','volume','oi','spot','basis','basis_pct','time','date','settlement'}"""
    def load():
        r = session().post(TAIFEX_Q, json={"MarketType": "1" if night else "0", "SymbolType": "F", "KindID": "1", "CID": "TXF",
                                            "ExpireMonth": "", "RowSize": "全部", "PageNo": "", "SortColumn": "", "AscDesc": "A"},
                           headers={"Referer": "https://mis.taifex.com.tw/futures/", "Content-Type": "application/json"}, timeout=20)
        r.raise_for_status()
        return r.json().get("RtData", {}).get("QuoteList", [])
    rows = cached(f"rt:tx:{'night' if night else 'day'}", config.TTL_REALTIME, load)
    spot = next((num(x["CLastPrice"]) or num(x["CRefPrice"]) for x in rows if x["SymbolID"].startswith("TXF-")), None)
    near = next((x for x in rows if x["SymbolID"].startswith("TXF") and not x["SymbolID"].startswith("TXF-")), None)
    if not near:
        return None
    last, ref = num(near["CLastPrice"]), num(near["CRefPrice"])
    out = {"symbol": near["DispCName"], "last": last, "ref": ref, "change": (last - ref) if last and ref else None,
           "change_pct": ((last / ref - 1) * 100) if last and ref else None, "volume": num(near["CTotalVolume"]),
           "oi": num(near["OpenInterest"]), "spot": spot, "time": near.get("CTime"), "date": near.get("CDate"),
           "settlement": num(near.get("SettlementPrice")), "night": night,
           "bid": num(near.get("CBidPrice1")), "ask": num(near.get("CAskPrice1")),
           "bid_size": num(near.get("CBidSize1")), "ask_size": num(near.get("CAskSize1"))}
    if last and spot:
        out["basis"] = last - spot
        out["basis_pct"] = (last / spot - 1) * 100
    return out


# ------------------------------------------------------------------ 權值股廣度 / 委買賣
def _sum_sizes(s: str | None) -> float:
    return sum(num(x) or 0 for x in (s or "").split("_") if x)


def large_caps() -> pd.DataFrame:
    """40 檔權值股即時報價：code, name, last, prev, chg_pct, volume, bid_size, ask_size, time"""
    codes = "|".join(f"tse_{c}.tw" for c in LARGE_CAPS)

    def load():
        r = session().get(MIS, params={"ex_ch": codes, "json": "1", "delay": "0"}, headers=MIS_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("msgArray", [])
    rows = cached("rt:large_caps", config.TTL_REALTIME, load)
    out = []
    for m in rows:
        last = num(m.get("z")) or num(m.get("pz"))
        prev = num(m.get("y"))
        if not last or not prev:
            continue
        out.append({"code": m.get("c"), "name": m.get("n"), "last": last, "prev": prev, "chg_pct": (last / prev - 1) * 100,
                    "volume": num(m.get("v")), "bid_size": _sum_sizes(m.get("f")), "ask_size": _sum_sizes(m.get("g")),
                    "time": m.get("%") or m.get("t")})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 綜合快照
def snapshot(scored: pd.DataFrame | None = None) -> dict:
    """一次組合所有即時資料，並計算量能進度、廣度、盤勢即時分。"""
    t = now_tw()
    phase = session_phase(t)
    snap: dict = {"ts": t.strftime("%Y-%m-%d %H:%M:%S"), "phase": phase}

    # 指數
    try:
        r = session().get(MIS, params={"ex_ch": "tse_t00.tw|otc_o00.tw", "json": "1", "delay": "0"}, headers=MIS_HEADERS, timeout=15).json()
        for m in r.get("msgArray", []):
            last, prev = num(m.get("z")), num(m.get("y"))
            key = "taiex" if m.get("c") == "t00" else "otc"
            snap[key] = {"last": last, "prev": prev, "open": num(m.get("o")), "high": num(m.get("h")), "low": num(m.get("l")),
                         "chg": (last - prev) if last and prev else None, "chg_pct": ((last / prev - 1) * 100) if last and prev else None,
                         "time": m.get("%"), "date": m.get("d")}
    except Exception as e:  # noqa: BLE001
        snap["error_index"] = str(e)

    # 分時與量能進度
    try:
        intra = taiex_intraday()
        snap["intraday"] = intra
        amt = intra.get("total_amount") or 0
        elapsed = SESSION_MINUTES
        if phase == "open":
            elapsed = max(15, (t.hour - 9) * 60 + t.minute)
        elif phase == "pre":
            elapsed = 0
        if elapsed:
            snap["amount_so_far"] = amt
            snap["amount_projected"] = amt / (elapsed / SESSION_MINUTES)
            snap["elapsed_frac"] = elapsed / SESSION_MINUTES
    except Exception as e:  # noqa: BLE001
        snap["error_intraday"] = str(e)

    # 期貨 (日盤 / 夜盤)
    try:
        snap["tx"] = futures_quotes(night=False)
        if phase in ("night", "closed", "pre"):
            snap["tx_night"] = futures_quotes(night=True)
    except Exception as e:  # noqa: BLE001
        snap["error_tx"] = str(e)

    # 廣度 / 委買賣
    try:
        lc = large_caps()
        if not lc.empty:
            up, down = int((lc["chg_pct"] > 0).sum()), int((lc["chg_pct"] < 0).sum())
            snap["breadth"] = {"n": len(lc), "up": up, "down": down, "flat": len(lc) - up - down,
                               "avg_chg": float(lc["chg_pct"].mean()), "bid_ask_ratio": float(lc["bid_size"].sum() / max(1, lc["ask_size"].sum()))}
            snap["large_caps"] = lc.sort_values("chg_pct", ascending=False).to_dict("records")
            tsmc = lc[lc["code"] == "2330"]
            if not tsmc.empty:
                snap["tsmc_chg"] = float(tsmc.iloc[0]["chg_pct"])
    except Exception as e:  # noqa: BLE001
        snap["error_breadth"] = str(e)

    # 國際盤即時 (Yahoo，延遲約 15 分)：美股前晚收盤、亞股盤中、比特幣/原油/黃金/匯率
    try:
        from .sources import global_markets as gm
        snap["global"] = gm.quotes(["sp500", "nasdaq", "sox", "vix", "tsm_adr", "nikkei", "kospi", "hsi", "btc", "oil", "gold", "usdtwd", "us10y"])
    except Exception as e:  # noqa: BLE001
        snap["error_global"] = str(e)

    # 臺指選擇權波動率指數 (VIXTWN，期交所每日檔，盤中約延遲數分鐘)
    try:
        from .sources import taifex_vix
        snap["vixtwn"] = taifex_vix.latest()
    except Exception as e:  # noqa: BLE001
        snap["error_vixtwn"] = str(e)

    # 與盤後資料結合：MA、量能基準
    if scored is not None and not scored.empty:
        last = scored.iloc[-1]
        closes = scored["close"].astype(float)
        same_day = snap.get("taiex", {}).get("date", "") == str(last["date"]).replace("-", "")
        price = snap.get("taiex", {}).get("last")
        if price:
            if same_day:
                ma5, ma20 = last["ma5"], last["ma20"]
            else:
                ma5, ma20 = (closes.tail(4).sum() + price) / 5, (closes.tail(19).sum() + price) / 20
            snap["ma5"], snap["ma20"], snap["ma60"] = float(ma5), float(ma20), float(last["ma60"]) if pd.notna(last["ma60"]) else None
            snap["bias20"] = (price / ma20 - 1) * 100
            snap["chip_date"] = str(last["date"])
            snap["same_day"] = bool(same_day)
        amt20 = last.get("amount_ma20")
        if pd.notna(amt20) and snap.get("amount_projected"):
            snap["vol_pace"] = snap["amount_projected"] / float(amt20)
            snap["amount_ma20"] = float(amt20)
    snap["score"] = intraday_score(snap)
    return snap


def intraday_score(s: dict) -> dict:
    """盤勢即時分 (-100..100)：價、期現價差、廣度、量能、委買賣、均線位置、台積電相對。"""
    parts = []

    def add(name, score, w, text):
        parts.append({"name": name, "score": max(-2, min(2, score)), "w": w, "text": text})

    idx = s.get("taiex") or {}
    chg = idx.get("chg_pct")
    if chg is not None:
        add("指數漲跌", chg / 1.2, 2.0, f"{chg:+.2f}%")
    tx = s.get("tx") or {}
    bp = tx.get("basis_pct")
    if bp is not None:
        add("期現價差", bp / 0.35, 1.5, f"{tx.get('basis', 0):+.0f} 點 ({bp:+.2f}%) {'正價差' if bp > 0 else '逆價差'}")
    b = s.get("breadth")
    if b:
        add("權值廣度", (b["up"] - b["down"]) / b["n"] * 2.5, 1.5, f"漲 {b['up']} / 跌 {b['down']} (均 {b['avg_chg']:+.2f}%)")
        add("委買賣比", (b["bid_ask_ratio"] - 1) * 3, 1.0, f"委買/委賣 {b['bid_ask_ratio']:.2f}")
    vp = s.get("vol_pace")
    if vp is not None and chg is not None and s.get("phase") in ("open", "post", "night", "closed"):
        sign = 1 if chg > 0 else -1 if chg < 0 else 0
        add("量能", sign * (vp - 1) * 2.5, 1.0, f"推估全日 {s.get('amount_projected', 0):,.0f} 億 = 20 日均 {vp:.2f}x")
    if s.get("ma20") and idx.get("last"):
        above20, above5 = idx["last"] > s["ma20"], idx["last"] > s.get("ma5", 0)
        add("均線位置", (1 if above20 else -1) + (0.5 if above5 else -0.5), 1.5,
            f"{'站上' if above20 else '跌破'}月線 / {'站上' if above5 else '跌破'} 5 日線 (乖離 {s.get('bias20', 0):+.2f}%)")
    if s.get("tsmc_chg") is not None and chg is not None:
        add("台積電相對", (s["tsmc_chg"] - chg) / 1.0, 0.5, f"台積電 {s['tsmc_chg']:+.2f}% vs 大盤 {chg:+.2f}%")
    gq = {q["key"]: q for q in (s.get("global") or [])}
    asia = [gq[k]["chg_pct"] for k in ("nikkei", "kospi") if gq.get(k) and gq[k].get("chg_pct") is not None]
    if asia and s.get("phase") in ("open", "pre"):
        add("亞股同步", float(np.mean(asia)) / 1.0, 1.0, "日經/KOSPI 盤中 " + " / ".join(f"{a:+.2f}%" for a in asia))
    if gq.get("sox") and gq["sox"].get("chg_pct") is not None and s.get("phase") in ("pre",):
        add("前晚費半", gq["sox"]["chg_pct"] / 1.5, 1.0, f"費半 {gq['sox']['chg_pct']:+.2f}% (決定跳空，開盤後影響小)")
    vt = s.get("vixtwn") or {}
    if vt.get("last") and vt.get("open"):
        chg = (vt["last"] / vt["open"] - 1) * 100
        add("台指VIX", -chg / 5.0, 0.75, f"VIXTWN {vt['last']:.2f} (今日 {chg:+.1f}%；>30 恐慌、<15 自滿)")
    if not parts:
        return {"score": 0, "label": "無資料", "parts": []}
    wsum = sum(p["w"] for p in parts)
    score = round(sum(p["score"] * p["w"] for p in parts) / (2 * wsum) * 100, 1)
    label = "強勢" if score >= 40 else "偏強" if score >= 15 else "中性" if score > -15 else "偏弱" if score > -40 else "弱勢"
    return {"score": score, "label": label, "parts": parts}


def combined_view(chip_smooth: float, chip_regime: str, rt_score: float, rt_label: str, phase: str) -> str:
    """籌碼 (盤後) × 盤勢 (即時) 交叉解讀。"""
    chip_bull, chip_bear = chip_smooth >= 15, chip_smooth <= -15
    rt_bull, rt_bear = rt_score >= 15, rt_score <= -15
    if phase not in ("open", "pre"):
        pre = "非交易時段，盤勢分為最後成交狀態。"
    else:
        pre = ""
    if chip_bull and rt_bull:
        return pre + "籌碼偏多 + 盤勢強：順勢，可依計畫加碼，拉回 5 日線為進場點。"
    if chip_bull and rt_bear:
        return pre + "籌碼偏多但盤勢弱：多頭拉回，不急著停損；若收盤跌破月線且量增再減碼。"
    if chip_bear and rt_bull:
        return pre + "籌碼偏空但盤勢強：反彈性質，不追高；等籌碼 (外資/期貨) 轉向再確認。"
    if chip_bear and rt_bear:
        return pre + "籌碼偏空 + 盤勢弱：空方共振，避免進場，已持股者控管部位。"
    if rt_bull:
        return pre + "籌碼中性、盤勢偏強：可小量試單，依收盤籌碼再決定加碼。"
    if rt_bear:
        return pre + "籌碼中性、盤勢偏弱：觀望，等待止跌訊號。"
    return pre + "籌碼與盤勢皆中性：區間操作或觀望。"


# ------------------------------------------------------------------ 警示
ALERT_RULES = [
    # (key, condition(snap, prev) -> str | None)
    ("break_ma20_down", lambda s, p: "指數盤中跌破月線" if s.get("ma20") and s["taiex"]["last"] < s["ma20"] and (p or {}).get("taiex", {}).get("last", 0) >= (p or {}).get("ma20", 0) and p else None),
    ("break_ma20_up", lambda s, p: "指數盤中站回月線" if s.get("ma20") and s["taiex"]["last"] > s["ma20"] and p and (p.get("taiex", {}).get("last", 1e9) <= p.get("ma20", 1e9)) else None),
    ("drop2", lambda s, p: f"大盤重挫 {s['taiex']['chg_pct']:+.2f}%" if (s.get("taiex", {}).get("chg_pct") or 0) <= -2 else None),
    ("rise2", lambda s, p: f"大盤大漲 {s['taiex']['chg_pct']:+.2f}%" if (s.get("taiex", {}).get("chg_pct") or 0) >= 2 else None),
    ("backwardation", lambda s, p: f"台指期逆價差擴大 {s['tx']['basis']:+.0f} 點 ({s['tx']['basis_pct']:+.2f}%)" if (s.get("tx") or {}).get("basis_pct", 0) <= -0.4 else None),
    ("contango", lambda s, p: f"台指期正價差 {s['tx']['basis']:+.0f} 點 ({s['tx']['basis_pct']:+.2f}%)" if (s.get("tx") or {}).get("basis_pct", 0) >= 0.4 else None),
    ("heavy_vol_down", lambda s, p: f"量能放大 {s['vol_pace']:.2f}x 且下跌 → 賣壓沉重" if s.get("vol_pace", 0) >= 1.5 and (s.get("taiex", {}).get("chg_pct") or 0) < -0.5 else None),
    ("heavy_vol_up", lambda s, p: f"量能放大 {s['vol_pace']:.2f}x 且上漲 → 攻擊量" if s.get("vol_pace", 0) >= 1.5 and (s.get("taiex", {}).get("chg_pct") or 0) > 0.5 else None),
    ("breadth_weak", lambda s, p: f"權值股普跌 (跌 {s['breadth']['down']}/{s['breadth']['n']})" if s.get("breadth") and s["breadth"]["down"] >= s["breadth"]["n"] * 0.8 else None),
    ("breadth_strong", lambda s, p: f"權值股普漲 (漲 {s['breadth']['up']}/{s['breadth']['n']})" if s.get("breadth") and s["breadth"]["up"] >= s["breadth"]["n"] * 0.8 else None),
    ("ask_heavy", lambda s, p: f"委賣明顯大於委買 (比 {s['breadth']['bid_ask_ratio']:.2f})" if s.get("breadth") and s["breadth"]["bid_ask_ratio"] <= 0.6 else None),
    ("tsmc_diverge", lambda s, p: f"台積電 {s['tsmc_chg']:+.2f}% 與大盤 {s['taiex']['chg_pct']:+.2f}% 背離" if s.get("tsmc_chg") is not None and abs(s["tsmc_chg"] - (s.get("taiex", {}).get("chg_pct") or 0)) >= 1.5 else None),
    ("night_drop", lambda s, p: f"夜盤台指期 {s['tx_night']['change_pct']:+.2f}%" if (s.get("tx_night") or {}).get("change_pct") is not None and abs(s["tx_night"]["change_pct"]) >= 1 else None),
    ("bias_hot", lambda s, p: f"月線乖離 {s['bias20']:+.2f}% 過大" if abs(s.get("bias20") or 0) >= 6 else None),
]


def check_alerts(snap: dict, prev: dict | None, fired: set[str]) -> list[dict]:
    """回傳本次新觸發的警示 (同一日同一規則只觸發一次；fired 由呼叫端維護)。"""
    events = []
    if not snap.get("taiex"):
        return events
    for key, fn in ALERT_RULES:
        try:
            msg = fn(snap, prev)
        except Exception:  # noqa: BLE001
            msg = None
        if msg and key not in fired:
            fired.add(key)
            events.append({"ts": snap["ts"], "key": key, "message": msg})
    return events


def persist(snap: dict) -> None:
    """把快照關鍵值寫入 SQLite (盤中時間序列)。"""
    idx, tx, b = snap.get("taiex") or {}, snap.get("tx") or {}, snap.get("breadth") or {}
    store.upsert_intraday(snap["ts"], {
        "taiex": idx.get("last"), "taiex_chg_pct": idx.get("chg_pct"), "tx": tx.get("last"), "basis": tx.get("basis"),
        "basis_pct": tx.get("basis_pct"), "tx_oi": tx.get("oi"), "breadth_up": b.get("up"), "breadth_down": b.get("down"),
        "bid_ask_ratio": b.get("bid_ask_ratio"), "vol_pace": snap.get("vol_pace"), "amount_projected": snap.get("amount_projected"),
        "rt_score": (snap.get("score") or {}).get("score"),
    })
