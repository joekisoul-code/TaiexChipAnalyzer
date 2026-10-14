"""臺灣證券交易所 (TWSE) 官方資料。

- BFI82U  三大法人買賣金額統計 (當日/指定日)
- MI_MARGN 信用交易統計 (大盤融資融券餘額) + 個股融資融券
- FMTQIK  每日市場成交資訊 (成交金額、加權指數)
- T86     三大法人買賣超日報 (個股)
- TWT93U  信用額度總量管制餘額表 (含借券賣出餘額)
- mis.twse.com.tw 即時指數 / 個股報價
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import numpy as np
import pandas as pd

from .. import config
from ..http import cache_delete, cache_get, cache_put, cached, get_json, num
from . import NoData

log = logging.getLogger(__name__)
BASE = "https://www.twse.com.tw/rwd/zh"
MIS = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
YI = 1e8  # 億


def roc_to_iso(s: str) -> str:
    """'115/09/01' -> '2026-09-01'"""
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}-{int(d):02d}"


def ymd_to_iso(s: str) -> str:
    """'20260911' -> '2026-09-11'"""
    s = s.strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _ok(j: dict) -> bool:
    return isinstance(j, dict) and j.get("stat") == "OK"


def _ttl(date: str | None) -> int:
    return config.TTL_HISTORY if date else config.TTL_INTRADAY


# ---------------------------------------------------------------- 三大法人
def institutional_daily(date: str | None = None) -> dict | None:
    """三大法人買賣金額 (億)。date='YYYYMMDD'，None=最新。無資料回 None。"""
    def load():
        j = get_json(f"{BASE}/fund/BFI82U", params={"response": "json", "dayDate": date or "", "type": "day"})
        if not _ok(j):
            return None
        out = {"date": ymd_to_iso(j["date"])}
        keymap = {
            "自營商(自行買賣)": "dealer_self",
            "自營商(避險)": "dealer_hedge",
            "投信": "trust",
            "外資及陸資(不含外資自營商)": "foreign",
            "外資自營商": "foreign_dealer",
            "合計": "total",
        }
        for row in j["data"]:
            k = keymap.get(row[0].strip())
            if k:
                out[k] = num(row[3]) / YI
                out[k + "_buy"] = num(row[1]) / YI
                out[k + "_sell"] = num(row[2]) / YI
        out["dealer"] = out.get("dealer_self", 0) + out.get("dealer_hedge", 0)
        return out
    return cached(f"twse:BFI82U:{date}", _ttl(date), load)


# ---------------------------------------------------------------- 融資融券 (大盤)
def margin_daily(date: str | None = None) -> dict | None:
    """大盤信用交易統計：融資/融券張數、融資金額 (億)。"""
    def load():
        j = get_json(f"{BASE}/marginTrading/MI_MARGN",
                     params={"response": "json", "selectType": "MS", "date": date or ""})
        if not _ok(j) or not j.get("tables"):
            return None
        t = j["tables"][0]
        out = {"date": ymd_to_iso(j["date"])}
        for row in t["data"]:
            item = row[0]
            buy, sell, repay, prev, today = (num(x) for x in row[1:6])
            if item.startswith("融資(交易"):
                out.update(margin_lots_prev=prev, margin_lots=today, margin_buy_lots=buy, margin_sell_lots=sell)
            elif item.startswith("融券"):
                out.update(short_lots_prev=prev, short_lots=today, short_buy_lots=buy, short_sell_lots=sell)
            elif item.startswith("融資金額"):
                out.update(margin_amt_prev=prev * 1e3 / YI, margin_amt=today * 1e3 / YI,
                           margin_buy_amt=buy * 1e3 / YI, margin_sell_amt=sell * 1e3 / YI)
        return out
    return cached(f"twse:MI_MARGN:MS:{date}", _ttl(date), load)


def margin_stocks(date: str | None = None) -> pd.DataFrame:
    """個股融資融券餘額 (張)。"""
    def load():
        j = get_json(f"{BASE}/marginTrading/MI_MARGN",
                     params={"response": "json", "selectType": "ALL", "date": date or ""})
        if not _ok(j) or len(j.get("tables", [])) < 2:
            return []
        rows = []
        for r in j["tables"][1]["data"]:
            rows.append({
                "code": r[0].strip(), "name": r[1].strip(),
                "margin_buy": num(r[2]), "margin_sell": num(r[3]), "margin_repay": num(r[4]),
                "margin_prev": num(r[5]), "margin_today": num(r[6]), "margin_limit": num(r[7]),
                "short_buy": num(r[8]), "short_sell": num(r[9]), "short_repay": num(r[10]),
                "short_prev": num(r[11]), "short_today": num(r[12]), "short_limit": num(r[13]),
                "offset": num(r[14]), "date": ymd_to_iso(j["date"]),
            })
        return rows
    return pd.DataFrame(cached(f"twse:MI_MARGN:ALL:{date}", _ttl(date), load))


# ---------------------------------------------------------------- 融資融券 歷史回補 (margin_dyn 資料層, 2026-10-14)
# 與上面 margin_stocks()/margin_daily() 的差別：10 年 TTL、stat≠OK 丟例外不寫快取 (#87 地雷)、含 note 欄與 tables[0] 的現償欄、
# 同主機節流 ≥2.2 s。快取格式刻意與本機 491 日快取 / margin_stocks() 相容：ALL key 存「列 dict 的 list」，全市場三列另存 MS key
# (margin_daily 同格式 + 三個 *_repay_* 欄)，讀取時再組成 {"date", "market", "stocks"}。
TTL_FOREVER = 10 * 365 * 86400
MIN_GAP = 2.2                 # 同主機兩次網路請求的最小間隔 (秒)；測試可設 0
CLOSES_FROM = "20040211"      # MI_INDEX 可查的最早日期 (查更早回「查詢日期小於93年2月11日」)
MARGIN_ALL_COLS = ("code", "name", "margin_buy", "margin_sell", "margin_repay", "margin_prev", "margin_today", "margin_limit",
                   "short_buy", "short_sell", "short_repay", "short_prev", "short_today", "short_limit", "offset", "note", "date")
MARGIN_MARKET_KEYS = ("margin_lots_prev", "margin_lots", "margin_buy_lots", "margin_sell_lots", "margin_repay_lots",
                      "short_lots_prev", "short_lots", "short_buy_lots", "short_sell_lots", "short_repay_lots",
                      "margin_amt_prev", "margin_amt", "margin_buy_amt", "margin_sell_amt", "margin_repay_amt")
_CODE_RE = re.compile(r"\d{4,6}[A-Z]?")
_last_net = 0.0


def _throttle() -> None:
    """只在真的要打網路時呼叫 (快取命中不等)。"""
    global _last_net
    wait = MIN_GAP - (time.time() - _last_net)
    if wait > 0:
        time.sleep(wait)
    _last_net = time.time()


def _nodata_stat(stat: str) -> bool:
    return any(k in stat for k in ("沒有符合", "無資料", "查詢日期"))


def _check_stat(j, what: str) -> dict:
    """stat≠OK：「沒有符合條件 / 查詢日期…」→ NoData；其他 (伺服器忙碌、非 JSON) → RuntimeError。兩者都不會被 cached() 寫入。"""
    if not isinstance(j, dict):
        raise RuntimeError(f"{what}: 非 dict 回應 ({type(j).__name__})")
    stat = str(j.get("stat") or "")
    if stat.upper() != "OK":
        if _nodata_stat(stat):
            raise NoData(f"{what}: {stat}")
        raise RuntimeError(f"{what}: stat={stat[:60] or '(空)'}")
    return j


def parse_margin_market(table: dict) -> dict:
    """MI_MARGN tables[0] 三列 (融資(交易單位) / 融券(交易單位) / 融資金額(仟元)) → 張與億 (含現償)。"""
    out: dict = {}
    for row in table.get("data") or []:
        item = str(row[0])
        buy, sell, repay, prev, today = (num(x) for x in row[1:6])
        if item.startswith("融資(交易"):
            out.update(margin_lots_prev=prev, margin_lots=today, margin_buy_lots=buy, margin_sell_lots=sell, margin_repay_lots=repay)
        elif item.startswith("融券"):
            out.update(short_lots_prev=prev, short_lots=today, short_buy_lots=buy, short_sell_lots=sell, short_repay_lots=repay)
        elif item.startswith("融資金額"):
            k = 1e3 / YI
            out.update(margin_amt_prev=_mul(prev, k), margin_amt=_mul(today, k), margin_buy_amt=_mul(buy, k),
                       margin_sell_amt=_mul(sell, k), margin_repay_amt=_mul(repay, k))
    return out


def _mul(v, k):
    return None if v is None else v * k


def parse_margin_all(j, ymd: str) -> dict:
    """MI_MARGN selectType=ALL 回應 → {"date": ISO, "market": 三列 dict, "stocks": [16 欄 dict (含 note)]}。
    stat 含「沒有符合」→ NoData；其他非 OK / 缺表 → RuntimeError (不可快取)。"""
    _check_stat(j, f"MI_MARGN ALL {ymd}")
    tables = j.get("tables") or []
    if len(tables) < 2 or not tables[1].get("data"):
        raise RuntimeError(f"MI_MARGN ALL {ymd}: stat OK 但無個股表 ({len(tables)} tables)")
    iso = ymd_to_iso(str(j.get("date") or ymd))
    rows = []
    for r in tables[1]["data"]:
        rows.append({
            "code": str(r[0]).strip(), "name": str(r[1]).strip(),
            "margin_buy": num(r[2]), "margin_sell": num(r[3]), "margin_repay": num(r[4]),
            "margin_prev": num(r[5]), "margin_today": num(r[6]), "margin_limit": num(r[7]),
            "short_buy": num(r[8]), "short_sell": num(r[9]), "short_repay": num(r[10]),
            "short_prev": num(r[11]), "short_today": num(r[12]), "short_limit": num(r[13]),
            "offset": num(r[14]), "note": (str(r[15]).strip() if len(r) > 15 and r[15] is not None else ""),
            "date": iso,
        })
    return {"date": iso, "market": parse_margin_market(tables[0]), "stocks": rows}


def _norm_stock_row(r: dict, iso: str) -> dict:
    """舊快取列可能缺 note (2026-10-13 那天由 margin_stocks() 寫入) / date。"""
    out = {k: r.get(k) for k in MARGIN_ALL_COLS}
    out["note"] = str(r.get("note") or "").strip()
    out["date"] = str(r.get("date") or iso)
    return out


def _norm_market(ms) -> dict | None:
    """MS 快取 (margin_daily 格式) → 補齊三個現償欄：恆等式 今日 = 前日 + 新增 − 了結 − 現償 (492 日 0 違反) 反推。"""
    if not isinstance(ms, dict) or ms.get("margin_lots") is None:
        return None
    out = {k: ms.get(k) for k in MARGIN_MARKET_KEYS}

    def fill(repay, prev, add, sub, today, nd):
        if out.get(repay) is None and all(out.get(k) is not None for k in (prev, add, sub, today)):
            out[repay] = round(out[prev] + out[add] - out[sub] - out[today], nd)
    fill("margin_repay_lots", "margin_lots_prev", "margin_buy_lots", "margin_sell_lots", "margin_lots", 0)
    fill("short_repay_lots", "short_lots_prev", "short_sell_lots", "short_buy_lots", "short_lots", 0)   # 融券：賣出 = 新增、買進 = 回補
    fill("margin_repay_amt", "margin_amt_prev", "margin_buy_amt", "margin_sell_amt", "margin_amt", 5)
    return out


def margin_stocks_hist(ymd: str) -> dict:
    """MI_MARGN selectType=ALL 單日 (可回補 2001-06~)：
    {"date": ISO, "market": {margin_lots_prev, margin_lots, margin_buy_lots, margin_sell_lots, margin_repay_lots,
                             short_lots_prev, short_lots, short_buy_lots(回補), short_sell_lots(新增), short_repay_lots,
                             margin_amt_prev, margin_amt, margin_buy_amt, margin_sell_amt, margin_repay_amt (億)} | None,
     "stocks": [{code, name, margin_buy, margin_sell, margin_repay, margin_prev, margin_today, margin_limit,
                 short_buy, short_sell, short_repay, short_prev, short_today, short_limit, offset, note, date}, ...]}
    stat 含「沒有符合」(休市 / 週間假日 / 尚未公布，同一回應) → raise NoData；其他非 OK → raise RuntimeError；兩者皆不快取。
    快取 key twse:MI_MARGN:ALL:{ymd} (list 格式，與本機 491 日快取及 margin_stocks() 相容) 讀取 TTL 10 年；
    舊 margin_stocks() 寫入的 [] 會被清掉重抓。"""
    ymd = ymd.replace("-", "")
    key, ms_key = f"twse:MI_MARGN:ALL:{ymd}", f"twse:MI_MARGN:MS:{ymd}"

    def load():
        _throttle()
        j = get_json(f"{BASE}/marginTrading/MI_MARGN", params={"response": "json", "selectType": "ALL", "date": ymd}, timeout=60)
        parsed = parse_margin_all(j, ymd)
        cache_put(ms_key, {"date": parsed["date"], **parsed["market"]})     # 同一請求順手補 margin_daily 的快取 (含現償三欄)
        return parsed["stocks"]

    rows = cached(key, TTL_FOREVER, load, allow_stale=False)
    if not isinstance(rows, list) or not rows:                             # 舊 margin_stocks() stat≠OK 寫的 [] / 壞格式 → 清掉重抓
        cache_delete(key)
        rows = cached(key, TTL_FOREVER, load, allow_stale=False)
    iso = str((rows[0] or {}).get("date") or ymd_to_iso(ymd))
    stocks = [_norm_stock_row(r, iso) for r in rows if isinstance(r, dict)]
    market = _norm_market(cache_get(ms_key, TTL_FOREVER))
    if market is None:                                                       # 舊快取日沒有 MS 快取 (理論上 494/492 日皆有)：補抓一次
        try:
            _throttle()
            market = _norm_market(margin_daily(ymd))
        except Exception as e:  # noqa: BLE001
            log.warning("MI_MARGN MS %s 補抓失敗: %s", ymd, e)
            market = None
    return {"date": iso, "market": market, "stocks": stocks}


def identity_err(rows, tol: float = 0.5):
    """恆等式檢查：融資 today = prev + buy − sell − repay；融券 today = prev + sell(新增) − buy(回補) − repay。
    任一不成立 (|殘差| > tol) 或欄位缺 → True (該列標 identity_err、不進統計)。
    rows：margin_stocks_hist 的 stocks 列 (margin_*/short_*) 或 store.margin_stock_daily 列 (m_*/s_*)；list → list[bool]、DataFrame → Series。"""
    is_df = isinstance(rows, pd.DataFrame)
    df = rows if is_df else pd.DataFrame(list(rows))
    if df.empty:
        return pd.Series([], dtype=bool) if is_df else []
    if "m_today" in df.columns:
        m, s = ("m_prev", "m_buy", "m_sell", "m_repay", "m_today"), ("s_prev", "s_sell", "s_buy", "s_repay", "s_today")
    else:
        m, s = ("margin_prev", "margin_buy", "margin_sell", "margin_repay", "margin_today"), ("short_prev", "short_sell", "short_buy", "short_repay", "short_today")

    def bad(cols):
        v = [pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index) for c in cols]
        prev, add, sub, rep, today = v
        resid = today - (prev + add - sub - rep)
        return ~(resid.abs() <= tol)          # NaN → True
    err = bad(m) | bad(s)
    return err if is_df else err.tolist()


def parse_closes(j, ymd: str) -> dict[str, list]:
    """MI_INDEX ALLBUT0999 → {code: [close, vol_lots]}；只留代號 (4~6 碼 + 可選英文字尾) 有收盤價的列 (無成交「--」不留)。"""
    _check_stat(j, f"MI_INDEX {ymd}")
    tbl = next((t for t in (j.get("tables") or []) if "證券代號" in (t.get("fields") or [])), None)
    if tbl is None or not tbl.get("data"):
        raise RuntimeError(f"MI_INDEX {ymd}: stat OK 但無個股表 (尚未公布 / 格式變更)")
    f = tbl["fields"]
    ic, ipx, ivol = f.index("證券代號"), f.index("收盤價"), f.index("成交股數")
    out: dict[str, list] = {}
    for row in tbl["data"]:
        code = str(row[ic]).strip()
        if not _CODE_RE.fullmatch(code):
            continue
        px = num(row[ipx])
        if px is None or px <= 0:
            continue
        out[code] = [px, (num(row[ivol]) or 0.0) / 1000.0]
    if not out:
        raise RuntimeError(f"MI_INDEX {ymd}: 無有效收盤列")
    return out


def closes_all(ymd: str) -> dict[str, tuple[float, float]]:
    """MI_INDEX type=ALLBUT0999 單日全市場收盤：{code: (close, vol_lots)} (約 30 KB/日，原始 249 KB 不留)。
    2004-02-11 之前 (本機判定或 TWSE 回「查詢日期小於93年2月11日」) / 休市 → raise NoData；其他失敗 raise；不快取失敗。
    快取 key twse:close_all:{ymd} 10 年。"""
    ymd = ymd.replace("-", "")
    if ymd < CLOSES_FROM:
        raise NoData(f"MI_INDEX {ymd}: 查詢日期小於 93 年 2 月 11 日")

    def load():
        _throttle()
        j = get_json(f"{BASE}/afterTrading/MI_INDEX", params={"response": "json", "type": "ALLBUT0999", "date": ymd}, timeout=60)
        return parse_closes(j, ymd)
    d = cached(f"twse:close_all:{ymd}", TTL_FOREVER, load, allow_stale=False)
    return {k: (float(v[0]), float(v[1])) for k, v in d.items()}


# ---------------------------------------------------------------- 市場成交 / 指數
def market_daily(month: str | None = None) -> pd.DataFrame:
    """每月市場成交資訊：date, shares, amount(億), trades, close, change。month='YYYYMM'。"""
    def load():
        j = get_json(f"{BASE}/afterTrading/FMTQIK",
                     params={"response": "json", "date": (month + "01") if month else ""})
        if not _ok(j):
            return []
        return [{
            "date": roc_to_iso(r[0]), "shares": num(r[1]), "amount": num(r[2]) / YI,
            "trades": num(r[3]), "close": num(r[4]), "change": num(r[5]),
        } for r in j["data"]]
    ttl = config.TTL_HISTORY if (month and month < dt.date.today().strftime("%Y%m")) else config.TTL_INTRADAY
    return pd.DataFrame(cached(f"twse:FMTQIK:{month}", ttl, load))


def index_realtime(codes: tuple[str, ...] = ("tse_t00.tw", "otc_o00.tw")) -> list[dict]:
    """即時報價 (5 秒快取)。指數代碼 tse_t00.tw (加權) / otc_o00.tw (櫃買)，個股 tse_2330.tw / otc_6488.tw。"""
    def load():
        j = get_json(MIS, params={"ex_ch": "|".join(codes), "json": "1", "delay": "0"},
                     headers={"Referer": "https://mis.twse.com.tw/stock/index.jsp"})
        out = []
        for m in j.get("msgArray", []):
            last = num(m.get("z")) or num(m.get("pz"))
            prev = num(m.get("y"))
            rec = {
                "code": m.get("c"), "name": m.get("n"), "market": m.get("ex"),
                "time": m.get("%") or m.get("t"), "date": m.get("d"),
                "last": last, "open": num(m.get("o")), "high": num(m.get("h")),
                "low": num(m.get("l")), "prev_close": prev, "volume": num(m.get("v")),
                "bid": (m.get("b") or "").split("_")[0], "ask": (m.get("a") or "").split("_")[0],
            }
            if last and prev:
                rec["change"] = last - prev
                rec["change_pct"] = (last / prev - 1) * 100
            out.append(rec)
        return out
    return cached(f"twse:mis:{'|'.join(codes)}", config.TTL_REALTIME, load)


def stock_realtime(stock_id: str) -> dict | None:
    for ex in ("tse", "otc"):
        rows = index_realtime((f"{ex}_{stock_id}.tw",))
        if rows and rows[0].get("last") is not None:
            return rows[0]
    return None


# ---------------------------------------------------------------- 個股三大法人
def t86(date: str | None = None) -> pd.DataFrame:
    """三大法人買賣超日報 (張)。"""
    def load():
        j = get_json(f"{BASE}/fund/T86", params={"response": "json", "selectType": "ALL", "date": date or ""})
        if not _ok(j):
            return []
        rows = []
        for r in j["data"]:
            rows.append({
                "code": r[0].strip(), "name": r[1].strip(),
                "foreign_net": num(r[4]) / 1000, "foreign_dealer_net": num(r[7]) / 1000,
                "trust_net": num(r[10]) / 1000, "dealer_net": num(r[11]) / 1000,
                "dealer_self_net": num(r[14]) / 1000, "dealer_hedge_net": num(r[17]) / 1000,
                "total_net": num(r[18]) / 1000, "date": ymd_to_iso(j["date"]),
            })
        return rows
    return pd.DataFrame(cached(f"twse:T86:{date}", _ttl(date), load))


# ---------------------------------------------------------------- 盤中分時 (歷史)
def index_1min(date: str) -> list[list]:
    """某日加權指數 1 分鐘收盤序列 (由每 5 秒指數統計 MI_5MINS_INDEX 取每分鐘最後一筆)。
    回傳 [[ 'HH:MM', close ], ...]；第一筆 '09:00' 為前一日收盤 (開盤參考)。date='YYYYMMDD'。"""
    def load():
        j = get_json(f"{BASE}/TAIEX/MI_5MINS_INDEX", params={"date": date, "response": "json"})
        if not _ok(j):
            return []
        out, last_min = [], None
        for r in j["data"]:
            t, v = r[0][:5], num(r[1])
            if v is None:
                continue
            if r[0] == "09:00:00":
                out.append(["09:00", v])      # 前一日收盤
                continue
            if t == last_min:
                out[-1][1] = v
            else:
                out.append([t, v])
                last_min = t
        return out
    return cached(f"twse:idx1m:{date}", 10 * 365 * 86400, load)


# ---------------------------------------------------------------- 盤中時間點 (小時模型訓練資料，2016~)
# 2026-10-12：小時模型原本用 Yahoo ^TWII 60m (只有約 2 年，且「開盤價」是 09:00:05 失真的第一筆，只反映約 53% 的真實跳空，
# 與盤中即時用 09:05 的定義不一致)。改用 TWSE 每 5 秒指數 (MI_5MINS_INDEX) 萃取各時間點價格 + 累積高低；開盤價一律取 09:05。
# 歷史 (2016-01~) 存在 data/intraday_marks.json (進 git，約 1 MB)；之後的交易日在訓練時逐日補抓 (每日 1 次請求，結果進快取)。
INTRADAY_MARKS_FILE = config.DATA_DIR / "intraday_marks.json"
MARK_TIMES = ("09:15", "09:30", "10:00", "11:00", "12:00", "13:00", "13:30")
OPEN_AT = "09:05"


def day_marks(rows: list) -> dict | None:
    """MI_5MINS_INDEX 一日資料 [['HH:MM:SS', 加權指數], ...] → {'prev_close','open','09:15'..'13:30','hi':{mark:累積高},'lo':{...}}。
    09:00:00 那筆 = 前一日收盤；沒有 13:30:00 (盤中/資料不完整) → None。高低從 09:05 起算 (避開開盤失真)。"""
    pts = [(str(t)[:8], float(v)) for t, v in rows if v is not None]
    if not pts or pts[0][0] != "09:00:00" or not any(t == "13:30:00" for t, _ in pts):
        return None
    prev_close = pts[0][1]
    xs = [(t, v) for t, v in pts if f"{OPEN_AT}:00" <= t <= "13:30:00"]
    if len(xs) < 100:
        return None
    rec = {"prev_close": round(prev_close, 2), "open": round(xs[0][1], 2), "hi": {}, "lo": {}}
    hi = lo = xs[0][1]
    i = 0
    for m in MARK_TIMES:
        T = m + ":00"
        while i < len(xs) and xs[i][0] <= T:
            hi, lo = max(hi, xs[i][1]), min(lo, xs[i][1])
            last = xs[i][1]
            i += 1
        rec[m], rec["hi"][m], rec["lo"][m] = round(last, 2), round(hi, 2), round(lo, 2)
    return rec


def _fetch_day_marks(ymd: str) -> dict | None:
    """單日：抓 MI_5MINS_INDEX → day_marks。非 OK 丟例外 (不讓 cached 把暫時失敗存成空值)。"""
    def load():
        j = get_json(f"{BASE}/TAIEX/MI_5MINS_INDEX", params={"date": ymd, "response": "json"}, timeout=60)
        if not _ok(j) or not j.get("data"):
            raise RuntimeError(f"MI_5MINS_INDEX {ymd}: {j.get('stat') if isinstance(j, dict) else type(j)}")
        rec = day_marks([[r[0], num(r[1])] for r in j["data"]])
        if rec is None:
            raise RuntimeError(f"MI_5MINS_INDEX {ymd}: 資料不完整")
        return rec
    return cached(f"twse:mi5marks:{ymd}", 10 * 365 * 86400, load, allow_stale=False)


def intraday_marks(fetch_missing: bool = True, max_fetch: int = 60) -> dict[str, dict]:
    """{ISO 日期: day_marks}。先讀 data/intraday_marks.json，再補抓檔案最後一天之後到最近收盤日 (最多 max_fetch 天)。"""
    import json as _json
    import time as _t
    days: dict[str, dict] = {}
    if INTRADAY_MARKS_FILE.exists():
        try:
            days = dict(_json.loads(INTRADAY_MARKS_FILE.read_text(encoding="utf-8")).get("days") or {})
        except Exception as e:  # noqa: BLE001
            log.warning("intraday_marks.json 讀取失敗: %s", e)
    if fetch_missing:
        now = dt.datetime.now(config.TZ)
        end = now.date() if now.strftime("%H:%M") >= "13:45" else now.date() - dt.timedelta(days=1)
        d = dt.date.fromisoformat(max(days)) + dt.timedelta(days=1) if days else dt.date(2016, 1, 1)
        hol: dict[int, set[str]] = {}
        n = 0
        while d <= end and n < max_fetch:
            iso = d.isoformat()
            if d.weekday() < 5:
                if d.year not in hol:
                    try:
                        hol[d.year] = holidays(d.year)
                    except Exception:  # noqa: BLE001
                        hol[d.year] = set()
                if iso not in hol[d.year]:
                    n += 1
                    try:
                        days[iso] = _fetch_day_marks(d.strftime("%Y%m%d"))
                    except Exception as e:  # noqa: BLE001   未列入休市表的休市日 / 暫時失敗：略過，下次再試
                        log.info("intraday marks %s skipped: %s", iso, e)
                    _t.sleep(1.0)
            d += dt.timedelta(days=1)
    return dict(sorted(days.items()))


def save_intraday_marks(days: dict[str, dict]) -> None:
    import json as _json
    INTRADAY_MARKS_FILE.write_text(_json.dumps({"source": "TWSE MI_5MINS_INDEX", "open_at": OPEN_AT, "marks": list(MARK_TIMES),
                                                "days": days}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def holidays(year: int | None = None) -> set[str]:
    """TWSE 休市日 (ISO 日期)。"""
    year = year or dt.date.today().year

    def load():
        # queryYear 參數會被 TWSE 忽略 (一律回今年)；要用 date=YYYY0101 指定年度，並核對回傳的 queryYear
        j = get_json(f"{BASE}/holidaySchedule/holidaySchedule", params={"response": "json", "date": f"{year}0101"})
        if not isinstance(j, dict) or j.get("stat", "").lower() != "ok":
            # 丟例外而不是回 []：避免把「還沒公布/暫時失敗」當成「沒有休市日」快取 12 小時 (選擇權剩餘天數會高估、IV 低估)
            raise RuntimeError(f"holidaySchedule {year}: {j.get('stat') if isinstance(j, dict) else type(j)}")
        if str(j.get("queryYear") or "") != str(year) or not j.get("data"):
            raise RuntimeError(f"holidaySchedule {year}: 回傳 {j.get('queryYear')} 年 / {len(j.get('data') or [])} 列 (尚未公布)")
        out = []
        for r in j["data"]:
            name, desc = r[1], r[2]
            if "開始交易" in name or "最後交易日" in name:
                continue
            if any(k in name + desc for k in ("放假", "無交易", "休市", "春節", "紀念日", "補假")):
                out.append(r[0])
        return out
    return set(cached(f"twse:holidays2:{year}", config.TTL_HISTORY, load))


_HOL_FAIL: dict[int, float] = {}      # 休市表抓不到的年度 → 時間 (1 小時內不重抓，避免每次呼叫都打 TWSE)


def next_trading_days(from_date: str, n: int = 3) -> list[str]:
    """from_date 之後的 n 個交易日 (跳過週末與休市日)。"""
    import time as _t
    d = dt.date.fromisoformat(from_date)
    hol_by_year: dict[int, set[str]] = {}

    def hol(y: int) -> set[str]:
        if y not in hol_by_year:
            fixed = {f"{y}-01-01", f"{y}-02-28", f"{y}-04-04", f"{y}-05-01", f"{y}-10-10"}   # 固定日期國定假日 (農曆節日無法推)
            if _t.time() - _HOL_FAIL.get(y, 0) < 3600:
                hol_by_year[y] = fixed
                return fixed
            try:
                hol_by_year[y] = holidays(y)
            except Exception as e:  # noqa: BLE001   休市表抓不到/尚未公布 → 週末 + 固定日期 (需要嚴格日曆的呼叫端先自行呼叫 holidays())
                log.warning("holidays %s unavailable, weekends + fixed dates only: %s", y, e)
                _HOL_FAIL[y] = _t.time()
                hol_by_year[y] = fixed
        return hol_by_year[y]
    out = []
    while len(out) < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in hol(d.year):
            out.append(d.isoformat())
    return out


# ---------------------------------------------------------------- 借券賣出餘額
def sbl_balance(date: str | None = None) -> dict | None:
    """借券賣出餘額 (全市場加總，張) 與個股明細。"""
    def load():
        j = get_json(f"{BASE}/marginTrading/TWT93U", params={"response": "json", "date": date or ""})
        if not _ok(j):
            return None
        rows, prev_sum, today_sum = [], 0.0, 0.0
        for r in j["data"]:
            prev, sell, ret, adj, today = (num(r[8]) or 0, num(r[9]) or 0, num(r[10]) or 0, num(r[11]) or 0, num(r[12]) or 0)
            prev_sum += prev
            today_sum += today
            rows.append({"code": r[0].strip(), "name": r[1].strip(),
                         "sbl_prev": prev / 1000, "sbl_sell": sell / 1000, "sbl_return": ret / 1000,
                         "sbl_today": today / 1000,
                         "short_prev": (num(r[2]) or 0) / 1000, "short_today": (num(r[6]) or 0) / 1000})
        return {"date": ymd_to_iso(j["date"]), "sbl_prev": prev_sum / 1000, "sbl_today": today_sum / 1000,
                "sbl_change": (today_sum - prev_sum) / 1000, "stocks": rows}
    return cached(f"twse:TWT93U:{date}", _ttl(date), load)
