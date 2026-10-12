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

import pandas as pd

from .. import config
from ..http import cached, get_json, num

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
