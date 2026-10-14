"""證券櫃檯買賣中心 (TPEx) 上櫃信用交易餘額 (margin_dyn 資料層, 2026-10-14)。

端點 www/zh-tw/margin/balance?date=YYYY/MM/DD&id=&response=json (實測 2016-10-14 611 檔 / 2024-10-14 799 / 2026-10-13 919，0.5 s、~100 KB)：
  tables[0].data   個股 20 欄：代號 名稱 前資餘額(張) 資買 資賣 現償 資餘額 資屬證金 資使用率(%) 資限額
                              前券餘額(張) 券賣 券買 券償 券餘額 券屬證金 券使用率(%) 券限額 資券相抵(張) 備註
  tables[0].summary 兩列：合計(張) (資 / 券 的 前日 新增 了結 現償 今日) 與 融資金(仟元) (五欄金額)
休市與尚未公布皆回 stat "ok" + 0 列 (C2)，無法從回應分辨 → 交易日曆 (twse.holidays + 週末) 與時刻判定 (見 margin_balance)。
只做 GET；兩次網路請求間隔 ≥ MIN_GAP 秒。
"""
from __future__ import annotations

import datetime as dt
import logging
import time

from .. import config
from ..http import cached, get_json, num
from . import NoData, NotYet

log = logging.getLogger(__name__)
BASE = "https://www.tpex.org.tw/www/zh-tw/margin/balance"
YI = 1e8
TTL_FOREVER = 10 * 365 * 86400
MIN_GAP = 2.0
PUBLISH_HOUR = 21              # 預期公布時刻 (台北)；只影響 NotYet 訊息
NODATA_GRACE_DAYS = 1          # 交易日 0 列：資料日 ≥ 今天 − 1 日 → NotYet (可能晚到)；更早 → NoData (寫 margin_nodata)
STOCK_COLS = ("code", "name", "margin_prev", "margin_buy", "margin_sell", "margin_repay", "margin_today", "margin_limit", "margin_util",
              "short_prev", "short_sell", "short_buy", "short_repay", "short_today", "short_limit", "short_util", "offset", "note", "date")
_last_net = 0.0


def _throttle() -> None:
    global _last_net
    wait = MIN_GAP - (time.time() - _last_net)
    if wait > 0:
        time.sleep(wait)
    _last_net = time.time()


def ymd_to_iso(s: str) -> str:
    s = s.strip().replace("/", "").replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _default_trading_day(iso: str) -> bool:
    """週末 → False；TWSE 休市表 (抓不到則只看週末) → False；其餘 True。"""
    d = dt.date.fromisoformat(iso)
    if d.weekday() >= 5:
        return False
    try:
        from . import twse
        return iso not in twse.holidays(d.year)
    except Exception as e:  # noqa: BLE001
        log.warning("holidays %s unavailable (%s)；只以週末判定", d.year, e)
        return True


def parse_balance(j, ymd: str) -> dict:
    """回應 → {"date", "status": "ok"|"empty", "market": {...} | None, "stocks": [...]}；stat≠ok / 非 dict → RuntimeError。"""
    if not isinstance(j, dict):
        raise RuntimeError(f"TPEx margin {ymd}: 非 dict 回應 ({type(j).__name__})")
    if str(j.get("stat") or "").lower() != "ok":
        raise RuntimeError(f"TPEx margin {ymd}: stat={str(j.get('stat'))[:60]}")
    iso = ymd_to_iso(str(j.get("date") or ymd))
    tables = j.get("tables") or []
    tbl = tables[0] if tables else {}
    data = tbl.get("data") or []
    if not data:
        return {"date": iso, "status": "empty", "market": None, "stocks": []}
    stocks = []
    for r in data:
        if len(r) < 19:
            continue
        stocks.append({
            "code": str(r[0]).strip(), "name": str(r[1]).strip(),
            "margin_prev": num(r[2]), "margin_buy": num(r[3]), "margin_sell": num(r[4]), "margin_repay": num(r[5]),
            "margin_today": num(r[6]), "margin_limit": num(r[9]), "margin_util": num(r[8]),
            "short_prev": num(r[10]), "short_sell": num(r[11]), "short_buy": num(r[12]), "short_repay": num(r[13]),
            "short_today": num(r[14]), "short_limit": num(r[17]), "short_util": num(r[16]),
            "offset": num(r[18]), "note": (str(r[19]).strip() if len(r) > 19 and r[19] is not None else ""), "date": iso,
        })
    market: dict = {}
    for row in tbl.get("summary") or []:
        label = str(row[1]) if len(row) > 1 else ""
        if label.startswith("合計"):
            market.update(margin_lots_prev=num(row[2]), margin_buy_lots=num(row[3]), margin_sell_lots=num(row[4]),
                          margin_repay_lots=num(row[5]), margin_lots=num(row[6]),
                          short_lots_prev=num(row[10]), short_sell_lots=num(row[11]), short_buy_lots=num(row[12]),
                          short_repay_lots=num(row[13]), short_lots=num(row[14]))
        elif label.startswith("融資金"):
            k = 1e3 / YI
            vals = [num(x) for x in row[2:7]]
            market.update(margin_amt_prev=_mul(vals[0], k), margin_buy_amt=_mul(vals[1], k), margin_sell_amt=_mul(vals[2], k),
                          margin_repay_amt=_mul(vals[3], k), margin_amt=_mul(vals[4], k))
    if market.get("margin_lots") is None:      # summary 缺 → 以個股加總補 (張)；金額無法補
        market.setdefault("margin_lots", sum((s["margin_today"] or 0) for s in stocks))
        market.setdefault("short_lots", sum((s["short_today"] or 0) for s in stocks))
    return {"date": iso, "status": "ok", "market": market, "stocks": stocks}


def _mul(v, k):
    return None if v is None else v * k


def margin_balance(ymd: str, *, now: dt.datetime | None = None, trading_day=None) -> dict:
    """上櫃信用交易餘額單日：{"date", "status": "ok"|"closed", "market": {...}, "stocks": [...]} (欄位見 parse_balance / STOCK_COLS；
    market 與 twse.margin_stocks_hist()["market"] 同鍵名，金額為億)。
    0 列判定：非交易日 (週末 / TWSE 休市表) → status "closed" (快取)；交易日且資料日 ≥ 今天 − NODATA_GRACE_DAYS → raise NotYet (不快取，
    訊息註明是否已過 21:00)；交易日且更早 → raise NoData (不快取；回補端寫 margin_nodata)。stat≠ok / 連線失敗 → raise (不快取)。
    快取 key tpex:margin:{ymd} 10 年。now / trading_day 可注入 (測試)。"""
    ymd = ymd.replace("/", "").replace("-", "")
    iso = ymd_to_iso(ymd)
    now = now or dt.datetime.now(config.TZ)
    is_td = trading_day or _default_trading_day

    def load():
        _throttle()
        j = get_json(BASE, params={"date": f"{ymd[:4]}/{ymd[4:6]}/{ymd[6:8]}", "id": "", "response": "json"}, timeout=60)
        p = parse_balance(j, ymd)
        if p["status"] == "ok":
            return p
        if not is_td(iso):
            return {"date": iso, "status": "closed", "market": None, "stocks": []}
        if iso >= (now.date() - dt.timedelta(days=NODATA_GRACE_DAYS)).isoformat():
            late = "已過" if (iso == now.date().isoformat() and now.hour >= PUBLISH_HOUR) or iso < now.date().isoformat() else "預期"
            raise NotYet(f"TPEx margin {iso}: 交易日 0 列，尚未公布 ({late} {PUBLISH_HOUR}:00)")
        raise NoData(f"TPEx margin {iso}: 交易日 0 列 (資料缺；TPEx 回補下限約 2007)")

    return cached(f"tpex:margin:{ymd}", TTL_FOREVER, load, allow_stale=False)
