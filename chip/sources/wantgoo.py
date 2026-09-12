"""玩股網 WantGoo (需 Playwright 真實瀏覽器；純 HTTP 會被 Cloudflare 擋 400)。

取得：
- 資券進出行情 (含 大盤融資維持率、券資比) 表格
- 借券賣出餘額增減、外資買賣超
- 八大公股銀行買賣動向 (各行庫每日金額/張數)

安裝：pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import pandas as pd

from .. import config
from ..http import cached, num

log = logging.getLogger(__name__)
UA = config.USER_AGENT
STEALTH_JS = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
TAIEX_PAGE = "https://www.wantgoo.com/stock/margin-trading/market-price/taiex"
BANK_PAGE = "https://www.wantgoo.com/stock/public-bank/trend"
BANK_MAP = {"tcb": "合庫", "land": "土銀", "bot": "台銀", "tbb": "台企銀",
            "chb": "彰銀", "first": "第一金", "mega": "兆豐銀", "hncb": "華南永昌"}


def available() -> bool:
    if not config.WANTGOO_ENABLED:
        return False
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_all(jobs: dict[str, tuple[str, list[str]]]) -> dict:
    """jobs = {name: (page_url, [json paths to fetch from that page])} -> {name: {"text": main_text, path: json}}"""
    from playwright.sync_api import sync_playwright

    out: dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA, locale="zh-TW", viewport={"width": 1366, "height": 900})
        ctx.add_init_script(STEALTH_JS)
        page = ctx.new_page()
        for name, (url, paths) in jobs.items():
            page.goto(url, wait_until="networkidle", timeout=90_000)
            time.sleep(2.5)
            res = {"text": page.inner_text("main")}
            if paths:
                res.update(page.evaluate(
                    """async (paths) => { const out = {};
                       for (const u of paths) {
                         try { const r = await fetch(u, {headers: {'X-Requested-With': 'XMLHttpRequest'}});
                               out[u] = r.ok ? await r.json() : {"_error": r.status}; }
                         catch (e) { out[u] = {"_error": String(e)}; } }
                       return out; }""", paths))
            out[name] = res
        browser.close()
    return out


def _parse_margin_table(text: str, year_hint: int) -> list[dict]:
    """解析『資券進出行情』表格文字行：MM/DD 融資餘額 增減 維持率% 融券餘額 增減 券資比 收盤 漲跌% 成交量"""
    rows = []
    pat = re.compile(r"^(\d{2})/(\d{2})\t([-\d.,]+)\t([-\d.,]+)\t([\d.]+)%\t([-\d,]+)\t([-\d,]+)\t([\d.]+)\t([\d.,]+)\t([-\d.]+)\t([\d,]+)")
    prev_month = None
    year = year_hint
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        mo, d = int(m.group(1)), int(m.group(2))
        if prev_month is not None and mo > prev_month:   # 表格由新到舊，月份回升代表跨年
            year -= 1
        prev_month = mo
        rows.append({
            "date": f"{year:04d}-{mo:02d}-{d:02d}",
            "margin_amt": num(m.group(3)), "margin_amt_chg": num(m.group(4)),
            "maint_ratio": num(m.group(5)), "short_lots": num(m.group(6)), "short_lots_chg": num(m.group(7)),
            "short_margin_ratio": num(m.group(8)), "close": num(m.group(9)), "change_pct": num(m.group(10)),
            "volume": num(m.group(11)),
        })
    return rows


def _parse_headline(text: str) -> dict:
    out = {}
    for label, key in [("大盤融資維持率", "maint_ratio"), ("融資餘額(億)", "margin_amt"), ("融券餘額(張)", "short_lots"),
                       ("借券賣出餘額增減張數", "sbl_chg_lots"), ("借券賣出餘額增減金額(億)", "sbl_chg_amt"),
                       ("外資買賣超(億)", "foreign_net")]:
        m = re.search(re.escape(label) + r"\s*\n\s*([-\d.,]+)%?", text)
        if m:
            out[key] = num(m.group(1))
    m = re.search(r"K線\s*\n\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        out["date"] = m.group(1)
    return out


def fetch_market() -> dict | None:
    """回傳 {'headline': {...}, 'margin_table': DataFrame, 'banks': DataFrame, 'raw': {...}}；無 Playwright 回 None。"""
    if not available():
        return None

    def load():
        jobs = {
            "taiex": (TAIEX_PAGE, [
                "/stock/total0000/margin-trading/historical-foreign-short-lending-statistic-long-term",
                "/stock/0000/margin-trading/historical-borrowing-balance-long-term",
            ]),
            "banks": (BANK_PAGE, ["/stock/public-bank/trend-data?market=-1"]),
        }
        raw = _fetch_all(jobs)
        head = _parse_headline(raw["taiex"]["text"])
        year = int(head["date"][:4]) if head.get("date") else dt.date.today().year
        table = _parse_margin_table(raw["taiex"]["text"], year)
        banks = []
        for r in raw["banks"].get("/stock/public-bank/trend-data?market=-1", []) or []:
            if not isinstance(r, dict) or "date" not in r:
                continue
            rec = {"date": r["date"][:10]}
            total_amt = total_cnt = 0.0
            for k, cn in BANK_MAP.items():
                v = r.get(k) or {}
                rec[f"money_{cn}"] = v.get("amount")
                rec[f"lots_{cn}"] = v.get("count")
                total_amt += v.get("amount") or 0
                total_cnt += v.get("count") or 0
            rec["gov8_net"] = total_amt       # 萬元
            rec["gov8_lots"] = total_cnt
            banks.append(rec)
        sbl = []
        for r in raw["taiex"].get("/stock/total0000/margin-trading/historical-foreign-short-lending-statistic-long-term", []) or []:
            if isinstance(r, dict) and "date" in r:
                sbl.append({"date": _ms(r["date"]), "sbl_chg_lots": r.get("todayVolumeChange"),
                            "sbl_chg_amt": r.get("todayAmountChange")})
        return {"headline": head, "margin_table": table, "banks": banks, "sbl": sbl}

    try:
        d = cached("wantgoo:market", config.TTL_INTRADAY, load)
    except Exception as e:  # noqa: BLE001
        log.warning("wantgoo 抓取失敗: %s", e)
        return None
    return {
        "headline": d["headline"],
        "margin_table": pd.DataFrame(d["margin_table"]).sort_values("date").reset_index(drop=True) if d["margin_table"] else pd.DataFrame(),
        "banks": pd.DataFrame(d["banks"]).sort_values("date").reset_index(drop=True) if d["banks"] else pd.DataFrame(),
        "sbl": pd.DataFrame(d["sbl"]).sort_values("date").reset_index(drop=True) if d["sbl"] else pd.DataFrame(),
    }


STOCK_ENDPOINTS = {
    "main_trend": "/stock/{id}/major-investors/main-trend-data",
    "broker": "/stock/{id}/major-investors/broker-buysell-data",
    "concentration": "/stock/{id}/major-investors/concentration-data",
    "shareholding": "/stock/{id}/historical-shareholding-distribution?top=100",
    "inst_trend": "/stock/{id}/institutional-investors/trend-data?topdays=250",
    "foreign_hist": "/stock/{id}/institutional-investors/foreign/historical-net-buy-sell",
    "trust_hist": "/stock/{id}/institutional-investors/investment-trust/historical-net-buy-sell",
    "dealer_hist": "/stock/{id}/institutional-investors/dealer/historical-net-buy-sell",
    "margin_hist": "/stock/{id}/margin-trading/historical-lending-balance",
    "short_hist": "/stock/{id}/margin-trading/historical-borrowing-balance",
    "sbl_hist": "/stock/{id}/margin-trading/historical-foreign-short-lending",
    "candles": "/investrue/{id}/historical-daily-candlesticks?before={before}&top=300",
}


def fetch_stocks(stock_ids: list[str], ttl: int = config.TTL_INTRADAY) -> dict[str, dict]:
    """一次瀏覽器工作階段抓多檔個股的籌碼 JSON (主力、券商均價、大戶、股權分散、法人、融資券、借券、K 線)。
    回傳 {stock_id: {key: json}}；無 Playwright 回 {}。"""
    if not available() or not stock_ids:
        return {}
    before = int(dt.datetime.now(config.TZ).timestamp() * 1000)
    # 部分端點只接受其所屬頁面發出的請求 (Referer 檢查)
    PAGE_FOR = {"broker": "major-investors/broker-buysell", "concentration": "major-investors/concentration",
                "inst_trend": "institutional-investors/trend"}
    todo = [s for s in stock_ids if not _cached_ok(f"wantgoo:stock:{s}", ttl)]
    out: dict[str, dict] = {}
    if todo:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
                ctx = browser.new_context(user_agent=UA, locale="zh-TW", viewport={"width": 1366, "height": 900})
                ctx.add_init_script(STEALTH_JS)
                page = ctx.new_page()
                js = """async (paths) => { const out = {};
                         for (const [k, u] of Object.entries(paths)) {
                           try { const r = await fetch(u, {headers: {'X-Requested-With': 'XMLHttpRequest'}});
                                 out[k] = r.ok ? await r.json() : {"_error": r.status}; }
                           catch (e) { out[k] = {"_error": String(e)}; } }
                         return out; }"""
                for sid in todo:
                    try:
                        res: dict = {}
                        # ETF 會被導向 /stock/etf/{id 小寫}/...，端點路徑也要跟著改
                        page.goto(f"https://www.wantgoo.com/stock/{sid}/major-investors/main-trend", wait_until="load", timeout=90_000)
                        time.sleep(2)
                        m = re.match(r"https://www\.wantgoo\.com(/stock/(?:etf/)?[^/]+)/", page.url)
                        base = m.group(1) if m else f"/stock/{sid}"
                        groups: dict[str, dict] = {"major-investors/main-trend": {}}
                        for k, v in STOCK_ENDPOINTS.items():
                            path = v.format(id=sid, before=before)
                            if path.startswith(f"/stock/{sid}/"):
                                path = base + path[len(f"/stock/{sid}"):]
                            groups.setdefault(PAGE_FOR.get(k, "major-investors/main-trend"), {})[k] = path
                        for pg, paths in groups.items():
                            for attempt in range(2):
                                try:
                                    page.goto(f"https://www.wantgoo.com{base}/{pg}", wait_until="load", timeout=90_000)
                                    time.sleep(3.5)
                                    res.update(page.evaluate(js, paths))
                                    break
                                except Exception as e:  # noqa: BLE001
                                    if attempt == 1:
                                        log.warning("wantgoo %s %s: %s", sid, pg, e)
                                    time.sleep(2)
                        _cache_put(f"wantgoo:stock:{sid}", res)
                    except Exception as e:  # noqa: BLE001
                        log.warning("wantgoo stock %s failed: %s", sid, e)
                browser.close()
        except Exception as e:  # noqa: BLE001
            log.warning("wantgoo stocks failed: %s", e)
    for sid in stock_ids:
        d = _cache_get(f"wantgoo:stock:{sid}")
        if d:
            out[sid] = d
    return out


def _cached_ok(key: str, ttl: int) -> bool:
    from ..http import is_cached
    return is_cached(key, ttl)


def _cache_put(key: str, data) -> None:
    cached(key, 0, lambda: data)


def _cache_get(key: str):
    try:
        return cached(key, 10 * 365 * 86400, lambda: (_ for _ in ()).throw(RuntimeError("no cache")))
    except Exception:  # noqa: BLE001
        return None


def _ms(v) -> str:
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(v / 1000, dt.UTC).date().isoformat()
    return str(v)[:10]
