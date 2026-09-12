"""探勘玩股網個股/排行頁面的 JSON 端點 (Playwright)。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from playwright.sync_api import sync_playwright  # noqa: E402

from chip.sources.wantgoo import STEALTH_JS, UA  # noqa: E402

PAGES = [
    "https://www.wantgoo.com/stock/2330/major-investors/main-trend",
    "https://www.wantgoo.com/stock/2330/major-investors/broker-buysell",
    "https://www.wantgoo.com/stock/2330/major-investors/concentration",
    "https://www.wantgoo.com/stock/2330/shareholding-distribution",
    "https://www.wantgoo.com/stock/2330/institutional-investors/trend",
    "https://www.wantgoo.com/stock/2330/margin-trading/synopsis",
    "https://www.wantgoo.com/stock/major-investors/broker-buy-sell-rank",
    "https://www.wantgoo.com/stock/institutional-investors/three-trade-for-trading-amount",
]
SKIP = ("all-alive", "toppopular", "all-quote-info", "beta-value", "company-profile", "monthly-revenue", "/pbr", "/per", "warrant-count",
        "commoditystate", "candlestick", "average-price", "advertisement", "/visit", "member", "payment", "event/")
captured = {}
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
    ctx = b.new_context(user_agent=UA, locale="zh-TW", viewport={"width": 1366, "height": 900})
    ctx.add_init_script(STEALTH_JS)
    page = ctx.new_page()

    def on_resp(r):
        u = r.url
        if "wantgoo.com" in u and not any(k in u for k in SKIP) and not any(k in u for k in ("/js/", "/css/", "/img/", "/lib/")):
            try:
                if "json" in r.headers.get("content-type", ""):
                    captured[u] = r.text()[:400]
            except Exception:  # noqa: BLE001
                pass
    page.on("response", on_resp)
    for url in PAGES:
        captured.clear()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=90000)
            time.sleep(7)
            print(f"\n=== {url}  title={page.title()}")
            for u, body in captured.items():
                print("  ", u.replace("https://www.wantgoo.com", ""), "\n      ", body.replace("\n", " ")[:300])
        except Exception as e:  # noqa: BLE001
            print("ERR", url, e)
    b.close()
