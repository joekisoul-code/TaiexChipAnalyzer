"""探測玩股網以外的資料源。"""
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0"}
s = requests.Session()
s.headers.update(UA)


def show(name, url, n=300, **kw):
    t = time.time()
    try:
        r = s.get(url, timeout=40, **kw)
        body = r.content.decode("utf-8-sig", errors="replace")
        print(f"[{r.status_code}] {name} len={len(body)} {time.time()-t:.1f}s\n    {body[:n].replace(chr(10), ' | ')}")
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {name}: {e}")


# FRED (免 key CSV)
for sid in ["DGS2", "DGS10", "T10Y2Y", "T10Y3M", "VIXCLS", "DTWEXBGS", "DCOILBRENTEU", "BAMLH0A0HYM2"]:
    show(f"FRED {sid}", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", 120)
# Stooq
show("stooq ^twii", "https://stooq.com/q/d/l/?s=^twii&i=d", 200)
show("stooq bdi?", "https://stooq.com/q/d/l/?s=bdi&i=d", 200)
# TAIFEX 台指 VIX
show("taifex vix openapi", "https://openapi.taifex.com.tw/v1/DailyOptionsDelta", 150)
show("taifex vix page", "https://www.taifex.com.tw/cht/7/vixQuote", 200)
show("taifex vix hist", "https://www.taifex.com.tw/cht/7/vixQuoteDown", 200)
# 台銀牌告匯率
show("BOT rates csv", "https://rate.bot.com.tw/xrt/flcsv/0/day", 300)
show("BOT USD history", "https://rate.bot.com.tw/xrt/quote/l6m/USD", 200)
# Yahoo extra symbols
for sym in ["BZ=F", "SI=F", "NG=F", "ZS=F", "ZC=F", "PL=F", "PA=F", "JPY=X", "KRW=X", "EURUSD=X", "^VXN", "^SKEW", "^VIX3M", "^FVX", "^TYX", "^IRX", "^TWII", "^KS11", "^N225", "000001.SS", "^BDI", "BDRY"]:
    try:
        r = s.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}", params={"interval": "1d", "range": "5d"}, timeout=20)
        j = r.json()
        res = j.get("chart", {}).get("result")
        if res:
            m = res[0]["meta"]
            print(f"[OK] yahoo {sym:<10} {m.get('shortName') or m.get('longName')!s:<30} last={m.get('regularMarketPrice')} tz={m.get('exchangeTimezoneName')}")
        else:
            print(f"[NO] yahoo {sym}: {j.get('chart', {}).get('error')}")
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] yahoo {sym}: {e}")
