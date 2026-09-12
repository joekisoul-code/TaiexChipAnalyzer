"""哪些資料源允許瀏覽器直接跨域讀取 (CORS)？"""
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
H = {"User-Agent": "Mozilla/5.0", "Origin": "https://example.github.io", "Referer": "https://example.github.io/"}
URLS = {
    "TWSE MIS 即時": "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0",
    "TWSE MIS 分時": "https://mis.twse.com.tw/stock/data/mis_ohlc_TSE.txt",
    "TWSE rwd BFI82U": "https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=json",
    "TWSE openapi": "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
    "TAIFEX openapi": "https://openapi.taifex.com.tw/v1/PutCallRatio",
    "TAIFEX mis quote": "https://mis.taifex.com.tw/futures/api/getQuoteList",
    "FinMind": "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=TAIEX&start_date=2026-09-01",
    "Yahoo chart": "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=5d",
    "HiStock": "https://histock.tw/stock/broker8.aspx",
    "TDCC opendata": "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5",
    "TAIFEX vix": "https://www.taifex.com.tw/cht/7/getVixData?filesname=20260911",
    "GitHub raw": "https://raw.githubusercontent.com/python/cpython/main/README.rst",
}
for name, url in URLS.items():
    try:
        if "getQuoteList" in url:
            r = requests.post(url, json={"MarketType": "0", "SymbolType": "F", "KindID": "1", "CID": "TXF", "ExpireMonth": "", "RowSize": "全部", "PageNo": "", "SortColumn": "", "AscDesc": "A"}, headers=H, timeout=20)
        else:
            r = requests.get(url, headers=H, timeout=20)
        acao = r.headers.get("Access-Control-Allow-Origin")
        print(f"{name:<18} [{r.status_code}] CORS={acao!s:<32} len={len(r.content)}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:<18} ERR {e}")
