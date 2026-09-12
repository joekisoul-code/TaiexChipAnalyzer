import re
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0"}
r = requests.get("https://www.taifex.com.tw/cht/7/vixMinNew", headers=UA, timeout=30)
t = r.text
print("vixMinNew", r.status_code, len(t))
for m in re.finditer(r"<form[^>]*>", t):
    print(m.group(0)[:200])
for m in re.finditer(r'<(input|select)[^>]*name="([^"]+)"[^>]*>', t):
    print("  ", m.group(0)[:160])
i = t.find("下載")
print(re.sub(r"\s+", " ", t[max(0, i - 500):i + 300]))
r2 = requests.get("https://mis.taifex.com.tw/futures/VolatilityQuotes/", headers=UA, timeout=30)
print("mis vol page", r2.status_code, len(r2.text))
for m in set(re.findall(r"(api/[A-Za-z/]+)", r2.text)):
    print("  api:", m)
# 猜 API
for body in [{"MarketType": "0", "SymbolType": "V"}, {}]:
    try:
        r3 = requests.post("https://mis.taifex.com.tw/futures/api/getVolatilityQuotes", json=body, headers={**UA, "Referer": "https://mis.taifex.com.tw/futures/VolatilityQuotes/"}, timeout=20)
        print("getVolatilityQuotes", body, r3.status_code, r3.text[:300])
    except Exception as e:  # noqa: BLE001
        print("ERR", e)
