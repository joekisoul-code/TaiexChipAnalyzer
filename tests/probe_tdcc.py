import re
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0"}
s = requests.Session()
s.headers.update(UA)
r = s.get("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", timeout=30)
print("GET", r.status_code, len(r.text))
tok = re.search(r'name="_token"\s+value="([^"]+)"', r.text) or re.search(r'_token"[^>]*value="([^"]+)"', r.text)
dates = re.findall(r'<option value="(\d{8})"', r.text)
print("token", bool(tok), "dates", dates[:3], "...", dates[-2:], len(dates))
if dates:
    data = {"scaDates": dates[1], "scaDate": dates[1], "SqlMethod": "StockNo", "StockNo": "2330", "radioStockNo": "2330", "StockName": "", "sub": "查詢"}
    if tok:
        data["_token"] = tok.group(1)
    r2 = s.post("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", data=data, timeout=30)
    print("POST", r2.status_code, len(r2.text))
    rows = re.findall(r"<tr[^>]*>\s*<td[^>]*>\s*(\d+)\s*</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>", r2.text, re.S)
    print(rows[:3], rows[-3:], len(rows))
# open data: check 2330 exists
t = requests.get("https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5", headers=UA, timeout=60).content.decode("utf-8-sig", errors="replace")
lines = [ln for ln in t.splitlines() if ",2330," in ln]
print("opendata 2330:", lines[:17])
