import re
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0"}
t = requests.get("https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5", headers=UA, timeout=60).content.decode("utf-8-sig", errors="replace")
lines = t.splitlines()
print("total lines", len(lines))
hits = [ln for ln in lines if "2330" in ln][:20]
print("lines containing 2330:", hits)
codes = sorted({ln.split(",")[1] for ln in lines[1:] if "," in ln})
print("n codes", len(codes), codes[:5], [c for c in codes if c.startswith("23")][:10], [c for c in codes if c.startswith("006")][:12])
s = requests.Session()
s.headers.update(UA)
r = s.get("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", timeout=30)
dates = re.findall(r'<option value="(\d{8})"', r.text)
i = r.text.find("_token")
print("token ctx:", r.text[i - 80:i + 120].replace("\n", " ") if i >= 0 else "none")
data = {"scaDates": dates[1], "scaDate": dates[1], "SqlMethod": "StockNo", "StockNo": "2330", "radioStockNo": "2330", "StockName": "", "sub": "查詢"}
r2 = s.post("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", data=data, timeout=30)
j = r2.text.find("持股分級")
print("POST ctx:", re.sub(r"\s+", " ", r2.text[j:j + 1500]) if j >= 0 else r2.text[:500])
