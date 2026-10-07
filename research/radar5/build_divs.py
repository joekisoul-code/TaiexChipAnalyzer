"""研究 r7 共用資料：全上市個股 Yahoo 除息/分割事件 (range=10y, events=div,split) → divs.parquet (code, date, amount, kind)。
只讀；快取鍵 yh:events:{sym}:10y (chip.http.cached，長 TTL)。日期以台北時間 (Yahoo 時戳 +8h) 取日。"""
import datetime as dt, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

sys.path.insert(0, "D:/TaiexChipAnalyzer")
from chip.http import cached  # noqa: E402
from chip.predict import treasure_live as L  # noqa: E402

DIR = Path(__file__).parent
codes = sorted(pd.read_parquet(DIR / "twse_10y.parquet", columns=["code"])["code"].unique())
TZ = dt.timezone(dt.timedelta(hours=8))


def fetch(sym):
    j = L.session().get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"range": "10y", "interval": "1d", "events": "div,split"}, headers=L.UA, timeout=40).json()
    res = j["chart"]["result"][0]; ev = res.get("events") or {}
    out = []
    for v in (ev.get("dividends") or {}).values():
        out.append({"date": dt.datetime.fromtimestamp(int(v["date"]), TZ).date().isoformat(), "amount": float(v["amount"]), "kind": "div"})
    for v in (ev.get("splits") or {}).values():
        out.append({"date": dt.datetime.fromtimestamp(int(v["date"]), TZ).date().isoformat(), "amount": float(v.get("numerator", 1)) / float(v.get("denominator", 1) or 1), "kind": "split"})
    return out


def one(c):
    for suf in (".TW", ".TWO"):
        for _ in range(3):
            try:
                rows = cached(f"yh:events:{c}{suf}:10y", 10 ** 9, lambda s=c + suf: fetch(s), allow_stale=True)
                return [dict(r, code=c) for r in rows]
            except Exception:  # noqa: BLE001
                time.sleep(2)
        # .TW 失敗才試 .TWO
    return []


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    rows = [r for rs in ex.map(one, codes) for r in rs]
D = pd.DataFrame(rows).drop_duplicates(["code", "date", "kind"]).sort_values(["code", "date"])
D.to_parquet(DIR / "divs.parquet", index=False)
print("codes", len(codes), "events", len(D), "with div", D[D.kind == "div"]["code"].nunique(), "splits", int((D.kind == "split").sum()), "secs", round(time.time() - t0))
