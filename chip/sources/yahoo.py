"""Yahoo Finance 加權指數 (^TWII) 小時 K，最多 730 天。用於小時級預測模型的訓練資料。"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .. import config
from ..http import cached, get_json

log = logging.getLogger(__name__)
URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"


def taiex_hourly(range_: str = "730d") -> pd.DataFrame:
    """欄位：date, time (HH:MM, bar 開始時間), open, high, low, close。台北時間。"""
    def load():
        j = get_json(URL, params={"interval": "60m", "range": range_}, timeout=60)
        res = j["chart"]["result"][0]
        ts, q = res["timestamp"], res["indicators"]["quote"][0]
        off = res["meta"].get("gmtoffset", 28800)
        rows = []
        for i, t in enumerate(ts):
            if q["close"][i] is None:
                continue
            d = dt.datetime.fromtimestamp(t, dt.UTC) + dt.timedelta(seconds=off)
            rows.append({"date": d.strftime("%Y-%m-%d"), "time": d.strftime("%H:%M"),
                         "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i]})
        return rows
    return pd.DataFrame(cached(f"yahoo:twii:60m:{range_}", config.TTL_INTRADAY, load))


def daily_marks(range_: str = "730d") -> dict[str, dict]:
    """每日各時間點價格：{date: {'prev_close','open','10:00','11:00','12:00','13:00','13:30', 'hi': {mark: cum_high}, 'lo': {...}}}
    小時 K 的 close = 該 bar 結束時的指數 (09:00 bar 的 close ≈ 10:00 價)。"""
    df = taiex_hourly(range_)
    out: dict[str, dict] = {}
    prev_close = None
    for date, g in df.groupby("date", sort=True):
        g = g.sort_values("time")
        bars = {r["time"]: r for _, r in g.iterrows()}
        if "09:00" not in bars:
            continue
        rec = {"prev_close": prev_close, "open": bars["09:00"]["open"], "hi": {}, "lo": {}}
        end_of = {"09:00": "10:00", "10:00": "11:00", "11:00": "12:00", "12:00": "13:00", "13:00": "13:30"}
        hi, lo = -1e18, 1e18
        for start, end in end_of.items():
            if start not in bars:
                break
            b = bars[start]
            hi, lo = max(hi, b["high"]), min(lo, b["low"])
            rec[end] = b["close"]
            rec["hi"][end], rec["lo"][end] = hi, lo
        if "13:30" in bars:
            rec["13:30"] = bars["13:30"]["close"]
            rec["hi"]["13:30"], rec["lo"]["13:30"] = max(hi, bars["13:30"]["high"]), min(lo, bars["13:30"]["low"])
        if "13:30" in rec:
            out[date] = rec
            prev_close = rec["13:30"]
    return out
