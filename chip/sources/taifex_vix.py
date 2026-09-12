"""臺指選擇權波動率指數 (VIXTWN)：期交所每日 15 秒檔 `cht/7/getVixData?filesname=YYYYMMDD`。
每日檔永久快取；日收盤值累積到 SQLite (key 'vixtwn_close') 供歷史。"""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from .. import config, store
from ..http import cached, get_text, is_cached, num

log = logging.getLogger(__name__)
URL = "https://www.taifex.com.tw/cht/7/getVixData?filesname={d}"


def day(date: str) -> list[list]:
    """date 'YYYYMMDD' → [['HH:MM:SS', value], ...] (無資料回 [])。"""
    def load():
        txt = get_text(URL.format(d=date), timeout=30)
        rows = []
        for line in txt.splitlines()[2:]:
            parts = [p for p in line.split("\t") if p != ""]
            if len(parts) < 3 or not parts[1].strip().isdigit():
                continue
            t = parts[1].strip().zfill(8)
            rows.append([f"{t[:2]}:{t[2:4]}:{t[4:6]}", num(parts[2])])
        return rows
    ttl = 10 * 365 * 86400 if date < dt.date.today().strftime("%Y%m%d") else 60
    return cached(f"vixtwn:{date}", ttl, load)


def latest() -> dict | None:
    """今日 (或最近有檔的日子) 的最新值與開盤值。"""
    d = dt.date.today()
    for _ in range(7):
        rows = day(d.strftime("%Y%m%d"))
        if rows:
            vals = [v for _, v in rows if v is not None]
            out = {"date": d.isoformat(), "time": rows[-1][0], "last": vals[-1], "open": vals[0], "high": max(vals), "low": min(vals)}
            store.upsert_metrics(d.isoformat(), {"vixtwn_close": vals[-1]})
            return out
        d -= dt.timedelta(days=1)
    return None


def history(days: int = 60, throttle: float = 1.0) -> pd.DataFrame:
    """累積近 days 個交易日的日收盤 (逐日抓，節流)。回傳 date, vixtwn_close。"""
    hist = store.load_metrics(["vixtwn_close"])
    have = set(hist["date"]) if not hist.empty else set()
    d = dt.date.today()
    fetched = 0
    while fetched < days and (dt.date.today() - d).days < days * 2:
        ds = d.isoformat()
        if d.weekday() < 5 and ds not in have:
            key = f"vixtwn:{d.strftime('%Y%m%d')}"
            fresh = not is_cached(key, 10 * 365 * 86400)
            try:
                rows = day(d.strftime("%Y%m%d"))
                if rows:
                    vals = [v for _, v in rows if v is not None]
                    store.upsert_metrics(ds, {"vixtwn_close": vals[-1]})
                    fetched += 1
            except Exception as e:  # noqa: BLE001
                log.warning("vixtwn %s: %s", ds, e)
                break
            if fresh:
                time.sleep(throttle)
        d -= dt.timedelta(days=1)
    hist = store.load_metrics(["vixtwn_close"])
    return hist.sort_values("date").reset_index(drop=True) if not hist.empty else hist
