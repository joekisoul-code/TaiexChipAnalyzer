"""SQLite：把每日只提供「最新一日」的指標 (PCR、大額交易人、借券、維持率…) 累積成歷史。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from . import config


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    try:
        c.execute("CREATE TABLE IF NOT EXISTS daily_metrics (date TEXT, key TEXT, value REAL, PRIMARY KEY(date,key))")
        c.execute("CREATE TABLE IF NOT EXISTS snapshots (date TEXT, source TEXT, payload TEXT, PRIMARY KEY(date,source))")
        c.execute("CREATE TABLE IF NOT EXISTS intraday (ts TEXT, key TEXT, value REAL, PRIMARY KEY(ts,key))")
        yield c
        c.commit()
    finally:
        c.close()


def upsert_metrics(date: str, metrics: dict) -> None:
    rows = [(date, k, float(v)) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v]
    if not rows:
        return
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO daily_metrics VALUES (?,?,?)", rows)


def upsert_frame(df: pd.DataFrame, cols: list[str]) -> None:
    if df is None or df.empty:
        return
    rows = []
    for _, r in df.iterrows():
        for k in cols:
            v = r.get(k)
            if v is not None and v == v:
                rows.append((str(r["date"])[:10], k, float(v)))
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO daily_metrics VALUES (?,?,?)", rows)


def load_metrics(keys: list[str] | None = None) -> pd.DataFrame:
    with conn() as c:
        if keys:
            q = f"SELECT date,key,value FROM daily_metrics WHERE key IN ({','.join('?' * len(keys))})"
            df = pd.read_sql_query(q, c, params=keys)
        else:
            df = pd.read_sql_query("SELECT date,key,value FROM daily_metrics", c)
    if df.empty:
        return pd.DataFrame(columns=["date"] + (keys or []))
    return df.pivot_table(index="date", columns="key", values="value", aggfunc="last").reset_index()


def upsert_intraday(ts: str, metrics: dict) -> None:
    rows = [(ts, k, float(v)) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v]
    if rows:
        with conn() as c:
            c.executemany("INSERT OR REPLACE INTO intraday VALUES (?,?,?)", rows)


def load_intraday(date: str) -> pd.DataFrame:
    """某日的盤中時間序列 (ts 為索引，欄位為指標)。"""
    with conn() as c:
        df = pd.read_sql_query("SELECT ts,key,value FROM intraday WHERE ts LIKE ?", c, params=(date + "%",))
    if df.empty:
        return df
    return df.pivot_table(index="ts", columns="key", values="value", aggfunc="last").reset_index()


def save_snapshot(date: str, source: str, payload) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?)",
                  (date, source, json.dumps(payload, ensure_ascii=False, default=str)))
