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
        # margin_dyn (2026-10-14)：個股融資融券日面板 (Tier-2 可重建；TWSE MI_MARGN ALL + MI_INDEX 收盤) 與 無資料日 (休市 / 缺日不重抓)
        c.execute("CREATE TABLE IF NOT EXISTS margin_stock_daily (date TEXT, code TEXT, mkt TEXT, name TEXT, note TEXT,"
                  " m_buy REAL, m_sell REAL, m_repay REAL, m_prev REAL, m_today REAL, m_limit REAL,"
                  " s_buy REAL, s_sell REAL, s_repay REAL, s_prev REAL, s_today REAL, s_limit REAL, offset REAL,"
                  " close REAL, vol_lots REAL, PRIMARY KEY(date, code))")
        c.execute("CREATE INDEX IF NOT EXISTS ix_msd_code ON margin_stock_daily(code, date)")
        c.execute("CREATE TABLE IF NOT EXISTS margin_nodata (date TEXT, src TEXT, PRIMARY KEY(date, src))")
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


BID_ASK_FIX_DATE = "2026-10-18"   # 同 realtime.BID_ASK_FIX_DATE (RT-1：此日前 bid_ask_ratio 實為 委賣/委買)


def normalize_bid_ask(df: pd.DataFrame) -> pd.DataFrame:
    """10-18 RT-1 審查：把舊口徑 bid_ask_ratio (無 bid_ask_conv 旗標且 ts < BID_ASK_FIX_DATE，實為 委賣/委買) 取倒數成 委買/委賣；
    新列 (bid_ask_conv=1) 不動。不改資料庫，只改讀出的副本。"""
    if df is None or df.empty or "bid_ask_ratio" not in df:
        return df
    d = df.copy()
    conv = d["bid_ask_conv"] if "bid_ask_conv" in d else pd.Series(float("nan"), index=d.index)
    old = conv.isna() & (d["ts"].astype(str).str[:10] < BID_ASK_FIX_DATE) & (pd.to_numeric(d["bid_ask_ratio"], errors="coerce") > 0)
    d.loc[old, "bid_ask_ratio"] = 1.0 / d.loc[old, "bid_ask_ratio"].astype(float)
    return d


def load_intraday(date: str, fix_bid_ask: bool = False) -> pd.DataFrame:
    """某日的盤中時間序列 (ts 為索引，欄位為指標)。
    注意 (RT-1)：BID_ASK_FIX_DATE 前寫入的 bid_ask_ratio 方向相反 (委賣/委買)；fix_bid_ask=True 時讀出已換成 委買/委賣。"""
    with conn() as c:
        df = pd.read_sql_query("SELECT ts,key,value FROM intraday WHERE ts LIKE ?", c, params=(date + "%",))
    if df.empty:
        return df
    df = df.pivot_table(index="ts", columns="key", values="value", aggfunc="last").reset_index()
    return normalize_bid_ask(df) if fix_bid_ask else df


def save_snapshot(date: str, source: str, payload) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?)",
                  (date, source, json.dumps(payload, ensure_ascii=False, default=str)))


def load_snapshots(source: str, limit: int = 10) -> list[tuple[str, dict]]:
    """某來源最近 limit 天的快照，最新在前：[(date, payload), ...]"""
    with conn() as c:
        rows = c.execute("SELECT date,payload FROM snapshots WHERE source=? ORDER BY date DESC LIMIT ?", (source, limit)).fetchall()
    out = []
    for d, p in rows:
        try:
            out.append((d, json.loads(p)))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------- margin_dyn：個股融資融券日面板 / 無資料日
MARGIN_STOCK_COLS = ["date", "code", "mkt", "name", "note",
                     "m_buy", "m_sell", "m_repay", "m_prev", "m_today", "m_limit",
                     "s_buy", "s_sell", "s_repay", "s_prev", "s_today", "s_limit", "offset",
                     "close", "vol_lots"]
_SRC2COL = {"margin_buy": "m_buy", "margin_sell": "m_sell", "margin_repay": "m_repay", "margin_prev": "m_prev", "margin_today": "m_today",
            "margin_limit": "m_limit", "short_buy": "s_buy", "short_sell": "s_sell", "short_repay": "s_repay", "short_prev": "s_prev",
            "short_today": "s_today", "short_limit": "s_limit", "offset": "offset"}
_MSD_SQL = f"INSERT OR REPLACE INTO margin_stock_daily ({','.join(MARGIN_STOCK_COLS)}) VALUES ({','.join('?' * len(MARGIN_STOCK_COLS))})"


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def margin_rows(stocks, date: str | None = None, mkt: str = "twse", closes: dict | None = None) -> list[tuple]:
    """來源列 (twse.margin_stocks_hist()["stocks"] / tpex.margin_balance()["stocks"]，鍵 margin_* / short_* / offset / note)
    → margin_stock_daily 表列 (MARGIN_STOCK_COLS 順序的 tuple)。closes = twse.closes_all(ymd) {code: (close, vol_lots)}，缺 → None。"""
    out = []
    for s in stocks or []:
        code = str(s.get("code") or "").strip()
        if not code:
            continue
        px = (closes or {}).get(code)
        row = [str(s.get("date") or date or "")[:10], code, mkt, str(s.get("name") or ""), str(s.get("note") or "")]
        row += [_f(s.get(k)) for k in ("margin_buy", "margin_sell", "margin_repay", "margin_prev", "margin_today", "margin_limit",
                                        "short_buy", "short_sell", "short_repay", "short_prev", "short_today", "short_limit", "offset")]
        row += [_f(px[0]) if px else None, _f(px[1]) if px and len(px) > 1 else None]
        out.append(tuple(row))
    return out


def upsert_margin_stocks(rows) -> int:
    """批次寫入 (executemany, INSERT OR REPLACE；同 (date, code) 重入覆蓋)。rows：margin_rows() 的 tuple 列、
    表欄名 dict 列、或含表欄名的 DataFrame。回寫入列數。"""
    if rows is None:
        return 0
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return 0
        df = rows.copy()
        for c in MARGIN_STOCK_COLS:
            if c not in df:
                df[c] = None
        df = df[MARGIN_STOCK_COLS].astype(object).where(df[MARGIN_STOCK_COLS].notna(), None)
        data = [tuple(r) for r in df.itertuples(index=False, name=None)]
    else:
        rows = list(rows)
        if not rows:
            return 0
        if isinstance(rows[0], dict):
            data = [tuple(str(r.get(c) or "") if c in ("date", "code", "mkt", "name", "note") else _f(r.get(c)) for c in MARGIN_STOCK_COLS)
                    for r in rows]
        else:
            data = [tuple(r) for r in rows]
    with conn() as c:
        c.executemany(_MSD_SQL, data)
    return len(data)


def update_margin_closes(date: str, closes: dict, mkt: str = "twse") -> int:
    """既有面板列補收盤 / 成交張 (closes = {code: (close, vol_lots)})。回更新列數。"""
    if not closes:
        return 0
    rows = [(_f(v[0]), _f(v[1]) if len(v) > 1 else None, date[:10], str(k), mkt) for k, v in closes.items()]
    with conn() as c:
        cur = c.executemany("UPDATE margin_stock_daily SET close=?, vol_lots=? WHERE date=? AND code=? AND mkt=?", rows)
        return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0


def load_margin_stocks(start: str | None = None, end: str | None = None, codes=None, mkt: str | None = "twse",
                       cols: list[str] | None = None) -> pd.DataFrame:
    """讀面板：date 區間 [start, end]、codes 子集、mkt (None = 全部)；依 (code, date) 排序。"""
    sel = cols or MARGIN_STOCK_COLS
    q, args = f"SELECT {','.join(sel)} FROM margin_stock_daily WHERE 1=1", []
    if start:
        q += " AND date>=?"; args.append(start[:10])
    if end:
        q += " AND date<=?"; args.append(end[:10])
    if mkt:
        q += " AND mkt=?"; args.append(mkt)
    if codes is not None:
        codes = [str(x) for x in codes]
        if not codes:
            return pd.DataFrame(columns=sel)
        q += f" AND code IN ({','.join('?' * len(codes))})"; args += codes
    q += " ORDER BY code, date"
    with conn() as c:
        return pd.read_sql_query(q, c, params=args)


def margin_dates(mkt: str = "twse", start: str | None = None, end: str | None = None) -> list[str]:
    """面板已有的日期 (遞增)。"""
    q, args = "SELECT DISTINCT date FROM margin_stock_daily WHERE mkt=?", [mkt]
    if start:
        q += " AND date>=?"; args.append(start[:10])
    if end:
        q += " AND date<=?"; args.append(end[:10])
    with conn() as c:
        return [r[0] for r in c.execute(q + " ORDER BY date", args).fetchall()]


def margin_coverage(mkt: str = "twse") -> dict:
    with conn() as c:
        d0, d1, n, rows = c.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date), COUNT(*) FROM margin_stock_daily WHERE mkt=?", (mkt,)).fetchone()
    return {"from": d0, "to": d1, "days": int(n or 0), "rows": int(rows or 0)}


def mark_nodata(date: str, src: str) -> None:
    """休市 / 無資料日 (src ∈ twse_all / twse_close / tpex …) → 回補不再重抓。"""
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO margin_nodata VALUES (?,?)", (date[:10], src))


def unmark_nodata(date: str, src: str | None = None) -> None:
    with conn() as c:
        if src is None:
            c.execute("DELETE FROM margin_nodata WHERE date=?", (date[:10],))
        else:
            c.execute("DELETE FROM margin_nodata WHERE date=? AND src=?", (date[:10], src))


def nodata_dates(src: str) -> set[str]:
    with conn() as c:
        return {r[0] for r in c.execute("SELECT date FROM margin_nodata WHERE src=?", (src,)).fetchall()}
