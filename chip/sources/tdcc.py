"""集保結算所 (TDCC) 集保戶股權分散表 open data (每週五更新，全部上市櫃含 ETF)。

級距 (持股分級)：1: 1-999 股, 2: 1,000-5,000, 3: 5,001-10,000, 4: 10,001-15,000, 5: 15,001-20,000, 6: 20,001-30,000,
7: 30,001-40,000, 8: 40,001-50,000, 9: 50,001-100,000, 10: 100,001-200,000, 11: 200,001-400,000, 12: 400,001-600,000,
13: 600,001-800,000, 14: 800,001-1,000,000, 15: 1,000,001 以上, 16: 差異數調整, 17: 合計。
大戶 (>400 張) = 12~15；>1000 張 = 15；散戶 (<10 張) = 1~3。
"""
from __future__ import annotations

import csv
import io
import logging

import pandas as pd

from .. import config, store
from ..http import cached, get_text

log = logging.getLogger(__name__)
URL = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"


def latest_all() -> pd.DataFrame:
    """最新一週全部證券：date, code, level, people, shares, ratio"""
    def load():
        txt = get_text(URL, timeout=120).lstrip("﻿")
        rows = []
        for r in csv.reader(io.StringIO(txt)):
            if len(r) < 6 or not r[0].isdigit():
                continue
            rows.append({"date": f"{r[0][:4]}-{r[0][4:6]}-{r[0][6:8]}", "code": r[1].strip(), "level": int(r[2]),
                         "people": int(r[3] or 0), "shares": int(r[4] or 0), "ratio": float(r[5] or 0)})
        return rows
    return pd.DataFrame(cached("tdcc:opendata:1-5", config.TTL_DAILY, load))


def holder_summary(code: str, persist: bool = True) -> dict | None:
    """某證券最新一週：big400 / big1000 / retail10 (%) 與 date；並累積到 SQLite 供歷史。"""
    df = latest_all()
    s = df[df["code"] == code]
    if s.empty:
        return None
    r = s.set_index("level")["ratio"]
    out = {"date": str(s["date"].iloc[0]), "big400": round(float(sum(r.get(i, 0) for i in (12, 13, 14, 15))), 2),
           "big1000": round(float(r.get(15, 0)), 2), "retail10": round(float(sum(r.get(i, 0) for i in (1, 2, 3))), 2),
           "people": int(s[s["level"] == 17]["people"].iloc[0]) if (s["level"] == 17).any() else None}
    if persist:
        try:
            store.upsert_metrics(out["date"], {f"tdcc:{code}:big400": out["big400"], f"tdcc:{code}:big1000": out["big1000"],
                                               f"tdcc:{code}:retail10": out["retail10"]})
        except Exception as e:  # noqa: BLE001
            log.debug("tdcc persist failed: %s", e)
    return out


def holder_history(code: str) -> pd.DataFrame:
    """由 SQLite 累積的週資料 (程式每週執行才會累積)。"""
    df = store.load_metrics([f"tdcc:{code}:big400", f"tdcc:{code}:big1000", f"tdcc:{code}:retail10"])
    if df.empty:
        return df
    return df.rename(columns={f"tdcc:{code}:big400": "big400", f"tdcc:{code}:big1000": "big1000", f"tdcc:{code}:retail10": "retail20"})
