"""FRED (美國聖路易聯準銀行) 免 key CSV：利率、殖利率曲線、廣義美元指數、高收益債利差、VIX 收盤。"""
from __future__ import annotations

import csv
import io
import logging

import pandas as pd

from .. import config
from ..http import cached, get_text

log = logging.getLogger(__name__)
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SERIES = {
    "us2y": ("DGS2", "美國 2 年期公債殖利率"),
    "us10y": ("DGS10", "美國 10 年期公債殖利率"),
    "curve_10_2": ("T10Y2Y", "殖利率曲線 10Y-2Y"),
    "curve_10_3m": ("T10Y3M", "殖利率曲線 10Y-3M"),
    "hy_spread": ("BAMLH0A0HYM2", "美國高收益債利差 (OAS)"),
    "dollar_broad": ("DTWEXBGS", "廣義美元指數"),
}


def series(sid: str) -> pd.DataFrame:
    """date, value (缺值列已剔除)"""
    def load():
        txt = get_text(URL.format(sid=sid), timeout=60)
        rows = []
        for r in csv.DictReader(io.StringIO(txt)):
            d, v = r.get("observation_date") or r.get("DATE"), r.get(sid)
            if d and v not in (None, "", "."):
                rows.append({"date": d, "value": float(v)})
        return rows
    return pd.DataFrame(cached(f"fred:{sid}", config.TTL_DAILY, load))


def all_series() -> dict[str, pd.DataFrame]:
    out = {}
    for key, (sid, _) in SERIES.items():
        try:
            df = series(sid)
            if not df.empty:
                out[key] = df
        except Exception as e:  # noqa: BLE001
            log.warning("FRED %s failed: %s", sid, e)
    return out
