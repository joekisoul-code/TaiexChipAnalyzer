"""嗨投資 HiStock：八大官股行庫 (公股銀行) 買賣超。

- broker8.aspx        全市場每日買賣總金額走勢 (億元) + 當日買超/賣超排行
- broker.aspx?no=XXXX 個股八大行庫每日買賣金額 (萬元) / 張數
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re

import pandas as pd

from .. import config
from ..http import cached, get_text, num

log = logging.getLogger(__name__)
BASE = "https://histock.tw/stock"
BANKS = ["合庫", "土銀", "台銀", "台企銀", "彰銀", "第一金", "兆豐銀", "華南永昌"]
BANK_KEYS = ["tcb", "land", "bot", "tbb", "chb", "first", "mega", "hncb"]


def _ts_to_date(ms: float) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).date().isoformat()


def _extract_chart_json(html: str, func: str) -> dict:
    m = re.search(func + r"\((\{.*?\})\);", html, re.S)
    if not m:
        raise ValueError(f"找不到 {func}() 圖表資料")
    return json.loads(m.group(1))


def _series(s: str) -> list[list[float]]:
    return json.loads(s) if s else []


def government_banks_history() -> pd.DataFrame:
    """八大行庫每日買賣總金額 (億元) 與加權指數。欄位: date, gov8_net, close"""
    def load():
        html = get_text(f"{BASE}/broker8.aspx")
        d = _extract_chart_json(html, "loadChart")
        money = {_ts_to_date(t): v for t, v in _series(d.get("SumMoney", ""))}
        price = {_ts_to_date(t): v for t, v in _series(d.get("Price", ""))}
        return [{"date": k, "gov8_net": v, "close": price.get(k)} for k, v in sorted(money.items())]
    return pd.DataFrame(cached("histock:broker8:history", config.TTL_INTRADAY, load))


def government_banks_ranking() -> dict[str, pd.DataFrame]:
    """當日八大行庫買超 / 賣超排行 (萬元)。回傳 {'buy': df, 'sell': df, 'date': 'YYYY-MM-DD'}"""
    def load():
        html = get_text(f"{BASE}/broker8.aspx")
        lists = re.findall(r'<ul class="stock-list">(.*?)</ul>', html, re.S)
        date = None
        m = re.search(r'title="(\d{4}-\d{2}-\d{2}) 八大公股行庫', html)
        if m:
            date = m.group(1)
        out = {"date": date, "buy": [], "sell": []}
        for idx, block in enumerate(lists[:2]):
            key = "buy" if idx == 0 else "sell"
            for li in re.findall(r"<li.*?</li>", block, re.S):
                code = re.search(r'goUrl\("(\w+)"\)', li)
                name = re.search(r'class="w100 name">&nbsp;([^<]+)<', li)
                vals = [num(v.replace("&nbsp;", "")) for v in re.findall(r'<span class="w70">([^<]*)</span>', li)]
                if not code or len(vals) < 9:
                    continue
                rec = {"code": code.group(1), "name": name.group(1).strip() if name else ""}
                rec.update(dict(zip(BANKS, vals[:8])))
                rec["total"] = vals[8]
                out[key].append(rec)
        return out
    d = cached("histock:broker8:rank", config.TTL_INTRADAY, load)
    return {"date": d["date"], "buy": pd.DataFrame(d["buy"]), "sell": pd.DataFrame(d["sell"])}


def government_bank_stock(stock_id: str) -> pd.DataFrame:
    """個股八大行庫每日買賣：date, gov8_net (萬元), gov8_lots (張), 各行庫 money/lots。"""
    def load():
        html = get_text(f"{BASE}/broker.aspx", params={"no": stock_id})
        d = _extract_chart_json(html, "initalChart_Broker8")
        table: dict[str, dict] = {}
        for key, raw in d.items():
            m = re.match(r"^([A-Za-z]+)(\d+)$", key)
            if not m or not isinstance(raw, str) or not raw.startswith("["):
                continue
            kind, n = m.group(1).lower(), int(m.group(2))
            if n < 1 or n > 8:
                continue
            col = f"{'money' if kind.startswith('money') else 'lots'}_{BANKS[n - 1]}"
            for t, v in _series(raw):
                table.setdefault(_ts_to_date(t), {})[col] = v
        rows = []
        for date in sorted(table):
            rec = {"date": date, **table[date]}
            rec["gov8_net"] = sum(v for k, v in rec.items() if k.startswith("money_") and v is not None)
            rec["gov8_lots"] = sum(v for k, v in rec.items() if k.startswith("lots_") and v is not None)
            rows.append(rec)
        return rows
    return pd.DataFrame(cached(f"histock:broker:{stock_id}", config.TTL_INTRADAY, load))
