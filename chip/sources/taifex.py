"""臺灣期貨交易所 OpenAPI (最新一日) 。歷史序列由 FinMind 補。"""
from __future__ import annotations

import csv
import io
import json
import logging

import pandas as pd

from .. import config
from ..http import cached, get_text, num

log = logging.getLogger(__name__)
BASE = "https://openapi.taifex.com.tw/v1"

# 期交所 OpenAPI 有時回 JSON、有時回 CSV (中文欄名)；CSV 時依欄位順序對應英文 key
FUT_INST_KEYS = ["Date", "ContractCode", "Item", "TradingVolume(Long)", "TradingValue(Long)(Thousands)",
                 "TradingVolume(Short)", "TradingValue(Short)(Thousands)", "TradingVolume(Net)",
                 "TradingValue(Net)(Thousands)", "OpenInterest(Long)", "ContractValueofOpenInterest(Long)(Thousands)",
                 "OpenInterest(Short)", "ContractValueofOpenInterest(Short)(Thousands)", "OpenInterest(Net)",
                 "ContractValueofOpenInterest(Net)(Thousands)"]
PCR_KEYS = ["Date", "PutVolume", "CallVolume", "PutCallVolumeRatio%", "PutOI", "CallOI", "PutCallOIRatio%"]
LARGE_KEYS = ["Date", "Contract", "ContractName", "SettlementMonth", "TypeOfTraders", "Top5Buy", "Top5Sell",
              "Top10Buy", "Top10Sell", "OIOfMarket"]


def _iso(s: str) -> str:
    s = s.strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _rows(endpoint: str, keys: list[str]) -> list[dict]:
    text = get_text(f"{BASE}/{endpoint}", headers={"Accept": "application/json"}, timeout=60).lstrip("﻿ \r\n")
    if text.startswith("["):
        return json.loads(text)
    rows = []
    for line in csv.reader(io.StringIO(text)):
        if not line or not line[0].strip() or not line[0].strip()[0].isdigit():
            continue
        rows.append(dict(zip(keys, [c.strip() for c in line])))
    return rows


def futures_institutional_latest() -> pd.DataFrame:
    """臺股期貨三大法人最新一日 (口數)。欄位同 finmind.tx_futures_institutional。"""
    def load():
        return _rows("MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate", FUT_INST_KEYS)
    rows = cached("taifex:fut_inst", config.TTL_INTRADAY, load)
    name = {"外資": "foreign", "外資及陸資": "foreign", "投信": "trust", "自營商": "dealer"}
    rec: dict = {}
    for r in rows:
        if r.get("ContractCode") != "臺股期貨":
            continue
        who = name.get(r.get("Item", "").strip())
        if not who:
            continue
        rec["date"] = _iso(r["Date"])
        lo, so = num(r["OpenInterest(Long)"]), num(r["OpenInterest(Short)"])
        rec[f"fut_{who}_long_oi"] = lo
        rec[f"fut_{who}_short_oi"] = so
        rec[f"fut_{who}_net_oi"] = lo - so
        rec[f"fut_{who}_net_trade"] = num(r["TradingVolume(Net)"])
    return pd.DataFrame([rec]) if rec else pd.DataFrame()


def put_call_ratio() -> pd.DataFrame:
    """臺指選擇權 Put/Call 比 (約 30 日)。pcr_vol / pcr_oi (%)"""
    def load():
        return _rows("PutCallRatio", PCR_KEYS)
    rows = cached("taifex:pcr", config.TTL_INTRADAY, load)
    df = pd.DataFrame([{
        "date": _iso(r["Date"]), "put_vol": num(r["PutVolume"]), "call_vol": num(r["CallVolume"]),
        "pcr_vol": num(r["PutCallVolumeRatio%"]), "put_oi": num(r["PutOI"]), "call_oi": num(r["CallOI"]),
        "pcr_oi": num(r["PutCallOIRatio%"]),
    } for r in rows])
    return df.sort_values("date").reset_index(drop=True)


def large_traders_tx() -> dict | None:
    """臺股期貨大額交易人未沖銷部位 (全部月份)。回傳前五/前十大、特定法人 淨部位。"""
    def load():
        return [r for r in _rows("OpenInterestOfLargeTradersFutures", LARGE_KEYS) if r.get("Contract") == "TX"]
    rows = cached("taifex:large", config.TTL_INTRADAY, load)
    out: dict = {}
    for r in rows:
        if r.get("Contract") != "TX" or r.get("SettlementMonth") != "999912":
            continue
        out["date"] = _iso(r["Date"])
        tag = "spec" if r.get("TypeOfTraders") == "1" else "all"   # 1=特定法人, 0=全體
        out[f"top5_{tag}_net"] = num(r["Top5Buy"]) - num(r["Top5Sell"])
        out[f"top10_{tag}_net"] = num(r["Top10Buy"]) - num(r["Top10Sell"])
        out[f"top10_{tag}_buy"] = num(r["Top10Buy"])
        out[f"top10_{tag}_sell"] = num(r["Top10Sell"])
        out["oi_market"] = num(r["OIOfMarket"])
    return out or None
