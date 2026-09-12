"""FinMind 開放資料 (歷史序列)。免費層不需 token；贊助層才有八大行庫個股 / 券商分點。"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .. import config
from ..http import cached, session

log = logging.getLogger(__name__)
YI = 1e8


class FinMindError(RuntimeError):
    pass


class FinMindPermissionError(FinMindError):
    """資料集需要贊助等級或 token。"""


def _default_start(days: int = config.HISTORY_DAYS) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def fetch(dataset: str, data_id: str | None = None, start_date: str | None = None,
          end_date: str | None = None, ttl: int = config.TTL_HISTORY, **extra) -> pd.DataFrame:
    start_date = start_date or _default_start()
    params = {"dataset": dataset, "start_date": start_date}
    if data_id:
        params["data_id"] = data_id
    if end_date:
        params["end_date"] = end_date
    params.update(extra)
    key = "finmind:" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) + f"&d={dt.date.today()}"

    def load():
        headers = {}
        if config.FINMIND_TOKEN:
            headers["Authorization"] = f"Bearer {config.FINMIND_TOKEN}"
        r = session().get(config.FINMIND_URL, params=params, headers=headers, timeout=60)
        if r.status_code in (400, 401, 402, 403):
            raise FinMindPermissionError(f"{dataset}: HTTP {r.status_code} {r.text[:120]}")
        r.raise_for_status()
        j = r.json()
        if j.get("status") != 200:
            msg = j.get("msg", "")
            if "permission" in msg.lower() or "sponsor" in msg.lower() or "token" in msg.lower():
                raise FinMindPermissionError(f"{dataset}: {msg}")
            raise FinMindError(f"{dataset}: {msg}")
        return j.get("data", [])

    return pd.DataFrame(cached(key, ttl, load))


# ---------------------------------------------------------------- 大盤序列
def taiex_price(start: str | None = None) -> pd.DataFrame:
    df = fetch("TaiwanStockPrice", "TAIEX", start)
    if df.empty:
        return df
    df = df.rename(columns={"max": "high", "min": "low", "Trading_money": "amount_raw",
                            "Trading_Volume": "volume", "spread": "change"})
    df["amount"] = df["amount_raw"] / YI
    return df[["date", "open", "high", "low", "close", "change", "volume", "amount"]]


def total_institutional(start: str | None = None) -> pd.DataFrame:
    """全市場三大法人淨買賣 (億)：foreign / trust / dealer_self / dealer_hedge / dealer / total"""
    df = fetch("TaiwanStockTotalInstitutionalInvestors", None, start)
    if df.empty:
        return df
    df["net"] = (df["buy"] - df["sell"]) / YI
    p = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").reset_index()
    p = p.rename(columns={"Foreign_Investor": "foreign", "Investment_Trust": "trust",
                          "Dealer_self": "dealer_self", "Dealer_Hedging": "dealer_hedge", "total": "total"})
    for c in ("foreign", "trust", "dealer_self", "dealer_hedge", "total"):
        if c not in p:
            p[c] = 0.0
    p["dealer"] = p["dealer_self"] + p["dealer_hedge"]
    return p[["date", "foreign", "trust", "dealer_self", "dealer_hedge", "dealer", "total"]]


def total_margin(start: str | None = None) -> pd.DataFrame:
    """全市場融資融券：margin_lots, short_lots (張), margin_amt (億)"""
    df = fetch("TaiwanStockTotalMarginPurchaseShortSale", None, start)
    if df.empty:
        return df
    p = df.pivot_table(index="date", columns="name", values="TodayBalance", aggfunc="last").reset_index()
    p = p.rename(columns={"MarginPurchase": "margin_lots", "ShortSale": "short_lots",
                          "MarginPurchaseMoney": "margin_amt"})
    p["margin_amt"] = p["margin_amt"] / YI
    return p[["date", "margin_lots", "short_lots", "margin_amt"]]


def tx_futures_institutional(start: str | None = None) -> pd.DataFrame:
    """臺股期貨三大法人：fut_{foreign,trust,dealer}_{net_oi,long_oi,short_oi,net_trade}"""
    df = fetch("TaiwanFuturesInstitutionalInvestors", "TX", start)
    if df.empty:
        return df
    name = {"外資": "foreign", "投信": "trust", "自營商": "dealer", "外資及陸資": "foreign"}
    df["who"] = df["institutional_investors"].map(name)
    df = df.dropna(subset=["who"])
    out = pd.DataFrame({"date": sorted(df["date"].unique())}).set_index("date")
    for who, g in df.groupby("who"):
        g = g.set_index("date")
        out[f"fut_{who}_long_oi"] = g["long_open_interest_balance_volume"]
        out[f"fut_{who}_short_oi"] = g["short_open_interest_balance_volume"]
        out[f"fut_{who}_net_oi"] = g["long_open_interest_balance_volume"] - g["short_open_interest_balance_volume"]
        out[f"fut_{who}_net_trade"] = g["long_deal_volume"] - g["short_deal_volume"]
    return out.reset_index()


def tx_night_history(start: str | None = None) -> pd.DataFrame:
    """臺指期夜盤 (after_market) 近月每日收盤與漲跌%：date, night_close, night_chg_pct, night_volume。
    夜盤 date 為該夜盤所屬的交易日 (15:00 開始)，對「隔日」開盤是前一晚資訊。"""
    df = fetch("TaiwanFuturesDaily", "TX", start)
    if df.empty:
        return df
    n = df[df["trading_session"] == "after_market"].copy()
    if n.empty:
        return n
    n = n.sort_values(["date", "contract_date"]).groupby("date", as_index=False).first()   # 近月
    return n.rename(columns={"close": "night_close", "spread_per": "night_chg_pct", "volume": "night_volume"})[
        ["date", "night_close", "night_chg_pct", "night_volume"]]


# ---------------------------------------------------------------- 個股序列
def stock_price(stock_id: str, start: str | None = None) -> pd.DataFrame:
    df = fetch("TaiwanStockPrice", stock_id, start)
    if df.empty:
        return df
    df = df.rename(columns={"max": "high", "min": "low", "Trading_money": "amount",
                            "Trading_Volume": "volume", "spread": "change"})
    df["volume_lots"] = df["volume"] / 1000
    return df[["date", "open", "high", "low", "close", "change", "volume", "volume_lots", "amount"]]


def stock_institutional(stock_id: str, start: str | None = None) -> pd.DataFrame:
    """個股法人買賣超 (張)"""
    df = fetch("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start)
    if df.empty:
        return df
    df["net"] = (df["buy"] - df["sell"]) / 1000
    p = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").reset_index()
    p = p.rename(columns={"Foreign_Investor": "foreign", "Investment_Trust": "trust",
                          "Dealer_self": "dealer_self", "Dealer_Hedging": "dealer_hedge",
                          "Foreign_Dealer_Self": "foreign_dealer"})
    for c in ("foreign", "trust", "dealer_self", "dealer_hedge"):
        if c not in p:
            p[c] = 0.0
    p["dealer"] = p["dealer_self"] + p["dealer_hedge"]
    p["total"] = p["foreign"] + p["trust"] + p["dealer"]
    return p[["date", "foreign", "trust", "dealer", "total"]]


def stock_margin(stock_id: str, start: str | None = None) -> pd.DataFrame:
    df = fetch("TaiwanStockMarginPurchaseShortSale", stock_id, start)
    if df.empty:
        return df
    return df.rename(columns={"MarginPurchaseTodayBalance": "margin_lots",
                              "ShortSaleTodayBalance": "short_lots",
                              "MarginPurchaseBuy": "margin_buy", "MarginPurchaseSell": "margin_sell",
                              "ShortSaleBuy": "short_buy", "ShortSaleSell": "short_sell"})[
        ["date", "margin_lots", "short_lots", "margin_buy", "margin_sell", "short_buy", "short_sell"]]


def stock_shareholding(stock_id: str, start: str | None = None) -> pd.DataFrame:
    df = fetch("TaiwanStockShareholding", stock_id, start)
    if df.empty:
        return df
    return df.rename(columns={"ForeignInvestmentSharesRatio": "foreign_ratio"})[["date", "foreign_ratio"]]


# ---------------------------------------------------------------- 贊助等級
def government_bank_stock(stock_id: str, start: str | None = None) -> pd.DataFrame:
    """八大行庫個股買賣 (贊助等級)。"""
    return fetch("TaiwanStockGovernmentBankBuySell", stock_id, start)


def broker_daily(stock_id: str, date: str) -> pd.DataFrame:
    """券商分點 (贊助等級)：TaiwanStockTradingDailyReport"""
    return fetch("TaiwanStockTradingDailyReport", stock_id, date, date)
