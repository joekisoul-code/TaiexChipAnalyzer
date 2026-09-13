"""特徵工程：把 market.add_features/score_frame 的欄位整理成模型輸入，加上時間特徵。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..analysis.common import zscore

# 有 2010~ 長歷史、可訓練的特徵 (八大行庫/PCR/維持率/借券/大額交易人 只有數月歷史，不進模型)
MARKET_FEATURES = [
    # 外資/投信/自營 現貨
    "foreign_z1", "foreign_z5", "foreign_z20", "trust_z5", "trust_z20", "dealer_z1", "foreign_streak", "trust_streak",
    # 外資期貨
    "fut_foreign_pct", "fut_foreign_chg1_z", "fut_foreign_chg5_z", "foreign_consistency",
    # 融資融券
    "margin_pct20", "margin_div20", "margin_chg5_pct", "short_chg5_z", "short_ratio",
    # 價量
    "ret1", "ret5", "ret20", "ret60", "bias5", "bias20", "bias60", "ma20_slope", "vol_ratio", "amount_5d_ratio",
    "hi20_dist", "lo20_dist", "vola20", "vola_ratio",
    # 狀態與因子分
    "state_bull", "state_bear", "f_foreign", "f_trust", "f_dealer", "f_fut_foreign", "f_margin", "f_short",
    "f_volume", "f_trend", "f_reversion", "composite", "composite_smooth", "composite_chg5",
    # 時間
    "dow", "month", "days_to_settle", "settle_week", "days_to_month_end", "days_to_quarter_end",
    # 國際盤 (台股當日開盤前已知)
    "g_vix_level", "g_vix_chg", "g_sox_r1", "g_sox_r5", "g_sox_r20", "g_sox_hi20", "g_sp500_r1", "g_sp500_hi20", "g_nasdaq_r1",
    "g_tsm_adr_r1", "g_btc_r20", "g_hsi_r5", "g_us10y_r5", "g_kospi_r1", "g_kospi_r5", "g_usdtwd_r5", "g_copper_r1", "g_gold_r5",
    "g_oil_r20", "g_dxy_r1", "f_global",
    "g_copper_gold_r20", "g_natgas_r20", "g_usdjpy_r5", "g_curve_10y_3m", "g_vix_term", "g_bdry_r20", "g_us5y_r5",
    "g_usdtwd_r20", "g_usdtwd_r60", "g_usdtwd_streak", "g_oil_r60", "f_fx_flow",
]
FEATURE_NAMES = {
    "foreign_z1": "外資當日買賣超 z", "foreign_z5": "外資 5 日累計 z", "foreign_z20": "外資 20 日累計 z",
    "trust_z5": "投信 5 日 z", "trust_z20": "投信 20 日 z", "dealer_z1": "自營商當日 z",
    "foreign_streak": "外資連買/賣天數", "trust_streak": "投信連買/賣天數",
    "fut_foreign_pct": "外資期貨淨部位一年百分位", "fut_foreign_chg1_z": "外資期貨日變化 z", "fut_foreign_chg5_z": "外資期貨 5 日變化 z",
    "foreign_consistency": "外資現貨期貨一致性", "margin_pct20": "融資 20 日變化%", "margin_div20": "融資-指數背離",
    "margin_chg5_pct": "融資 5 日變化%", "short_chg5_z": "融券 5 日變化 z", "short_ratio": "券資比",
    "ret1": "當日漲跌%", "ret5": "5 日漲跌%", "ret20": "20 日漲跌%", "ret60": "60 日漲跌%",
    "bias5": "5 日線乖離", "bias20": "月線乖離", "bias60": "季線乖離", "ma20_slope": "月線斜率",
    "vol_ratio": "量能/20 日均", "amount_5d_ratio": "5 日量能/20 日均", "hi20_dist": "距 20 日高點%", "lo20_dist": "距 20 日低點%",
    "vola20": "20 日波動率", "vola_ratio": "波動率/60 日", "state_bull": "多頭狀態", "state_bear": "空頭狀態",
    "f_foreign": "因子:外資現貨", "f_trust": "因子:投信", "f_dealer": "因子:自營商", "f_fut_foreign": "因子:外資期貨",
    "f_margin": "因子:融資象限", "f_short": "因子:融券", "f_volume": "因子:量價", "f_trend": "因子:趨勢", "f_reversion": "因子:超跌回歸",
    "composite": "綜合籌碼分", "composite_smooth": "綜合分(平滑)", "composite_chg5": "綜合分 5 日動能",
    "dow": "星期", "month": "月份", "days_to_settle": "距期貨結算日", "settle_week": "結算週",
    "days_to_month_end": "距月底交易日", "days_to_quarter_end": "距季底交易日", "f_global": "因子:國際盤",
    "g_vix_level": "VIX 水準", "g_vix_chg": "VIX 日變化%", "g_sox_r1": "費半前晚%", "g_sox_r5": "費半 5 日%", "g_sox_r20": "費半 20 日%",
    "g_sox_hi20": "費半距 20 日高", "g_sp500_r1": "S&P500 前晚%", "g_sp500_hi20": "S&P500 距 20 日高", "g_nasdaq_r1": "Nasdaq 前晚%",
    "g_tsm_adr_r1": "台積電 ADR 前晚%", "g_btc_r20": "比特幣 20 日%", "g_hsi_r5": "恆生 5 日%", "g_us10y_r5": "美債殖利率 5 日%",
    "g_kospi_r1": "KOSPI 前日%", "g_kospi_r5": "KOSPI 5 日%", "g_usdtwd_r5": "美元/台幣 5 日%", "g_copper_r1": "銅前晚%",
    "g_gold_r5": "黃金 5 日%", "g_oil_r20": "原油 20 日%", "g_dxy_r1": "美元指數前晚%",
    "g_copper_gold_r20": "銅金比 20 日%", "g_natgas_r20": "天然氣 20 日%", "g_usdjpy_r5": "美元/日圓 5 日%", "g_curve_10y_3m": "殖利率曲線 10Y-3M",
    "g_vix_term": "VIX 期限結構", "g_bdry_r20": "乾散貨 ETF 20 日%", "g_us5y_r5": "美債 5Y 5 日%",
    "g_usdtwd_r20": "美元/台幣 20 日%", "g_usdtwd_r60": "美元/台幣 60 日%", "g_usdtwd_streak": "美元/台幣連漲天數", "g_oil_r60": "油價 60 日%", "f_fx_flow": "因子:匯率資金流",
}


def third_wednesday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    offset = (2 - d.weekday()) % 7          # Wednesday = 2
    return d + dt.timedelta(days=offset + 14)


def _days_to_settle(dates: pd.Series) -> pd.Series:
    out = []
    for s in dates:
        d = dt.date.fromisoformat(str(s)[:10])
        tw = third_wednesday(d.year, d.month)
        if tw < d:
            y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
            tw = third_wednesday(y, m)
        out.append((tw - d).days)
    return pd.Series(out, index=dates.index)


def busdays_to_period_end(dates: pd.Series, freq: str) -> pd.Series:
    """該日之後到 月底/季底 (含) 還有幾個「週一~週五」；freq='M' 或 'Q'。

    純日曆定義 (np.busday_count，不扣 TWSE 休市日)：訓練與即時推論算出來完全相同，
    不像 groupby(period).cumcount(ascending=False) 會讓「任何 frame 的最後一列」永遠是 0 (訓練/推論偏差)。
    不扣休市日的原因：TWSE 休市日快取只有近兩年，2010~ 訓練列無法一致扣除；週末以外的假日 (春節等) 只讓數值略偏大，
    且訓練/推論一致。當月最後一個週間日 = 0，與舊定義在「完整月份」的語意相同。
    """
    dts = pd.to_datetime(dates)
    end = dts.dt.to_period(freq).dt.end_time.dt.normalize()
    a = dts.values.astype("datetime64[D]") + np.timedelta64(1, "D")      # 從隔天起算 (不含當日)
    b = end.values.astype("datetime64[D]") + np.timedelta64(1, "D")      # busday_count 的 end 不含 → 含月底當天
    return pd.Series(np.busday_count(a, b), index=dates.index).astype(int)


def add_time_features(d: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(d["date"])
    d["dow"] = dates.dt.weekday
    d["month"] = dates.dt.month
    d["days_to_settle"] = _days_to_settle(d["date"])
    d["settle_week"] = (d["days_to_settle"] <= 4).astype(int)
    # 距月底/季底的週間日數 (日曆定義，訓練與推論一致；見 busdays_to_period_end)
    d["days_to_month_end"] = busdays_to_period_end(d["date"], "M")
    d["days_to_quarter_end"] = busdays_to_period_end(d["date"], "Q")
    return d


def market_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    """從 score_frame 輸出建立特徵矩陣 (保留 date/close 與 fwd 欄位)。"""
    d = scored.copy()
    c = d["close"].astype(float)
    d["foreign_z1"] = zscore(d["foreign"], 60)
    d["foreign_z5"] = zscore(d["foreign_5d"], 60)
    d["foreign_z20"] = zscore(d["foreign_20d"], 120)
    d["trust_z5"] = zscore(d["trust_5d"], 60)
    d["trust_z20"] = zscore(d["trust_20d"], 120)
    d["dealer_z1"] = zscore(d["dealer"], 60)
    d["fut_foreign_chg1_z"] = zscore(d["fut_foreign_chg1"], 60)
    d["fut_foreign_chg5_z"] = zscore(d["fut_foreign_chg5"], 60)
    d["margin_chg5_pct"] = d["margin_amt"].pct_change(5) * 100
    d["short_chg5_z"] = zscore(d["short_chg5"], 60)
    d["ret60"] = c.pct_change(60) * 100
    d["bias5"] = (c / d["ma5"] - 1) * 100
    d["bias60"] = (c / d["ma60"] - 1) * 100
    d["hi20_dist"] = (c / d["hi20"] - 1) * 100
    d["lo20_dist"] = (c / d["lo20"] - 1) * 100
    d["vola20"] = d["ret1"].rolling(20).std()
    d["vola_ratio"] = d["vola20"] / d["ret1"].rolling(60).std()
    d["state_bull"] = (d["state"] == "多頭").astype(int)
    d["state_bear"] = (d["state"] == "空頭").astype(int)
    d = add_time_features(d)
    for h in (1, 2, 3, 5, 10, 20):
        if f"fwd{h}" not in d:
            d[f"fwd{h}"] = (c.shift(-h) / c - 1) * 100
    for col in MARKET_FEATURES:
        if col not in d:
            d[col] = np.nan
    return d


def as_if_close(scored: pd.DataFrame, price: float, projected_amount: float | None) -> pd.DataFrame:
    """盤中：假設今日以 price 收盤，複製最後一列並重算價量類特徵 (籌碼類沿用前一日)。"""
    d = scored.copy()
    last = d.iloc[-1].copy()
    last_date = str(last["date"])
    today = dt.date.today().isoformat()
    if last_date == today:
        return d  # 已收盤且資料同日
    new = last.copy()
    new["date"] = today
    try:   # 今日開盤前已知的國際盤
        from ..sources import global_markets as gm
        for k, v in gm.latest_features().items():
            new[k] = v
    except Exception:  # noqa: BLE001
        pass
    new["close"] = price
    new["open"] = new["high"] = new["low"] = price
    new["amount"] = projected_amount if projected_amount else last["amount"]
    for col in ("foreign", "trust", "dealer", "total", "gov8_net", "fut_foreign_net_trade"):
        if col in new:
            new[col] = np.nan     # 今日籌碼未知
    # to_frame().T 會讓所有欄位變 object dtype → add_features 的 amount/amount_ma20 除法在 pandas 3 會 ZeroDivisionError；還原數值 dtype
    d = pd.concat([d, new.to_frame().T], ignore_index=True).infer_objects()
    return d
