"""國際市場日資料 (Yahoo Finance chart API)：美股、費半、VIX、日韓股、原油、黃金、比特幣、美元/台幣、美債殖利率、台積電 ADR、台灣 ETF。

台股是亞洲早盤，前一晚美股 (台北時間 04:00/05:00 收) 與當日早上的日韓開盤都在台股開盤前/同時發生：
- 美股/費半/VIX/TSM ADR/原油/黃金/美債：以「前一個美國交易日」的收盤變化作為台股當日特徵 (lag 0 = 台北日期 D 的清晨收盤)。
- 日經/韓股：與台股同日交易，收盤晚於台股 → 只能用前一日 (lag 1)，或盤中即時用當日開盤。
- 比特幣：24 小時交易，取台北 08:00 前的日 K。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from ..http import cached, get_json

log = logging.getLogger(__name__)
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

SYMBOLS = {
    "sp500": ("^GSPC", "S&P 500", "us"),
    "nasdaq": ("^IXIC", "Nasdaq", "us"),
    "sox": ("^SOX", "費城半導體", "us"),
    "vix": ("^VIX", "VIX 恐慌指數", "us"),
    "tsm_adr": ("TSM", "台積電 ADR", "us"),
    "ewt": ("EWT", "iShares 台灣 ETF", "us"),
    "us10y": ("^TNX", "美國 10 年期公債殖利率", "us"),
    "dxy": ("DX-Y.NYB", "美元指數", "us"),
    "oil": ("CL=F", "WTI 原油", "us"),
    "gold": ("GC=F", "黃金", "us"),
    "copper": ("HG=F", "銅", "us"),
    "btc": ("BTC-USD", "比特幣", "crypto"),
    "usdtwd": ("TWD=X", "美元/台幣", "fx"),
    "usdjpy": ("JPY=X", "美元/日圓", "fx"),
    "usdkrw": ("KRW=X", "美元/韓元", "fx"),
    "eurusd": ("EURUSD=X", "歐元/美元", "fx"),
    "nikkei": ("^N225", "日經 225", "asia"),
    "kospi": ("^KS11", "韓國 KOSPI", "asia"),
    "hsi": ("^HSI", "香港恆生", "asia"),
    "sse": ("000001.SS", "上證", "asia"),
    # 利率 / 波動率 / 原物料 (擴充)
    "us3m": ("^IRX", "美國 3 個月國庫券", "us"),
    "us5y": ("^FVX", "美國 5 年期公債殖利率", "us"),
    "us30y": ("^TYX", "美國 30 年期公債殖利率", "us"),
    "vxn": ("^VXN", "Nasdaq 100 VIX", "us"),
    "vix3m": ("^VIX3M", "VIX 3 個月", "us"),
    "skew": ("^SKEW", "黑天鵝指數 SKEW", "us"),
    "brent": ("BZ=F", "布蘭特原油", "us"),
    "silver": ("SI=F", "白銀", "us"),
    "natgas": ("NG=F", "天然氣", "us"),
    "soybean": ("ZS=F", "黃豆", "us"),
    "corn": ("ZC=F", "玉米", "us"),
    "platinum": ("PL=F", "白金", "us"),
    "bdry": ("BDRY", "乾散貨運價 ETF (BDI 代理)", "us"),
    "twii": ("^TWII", "加權指數 (Yahoo)", "asia"),
}


def history(symbol: str, range_: str = "20y") -> pd.DataFrame:
    """日 K：date(交易所當地日期), open, high, low, close。"""
    def load():
        j = get_json(URL.format(sym=symbol), params={"interval": "1d", "range": range_, "events": "div,splits"}, timeout=60)
        res = j["chart"]["result"][0]
        ts, q = res["timestamp"], res["indicators"]["quote"][0]
        off = res["meta"].get("gmtoffset", 0)
        rows = []
        for i, t in enumerate(ts):
            if q["close"][i] is None:
                continue
            d = dt.datetime.fromtimestamp(t, dt.UTC) + dt.timedelta(seconds=off)
            rows.append({"date": d.strftime("%Y-%m-%d"), "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i]})
        return rows
    return pd.DataFrame(cached(f"yahoo:{symbol}:1d:{range_}", config.TTL_DAILY, load))


def all_markets(range_: str = "20y") -> dict[str, pd.DataFrame]:
    out = {}
    for key, (sym, _, _) in SYMBOLS.items():
        try:
            df = history(sym, range_)
            if not df.empty:
                out[key] = df
        except Exception as e:  # noqa: BLE001
            log.warning("%s (%s) failed: %s", key, sym, e)
    return out


def aligned_features(taiex_dates: pd.Series, markets: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """對齊到台股交易日 D 的國際特徵 (皆為台股 D 日開盤前已知)：
    us/crypto/fx/commodity：取「日期 < D」的最後一筆 → 其 1 日 / 5 日報酬與距 20 日高點 (VIX 為水準與變化)。
    asia：取「日期 < D」的最後一筆 (即前一交易日) 的 1 日報酬。
    欄位：g_{key}_r1, g_{key}_r5, g_vix_level, g_vix_chg, g_tsm_adr_r1 ...
    """
    markets = markets or all_markets()
    dates = pd.to_datetime(pd.Series(taiex_dates).astype(str))
    out: dict = {"date": dates.dt.strftime("%Y-%m-%d").values}
    levels: dict[str, np.ndarray] = {}
    for key, df in markets.items():
        d = df.copy()
        d["dt"] = pd.to_datetime(d["date"])
        d = d.sort_values("dt")
        d["r1"] = d["close"].pct_change() * 100
        d["r5"] = d["close"].pct_change(5) * 100
        d["r20"] = d["close"].pct_change(20) * 100
        d["r60"] = d["close"].pct_change(60) * 100
        d["hi20_dist"] = (d["close"] / d["close"].rolling(20).max() - 1) * 100
        up = (d["close"].diff() > 0).astype(int)
        d["up_streak"] = up.groupby((up == 0).cumsum()).cumsum()
        # 對每個台股日期，找最後一個 < D 的資料列 (merge_asof, allow_exact_matches=False)
        m = pd.merge_asof(pd.DataFrame({"dt": dates}).sort_values("dt"), d[["dt", "close", "r1", "r5", "r20", "r60", "hi20_dist", "up_streak"]],
                          on="dt", direction="backward", allow_exact_matches=False)
        m = m.set_index("dt").reindex(dates)
        levels[key] = m["close"].values
        out[f"g_{key}_r1"] = m["r1"].values
        out[f"g_{key}_r5"] = m["r5"].values
        if key in ("oil", "brent", "usdtwd", "usdjpy", "copper", "gold"):
            out[f"g_{key}_r60"] = m["r60"].values
        if key == "usdtwd":
            out["g_usdtwd_r20"] = m["r20"].values
            out["g_usdtwd_streak"] = m["up_streak"].values
        if key == "vix":
            out["g_vix_level"] = m["close"].values
            out["g_vix_chg"] = m["r1"].values
        if key in ("sox", "sp500", "tsm_adr", "btc", "gold", "oil", "copper", "natgas", "usdjpy", "bdry", "silver"):
            out[f"g_{key}_r20"] = m["r20"].values
            out[f"g_{key}_hi20"] = m["hi20_dist"].values
    # 跨市場研究 (2007~) 驗證過的派生特徵
    if "copper" in levels and "gold" in levels:
        cg = pd.Series(levels["copper"] / levels["gold"])
        out["g_copper_gold_r20"] = ((cg / cg.shift(20) - 1) * 100).values
    if "us10y" in levels and "us3m" in levels:
        out["g_curve_10y_3m"] = levels["us10y"] - levels["us3m"]
    if "vix" in levels and "vix3m" in levels:
        out["g_vix_term"] = levels["vix"] / levels["vix3m"]
    return pd.DataFrame(out)


def latest_features(markets: dict[str, pd.DataFrame] | None = None) -> dict:
    """「現在」可用的最新國際特徵 (供下一個台股交易日 / 盤中使用)：以明天日期對齊取最後收盤。"""
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    f = aligned_features(pd.Series([tomorrow]), markets)
    return {k: (float(v) if pd.notna(v) else np.nan) for k, v in f.iloc[0].items() if k != "date"}


def quotes(keys: list[str] | None = None) -> list[dict]:
    """各市場最新報價 (Yahoo 1 日 5 分 K 的最後一筆，含盤中)；回傳 [{key,name,last,prev_close,chg_pct,time}]。"""
    out = []
    for key, (sym, name, region) in SYMBOLS.items():
        if keys and key not in keys:
            continue
        try:
            j = cached(f"yahoo:{sym}:5m:1d", 60, lambda s=sym: get_json(URL.format(sym=s), params={"interval": "5m", "range": "1d"}, timeout=20))
            res = j["chart"]["result"][0]
            meta = res["meta"]
            last = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            t = meta.get("regularMarketTime")
            off = meta.get("gmtoffset", 0)
            ts = (dt.datetime.fromtimestamp(t, dt.UTC) + dt.timedelta(hours=8)).strftime("%m-%d %H:%M") if t else ""
            out.append({"key": key, "name": name, "region": region, "last": last, "prev_close": prev,
                        "chg_pct": (last / prev - 1) * 100 if last and prev else None, "time_tw": ts})
        except Exception as e:  # noqa: BLE001
            log.debug("quote %s failed: %s", sym, e)
    return out


GLOBAL_FEATURES = ["g_vix_level", "g_vix_chg", "g_sox_r1", "g_sox_r5", "g_sox_r20", "g_sox_hi20", "g_sp500_r1", "g_sp500_hi20",
                   "g_nasdaq_r1", "g_tsm_adr_r1", "g_btc_r20", "g_hsi_r5", "g_us10y_r5", "g_kospi_r1", "g_kospi_r5",
                   "g_usdtwd_r5", "g_copper_r1", "g_gold_r5", "g_oil_r20", "g_dxy_r1",
                   "g_copper_gold_r20", "g_natgas_r20", "g_usdjpy_r5", "g_curve_10y_3m", "g_vix_term", "g_bdry_r20", "g_us5y_r5",
                   "g_usdtwd_r20", "g_usdtwd_r60", "g_usdtwd_streak", "g_oil_r60", "g_brent_r60"]
GLOBAL_NAMES = {"g_vix_level": "VIX 水準", "g_vix_chg": "VIX 日變化%", "g_sox_r1": "費半前晚%", "g_sox_r5": "費半 5 日%", "g_sox_r20": "費半 20 日%",
                "g_sox_hi20": "費半距 20 日高", "g_sp500_r1": "S&P500 前晚%", "g_sp500_hi20": "S&P500 距 20 日高", "g_nasdaq_r1": "Nasdaq 前晚%",
                "g_tsm_adr_r1": "台積電 ADR 前晚%", "g_btc_r20": "比特幣 20 日%", "g_hsi_r5": "恆生 5 日%", "g_us10y_r5": "美債殖利率 5 日%",
                "g_kospi_r1": "KOSPI 前日%", "g_kospi_r5": "KOSPI 5 日%", "g_usdtwd_r5": "美元/台幣 5 日%", "g_copper_r1": "銅前晚%",
                "g_gold_r5": "黃金 5 日%", "g_oil_r20": "原油 20 日%", "g_dxy_r1": "美元指數前晚%"}
