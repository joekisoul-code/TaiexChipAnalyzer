"""晚間美期叫牌 (2026-09-24)：台股收盤後，美股 S&P 期貨 (ES=F) 相對當天台北 13:00 的漲跌 → 隔一交易日台股收盤方向。

研究 (tools/es_evening_research.py，Yahoo ES=F 60m 2024-05~2026-09，台股 575 個交易日，隔天上漲基準 57%)：
基準 = 當天 13:00 (台北) 前最後一根小時 K 收盤；比較時點的 ES 價格 = 該整點前 3 小時內最後一根。
- 21:00 前樣本少且不穩 (18:00 ≥0.5%：2025 年只有 46%) → 不叫。
- 21:00 起 |Δ| ≥ 0.5%：75~82%，逐年最低 ≥69%；22:00 起 |Δ| ≥ 0.3%：71~79%。
- 越晚越準 (美股開盤後資訊越完整)；04:00 之後沿用 04:00 的統計。
只在「最後一個台股交易日 21:00 ~ 下一交易日 08:45」使用；早上 05:00 後雲端含夜盤預測 (約 84%) 也會出來，兩者可互相印證。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SYM = "ES=F"
# 整點 → {門檻: {hit, n, yr_min}}；門檻 0.5 自 21:00、0.3 自 22:00
TABLE = {
    "21": {"0.5": {"hit": 0.750, "n": 112, "yr_min": 0.69}},
    "22": {"0.5": {"hit": 0.763, "n": 173, "yr_min": 0.70}, "0.3": {"hit": 0.707, "n": 287, "yr_min": 0.67}},
    "23": {"0.5": {"hit": 0.797, "n": 212, "yr_min": 0.77}, "0.3": {"hit": 0.735, "n": 325, "yr_min": 0.70}},
    "00": {"0.5": {"hit": 0.820, "n": 222, "yr_min": 0.79}, "0.3": {"hit": 0.773, "n": 330, "yr_min": 0.76}},
    "01": {"0.5": {"hit": 0.816, "n": 245, "yr_min": 0.77}, "0.3": {"hit": 0.793, "n": 353, "yr_min": 0.75}},
    "02": {"0.5": {"hit": 0.823, "n": 248, "yr_min": 0.78}, "0.3": {"hit": 0.772, "n": 356, "yr_min": 0.75}},
    "03": {"0.5": {"hit": 0.825, "n": 257, "yr_min": 0.79}, "0.3": {"hit": 0.776, "n": 370, "yr_min": 0.76}},
    "04": {"0.5": {"hit": 0.824, "n": 262, "yr_min": 0.79}, "0.3": {"hit": 0.786, "n": 378, "yr_min": 0.77}},
}
BASE_UP = 0.569
# 個股隔天與美期同向 (23:00 |Δ|≥0.5%，2024-05~2026-09，權值股 40 檔；App 只在 hit ≥ 0.60 時顯示)。
# 大盤第 2/3 天累積同向 71%/73%，但第 2 天當日只 58% ≈ 基準 → 美期資訊幾乎只作用在隔天。
STOCKS = {"1101": {"hit": 0.493, "n": 211, "yr_min": 0.479}, "1216": {"hit": 0.545, "n": 211, "yr_min": 0.527}, "1303": {"hit": 0.55, "n": 211, "yr_min": 0.479}, "2207": {"hit": 0.538, "n": 210, "yr_min": 0.5}, "2301": {"hit": 0.592, "n": 211, "yr_min": 0.495}, "2303": {"hit": 0.597, "n": 211, "yr_min": 0.495}, "2308": {"hit": 0.716, "n": 211, "yr_min": 0.688}, "2317": {"hit": 0.635, "n": 211, "yr_min": 0.604}, "2327": {"hit": 0.675, "n": 209, "yr_min": 0.625}, "2330": {"hit": 0.739, "n": 211, "yr_min": 0.681}, "2344": {"hit": 0.55, "n": 211, "yr_min": 0.527}, "2345": {"hit": 0.621, "n": 211, "yr_min": 0.582}, "2357": {"hit": 0.645, "n": 211, "yr_min": 0.571}, "2379": {"hit": 0.597, "n": 211, "yr_min": 0.521}, "2382": {"hit": 0.621, "n": 211, "yr_min": 0.583}, "2395": {"hit": 0.626, "n": 211, "yr_min": 0.562}, "2408": {"hit": 0.569, "n": 211, "yr_min": 0.505}, "2412": {"hit": 0.412, "n": 211, "yr_min": 0.352}, "2454": {"hit": 0.654, "n": 211, "yr_min": 0.625}, "2603": {"hit": 0.498, "n": 211, "yr_min": 0.417}, "2609": {"hit": 0.536, "n": 211, "yr_min": 0.486}, "2615": {"hit": 0.502, "n": 211, "yr_min": 0.484}, "2880": {"hit": 0.555, "n": 211, "yr_min": 0.527}, "2881": {"hit": 0.635, "n": 211, "yr_min": 0.528}, "2882": {"hit": 0.616, "n": 211, "yr_min": 0.542}, "2883": {"hit": 0.578, "n": 211, "yr_min": 0.542}, "2884": {"hit": 0.6, "n": 210, "yr_min": 0.556}, "2885": {"hit": 0.602, "n": 211, "yr_min": 0.556}, "2886": {"hit": 0.564, "n": 211, "yr_min": 0.495}, "2891": {"hit": 0.602, "n": 211, "yr_min": 0.556}, "2892": {"hit": 0.578, "n": 211, "yr_min": 0.528}, "3008": {"hit": 0.659, "n": 211, "yr_min": 0.604}, "3017": {"hit": 0.63, "n": 211, "yr_min": 0.569}, "3034": {"hit": 0.63, "n": 211, "yr_min": 0.583}, "3231": {"hit": 0.611, "n": 211, "yr_min": 0.583}, "3443": {"hit": 0.607, "n": 211, "yr_min": 0.571}, "3661": {"hit": 0.654, "n": 211, "yr_min": 0.521}, "3711": {"hit": 0.682, "n": 211, "yr_min": 0.639}, "5880": {"hit": 0.54, "n": 211, "yr_min": 0.472}, "6669": {"hit": 0.626, "n": 211, "yr_min": 0.611}}
# 隔天開盤跳空 ≈ a + β × ES 漲跌 (同期資料最小平方；|預估| > 0.15% 時的方向命中、逐年最低)。
# 對照：夜盤收盤後用完整夜盤 R² 0.52、方向 91% (07:00 發布會用)；晚上夜盤進行中改用這張表。
GAP = {"21": {"beta": 0.533, "a": 0.091, "dir_hit": 0.716, "n": 296, "yr_min": 0.669, "r2": 0.106}, "22": {"beta": 0.49, "a": 0.094, "dir_hit": 0.733, "n": 329, "yr_min": 0.696, "r2": 0.131}, "23": {"beta": 0.514, "a": 0.098, "dir_hit": 0.764, "n": 365, "yr_min": 0.729, "r2": 0.173}, "00": {"beta": 0.554, "a": 0.093, "dir_hit": 0.791, "n": 373, "yr_min": 0.772, "r2": 0.233}, "01": {"beta": 0.495, "a": 0.086, "dir_hit": 0.796, "n": 368, "yr_min": 0.764, "r2": 0.278}, "02": {"beta": 0.52, "a": 0.083, "dir_hit": 0.809, "n": 377, "yr_min": 0.785, "r2": 0.312}, "03": {"beta": 0.472, "a": 0.079, "dir_hit": 0.803, "n": 360, "yr_min": 0.789, "r2": 0.307}, "04": {"beta": 0.486, "a": 0.079, "dir_hit": 0.827, "n": 370, "yr_min": 0.814, "r2": 0.351}}


def _es_hourly(range_: str = "5d") -> pd.Series:
    import requests
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{SYM}", params={"interval": "60m", "range": range_},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    j = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("Asia/Taipei")
    s = pd.Series(j["indicators"]["quote"][0]["close"], index=ts).dropna()
    return s


def build(date: str | None) -> dict | None:
    """date = 最後一個台股交易日 (YYYY-MM-DD)。回傳基準價與統計表，App 以即時 ES 價計算叫牌。"""
    if not date:
        return None
    try:
        s = _es_hourly()
        t0 = pd.Timestamp(f"{date} 13:00", tz="Asia/Taipei")
        b = s[(s.index <= t0) & (s.index > t0 - pd.Timedelta(hours=3))]
        if b.empty:
            return None
        out = {"date": date, "sym": SYM, "base": round(float(b.iloc[-1]), 2), "base_ts": b.index[-1].strftime("%Y-%m-%d %H:%M"),
               "table": TABLE, "gap": GAP, "stocks": STOCKS, "base_up": BASE_UP, "start_hour": 21, "th_main": 0.5, "th_late": 0.3, "late_hour": 22,
               "note": "收盤後美股 S&P 期貨相對當天 13:00 的漲跌；21:00 起 ≥0.5% (22:00 起 ≥0.3%) 叫隔天台股同方向，命中 71~83% (2024-05~2026-09，575 日)"}
        last = s[s.index > t0]
        if len(last):
            out["last"] = round(float(last.iloc[-1]), 2); out["last_ts"] = last.index[-1].strftime("%Y-%m-%d %H:%M")
            out["chg"] = round((out["last"] / out["base"] - 1) * 100, 3)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("es_evening: %s", e)
        return None
