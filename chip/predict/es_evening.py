"""晚間美期叫牌 (2026-09-24，2026-09-25 修正時間對齊並加入多資產跳空)。

台股收盤後，美股 S&P 期貨 (ES=F) 相對當天 13:00 的漲跌 → 隔一交易日台股收盤方向；多資產 (ES/NQ/NKD/^SOX/TSM) → 開盤跳空。

時間對齊：Yahoo 60m 的時間戳是 K 棒「開始」時間。整點 H 的已知價 = K 棒結束 ≤ H 的最後一根 (開始 ≤ H−1 時)。
2026-09-24 版用「開始 ≤ H」，等於偷看 1 小時，21~23 點命中被高估 2~5 個百分點 (21 點實為 70%、2025 年 60%) → 改自 22:00 起叫牌。
統計表由 tools/es_evening_research.py --write --panel=<pkl> 產生於 data/models/es_evening_research.json；只保留 |Δ|≥0.5% 且命中 ≥70%、逐年最低 ≥65% 的整點 (全期統計，非走動式)。

重要限制 (2026-09-25 驗證)：命中是「隔天收盤 vs 今天收盤」，幾乎全部來自開盤跳空。
以隔天台指期 08:45 開盤進場、持有到收盤，方向命中只有約 50% (23:00 叫牌 49.7%、04:00 50.0%) → 沒有開盤後的交易優勢。
方向叫牌加入多資產 (NQ/NKD/SOX/TSM) 的研究被對抗審查推翻 (增益只在 2026 年，2025 年同覆蓋較差)，維持只用 ES；
開盤跳空的多資產模型兩年都穩健 (固定 |預估|>0.15%：80.8% vs ES 75.6%，R² 0.149→0.223)，自 23:00 起使用。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)
SYM = "ES=F"
MULTI_FUT = ("NQ=F", "NKD=F")        # 與 ES 同樣以 13:00 開始那根 K 的收盤為基準
# 2026-09-25 審查修正：連續合約 (ES=F 等) 每季換月時 Yahoo 會切到下一季合約 (實測 2026-09-14 23:00，ES 跳 +0.88%、NQ +1.0%)，
# 13:00 基準與晚間即時價若落在不同合約就會產生假漲跌 → 基準與 App 即時價一律用同一個「具體合約」(例 ESZ26.CME)。
# 選法：到期日距離 date ≥ ROLL_DAYS 的最近季月合約 (ES/NQ 到期 = 季月第三個週五；NKD = 季月第二個週五前一營業日)。
FUT_ROOT = {"ES=F": "ES", "NQ=F": "NQ", "NKD=F": "NKD"}
QMONTH = {3: "H", 6: "M", 9: "U", 12: "Z"}
ROLL_DAYS = 10
MULTI_US = ("^SOX", "TSM")           # 美股正規盤：相對前一美股交易日收盤；美股未開盤時漲跌 = 0
START_HOUR, LATE_HOUR = 22, None     # 22:00 起 |Δ|≥0.5% 叫牌。0.3~0.5% 區間 (審查修正後改算區間命中) 只有 61~68%，8 個整點僅 02 點過線 → 停用
BANDS = ("0.5",)
MIN_HIT, MIN_YR = 0.70, 0.65
RESEARCH_PATH = os.path.join(config.DATA_DIR if hasattr(config, "DATA_DIR") else "data", "models", "es_evening_research.json")
# 多資產開盤跳空 (wf_evening_multi，逐季擴張走動式 2025-01~2026-09；hit = |預估|>0.15% 方向命中的樣本外值)
GAP_MULTI = {"23": {"a": 0.0866, "w": {"es": -0.291, "nq": 0.4064, "nkd": 0.136, "sox": 0.0025, "tsm": 0.1043}, "hit": 0.792, "n": 274, "yr_min": 0.732, "r2": 0.155}, "00": {"a": 0.0884, "w": {"es": -0.1141, "nq": 0.2712, "nkd": 0.1073, "sox": 0.0065, "tsm": 0.0933}, "hit": 0.823, "n": 294, "yr_min": 0.768, "r2": 0.193}, "01": {"a": 0.0767, "w": {"es": 0.0498, "nq": 0.1634, "nkd": 0.1147, "sox": 0.0034, "tsm": 0.0939}, "hit": 0.823, "n": 317, "yr_min": 0.79, "r2": 0.226}, "02": {"a": 0.072, "w": {"es": 0.032, "nq": 0.148, "nkd": 0.1418, "sox": -0.002, "tsm": 0.1008}, "hit": 0.834, "n": 313, "yr_min": 0.826, "r2": 0.311}, "03": {"a": 0.0656, "w": {"es": 0.066, "nq": 0.1353, "nkd": 0.1238, "sox": 0.0003, "tsm": 0.1053}, "hit": 0.87, "n": 301, "yr_min": 0.861, "r2": 0.364}, "04": {"a": 0.0604, "w": {"es": -0.0035, "nq": 0.1633, "nkd": 0.1273, "sox": 0.007, "tsm": 0.0949}, "hit": 0.853, "n": 306, "yr_min": 0.836, "r2": 0.348}}


def _research() -> dict:
    for p in (RESEARCH_PATH, os.path.join("data", "models", "es_evening_research.json")):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            continue
    return {}


def _tables() -> tuple[dict, dict, dict, float]:
    r = _research()
    table = {}
    for k, row in (r.get("table") or {}).items():
        keep = {th: v for th, v in row.items() if th in BANDS and v.get("hit", 0) >= MIN_HIT and v.get("yr_min", 0) >= MIN_YR}
        if keep:
            table[k] = keep
    return table, r.get("gap") or {}, r.get("stocks") or {}, float(r.get("base_up") or 0.569)


def _hourly(sym: str, range_: str = "5d", interval: str = "60m") -> pd.Series:
    import requests
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"interval": interval, "range": range_},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    j = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("Asia/Taipei")
    return pd.Series(j["indicators"]["quote"][0]["close"], index=ts).dropna()


def _es_hourly(range_: str = "5d") -> pd.Series:
    return _hourly(SYM, range_)


def _expiry(root: str, y: int, m: int) -> dt.date:
    d = dt.date(y, m, 1)
    fri1 = d + dt.timedelta(days=(4 - d.weekday()) % 7)
    if root == "NKD":
        return fri1 + dt.timedelta(days=7) - dt.timedelta(days=1)   # 第二個週五的前一天
    return fri1 + dt.timedelta(days=14)                              # 第三個週五


def contract_code(cont_sym: str, date: str) -> str:
    """連續合約代號 → date 當晚應使用的具體季月合約 (Yahoo 格式，例 ESZ26.CME)。"""
    root = FUT_ROOT[cont_sym]
    d0 = dt.date.fromisoformat(date)
    y, m = d0.year, d0.month
    for _ in range(6):
        qm = ((m - 1) // 3 + 1) * 3
        if qm > 12:
            y, qm = y + 1, qm - 12
        if (_expiry(root, y, qm) - d0).days >= ROLL_DAYS:
            return f"{root}{QMONTH[qm]}{str(y)[2:]}.CME"
        m = qm + 1
        if m > 12:
            y, m = y + 1, 1
    raise ValueError(f"no contract for {cont_sym} {date}")


def _base13(s: pd.Series, date: str) -> tuple[float | None, str | None]:
    t0 = pd.Timestamp(f"{date} 13:00", tz="Asia/Taipei")
    b = s[(s.index <= t0) & (s.index > t0 - pd.Timedelta(hours=3))]
    return (round(float(b.iloc[-1]), 4), b.index[-1].strftime("%Y-%m-%d %H:%M")) if len(b) else (None, None)


def _raw_series(sym: str, range_: str, interval: str) -> pd.Series:
    """不 dropna：保留 Yahoo 的缺值列，讓呼叫端知道那一天其實有交易但缺收盤。"""
    import requests
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"interval": interval, "range": range_},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    j = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True)
    return pd.Series(j["indicators"]["quote"][0]["close"], index=ts, dtype="float64")


def _us_prev(sym: str, date: str) -> tuple[float | None, str | None]:
    """台股 date 13:30 之前最後一個美股正規盤收盤 = 美國日期 < date 的最後一根日K 收盤，回傳 (價, 美國日期)。
    2026-09-25 審查修正：先依日期過濾再處理缺值 (Yahoo ^SOX 2026-09-22 日K 收盤是 null，舊版 dropna 會悄悄拿到 9/21)；
    缺值時改用該日 60 分K 最後一根非空收盤，仍取不到就回傳 None (不輸出 multi)。"""
    try:
        d = _raw_series(sym, "10d", "1d")
        ny = d.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
        d = d[ny < date]
        if not len(d):
            return None, None
        day = d.index[-1].tz_convert("America/New_York").strftime("%Y-%m-%d")
        v = d.iloc[-1]
        if not np.isfinite(v):
            h = _raw_series(sym, "10d", "60m").dropna()
            h = h[h.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == day]
            if not len(h):
                return None, day
            v = h.iloc[-1]
        return round(float(v), 4), day
    except Exception as e:  # noqa: BLE001
        log.warning("es_evening us prev %s: %s", sym, e)
        return None, None


def build(date: str | None) -> dict | None:
    """date = 最後一個台股交易日 (YYYY-MM-DD)。回傳基準價與統計表，App 以即時價 (Worker /idx) 計算叫牌與開盤跳空。"""
    if not date:
        return None
    try:
        table, gap, stocks, base_up = _tables()
        es_code = contract_code(SYM, date)
        s = _hourly(es_code)
        base, base_ts = _base13(s, date)
        if base is None or not table:
            return None
        try:
            from ..sources import twse
            nxt = twse.next_trading_days(date, 1)[0]
        except Exception:  # noqa: BLE001
            nxt = None
        out = {"date": date, "next": nxt, "sym": es_code, "cont": SYM, "base": round(base, 2), "base_ts": base_ts, "table": table, "gap": gap, "stocks": stocks, "base_up": base_up,
               "start_hour": START_HOUR, "th_main": 0.5, "th_late": None, "late_hour": LATE_HOUR, "tradable_hit": 0.50, "stats_kind": "全期統計 (2024-05 起，非走動式)",
               "note": "收盤後美股 S&P 期貨 (具體季月合約) 相對當天 13:00 的漲跌；22:00 起 ≥0.5% 叫隔天台股收盤同方向，全期統計命中 75~83% (2024-05 起，非走動式)。"
                       "命中幾乎全來自開盤跳空：隔天台指期開盤才進場、持有到收盤只有約 50%"}
        t0 = pd.Timestamp(f"{date} 13:00", tz="Asia/Taipei")
        last = s[s.index > t0]
        if len(last):
            out["last"] = round(float(last.iloc[-1]), 2); out["last_ts"] = last.index[-1].strftime("%Y-%m-%d %H:%M")
            out["chg"] = round((out["last"] / out["base"] - 1) * 100, 3)
        mb, mp = {}, {}
        codes = {"ES=F": es_code}
        for sym in MULTI_FUT:
            try:
                codes[sym] = contract_code(sym, date)
                v, _ = _base13(_hourly(codes[sym]), date)
                if v:
                    mb[codes[sym]] = v
            except Exception as e:  # noqa: BLE001
                log.warning("es_evening base %s: %s", sym, e)
        us_days = {}
        for sym in MULTI_US:
            v, day = _us_prev(sym, date)
            if v:
                mp[sym] = v; us_days[sym] = day
        same_day = len(set(us_days.values())) == 1   # ^SOX 與 TSM 的前收必須是同一個美股交易日
        if len(mb) == len(MULTI_FUT) and len(mp) == len(MULTI_US) and same_day:
            out["multi"] = {"base": {es_code: out["base"], **mb}, "prev": mp, "prev_day": next(iter(us_days.values())), "gap": GAP_MULTI,
                            "keys": {"es": es_code, "nq": codes["NQ=F"], "nkd": codes["NKD=F"], "sox": "^SOX", "tsm": "TSM"}}
        elif not same_day:
            log.warning("es_evening: 美股前收日期不一致 %s，不輸出 multi", us_days)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("es_evening: %s", e)
        return None
