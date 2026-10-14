"""融資行為動態追蹤與分析 (margin_dyn v1.0；規格 scratchpad/margin/SPEC.md)。

追蹤全市場 (TWSE 為主、TPEx 合計另列) 與個股的融資「行為」，不做方向判斷：
- market   §2.1 水準 / 動態 / 流量 / 擴散 / 維持率 (TWSE 自算 + 玩股網對照) / 融券 / 上櫃 / 60 日行為 / 週別 / 12 類事件統計
- history  2001~ 日序列 (Tier-1 不可丟：fresh ∪ SQLite daily_metrics ∪ prev ∪ 種子)；flows 2 年；derived 400 日 (App 圖表)
- stocks   追蹤清單 5 檔 + 雷達候選池 ≤100 檔 (§2.2：重述規則 §3.6、停資遮罩、FIFO 估計成本、融資壓力分數)
- ranking  §2.3 七榜 ×15 + ETF 小榜，連續上榜由 snapshots
- alerts   ≤12 則，全描述

誠實標籤：label ∈ {predictive, descriptive, risk_descriptive}；predictive 只來自 data/models/margin_validated.json (目前空 → 不出現)。
事件 fwd 自 D+1 收盤起算；對照 = 同日 ret1/ret5/ret20/bias20/rv20 最近鄰 k=10 + ret20 卡尺 ±2 pt、年差 ≤4；CI = episode 與年 cluster bootstrap (seed 固定)。
資料層 (twse.margin_stocks_hist / closes_all、tpex.margin_balance、finmind.total_margin_flows、wantgoo.maint_real) 以 lazy import 取用，
尚未實作或失敗時該段沿用 prev (meta.carried_parts)。本模組只寫 SQLite 的 margin_* 新表與 daily_metrics 的 mg_* / maint_* / tpex_* / 擴散 新 key。
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, store
from ..http import cache_delete, num, session
from ..serialize import clean
from . import gov8
from .common import streak, zscore

log = logging.getLogger(__name__)

PAGES_URL = gov8.PAGES_URL
NAME = "margin"
RAW_URL = "https://raw.githubusercontent.com/joekisoul-code/TaiexChipAnalyzer/gh-pages/data/margin.json"
MAX_STOCKS = 100
HISTORY_FROM = "2001-01-01"
MAINT_FROM = "2004-02-11"          # MI_INDEX 回補下限
VERSION = "1.0"
SEED = 20261014
BOOT_B = 1000
MATCH_K = 10
CALIPER_RET20 = 2.0
YEAR_DIFF = 4
EPISODE_GAP = 10
THROTTLE = 2.2                     # TWSE 節流秒數 (測試設 0)
SIZE_LIMIT = 900 * 1024
PANEL_FROM = os.getenv("MARGIN_PANEL_FROM") or "2024-10-01"   # 個股面板 / 回補起點；Actions 以 vars.MARGIN_PANEL_FROM 往前拉 (空字串 = 預設)
TPEX_FROM = "2007-01-01"           # TPEx 上櫃餘額可回補下限 (SPEC §2.1)
NODATA_GRACE_DAYS = 1              # TWSE ALL NoData：資料日 ≥ today − 1 日 → 視為尚未公布 (cron 延遲跨午夜時不把前一交易日永久寫進 margin_nodata；與 tpex.NODATA_GRACE_DAYS 一致)
MAINT_FULL_FROM = "2004-06-30"     # maint_pct2004 / 「2004 年來」文案只在自算維持率覆蓋期起點 ≤ 此日時輸出，否則 null 並改寫「{maint_from} 以來」
MIN_QUANTILE_DAYS = 2500           # ext_inc / ext_dec / turnover_hot 的全歷史分位門檻至少要這麼多日樣本，否則事件輸出 null (insufficient)
FFILL_MAX_AGE = 60                 # 維持率自算：缺價檔以最後已知收盤補價的最大年齡 (交易日)
LAST_PX_DAYS = 400                 # 維持率自算：往前找最後收盤的日曆天上限 (>60 日的只用來估缺價市值、不補價)
UNEST_LOTS_PARTIAL = 1000          # 維持率自算：完全估不出價的無價檔張數合計 ≥ 此值 → partial
RAW_CACHE_KEYS = ("twse:MI_MARGN:ALL:{ymd}", "tpex:margin:{ymd}")   # 個股列入 SQLite 後即刪的整份原始快取 (≈420 / 343 KB/日；SPEC §3.1 不留 data/cache)
VALIDATED_PATH = config.DATA_DIR / "models" / "margin_validated.json"
SEED_PATH = config.DATA_DIR / "margin_market.json"
WATCH_IDS = ["2330", "00631L", "00685L", "00981A", "00988A"]
LABELS = {"predictive": "樣本外配對對照顯著且逐年一致 (目前無)", "descriptive": "行為描述，對未來漲跌無可驗證差異",
          "risk_descriptive": "槓桿 / 強制賣壓的風險描述"}
LABEL_SET = tuple(LABELS)
QUADS = ("資增價漲", "資增價跌", "資減價跌", "資減價漲", "中性")
TESTS_TOTAL = {"indicators": 153, "events": 114, "stock_tests": 162, "module_events": 36}
FORBIDDEN_RE = re.compile("買進|賣出|加碼|減碼|放空|做多|做空|建議買|建議賣|抄底|逃頂|縮小部位|進場|持有|持股水位|續抱|試單|承接|停損|"
                          "斷頭|散戶|追價過熱|籌碼沉澱|清洗|護盤|偏弱|偏強")
HISTORY_COLS = ["date", "margin_amt", "margin_lots", "short_lots", "maint_twse", "tpex_margin_amt"]
FLOW_COLS = ["date", "buy_lots", "sell_lots", "repay_lots", "buy_amt", "sell_amt", "short_buy", "short_sell", "short_repay", "br_net", "usage_gt60_pct"]
DERIVED_COLS = ["date", "chg1", "z5", "pct20", "ret20", "div20", "short_ratio", "turnover_ratio", "maint", "regime"]
DAILY_COLS = ["md", "lots", "buy", "sell", "repay", "short", "offset", "close", "util"]
PARTS = ("market", "history", "flows", "derived", "ranking", "stocks", "alerts")
# history 欄 ↔ daily_metrics key (不碰 market.py 既有的 margin_amt / maint_ratio 等 key)
METRIC_KEYS = {"margin_amt": "mg_amt", "margin_lots": "mg_lots", "short_lots": "mg_short", "close": "twii_close", "amount": "twii_amount",
               "buy_lots": "mg_buy_lots", "sell_lots": "mg_sell_lots", "repay_lots": "mg_repay_lots", "buy_amt": "mg_buy_amt", "sell_amt": "mg_sell_amt",
               "repay_amt": "mg_repay_amt", "short_buy": "mg_short_buy", "short_sell": "mg_short_sell", "short_repay": "mg_short_repay",
               "maint_twse": "maint_twse", "maint_real": "maint_real", "maint_cov": "maint_cov",
               "br_up": "br_up", "br_dn": "br_dn", "br_net": "br_net", "conc_top20": "conc_top20", "usage_med": "usage_med", "usage_gt60_pct": "usage_gt60",
               "usage_gt80_n": "usage_gt80_n", "offset_vol_share": "offset_vol_share", "etf_lots_share": "etf_lots_share", "n_valid": "mg_n_valid",
               "tpex_margin_lots": "tpex_margin_lots", "tpex_margin_amt": "tpex_margin_amt", "tpex_short_lots": "tpex_short_lots"}
MARGIN_RATIO = 0.6
MIN_BASE_LOTS = 50
RECOVER_DAYS = 40
BURN_IN = 120
STOCK_KEYS = ["code", "name", "m_buy", "m_sell", "m_repay", "m_prev", "m_today", "m_limit", "s_buy", "s_sell", "s_repay", "s_prev", "s_today",
              "s_limit", "offset", "note"]
_ALIASES = {"margin_buy": "m_buy", "margin_sell": "m_sell", "margin_repay": "m_repay", "margin_prev": "m_prev", "margin_today": "m_today",
            "margin_limit": "m_limit", "short_buy": "s_buy", "short_sell": "s_sell", "short_repay": "s_repay", "short_prev": "s_prev",
            "short_today": "s_today", "short_limit": "s_limit", "stock_id": "code", "stock_name": "name"}
# 事件 id → (預設 label, grade, 內建證據)
EVENT_DEFS = {
    "streak_up7": ("descriptive", "weak", {"note": "h5 配對超額 +0.43~+0.50 [0.01~0.11, 0.77~0.93]、h10 +0.41、h20 ≈0；16/22 年；p≈0.028 不過 FDR",
                                           "placebo": "純價格安慰劑「近 7 日 ≥6 日上漲」h5 +0.90 [0.53, 1.28] 更大 → 連增與指數連漲性質相近"}),
    "streak_dn7": ("descriptive", "none", {"note": "CI 含 0"}),
    "wipeout": ("risk_descriptive", "weak_upper_bound", {"note": "2010~ 無條件 10 日 +1.41% / 勝率 65.9%；配對超額 +0.72 [−0.10, 1.74] ~ +0.90 [0.13, 2.00] 為上界 (SMD −0.66)",
                                                         "placebo": "對照 ret20 −3.7~−4.5% vs 事件 −7.5%；2015~ episode 第一日 −2.79 [−5.10, −0.34] (n=19)；效應為跌深反彈而非融資"}),
    "chase": ("descriptive", "none", {"note": "CI 含 0"}),
    "knife": ("descriptive", "none", {"note": "CI 含 0"}),
    "ext_inc": ("descriptive", "none", {"note": "CI 含 0"}),
    "ext_dec": ("descriptive", "none", {"note": "CI 含 0"}),
    "hi250": ("descriptive", "none", {"note": "年去均值後 h10 −0.17 [−1.17, 1.03]"}),
    "lo250": ("descriptive", "none", {"note": "2020~ 時代效應"}),
    "maint_lt150": ("risk_descriptive", "none", {"note": "真實維持率五分位對 fwd20 非單調 (A10)"}),
    "turnover_hot": ("descriptive", "none", {"note": "—"}),
    "deleverage_fast": ("risk_descriptive", "none", {"note": "A7 的維持率代理版已廢；本定義用 TWSE 自算值"}),
}
# 事件定義句 (active=False 時 text 只給定義、不帶今日數值；{y} = 序列起始年)
EVENT_DEFS_TEXT = {
    "streak_up7": "融資連增 ≥7 日", "streak_dn7": "融資連減 ≥7 日", "wipeout": "融資 20 日 <−4% 且指數 20 日 <−3%",
    "chase": "融資 20 日增幅高於指數 >5 個百分點", "knife": "融資 20 日 >+2% 且指數 20 日 <−2%",
    "ext_inc": "單日融資變化 ≥ {y} 年來第 95 百分位", "ext_dec": "單日融資變化 ≤ {y} 年來第 5 百分位",
    "hi250": "融資餘額創 250 日新高", "lo250": "融資餘額創 250 日新低", "maint_lt150": "維持率低於 150",
    "turnover_hot": "日周轉 ≥ {y} 年來第 95 百分位", "deleverage_fast": "5 日融資 <−3% 且維持率 5 日降 ≥8 個百分點",
}


try:                                           # 資料層共用例外 (chip/sources/__init__.py)；尚未存在時用同名本地類別，_is() 以類別名判斷兩者皆可
    from ..sources import NoData, NotYet
except Exception:  # noqa: BLE001
    class NoData(Exception):
        """休市 / 週間假日 / 未公布 (TWSE 三者同回應)。"""

    class NotYet(Exception):
        """交易日但尚未公布 (TPEx 0 列)。"""


class Unavailable(Exception):
    """資料層函式尚未實作 / 模組不存在。"""


def _is(e: BaseException, name: str) -> bool:
    return type(e).__name__ == name or any(k.__name__ == name for k in type(e).__mro__)


# ------------------------------------------------------------------ 資料層 (lazy import；同事同時在寫，簽名以 SPEC §3.1 為準)
def _fn(mod: str, name: str):
    try:
        m = importlib.import_module(f"chip.sources.{mod}")
    except Exception as e:  # noqa: BLE001
        raise Unavailable(f"chip.sources.{mod}: {e}") from e
    f = getattr(m, name, None)
    if f is None:
        raise Unavailable(f"chip.sources.{mod}.{name} 尚未實作")
    return f


def _fetch_margin_all(ymd: str):
    return _fn("twse", "margin_stocks_hist")(ymd)


def _fetch_closes(ymd: str):
    return _fn("twse", "closes_all")(ymd)


def _fetch_tpex(ymd: str):
    return _fn("tpex", "margin_balance")(ymd)


def _fetch_total_flows():
    return _fn("finmind", "total_margin_flows")()


def _fetch_total_margin():
    return _fn("finmind", "total_margin")(HISTORY_FROM)


def _fetch_twse_ms():
    return _fn("twse", "margin_daily")()


def _fetch_taiex():
    return _fn("finmind", "taiex_price")(HISTORY_FROM)


def _fetch_maint_real():
    return _fn("wantgoo", "maint_real")()


def _fetch_holidays(year: int):
    return _fn("twse", "holidays")(year)


def _sleep() -> None:
    """請求間節流。資料層 (chip.sources.twse / tpex) 自帶 MIN_GAP 節流且只在真的打網路時等 (快取命中不等)，
    有的話這裡不再重複等 (否則每日多 4.4 s，450 日回補會超過 60 分)；資料層沒有節流機制才用 THROTTLE。"""
    if THROTTLE <= 0:
        return
    try:
        tw = importlib.import_module("chip.sources.twse")
        if float(getattr(tw, "MIN_GAP", 0) or 0) > 0 and callable(getattr(tw, "_throttle", None)):
            return
    except Exception:  # noqa: BLE001
        pass
    time.sleep(THROTTLE)


def _ymd(date: str) -> str:
    return str(date)[:10].replace("-", "")


def _iso(s) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    if re.fullmatch(r"\d{2,3}/\d{1,2}/\d{1,2}", s):
        y, m, d = s.split("/")
        return f"{int(y) + 1911:04d}-{int(m):02d}-{int(d):02d}"
    return s[:10]


def _r(v, d: int = 2):
    try:
        if v is None or (isinstance(v, float) and v != v) or (isinstance(v, (np.floating,)) and np.isnan(v)):
            return None
        return round(float(v), d) if d else int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _f(v, fmt: str = "{:+.1f}", na: str = "n/a") -> str:
    v = _r(v, 6)
    return na if v is None else fmt.format(v)


def _fp(v, fmt: str = "{:+.1f}", unit: str = "%") -> str:
    """數值 + 單位；None → 'n/a' (不帶單位)。"""
    v = _r(v, 6)
    return "n/a" if v is None else fmt.format(v) + unit


# ------------------------------------------------------------------ 已發布資料 / 驗證名單 / 交易日曆
def published(prev: dict | None = None) -> dict:
    """Pages margin.json → raw gh-pages 備援 (同 gov8.published)。"""
    if prev is not None:
        return prev
    for url in (PAGES_URL.rstrip("/") + f"/data/{NAME}.json", RAW_URL):
        try:
            r = session().get(url, timeout=20)
            if r.ok:
                j = json.loads(r.content.decode("utf-8", errors="replace"))
                if isinstance(j, dict) and j:
                    return j
        except Exception as e:  # noqa: BLE001
            log.debug("published %s unavailable (%s): %s", NAME, url, e)
    return {}


def validated_ids(path: Path | str | None = None) -> list[str]:
    """data/models/margin_validated.json → 已驗證 id 清單 (目前空)。格式：[id,...] 或 {"validated":[{id,...}|id]}。"""
    p = Path(path) if path else VALIDATED_PATH
    try:
        j = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    items = j.get("validated", []) if isinstance(j, dict) else j
    out = []
    for it in items or []:
        sid = it.get("id") if isinstance(it, dict) else it
        if isinstance(sid, str) and sid:
            out.append(sid)
    return out


def _holidays_for(year: int) -> set[str]:
    try:
        return {str(x)[:10] for x in _fetch_holidays(year)}
    except Exception as e:  # noqa: BLE001
        p = config.DATA_DIR / f"holidays_{year}.json"
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                return {str(x)[:10] for x in (j if isinstance(j, list) else list(j))}
            except Exception:  # noqa: BLE001
                pass
        log.warning("holidays %s unavailable (%s) → 只排週末", year, e)
        return set()


def trading_days(start: str, end: str) -> list[str]:
    """TWSE 休市表 ∪ 週末 (絕不用 Yahoo 日曆)。"""
    s, e = dt.date.fromisoformat(str(start)[:10]), dt.date.fromisoformat(str(end)[:10])
    if e < s:
        return []
    hol: set[str] = set()
    for y in range(s.year, e.year + 1):
        hol |= _holidays_for(y)
    out = []
    d = s
    while d <= e:
        if d.weekday() < 5 and d.isoformat() not in hol:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


# ------------------------------------------------------------------ SQLite：margin_stock_daily / margin_nodata (規格 §3.1 schema)
def _ensure(c) -> None:
    c.execute("""CREATE TABLE IF NOT EXISTS margin_stock_daily (date TEXT, code TEXT, mkt TEXT, name TEXT, note TEXT,
        m_buy REAL, m_sell REAL, m_repay REAL, m_prev REAL, m_today REAL, m_limit REAL,
        s_buy REAL, s_sell REAL, s_repay REAL, s_prev REAL, s_today REAL, s_limit REAL, offset REAL,
        close REAL, vol_lots REAL, PRIMARY KEY(date, code))""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_msd_code ON margin_stock_daily(code, date)")
    c.execute("CREATE TABLE IF NOT EXISTS margin_nodata (date TEXT, src TEXT, PRIMARY KEY(date, src))")


def _norm_stock_row(r) -> dict | None:
    if isinstance(r, (list, tuple)):
        if len(r) < 15:
            return None
        out = {"code": str(r[0]).strip(), "name": str(r[1]).strip()}
        for k, v in zip(STOCK_KEYS[2:15], r[2:15]):
            out[k] = num(v)
        out["note"] = str(r[15]).strip() if len(r) > 15 and r[15] is not None else ""
        return out
    if isinstance(r, dict):
        out = {}
        for k, v in r.items():
            kk = _ALIASES.get(k, k)
            if kk in STOCK_KEYS:
                out[kk] = v
        if not out.get("code"):
            return None
        out["code"] = str(out["code"]).strip()
        out["name"] = str(out.get("name") or "").strip()
        out["note"] = str(out.get("note") or "").strip()
        for k in STOCK_KEYS[2:15]:
            out[k] = num(out.get(k))
        return out
    return None


def _norm_market(m) -> dict | None:
    """tables[0] 三列 (資 / 券 / 金額) 或已整理 dict → {margin_lots, margin_amt(億), buy_lots, sell_lots, repay_lots, buy_amt, sell_amt, repay_amt, short_*}。"""
    if m is None:
        return None
    out: dict = {}
    if isinstance(m, dict) and not any(str(k).startswith("融") for k in m):
        if "margin_lots" in m or "margin_amt" in m:
            amt = {k: m.get(k) for k in ("margin_amt", "margin_buy_amt", "margin_sell_amt", "margin_repay_amt", "margin_amt_prev")}
            out.update(margin_lots=m.get("margin_lots"), margin_lots_prev=m.get("margin_lots_prev"), buy_lots=m.get("margin_buy_lots", m.get("buy_lots")),
                       sell_lots=m.get("margin_sell_lots", m.get("sell_lots")), repay_lots=m.get("margin_repay_lots", m.get("repay_lots")),
                       short_lots=m.get("short_lots"), short_buy=m.get("short_buy_lots", m.get("short_buy")), short_sell=m.get("short_sell_lots", m.get("short_sell")),
                       short_repay=m.get("short_repay_lots", m.get("short_repay")), margin_amt=amt["margin_amt"], buy_amt=amt["margin_buy_amt"],
                       sell_amt=amt["margin_sell_amt"], repay_amt=amt["margin_repay_amt"])
            if out.get("repay_lots") is None and all(out.get(k) is not None for k in ("margin_lots", "margin_lots_prev", "buy_lots", "sell_lots")):
                out["repay_lots"] = out["margin_lots_prev"] + out["buy_lots"] - out["sell_lots"] - out["margin_lots"]
            return out
        return None
    rows = m.get("data", m) if isinstance(m, dict) else m
    if isinstance(rows, dict):
        rows = [[k] + list(v) for k, v in rows.items()]
    for row in rows or []:
        if not row:
            continue
        item = str(row[0])
        vals = [num(x) for x in row[1:6]]
        if len(vals) < 5:
            continue
        buy, sell, repay, prev, today = vals
        if item.startswith("融資(交易") or item.startswith("融資(") or item == "融資":
            out.update(buy_lots=buy, sell_lots=sell, repay_lots=repay, margin_lots_prev=prev, margin_lots=today)
        elif item.startswith("融券"):
            out.update(short_buy=buy, short_sell=sell, short_repay=repay, short_lots_prev=prev, short_lots=today)
        elif item.startswith("融資金額"):
            k = 1e3 / 1e8
            out.update(buy_amt=buy * k if buy is not None else None, sell_amt=sell * k if sell is not None else None,
                       repay_amt=repay * k if repay is not None else None, margin_amt_prev=prev * k if prev is not None else None,
                       margin_amt=today * k if today is not None else None)
    return out or None


def _norm_day(res) -> tuple[str | None, dict | None, list[dict]]:
    """margin_stocks_hist 回傳 {"date","market","stocks"} 或本機快取 (列 dict 清單) → (date, market, stock rows)。"""
    if res is None:
        return None, None, []
    if isinstance(res, list):
        rows = [x for x in (_norm_stock_row(r) for r in res) if x]
        date = _iso(res[0].get("date")) if res and isinstance(res[0], dict) else None
        return date, None, rows
    if isinstance(res, pd.DataFrame):
        return _norm_day(res.to_dict("records"))
    if isinstance(res, dict):
        stocks = res.get("stocks")
        if stocks is None:
            stocks = res.get("rows", [])
        if isinstance(stocks, pd.DataFrame):
            stocks = stocks.to_dict("records")
        rows = [x for x in (_norm_stock_row(r) for r in stocks or []) if x]
        return _iso(res.get("date")), _norm_market(res.get("market")), rows
    return None, None, []


def _norm_closes(x) -> dict[str, tuple[float | None, float | None]]:
    """closes_all → {code: (close, vol_lots)}；也接受 {date, rows:{code:{close, volume}}} (本機 mi_index_all 快取)。"""
    if x is None:
        return {}
    if isinstance(x, dict) and "rows" in x and isinstance(x["rows"], dict):
        x = x["rows"]
    if isinstance(x, pd.DataFrame):
        x = {str(r["code"]): (r.get("close"), r.get("vol_lots", r.get("volume"))) for _, r in x.iterrows()}
    out = {}
    for code, v in (x or {}).items():
        if isinstance(v, dict):
            cl = num(v.get("close"))
            vl = v.get("vol_lots")
            if vl is None and v.get("volume") is not None:
                vl = num(v["volume"])
                vl = vl / 1000 if vl is not None else None
            out[str(code)] = (cl, num(vl))
        elif isinstance(v, (list, tuple)):
            out[str(code)] = (num(v[0]) if len(v) else None, num(v[1]) if len(v) > 1 else None)
        else:
            out[str(code)] = (num(v), None)
    return out


def _panel_upsert(date: str, rows: list[dict], closes: dict | None = None, mkt: str = "twse") -> int:
    closes = closes or {}
    tuples = []
    for r in rows:
        cl, vl = closes.get(r["code"], (None, None))
        tuples.append((date, r["code"], mkt, r.get("name"), r.get("note") or "", r.get("m_buy"), r.get("m_sell"), r.get("m_repay"), r.get("m_prev"),
                       r.get("m_today"), r.get("m_limit"), r.get("s_buy"), r.get("s_sell"), r.get("s_repay"), r.get("s_prev"), r.get("s_today"),
                       r.get("s_limit"), r.get("offset"), cl, vl))
    if not tuples:
        return 0
    with store.conn() as c:
        _ensure(c)
        c.executemany("INSERT OR REPLACE INTO margin_stock_daily VALUES (" + ",".join("?" * 20) + ")", tuples)
    return len(tuples)


def _panel_update_closes(date: str, closes: dict) -> int:
    rows = [(cl, vl, date, code) for code, (cl, vl) in closes.items() if cl is not None]
    if not rows:
        return 0
    with store.conn() as c:
        _ensure(c)
        c.executemany("UPDATE margin_stock_daily SET close=?, vol_lots=? WHERE date=? AND code=?", rows)
    return len(rows)


def _panel_dates() -> list[str]:
    with store.conn() as c:
        _ensure(c)
        return [r[0] for r in c.execute("SELECT DISTINCT date FROM margin_stock_daily ORDER BY date").fetchall()]


def _panel_dates_without_close() -> list[str]:
    with store.conn() as c:
        _ensure(c)
        return [r[0] for r in c.execute("SELECT date FROM margin_stock_daily GROUP BY date HAVING SUM(close IS NOT NULL)=0 ORDER BY date DESC").fetchall()]


def _nodata(src: str) -> set[str]:
    with store.conn() as c:
        _ensure(c)
        return {r[0] for r in c.execute("SELECT date FROM margin_nodata WHERE src=?", (src,)).fetchall()}


def _mark_nodata(date: str, src: str) -> None:
    with store.conn() as c:
        _ensure(c)
        c.execute("INSERT OR REPLACE INTO margin_nodata VALUES (?,?)", (date, src))


def load_panel(since: str | None = None, codes: list[str] | None = None, cols: list[str] | None = None) -> pd.DataFrame:
    """margin_stock_daily → DataFrame (依 code, date 排序)。"""
    q = "SELECT " + (",".join(cols) if cols else "*") + " FROM margin_stock_daily"
    cond, params = [], []
    if since:
        cond.append("date>=?"); params.append(since)
    if codes:
        cond.append(f"code IN ({','.join('?' * len(codes))})"); params.extend(codes)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY code, date"
    with store.conn() as c:
        _ensure(c)
        return pd.read_sql_query(q, c, params=params)


def import_seed(cache_dir=None, parquet_dir=None) -> int:
    """一次性：本機 data/cache 的 twse:MI_MARGN:ALL:* 快取 (列 dict 清單) + 研究 twse_raw parquet → margin_stock_daily；回傳匯入列數。
    收盤價不在種子裡 (之後由 backfill 的 closes 階段補、維持率由 build 的 _recompute_maint 補)。MS 快取 (全市場流量) 同時寫 daily_metrics。"""
    have = set(_panel_dates())
    days: dict[str, list[dict]] = {}
    ms: dict[str, dict] = {}
    if cache_dir and Path(cache_dir).exists():
        for p in Path(cache_dir).glob("*.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    head = f.read(200)
                i = head.find('"key": "')
                if i < 0:
                    continue
                key = head[i + 8: head.find('"', i + 8)]
                if key.startswith("twse:MI_MARGN:ALL:"):
                    date = _iso(key.rsplit(":", 1)[1])
                    if date in have or not date:
                        continue
                    blob = json.loads(p.read_text(encoding="utf-8"))
                    d2, _, rows = _norm_day(blob.get("data"))
                    if rows:
                        days[d2 or date] = rows
                elif key.startswith("twse:MI_MARGN:MS:"):
                    blob = json.loads(p.read_text(encoding="utf-8"))
                    m = _norm_market(blob.get("data"))
                    if m and blob.get("data", {}).get("date"):
                        ms[_iso(blob["data"]["date"])] = m
            except Exception as e:  # noqa: BLE001
                log.debug("seed cache %s: %s", p.name, e)
    if parquet_dir and Path(parquet_dir).exists():
        for p in sorted(Path(parquet_dir).glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                date = _iso(df["date"].iloc[0]) if "date" in df and len(df) else _iso(p.stem)
                if not date or date in have or date in days:
                    continue
                _, _, rows = _norm_day(df)
                if rows:
                    days[date] = rows
            except Exception as e:  # noqa: BLE001
                log.debug("seed parquet %s: %s", p.name, e)
    n = 0
    for date in sorted(days):
        rows = days[date]
        n += _panel_upsert(date, rows, None)
        try:
            _day_metrics(date, rows, None, ms.get(date))
        except Exception as e:  # noqa: BLE001
            log.debug("seed metrics %s: %s", date, e)
    for date, m in ms.items():
        if date not in days:
            try:
                store.upsert_metrics(date, _flow_metrics(m))
            except Exception as e:  # noqa: BLE001
                log.debug("seed ms %s: %s", date, e)
    return n


# ------------------------------------------------------------------ 當日面板 → 擴散 / 維持率 / 流量 metrics
def _flow_metrics(m: dict | None) -> dict:
    if not m:
        return {}
    out = {}
    for col in ("buy_lots", "sell_lots", "repay_lots", "buy_amt", "sell_amt", "repay_amt", "short_buy", "short_sell", "short_repay",
                "margin_amt", "margin_lots", "short_lots"):
        v = m.get(col)
        if v is not None and v == v:
            out[METRIC_KEYS[col]] = float(v)
    return out


def _is_etf(code) -> bool:
    return str(code).startswith("00")


def cross_section(day_panel: pd.DataFrame, closes: dict | None = None) -> dict:
    """br_up / br_dn / br_net、conc_top20、usage_med / gt60_pct / gt80_n、offset_vol_share、etf_lots_share、n_valid (ETF 與無融資檔不入有效檔)。"""
    out = {"br_up": None, "br_dn": None, "br_net": None, "conc_top20": None, "usage_med": None, "usage_gt60_pct": None, "usage_gt80_n": None,
           "offset_vol_share": None, "etf_lots_share": None, "n_valid": 0}
    if day_panel is None or len(day_panel) == 0:
        return out
    d = day_panel.copy()
    for c in ("m_today", "m_prev", "m_limit", "offset", "vol_lots"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    d["code"] = d["code"].astype(str)
    note = d["note"].fillna("").astype(str) if "note" in d else pd.Series("", index=d.index)
    if closes:
        cl = {k: v for k, v in closes.items()}
        d["vol_lots"] = d["vol_lots"].where(d["vol_lots"].notna(), d["code"].map(lambda k: (cl.get(k) or (None, None))[1]))
    etf = d["code"].map(_is_etf)
    active = d[(~etf) & ((d["m_today"] > 0) | (d["m_prev"] > 0))]
    n = int(len(active))
    out["n_valid"] = n
    if n:
        chg = active["m_today"] - active["m_prev"]
        up, dn = int((chg > 0).sum()), int((chg < 0).sum())
        out.update(br_up=round(up / n * 100, 2), br_dn=round(dn / n * 100, 2), br_net=round((up - dn) / n * 100, 2))
        tot = float(chg.abs().sum())
        out["conc_top20"] = round(float(chg.abs().nlargest(20).sum() / tot * 100), 2) if tot > 0 else None
        ok = active["m_limit"].notna() & (active["m_limit"] > 0) & ~note.loc[active.index].str.contains("O")
        usage = (active.loc[ok, "m_today"] / active.loc[ok, "m_limit"] * 100).dropna()
        if len(usage):
            out.update(usage_med=round(float(usage.median()), 2), usage_gt60_pct=round(float((usage > 60).mean() * 100), 2), usage_gt80_n=int((usage > 80).sum()))
    vol = d["vol_lots"]
    if vol.notna().any() and float(vol.fillna(0).sum()) > 0:
        has = vol.notna()
        out["offset_vol_share"] = round(float(d.loc[has, "offset"].fillna(0).sum() / vol[has].sum() * 100), 3)
    tot_lots = float(d["m_today"].fillna(0).sum())
    if tot_lots > 0:
        out["etf_lots_share"] = round(float(d.loc[etf, "m_today"].fillna(0).sum() / tot_lots * 100), 2)
    return out


def _px_of(v):
    """closes / last_px / next_px 的值 → (close, extra...)；接受 float、(close,)、(close, age)、(close, age, lots)、(close, lots)。"""
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v,)


def _est_px(lots_now: float, ref_close, ref_lots) -> float | None:
    """以鄰近日的收盤估無價檔的價格：股本事件 (分割 / 反分割 / 面額變更) 時張數與價格反向變動，用張數比還原到當日股數基準
    (6949 面額變更：停牌 5,315 張、復牌 105,083 張 × 67.1 → 估 ≈1,327)；比值不在 [1/50, 50] 或任一端張數 ≤0 → 不調整。"""
    if ref_close is None or ref_close != ref_close or ref_close <= 0:
        return None
    ratio = 1.0
    try:
        if ref_lots is not None and ref_lots == ref_lots and ref_lots > 0 and lots_now > 0:
            r = float(ref_lots) / float(lots_now)
            if 1 / 50 <= r <= 50:
                ratio = r
    except (TypeError, ValueError):
        pass
    return float(ref_close) * ratio


def maint_direct(day_panel: pd.DataFrame, closes: dict | None, margin_amt: float | None, last_px: dict | None = None,
                 next_px: dict | None = None) -> tuple[float | None, float | None, str | None]:
    """自算維持率 = Σ(m_today × close × 1000) / (融資金額 億 × 1e8) × 100。
    缺收盤檔用 last_px {code: (close, age[, lots_then])} 補價 (age ≤ FFILL_MAX_AGE 交易日)；補不到價的檔不進分子，但以最近之後收盤 next_px
    {code: (close, lots_next)} 或較舊的 last_px (× 張數比，§3.6 股本事件還原) 估其市值：任一補價 / 無價檔估計貢獻 >0.5 pt、估計缺價市值占比 >1%、
    無價張數占比 >1%、或完全估不出價的無價檔張數合計 ≥ UNEST_LOTS_PARTIAL → 'partial'。
    回 (value, coverage%, src)；coverage = 100 − max(無價張數占比, 估計缺價市值占比)。"""
    if day_panel is None or len(day_panel) == 0 or margin_amt is None or not margin_amt or margin_amt != margin_amt:
        return None, None, None
    closes = closes or {}
    last_px = last_px or {}
    next_px = next_px or {}
    lots = pd.to_numeric(day_panel["m_today"], errors="coerce").fillna(0).to_numpy(float)
    codes = day_panel["code"].astype(str).to_numpy()
    px = np.full(len(lots), np.nan)
    est = np.full(len(lots), np.nan)                 # 無價檔的估計價 (只用來判 partial / coverage)
    filled = np.zeros(len(lots), dtype=bool)
    for i, c in enumerate(codes):
        if lots[i] <= 0:
            continue
        cl = _px_of(closes.get(c))[0] if closes.get(c) is not None else None
        if cl is not None and cl == cl and cl > 0:
            px[i] = float(cl)
            continue
        lp = last_px.get(c)
        if lp is not None:
            t = _px_of(lp)
            lcl, age = t[0], (t[1] if len(t) > 1 else 0)
            if lcl is not None and lcl == lcl and lcl > 0 and (age is None or age <= FFILL_MAX_AGE):
                px[i] = float(lcl)
                filled[i] = True
                continue
        npx = next_px.get(c)
        if npx is not None:
            t = _px_of(npx)
            e = _est_px(lots[i], t[0], t[1] if len(t) > 1 else None)
            if e is not None:
                est[i] = e
                continue
        if lp is not None:
            t = _px_of(lp)
            e = _est_px(lots[i], t[0], t[2] if len(t) > 2 else None)
            if e is not None:
                est[i] = e
    has = np.isfinite(px) & (lots > 0)
    tot_lots = float(lots.sum())
    if tot_lots <= 0 or not has.any():
        return None, None, None
    denom = float(margin_amt) * 1e8
    contrib = np.where(has, lots * np.nan_to_num(px) * 1000.0, 0.0) / denom * 100
    value = float(contrib.sum())
    missing = (lots > 0) & ~has
    missing_share = float(lots[missing].sum() / tot_lots * 100)
    est_ok = missing & np.isfinite(est)
    est_contrib = np.where(est_ok, lots * np.nan_to_num(est) * 1000.0, 0.0) / denom * 100
    est_share = float(est_contrib.sum() / (value + est_contrib.sum()) * 100) if (value + est_contrib.sum()) > 0 else 0.0
    unest_lots = float(lots[missing & ~np.isfinite(est)].sum())
    coverage = round(100 - max(missing_share, est_share), 2)
    src = "twse"
    if (missing_share > 1.0 or est_share > 1.0 or (filled.any() and float(contrib[filled].max()) > 0.5)
            or (est_ok.any() and float(est_contrib.max()) > 0.5) or unest_lots >= UNEST_LOTS_PARTIAL):
        src = "partial"
    return round(value, 2), coverage, src


def _last_closes(before: str, limit_days: int = LAST_PX_DAYS) -> dict[str, tuple[float, int, float | None]]:
    """各檔在 before 之前最後一筆收盤 (日曆天 ≤ limit_days) → {code: (close, age_trading_days 約, 當時張數)}；age > FFILL_MAX_AGE 的只供估市值。"""
    since = (dt.date.fromisoformat(before) - dt.timedelta(days=limit_days)).isoformat()
    with store.conn() as c:
        _ensure(c)
        rows = c.execute("SELECT code, close, m_today, MAX(date) FROM margin_stock_daily WHERE date<? AND date>=? AND close IS NOT NULL GROUP BY code",
                         (before, since)).fetchall()
    out = {}
    b = dt.date.fromisoformat(before)
    for code, cl, lt, d in rows:
        age = (b - dt.date.fromisoformat(d)).days * 5 // 7
        out[code] = (cl, age, lt)
    return out


def _next_closes(after: str, limit_days: int = LAST_PX_DAYS) -> dict[str, tuple[float, float | None]]:
    """各檔在 after 之後第一筆收盤 (日曆天 ≤ limit_days) → {code: (close, 當時張數)}；只供無價檔估市值 (停牌 → 復牌 / 冷啟動由新往舊補時用)。"""
    until = (dt.date.fromisoformat(after) + dt.timedelta(days=limit_days)).isoformat()
    with store.conn() as c:
        _ensure(c)
        rows = c.execute("SELECT code, close, m_today, MIN(date) FROM margin_stock_daily WHERE date>? AND date<=? AND close IS NOT NULL GROUP BY code",
                         (after, until)).fetchall()
    return {code: (cl, lt) for code, cl, lt, _d in rows}


def _maint_metrics(v, cov, src) -> dict:
    return {"maint_twse": v, "maint_cov": cov if src == "twse" else -abs(cov or 0)}      # 負值 = partial (daily_metrics 只存數值)


def _day_metrics(date: str, rows: list[dict], closes: dict | None, mkt: dict | None, margin_amt: float | None = None, *, maint: bool = True) -> dict:
    """當日面板 → 擴散 / 互抵 / ETF 占比 (+ maint=True 且有收盤、知金額時的自算維持率) + 全市場流量 → daily_metrics。
    回補由新往舊補收盤時 (run_closes) 傳 maint=False：更早日子尚無收盤、ffill 無效，維持率交給 build 的 _recompute_maint 升冪統一算。"""
    df = pd.DataFrame(rows)
    cs = cross_section(df, closes)
    metrics = {METRIC_KEYS[k]: float(v) for k, v in cs.items() if v is not None and k in METRIC_KEYS}
    amt = margin_amt if margin_amt is not None else (mkt or {}).get("margin_amt")
    if maint and closes and amt:
        try:
            v, cov, src = maint_direct(df, closes, amt, _last_closes(date), _next_closes(date))
            if v is not None:
                metrics.update(_maint_metrics(v, cov, src))
        except Exception as e:  # noqa: BLE001
            log.debug("maint %s: %s", date, e)
    metrics.update(_flow_metrics(mkt))
    if metrics:
        store.upsert_metrics(date, metrics)
    return metrics


def _norm_tpex(res) -> tuple[dict | None, str]:
    """tpex.margin_balance → ({margin_lots, margin_amt(億), short_lots}, status ok|closed)。"""
    if res is None:
        return None, "closed"
    m = res.get("market") if isinstance(res, dict) and "market" in res else res
    if not isinstance(m, dict):
        return None, "closed"
    status = str(res.get("status", "ok")) if isinstance(res, dict) else "ok"
    lots = m.get("margin_lots", m.get("tpex_margin_lots"))
    amt = m.get("margin_amt", m.get("tpex_margin_amt"))
    sh = m.get("short_lots", m.get("tpex_short_lots"))
    if amt is not None and amt > 1e6:                       # 仟元 → 億
        amt = amt * 1e3 / 1e8
    if lots is None and amt is None:
        return None, status if status != "ok" else "closed"
    return {"margin_lots": lots, "margin_amt": amt, "short_lots": sh}, "ok"


def _tpex_have(date: str) -> bool:
    """daily_metrics 已有該日上櫃合計 (原始快取入庫後即刪，不能再靠快取判斷「已抓過」)。"""
    try:
        with store.conn() as c:
            return c.execute("SELECT 1 FROM daily_metrics WHERE date=? AND key=? LIMIT 1", (date, METRIC_KEYS["tpex_margin_lots"])).fetchone() is not None
    except Exception:  # noqa: BLE001
        return False


def _tpex_day(date: str) -> str:
    """回 status：ok / notyet / closed / fail。已在 daily_metrics 的日子直接回 ok (不重抓)。"""
    if _tpex_have(date):
        return "ok"
    try:
        res = _fetch_tpex(_ymd(date))
    except Exception as e:  # noqa: BLE001
        if _is(e, "NotYet"):
            return "notyet"
        if _is(e, "NoData"):
            _mark_nodata(date, "tpex")
            return "closed"
        log.debug("tpex %s: %s", date, e)
        return "fail"
    m, status = _norm_tpex(res)
    if m is None or status != "ok":
        if status == "closed":
            _mark_nodata(date, "tpex")
        return status if status in ("notyet", "closed") else "closed"
    store.upsert_metrics(date, {METRIC_KEYS["tpex_" + k]: float(v) for k, v in m.items() if v is not None})
    _drop_raw_cache(date, tpex=True, twse=False)
    return "ok"


def _drop_raw_cache(date: str, *, twse: bool = True, tpex: bool = False) -> None:
    """個股列已入 SQLite → 刪整份原始快取 (TWSE ALL ≈420 KB/日、TPEx ≈343 KB/日)；MS / close_all 很小保留。
    否則 Actions cache (key 含 run_id、每次 fast 都整包上傳) 回補 450 日 ≈ +360 MB、補到 2004 ≈ +3.7 GB。"""
    ymd = _ymd(date)
    try:
        if twse:
            cache_delete(RAW_CACHE_KEYS[0].format(ymd=ymd))
        if tpex:
            cache_delete(RAW_CACHE_KEYS[1].format(ymd=ymd))
    except Exception as e:  # noqa: BLE001
        log.debug("drop raw cache %s: %s", date, e)


# ------------------------------------------------------------------ 回補 (每日 ≤3 請求；先補最近缺日再往回；NoData 寫 margin_nodata；可重入)
def backfill(max_days: int = 40, direction: str = "forward_then_back", *, today: str | None = None, panel_from: str | None = None,
             max_seconds: float | None = None) -> dict:
    """缺日回補。預算：max_days (請求日數) 與 max_seconds (秒；到時停止、剩下的回 pending 由下次續補，rep.timeout=True)。
    順序：最近缺日 (由新往舊) → 種子日補收盤 → 更早缺日 (由新往舊) → 上櫃合計補齊；面板為空 (冷啟動) 時全部缺日視為 back，第一個抓的是最新日。
    TWSE ALL NoData 於資料日 ≥ today − NODATA_GRACE_DAYS → 尚未公布 (不寫 margin_nodata)；更早 → 寫 margin_nodata。"""
    t0 = time.time()
    rep = {"done": 0, "pending": 0, "pending_closes": 0, "from": None, "to": None, "seconds": 0.0, "nodata": 0, "closes": 0, "tpex": 0, "failed": 0}
    today = today or dt.datetime.now(config.TZ).date().isoformat()
    panel_from = panel_from or PANEL_FROM
    try:
        cal = trading_days(panel_from, today)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"calendar: {e}"
        return rep
    have = set(_panel_dates())
    nod = _nodata("twse_all")
    missing = [d for d in cal if d not in have and d not in nod]
    last_have = max(have) if have else None
    fwd = sorted(d for d in missing if last_have is not None and d > last_have)
    back = sorted((d for d in missing if last_have is None or d < last_have), reverse=True)     # 冷啟動：全部缺日由新往舊
    grace = (dt.date.fromisoformat(today) - dt.timedelta(days=NODATA_GRACE_DAYS)).isoformat()
    nod_close = _nodata("twse_close")
    need_close = [d for d in _panel_dates_without_close() if d >= MAINT_FROM and d not in nod_close]
    budget = int(max_days or 0)
    done = set()

    def out_of_time() -> bool:
        if max_seconds is not None and time.time() - t0 >= float(max_seconds):
            rep["timeout"] = True
            return True
        return False

    def run_days(days: list[str]) -> bool:
        """缺日：ALL + MI_INDEX + TPEx (每日 ≤3 請求)。回 False = 暫時失敗中斷 (下次重入)。"""
        nonlocal budget
        for d in days:
            if budget <= 0 or out_of_time():
                return True
            try:
                res = _fetch_margin_all(_ymd(d))
            except Exception as e:  # noqa: BLE001
                budget -= 1
                if _is(e, "NoData"):
                    if d >= grace:                 # 當日 / 前一日 (TWSE 約 21:00 才公布、cron 可能延遲跨午夜) 的 NoData = 尚未公布，不寫 margin_nodata，下次再試 (仍計 pending)
                        rep["notyet"] = rep.get("notyet", 0) + 1; _sleep()
                        continue
                    _mark_nodata(d, "twse_all"); rep["nodata"] += 1; done.add(d); _sleep()
                    continue
                rep["failed"] += 1; rep["error"] = f"{d}: {e}"
                return False
            budget -= 1
            date, mkt, rows = _norm_day(res)
            date = date or d
            closes = None
            if date >= MAINT_FROM:
                _sleep()
                try:
                    closes = _norm_closes(_fetch_closes(_ymd(date)))
                    rep["closes"] += 1
                except Exception as e:  # noqa: BLE001
                    if _is(e, "NoData"):
                        _mark_nodata(date, "twse_close")
                    else:
                        log.warning("closes_all %s: %s", date, e)
            _panel_upsert(date, rows, closes)
            _drop_raw_cache(date)
            try:
                _day_metrics(date, rows, closes, mkt)
            except Exception as e:  # noqa: BLE001
                log.debug("day metrics %s: %s", date, e)
            done.add(d); rep["done"] += 1
            rep["from"] = min(rep["from"] or date, date); rep["to"] = max(rep["to"] or date, date)
            if _tpex_day(date) == "ok":
                rep["tpex"] += 1
            _sleep()
        return True

    def run_closes() -> bool:
        """種子匯入的日子沒有收盤 → 補 MI_INDEX (最近的先) 與互抵占比；維持率不在此算 (由新往舊補時更早日子尚無收盤、ffill 無效)，
        交給 build 的 _recompute_maint 升冪統一算 (coverage<100 的日子也會重算)。"""
        nonlocal budget
        for d in need_close:
            if budget <= 0 or out_of_time():
                return True
            budget -= 1
            try:
                closes = _norm_closes(_fetch_closes(_ymd(d)))
            except Exception as e:  # noqa: BLE001
                if _is(e, "NoData"):
                    _mark_nodata(d, "twse_close"); done.add(d); _sleep()
                    continue
                rep["failed"] += 1; rep["error"] = f"closes {d}: {e}"
                return False
            _panel_update_closes(d, closes)
            rep["closes"] += 1; done.add(d)
            try:
                pan = load_panel(since=d, cols=["date", "code", "m_today", "m_prev", "m_limit", "note", "offset", "vol_lots"])
                pan = pan[pan["date"] == d]
                _day_metrics(d, pan.to_dict("records"), closes, None, _hist_amt(d), maint=False)
            except Exception as e:  # noqa: BLE001
                log.debug("closes metrics %s: %s", d, e)
            _sleep()
        return True

    def run_tpex() -> None:
        """面板已有 (種子匯入) 但 daily_metrics 尚無上櫃合計的日子 → 補 TPEx (最近的先；NoData 寫 margin_nodata、NotYet 下次再試)。"""
        nonlocal budget
        key = METRIC_KEYS.get("tpex_margin_lots")
        try:
            got = store.load_metrics([key]) if key else pd.DataFrame()
            have_t = set(got.loc[got[key].notna(), "date"].astype(str).str[:10]) if not got.empty and key in got else set()
        except Exception:  # noqa: BLE001
            have_t = set()
        nod_t = _nodata("tpex")
        todo = [d for d in sorted(set(_panel_dates()), reverse=True) if d >= TPEX_FROM and d < today and d not in have_t and d not in nod_t]
        rep["pending_tpex"] = len(todo)
        for d in todo:
            if budget <= 0 or out_of_time():
                return
            budget -= 1
            st = _tpex_day(d)
            if st == "ok":
                rep["tpex"] += 1; rep["pending_tpex"] -= 1
            elif st == "closed":
                rep["pending_tpex"] -= 1
            elif st == "fail":
                rep["failed"] += 1; rep.setdefault("error", f"tpex {d}")
                return

    # 順序：最近缺日 (forward) → 種子日補收盤 (近 2 年，維持率 / 股價欄要用) → 更早的缺日 (back；冷啟動或 MARGIN_PANEL_FROM 往前拉時) → 上櫃合計補齊
    first, second = (fwd, back) if direction == "forward_then_back" else (back, fwd)
    ok = run_days(first)
    if ok:
        ok = run_closes()
    if ok:
        ok = run_days(second)
    if ok:
        run_tpex()
    rep["pending"] = len([d for d in missing if d not in done])
    rep["pending_closes"] = len([d for d in need_close if d not in done])
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def _hist_amt(date: str) -> float | None:
    try:
        m = store.load_metrics([METRIC_KEYS["margin_amt"]])
        if m.empty or METRIC_KEYS["margin_amt"] not in m:
            return None
        row = m[m["date"].astype(str).str[:10] == date]
        return float(row[METRIC_KEYS["margin_amt"]].iloc[0]) if len(row) else None
    except Exception:  # noqa: BLE001
        return None


def _recompute_maint(hist: pd.DataFrame, max_days: int = 600) -> int:
    """面板有收盤的日子中，daily_metrics 尚無 maint_twse 或 maint_cov ≠ 100 (partial / 有無價檔) 的 → 升冪重算 (用 hist 的 margin_amt)。
    升冪且面板整段載入一次：缺價檔的最後收盤 (≤ FFILL_MAX_AGE 補價、更舊只估市值) 與之後第一筆收盤皆由同一份面板 ffill / bfill 取得，
    冷啟動 (由新往舊回補) 補到更早收盤後自動修正；回寫入日數。只看最近 max_days 日。"""
    if hist is None or hist.empty or "margin_amt" not in hist:
        return 0
    amt = dict(zip(hist["date"].astype(str).str[:10], hist["margin_amt"]))
    full: set[str] = set()
    try:
        got = store.load_metrics(["maint_twse", "maint_cov"])
        if not got.empty and "maint_twse" in got:
            cov = pd.to_numeric(got["maint_cov"], errors="coerce") if "maint_cov" in got else pd.Series(np.nan, index=got.index)
            ok = got["maint_twse"].notna() & (cov == 100)
            full = set(got.loc[ok, "date"].astype(str).str[:10])
    except Exception:  # noqa: BLE001
        full = set()
    with store.conn() as c:
        _ensure(c)
        dates = [r[0] for r in c.execute("SELECT date FROM margin_stock_daily GROUP BY date HAVING SUM(close IS NOT NULL)>0 ORDER BY date DESC").fetchall()]
    todo = sorted(d for d in dates if d not in full and amt.get(d) is not None and amt.get(d) == amt.get(d))[-max_days:]
    if not todo:
        return 0
    since = (dt.date.fromisoformat(min(todo)) - dt.timedelta(days=LAST_PX_DAYS)).isoformat()
    pan = load_panel(since=since, cols=["date", "code", "m_today", "close"])
    if pan.empty:
        return 0
    pan["code"] = pan["code"].astype(str)
    pan["close"] = pd.to_numeric(pan["close"], errors="coerce")
    pan["m_today"] = pd.to_numeric(pan["m_today"], errors="coerce")
    pan = pan.drop_duplicates(["date", "code"], keep="last")
    close = pan.pivot(index="date", columns="code", values="close").sort_index()          # pivot (非 pivot_table)：全期無收盤的檔也要保留為欄 (其張數算缺價)
    lots = pan.pivot(index="date", columns="code", values="m_today").reindex_like(close)
    dates_all = list(close.index)
    pos = pd.Series(np.arange(len(dates_all)), index=close.index)
    has = close.notna()
    idx_mat = pd.DataFrame(np.where(has.to_numpy(), pos.to_numpy()[:, None], np.nan), index=close.index, columns=close.columns)
    last_idx = idx_mat.ffill().shift(1)                      # 之前最後一筆有收盤的列位置
    next_idx = idx_mat.bfill().shift(-1)                     # 之後第一筆
    close_np, lots_np = close.to_numpy(float), lots.to_numpy(float)
    codes = np.array(close.columns, dtype=str)
    n = 0
    for d in todo:
        if d not in pos.index:
            continue
        i = int(pos[d])
        li, ni = last_idx.iloc[i].to_numpy(float), next_idx.iloc[i].to_numpy(float)
        day_lots = lots_np[i]
        row = np.isfinite(day_lots) & (day_lots > 0)
        closes = {c: (close_np[i, j], None) for j, c in enumerate(codes) if row[j] and np.isfinite(close_np[i, j])}
        last_px, next_px = {}, {}
        for j in np.flatnonzero(row & ~np.isfinite(close_np[i])):
            if np.isfinite(li[j]):
                k = int(li[j]); last_px[codes[j]] = (close_np[k, j], i - k, lots_np[k, j])
            if np.isfinite(ni[j]):
                k = int(ni[j]); next_px[codes[j]] = (close_np[k, j], lots_np[k, j])
        day = pd.DataFrame({"code": codes[row], "m_today": day_lots[row]})
        try:
            v, cov, src = maint_direct(day, closes, amt[d], last_px, next_px)
        except Exception as e:  # noqa: BLE001
            log.debug("recompute maint %s: %s", d, e)
            continue
        if v is not None:
            store.upsert_metrics(d, _maint_metrics(v, cov, src))
            n += 1
    return n


# ------------------------------------------------------------------ 全市場序列 (Tier-1：fresh ∪ SQLite ∪ prev ∪ 種子，去重 keep-first)
def _to_yi(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mx = float(s.abs().max()) if s.notna().any() else 0.0
    if mx > 1e9:          # 元
        return s / 1e8
    if mx > 1e6:          # 仟元
        return s * 1e3 / 1e8
    return s              # 已是億


def _norm_total(df) -> pd.DataFrame:
    """FinMind TaiwanStockTotalMarginPurchaseShortSale (原始列或已整理) → date, margin_lots, short_lots, margin_amt(億), buy/sell/repay 張與金額, short_*。"""
    if df is None:
        return pd.DataFrame()
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame) or df.empty or "date" not in df:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = d["date"].astype(str).str[:10]
    if "name" in d and "TodayBalance" in d:
        out = pd.DataFrame({"date": sorted(d["date"].unique())}).set_index("date")
        spec = {"MarginPurchase": ("margin_lots", "buy_lots", "sell_lots", "repay_lots", 1.0),
                "ShortSale": ("short_lots", "short_buy", "short_sell", "short_repay", 1.0),
                "MarginPurchaseMoney": ("margin_amt", "buy_amt", "sell_amt", "repay_amt", 1e-8)}
        for nm, (bal, b, s, r, k) in spec.items():
            g = d[d["name"] == nm].drop_duplicates("date", keep="last").set_index("date")
            if g.empty:
                continue
            out[bal] = pd.to_numeric(g["TodayBalance"], errors="coerce") * k
            for src, dst in (("buy", b), ("sell", s), ("Return", r)):
                if src in g:
                    out[dst] = pd.to_numeric(g[src], errors="coerce") * k
        return out.reset_index()
    ren = {"buy": "buy_lots", "sell": "sell_lots", "repay": "repay_lots", "Return": "repay_lots", "margin_buy_lots": "buy_lots", "margin_sell_lots": "sell_lots",
           "margin_repay_lots": "repay_lots", "margin_buy_amt": "buy_amt", "margin_sell_amt": "sell_amt", "margin_repay_amt": "repay_amt",
           "margin_buy": "buy_lots", "margin_sell": "sell_lots", "margin_repay": "repay_lots", "short_buy_lots": "short_buy", "short_sell_lots": "short_sell",
           "short_repay_lots": "short_repay", "MarginPurchaseTodayBalance": "margin_lots", "ShortSaleTodayBalance": "short_lots",
           "MarginPurchaseMoneyTodayBalance": "margin_amt"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d and v not in d})
    keep = ["date"] + [c for c in ("margin_lots", "short_lots", "margin_amt", "buy_lots", "sell_lots", "repay_lots", "buy_amt", "sell_amt", "repay_amt",
                                   "short_buy", "short_sell", "short_repay") if c in d]
    d = d[keep].drop_duplicates("date", keep="last")
    for c in ("margin_amt", "buy_amt", "sell_amt", "repay_amt"):
        if c in d:
            d[c] = _to_yi(d[c])
    return d


def _table_df(tbl) -> pd.DataFrame:
    """{cols, rows} 或 list[dict] → DataFrame；壞格式 → 空。"""
    try:
        if isinstance(tbl, dict) and "cols" in tbl and "rows" in tbl:
            df = pd.DataFrame(tbl["rows"], columns=tbl["cols"])
        elif isinstance(tbl, list):
            df = pd.DataFrame(tbl)
        else:
            return pd.DataFrame()
        if "date" not in df:
            return pd.DataFrame()
        df["date"] = df["date"].astype(str).str[:10]
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _fix_amt_anomaly(df: pd.DataFrame, alt: pd.Series | None) -> int:
    """FinMind 金額側單日 >10% 跳動而張數未跟 (2022-01-03 / 2024-06-19) → 以 TWSE 值 (alt) 覆寫，無則內插。"""
    if "margin_amt" not in df or len(df) < 3:
        return 0
    a = pd.to_numeric(df["margin_amt"], errors="coerce")
    pc = a.pct_change().abs()
    lots_pc = pd.to_numeric(df["margin_lots"], errors="coerce").pct_change().abs() if "margin_lots" in df else pd.Series(0.0, index=df.index)
    bad = (pc > 0.10) & (lots_pc.fillna(0) < 0.03)
    bad &= (a.shift(-1) / a - 1).abs().fillna(1) > 0.08         # 隔日回彈 (單日異常)
    if not bad.any():
        return 0
    fixed = a.copy()
    for i in np.flatnonzero(bad.to_numpy()):
        v = None
        if alt is not None:
            v = alt.get(df["date"].iloc[i])
            if v is not None and (v != v or abs(v / a.iloc[i - 1] - 1) > 0.10):
                v = None
        fixed.iloc[i] = v if v is not None else np.nan
    fixed = fixed.interpolate(limit=2)
    df["margin_amt"] = fixed
    df.loc[bad, "amt_fixed"] = True
    return int(bad.sum())


def market_history(prev: dict | None = None) -> pd.DataFrame:
    """date, margin_amt (億), margin_lots, short_lots, 流量, maint_twse/real, tpex_*, 擴散, close/amount/volume_lots。
    來源優先序：FinMind total (餘額 + flows) → TWSE MS 當日 → SQLite daily_metrics → prev.history / flows → 種子；日期去重 keep-first (逐欄取第一個非空)。"""
    prev = prev or {}
    parts: list[pd.DataFrame] = []
    alt_amt = []
    fm = pd.DataFrame()
    try:
        fm = _norm_total(_fetch_total_flows())
    except Exception as e:  # noqa: BLE001
        log.info("finmind total_margin_flows 不可用 (%s)，改用 total_margin", e)
        try:
            fm = _norm_total(_fetch_total_margin())
        except Exception as e2:  # noqa: BLE001
            log.warning("FinMind 融資序列失敗: %s", e2)
    if not fm.empty:
        parts.append(fm)
    try:
        ms = _norm_market(_fetch_twse_ms())
        if ms and ms.get("margin_amt"):
            raw = _fetch_twse_ms()
            date = _iso((raw or {}).get("date")) if isinstance(raw, dict) else None
            if date:
                row = {k: v for k, v in ms.items() if k in ("margin_lots", "short_lots", "margin_amt", "buy_lots", "sell_lots", "repay_lots", "buy_amt", "sell_amt",
                                                               "repay_amt", "short_buy", "short_sell", "short_repay")}
                row["date"] = date
                tw = pd.DataFrame([row])
                parts.insert(0, tw)                      # TWSE 當日最優先 (FinMind 落後 1 日；金額異常以 TWSE 為準)
                alt_amt.append(tw[["date", "margin_amt"]])
    except Exception as e:  # noqa: BLE001
        log.debug("TWSE MS: %s", e)
    try:
        px = _fetch_taiex()
        if isinstance(px, pd.DataFrame) and not px.empty:
            p = px[["date", "close"] + [c for c in ("amount", "volume") if c in px]].copy()
            p["date"] = p["date"].astype(str).str[:10]
            if "amount" in p:
                p.loc[pd.to_numeric(p["amount"], errors="coerce") <= 0, "amount"] = np.nan
            if "volume" in p:
                p["volume_lots"] = pd.to_numeric(p["volume"], errors="coerce") / 1000
                p = p.drop(columns=["volume"])
            parts.append(p)
    except Exception as e:  # noqa: BLE001
        log.warning("FinMind TAIEX 失敗: %s", e)
    try:
        inv = {v: k for k, v in METRIC_KEYS.items()}
        st = store.load_metrics(list(METRIC_KEYS.values()))
        if not st.empty:
            st = st.rename(columns=inv)
            st["date"] = st["date"].astype(str).str[:10]
            if "margin_amt" in st:
                alt_amt.append(st[["date", "margin_amt"]])
            parts.append(st)
    except Exception as e:  # noqa: BLE001
        log.debug("store margin metrics: %s", e)
    for key in ("history", "flows"):
        t = _table_df(prev.get(key))
        if not t.empty:
            if key == "history":
                t = t.rename(columns={"maint": "maint_twse"})
                if "margin_amt" in t:
                    alt_amt.append(t[["date", "margin_amt"]])
            parts.append(t)
    try:
        if SEED_PATH.exists():
            sj = json.loads(SEED_PATH.read_text(encoding="utf-8"))
            for key in ("history", "flows"):
                t = _table_df(sj.get(key))
                if not t.empty:
                    parts.append(t)
    except Exception as e:  # noqa: BLE001
        log.debug("seed json: %s", e)
    if not parts:
        return pd.DataFrame(columns=["date", "margin_amt", "margin_lots", "short_lots"])
    allc = sorted({c for p in parts for c in p.columns if c != "date"})
    frames = []
    for p in parts:
        q = p.copy()
        q["date"] = q["date"].astype(str).str[:10]
        for c in allc:
            if c not in q:
                q[c] = np.nan
            q[c] = pd.to_numeric(q[c], errors="coerce")
        frames.append(q[["date"] + allc])
    df = pd.concat(frames, ignore_index=True)
    df = df.groupby("date", sort=True).first().reset_index()
    df = df[df["date"] >= HISTORY_FROM]
    alt = None
    if alt_amt:
        a = pd.concat(alt_amt, ignore_index=True).dropna()
        a["date"] = a["date"].astype(str).str[:10]
        alt = a.drop_duplicates("date", keep="first").set_index("date")["margin_amt"]
    df["amt_fixed"] = False
    _fix_amt_anomaly(df, alt)
    df = df.dropna(subset=["margin_amt"]).reset_index(drop=True)
    try:   # Tier-1 持久化到 daily_metrics (餘額 / 大盤收盤 / 流量 2001~)，離線時可重建 (FinMind 失敗時 turnover_hot 的 p95 門檻與統計才不會隨來源可用性漂移)
        cols = [c for c in ("margin_amt", "margin_lots", "short_lots", "close", "amount", "buy_lots", "sell_lots", "repay_lots", "buy_amt", "sell_amt",
                            "repay_amt", "short_buy", "short_sell", "short_repay") if c in df]
        if cols and not fm.empty:
            sub = df[["date"] + cols].rename(columns={c: METRIC_KEYS[c] for c in cols})
            store.upsert_frame(sub, [METRIC_KEYS[c] for c in cols])
    except Exception as e:  # noqa: BLE001
        log.debug("persist tier1: %s", e)
    return df


# ------------------------------------------------------------------ 衍生 (§2.1)
def _pct_excl(s: pd.Series, w: int, minp: int) -> pd.Series:
    """今日值在過去 w 日 (不含今日) 的百分位 0~100。"""
    r = s.rolling(w + 1, min_periods=minp + 1).rank(pct=True)
    cnt = s.rolling(w + 1, min_periods=minp + 1).count()
    return ((r * cnt - 1) / (cnt - 1).replace(0, np.nan) * 100)


def _run_length(state: pd.Series) -> pd.Series:
    out = np.zeros(len(state), dtype=int)
    prev, run = None, 0
    for i, v in enumerate(state.to_numpy()):
        run = run + 1 if v == prev else 1
        out[i] = run
        prev = v
    return pd.Series(out, index=state.index)


def _quad(mp, r20) -> np.ndarray:
    mp, r20 = np.asarray(mp, float), np.asarray(r20, float)
    q = np.select([(mp > 1) & (r20 > 1), (mp > 1) & (r20 < -1), (mp < -1) & (r20 < -1), (mp < -1) & (r20 > 1)],
                  [QUADS[0], QUADS[1], QUADS[2], QUADS[3]], QUADS[4]).astype(object)
    q[~(np.isfinite(mp) & np.isfinite(r20))] = ""
    return q


def _derive(hist: pd.DataFrame, scored: pd.DataFrame | None) -> pd.DataFrame:
    h = hist.dropna(subset=["margin_amt"]).sort_values("date").reset_index(drop=True).copy()
    h["date"] = h["date"].astype(str).str[:10]
    for c in ("close", "amount", "volume_lots", "margin_lots", "short_lots", "buy_lots", "sell_lots", "repay_lots", "maint_twse", "maint_real", "maint_cov",
              "tpex_margin_amt", "tpex_margin_lots", "tpex_short_lots", "br_net", "usage_gt60_pct"):
        if c not in h:
            h[c] = np.nan
        h[c] = pd.to_numeric(h[c], errors="coerce")
    if scored is not None and not scored.empty and "date" in scored:
        sc = scored.copy()
        sc["date"] = sc["date"].astype(str).str[:10]
        cols = [c for c in ("close", "amount", "margin_div20", "f_margin", "maint_ratio", "margin_pct20", "short_ratio") if c in sc]
        sc = sc[["date"] + cols].drop_duplicates("date", keep="last").rename(columns={"close": "sc_close", "amount": "sc_amount", "short_ratio": "sc_short_ratio"})
        h = h.merge(sc, on="date", how="left")
        for c in ("close", "amount"):
            if f"sc_{c}" in h:
                h[c] = h[f"sc_{c}"].where(h[f"sc_{c}"].notna(), h[c])
    c = h["close"].astype(float)
    a = h["margin_amt"].astype(float)
    h["ret1"], h["ret5"], h["ret20"] = c.pct_change() * 100, c.pct_change(5) * 100, c.pct_change(20) * 100
    h["bias20"] = (c / c.rolling(20).mean() - 1) * 100
    h["rv20"] = h["ret1"].rolling(20).std()
    for k in (5, 10, 20):
        h[f"fwd{k}"] = (c.shift(-(k + 1)) / c.shift(-1) - 1) * 100          # D+1 收盤起算
    h["fwd10_d0"] = (c.shift(-10) / c - 1) * 100                             # 研究欄 (D 收盤起算)
    for k in (1, 5, 20, 60):
        h[f"chg{k}"] = a.diff(k)
        h[f"pct{k}"] = a.pct_change(k) * 100
    h["acc5"] = h["pct5"] - h["pct5"].shift(5)
    h["streak"] = streak(h["chg1"])
    h["pct250"], h["pct1250"] = _pct_excl(a, 250, 120), _pct_excl(a, 1250, 500)
    hi, lo = a.rolling(250, min_periods=120).max(), a.rolling(250, min_periods=120).min()
    h["dhi250"], h["dlo250"] = (a / hi - 1) * 100, (a / lo - 1) * 100
    h["hi250"], h["lo250"] = (a >= hi) & hi.notna(), (a <= lo) & lo.notna()
    h["z1"], h["z5"], h["z20"] = zscore(h["chg1"], 60), zscore(h["chg5"], 60), zscore(h["chg20"], 120)
    h["amount_ma20"] = h["amount"].rolling(20, min_periods=10).mean()
    h["lev_turn"] = a / h["amount_ma20"]
    h["lev_turn_pct1250"] = _pct_excl(h["lev_turn"], 1250, 500)
    mp = h["margin_pct20"].where(h["margin_pct20"].notna(), h["pct20"]) if "margin_pct20" in h else h["pct20"]
    h["quad"] = _quad(mp, h["ret20"])
    h["quad_days"] = _run_length(h["quad"])
    h["quad_switch60"] = (h["quad"] != h["quad"].shift(1)).astype(int).rolling(60, min_periods=1).sum()
    div = h["pct20"] - h["ret20"]
    h["div20"] = h["margin_div20"].where(h["margin_div20"].notna(), div) if "margin_div20" in h else div
    h["div20_z250"] = zscore(h["div20"], 250)
    h["corr60"] = h["pct1"].rolling(60).corr(h["ret1"])
    h["beta60"] = h["pct1"].rolling(60).cov(h["ret1"]) / h["ret1"].rolling(60).var()
    sign = np.sign(h["pct20"]).fillna(0)
    run = _run_length(sign)
    h["regime"] = np.where((run >= 20) & (sign > 0), "exp", np.where((run >= 20) & (sign < 0), "con", ""))
    h["regime_days"] = np.where(h["regime"] != "", run, 0)
    prev_lots = h["margin_lots"].shift(1)
    h["bs_ratio"] = h["buy_lots"] / h["sell_lots"].replace(0, np.nan)
    h["turnover_ratio"] = (h["buy_lots"] + h["sell_lots"]) / prev_lots * 100
    h["new_rate"] = h["buy_lots"] / prev_lots * 100
    h["repay_share"] = h["repay_lots"] / (h["sell_lots"] + h["repay_lots"]).replace(0, np.nan) * 100
    h["buy_vol_share"] = h["buy_lots"] / h["volume_lots"].replace(0, np.nan) * 100
    for col in ("bs_ratio", "turnover_ratio", "br_net"):
        h[f"{col}_z60"] = zscore(h[col], 60)
    s = h["short_lots"].astype(float)
    h["s_pct5"], h["s_pct20"] = s.pct_change(5) * 100, s.pct_change(20) * 100
    h["s_streak"] = streak(s.diff())
    sr = s / h["margin_lots"].replace(0, np.nan) * 100
    h["short_ratio"] = h["sc_short_ratio"].where(h["sc_short_ratio"].notna(), sr) if "sc_short_ratio" in h else sr
    h["sr_pct250"] = _pct_excl(h["short_ratio"], 250, 120)
    h["sr_chg5"] = h["short_ratio"].diff(5)
    t = h["tpex_margin_amt"].astype(float)
    h["tpex_pct5"], h["tpex_pct20"] = t.pct_change(5) * 100, t.pct_change(20) * 100
    h["both_amt"] = a + t
    m = h["maint_twse"].astype(float)
    h["maint"] = m
    h["maint_chg5"] = m.diff(5)
    # 「2004 年來」百分位只在覆蓋期起點 ≤ MAINT_FULL_FROM 時輸出 (否則 120 日就會輸出名為 2004 的百分位)；pct_hist 為可用期間的 expanding 百分位 (附 maint_from)
    h["maint_pct_hist"] = m.expanding(min_periods=120).rank(pct=True) * 100
    maint_from = str(h.loc[m.notna(), "date"].iloc[0]) if m.notna().any() else None
    h["maint_pct2004"] = h["maint_pct_hist"] if (maint_from is not None and maint_from <= MAINT_FULL_FROM) else np.nan
    h.attrs["maint_from"] = maint_from
    h["lt150_days250"] = (m < 150).astype(float).where(m.notna()).rolling(250, min_periods=1).sum()
    return h


def _behav60(h: pd.DataFrame) -> dict:
    t = h.tail(60)
    g, r = t["chg1"].astype(float), t["ret1"]
    down, up = t[r < 0], t[r > 0]
    return {"inc_days_pct": _r((g > 0).mean() * 100, 0), "avg_inc": _r(g[g > 0].mean(), 1) if (g > 0).any() else None,
            "avg_dec": _r(g[g < 0].mean(), 1) if (g < 0).any() else None,
            "inc_on_down_rate": _r((down["chg1"] > 0).mean() * 100, 0) if len(down) else None,
            "dec_on_up_rate": _r((up["chg1"] < 0).mean() * 100, 0) if len(up) else None,
            "down_days": int(len(down)), "up_days": int(len(up))}


def _weekly(h: pd.DataFrame) -> list[dict]:
    out = []
    try:
        wk = h.copy()
        wk["wk"] = pd.to_datetime(wk["date"]).dt.to_period("W").astype(str)
        for _k, grp in list(wk.groupby("wk"))[-12:]:
            cl = grp["close"].astype(float).dropna()
            out.append({"week": str(grp["date"].iloc[0])[:10], "chg": _r(grp["chg1"].sum(), 0), "days": int(len(grp)),
                        "ret": _r((cl.iloc[-1] / cl.iloc[0] - 1) * 100, 2) if len(cl) >= 2 else None})
    except Exception as e:  # noqa: BLE001
        log.debug("weekly: %s", e)
    return out


def market_view(hist: pd.DataFrame, scored: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    if hist is None or hist.empty or "margin_amt" not in hist:
        return {}, pd.DataFrame()
    h = _derive(hist, scored)
    if h.empty:
        return {}, h
    L = h.iloc[-1]
    quad = L["quad"] or QUADS[4]
    qd = int(L["quad_days"]) if quad else 0
    st = int(L["streak"])
    reg_id = L["regime"]
    base_rate = _r((h["regime"] != "").mean() * 100, 0)
    reg = {"id": {"exp": "expansion", "con": "contraction", "": "none"}[reg_id], "text": {"exp": "融資擴張期", "con": "融資收縮期", "": "無連續 20 日同向"}[reg_id],
           "since": str(h["date"].iloc[len(h) - int(L["regime_days"])]) if reg_id else None, "days": int(L["regime_days"]), "base_rate": base_rate, "label": "descriptive"}
    if reg_id:
        reg["text"] = f"{reg['text']}自 {reg['since']} 起第 {reg['days']} 日 (歷史 {base_rate:.0f}% 的日子處於擴張或收縮期)"
    acc = _r(L["acc5"], 1)
    acc_txt = "持平" if acc is None or abs(acc) < 0.5 else ("加快" if acc > 0 else "放緩")
    maint_v = _r(L["maint"], 1)
    real_rows = h.dropna(subset=["maint_real"])
    real_v = _r(real_rows["maint_real"].iloc[-1], 1) if len(real_rows) else None
    real_date = str(real_rows["date"].iloc[-1]) if len(real_rows) else None
    mt_rows = h.dropna(subset=["maint"])
    maint_date = str(mt_rows["date"].iloc[-1]) if len(mt_rows) else None
    cov = _r(L["maint_cov"], 1)
    maint_src = None if maint_v is None else ("partial" if (cov is not None and cov < 0) else "twse")
    check = bool(maint_v is not None and real_v is not None and real_date == maint_date and abs(maint_v - real_v) > 1.5)
    mv = maint_v if maint_v is not None else real_v
    mode_txt = (f"融資 20 日 {_f(L['pct20'])}% vs 指數 {_f(L['ret20'])}% → 『{quad}』第 {qd} 日" + (f"，維持率 {mv:.1f}%" if mv is not None else "")
                + "；以 D 日收盤後公布的資料")
    verdict = ("融資行為指標對大盤 1/5/10/20 日方向：153 項走動式五分位檢定 8 個 CI 不含 0 (5% 期望 7.7)、114 項事件檢定 2 個 (期望 5.7)，"
               "多重比較校正後 0 項存活；本區為行為描述")
    tp_rows = h.dropna(subset=["tpex_margin_amt"])
    tpL = tp_rows.iloc[-1] if len(tp_rows) else None
    view = {
        "date": str(L["date"]), "amt": _r(L["margin_amt"], 1), "lots": _r(L["margin_lots"], 0),       # 融券在下方 short{lots,...} (同 key 重複會被蓋掉)
        "chg1": _r(L["chg1"], 1), "chg5": _r(L["chg5"], 1), "chg20": _r(L["chg20"], 1), "chg60": _r(L["chg60"], 1),
        "pct5": _r(L["pct5"], 2), "pct20": _r(L["pct20"], 2), "pct60": _r(L["pct60"], 2), "acc5": acc, "acc_text": acc_txt, "streak": st,
        "pct250": _r(L["pct250"], 0), "pct1250": _r(L["pct1250"], 0), "dhi250": _r(L["dhi250"], 1), "dlo250": _r(L["dlo250"], 1),
        "z1": _r(L["z1"]), "z5": _r(L["z5"]), "z20": _r(L["z20"]),
        "lev_turn": _r(L["lev_turn"]), "lev_turn_pct1250": _r(L["lev_turn_pct1250"], 0),
        "quad": {"id": QUADS.index(quad) if quad in QUADS else 4, "text": quad, "days": qd, "switch60": int(L["quad_switch60"]) if pd.notna(L["quad_switch60"]) else 0},
        "ret20": _r(L["ret20"], 2), "ret5": _r(L["ret5"], 2), "ret1": _r(L["ret1"], 2), "close": _r(L["close"], 2),
        "div20": _r(L["div20"], 1), "div20_z250": _r(L["div20_z250"]), "corr60": _r(L["corr60"]), "beta60": _r(L["beta60"]),
        "f_margin": _r(L["f_margin"], 1) if "f_margin" in h else None,
        "regime": reg,
        "flows": {"buy": _r(L["buy_lots"], 0), "sell": _r(L["sell_lots"], 0), "repay": _r(L["repay_lots"], 0), "bs_ratio": _r(L["bs_ratio"]),
                  "bs_z60": _r(L["bs_ratio_z60"]), "turnover_ratio": _r(L["turnover_ratio"]), "turnover_z60": _r(L["turnover_ratio_z60"]),
                  "new_rate": _r(L["new_rate"]), "repay_share": _r(L["repay_share"]), "buy_vol_share": _r(L["buy_vol_share"]),
                  "bs_ratio_5d": _r(h["bs_ratio"].tail(5).mean()), "turnover_5d": _r(h["turnover_ratio"].tail(5).mean()),
                  "date": str(h.dropna(subset=["buy_lots"])["date"].iloc[-1]) if h["buy_lots"].notna().any() else None},
        "cross": {k: _r(L[c], 2) if c in h else None for k, c in (("br_up", "br_up"), ("br_dn", "br_dn"), ("br_net", "br_net"), ("br_net_z60", "br_net_z60"),
                                                                 ("conc_top20", "conc_top20"), ("usage_med", "usage_med"), ("usage_gt60_pct", "usage_gt60_pct"),
                                                                 ("offset_vol_share", "offset_vol_share"), ("etf_lots_share", "etf_lots_share"))},
        "maint": {"value": maint_v, "src": maint_src, "date": maint_date, "real": real_v, "real_date": real_date, "coverage": abs(cov) if cov is not None else None,
                  "check": check, "d130": _r(mv - 130, 1) if mv is not None else None, "d150": _r(mv - 150, 1) if mv is not None else None,
                  "chg5": _r(L["maint_chg5"], 1), "pct2004": _r(L["maint_pct2004"], 0), "pct_hist": _r(L["maint_pct_hist"], 0),
                  "since": str(mt_rows["date"].iloc[0]) if len(mt_rows) else None, "days": int(len(mt_rows)),
                  "lt150_days250": _r(L["lt150_days250"], 0), "label": "risk_descriptive"},
        "short": {"lots": _r(L["short_lots"], 0), "pct5": _r(L["s_pct5"]), "pct20": _r(L["s_pct20"]), "streak": int(L["s_streak"]), "ratio": _r(L["short_ratio"]),
                  "ratio_pct250": _r(L["sr_pct250"], 0), "ratio_chg5": _r(L["sr_chg5"])},
        "tpex": {"date": str(tpL["date"]) if tpL is not None else None, "amt": _r(tpL["tpex_margin_amt"], 1) if tpL is not None else None,
                 "lots": _r(tpL["tpex_margin_lots"], 0) if tpL is not None else None, "pct5": _r(tpL["tpex_pct5"]) if tpL is not None else None,
                 "pct20": _r(tpL["tpex_pct20"]) if tpL is not None else None, "short_lots": _r(tpL["tpex_short_lots"], 0) if tpL is not None else None,
                 "both_amt": _r(tpL["both_amt"], 1) if tpL is not None else None, "status": "ok" if tpL is not None else "none"},
        "behav60": _behav60(h), "weekly": _weekly(h),
        "verdict": {"text": verdict, "label": "descriptive"},
        "mode": {"id": quad, "text": mode_txt, "label": "descriptive"},
        "max_inc": {"date": str(h.loc[h["chg1"].idxmax(), "date"]), "v": _r(h["chg1"].max(), 1)} if h["chg1"].notna().any() else None,
        "max_dec": {"date": str(h.loc[h["chg1"].idxmin(), "date"]), "v": _r(h["chg1"].min(), 1)} if h["chg1"].notna().any() else None,
        "history_days": int(len(h)), "history_from": str(h["date"].iloc[0]),
    }
    for k, c in (("usage_gt80_n", "usage_gt80_n"), ("n_valid", "n_valid")):
        view["cross"][k] = _r(L[c], 0) if c in h else None
    view["cross"]["label"] = "risk_descriptive"
    view["cross"]["date"] = str(h.dropna(subset=["br_up"])["date"].iloc[-1]) if "br_up" in h and h["br_up"].notna().any() else None
    return view, h


# ------------------------------------------------------------------ 事件 (§2.4：D+1 起算、k=10 配對 + ret20 卡尺、episode / 年 cluster bootstrap)
def _quantile_insufficient(h: pd.DataFrame) -> set[str]:
    """全歷史分位門檻的事件 (ext_inc / ext_dec 用 chg1、turnover_hot 用 turnover_ratio) 樣本 < MIN_QUANTILE_DAYS → 不輸出 (門檻會隨來源可用性漂移)。"""
    out = set()
    if int(h["chg1"].notna().sum()) < MIN_QUANTILE_DAYS:
        out |= {"ext_inc", "ext_dec"}
    if int(h["turnover_ratio"].notna().sum()) < MIN_QUANTILE_DAYS:
        out.add("turnover_hot")
    return out


def _event_masks(h: pd.DataFrame) -> dict[str, pd.Series]:
    chg1 = h["chg1"]
    hi, lo = chg1.quantile(0.95), chg1.quantile(0.05)
    tr = h["turnover_ratio"]
    tr95 = tr.quantile(0.95) if tr.notna().any() else np.nan
    m = h["maint"]
    no = pd.Series(False, index=h.index)
    bad = _quantile_insufficient(h)
    return {
        "streak_up7": h["streak"] >= 7,
        "streak_dn7": h["streak"] <= -7,
        "wipeout": (h["pct20"] < -4) & (h["ret20"] < -3),
        "chase": h["div20"] > 5,
        "knife": (h["pct20"] > 2) & (h["ret20"] < -2),
        "ext_inc": (chg1 >= hi) if "ext_inc" not in bad else no, "ext_dec": (chg1 <= lo) if "ext_dec" not in bad else no,
        "hi250": h["hi250"].astype(bool), "lo250": h["lo250"].astype(bool),
        "maint_lt150": m < 150,
        "turnover_hot": (tr >= tr95) if (tr95 == tr95 and "turnover_hot" not in bad) else no,
        "deleverage_fast": (h["pct5"] < -3) & (m.diff(5) <= -8),
    }


def _episodes(idx: np.ndarray, gap: int = EPISODE_GAP) -> np.ndarray:
    ep = np.zeros(len(idx), dtype=int)
    for i in range(1, len(idx)):
        ep[i] = ep[i - 1] + (1 if idx[i] - idx[i - 1] > gap else 0)
    return ep


def _cluster_ci(x: np.ndarray, groups: np.ndarray, rng: np.random.Generator, reps: int = BOOT_B) -> list | None:
    """以 groups (episode 或年) 為單位重抽樣的均值 95% CI (向量化)。"""
    if len(x) == 0:
        return None
    g, inv = np.unique(groups, return_inverse=True)
    ng = len(g)
    if ng < 2:
        return None
    gs = np.bincount(inv, weights=x, minlength=ng)
    gn = np.bincount(inv, minlength=ng).astype(float)
    pick = rng.integers(0, ng, size=(reps, ng))
    means = gs[pick].sum(1) / gn[pick].sum(1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def _controls(h: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """事件日 (有 fwd20 與配對特徵) 與其對照：同日 ret1/ret5/ret20/bias20/rv20 標準化最近鄰 k=MATCH_K、ret20 卡尺、年差 ≤YEAR_DIFF、排除任一事件日 ±3 日。
    回 (ev_idx, episode, ctrl_list (與 keep 對齊), keep = 找得到對照的事件位置)。"""
    n_all = len(h)
    Z = h[["ret1", "ret5", "ret20", "bias20", "rv20"]].astype(float)
    Z = (Z - Z.mean()) / Z.std().replace(0, np.nan)
    zarr = Z.to_numpy()
    ok = np.isfinite(h["fwd20"].to_numpy(float)) & np.isfinite(zarr).all(1)
    years = pd.to_datetime(h["date"]).dt.year.to_numpy()
    r20 = h["ret20"].to_numpy(float)
    ev_all = np.flatnonzero(mask)
    ev_idx = np.flatnonzero(mask & ok)
    ep = _episodes(ev_idx)
    near = np.zeros(n_all, dtype=bool)
    for off in range(-3, 4):
        sh = ev_all + off
        near[sh[(sh >= 0) & (sh < n_all)]] = True
    pool = np.flatnonzero(ok & ~near)
    ctrl: list[np.ndarray] = []
    keep = []
    if len(pool):
        for j, i in enumerate(ev_idx):
            cand = pool[(np.abs(years[pool] - years[i]) <= YEAR_DIFF) & (np.abs(r20[pool] - r20[i]) <= CALIPER_RET20)]
            if len(cand) == 0:
                continue
            dist = ((zarr[cand] - zarr[i]) ** 2).sum(1)
            ctrl.append(cand[np.argsort(dist, kind="stable")[:MATCH_K]])
            keep.append(j)
    return ev_idx, ep, ctrl, np.array(keep, dtype=int)


def _match_study(h: pd.DataFrame, mask: pd.Series, eid: str) -> dict:
    n_all = len(h)
    mask = mask.fillna(False).astype(bool).to_numpy()
    fwd = {k: h[f"fwd{k}"].to_numpy(float) for k in (5, 10, 20)}
    years = pd.to_datetime(h["date"]).dt.year.to_numpy()
    r20 = h["ret20"].to_numpy(float)
    ev_all = np.flatnonzero(mask)
    out = {"n": 0, "n_total": int(len(ev_all)), "episodes": 0, "last": str(h["date"].iloc[ev_all[-1]]) if len(ev_all) else None,
           "since_days": int(n_all - 1 - ev_all[-1]) if len(ev_all) else None}
    for k in (5, 10, 20):
        out[f"fwd{k}"] = out[f"win{k}"] = out[f"excess{k}"] = None
    out.update(ci_ep=None, ci_year=None, years_pos=0, years_n=0, smd_ret20=None, n_matched=0, d0_excess10=None, sub2010=None)
    ev_idx, ep, ctrl, keep = _controls(h, mask)
    if len(ev_idx) == 0:
        return out
    out["n"], out["episodes"] = int(len(ev_idx)), int(ep.max() + 1)
    for k in (5, 10, 20):
        fe = fwd[k][ev_idx]
        out[f"fwd{k}"], out[f"win{k}"] = round(float(np.nanmean(fe)), 3), round(float((fe > 0).mean() * 100), 1)
    sub = ev_idx[years[ev_idx] >= 2010]                       # 2010~ 子樣本 (無條件平均；文案引用同物件數字、不硬編碼研究值)
    if len(sub):
        f10 = fwd[10][sub]
        out["sub2010"] = {"from": "2010-01-01", "n": int(len(sub)), "fwd10": round(float(np.nanmean(f10)), 3), "win10": round(float((f10 > 0).mean() * 100), 1),
                          "fwd5": round(float(np.nanmean(fwd[5][sub])), 3), "fwd20": round(float(np.nanmean(fwd[20][sub])), 3)}
    if not len(keep):
        return out
    ev_m, ep_m, yr_m = ev_idx[keep], ep[keep], years[ev_idx[keep]]
    out["n_matched"] = int(len(ev_m))
    rng = np.random.default_rng(SEED + sum(map(ord, eid)))
    for k in (5, 10, 20):
        fe = fwd[k][ev_m]
        fc = np.array([np.nanmean(fwd[k][c]) for c in ctrl])
        diff = fe - fc
        m = np.isfinite(diff)
        out[f"excess{k}"] = round(float(diff[m].mean()), 3) if m.any() else None
        if k == 10:
            out["ci_ep"] = _cluster_ci(diff[m], ep_m[m], rng)
            out["ci_year"] = _cluster_ci(diff[m], yr_m[m], rng)
            yr = pd.DataFrame({"y": yr_m[m], "d": diff[m]}).groupby("y")["d"].agg(["size", "mean"])
            yr = yr[yr["size"] >= 3]
            out["years_pos"], out["years_n"] = int((yr["mean"] > 0).sum()), int(len(yr))
            d0 = h["fwd10_d0"].to_numpy(float)
            dd = d0[ev_m] - np.array([np.nanmean(d0[c]) for c in ctrl])
            out["d0_excess10"] = round(float(np.nanmean(dd)), 3) if np.isfinite(dd).any() else None
    re_, rc = r20[ev_m], np.concatenate([r20[c] for c in ctrl])
    sd = np.sqrt((np.nanvar(re_, ddof=1) + np.nanvar(rc, ddof=1)) / 2) if len(re_) > 1 and len(rc) > 1 else np.nan
    out["smd_ret20"] = round(float((np.nanmean(re_) - np.nanmean(rc)) / sd), 3) if sd and np.isfinite(sd) and sd > 0 else None
    return out


def _ci_txt(ci) -> str:
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(x is not None for x in ci) else "CI n/a"


def _event_text(eid: str, L: pd.Series, ev: dict) -> str:
    """事件文案。active=False → 只給定義句 (不帶今日數值，由 App 依 active / since_days 呈現)；insufficient → 定義句 + 樣本不足。
    active=True → 今日數值 + 同物件計算值 (樣本內 fwd / 配對超額 / 2010~ 子樣本)，不硬編碼研究數字 (研究證據只在 evidence.note / placebo)。"""
    y = ev.get("since_year") or str(HISTORY_FROM)[:4]
    base = EVENT_DEFS_TEXT.get(eid, eid).format(y=y)
    if ev.get("insufficient"):
        return f"{base} (樣本不足，分位門檻需 ≥{MIN_QUANTILE_DAYS} 日)"
    if not ev.get("active"):
        s = f"{base} (目前未成立)"
    else:
        g = lambda c, fmt="{:+.1f}": _f(L.get(c), fmt)  # noqa: E731
        if eid == "streak_up7":
            s = (f"融資連增 {int(L.get('streak') or 0)} 日 (≥7)：樣本內事件後 5 日較對照 {_f(ev.get('excess5'), '{:+.2f}')} 個百分點 "
                 f"(n={ev.get('n') or 0}，10 日 {_f(ev.get('excess10'), '{:+.2f}')} {_ci_txt(ev.get('ci_ep'))}) → 描述")
        elif eid == "streak_dn7":
            s = f"融資連減 {abs(int(L.get('streak') or 0))} 日 (≥7)"
        elif eid == "wipeout":
            sub = ev.get("sub2010") or {}
            s = (f"融資 20 日 {g('pct20')}%、指數 {g('ret20')}% 第 {int(L.get('quad_days') or 0)} 日：樣本內事件後 10 日平均 {_f(ev.get('fwd10'), '{:+.2f}')}% "
                 f"(n={ev.get('n') or 0})" + (f"、2010~ 子樣本 {_f(sub.get('fwd10'), '{:+.2f}')}% / 勝率 {_f(sub.get('win10'), '{:.0f}')}% (n={sub.get('n')})" if sub else "")
                 + f"；配對超額 {_f(ev.get('excess10'), '{:+.2f}')} {_ci_txt(ev.get('ci_ep'))} → 描述")
        elif eid == "chase":
            s = f"融資 20 日增幅高於指數 {g('div20')} 個百分點 (>5)"
        elif eid == "knife":
            s = f"指數跌融資增 (資增價跌) 第 {int(L.get('quad_days') or 0)} 日"
        elif eid in ("ext_inc", "ext_dec"):
            s = f"單日融資 {g('chg1', '{:+.0f}')} 億，為 {y} 年來第 {_f(ev.get('chg1_pct'), '{:.0f}')} 百分位"
        elif eid == "hi250":
            s = "融資餘額創 250 日新高"
        elif eid == "lo250":
            s = "融資餘額創 250 日新低"
        elif eid == "maint_lt150":
            s = f"維持率 {g('maint', '{:.1f}')}% 低於 150 ({ev.get('share_since') or '覆蓋期'} {ev.get('n_total') or 0} 日、占 {_f(ev.get('share'), '{:.1f}')}%)"
        elif eid == "turnover_hot":
            s = f"日周轉 {g('turnover_ratio', '{:.1f}')}% 為 {y} 年來第 {_f(ev.get('turnover_pct'), '{:.0f}')} 百分位"
        else:
            s = f"5 日融資 {g('pct5')}%、維持率 5 日 {g('maint_chg5')} 個百分點"
    if ev.get("smd_ret20") is not None and abs(ev["smd_ret20"]) > 0.3:
        s += "；對照組跌幅仍小於事件日，超額為上界"
    return s


def events(h: pd.DataFrame, validated: list[str] | None = None) -> dict:
    """§2.4 12 事件 + baseline。label：validated 內 → predictive，否則 EVENT_DEFS 預設；grade 只進 evidence。"""
    validated = list(validated or [])
    out: dict = {"events": {}, "baseline": {}, "tests_total": TESTS_TOTAL["module_events"]}
    if h is None or h.empty:
        return out
    masks = _event_masks(h)
    insufficient = _quantile_insufficient(h)
    L = h.iloc[-1]
    m_ok = h["maint"].notna()
    maint_from = str(h.loc[m_ok, "date"].iloc[0])[:10] if m_ok.any() else None
    share_since = ("2004 年來" if maint_from is not None and maint_from <= MAINT_FULL_FROM else (f"{maint_from} 以來" if maint_from else None))
    extras = {"chg1_pct": float((h["chg1"] <= L["chg1"]).mean() * 100) if pd.notna(L["chg1"]) else None,
              "turnover_pct": float((h["turnover_ratio"].dropna() <= L["turnover_ratio"]).mean() * 100) if pd.notna(L["turnover_ratio"]) else None,
              "share": float(masks["maint_lt150"].sum() / max(int(m_ok.sum()), 1) * 100), "since_year": str(h["date"].iloc[0])[:4]}
    for eid, (lab, grade, evid) in EVENT_DEFS.items():
        st = _match_study(h, masks[eid], eid)
        st["active"] = bool(masks[eid].fillna(False).iloc[-1])
        st["insufficient"] = eid in insufficient
        st["label"] = "predictive" if eid in validated else lab
        st["evidence"] = {"grade": grade, "d0_excess10": st.pop("d0_excess10", None), "placebo": evid.get("placebo"), "note": evid.get("note"),
                          "tests_total": TESTS_TOTAL["module_events"], "matching": f"k={MATCH_K}, ret20 卡尺 ±{CALIPER_RET20:g} pt, 年差 ≤{YEAR_DIFF}, D+1 起算"}
        if eid == "maint_lt150":
            st["share"], st["share_since"], st["hist_from"] = _r(extras["share"], 2), share_since, maint_from
        ev = {**st, **extras}
        st["text"] = _event_text(eid, L, ev)
        out["events"][eid] = st
    for key, since in (("all", HISTORY_FROM), ("2010", "2010-01-01")):
        sub = h[h["date"] >= since]
        b = {"from": since, "n": int(sub["fwd10"].notna().sum())}
        for k in (5, 10, 20):
            v = sub[f"fwd{k}"].dropna()
            b[f"fwd{k}"], b[f"win{k}"] = (_r(v.mean(), 3), _r((v > 0).mean() * 100, 1)) if len(v) else (None, None)
        out["baseline"][key] = b
    out["note"] = (f"樣本內、D+1 收盤起算、對照 = 同日最近鄰配對 (k={MATCH_K}, ret20 卡尺)、CI = episode 與年 cluster bootstrap (B={BOOT_B}, seed {SEED})；"
                   "12 事件 × 3 視野 = 36 檢定，5% 期望 1.8 個假陽性；fwd20 含季節 (12 月 +2.1% ~ 7 月 −0.8%)")
    return out


# ------------------------------------------------------------------ 個股 (§2.2 / §3.6)
def restatements(df_code: pd.DataFrame) -> pd.DataFrame:
    """f = prev(D)/today(D−1)；kind ∈ {capital, adjust, settlement, ''}；adj = 連續化係數 (只用 capital 的 f；lots_adj = lots × adj)。"""
    d = df_code.sort_values("date").reset_index(drop=True)
    today = pd.to_numeric(d["m_today"], errors="coerce")
    prev = pd.to_numeric(d["m_prev"], errors="coerce")
    tp = today.shift(1)
    f = np.where((tp > 0) & (prev > 0), prev / tp, 1.0)
    f = np.where(np.isfinite(f), f, 1.0)
    dev = np.abs(f - 1)
    lim = pd.to_numeric(d["m_limit"], errors="coerce") if "m_limit" in d else pd.Series(np.nan, index=d.index)
    limchg = ((lim / lim.shift(1) - 1).abs() > 0.01).fillna(False).to_numpy()
    gap = pd.to_datetime(d["date"]).diff().dt.days.fillna(1).to_numpy()
    kind = np.full(len(d), "", dtype=object)
    settle = (gap >= 6) & (dev > 0.005) & (dev <= 0.4) & (f < 1)
    capital = (dev > 0.4) | ((dev > 0.10) & (dev <= 0.4) & limchg)
    adjust = (dev > 0.005) & ~capital
    kind[adjust] = "adjust"
    kind[capital] = "capital"
    kind[settle & ~capital] = "settlement"
    use = np.where(kind == "capital", f, 1.0)
    cum = np.cumprod(use[::-1])[::-1]
    adj = cum / use
    return pd.DataFrame({"date": d["date"].to_numpy(), "f": f, "kind": kind, "adj": adj, "limit_chg": limchg})


def _safe_div(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(len(a), np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def _fifo_cost(price: np.ndarray, buy: np.ndarray, exits: np.ndarray, seed_lots: float, seed_px: float, div: np.ndarray | None = None) -> np.ndarray:
    """FIFO 估計融資平均成本 (了結與現償先沖最早新增)；除息日在庫成本扣股利。"""
    lots: deque = deque()
    tot_amt = tot_lots = 0.0
    if seed_lots > 0 and np.isfinite(seed_px):
        lots.append([seed_px, seed_lots]); tot_amt, tot_lots = seed_px * seed_lots, seed_lots
    out = np.full(len(price), np.nan)
    for i in range(len(price)):
        px = price[i]
        if div is not None and div[i] > 0 and lots:
            for l in lots:
                l[0] = max(l[0] - div[i], 0.01)
            tot_amt = sum(l[0] * l[1] for l in lots)
        rem = exits[i] if np.isfinite(exits[i]) else 0.0
        while rem > 0 and lots:
            if lots[0][1] <= rem + 1e-9:
                rem -= lots[0][1]; tot_amt -= lots[0][0] * lots[0][1]; tot_lots -= lots[0][1]; lots.popleft()
            else:
                lots[0][1] -= rem; tot_amt -= lots[0][0] * rem; tot_lots -= rem; rem = 0.0
        b = buy[i] if np.isfinite(buy[i]) else 0.0
        if b > 0 and np.isfinite(px):
            lots.append([px, b]); tot_amt += px * b; tot_lots += b
        out[i] = tot_amt / tot_lots if tot_lots > 1e-9 else np.nan
    return out


def _div_array(dates: np.ndarray, exdiv_events) -> np.ndarray:
    div = np.zeros(len(dates))
    evs = exdiv_events or []
    if isinstance(evs, dict):
        evs = (evs.get("realized") or []) + (evs.get("upcoming") or [])
    pos = {str(d)[:10]: i for i, d in enumerate(dates)}
    for e in evs:
        try:
            ex, cash = str(e.get("ex_date"))[:10], float(e.get("cash") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if cash > 0 and ex in pos:
            div[pos[ex]] += cash
    return div


def stock_features(df_code: pd.DataFrame, closes=None, exdiv_events=None) -> pd.DataFrame:
    """單一個股 (margin_stock_daily 列) → §2.2 指標。移植研究 margin_lib.stock_features，改：重述規則 §3.6、停資恢復遮罩 40 日、基期 <50 → null、chg1 用 TWSE prev 欄、
    恆等式違反列 (identity_err) 流量不進統計、除息調整報酬與 FIFO 成本、分割時價格同步換基準。"""
    d = df_code.sort_values("date").reset_index(drop=True).copy()
    n = len(d)
    cols = ["date", "code", "lots", "lots_adj", "chg1", "chg5", "chg20", "pct5", "pct20", "util", "buy", "sell", "repay", "turnover5", "buy_vol_share",
            "repay_share", "offset_share5", "close", "ret5", "ret20", "quad20", "quad20_days", "div20", "short_lots", "short_ratio", "short_chg5",
            "short_util", "bal_pos250", "bal_vs_ma60", "cost_fifo", "pnl_fifo", "maint_est", "pressure_score", "pressure_partial", "streak", "identity_err", "halted",
            "short_halted", "alloc", "recovering", "burn_in", "kind", "f", "limit"]
    if n == 0:
        return pd.DataFrame(columns=cols)
    for c in ("m_buy", "m_sell", "m_repay", "m_prev", "m_today", "m_limit", "s_buy", "s_sell", "s_repay", "s_prev", "s_today", "s_limit", "offset", "close", "vol_lots"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    d["date"] = d["date"].astype(str).str[:10]
    if closes is not None:
        cm = closes if isinstance(closes, dict) else dict(zip(closes["date"].astype(str).str[:10], pd.to_numeric(closes["close"], errors="coerce")))
        fill = d["date"].map(lambda k: (cm.get(k)[0] if isinstance(cm.get(k), (tuple, list)) else cm.get(k)))
        d["close"] = d["close"].where(d["close"].notna(), pd.to_numeric(fill, errors="coerce"))
    note = d["note"].fillna("").astype(str) if "note" in d else pd.Series([""] * n)
    rs = restatements(d)
    adj = rs["adj"].to_numpy(float)
    lots = d["m_today"].to_numpy(float)
    lots_adj = lots * adj
    out = pd.DataFrame({"date": d["date"].to_numpy(), "code": d["code"].astype(str).to_numpy() if "code" in d else ""})
    out["lots"], out["lots_adj"] = lots, lots_adj
    out["kind"], out["f"] = rs["kind"].to_numpy(), rs["f"].to_numpy()
    out["limit"] = d["m_limit"].to_numpy(float)
    out["chg1"] = (d["m_today"] - d["m_prev"]).to_numpy(float)
    ident = d["m_prev"] + d["m_buy"] - d["m_sell"] - d["m_repay"]
    err = ((ident - d["m_today"]).abs() > 0.5) & ident.notna() & d["m_today"].notna()
    out["identity_err"] = err.to_numpy()
    scale = np.where(err.to_numpy(), np.nan, adj)
    buy, sell, repay = d["m_buy"].to_numpy(float) * scale, d["m_sell"].to_numpy(float) * scale, d["m_repay"].fillna(0).to_numpy(float) * scale
    exits = sell + repay
    halted = note.str.contains("O").to_numpy()
    out["halted"], out["short_halted"], out["alloc"] = halted, note.str.contains("X").to_numpy(), note.str.contains("[@!]", regex=True).to_numpy()
    start = (lots > 0) & (np.roll(lots, 1) == 0)
    start[0] = False
    recov = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(start):
        recov[i: i + RECOVER_DAYS] = True
    out["recovering"] = recov
    s = pd.Series(lots_adj)
    for k in (5, 20):
        base = s.shift(k).to_numpy()
        chg = s.to_numpy() - base
        pct = _safe_div(chg, base) * 100
        pct[~(base >= MIN_BASE_LOTS)] = np.nan
        pct[recov] = np.nan
        out[f"chg{k}"], out[f"pct{k}"] = chg, pct
    lim = d["m_limit"].to_numpy(float)
    util = _safe_div(lots, lim) * 100
    util[~(lim > 0) | halted] = np.nan
    out["util"] = util
    out["buy"], out["sell"], out["repay"] = buy, sell, repay
    vol = d["vol_lots"].to_numpy(float)
    out["turnover5"] = _safe_div(pd.Series(buy + sell).rolling(5, min_periods=1).sum().to_numpy(), lots_adj) * 100      # §2.2 (buy+sell)/lots，不含現償 (同市場級 turnover_ratio)
    out["buy_vol_share"] = _safe_div(buy, vol) * 100
    out["repay_share"] = _safe_div(repay, exits) * 100
    v5 = pd.Series(vol).rolling(5, min_periods=1).sum().to_numpy()
    out["offset_share5"] = _safe_div(d["offset"].rolling(5, min_periods=1).sum().to_numpy(float), v5) * 100
    close = d["close"].to_numpy(float)
    out["close"] = close
    div = _div_array(out["date"].to_numpy(), exdiv_events)
    fac = np.ones(n)                                         # 除息還原係數 (事件日前的收盤乘上 (prev−cash)/prev)
    for i in np.flatnonzero(div > 0):
        if i > 0 and np.isfinite(close[i - 1]) and close[i - 1] > div[i]:
            fac[:i] *= (close[i - 1] - div[i]) / close[i - 1]
    cadj = pd.Series(close * fac)
    cap_idx = np.flatnonzero(out["kind"].to_numpy() == "capital")
    for k in (5, 20):
        rk = (cadj / cadj.shift(k) - 1).to_numpy() * 100
        for i in cap_idx:                                     # §3.6(a)：股本事件日起 ret 視窗重算 (跨事件日的 k 日報酬不可比)
            rk[i: i + k] = np.nan
        out[f"ret{k}"] = rk
    m20, r20 = out["pct20"].to_numpy(float), out["ret20"].to_numpy(float)
    q = _quad(m20, r20)
    out["quad20"] = q
    run = np.zeros(n, dtype=int); last = None; cnt = 0
    for i in range(n):
        if q[i] == "" or q[i] != last:
            cnt = 1 if q[i] != "" else 0; last = q[i]
        else:
            cnt += 1
        run[i] = cnt
    out["quad20_days"] = run
    out["div20"] = m20 - r20
    sb = d["s_today"].to_numpy(float)
    out["short_lots"] = sb
    out["short_ratio"] = _safe_div(sb, lots) * 100
    out["short_chg5"] = pd.Series(sb).diff(5).to_numpy()
    out["short_util"] = _safe_div(sb, d["s_limit"].to_numpy(float)) * 100
    out["bal_pos250"] = _pct_excl(s, 250, 60).to_numpy()
    out["bal_vs_ma60"] = (s / s.rolling(60).mean() - 1).to_numpy() * 100
    padj = close / adj                                        # 分割時價格同步換到今日股數基準
    okp = np.isfinite(lots_adj) & np.isfinite(padj) & (lots_adj > 0)
    cost = np.full(n, np.nan)
    burn = np.arange(n) < BURN_IN
    if okp.any():
        # 起點 = 第一個「有餘額且有收盤」的列 (冷啟動時面板前段沒有收盤：種子匯入後收盤由 backfill 逐日補)；起點之前的流量不進帳本
        # (否則起點前的了結會沖掉代表起點餘額的種子批次)；burn-in 以起點起算且要求 ≥ BURN_IN 個有價列
        first = int(np.argmax(okp))
        cost[first:] = _fifo_cost(padj[first:], buy[first:], exits[first:], float(lots_adj[first]), float(padj[first]), div[first:] if div is not None else None)
        burn = burn | (np.arange(n) < first + BURN_IN) | (np.cumsum(np.isfinite(padj)) < BURN_IN)
    else:
        burn[:] = True
    cost[burn] = np.nan
    out["burn_in"] = burn
    out["cost_fifo"] = cost
    out["pnl_fifo"] = (close / cost - 1) * 100
    out["maint_est"] = close / (cost * MARGIN_RATIO) * 100
    # 融資壓力分數：35% 使用率 + 35% 估計浮虧 + 15% 5 日融資減幅 + 15% 5 日跌幅。兩個 35% 分項 (util / pnl_fifo) 缺 (停資、burn-in、無收盤) → null，
    # 不以 0 代入充當完整分數；15% 分項 (pct5 / ret5) 缺 → 以 0 代入並標 pressure_partial
    pnl = out["pnl_fifo"].to_numpy(float)
    p5, r5 = out["pct5"].to_numpy(float), out["ret5"].to_numpy(float)
    u = np.clip(np.nan_to_num(util, nan=0.0), 0, 100) / 100
    loss = np.clip(-np.nan_to_num(pnl, nan=0.0), 0, 30) / 30
    dec = np.clip(-np.nan_to_num(p5, nan=0.0), 0, 15) / 15
    drop = np.clip(-np.nan_to_num(r5, nan=0.0), 0, 15) / 15
    score = 35 * u + 35 * loss + 15 * dec + 15 * drop
    score[~(np.isfinite(util) & np.isfinite(pnl))] = np.nan
    out["pressure_score"] = score
    out["pressure_partial"] = np.isfinite(score) & ~(np.isfinite(p5) & np.isfinite(r5))
    out["streak"] = streak(out["chg1"]).to_numpy()
    return out[cols]


def _flags_at(F: pd.DataFrame, i: int) -> list[str]:
    r = F.iloc[i]
    fl = []
    if r["halted"]:
        fl.append("halted")
    if r["short_halted"]:
        fl.append("short_halted")
    if r["alloc"]:
        fl.append("alloc")
    kinds = F["kind"].iloc[max(0, i - 19): i + 1]
    if kinds.isin(["capital", "adjust"]).any():
        fl.append("restated")
    if r["recovering"]:
        fl.append("recovering")
    if r["burn_in"] or not np.isfinite(r["cost_fifo"]):
        fl.append("burn_in")
    if not np.isfinite(r["close"]):
        fl.append("no_close")
    return fl


def _restated_obj(F: pd.DataFrame) -> dict | None:
    sub = F.tail(250)
    cap = sub[sub["kind"] == "capital"]
    if cap.empty:
        return None
    r = cap.iloc[-1]
    return {"date": str(r["date"]), "f": _r(r["f"], 3), "kind": "capital"}


def _pool_codes(pools) -> list[str]:
    out: list[str] = []
    for p in pools or []:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            cols, rows = p.get("cols") or [], p.get("rows") or []
            if "code" in cols:
                ci = cols.index("code")
                out.extend(str(r[ci]) for r in rows if isinstance(r, (list, tuple)) and len(r) > ci)
            elif "code" in p:
                out.append(str(p["code"]))
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def stocks(panel: pd.DataFrame, ids: list[str], closes_hist: dict | None = None, exdiv_map: dict | None = None) -> dict:
    """追蹤 5 + pools ≤100 (ids 已排好)；每檔 60 列 daily；TPEx 檔 (mkt='tpex') v1 不含。"""
    out: dict = {}
    if panel is None or panel.empty:
        return out
    panel = panel.copy()
    panel["code"] = panel["code"].astype(str)
    groups = dict(tuple(panel.groupby("code")))
    for sid in ids[:MAX_STOCKS]:
        g = groups.get(sid)
        if g is None or g.empty:
            continue
        try:
            F = stock_features(g, (closes_hist or {}).get(sid), (exdiv_map or {}).get(sid))
        except Exception as e:  # noqa: BLE001
            log.debug("stock_features %s: %s", sid, e)
            continue
        if F.empty:
            continue
        L = F.iloc[-1]
        quad = L["quad20"] or QUADS[4]
        mode_txt = f"融資 20 日 {_fp(L['pct20'])} vs 股價 {_fp(L['ret20'])}，使用率 {_fp(L['util'], '{:.0f}')}"
        if pd.notna(L["pressure_score"]):
            mode_txt += f"，融資壓力分數 {L['pressure_score']:.0f}/100"
        tail = F.tail(60)
        daily = [[str(r.date)[5:10], _r(r.lots, 0), _r(r.buy, 0), _r(r.sell, 0), _r(r.repay, 0), _r(r.short_lots, 0), None, _r(r.close, 2), _r(r.util, 1)]
                 for r in tail.itertuples()]
        offs = pd.to_numeric(g.sort_values("date")["offset"], errors="coerce").tail(len(tail)).to_numpy() if "offset" in g else np.full(len(tail), np.nan)
        for row, o in zip(daily, offs):
            row[6] = _r(o, 0)
        out[sid] = {"name": str(g["name"].dropna().iloc[-1]) if "name" in g and g["name"].notna().any() else sid, "mkt": str(g["mkt"].iloc[-1]) if "mkt" in g else "twse",
                    "date": str(L["date"]), "lots": _r(L["lots"], 0), "chg1": _r(L["chg1"], 0), "chg5": _r(L["chg5"], 0), "chg20": _r(L["chg20"], 0),
                    "pct5": _r(L["pct5"], 1), "pct20": _r(L["pct20"], 1), "ret5": _r(L["ret5"], 1), "ret20": _r(L["ret20"], 1), "div20": _r(L["div20"], 1),
                    "util": _r(L["util"], 1), "limit": _r(L["limit"], 0), "short_lots": _r(L["short_lots"], 0), "short_ratio": _r(L["short_ratio"], 1),
                    "short_chg5": _r(L["short_chg5"], 0), "offset_share5": _r(L["offset_share5"], 2), "turnover5": _r(L["turnover5"], 1),
                    "bal_pos250": _r(L["bal_pos250"], 0), "streak": int(L["streak"]),
                    "quad": {"id": QUADS.index(quad) if quad in QUADS else 4, "text": quad, "days": int(L["quad20_days"])},
                    "cost_fifo": _r(L["cost_fifo"], 2), "pnl_fifo": _r(L["pnl_fifo"], 1), "maint_est": _r(L["maint_est"], 0),
                    "pressure_score": _r(L["pressure_score"], 0), "pressure_partial": bool(L["pressure_partial"]), "flags": _flags_at(F, len(F) - 1), "restated": _restated_obj(F),
                    "mode": {"id": quad, "text": mode_txt, "label": "descriptive"}, "cols": DAILY_COLS, "daily": daily, "n_days": int(len(F))}
    return out


# ------------------------------------------------------------------ 排行 (§2.3)
def _rank_text(board: str, r: dict) -> str:
    nm = r.get("name") or r.get("code")
    if board in ("inc5", "dec5", "etf_inc5", "etf_dec5"):
        s = f"{nm} 融資 5 日 {_f(r.get('chg5'), '{:+,.0f}')} 張 ({_fp(r.get('pct5'))})，股價 5 日 {_fp(r.get('ret5'))}"
        if r.get("ret5") is not None and r.get("pct5") is not None:
            s += f"；融資增幅與股價漲幅差 {r['pct5'] - r['ret5']:+.1f} 個百分點"
    elif board == "util":
        s = f"{nm} 融資使用率 {_fp(r.get('util'), '{:.0f}')} (餘額 {_f(r.get('lots'), '{:,.0f}')} / 限額 {_f(r.get('limit'), '{:,.0f}')} 張)"
    elif board == "pressure":
        s = (f"{nm} 融資壓力分數 {_f(r.get('v'), '{:.0f}')}：使用率 {_fp(r.get('util'), '{:.0f}')}、估計維持率約 {_fp(r.get('maint_est'), '{:.0f}')}、"
             f"5 日融資 {_fp(r.get('pct5'))}" + (" (5 日分項缺，以 0 計)" if r.get("pressure_partial") else ""))
    elif board == "div20":
        s = f"{nm} 20 日融資 {_fp(r.get('pct20'))} 而股價 {_fp(r.get('ret20'))} (背離 {_f(r.get('v'))} 個百分點)"
    elif board == "offset":
        s = f"{nm} 近 5 日資券互抵占成交量 {_fp(r.get('v'), '{:.2f}')} (資券當沖)"
    else:
        s = f"{nm} 券資比 {_fp(r.get('v'), '{:.0f}')}，融券 5 日 {_f(r.get('short_chg5'), '{:+,.0f}')} 張"
    if (r.get("days_on") or 0) >= 2:
        s += f"；已連續上榜 {r['days_on']} 日"
    return s


RANK_THRESHOLDS = {"lots_min": 500, "chg5_abs_min": 100, "util_min": 60, "short_ratio_min": 20,
                   "pressure_lots_min": 500}      # 偏離 SPEC §2.3 (「pressure_score 前 15」未設張數門檻)：壓力榜同樣要求 lots ≥500，避免數十張的檔混入；輸出於 ranking.thresholds


def ranking(panel_last60: pd.DataFrame, closes: dict | None = None, snapshots: list | None = None, save: bool = True) -> dict:
    """七榜 ×15 + ETF 小榜；days_on / cum_on 由 snapshots (None → store.load_snapshots('margin_rank'))；stats 以同日等權宇宙為基準。
    門檻見 RANK_THRESHOLDS (輸出 ranking.thresholds)；壓力榜只列 pressure_score 非 null (util 與 pnl_fifo 皆有) 的檔。"""
    boards = ["inc5", "dec5", "etf_inc5", "etf_dec5", "util", "pressure", "div20", "offset", "short"]
    out: dict = {"date": "", "n_days": 0, "stats": {}, "label": {"util": "risk_descriptive", "pressure": "risk_descriptive"}, "thresholds": dict(RANK_THRESHOLDS)}
    for b in boards:
        out[b] = []
    if panel_last60 is None or panel_last60.empty:
        return out
    panel = panel_last60.copy()
    panel["code"] = panel["code"].astype(str)
    rows = []
    for code, g in panel.groupby("code"):
        try:
            F = stock_features(g, (closes or {}).get(code))
        except Exception as e:  # noqa: BLE001
            log.debug("ranking features %s: %s", code, e)
            continue
        if F.empty:
            continue
        L = F.iloc[-1].to_dict()
        L["flags"] = _flags_at(F, len(F) - 1)
        L["name"] = str(g["name"].dropna().iloc[-1]) if "name" in g and g["name"].notna().any() else code
        L["mkt"] = str(g["mkt"].iloc[-1]) if "mkt" in g else "twse"
        rows.append(L)
    if not rows:
        return out
    F = pd.DataFrame(rows)
    date = str(F["date"].max())
    F = F[F["date"] == date].reset_index(drop=True)
    out["date"] = date
    etf = F["code"].map(_is_etf)
    T = RANK_THRESHOLDS
    big = F["lots"] >= T["lots_min"]
    mv5 = (F["chg5"].abs() >= T["chg5_abs_min"]) & F["pct5"].notna()
    sel = {
        "inc5": F[~etf & big & mv5].sort_values("pct5", ascending=False),
        "dec5": F[~etf & big & mv5].sort_values("pct5", ascending=True),
        "etf_inc5": F[etf & big & mv5].sort_values("pct5", ascending=False),
        "etf_dec5": F[etf & big & mv5].sort_values("pct5", ascending=True),
        "util": F[F["util"] >= T["util_min"]].sort_values("util", ascending=False),
        "pressure": F[F["pressure_score"].notna() & (F["lots"] >= T["pressure_lots_min"])].sort_values("pressure_score", ascending=False),
        "div20": F[big & F["div20"].notna()].assign(_a=lambda x: x["div20"].abs()).sort_values("_a", ascending=False),
        "offset": F[F["offset_share5"] > 0].sort_values("offset_share5", ascending=False),
        "short": F[(F["short_ratio"] >= T["short_ratio_min"]) & big].sort_values("short_ratio", ascending=False),
    }
    vcol = {"inc5": "pct5", "dec5": "pct5", "etf_inc5": "pct5", "etf_dec5": "pct5", "util": "util", "pressure": "pressure_score", "div20": "div20",
            "offset": "offset_share5", "short": "short_ratio"}
    for b, df in sel.items():
        for _, r in df.head(15).iterrows():
            row = {"code": r["code"], "name": r["name"], "mkt": r["mkt"], "v": _r(r[vcol[b]], 2), "lots": _r(r["lots"], 0), "chg5": _r(r["chg5"], 0),
                   "pct5": _r(r["pct5"], 1), "util": _r(r["util"], 1), "ret5": _r(r["ret5"], 1), "days_on": 0, "cum_on": 0, "flags": list(r["flags"])}
            if b == "util":
                row["limit"] = _r(r["limit"], 0)
            if b == "pressure":
                row["maint_est"], row["pressure_partial"] = _r(r["maint_est"], 0), bool(r.get("pressure_partial", False))
            if b == "div20":
                row["pct20"], row["ret20"] = _r(r["pct20"], 1), _r(r["ret20"], 1)
            if b == "short":
                row["short_chg5"] = _r(r["short_chg5"], 0)
            out[b].append(row)
    if snapshots is None:
        try:
            if save:
                store.save_snapshot(date, "margin_rank", {b: [{"code": r["code"], "v": r["v"]} for r in out[b]] for b in boards})
            snapshots = store.load_snapshots("margin_rank", 60)
        except Exception as e:  # noqa: BLE001
            log.debug("margin_rank snapshots: %s", e)
            snapshots = []
    snaps = [(d, p) for d, p in snapshots if d <= date]
    if not any(d == date for d, _ in snaps):
        snaps.insert(0, (date, {b: [{"code": r["code"], "v": r["v"]} for r in out[b]] for b in boards}))
    snaps.sort(key=lambda x: x[0], reverse=True)
    out["n_days"] = len(snaps)
    prev_codes = {b: {x["code"] for x in (snaps[1][1].get(b) or [])} if len(snaps) > 1 else set() for b in boards}
    uni_ret5 = F["ret5"].dropna()
    out["stats"]["universe"] = {"date": date, "n": int(len(F)), "ret5_eqw": _r(uni_ret5.mean(), 2) if len(uni_ret5) else None, "basis": "同日等權宇宙 (全部有融資列的個股)"}
    for b in boards:
        for r in out[b]:
            hits = [any(x.get("code") == r["code"] for x in (payload.get(b) or [])) for _, payload in snaps]
            days = 0
            for hit in hits:                     # days_on：從最新快照往回連續；cum_on：快照視窗 (≤60 日) 內上榜總日數
                if not hit:
                    break
                days += 1
            r["days_on"], r["cum_on"] = days, int(sum(hits))
            r["text"] = _rank_text(b, r)
        if out[b]:
            codes = {r["code"] for r in out[b]}
            rb = [r["ret5"] for r in out[b] if r["ret5"] is not None]
            out["stats"][b] = {"n": len(out[b]), "avg_days_on": _r(np.mean([r["days_on"] for r in out[b]]), 1),
                               "turnover": _r(len(codes - prev_codes[b]) / len(codes) * 100, 0) if prev_codes[b] else None,
                               "ret5_eqw": _r(np.mean(rb), 2) if rb else None, "universe_ret5_eqw": out["stats"]["universe"]["ret5_eqw"]}
    return out


# ------------------------------------------------------------------ 警示 (全描述；≤12)
def alerts(view: dict, rank: dict, stk: dict) -> list[str]:
    al: list[str] = []
    if view:
        q = view.get("quad") or {}
        if q.get("days") == 1 and q.get("text"):
            al.append(f"融資-指數象限轉為『{q['text']}』(20 日融資 {_f(view.get('pct20'))}% vs 指數 {_f(view.get('ret20'))}%) (描述)")
        st = int(view.get("streak") or 0)
        if abs(st) >= 7:
            al.append(f"融資連{'增' if st > 0 else '減'} {abs(st)} 日 (2010~ 連增 ≥7 日占 7.8% 的日子；對後市無可驗證差異) (描述)")
        z1 = view.get("z1")
        if z1 is not None and abs(z1) >= 2:
            al.append(f"單日融資 {_f(view.get('chg1'), '{:+.0f}')} 億，為 60 日 {z1:+.1f} 個標準差 (描述)")
        if view.get("pct250") is not None and view["pct250"] >= 99 and (view.get("dhi250") or 0) >= 0:
            al.append("融資餘額創 250 日新高 (年去均值後對後市無差異) (描述)")
        if view.get("pct250") is not None and view["pct250"] <= 1 and (view.get("dlo250") or 0) <= 0:
            al.append("融資餘額創 250 日新低 (描述)")
        m = view.get("maint") or {}
        mv = m.get("value") if m.get("value") is not None else m.get("real")
        if mv is not None and mv < 150:
            al.append(f"大盤融資維持率 {mv:.1f}% 低於 150 觀察線 (TWSE 自算；維持率五分位對未來報酬非單調) (風險描述)")
        if m.get("check"):
            al.append(f"維持率自算 {m['value']:.1f}% 與玩股網 {m['real']:.1f}% 差 >1.5 個百分點，請以日期與來源對照 (描述)")
        cr = view.get("cross") or {}
        if cr.get("br_net_z60") is not None and abs(cr["br_net_z60"]) >= 2:
            al.append(f"融資增加家數比 {_f(cr.get('br_up'), '{:.0f}')}% vs 減少 {_f(cr.get('br_dn'), '{:.0f}')}% (淨 60 日 z {cr['br_net_z60']:+.1f}) (描述)")
        if (cr.get("usage_gt80_n") or 0) >= 10:
            al.append(f"融資使用率 >80% 共 {cr['usage_gt80_n']} 檔 (餘額 / 限額；接近上限時新增融資額度受限) (風險描述)")
        reg = view.get("regime") or {}
        if reg.get("days") == 20:
            al.append(f"{reg.get('text')} (描述)")
    for b, txt in (("inc5", "融資 5 日增幅"), ("dec5", "融資 5 日減幅")):
        for r in (rank or {}).get(b, [])[:15]:
            if (r.get("days_on") or 0) >= 3:
                al.append(f"{r['code']} {r.get('name') or ''} 連續 {r['days_on']} 日列{txt}榜，5 日 {_f(r.get('chg5'), '{:+,.0f}')} 張 ({_f(r.get('pct5'))}%) (描述)")
    for sid, r in (stk or {}).items():
        fl = r.get("flags") or []
        if "halted" in fl:
            al.append(f"{sid} {r.get('name') or ''} 停資 (TWSE 註記 O) (描述)")
        rs = r.get("restated")
        if rs and rs.get("date") == r.get("date"):
            al.append(f"{sid} {r.get('name') or ''} 融資張數重述 ×{rs['f']:.2f} (股本事件)，20 日統計自該日重算 (描述)")
        if r.get("maint_est") is not None and r["maint_est"] < 130:
            al.append(f"{sid} {r.get('name') or ''} 估計維持率約 {r['maint_est']:.0f}% (FIFO 估計值非個別帳戶) (風險描述)")
    return al[:12]


# ------------------------------------------------------------------ 體積守門 / 組裝
def _size(out: dict) -> int:
    return len(json.dumps(clean(out), ensure_ascii=False, default=str).encode("utf-8"))


def _cut_table(tbl: dict, keep_rows) -> dict:
    rows = tbl.get("rows") or []
    return {"cols": tbl.get("cols"), "rows": keep_rows(rows)}


def size_guard(out: dict, limit: int = SIZE_LIMIT) -> dict:
    """raw >900 KB → history 砍到 2010~ → stocks 砍到 60 檔 → derived 砍到 250 日。"""
    steps = []
    size = _size(out)
    if size > limit and out.get("history"):
        out["history"] = _cut_table(out["history"], lambda rows: [r for r in rows if str(r[0]) >= "2010-01-01"])
        steps.append("history_2010"); size = _size(out)
    if size > limit and out.get("stocks"):
        keep = (out.get("watch_ids") or []) + [c for c in (out.get("pool_ids") or []) if c not in (out.get("watch_ids") or [])]
        keep = [c for c in keep if c in out["stocks"]][:60] or list(out["stocks"])[:60]
        out["stocks"] = {k: out["stocks"][k] for k in keep}
        out["pool_ids"] = [c for c in (out.get("pool_ids") or []) if c in out["stocks"]]
        steps.append("stocks_60"); size = _size(out)
    if size > limit and out.get("derived"):
        out["derived"] = _cut_table(out["derived"], lambda rows: rows[-250:])
        steps.append("derived_250"); size = _size(out)
    out.setdefault("meta", {})["size_guard"] = {"steps": steps, "bytes": size}
    return out


def _table(h: pd.DataFrame, cols: list[str], digits: dict, ren: dict | None = None, mask=None) -> dict:
    ren = ren or {}
    src = [ren.get(c, c) for c in cols]
    d = h.copy()
    for c in src:
        if c not in d:
            d[c] = np.nan
    if mask is not None:
        d = d[mask]
    rows = []
    for rec in d[src].itertuples(index=False):
        row = []
        for c, v in zip(cols, rec):
            if c == "date":
                row.append(str(v)[:10])
            elif c == "regime":
                row.append(v if isinstance(v, str) else "")
            else:
                row.append(_r(v, digits.get(c, 2)))
        rows.append(row)
    return {"cols": cols, "rows": rows}


def _skeleton(meta: dict) -> dict:
    return {"meta": meta, "market": {}, "history": {"cols": HISTORY_COLS, "rows": []}, "flows": {"cols": FLOW_COLS, "rows": []},
            "derived": {"cols": DERIVED_COLS, "rows": []}, "ranking": {"date": "", "n_days": 0}, "stocks": {}, "watch_ids": [], "pool_ids": [], "alerts": []}


def _watch_ids() -> list[str]:
    try:
        from . import chips
        return list(chips.WATCHLIST)
    except Exception:  # noqa: BLE001
        return list(WATCH_IDS)


def build(scored: pd.DataFrame | None = None, watch_res: dict | None = None, pools=None, prev: dict | None = None, fast: bool = False) -> dict:
    """fast：不連網 (除 published)，整份沿用 prev、meta.carried=True；full：單段失敗沿用 prev 對應區塊並列 meta.carried_parts。"""
    prev = published(prev)
    now = dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S")
    latest = None
    try:
        if scored is not None and not scored.empty and "date" in scored:
            latest = str(pd.Series(scored["date"]).astype(str).max())[:10]
    except Exception:  # noqa: BLE001
        latest = None
    pmeta = dict(prev.get("meta") or {})
    meta = {"generated": now, "latest_trading": latest or pmeta.get("latest_trading"), "mode": "fast" if fast else "full", "version": VERSION,
            "carried": False, "carried_parts": [], "sources": dict(pmeta.get("sources") or {}), "coverage": dict(pmeta.get("coverage") or {}),
            "labels": LABELS, "validated": [], "tests_total": TESTS_TOTAL, "publish_log": list(pmeta.get("publish_log") or [])[-30:]}
    try:
        meta["validated"] = validated_ids()
    except Exception:  # noqa: BLE001
        meta["validated"] = []
    if fast:
        out = dict(prev) if prev else _skeleton(meta)
        meta["carried"], meta["carried_parts"] = True, list(PARTS)
        out["meta"] = meta
        return out
    out = _skeleton(meta)
    carried: list[str] = []
    src = meta["sources"]
    # 1. 當日面板 (TWSE ALL + MI_INDEX) + TPEx + 玩股網 → SQLite
    day = latest or dt.datetime.now(config.TZ).date().isoformat()
    try:
        have = set(_panel_dates())
        if day not in have and day not in _nodata("twse_all"):
            try:
                res = _fetch_margin_all(_ymd(day))
                date, mkt, rows = _norm_day(res)
                date = date or day
                closes = None
                try:
                    _sleep(); closes = _norm_closes(_fetch_closes(_ymd(date)))
                except Exception as e:  # noqa: BLE001
                    log.debug("closes_all %s: %s", date, e)
                _panel_upsert(date, rows, closes)
                _drop_raw_cache(date)
                _day_metrics(date, rows, closes, mkt)
                src["twse_all"] = {"date": date, "ok": True}
            except Exception as e:  # noqa: BLE001
                src["twse_all"] = {"date": day, "ok": False, "reason": "notyet" if _is(e, "NoData") else str(e)[:80]}
        else:
            src["twse_all"] = {"date": day, "ok": day in have}
        src["tpex"] = {"date": day, "status": _tpex_day(day) if day not in _nodata("tpex") else "closed"}
    except Exception as e:  # noqa: BLE001
        log.warning("margin day panel: %s", e)
    try:
        wg = _fetch_maint_real()
        if isinstance(wg, pd.DataFrame) and not wg.empty:
            col = next((c for c in ("maint_ratio", "maint_real", "marginRatio", "maint", "value") if c in wg), None)
            if col:
                w = pd.DataFrame({"date": wg["date"].astype(str).str[:10], "maint_real": pd.to_numeric(wg[col], errors="coerce")}).dropna()
                if w["maint_real"].max() < 10:
                    w["maint_real"] *= 100
                store.upsert_frame(w, ["maint_real"])
                src["wantgoo"] = {"date": str(w["date"].max()), "ok": True}
        else:
            src["wantgoo"] = {"date": None, "ok": False}
    except Exception as e:  # noqa: BLE001
        log.info("wantgoo maint_real: %s", e)
        src["wantgoo"] = {"date": None, "ok": False}
    # 2. Tier-1 序列 → 自算維持率回補 → 衍生 / 事件
    hist = pd.DataFrame()
    h = pd.DataFrame()
    try:
        hist = market_history(prev)
        if hist.empty:
            raise RuntimeError("history empty")
        if "close" not in hist or hist["close"].notna().sum() < 20:
            raise RuntimeError("no price series (FinMind TAIEX / daily_metrics twii_close 皆無)")
        for c in ("buy_lots", "maint_twse", "tpex_margin_amt"):
            if c not in hist:
                hist[c] = np.nan
        try:
            n_m = _recompute_maint(hist)
            if n_m:
                hist = market_history(prev)
        except Exception as e:  # noqa: BLE001
            log.debug("recompute maint: %s", e)
        view, h = market_view(hist, scored)
        ev = events(h, meta["validated"])
        view["events"], view["baseline"], view["events_note"] = ev["events"], ev["baseline"], ev.get("note")
        out["market"] = view
        src["finmind_total"] = {"date": str(hist["date"].max())}
        ms_rows = hist.dropna(subset=["buy_lots"])
        src["twse_ms"] = {"date": str(ms_rows["date"].max()) if len(ms_rows) else None, "ok": bool(len(ms_rows))}
        src["maint"] = {"src": view["maint"]["src"], "real_date": view["maint"]["real_date"]}
        out["history"] = _table(h, HISTORY_COLS, {"margin_amt": 1, "margin_lots": 0, "short_lots": 0, "maint_twse": 2, "tpex_margin_amt": 1})
        fl_mask = h["buy_lots"].notna()
        out["flows"] = _table(h.tail(520), FLOW_COLS, {"buy_lots": 0, "sell_lots": 0, "repay_lots": 0, "buy_amt": 1, "sell_amt": 1, "short_buy": 0, "short_sell": 0,
                                                       "short_repay": 0, "br_net": 1, "usage_gt60_pct": 2}, mask=fl_mask.tail(520))
        out["derived"] = _table(h.tail(400), DERIVED_COLS, {"chg1": 1, "z5": 2, "pct20": 2, "ret20": 2, "div20": 1, "short_ratio": 2, "turnover_ratio": 2, "maint": 1})
        maint_rows = h.dropna(subset=["maint"])
        tp_rows = h.dropna(subset=["tpex_margin_amt"])
        meta["coverage"].update({"history_from": str(h["date"].iloc[0]), "history_days": int(len(h)),
                                 "flows_from": str(ms_rows["date"].min()) if len(ms_rows) else None, "flows_days": int(len(ms_rows)),
                                 "maint_from": str(maint_rows["date"].min()) if len(maint_rows) else None, "maint_days": int(len(maint_rows)),
                                 "tpex_from": str(tp_rows["date"].min()) if len(tp_rows) else None})
    except Exception as e:  # noqa: BLE001
        log.warning("margin market/history: %s", e)
        for part in ("market", "history", "flows", "derived"):
            if prev.get(part):
                out[part] = prev[part]
            carried.append(part)
    # 3. 個股面板 → stocks / ranking
    watch = _watch_ids()
    pool_ids = [c for c in _pool_codes(pools) if c not in watch][: max(0, MAX_STOCKS - len(watch))]
    if not pool_ids:
        pool_ids = [c for c in (prev.get("pool_ids") or []) if c not in watch][: max(0, MAX_STOCKS - len(watch))]
    out["watch_ids"], out["pool_ids"] = watch, pool_ids
    panel = pd.DataFrame()
    try:
        dates = _panel_dates()
        meta["coverage"].update({"panel_from": dates[0] if dates else None, "panel_days": len(dates)})
        if dates:
            since = dates[-min(len(dates), 320)]
            panel = load_panel(since=since)
    except Exception as e:  # noqa: BLE001
        log.warning("margin panel load: %s", e)
    try:
        if panel.empty:
            raise RuntimeError("panel empty")
        exmap = {}
        if isinstance(watch_res, dict):
            for sid, a in watch_res.items():
                if isinstance(a, dict) and isinstance(a.get("exdiv"), dict):
                    exmap[sid] = a["exdiv"]
        out["stocks"] = stocks(panel, watch + pool_ids, None, exmap)
        meta["coverage"]["stocks_n"] = len(out["stocks"])
    except Exception as e:  # noqa: BLE001
        log.warning("margin stocks: %s", e)
        out["stocks"] = prev.get("stocks") or {}
        carried.append("stocks")
    try:
        if panel.empty:
            raise RuntimeError("panel empty")
        out["ranking"] = ranking(panel)
    except Exception as e:  # noqa: BLE001
        log.warning("margin ranking: %s", e)
        out["ranking"] = prev.get("ranking") or {"date": "", "n_days": 0}
        carried.append("ranking")
    try:
        out["alerts"] = alerts(out.get("market") or {}, out.get("ranking") or {}, out.get("stocks") or {})
    except Exception as e:  # noqa: BLE001
        log.warning("margin alerts: %s", e)
        out["alerts"] = prev.get("alerts") or []
        carried.append("alerts")
    try:
        meta["coverage"]["backfill"] = {"pending_days": None, "last_run": now}
        cal = trading_days(PANEL_FROM, day)
        have = set(_panel_dates()) | _nodata("twse_all")
        meta["coverage"]["backfill"]["pending_days"] = len([d for d in cal if d not in have])
    except Exception as e:  # noqa: BLE001
        log.debug("pending days: %s", e)
    mdate = (out.get("market") or {}).get("date")
    if mdate and latest and mdate == latest and not any(p.get("date") == mdate for p in meta["publish_log"]):
        meta["publish_log"] = (meta["publish_log"] + [{"date": mdate, "first_seen": now}])[-30:]
    meta["carried"] = bool(carried) and set(carried) >= set(PARTS)
    meta["carried_parts"] = carried
    out["meta"] = meta
    return size_guard(out)


def write_seed(out: dict, path: Path | str | None = None) -> Path | None:
    """Tier-1 精簡種子 (§3.4)：history (2001~ 6 欄) + flows (2 年) → data/margin_market.json (≈350 KB)，train 模式隨 models 一起 git add；
    market_history() 讀它當最後備援 (Actions cache 與 Pages 都沒有時仍能重建全市場序列)。沿用 (carried) 或空的 history 不覆寫既有種子；回寫入路徑或 None。"""
    p = Path(path) if path else SEED_PATH
    hist = (out or {}).get("history") or {}
    rows = hist.get("rows") if isinstance(hist, dict) else None
    if not rows or list(hist.get("cols") or []) != HISTORY_COLS or "history" in (((out or {}).get("meta") or {}).get("carried_parts") or []):
        return None
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
            if len(((old.get("history") or {}).get("rows") or [])) > len(rows) * 1.02:      # 縮水保護：新種子比舊的少 >2% 不覆寫
                return None
        except Exception:  # noqa: BLE001
            pass
    seed = {"meta": {"generated": ((out.get("meta") or {}).get("generated")), "version": VERSION, "history_days": len(rows)},
            "history": {"cols": list(hist["cols"]), "rows": rows}}
    fl = out.get("flows") or {}
    if isinstance(fl, dict) and fl.get("rows"):
        seed["flows"] = {"cols": list(fl.get("cols") or FLOW_COLS), "rows": fl["rows"]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(seed), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return p
