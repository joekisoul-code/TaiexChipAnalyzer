"""事件預判模組 (2026-09-27)：休市 / 選舉 / 總經 / 台積電法說等「已排程事件」的歷史統計與已驗證的寬度、跳空、IV 預期。

研究：四族 810 項檢定 (台灣選舉/兩岸、美國選舉、FOMC/CPI/NFP/央行/法說/MSCI、節日與年底)，每項獨立驗證；
參數全部在 data/models/event_params.json (研究產出，照原樣出貨)。結論：
- 方向 (漲跌)：**沒有任何事件通過** (Holm/BH 校正後全滅；walk-forward 扣成本全輸單純持有) → 對 short_term / five_day / verdict
  的投票權重固定 0，本模組**永不輸出方向訊號**，方向統計只當描述並標「未通過驗證」。
- 休市後首日：實現波動 k=1 帶寬 × √max(1, n_US) (n_US = 休市期間已收盤的美股交易日數；無擬合參數)。
- 休市後 0050 開盤跳空 ≈ 正常日 β × 休市期間費半累積 log 報酬 (樣本外 R² 0.62~0.68)；前一晚美股休市 → 跳空幅度約 ×0.76。
- 跨休市 TXO ivk5 機械下降 −0.084 (春節 −0.216)；選擇權 IV 機率帶的事件乘數一律 1.0，
  唯一例外：台積電法說 (日期經 IR 確認) 當日收盤發布的操盤台 2330 k=1 帶 ×1.8 (證據「中」)。
其餘一律只描述 (「歷史上無可辨識效應」)。

主要函式：load_params、refresh_calendar、n_us_between、n_us_uncovered、session_factor、gap_forecast、iv_expect、build。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)
MODEL_DIR = config.DATA_DIR / "models"
PARAMS_PATH = MODEL_DIR / "event_params.json"
CAL_PATH = MODEL_DIR / "event_calendar.json"
DISCLAIMER = "歷史統計 / 機率帶描述，非買賣訊號；方向統計全部未通過驗證。"
NO_EFFECT = "歷史上無可辨識效應"
UNVERIFIED = "未通過驗證"
_P_CACHE: dict = {}

# ============================================================ 參數
def load_params(path: Path | str | None = None) -> dict:
    """讀 event_params.json 並檢查 schema；direction.vote_weight 必須為 0、votes 必須為空 (事件永不參與方向投票)。"""
    p = Path(path or PARAMS_PATH)
    key = (str(p), p.stat().st_mtime if p.exists() else 0)
    if key in _P_CACHE:
        return _P_CACHE[key]
    P = json.loads(p.read_text(encoding="utf-8"))
    if P.get("schema") != "event_params/v1":
        raise ValueError(f"event_params schema {P.get('schema')!r} != 'event_params/v1'")
    d = P.get("direction") or {}
    if float(d.get("vote_weight", -1)) != 0.0 or d.get("votes") != {}:
        raise ValueError("event_params.direction.vote_weight 必須為 0 且 votes 為空")
    for k in ("band_rules", "gap_rules", "iv_expectations", "event_types"):
        if not isinstance(P.get(k), dict):
            raise ValueError(f"event_params 缺少 {k}")
    _P_CACHE.clear()
    _P_CACHE[key] = P
    return P


def banned_phrases(P: dict | None = None) -> list[str]:
    return list(((P or load_params()).get("direction") or {}).get("banned_phrases_zh") or [])


def clean_text(s: str | None, P: dict | None = None) -> str | None:
    """文案保險：禁用字 (買進/賣出/加碼…) 不應出現；萬一出現就改成中性字眼並記 warning (測試會強制檢查)。"""
    if s is None:
        return None
    out = str(s)
    for w in banned_phrases(P):
        if w in out:
            log.warning("events 文案含禁用字 %s：%s", w, out)
            out = out.replace(w, "〔略〕")
    return out


def range_levels_enabled(P: dict | None = None) -> bool:
    """√n_US 是否套到 range_levels 路徑帶 (k=1)。研究只驗證了收盤 sigma60 帶 → 需驗收 A 通過才開 (見 range_levels_path.acceptance)。"""
    r = ((P or load_params()).get("band_rules") or {}).get("post_closure_sqrt_nus") or {}
    return bool(r.get("enabled")) and bool((r.get("range_levels_path") or {}).get("enabled"))


# ============================================================ 日曆：農曆表 / 台股規則假日 / NYSE
LUNAR = {  # 公曆日期 (中央氣象署農民曆)
    "LNY": {2021: "02-12", 2022: "02-01", 2023: "01-22", 2024: "02-10", 2025: "01-29", 2026: "02-17", 2027: "02-06", 2028: "01-26", 2029: "02-13", 2030: "02-03"},
    "DRAGON": {2021: "06-14", 2022: "06-03", 2023: "06-22", 2024: "06-10", 2025: "05-31", 2026: "06-19", 2027: "06-09", 2028: "05-28", 2029: "06-16", 2030: "06-05"},
    "MIDAUTUMN": {2021: "09-21", 2022: "09-10", 2023: "09-29", 2024: "09-17", 2025: "10-06", 2026: "09-25", 2027: "09-15", 2028: "10-03", 2029: "09-22", 2030: "09-12"},
    "QINGMING": {2021: "04-04", 2022: "04-05", 2023: "04-05", 2024: "04-04", 2025: "04-04", 2026: "04-05", 2027: "04-05", 2028: "04-04", 2029: "04-04", 2030: "04-05"},
}


def _d(s) -> dt.date:
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(str(s)[:10])


def _lunar(kind: str, y: int) -> dt.date | None:
    v = LUNAR[kind].get(y)
    return dt.date.fromisoformat(f"{y}-{v}") if v else None


def _observe(d: dt.date) -> dt.date:
    """台灣紀念日：逢週六於前一日、逢週日於後一日放假。"""
    return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d


def tw_rule_holidays(y: int) -> dict[str, str]:
    """台股休市日規則估計 (ISO 日期 → 名稱，只含平日)：固定國定假日 (含 2025 起新增的教師節、光復節、行憲紀念日) + 農曆表。
    春節：小年夜~初三 (LNY−2~LNY+2)，區塊內每個週末日往後補 1 個平日；區塊前 2 個平日為「僅辦理結算交割、無交易」。"""
    out: dict[str, str] = {}

    def add(d: dt.date, name: str, observe: bool = True):
        d2 = _observe(d) if observe else d
        # 與其他假日撞期：逢週日 (補假往後) → 往後找下一個空的平日 (例：2027 兒童節 04-04 週日，04-05 已是清明 → 04-06 週二，
        # 人事行政總處 116 年行事曆)；其餘 (逢週六往前補、或平日本身撞期) → 往前一個平日
        step = dt.timedelta(days=1 if (observe and d.weekday() == 6) else -1)
        while d2.isoformat() in out:
            d2 += step
            while d2.weekday() >= 5:
                d2 += step
        if d2.weekday() < 5 and d2.year == y:
            out[d2.isoformat()] = name
    fixed = [(1, 1, "元旦"), (2, 28, "和平紀念日"), (5, 1, "勞動節"), (10, 10, "國慶日")]
    if y >= 2025:
        fixed += [(9, 28, "教師節"), (10, 25, "光復節"), (12, 25, "行憲紀念日")]
    for m, dd, nm in fixed:
        if (m, dd) == (1, 1) and dt.date(y, 1, 1).weekday() == 5:
            continue                                              # 元旦逢週六 → 前一年 12-31 放假 (下面處理)
        add(dt.date(y, m, dd), nm)
    if dt.date(y + 1, 1, 1).weekday() == 5:
        out[dt.date(y, 12, 31).isoformat()] = "元旦 (補假)"
    lny = _lunar("LNY", y)
    if lny:
        block = [lny + dt.timedelta(days=i) for i in range(-2, 3)]
        extra = sum(1 for d in block if d.weekday() >= 5)
        for d in block:
            if d.weekday() < 5:
                out[d.isoformat()] = "春節"
        d = block[-1]
        while extra:
            d += dt.timedelta(days=1)
            if d.weekday() < 5 and d.isoformat() not in out:
                out[d.isoformat()] = "春節 (補假)"
                extra -= 1
        d, k = block[0], 0
        while k < 2:
            d -= dt.timedelta(days=1)
            if d.weekday() < 5:
                out[d.isoformat()] = "春節 (僅結算交割、無交易)"
                k += 1
    qm = _lunar("QINGMING", y)
    if qm:
        cd = dt.date(y, 4, 4)
        if cd == qm:          # 兒童節與清明同日：前一日放假，逢週四則後一日
            cd = cd + dt.timedelta(days=1) if cd.weekday() == 3 else cd - dt.timedelta(days=1)
        add(qm, "清明節")
        add(cd, "兒童節")
    for kind, nm in (("DRAGON", "端午節"), ("MIDAUTUMN", "中秋節")):
        d = _lunar(kind, y)
        if d:
            add(d, nm)
    return {k: out[k] for k in sorted(out)}


def _nth_weekday(y: int, m: int, wd: int, n: int) -> dt.date:
    d = dt.date(y, m, 1)
    d += dt.timedelta(days=(wd - d.weekday()) % 7)
    return d + dt.timedelta(days=7 * (n - 1))


def _last_weekday(y: int, m: int, wd: int) -> dt.date:
    d = (dt.date(y, m + 1, 1) if m < 12 else dt.date(y + 1, 1, 1)) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - wd) % 7)


def _easter(y: int) -> dt.date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return dt.date(y, month, day)


NYSE_SPECIAL = {"2001-09-11": "911", "2001-09-12": "911", "2001-09-13": "911", "2001-09-14": "911", "2004-06-11": "雷根國葬",
                "2007-01-02": "福特國葬", "2012-10-29": "颶風 Sandy", "2012-10-30": "颶風 Sandy", "2018-12-05": "老布希國葬",
                "2025-01-09": "卡特國葬"}


def nyse_holidays(y: int) -> dict[str, str]:
    """NYSE 休市日規則 (ISO → 名稱)。元旦逢週六不補 (12-31 照常交易)；其餘逢週六提前週五、逢週日順延週一。"""
    out: dict[str, str] = {}

    def obs(d: dt.date) -> dt.date:
        return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d
    nd = dt.date(y, 1, 1)
    if nd.weekday() != 5:
        out[obs(nd).isoformat()] = "元旦"
    if y >= 1998:
        out[_nth_weekday(y, 1, 0, 3).isoformat()] = "馬丁路德金恩紀念日"
    out[_nth_weekday(y, 2, 0, 3).isoformat()] = "總統日"
    out[(_easter(y) - dt.timedelta(days=2)).isoformat()] = "耶穌受難日"
    out[_last_weekday(y, 5, 0).isoformat()] = "陣亡將士紀念日"
    if y >= 2022:
        out[obs(dt.date(y, 6, 19)).isoformat()] = "六月節"
    out[obs(dt.date(y, 7, 4)).isoformat()] = "獨立紀念日"
    out[_nth_weekday(y, 9, 0, 1).isoformat()] = "勞動節"
    out[_nth_weekday(y, 11, 3, 4).isoformat()] = "感恩節"
    out[obs(dt.date(y, 12, 25)).isoformat()] = "聖誕節"
    for k, v in NYSE_SPECIAL.items():
        if k.startswith(str(y)):
            out[k] = v
    return {k: out[k] for k in sorted(out) if k.startswith(str(y))}


# ------------------------------------------------------------ 台股交易日 (TWSE API 優先，失敗退回規則並標 estimated)
_TW_HOL: dict[int, tuple[set[str], str]] = {}
_TW_FAIL: dict[int, float] = {}


def tw_holidays(y: int) -> tuple[set[str], str]:
    """(休市日集合, 來源 'api' | 'rule')。API 失敗 (尚未公布、402、逾時) → 規則估計；1 小時內不重抓失敗年度。"""
    if y in _TW_HOL:
        return _TW_HOL[y]
    if time.time() - _TW_FAIL.get(y, 0) > 3600:
        try:
            from ..sources import twse
            h = set(twse.holidays(y))
            if h:
                _TW_HOL[y] = (h, "api")
                return _TW_HOL[y]
            raise RuntimeError("empty")
        except Exception as e:  # noqa: BLE001
            log.info("TWSE holidays %s unavailable → rule estimate (%s)", y, e)
            _TW_FAIL[y] = time.time()
    return set(tw_rule_holidays(y)), "rule"


def reset_calendar_cache() -> None:
    _TW_HOL.clear()
    _TW_FAIL.clear()
    _US_ACTUAL.clear()
    _MEMO.clear()


def is_tw_trading(d) -> bool:
    d = _d(d)
    return d.weekday() < 5 and d.isoformat() not in tw_holidays(d.year)[0]


def next_td(d, n: int = 1) -> str:
    d = _d(d)
    k = 0
    while k < n:
        d += dt.timedelta(days=1)
        if is_tw_trading(d):
            k += 1
    return d.isoformat()


def prev_td(d) -> str:
    d = _d(d) - dt.timedelta(days=1)
    while not is_tw_trading(d):
        d -= dt.timedelta(days=1)
    return d.isoformat()


def tw_sessions(start, end) -> list[str]:
    """[start, end] 內的台股交易日。"""
    d, e, out = _d(start), _d(end), []
    while d <= e:
        if is_tw_trading(d):
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def closed_weekdays_between(a, b) -> int:
    """a、b 之間 (不含兩端) 台股休市的平日數 (a、b 為相鄰交易日時 = 休市長度)。"""
    d, e, n = _d(a) + dt.timedelta(days=1), _d(b), 0
    while d < e:
        if d.weekday() < 5 and not is_tw_trading(d):
            n += 1
        d += dt.timedelta(days=1)
    return n


# ------------------------------------------------------------ 美股交易日 (NYSE 規則；過去日期可用 ^GSPC 實際交易日覆寫)
_US_ACTUAL: dict[str, object] = {}


def set_us_actual(dates) -> None:
    """注入實際美股交易日 (例如 ^GSPC 歷史)；範圍內以實際為準，範圍外用規則。"""
    ds = sorted({str(x)[:10] for x in dates})
    if ds:
        _US_ACTUAL["set"], _US_ACTUAL["first"], _US_ACTUAL["last"] = set(ds), ds[0], ds[-1]


def _load_us_actual() -> None:
    if "tried" in _US_ACTUAL:
        return
    _US_ACTUAL["tried"] = True
    try:
        from ..sources import global_markets
        g = global_markets.history("^GSPC", "20y")        # 與 aligned_features 同一快取 key
        if g is not None and len(g):
            ds = g["date"].astype(str).str[:10]
            ds = ds[ds < dt.date.today().isoformat()]      # 今天的 bar 可能進行中 → 只信過去
            set_us_actual(ds)
    except Exception as e:  # noqa: BLE001
        log.debug("^GSPC actual sessions unavailable: %s", e)


def is_us_session(d) -> bool:
    d = _d(d)
    s = d.isoformat()
    if _US_ACTUAL.get("set") and _US_ACTUAL["first"] <= s <= _US_ACTUAL["last"]:
        return s in _US_ACTUAL["set"]
    return d.weekday() < 5 and s not in nyse_holidays(d.year)


def us_sessions(start, end_exclusive) -> list[str]:
    d, e, out = _d(start), _d(end_exclusive), []
    while d < e:
        if is_us_session(d):
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def n_us_between(prev_td_, td) -> int:
    """上一台股交易日 prev_td 收盤後、td 開盤前已收盤的美股交易日數 = #{美股交易日 U : prev_td ≤ U < td}
    (美股 U 日收盤在台北 U+1 清晨 04:00~05:00，晚於 prev_td 13:30、早於 td 09:00 ⇔ prev_td ≤ U ≤ td−1)。"""
    return len(us_sessions(prev_td_, td))


def n_us_uncovered(td, night_meta: dict | None = None, prev: str | None = None) -> int:
    """扣掉被「對齊且完整」的台指期夜盤涵蓋的美股日：夜盤只在台股交易日晚上開 → 最多涵蓋 prev_td 當晚那一個美股交易日。"""
    prev = prev or prev_td(td)
    n = n_us_between(prev, td)
    if night_meta and night_meta.get("used") and is_us_session(prev):
        n -= 1
    return max(0, n)


def _lny_in(a, b) -> bool:
    """(a, b) 之間 (不含兩端) 是否含農曆春節初一。"""
    a, b = _d(a), _d(b)
    for y in {a.year, b.year}:
        l_ = _lunar("LNY", y)
        if l_ and a < l_ < b:
            return True
    return False


def _hol_names(a, b) -> list[str]:
    """(a, b) 之間的台股休市日名稱 (依規則表命名；API 年度只給日期)。"""
    a, b = _d(a), _d(b)
    names: list[str] = []
    d = a + dt.timedelta(days=1)
    while d < b:
        if d.weekday() < 5 and not is_tw_trading(d):
            nm = tw_rule_holidays(d.year).get(d.isoformat()) or "休市"
            nm = nm.replace(" (補假)", "").replace(" (僅結算交割、無交易)", "")
            if nm not in names:
                names.append(nm)
        d += dt.timedelta(days=1)
    return names


# ============================================================ 內建事件表 (手動抄錄；缺年份 → 規則估計並標 estimated)
FOMC = {  # 決策日 (美東)；True = 附經濟預測 SEP。來源 federalreserve.gov fomccalendars (2026-09-27 抄錄；2027 暫定)
    "2026-01-28": False, "2026-03-18": True, "2026-04-29": False, "2026-06-17": True, "2026-07-29": False, "2026-09-16": True,
    "2026-10-28": False, "2026-12-09": True,
    "2027-01-27": False, "2027-03-17": True, "2027-04-28": False, "2027-06-09": True, "2027-07-28": False, "2027-09-15": True,
    "2027-10-27": False, "2027-12-08": True,
}
FOMC_COVER_UNTIL = "2027-12-31"
CPI = {  # 公布日 (美東 08:30) → 參考月份。來源 bls.gov CPI schedule (2026)
    "2026-01-13": "2025-12", "2026-02-13": "2026-01", "2026-03-11": "2026-02", "2026-04-10": "2026-03", "2026-05-12": "2026-04",
    "2026-06-10": "2026-05", "2026-07-14": "2026-06", "2026-08-12": "2026-07", "2026-09-11": "2026-08", "2026-10-14": "2026-09",
    "2026-11-10": "2026-10", "2026-12-10": "2026-11",
}
NFP = {"2026-10-02": "2026-09", "2026-11-06": "2026-10", "2026-12-04": "2026-11"}   # bls.gov empsit schedule
CBC = {"2026-03-19": "scheduled", "2026-06-18": "scheduled", "2026-09-17": "scheduled", "2026-12-17": "scheduled"}  # cbc.gov.tw 2026 行事曆 (約 16:00 公布)
TSMC_CALLS = {  # 台北日期 14:00；confirmed = TSMC IR 已公告
    "2026-10-15": {"q": "2026 Q3", "confidence": "confirmed", "source": "TSMC IR financial calendar / teleconference (2026-09-27 查核：10-15 14:00~15:30 台北)"},
}
ELECTIONS = [
    {"date": "2026-11-28", "event_type": "tw_local_election", "title_zh": "九合一地方選舉 (週六投票)", "confidence": "confirmed", "source": "中選會",
     "ledger_tag": "tw_local_2026"},
    {"date": "2026-11-03", "event_type": "us_midterm", "title_zh": "美國期中選舉", "confidence": "scheduled", "source": "選舉日規則 (11 月第一個週一後的週二)"},
    {"date": "2028-11-07", "event_type": "us_presidential", "title_zh": "美國總統大選", "confidence": "scheduled", "source": "選舉日規則"},
]


def _first_thu_after_12(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 13)
    return d + dt.timedelta(days=(3 - d.weekday()) % 7)


def tsmc_calls(start: dt.date, end: dt.date) -> list[dict]:
    """內建確認表 + 規則估計 (1/4/7/10 月 12 日後首個週四，EDGAR 6-K 規則) → estimated，乘數不啟用。"""
    out = {k: dict(v, date=k) for k, v in TSMC_CALLS.items()}
    for y in range(start.year, end.year + 1):
        for m, q in ((1, f"{y - 1} Q4"), (4, f"{y} Q1"), (7, f"{y} Q2"), (10, f"{y} Q3")):
            d = _first_thu_after_12(y, m)
            if not any(v["q"] == q for v in out.values()):
                out[d.isoformat()] = {"q": q, "confidence": "estimated", "source": "規則：1/4/7/10 月 12 日後首個週四；待 TSMC IR 確認", "date": d.isoformat()}
    return [v for k, v in sorted(out.items()) if start <= _d(k) <= end]


def tsmc_2330_k1(issue_date: str, now: dt.datetime | None = None, P: dict | None = None) -> dict:
    """操盤台 2330 k=1 帶的法說乘數：只在『法說日期經 IR 確認』且帶是『法說當日收盤發布』(ADR 收盤前) 時 ×λ。"""
    P = P or load_params()
    rule = (P.get("band_rules") or {}).get("tsmc_call_2330_k1") or {}
    lam = float(rule.get("lambda") or 1.0)
    now = now or dt.datetime.now(config.TZ)
    for c in tsmc_calls(_d(issue_date) - dt.timedelta(days=1), _d(issue_date) + dt.timedelta(days=1)):
        if c["date"] != str(issue_date)[:10]:
            continue
        info = {"call_date": c["date"], "q": c["q"], "date_confidence": c["confidence"], "lambda": lam, "grade": rule.get("grade")}
        if not rule.get("enabled"):
            return {**info, "factor": 1.0, "applied": False, "why": "規則停用"}
        if c["confidence"] != "confirmed":
            return {**info, "factor": 1.0, "applied": False, "why": "法說日期為估計值，乘數不啟用"}
        adr_close = dt.datetime.combine(_d(c["date"]) + dt.timedelta(days=1), dt.time(5, 30), tzinfo=config.TZ)
        if now >= adr_close:
            return {**info, "factor": 1.0, "applied": False, "why": "ADR 已收盤，改看 ADR 跳空估計，乘數不套"}
        return {**info, "factor": lam, "applied": True, "why": f"台積電法說當日收盤發布：2330 1 日帶 ×{lam:.1f} (證據中等，BH 0.20)"}
    return {"factor": 1.0, "applied": False}


def _nfp_rule(y: int, m: int) -> dt.date:
    """BLS 就業報告：參考週 (含上月 12 日的週日~週六) 結束後第三個週五；遇 NYSE 假日延一週 (例：2027-01-01 → 01-08)。"""
    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    d12 = dt.date(py, pm, 12)
    sat = d12 + dt.timedelta(days=(5 - d12.weekday()) % 7)
    d = sat + dt.timedelta(days=6 + 14)
    while d.isoformat() in nyse_holidays(d.year):
        d += dt.timedelta(days=7)
    return d


def _cpi_rule(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 12)                     # 10~15 日的週中 (取 12 日起第一個週二~週四)
    while d.weekday() not in (1, 2, 3):
        d += dt.timedelta(days=1)
    return d


def _cbc_rule(y: int, m: int) -> dt.date:
    return _nth_weekday(y, m, 3, 3)


def _msci_rule(y: int, m: int) -> dt.date:
    d = (dt.date(y, m + 1, 1) if m < 12 else dt.date(y + 1, 1, 1)) - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


# ============================================================ 行事曆 (合併：API > 內建表 > 規則)
def _react(date_local: str) -> str:
    """美國事件 / 台灣收盤後事件 / 週末投票 → 下一個台北交易日。"""
    return next_td(date_local)


def refresh_calendar(today=None, write: bool = True, horizon: int | None = None, path: Path | str | None = None, P: dict | None = None) -> dict:
    """合併未來事件 (TWSE API + 內建表 + 規則估計)，每筆帶 date_confidence；只在內容改變時寫 event_calendar.json。"""
    P = P or load_params()
    today = _d(today or dt.datetime.now(config.TZ).date())
    horizon = int(horizon or P.get("horizon_days") or 120)
    end = today + dt.timedelta(days=horizon + 40)           # 多抓 40 天 (春節常落在視野外幾天)，超出者標 beyond_horizon
    _load_us_actual()
    srcs = {y: tw_holidays(y)[1] for y in range(today.year, end.year + 2)}
    conf_tw = lambda d: "confirmed" if srcs.get(_d(d).year) == "api" else "estimated"   # noqa: E731
    src_tw = lambda d: "TWSE holidaySchedule API" if srcs.get(_d(d).year) == "api" else f"{_d(d).year} TWSE 行事曆未公布；平日規則 + 農曆表"   # noqa: E731
    ev: list[dict] = []
    sess = tw_sessions(today, end)
    for td in sess:
        pv = prev_td(td)
        nus = n_us_between(pv, td)
        cwd = closed_weekdays_between(pv, td)
        conf = "confirmed" if conf_tw(pv) == "confirmed" and conf_tw(td) == "confirmed" else "estimated"
        if cwd >= 1:
            lny = _lny_in(pv, td)
            names = "、".join(_hol_names(pv, td))
            f = math.sqrt(max(1, nus))
            ev.append({"date": (_d(pv) + dt.timedelta(days=1)).isoformat(), "tw_session": td, "prev_td": pv,
                       "event_type": "lny_reopen" if lny else "post_closure_session",
                       "title_zh": (f"農曆年開紅盤 (休 {cwd} 個平日，含 {nus} 個美股交易日)" if lny else
                                    f"{names}連假後復市 (休 {cwd} 個平日，含 {nus} 個美股交易日)"),
                       "date_confidence": conf, "source": src_tw(td), "n_us": nus, "closed_weekdays": cwd, "range_k1_factor": round(f, 3)})
        if nus == 0:
            us_names = [nyse_holidays(_d(u).year).get(u) for u in _weekdays(pv, td) if not is_us_session(u)]
            us_names = [x for x in us_names if x]
            ev.append({"date": (_d(td) - dt.timedelta(days=1)).isoformat(), "tw_session": td, "prev_td": pv, "event_type": "us_holiday_no_session",
                       "title_zh": f"前一晚美股休市 ({'、'.join(us_names) or '美股假日'})", "date_confidence": "scheduled", "source": "NYSE 規則",
                       "n_us": 0, "closed_weekdays": cwd})
        nx = next_td(td)
        cwd_after = closed_weekdays_between(td, nx)
        if cwd_after >= 1:
            lny = _lny_in(td, nx)
            names = "、".join(_hol_names(td, nx))
            ev.append({"date": td, "tw_session": td, "event_type": "lny_fengguan" if lny else "pre_holiday_session",
                       "title_zh": "農曆年封關日" if lny else f"{names}連假前最後交易日",
                       "date_confidence": conf_tw(nx), "source": src_tw(nx), "n_us": nus, "next_td": nx})
    # 年底最後 5 日 / 年度最後交易日 / 年度首個交易日
    for y in range(today.year, end.year + 1):
        yr = tw_sessions(dt.date(y, 12, 1), dt.date(y, 12, 31))
        if len(yr) >= 5:
            w = yr[-5:]
            if today <= _d(w[-1]) and _d(w[0]) <= end:
                ev.append({"date": w[0], "tw_session": w[0], "event_type": "tw_yearend_last5", "title_zh": f"台股年底最後 5 個交易日 ({w[0][5:]}~{w[-1][5:]})",
                           "date_confidence": conf_tw(w[0]), "source": src_tw(w[0]), "window": w})
            if today <= _d(w[-1]) <= end:
                ev.append({"date": w[-1], "tw_session": w[-1], "event_type": "tw_yearend_last_day", "title_zh": "台股年度最後交易日",
                           "date_confidence": conf_tw(w[-1]), "source": src_tw(w[-1])})
        fy = next_td(dt.date(y, 12, 31))
        if today <= _d(fy) <= end:
            ev.append({"date": fy, "tw_session": fy, "event_type": "jan_first", "title_zh": "新年度首個交易日",
                       "date_confidence": conf_tw(fy), "source": src_tw(fy)})
    # 總經 (美國事件 → 下一個台北交易日)
    fomc_cov = True
    for d, sep in FOMC.items():
        if today - dt.timedelta(days=1) <= _d(d) <= end:
            ev.append({"date": d, "tw_session": _react(d), "event_type": "fomc", "title_zh": "FOMC 利率決策" + (" + 經濟預測 (SEP)" if sep else ""),
                       "date_confidence": "scheduled", "source": "federalreserve.gov fomccalendars" + (" (暫定)" if d >= "2027-01-01" else "")})
    if end.isoformat() > FOMC_COVER_UNTIL:
        fomc_cov = False
    months = pd.period_range(pd.Period(today, "M"), pd.Period(end, "M"), freq="M")
    for per in months:
        y, m = per.year, per.month
        ref = (per - 1).strftime("%Y-%m")
        cpi_d = next((k for k, v in CPI.items() if v == ref), None)
        cpi_conf = "scheduled" if cpi_d else "estimated"
        cpi_d = cpi_d or _cpi_rule(y, m).isoformat()
        nfp_d = next((k for k, v in NFP.items() if v == ref), None)
        nfp_conf = "scheduled" if nfp_d else "estimated"
        nfp_d = nfp_d or _nfp_rule(y, m).isoformat()
        for d, typ, t, conf, src in ((cpi_d, "us_cpi", f"美國 {per.month - 1 or 12} 月 CPI", cpi_conf, "bls.gov CPI schedule" if cpi_conf == "scheduled" else "規則估計：月 10~15 日週中；待 BLS 公告"),
                                     (nfp_d, "us_nfp", f"美國 {per.month - 1 or 12} 月非農就業", nfp_conf, "bls.gov empsit schedule" if nfp_conf == "scheduled" else "規則估計：月初第一個週五；待 BLS 公告")):
            if today - dt.timedelta(days=1) <= _d(d) <= end:
                ev.append({"date": d, "tw_session": _react(d), "event_type": typ, "title_zh": t + (" (估計)" if conf == "estimated" else ""),
                           "date_confidence": conf, "source": src})
        if m in (3, 6, 9, 12):
            d = next((k for k in CBC if k.startswith(f"{y}-{m:02d}")), None)
            conf = "scheduled" if d else "estimated"
            d = d or _cbc_rule(y, m).isoformat()
            if today - dt.timedelta(days=1) <= _d(d) <= end:
                ev.append({"date": d, "tw_session": _react(d), "event_type": "tw_cbc", "title_zh": "央行理監事會 (約 16:00 公布)" + (" (估計)" if conf == "estimated" else ""),
                           "date_confidence": conf, "source": "cbc.gov.tw 年度行事曆" if conf == "scheduled" else "規則估計：季末月第三個週四"})
        if m in (2, 5, 8, 11):
            d = _msci_rule(y, m).isoformat()
            if today <= _d(d) <= end and is_tw_trading(d):
                ev.append({"date": d, "tw_session": d, "event_type": "msci_rebalance", "title_zh": "MSCI 季度調整生效 (規則估計，日期待 MSCI 公告)",
                           "date_confidence": "low", "source": "規則：2/5/8/11 月最後交易日；11 月遇感恩節可能提前"})
    for c in tsmc_calls(today - dt.timedelta(days=1), end):
        ev.append({"date": c["date"], "tw_session": _react(c["date"]), "issue_date": c["date"], "event_type": "tsmc_call",
                   "title_zh": f"台積電 {c['q']} 法說會 (14:00 台北)" + (" (估計)" if c["confidence"] != "confirmed" else ""),
                   "date_confidence": c["confidence"], "source": c["source"]})
    for e in ELECTIONS:
        if not (today - dt.timedelta(days=1) <= _d(e["date"]) <= end):
            continue
        t0 = _react(e["date"])
        row = {"date": e["date"], "tw_session": t0, "event_type": e["event_type"], "date_confidence": e["confidence"], "source": e["source"]}
        if e["event_type"] == "tw_local_election":
            tm1 = prev_td(t0)
            ev.append({**row, "tw_session": tm1, "role": "T-1", "ledger_tag": e["ledger_tag"] + "_T-1",
                       "title_zh": f"{e['title_zh']}：選前最後交易日 (T-1)，當日收盤發布的機率帶涵蓋投票週末 (寫入帳本事後檢查)"})
            ev.append({**row, "role": "T0", "ledger_tag": e["ledger_tag"] + "_T0", "title_zh": f"{e['title_zh']}；T0 = {t0[5:]} 投票後首個交易日"})
        else:
            ev.append({**row, "title_zh": f"{e['title_zh']} (台北反應日 {t0[5:]})"})
    lim = (today + dt.timedelta(days=horizon)).isoformat()
    for e in ev:
        e["beyond_horizon"] = e["tw_session"] > lim
    ev = [e for e in ev if e["tw_session"] >= today.isoformat()]
    ev.sort(key=lambda e: (e["tw_session"], _ORDER.get(e["event_type"], 50), e["date"]))
    rule_years = sorted(y for y, s in srcs.items() if s != "api")
    api_years = sorted(y for y, s in srcs.items() if s == "api")
    stale_years = [y for y in rule_years if dt.date(y, 1, 1) <= today + dt.timedelta(days=horizon)]
    health = {"tw_confirmed_until": f"{max(api_years)}-12-31" if api_years else None,
              "tw_sources": {str(y): s for y, s in srcs.items()},
              "fomc_covered_until": FOMC_COVER_UNTIL, "fomc_covers_horizon": fomc_cov,
              "stale": bool(stale_years) or not fomc_cov,
              "fallback_used": [f"{y} TW 規則估計" for y in stale_years] + ([] if fomc_cov else ["FOMC 表未涵蓋視野"]),
              "us_calendar": "NYSE 規則" + (" + ^GSPC 實際交易日 (過去)" if _US_ACTUAL.get("set") else "")}
    cal = {"schema": "event_calendar/v1", "asof": today.isoformat(), "params_version": P.get("version"), "horizon_days": horizon,
           "health": health, "events": ev}
    if write:
        _write_if_changed(Path(path or CAL_PATH), cal)
    return cal


_ORDER = {"post_closure_session": 0, "lny_reopen": 0, "us_holiday_no_session": 1, "tw_local_election": 2, "tsmc_call": 3, "fomc": 4, "us_cpi": 5,
          "us_nfp": 6, "tw_cbc": 7, "us_midterm": 8, "msci_rebalance": 9, "pre_holiday_session": 10, "lny_fengguan": 10, "tw_yearend_last5": 11,
          "tw_yearend_last_day": 12, "jan_first": 13}


def _weekdays(a, b) -> list[str]:
    d, e, out = _d(a), _d(b), []
    while d < e:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _write_if_changed(p: Path, cal: dict) -> None:
    try:
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if {k: v for k, v in old.items() if k != "generated"} == cal:
                return
        body = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), **cal}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.debug("event_calendar write: %s", e)


# ============================================================ 每個交易日的因子 / 標籤
def session_tags(td: str, cal: dict | None = None) -> list[str]:
    """td 這個台股交易日的事件標籤 (休市後首日、前晚美股休市、連假前一日、年底週、法說日、選舉 T-1/T0…)。"""
    tags: list[str] = []
    try:
        pv = prev_td(td)
        if closed_weekdays_between(pv, td) >= 1:
            tags.append("lny_reopen" if _lny_in(pv, td) else "post_closure_session")
        if n_us_between(pv, td) == 0:
            tags.append("us_holiday_no_session")
        nx = next_td(td)
        if closed_weekdays_between(td, nx) >= 1:
            tags.append("lny_fengguan" if _lny_in(td, nx) else "pre_holiday_session")
        elif closed_weekdays_between(nx, next_td(nx)) >= 1:
            tags.append("pre_pre_holiday")
        yr = tw_sessions(dt.date(_d(td).year, 12, 1), dt.date(_d(td).year, 12, 31))
        if td in yr[-5:]:
            tags.append("tw_yearend_last5")
    except Exception as e:  # noqa: BLE001
        log.debug("session_tags %s: %s", td, e)
    for e in (cal or {}).get("events") or []:
        if e.get("tw_session") == td and e["event_type"] not in tags and e["event_type"] not in ("post_closure_session", "lny_reopen", "us_holiday_no_session",
                                                                                                 "pre_holiday_session", "lny_fengguan", "tw_yearend_last5"):
            tags.append(e["event_type"] + (f"_{e['role']}" if e.get("role") else ""))
        if e.get("event_type") == "tsmc_call" and e.get("issue_date") == td:
            tags.append("tsmc_call_issue_day")
    return tags


def session_factor(td: str, night_meta: dict | None = None, prev: str | None = None, P: dict | None = None, cal: dict | None = None) -> dict:
    """{'range_k1': √max(1,n_US) 或 1, 'iv': 1.0, 'n_us', 'n_us_uncovered', 'range_mode', 'tags', 'why'}。
    規則 (design §3)：驗證用的帶不含任何美股資訊 → 只要 n_US_total ≥ 2 (必有未被夜盤涵蓋的美股日) 就退回 base 模式乘 √n_US_total
    ('base_event')；n_US ≤ 1 維持原本 (夜盤照舊)；n_US = 0 → 1.0 (收盤波動不縮)。"""
    P = P or load_params()
    prev = prev or prev_td(td)
    n = n_us_between(prev, td)
    used = bool((night_meta or {}).get("used"))
    unc = n_us_uncovered(td, night_meta, prev)
    rule = (P.get("band_rules") or {}).get("post_closure_sqrt_nus") or {}
    if n >= 2 and rule.get("enabled"):
        f, mode = math.sqrt(n), "base_event"
    else:
        f, mode = 1.0, ("night" if used else "base")
    why = (f"含休市寬度 ×{f:.2f} (休市期間 {n} 個美股交易日)" if f > 1 else
           "前一晚美股休市：跳空幅度通常較小 (約 ×0.76)，機率帶寬度不變" if n == 0 else None)
    return {"td": td, "prev_td": prev, "n_us": n, "n_us_uncovered": unc, "closed_weekdays": closed_weekdays_between(prev, td),
            "range_k1": round(f, 4), "iv": 1.0, "range_mode": mode, "night_used": used, "tags": session_tags(td, cal), "why": why}


# ============================================================ IV 預期
def iv_expect(prev: str, td: str, opt_hist: list[dict] | None = None, P: dict | None = None, cal: dict | None = None) -> dict | None:
    P = P or load_params()
    ie = P.get("iv_expectations") or {}
    out: dict = {}
    if closed_weekdays_between(prev, td) >= 1:
        cm = ie.get("closure_mechanical") or {}
        lny = _lny_in(prev, td)
        out.update({"ivk5_dlog": (cm.get("ivk5_dlog") or {}).get("LNY" if lny else "any_closure"),
                    "iv21_dlog": (cm.get("iv21_dlog") or {}).get("LNY" if lny else "any_closure"), "kind": "lny" if lny else "closure",
                    "grade": cm.get("grade"), "note": "IV 下降屬交易日年化的機械效果 (跨休市)；IV 變化特徵應看「實際 − 預期」"})
    t0 = [e for e in (cal or {}).get("events") or [] if e.get("event_type") == "tw_local_election" and e.get("role") == "T0" and e.get("tw_session") == td]
    if t0:
        out["election_iv5"] = election_iv5(prev, opt_hist, P)
    return out or None


def election_iv5(prev: str, opt_hist: list[dict] | None = None, P: dict | None = None) -> dict:
    """台灣選舉 T0 (投票後首個交易日) 的 iv5 預期：平均 −13% (n=8，6/8 下降)；T-1 iv5 ÷ 前 20 日均 ≥ 1.2 的 2 次 (−21%、−40%)
    降幅較大，< 1.2 時幅度不定 (約 −25%~+16%)。數字全部取自 event_params.iv_expectations.tw_election_T0_iv5.cases。"""
    P = P or load_params()
    el = (P.get("iv_expectations") or {}).get("tw_election_T0_iv5") or {}
    thr = float(el.get("ratio_threshold") or 1.2)
    cases = el.get("cases") or []
    hi = sorted((c["pct"] for c in cases if c.get("t1_ratio") is not None and c["t1_ratio"] >= thr), reverse=True)
    lo = [c["pct"] for c in cases if c.get("t1_ratio") is not None and c["t1_ratio"] < thr]
    mean = el.get("mean_pct")
    if mean is None and el.get("iv5_dlog_mean") is not None:
        mean = round((math.exp(float(el["iv5_dlog_mean"])) - 1) * 100)
    n = el.get("n")
    neg = el.get("neg")
    head = f"選後首日 iv5 歷史平均下降約 {abs(mean or 0):.0f}% (n={n}" + (f"，{neg} 下降)" if neg else ")")
    hi_txt = "、".join(f"{abs(x):.0f}%" for x in hi)
    lo_rng = (f"約 {min(lo):+.0f}%~{max(lo):+.0f}%" if lo else "")
    ratio = None
    try:
        h = pd.DataFrame(opt_hist or [])
        if len(h) and "iv5" in h:
            h = h[h["date"].astype(str) <= prev].dropna(subset=["iv5"])
            if len(h) >= 21 and str(h["date"].iloc[-1]) == prev:
                ratio = float(h["iv5"].iloc[-1] / h["iv5"].iloc[-21:-1].mean())     # T-1 ÷ (T-21..T-2) 平均，與研究同定義
    except Exception:  # noqa: BLE001
        ratio = None
    big = ratio is not None and ratio >= thr
    if big:
        text = f"{head}；本次選前 iv5 溢價 {ratio:.2f} 倍 (≥{thr:g})，歷史上同樣 ≥{thr:g} 倍的 {len(hi)} 次分別下降約 {hi_txt}"
    elif ratio is not None:
        text = f"{head}；本次選前 iv5 溢價 {ratio:.2f} 倍 (<{thr:g})，歷史上此情況幅度不定 ({lo_rng}，n={len(lo)})"
    else:
        text = f"{head}；幅度視 T-1 收盤 iv5 溢價而定 (≥{thr:g} 倍的 {len(hi)} 次下降約 {hi_txt}；<{thr:g} 倍時幅度不定)"
    return {"expect_pct_range": ([min(hi), max(hi)] if big and hi else None), "mean_pct": mean, "neg": neg,
            "t1_ratio": None if ratio is None else round(ratio, 2), "ratio_threshold": thr,
            "condition": f"T-1 iv5 ÷ 前 20 日均 ≥ {thr:g} 時歷史上降幅較大 (" + "、".join(f"{x:+.0f}%" for x in hi) + ")",
            "cases_hi_pct": hi, "cases_lo_pct_range": ([min(lo), max(lo)] if lo else None),
            "grade": el.get("grade"), "n": n, "text": clean_text(text, P)}


# ============================================================ 休市後 0050 跳空預估卡
GAP_CLASS = {"lny": "gap_LNY", "ge2wd": "gap_ge2wd", "1wd": "gap_1wd"}


def _logret_by_date(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or len(df) == 0:
        return {}
    d = df[["date", "close"]].copy()
    d["date"] = d["date"].astype(str).str[:10]
    d = d.dropna().drop_duplicates("date", keep="last").sort_values("date")
    r = np.log(d["close"].astype(float) / d["close"].astype(float).shift(1))
    return {k: float(v) for k, v in zip(d["date"], r) if np.isfinite(v)}


def gap_beta(px0050: pd.DataFrame, xret: dict[str, float], before: str, exdiv_dates=(), win: int = 750, min_n: int = 250) -> tuple[float | None, int, float | None]:
    """β_gap：前 win 個『正常日』(前一台股日與當日間無休市平日、恰 1 個美股交易日) 的 0050 開盤跳空 log 對前一美股日 x 報酬
    無截距 OLS (研究 h3 Model A)。剔除除息日、開盤 = 前收 (缺開盤)、|跳空| > 15% (分割未還原)。回傳 (β, n, 殘差 sd%)。"""
    d = px0050[["date", "open", "close"]].copy()
    d["date"] = d["date"].astype(str).str[:10]
    d = d.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    d["pdate"], d["pclose"] = d["date"].shift(1), d["close"].shift(1)
    d = d[(d["date"] < before) & d["pdate"].notna()].tail(win * 2 + 200)
    exd = {str(x)[:10] for x in exdiv_dates or ()}
    rows = []
    for r in d.itertuples():
        if r.date in exd or not r.open or not r.pclose or float(r.open) == float(r.pclose):
            continue
        y = math.log(float(r.open) / float(r.pclose))
        if abs(y) > 0.15:
            continue
        pv, cur = _d(r.pdate), _d(r.date)
        if any(x.weekday() < 5 for x in (pv + dt.timedelta(days=i) for i in range(1, (cur - pv).days))):
            continue                                   # 中間有平日沒交易 = 休市後首日，不是正常日
        us = us_sessions(pv, cur)
        if len(us) != 1 or us[0] not in xret:
            continue
        rows.append((xret[us[0]], y))
    rows = rows[-win:]
    if len(rows) < min_n:
        return None, len(rows), None
    x, y = np.array(rows).T
    b = float((x * y).sum() / (x * x).sum())
    return b, len(rows), float(np.std(y - b * x) * 100)


def gap_forecast(td: str, prev: str | None = None, px0050: pd.DataFrame | None = None, sox: pd.DataFrame | None = None,
                 tsm: pd.DataFrame | None = None, exdiv_dates=(), sigma60_pct: float | None = None, now: dt.datetime | None = None,
                 P: dict | None = None) -> dict | None:
    """復市日 0050 跳空預估：β_gap × Σ log SOX (休市期間所有美股交易日；SOX 缺值改用 TSM ADR 與其自己的 β)。
    只在休市後首日 (中間有休市平日) 輸出；美股尚未全部收盤 → status 'pending_us'；0050 除息日 → 'exdiv_excluded'。
    前一晚美股休市 (n_US = 0) → 不預估，只給 ×0.76 說明。"""
    P = P or load_params()
    prev = prev or prev_td(td)
    gr = (P.get("gap_rules") or {})
    rule = gr.get("post_closure_gap_forecast") or {}
    cwd = closed_weekdays_between(prev, td)
    us = us_sessions(prev, td)
    if not us:
        nu = gr.get("no_us_session_gap") or {}
        return {"status": "no_us_session", "n_us": 0, "gap_size_factor": nu.get("gap_size_factor"), "grade": nu.get("grade"),
                "note": clean_text(f"前一晚美股休市：0050 開盤跳空幅度通常約平常 ×{nu.get('gap_size_factor')} (n={nu.get('n')})，全日波動不變", P)}
    if cwd < 1 or not rule.get("enabled"):
        return None
    lny = _lny_in(prev, td)
    cls = "lny" if lny else ("ge2wd" if cwd >= 2 else "1wd")
    oos = (rule.get("oos") or {}).get(GAP_CLASS[cls]) or {}
    out = {"td": td, "prev_td": prev, "n_us": len(us), "us_sessions": us, "closed_weekdays": cwd, "class": cls, "grade": rule.get("grade"),
           "r2_oos": oos.get("r2_sox"), "n_oos": oos.get("n_oos"), "sign_hit_oos": oos.get("sign_hit_sox", oos.get("sign_hit")), "beta_ref": (rule.get("beta_ref") or {}).get("gap_normal_day"),
           "formula": "0050 跳空 ≈ β × 休市期間費半累積 log 報酬 (β = 前 750 個正常日無截距 OLS)", "status": "pending_us", "est_0050_pct": None}
    if str(td) in {str(x)[:10] for x in exdiv_dates or ()}:
        out.update(status="exdiv_excluded", note="復市日為 0050 除息日，開盤跳空含除息，不預估")
        return out
    now = now or dt.datetime.now(config.TZ)
    last_close = dt.datetime.combine(_d(us[-1]) + dt.timedelta(days=1), dt.time(5, 30), tzinfo=config.TZ)
    sx, tx = _logret_by_date(sox), _logret_by_date(tsm)
    src, xr = ("SOX", sx) if all(u in sx for u in us) else ("TSM", tx) if all(u in tx for u in us) else (None, None)
    if now < last_close:                 # 最後一個美股日尚未收盤 (台北 U+1 05:30 前)：即使資料列存在也可能是盤中價
        src = None
    if src is None:
        have = [u for u in us if u in sx]
        out["sum_sox_pct_partial"] = round(sum(sx[u] for u in have) * 100, 2) if have else None
        out["n_us_closed"] = len(have)
        out["note"] = (f"待美股收盤 ({us[-1]} 美股收盤後、台北 {(_d(us[-1]) + dt.timedelta(days=1)).isoformat()[5:]} 清晨更新)" if now < last_close
                       else "待美股收盤資料更新")
        return out
    if px0050 is None or len(px0050) < 300:
        out.update(status="no_data", note="0050 價格資料不足，無法估 β")
        return out
    b, n, rsd = gap_beta(px0050, xr, td, exdiv_dates)
    if b is None:
        out.update(status="no_data", note=f"正常日樣本不足 ({n} < 250)")
        return out
    s = sum(xr[u] for u in us)
    est = b * s * 100
    rms = oos.get("resid_rms_sigma60")
    if isinstance(rms, dict):            # 殘差依預估來源 (SOX / TSM) 分開；舊格式 (單一數字) 照用
        rms = rms.get(src.lower())
    sl = src.lower()
    out.update(r2_oos=oos.get(f"r2_{sl}", out["r2_oos"]), sign_hit_oos=oos.get(f"sign_hit_{sl}", out["sign_hit_oos"]))
    out.update(status="ready", source=src, beta=round(b, 3), beta_n=n, sum_sox_pct=round(s * 100, 2) if src == "SOX" else None,
               sum_x_pct=round(s * 100, 2), est_0050_pct=round(est, 2),
               resid_sd_pct=round(rms * sigma60_pct, 2) if (rms and sigma60_pct) else None, resid_rms_sigma60=rms,
               note=clean_text(f"休市期間 {len(us)} 個美股交易日{'費半' if src == 'SOX' else '台積電 ADR'}累積 {s * 100:+.2f}% × β {b:.2f} → 0050 開盤跳空約 {est:+.2f}%"
                               + (f" (殘差約 ±{rms * sigma60_pct:.2f}%)" if rms and sigma60_pct else "") + "；歷史統計，非買賣訊號", P))
    return out


# ============================================================ 每筆事件的描述 (只用已驗證數字)
def _g(x: dict, *path, default=None):
    for p in path:
        if not isinstance(x, dict):
            return default
        x = x.get(p)
    return default if x is None else x


def _short(g: str | None) -> str:
    g = str(g or "無")
    for c in "強中弱無":
        if g.startswith(c):
            return c
    return "無"


def dlog_pct(dlog: float | None) -> float:
    """log 變化 → 百分比降幅 (正數)：(1 − e^dlog) × 100。例：−0.216 → 19.4 (不是 21.6)。"""
    return (1.0 - math.exp(float(dlog or 0.0))) * 100.0


def _band_status(rule: dict) -> str:
    """候選帶寬規則的狀態文字：驗收 B 已跑且未通過 → 明講未通過；未跑 → 待驗收。"""
    acc = (rule or {}).get("acceptance_B") or {}
    if acc and acc.get("pass") is False:
        why = str(acc.get("note") or "").split("→")[0].strip()
        return f"帶寬驗收 B 未通過 ({why})，維持關閉、只寫帳本" if why else "帶寬驗收 B 未通過，維持關閉、只寫帳本"
    if acc and acc.get("pass") is True:
        return "帶寬驗收 B 通過" + ("" if rule.get("enabled") else "，尚未啟用")
    return "尚未做帶寬驗收，預設關閉、只寫帳本"


def _texts(e: dict, P: dict) -> dict:
    t = e["event_type"]
    et = (P.get("event_types") or {}).get(t) or {}
    br, gr = P.get("band_rules") or {}, P.get("gap_rules") or {}
    n, vol, dir_ = None, NO_EFFECT, f"方向：{NO_EFFECT} ({UNVERIFIED})"
    band: dict = {"desk_iv": 1.0}
    ids: list[str] = []
    if t == "post_closure_session":
        cls = "ge2wd" if (e.get("closed_weekdays") or 0) >= 2 else "1wd"
        n = _g(et, "vol", "n", cls)
        f = e.get("range_k1_factor") or 1.0
        vol = (f"休市後首日 |報酬| 約平常 ×{_g(et, 'vol', 'abs_ret_x_sigma60', '1wd')} (休 1 個平日)~×{_g(et, 'vol', 'abs_ret_x_sigma60', 'ge2wd')} (休 ≥2 個平日)；"
               + (f"本次含 {e.get('n_us')} 個美股交易日 → 1 日實現波動帶寬度 ×{f:.2f}；0050 跳空 ≈ β × 休市期間費半累積 (樣本外 R² 0.62~0.68)"
                  if f > 1 else f"美股同步休市，本次只含 {e.get('n_us')} 個美股交易日 → 寬度不放寬"))
        dir_ = f"方向由休市期間美股漲跌決定，無日曆效應 ({UNVERIFIED})"
        band = {"range_levels_k1_factor": round(f, 3), "desk_iv": 1.0}
        ids = (br.get("post_closure_sqrt_nus") or {}).get("finding_ids") or []
    elif t == "lny_reopen":
        n = _g(et, "vol", "n")
        f = e.get("range_k1_factor") or 1.0
        vol = (f"開紅盤單日 |報酬| 約平常 ×{_g(et, 'vol', 'abs_ret_x')}、0050 跳空約 ×{_g(et, 'vol', 'gap0050_x')}，只持續一天；"
               f"含 {e.get('n_us')} 個美股交易日 → 1 日實現波動帶寬度 ×{f:.2f}；ivk5 預期機械下降約 {dlog_pct(_g(P, 'iv_expectations', 'closure_mechanical', 'ivk5_dlog', 'LNY', default=0)):.0f}%")
        dir_ = f"開紅盤漲跌幾乎完全來自休市期間美股 (扣除後 +0.04%，{UNVERIFIED})"
        band = {"range_levels_k1_factor": round(f, 3), "desk_iv": 1.0}
        ids = ["B:realised:LNY", "G:LNY:gap", "IV02"]
    elif t == "lny_fengguan":
        n = _g(et, "direction", "n")
        vol = f"封關日盤中區間約平常 ×{_g(et, 'vol', 'range_x')} (p {_g(et, 'vol', 'p')}，證據弱)；選擇權 k=1 帶上緣候選 ×{_g(br, 'fengguan_iv_k1_widen', 'candidate_factor')} 只寫帳本"
        dir_ = f"封關日多數收紅但平均僅 +{_g(et, 'direction', 'since2010_excess_pct')}%，低於來回成本 0.371% ({UNVERIFIED})"
        band = {"desk_iv": 1.0, "desk_iv_k1_candidate": _g(br, "fengguan_iv_k1_widen", "candidate_factor"), "enabled": False}
        ids = ["B:IV:k1:LNY"]
    elif t == "pre_holiday_session":
        n = _g(et, "vol", "n")
        vol = f"連假前一日盤中區間約平常 ×{_g(et, 'vol', 'range_x')} (n={n}, p {_g(et, 'vol', 'p')})；帶寬候選 ×{_g(br, 'pre_holiday_session_range', 'candidate_factor')}：{_band_status(br.get('pre_holiday_session_range'))}"
        band = {"candidate_range_factor": _g(br, "pre_holiday_session_range", "candidate_factor"), "enabled": False, "desk_iv": 1.0,
                "acceptance_B_pass": _g(br, "pre_holiday_session_range", "acceptance_B", "pass")}
        ids = ["TV08", "TV08b"]
    elif t == "tw_yearend_last5":
        n = _g(et, "vol", "n")
        vol = f"年底最後 5 日盤中區間約平常 ×{_g(et, 'vol', 'range_x')} (n={n})；帶寬候選 ×{_g(br, 'yearend_week_range', 'candidate_factor')}：{_band_status(br.get('yearend_week_range'))}"
        dir_ = f"2010 年後 16 年有 13 年上漲、平均超額約 +1%，未通過 Holm 校正 ({UNVERIFIED})"
        band = {"candidate_range_factor": _g(br, "yearend_week_range", "candidate_factor"), "enabled": False, "desk_iv": 1.0,
                "acceptance_B_pass": _g(br, "yearend_week_range", "acceptance_B", "pass")}
        ids = ["TV14", "TV05"]
    elif t == "us_holiday_no_session":
        n = _g(et, "vol", "n")
        vol = f"前一晚美股休市：0050 開盤跳空幅度約平常 ×{_g(et, 'vol', 'gap0050_x')}，全日波動不變 (×{_g(et, 'vol', 'cc_x')})，機率帶不縮"
        band = {"range_levels_k1_factor": 1.0, "gap_size_factor": _g(gr, "no_us_session_gap", "gap_size_factor"), "desk_iv": 1.0}
        ids = ["TV19", "TV16"]
    elif t == "fomc":
        n = _g(et, "vol", "tw_n")
        vol = (f"美股當晚區間約落在第 {_g(et, 'vol', 'spx_range_pctile', default=0) * 100:.0f} 百分位；台股隔日 |報酬| 約第 {_g(et, 'vol', 'tw_absret_pctile', default=0) * 100:.0f} 百分位，"
               "現有選擇權機率帶已足夠涵蓋，不調整")
        ids = ["band:iv:fomc"]
    elif t == "us_cpi":
        n = _g(et, "vol", "tw_n")
        vol = f"台股對 CPI {NO_EFFECT} (台股隔日 |報酬| 第 {_g(et, 'vol', 'tw_absret_pctile', default=0) * 100:.0f} 百分位，2021~23 通膨期也沒有)"
        ids = ["band:iv:cpi"]
    elif t == "us_nfp":
        n = _g(et, "vol", "tw_n")
        vol = f"美股當日區間約第 {_g(et, 'vol', 'spx_range_pctile', default=0) * 100:.0f} 百分位；台股週一 |報酬| 約第 {_g(et, 'vol', 'tw_absret_pctile', default=0) * 100:.0f} 百分位 (影響很小)"
        ids = ["band:iv:nfp"]
    elif t == "tw_cbc":
        n = _g(et, "vol", "n")
        vol = NO_EFFECT
        ids = ["band:iv:cbc"]
    elif t == "msci_rebalance":
        vol = f"{NO_EFFECT} (看似波動較大其實是月底效應)"
        ids = ["band:iv:msci"]
    elif t == "tsmc_call":
        n = _g(et, "vol", "n")
        ok = e.get("date_confidence") == "confirmed" and _g(br, "tsmc_call_2330_k1", "enabled")
        lam = _g(br, "tsmc_call_2330_k1", "lambda")
        vol = (f"台積電隔日 |報酬| 中位數約平常 ×{_g(et, 'vol', 'tw2330_median_ratio')}；0050 跳空約落在正常日第 {_g(et, 'vol', 'gap0050_pctile', default=0) * 100:.0f} 百分位；"
               + (f"操盤台 2330 的 1 日帶於 {e.get('issue_date', '')[5:]} 收盤發布時 ×{lam} (證據中等)" if ok else f"日期為估計，2330 帶乘數 ×{lam} 待 IR 確認後才啟用"))
        band = {"desk_2330_k1_lambda": lam, "enabled": bool(ok), "issued_on": f"{e.get('issue_date')} 收盤", "index": 1.0, "desk_iv": 1.0}
        ids = _g(br, "tsmc_call_2330_k1", "finding_ids", default=[])
    elif t == "tw_local_election":
        n = _g(et, "direction", "n")
        if e.get("role") == "T-1":
            vol = "當日收盤發布的機率帶涵蓋投票週末；選擇權帶乘數 1.0 (選舉帶測試全數未改善)，寫入帳本事後檢查"
        else:
            vol = (f"選後首日 5 日期 IV 平均下降約 13% (n=8)；實現波動 2010 後約 ×{_g(et, 'vol', 'exhv1_x_since2010')} (p {_g(et, 'vol', 'p_since2010')}，不顯著)，"
                   f"候選 ×{_g(br, 'tw_election_T0_hv', 'candidate_factor')} 預設關閉")
        dir_ = f"選前 20 日多上漲，但與同期 11 月季節性重疊、未過校正 ({UNVERIFIED})"
        band = {"desk_iv": 1.0, "ledger_tag": e.get("ledger_tag")}
        if e.get("role") == "T0":
            band.update(range_levels_candidate=_g(br, "tw_election_T0_hv", "candidate_factor"), enabled=False)
        ids = ["E1", "E2", "E15"]
    elif t == "us_midterm":
        n = _g(et, "vol", "n_taiex")
        vol = (f"反應日 S&P 區間約 ×{_g(et, 'vol', 'spx_range_x')} (p {_g(et, 'vol', 'spx_p')})、加權約 ×{_g(et, 'vol', 'taiex_range_x')} (p {_g(et, 'vol', 'taiex_p')})，"
               "均不顯著；機率帶維持不變，VIX 不一定下降")
        dir_ = f"期中選舉後的總統週期漲幅 1990 年後已不顯著 ({UNVERIFIED})"
        band = {"desk_iv": 1.0, "range_levels": 1.0}
        ids = ["USE-MID-RANGE"]
    elif t == "us_presidential":
        vol = "S&P 反應日區間約 1.3~1.5 倍 (候選，只套 S&P)；台股選擇權帶已定價"
    elif t in ("tw_yearend_last_day", "jan_first"):
        vol = NO_EFFECT + ("；首日波動偏大只是元旦休市 (已含在休市規則)" if t == "jan_first" else "")
    gv, gd = et.get("grade_vol"), et.get("grade_dir")
    if t == "tw_local_election" and e.get("role") == "T-1":
        gv = "無 (T-1 未檢定波動效應；「中」屬 T0 的 iv5 下降)"
    out = {"grade_vol": _short(gv), "grade_vol_detail": gv, "grade_dir": _short(gd), "n": n,
            "vol_text": clean_text(vol, P), "dir_text": clean_text(dir_, P), "notes_zh": clean_text(et.get("notes_zh") or "", P),
            "name_zh": et.get("name_zh"), "band": band, "finding_ids": ids,
            "stats": {k: et.get(k) for k in ("vol", "direction", "band") if et.get(k) is not None}}
    if et.get("grade_gap"):          # 跳空幅度的證據 (與全日波動分開；例：前一晚美股休市 → 波動 無、跳空 強)
        out["grade_gap"] = _short(et.get("grade_gap"))
    return out


def upcoming(cal: dict, since: str, P: dict | None = None, today: str | None = None) -> list[dict]:
    """tw_session ≥ since 的事件 (依日期排序)；days_to = 台股反應日距 today 的日曆天數。"""
    P = P or load_params()
    asof = today or since
    out = []
    for e in cal.get("events") or []:
        if e["tw_session"] < since:
            continue
        x = {k: e.get(k) for k in ("date", "tw_session", "event_type", "title_zh", "date_confidence", "source", "n_us", "closed_weekdays", "role",
                                    "ledger_tag", "beyond_horizon", "window", "issue_date") if e.get(k) is not None}
        x["title_zh"] = clean_text(x.get("title_zh"), P)
        x["days_to"] = (_d(e["tw_session"]) - _d(asof)).days
        if e.get("event_type") in ("post_closure_session", "lny_reopen"):
            x["range_k1_factor"] = e.get("range_k1_factor")
        x.update(_texts(e, P))
        cav = []
        if e["event_type"] == "tw_local_election" and e.get("role") == "T0":
            cav = ["MSCI 11 月季度調整生效日可能重疊 (待公告)", "前一交易日為美國感恩節後 (黑色星期五半日盤)"]
        if e["event_type"] == "msci_rebalance":
            cav = ["11 月生效日常因感恩節提前；日期待 MSCI 公告"]
        if cav:
            x["caveats"] = cav
        out.append(x)
    return out


# ============================================================ 組裝
_MEMO: dict = {}


def _sigma60(tw: pd.DataFrame | None, before: str) -> float | None:
    try:
        c = tw[tw["date"].astype(str) < before]["close"].astype(float)
        r = np.log(c / c.shift(1)).dropna().tail(60)
        return float(r.std() * 100) if len(r) >= 40 else None
    except Exception:  # noqa: BLE001
        return None


def _load_0050() -> tuple[pd.DataFrame | None, list[str]]:
    try:
        from ..analysis import chips
        from ..sources import finmind
        y = dt.date.today().year
        px = finmind.stock_price("0050", f"{y - 5}-01-01")
        px, _ = chips.adjust_splits(px, "0050")
        exd = []
        try:
            dv = finmind.fetch("TaiwanStockDividendResult", "0050", f"{y - 5}-01-01")
            exd = [str(x)[:10] for x in (dv["date"] if dv is not None and len(dv) else [])]
        except Exception:  # noqa: BLE001
            pass
        return px, exd
    except Exception as e:  # noqa: BLE001
        log.info("events 0050 prices unavailable: %s", e)
        return None, []


def build(today=None, ctx: dict | None = None, write_calendar: bool = True) -> dict:
    """desk.json["events"] 的完整內容 (§4.1)。ctx (皆可省略，省略時自行以快取載入)：
    last_td (最後一個有資料的台股交易日)、px0050 (date/open/close，已還原分割)、exdiv_0050 (除息日清單)、sox/tsm (美股日K)、
    twii (加權 date/close，用來算 sigma60)、night_used (夜盤模式是否會用在下一交易日)、opt_hist (TXO iv 歷史)。"""
    P = load_params()
    ctx = dict(ctx or {})
    now = ctx.get("now") or dt.datetime.now(config.TZ)
    today = _d(today or now.date())
    cal = refresh_calendar(today, write=write_calendar, P=P)
    last = str(ctx.get("last_td") or "")[:10] or prev_td(today + dt.timedelta(days=1) if (now.hour >= 14 and is_tw_trading(today)) else today)
    td = next_td(last)
    sf = session_factor(td, {"used": bool(ctx.get("night_used"))}, prev=last, P=P, cal=cal)
    acc = range_levels_enabled(P)
    ns: dict = {"date": td, "prev_td": last, "n_us": sf["n_us"], "n_us_uncovered": sf["n_us_uncovered"], "closed_weekdays": sf["closed_weekdays"],
                "range_k1_factor": sf["range_k1"], "range_k1_factor_applied": bool(acc and sf["range_k1"] > 1),
                "range_mode": sf["range_mode"] if acc else ("base" if not ctx.get("night_used") else "night"),
                "iv_band_factor": 1.0, "tags": sf["tags"], "why": clean_text(sf["why"], P),
                "applied_to": (["range_levels k=1 (路徑帶，驗收 A 通過)"] if acc else ["跳空預估卡與說明 (range_levels 路徑帶未套用：驗收 A 未通過)"]) if sf["range_k1"] > 1 else []}
    if ctx.get("px0050") is None and (sf["closed_weekdays"] >= 1):
        ctx["px0050"], ex = _load_0050()
        ctx.setdefault("exdiv_0050", ex)
    for key, sym, rg in (("sox", "^SOX", "5y"), ("tsm", "TSM", "20y")):
        if ctx.get(key) is None and sf["closed_weekdays"] >= 1:
            try:
                from ..sources import global_markets
                ctx[key] = global_markets.history_closed(sym, rg)
            except Exception as e:  # noqa: BLE001
                log.info("events %s unavailable: %s", sym, e)
    try:
        ns["gap"] = gap_forecast(td, last, ctx.get("px0050"), ctx.get("sox"), ctx.get("tsm"), ctx.get("exdiv_0050") or (),
                                 _sigma60(ctx.get("twii"), td) if ctx.get("twii") is not None else None, now=now, P=P)
    except Exception as e:  # noqa: BLE001
        log.warning("events gap_forecast: %s", e)
        ns["gap"] = {"status": "error", "note": str(e)[:120]}
    ns["iv_expect"] = iv_expect(last, td, ctx.get("opt_hist"), P, cal)
    t = tsmc_2330_k1(last, now=now, P=P)
    if t.get("call_date"):
        ns["tsmc_2330_k1"] = t
    ups = upcoming(cal, td, P, today=today.isoformat())
    return {"asof": today.isoformat(), "generated": now.strftime("%Y-%m-%d %H:%M:%S"), "params_version": P.get("version"),
            "horizon_days": P.get("horizon_days"), "disclaimer": DISCLAIMER, "direction_vote_weight": 0.0,
            "range_levels_path_enabled": acc, "next_session": ns, "upcoming": ups, "calendar_health": cal.get("health")}


def for_forecast(ev: dict) -> dict:
    """forecast.json["events"]：next_session + 前 10 筆 upcoming (精簡)。"""
    if not ev or ev.get("error"):
        return ev or {}
    keep = ("date", "tw_session", "event_type", "title_zh", "days_to", "grade_vol", "grade_gap", "grade_dir", "n", "vol_text", "dir_text", "date_confidence", "band")
    return {k: ev.get(k) for k in ("asof", "generated", "params_version", "disclaimer", "direction_vote_weight", "range_levels_path_enabled", "next_session", "calendar_health")} | {
        "upcoming": [{k: u.get(k) for k in keep if k in u} for u in (ev.get("upcoming") or [])[:10]]}


def tags_for_window(start_exclusive: str, end_inclusive: str, cal: dict | None = None) -> list[str]:
    """(start, end] 期間各交易日的事件標籤聯集 (band_ledger 用)。"""
    tags: list[str] = []
    for d in tw_sessions(_d(start_exclusive) + dt.timedelta(days=1), end_inclusive):
        for t in session_tags(d, cal):
            if t not in tags:
                tags.append(t)
    return tags


def all_texts(ev: dict) -> list[str]:
    """build() 輸出內所有使用者可見中文 (測試用：禁用字檢查)。"""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("title_zh", "notes_zh", "dir_text", "vol_text", "note", "why", "text") and isinstance(v, str):
                    out.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(ev)
    return out
