"""挖寶雷達 / 飆股雷達 雲端每日掃描 + 永久追蹤帳本 (2026-10-05)。

目的：App 的掃描只在裝置開著時跑、追蹤紀錄只存在各裝置 localStorage (加密同步，雲端讀不到)，
所以沒開 App 的日子沒有紀錄、換裝置看不到完整歷史。這裡在 GitHub Actions 盤後用「和 App 相同的模型與候選規則」
每天掃一次並永久歸檔，隔天起逐檔對帳，統計真正的上線成績 (依等級/年月)，失準時警示。

與 App (learning.js / app.js) 的對應：
- 候選池：TWSE 全市場盤後表 (MI_INDEX ALLBUT0999，同 App 13:35 後的 after-close 來源) + BWIBBU_ALL 本益比/淨值比 →
  scoring.js screen() 的 power 公式 (治學倍率 treasureMult 固定 1：那是各裝置自己的統計)，取前 80 並排除 ETF (代號 00 開頭) → 挖寶看前 40、飆股看前 80。
- 特徵：treasure.features (與 App tmFeatures 同款)；日 K 用 Yahoo 2 年 (Worker /idxh，失敗改直連)，D 日那根以盤後表取代 (含成交值)；大盤用 Yahoo ^TWII。
- 分級：p ≥ th_A 且大盤月線乖離 < 0 → A (≥ th_Aplus → A+)；高分但大盤在月線上 → B+；其餘 B。飆股 = 前 80 中 ps 前 3 名。
- 記錄：挖寶每天最多 6 檔 (池內前 20 依 p 排序、pct < 9.4、同檔 60 日內未命中不重複、追蹤中不超過 40)；飆股前 3 名 (同檔 28 日內不重複)。
- 結案 (改用回測定義，App 的 30 日曆天/不分先後 與回測不同，所以兩邊成績對不上)：
  挖寶 21 交易日：收盤峰值 ≥ +6% 或相對大盤任一日 ≥ +4pt (先盤中低點 ≤ −8% 且當時峰值 < 6% 為停損未命中)，結案收盤 > 0 才算命中；曾達標但結案 ≤ 0 為「回落」。
  飆股 20 交易日：最高價先到 +20% 且之前最低價未破 −10% 為「飆」；出場規則：漲 10% 後自最高點回落 8% 出場，否則第 20 日收盤。
輸出 data/treasure_live.json：{asof, scan{date, treasure[], surge[], pool_n}, ledger{treasure[], surge[]}, stats, alerts, model{trained_at, th_A, th_Aplus}}。
歸檔跨次累積：Pages 上一版 ∪ 本機 data/cache (同 gov8 的保險機制)；縮水保護。

pr2 (2026-10-05，LC-04 pass / T5-01~03 / T5-missed / LC-05~06 fail)：
- A/A+ 當訊號 (role=signal)、B/B+ 當描述 (role=descriptive；回測即無選股力：B 39% ≈ 同日池 40%，上線 12~22% 反映 7 月大盤 −10.7%)。
- A∪A+ 的預期命中改為 55~57% (回測 A 0.548 / A+ 0.638；OOS 分數器重掃回填 0.568)，不是帳本的 64%/75% (分數器看過答案的回填、門檻為樣本內)；
  大半優勢來自「大盤月線下的日子」(同日池 52%)，等級內 p 高低與命中無關 (Spearman 0.05) → 警報只寫「模型分」，不寫「命中機率」。
- 帳本誠實化：每筆記 model_ver / th_A / th_Aplus / mkt_bias20 / scan_src (live|backfill)；停損列也追到第 21 日補 fin21 (cur 仍是結案日報酬)；
  stats 分 by_source (真實發布 vs 回填)、signal (A∪A+ 主數字)；漂移只對訊號級且真實發布 ≥15 筆判定，B/B+ 不再產生「失準」。
- 門檻 th_A 0.5062 / th_Aplus 0.6362、飆股 th_top10 0.3977 與前 3 名維持 (LC-05/06)；模型檔不動、無重訓。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..http import cached, session
from . import model as M
from . import treasure as T

log = logging.getLogger(__name__)
PAGES = "https://joekisoul-code.github.io/TaiexChipAnalyzer/data/treasure_live.json"
WORKER = "https://skynet-proxy.joekisoul.workers.dev"
LOCAL = config.CACHE_DIR / "treasure_live.json"
UA = {"User-Agent": "Mozilla/5.0"}
POOL, TREASURE_POOL, PER_DAY, MAX_ACTIVE, REENTRY_MISS_DAYS = 80, 40, 6, 40, 60
SURGE_DEDUP_DAYS = 28
LEDGER_MAX = 1500
EXPECT_AAPLUS = (0.55, 0.57)   # pr2 LC-04：A∪A+ 預期命中區間 (回測 A 0.548 / A+ 0.638、OOS 分數器重掃 0.568)
ROLE = {"A+": "signal", "A": "signal", "B+": "descriptive", "B": "descriptive"}
DRIFT_MIN_PUBLISHED = 15       # 訊號級漂移判定：真實發布結案 ≥15 筆且命中低於預期下限 15pt
NOTE_TIERS = "A/A+ 為訊號級；B/B+ 為描述級 (回測即無選股力 #40/#84)"
MAX_FIN21_WAIT_DAYS = 60       # 結案列補 fin21：日 K 已延伸 60 個日曆日仍湊不到 21 根 (下市/停牌) → 放棄


# ------------------------------------------------------------------ 資料
def _num(x):
    try:
        s = str(x).replace(",", "").strip()
        return float(s) if s not in ("", "--", "-", "X") else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def market_snapshot(date: str | None = None) -> tuple[str | None, dict]:
    """TWSE 盤後全市場表 (rwd MI_INDEX)。回傳 (資料日, {code: {name, open, high, low, close, volume, value, change, pct, amp, per}})。
    date 省略 → 今天；當天沒有資料 (休市/未公布) 往前找最多 6 天。"""
    d0 = dt.date.fromisoformat(date) if date else dt.datetime.now(config.TZ).date()
    for back in range(0, 7):
        d = d0 - dt.timedelta(days=back)
        if d.weekday() >= 5:
            continue
        ds = d.strftime("%Y%m%d")
        def load(ds=ds):
            r = session().get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", params={"date": ds, "type": "ALLBUT0999", "response": "json", "_": int(dt.datetime.now().timestamp())},
                              headers=UA, timeout=60)
            r.raise_for_status()
            j = r.json()
            tbl = [t for t in (j.get("tables") or []) if len(t.get("data") or []) > 500]
            if not tbl:
                return None
            f = tbl[0]["fields"]; ix = {k: f.index(k) for k in f}
            out = {}
            for row in tbl[0]["data"]:
                code = str(row[ix["證券代號"]]).strip()
                if not re.fullmatch(r"\d{4,6}[A-Z]?", code):
                    continue
                c, o, h, l = _num(row[ix["收盤價"]]), _num(row[ix["開盤價"]]), _num(row[ix["最高價"]]), _num(row[ix["最低價"]])
                if not (c > 0):
                    continue
                sign = -1 if "-" in str(row[ix["漲跌(+/-)"]]) else 1
                chg = sign * _num(row[ix["漲跌價差"]])
                if not np.isfinite(chg):
                    chg = 0.0
                out[code] = {"name": str(row[ix["證券名稱"]]).strip(), "open": o, "high": h, "low": l, "close": c, "volume": _num(row[ix["成交股數"]]),
                             "value": _num(row[ix["成交金額"]]), "change": chg, "pct": chg / ((c - chg) or 1e-9) * 100, "amp": (h - l) / (l or 1e-9) * 100 if h > 0 and l > 0 else 0.0,
                             "per": _num(row[ix["本益比"]])}
            return {"date": d.isoformat(), "rows": out}
        try:
            j = cached(f"twse:mi_index_all:{ds}", 30 * 86400, load, allow_stale=False)
        except Exception as e:  # noqa: BLE001
            log.warning("MI_INDEX %s: %s", ds, e)
            j = None
        if j and j.get("rows"):
            return j["date"], j["rows"]
    return None, {}


def _twii_close_from_twse(D: str) -> float:
    """MI_INDEX 第一張表 (大盤統計資訊) 的「發行量加權股價指數」收盤。"""
    r = session().get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", params={"date": D.replace("-", ""), "type": "ALLBUT0999", "response": "json"}, headers=UA, timeout=60)
    for t in r.json().get("tables") or []:
        for row in t.get("data") or []:
            if row and "發行量加權股價指數" in str(row[0]) and len(row) > 1:
                v = _num(row[1])
                if v > 0:
                    return v
    raise ValueError("no TWII row")


def valuation() -> dict:
    """BWIBBU_ALL (經 Worker；Actions 直連 openapi 不通)：{code: {per, pbr}}。失敗 → {} (只影響候選池排序的 nav/valu 項)。"""
    def load():
        j = session().get(WORKER + "/twse/v1/exchangeReport/BWIBBU_ALL", headers=UA, timeout=60).json()
        return {str(x.get("Code")).strip(): {"per": _num(x.get("PEratio")), "pbr": _num(x.get("PBratio"))} for x in j if x.get("Code")}
    try:
        return cached("twse:bwibbu_all", config.TTL_DAILY, load) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("BWIBBU_ALL: %s", e)
        return {}


def screen(rows: dict, val: dict, mkt_pct: float, limit: int = POOL) -> list[dict]:
    """scoring.js screen()：power = mom + liq + eng + nav + valu + lag + near (treasureMult 固定 1)；成交值 > 5 千萬、pct < 9.3，前 limit。"""
    out = []
    for code, s in rows.items():
        if not (s["value"] > 5e7) or not np.isfinite(s["close"]):
            continue
        pct, amp = s["pct"], s["amp"]
        v = val.get(code) or {}
        pbr, per = v.get("pbr", float("nan")), v.get("per", float("nan"))
        mom = min(pct, 7) * 1.6 if pct >= 0 else pct * 0.6
        liq = math.log10(s["value"]) * 2
        eng = min(amp, 8) * 0.8
        nav = (1.2 - pbr) * 10 if pbr > 0 and pbr < 1.2 else 0
        valu = (15 - per) * 0.5 if per > 0 and per < 15 else 0
        lag = min(mkt_pct - pct, 5) * 1.2 if pct < mkt_pct else 0
        near = (pct - 3) * 0.6 if 3 <= pct < 9 else 0
        if pct < 9.3:
            out.append({"code": code, **s, "pbr": pbr, "power": mom + liq + eng + nav + valu + lag + near})
    out.sort(key=lambda x: -x["power"])
    return out[:limit]


def bars(code: str, years: str = "2y") -> list[dict]:
    """Yahoo 日 K (date/open/high/low/close/volume)：Worker /idxh 優先 (與 App 同源)，失敗改直連 Yahoo。.TW 不夠長再試 .TWO。"""
    def via_worker(sym):
        j = session().get(WORKER + "/idxh", params={"sym": sym, "range": years, "interval": "1d"}, headers=UA, timeout=60).json()
        return [x for x in (j.get("data") or []) if x.get("close")]
    def via_yahoo(sym):
        j = session().get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"interval": "1d", "range": years}, headers=UA, timeout=60).json()
        res = j["chart"]["result"][0]; ts, q = res["timestamp"], res["indicators"]["quote"][0]; off = res["meta"].get("gmtoffset", 0)
        return [{"date": (dt.datetime.fromtimestamp(t, dt.UTC) + dt.timedelta(seconds=off)).strftime("%Y-%m-%d"), "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i], "volume": q["volume"][i]}
                for i, t in enumerate(ts) if q["close"][i]]
    for suf in (".TW", ".TWO"):
        sym = code + suf
        for fn in (via_worker, via_yahoo):
            try:
                rows = cached(f"tl:bars:{sym}:{years}", config.TTL_INTRADAY * 3, lambda fn=fn, sym=sym: fn(sym), allow_stale=False)
                if rows and len(rows) > 60:
                    return rows
            except Exception as e:  # noqa: BLE001
                log.debug("bars %s %s: %s", sym, fn.__name__, e)
    return []


def twii_hist() -> dict:
    """{date: close} (Yahoo ^TWII 10y，Worker 優先)。"""
    rows = []
    try:
        rows = cached("tl:twii:10y", config.TTL_INTRADAY * 3, lambda: session().get(WORKER + "/idxh", params={"sym": "^TWII", "range": "10y", "interval": "1d"}, headers=UA, timeout=60).json().get("data"),
                      allow_stale=False) or []
    except Exception as e:  # noqa: BLE001
        log.debug("twii via worker: %s", e)
    if not rows:
        try:
            from ..sources import global_markets
            df = global_markets.history("^TWII", "10y")
            rows = df.to_dict("records")
        except Exception as e:  # noqa: BLE001
            log.warning("twii fallback: %s", e)
    return {str(x["date"])[:10]: float(x["close"]) for x in rows if x.get("close")}


# ------------------------------------------------------------------ 評分 (與 App tmScore 相同)
def _mk_frame(mk: dict, upto: str) -> pd.DataFrame:
    d = sorted(k for k in mk if k <= upto)
    s = pd.Series([mk[k] for k in d], index=d)
    return pd.DataFrame({"date": d, "m_close": s.values, "m_ret1": s.pct_change().values * 100, "m_ret20": s.pct_change(20).values * 100,
                         "m_bias20": (s / s.rolling(20).mean() - 1).values * 100})


def score_one(code: str, snap_row: dict, D: str, mkf: pd.DataFrame, tm: dict) -> dict | None:
    b = bars(code)
    rows = [{"date": str(x["date"])[:10], "open": x.get("open"), "high": x.get("high"), "low": x.get("low"), "close": x["close"], "volume": x.get("volume") or 0, "amount": np.nan}
            for x in b if x.get("close") and str(x["date"])[:10] <= D]
    if not rows:
        return None
    r = snap_row
    bar = {"date": D, "open": r["open"] if r["open"] > 0 else r["close"], "high": r["high"] if r["high"] > 0 else r["close"], "low": r["low"] if r["low"] > 0 else r["close"],
           "close": r["close"], "volume": r["volume"] or (rows[-1]["volume"] if rows[-1]["date"] == D else 0), "amount": r["value"]}
    if rows[-1]["date"] == D:
        rows[-1] = bar
    elif D > rows[-1]["date"]:
        rows.append(bar)
    if rows[-1]["date"] != D or len(rows) < 62:
        return None
    g = pd.DataFrame(rows)
    g["amount"] = g["amount"].fillna(g["close"] * g["volume"])      # App：amount 只有 D 日有值，其餘以 close×volume (lval 只用 D 日)
    f = T.features(g, mkf, label=False).iloc[-1]
    x = {k: (float(f[k]) if k in f and pd.notna(f[k]) else float("nan")) for k in set(tm["features"]) | set((tm.get("surge") or {}).get("features") or [])}
    arr = [x[k] for k in tm["features"]]
    if any(not np.isfinite(v) for v in arr):
        return None
    p = 1 / (1 + math.exp(-T._eval(tm, arr)))
    gl = (tm.get("gate") or {}).get("m_bias20_lt")
    hi = p >= tm.get("th_A", 0.5); gate = gl is None or x["m_bias20"] < gl
    tier = "B" if not hi else ("B+" if not gate else ("A+" if tm.get("th_Aplus") is not None and p >= tm["th_Aplus"] else "A"))
    ps = None
    sg = tm.get("surge")
    if sg and sg.get("trees"):
        try:
            ps = 1 / (1 + math.exp(-T._eval(sg, [x[k] if np.isfinite(x[k]) else None for k in sg.get("features") or tm["features"]])))
        except Exception as e:  # noqa: BLE001
            log.debug("surge eval %s: %s", code, e)
    return {"p": round(p, 4), "tier": tier, "ps": round(ps, 4) if ps is not None else None, "m_bias20": round(x["m_bias20"], 3)}


def scan(tm: dict, date: str | None = None) -> dict:
    """當日掃描：回傳 {date, pool_n, treasure[(前 40 中 pct<9.4、依 p 排序)], surge[前 3], mkt_close, mkt_bias20, errors}。"""
    D, rows = market_snapshot(date)
    if not D:
        return {"date": None, "error": "no market snapshot"}
    mk = twii_hist()
    if D not in mk:                       # Yahoo ^TWII 尚未有 D 日 (常晚數小時) → 以盤後表內的加權指數收盤補上
        try:
            mk = dict(mk); mk[D] = _twii_close_from_twse(D)
        except Exception as e:  # noqa: BLE001
            log.warning("TWII %s missing (%s)：大盤特徵以前一日計", D, e)
    mkf = _mk_frame(mk, D)
    mkt_pct = float(mkf["m_ret1"].iloc[-1]) if len(mkf) and pd.notna(mkf["m_ret1"].iloc[-1]) else 0.0
    pool = [r for r in screen(rows, valuation(), mkt_pct, POOL) if not r["code"].startswith("00")][:POOL]
    res, errors = {}, 0
    for r in pool:
        try:
            s = score_one(r["code"], r, D, mkf, tm)
            if s:
                res[r["code"]] = s
        except Exception as e:  # noqa: BLE001
            errors += 1; log.debug("score %s: %s", r["code"], e)
    tre = [{"code": r["code"], "name": r["name"], "close": r["close"], "pct": round(r["pct"], 2), "value": r["value"], "pbr": r.get("pbr"), **res[r["code"]]}
           for r in pool[:TREASURE_POOL] if r["code"] in res and r["pct"] < 9.4]
    tre.sort(key=lambda x: -x["p"])
    sg_k = int(((tm.get("surge") or {}).get("def") or {}).get("topk") or 3)
    sur = sorted([{"code": r["code"], "name": r["name"], "close": r["close"], "pct": round(r["pct"], 2), "ps": res[r["code"]]["ps"]} for r in pool if r["code"] in res and res[r["code"]]["ps"] is not None and r["pct"] < 9.4],
                 key=lambda x: -x["ps"])[:sg_k]
    return {"date": D, "pool_n": len(pool), "scored_n": len(res), "errors": errors, "treasure": tre, "surge": sur,
            "mkt_close": mk.get(D), "mkt_bias20": round(float(mkf["m_bias20"].iloc[-1]), 3) if len(mkf) and pd.notna(mkf["m_bias20"].iloc[-1]) else None,
            "mkt_pct": round(mkt_pct, 2)}


# ------------------------------------------------------------------ 帳本
def load_prev() -> dict:
    """上次發布 (Pages) ∪ 本機備份，帳本以 (kind, code, date) 去重、取較完整者。"""
    out = {"ledger": {"treasure": [], "surge": []}, "scans": []}
    srcs = []
    try:
        r = session().get(PAGES, headers=UA, timeout=30)
        if r.ok and r.text.strip().startswith("{"):
            srcs.append(r.json())
    except Exception as e:  # noqa: BLE001
        log.debug("published treasure_live: %s", e)
    try:
        if LOCAL.exists():
            srcs.append(json.loads(LOCAL.read_text(encoding="utf-8")))
    except Exception as e:  # noqa: BLE001
        log.debug("local treasure_live: %s", e)
    for s in srcs:
        for kind in ("treasure", "surge"):
            have = {(x["code"], x["date"]): i for i, x in enumerate(out["ledger"][kind])}
            for x in ((s.get("ledger") or {}).get(kind) or []):
                k = (x.get("code"), x.get("date"))
                if k in have:
                    old = out["ledger"][kind][have[k]]
                    if (x.get("status") != "追蹤" and old.get("status") == "追蹤") or (len(json.dumps(x)) > len(json.dumps(old)) and old.get("status") == "追蹤")                             or (old.get("status") != "追蹤" and x.get("status") == old.get("status") and x.get("fin21") is not None and old.get("fin21") is None):   # pr2：已結案但另一份已補 fin21
                        out["ledger"][kind][have[k]] = x
                else:
                    have[k] = len(out["ledger"][kind]); out["ledger"][kind].append(x)
        seen = {x["date"] for x in out["scans"]}
        for x in (s.get("scans") or []):
            if x.get("date") and x["date"] not in seen:
                out["scans"].append(x); seen.add(x["date"])
    for kind in ("treasure", "surge"):
        out["ledger"][kind].sort(key=lambda x: (x.get("date") or "", x.get("code") or ""))
    out["scans"].sort(key=lambda x: x["date"])
    return out


def record(prev: dict, sc: dict, tm: dict | None = None, scan_src: str = "live") -> dict:
    """把本日掃描結果記入帳本 (同 App recordDiscoveries / recordSurge 的規則)。
    pr2：每筆加 model_ver (掃描時模型 trained_at)、th_A / th_Aplus (當時門檻)、mkt_bias20 (閘門狀態)、scan_src ('live' 真實發布 | 'backfill' 回填)，
    讓帳本不再路徑相依、可事後重算等級。"""
    D = sc.get("date")
    if not D:
        return prev
    tm = tm or {}
    meta = {"model_ver": tm.get("trained_at"), "th_A": tm.get("th_A"), "th_Aplus": tm.get("th_Aplus"), "mkt_bias20": sc.get("mkt_bias20"), "scan_src": scan_src}
    led = prev["ledger"]
    tre = led["treasure"]
    active = [x for x in tre if x.get("status") == "追蹤"]
    act_codes = {x["code"] for x in active}
    recent_miss = {x["code"] for x in tre if x.get("status") == "未命中" and x.get("evalAt") and (pd.Timestamp(D) - pd.Timestamp(x["evalAt"])).days < REENTRY_MISS_DAYS}
    have = {(x["code"], x["date"]) for x in tre}
    n = 0
    for r in sc["treasure"][:20]:
        if n >= PER_DAY or len(active) + n >= MAX_ACTIVE:
            break
        if r["code"] in act_codes or r["code"] in recent_miss or (r["code"], D) in have:
            continue
        tre.append({"code": r["code"], "name": r["name"], "date": D, "entry": r["close"], "mktEntry": sc.get("mkt_close"), "p": r["p"], "tier": r["tier"], "ps": r.get("ps"),
                    "pct": r["pct"], "status": "追蹤", "peak": 0.0, "trough": 0.0, "cur": 0.0, "rel": 0.0, "days": 0, **meta})
        n += 1
    sur = led["surge"]
    for r in sc["surge"]:
        if any(x["code"] == r["code"] and abs((pd.Timestamp(D) - pd.Timestamp(x["date"])).days) < SURGE_DEDUP_DAYS for x in sur):
            continue
        sur.append({"code": r["code"], "name": r["name"], "date": D, "entry": r["close"], "ps": r["ps"], "status": "追蹤", "days": 0,
                    "model_ver": meta["model_ver"], "th_top10": (tm.get("surge") or {}).get("th_top10"), "scan_src": scan_src})
    scans = [x for x in prev.get("scans") or [] if x.get("date") != D]
    scans.append({"date": D, "n_treasure": len(sc["treasure"]), "n_A": sum(1 for r in sc["treasure"] if r["tier"] in ("A", "A+")), "mkt_bias20": sc.get("mkt_bias20"),
                  "top": [{"code": r["code"], "tier": r["tier"], "p": r["p"]} for r in sc["treasure"][:6]], "surge": [{"code": r["code"], "ps": r["ps"]} for r in sc["surge"]]})
    prev["scans"] = scans[-400:]
    for kind in ("treasure", "surge"):
        led[kind] = led[kind][-LEDGER_MAX:]
    return prev


def _fin21_of(x: dict):
    """第 21 個交易日收盤報酬 (回測口徑)：新列有 fin21；舊結案列若在第 21 日結案，cur 即 fin21。"""
    if x.get("fin21") is not None:
        return x["fin21"]
    if x.get("status") not in (None, "追蹤") and (x.get("days") or 0) >= T.H and x.get("cur") is not None:
        return x["cur"]
    return None


def _eval_treasure(x: dict, b: list[dict], mk: dict) -> None:
    """回測定義 (treasure.features 的 label)：21 交易日；峰值取收盤、停損取盤中低點 (先後順序)、相對大盤任一日 ≥ 4pt。
    pr2：fin21 = 第 21 個交易日收盤報酬，不論是否已停損 (停損後仍追到 21 日)；已結案列只補 fin21 (cur/peak/days 等結案值不動)。"""
    e = float(x["entry"]); since = [r for r in b if str(r["date"])[:10] > x["date"] and r.get("close")][:T.H]
    if not since:
        return
    if len(since) >= T.H and x.get("fin21") is None:
        x["fin21"] = round((float(since[T.H - 1]["close"]) / e - 1) * 100, 2); x["fin21Date"] = str(since[T.H - 1]["date"])[:10]
    elif len(since) < T.H and x.get("fin21") is None and (pd.Timestamp(str(b[-1]["date"])[:10]) - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
        x["fin21_na"] = True      # 日 K 不足 (下市/停牌/資料斷)：不再重試
    if x.get("status") not in (None, "追蹤"):
        if x.get("trail") is None:   # 舊結案列 (理論上結案時已填)：補移動停利
            pk = e
            for r in since:
                pk = max(pk, float(r["close"]))
                if pk >= e * 1.1 and float(r["close"]) <= pk * 0.92:
                    x["trail"] = round((float(r["close"]) / e - 1) * 100, 2); x["trailDate"] = str(r["date"])[:10]; break
            if x.get("trail") is None and len(since) >= T.H:
                x["trail"] = x.get("cur")
        return
    me = x.get("mktEntry") or (mk.get(x["date"]) if x["date"] in mk else None)
    cm = -1e9; stop = False; reached = False; peak = -1e9; trough = 1e9; rel_max = -1e9
    for r in since:
        cr = (float(r["close"]) / e - 1) * 100; lr = (float(r.get("low") or r["close"]) / e - 1) * 100
        cm = max(cm, cr); peak = max(peak, cr); trough = min(trough, lr)
        mr = ((mk.get(str(r["date"])[:10]) or float("nan")) / me - 1) * 100 if me else 0.0
        if not np.isfinite(mr):
            mr = 0.0
        rel_max = max(rel_max, cr - mr)
        if not stop and not reached and lr <= T.STOP and cm < T.TGT:
            stop = True
        if not stop and (cr >= T.TGT or (cr - mr) >= T.REL):
            reached = True
    cur = (float(since[-1]["close"]) / e - 1) * 100
    last_m = mk.get(str(since[-1]["date"])[:10]); rel = (cur - ((last_m / me - 1) * 100)) if (me and last_m) else None
    x.update(peak=round(peak, 2), trough=round(trough, 2), cur=round(cur, 2), rel=round(rel, 2) if rel is not None else None, relMax=round(rel_max, 2), days=len(since), lastDate=str(since[-1]["date"])[:10])
    # 移動停利 (A 級出場建議：漲 10% 後自最高收盤回落 8%)
    if x.get("trail") is None:
        pk = e
        for r in since:
            pk = max(pk, float(r["close"]))
            if pk >= e * 1.1 and float(r["close"]) <= pk * 0.92:
                x["trail"] = round((float(r["close"]) / e - 1) * 100, 2); x["trailDate"] = str(r["date"])[:10]; break
    if x.get("status") == "追蹤" and (stop or len(since) >= T.H):
        if stop:
            x["status"], x["reason"] = "未命中", f"先跌破 {T.STOP:.0f}% (盤中低點 {trough:.1f}%)"
        elif reached and cur > 0:
            x["status"], x["reason"] = "命中", f"峰值 {peak:+.1f}%、結案 {cur:+.1f}%"
        elif reached:
            x["status"], x["reason"] = "回落", f"曾達標 (峰值 {peak:+.1f}%) 但結案 {cur:+.1f}%"
        else:
            x["status"], x["reason"] = "未命中", f"21 日內未達 +6% / 相對 +4pt (峰值 {peak:+.1f}%)"
        x["evalAt"] = str(since[-1]["date"])[:10]
        if x.get("trail") is None:
            x["trail"] = round(cur, 2)


def _eval_surge(x: dict, b: list[dict]) -> None:
    e = float(x["entry"]); since = [r for r in b if str(r["date"])[:10] > x["date"] and r.get("close")][:T.SURGE_N]
    if not since:
        return
    peak = e; res = None; trail = None; tdate = None
    for r in since:
        h = float(r.get("high") or r["close"]); l = float(r.get("low") or r["close"]); c = float(r["close"])
        if res is None and l <= e * (1 + T.SURGE_DN / 100):
            res = "未飆"
        if res is None and h >= e * (1 + T.SURGE_UP / 100):
            res = "飆"
        peak = max(peak, h)
        if trail is None and peak >= e * 1.1 and c <= peak * 0.92:
            trail, tdate = (c / e - 1) * 100, str(r["date"])[:10]
    last = float(since[-1]["close"])
    x.update(cur=round((last / e - 1) * 100, 2), peak=round((peak / e - 1) * 100, 2), days=len(since), lastDate=str(since[-1]["date"])[:10])
    if trail is not None:
        x["trail"], x["trailDate"] = round(trail, 2), tdate
    if x.get("status") == "追蹤" and (res in ("飆", "未飆") or len(since) >= T.SURGE_N):
        x["status"] = "飆" if res == "飆" else "未飆"; x["evalAt"] = str(since[-1]["date"])[:10]
    if x.get("trail") is None and len(since) >= T.SURGE_N:
        x["trail"] = x["cur"]


def evaluate(prev: dict, mk: dict, max_fetch: int = 200, upto: str | None = None) -> int:
    """逐檔對帳 (未結案或尚未出場的紀錄)；每次最多抓 max_fetch 檔 (控制 Yahoo 呼叫)。upto：只用該日 (含) 以前的 K 棒 (回填時逐日模擬)。"""
    n = 0
    todo = []
    for x in prev["ledger"]["treasure"]:
        if x.get("status") == "追蹤" or x.get("trail") is None or (x.get("fin21") is None and not x.get("fin21_na") and (x.get("days") or 0) < T.H):
            todo.append(("t", x))     # pr2：停損提早結案的列繼續追到第 21 日補 fin21
    for x in prev["ledger"]["surge"]:
        if x.get("status") == "追蹤" or (x.get("trail") is None and (x.get("days") or 0) < T.SURGE_N):
            todo.append(("s", x))
    cache: dict[str, list] = {}
    for kind, x in todo[:max_fetch * 2]:
        if n >= max_fetch and x["code"] not in cache:
            break
        try:
            b = cache.get(x["code"])
            if b is None:
                b = bars(x["code"]); cache[x["code"]] = b; n += 1          # 與評分同一份 2 年日 K 快取
            if upto:
                b = [r for r in b if str(r["date"])[:10] <= upto]
            if not b:
                # 取不到任何日 K (下市/改代號)：已結案、尚缺 fin21 的列超過 MAX_FIN21_WAIT_DAYS 個日曆日 → 標 fin21_na，不再每次重抓
                if kind == "t" and x.get("status") not in (None, "追蹤") and x.get("fin21") is None and not x.get("fin21_na"):
                    ref = pd.Timestamp(upto) if upto else pd.Timestamp(dt.datetime.now(config.TZ).strftime("%Y-%m-%d"))
                    if (ref - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
                        x["fin21_na"] = True
                continue
            (_eval_treasure(x, b, mk) if kind == "t" else _eval_surge(x, b))
        except Exception as e:  # noqa: BLE001
            log.debug("evaluate %s: %s", x["code"], e)
    return n


def stats(prev: dict, tm: dict) -> dict:
    for kind in ("treasure", "surge"):
        prev["ledger"][kind].sort(key=lambda x: (x.get("date") or "", x.get("code") or ""))
    tre = prev["ledger"]["treasure"]; sur = prev["ledger"]["surge"]
    done = [x for x in tre if x.get("status") != "追蹤"]
    def agg(rows):
        if not rows:
            return None
        hit = sum(1 for x in rows if x["status"] == "命中")
        cur = [x["cur"] for x in rows if x.get("cur") is not None]; rel = [x["rel"] for x in rows if x.get("rel") is not None]
        tr = [x["trail"] for x in rows if x.get("trail") is not None]
        f21 = [v for v in (_fin21_of(x) for x in rows) if v is not None]
        return {"n": len(rows), "hit": hit, "rate": round(hit / len(rows), 3), "fallback": sum(1 for x in rows if x["status"] == "回落"), "miss": sum(1 for x in rows if x["status"] == "未命中"),
                "avg_fin": round(float(np.mean(cur)), 2) if cur else None, "win": round(float(np.mean([c > 0 for c in cur])), 3) if cur else None,
                "avg_rel": round(float(np.mean(rel)), 2) if rel else None, "trail_avg": round(float(np.mean(tr)), 2) if tr else None, "trail_win": round(float(np.mean([t > 0 for t in tr])), 3) if tr else None,
                # pr2：21 日報酬 (同回測口徑，停損列也追到 21 日)、真實發布 / 回填筆數
                "avg_fin21": round(float(np.mean(f21)), 2) if f21 else None, "n_fin21": len(f21), "win21": round(float(np.mean([v > 0 for v in f21])), 3) if f21 else None,
                "n_published": sum(1 for x in rows if not x.get("backfill")), "n_backfill": sum(1 for x in rows if x.get("backfill"))}
    oos = ((tm.get("oos") or {}).get("tiers") or {})
    pub = [x for x in done if not x.get("backfill")]
    by_tier = {}
    for t in ("A+", "A", "B+", "B"):
        g = agg([x for x in done if x.get("tier") == t])
        if g:
            bt = oos.get("A-" if t == "A" else t) or {}
            g["bt_hit"] = bt.get("hit"); g["bt_fin"] = bt.get("fin")
            g["role"] = ROLE[t]
            gp = agg([x for x in pub if x.get("tier") == t])
            g["published"] = {k: gp[k] for k in ("n", "hit", "rate", "avg_fin", "avg_fin21")} if gp else None
            g["below_bt"] = bool(g["n"] >= 15 and bt.get("hit") is not None and g["rate"] < bt["hit"] - 0.15)   # 資訊旗標 (含回填)，不是失準判定
            if g["role"] == "signal":
                g["expect"] = list(EXPECT_AAPLUS)
                g["drift"] = bool(gp and gp["n"] >= DRIFT_MIN_PUBLISHED and gp["rate"] < EXPECT_AAPLUS[0] - 0.15)   # 只看真實發布結案
            else:
                g["drift"] = False
                g["note"] = "描述級：不作失準判定 (回測即無選股力：B 39% ≈ 同日池 40%；上線 12~22% 反映 7 月大盤 −10.7%)"
            by_tier[t] = g
    sig_rows = [x for x in done if x.get("tier") in ("A+", "A")]
    signal = agg(sig_rows)
    if signal:
        signal["role"] = "signal"; signal["expect"] = list(EXPECT_AAPLUS)
        gp = agg([x for x in sig_rows if not x.get("backfill")])
        signal["published"] = {k: gp[k] for k in ("n", "hit", "rate", "avg_fin", "avg_fin21")} if gp else None
        signal["drift"] = bool(gp and gp["n"] >= DRIFT_MIN_PUBLISHED and gp["rate"] < EXPECT_AAPLUS[0] - 0.15)
    by_source = {"published": agg(pub), "backfill": agg([x for x in done if x.get("backfill")])}
    by_month = {}
    for x in done:
        by_month.setdefault(x["date"][:7], []).append(x)
    by_month = {m: {"n": len(v), "rate": round(sum(1 for x in v if x["status"] == "命中") / len(v), 3), "avg_fin": round(float(np.mean([x["cur"] for x in v if x.get("cur") is not None] or [0])), 2),
                    "avg_fin21": (lambda f: round(float(np.mean(f)), 2) if f else None)([q for q in (_fin21_of(x) for x in v) if q is not None]),
                    "n_A": sum(1 for x in v if x.get("tier") in ("A+", "A"))} for m, v in sorted(by_month.items())}
    sd = [x for x in sur if x.get("status") != "追蹤"]; trl = [x for x in sur if x.get("trail") is not None]
    sg_oos = ((tm.get("surge") or {}).get("oos") or {})
    surge = {"n": len(sur), "done": len(sd), "hits": sum(1 for x in sd if x["status"] == "飆"), "rate": round(sum(1 for x in sd if x["status"] == "飆") / len(sd), 3) if sd else None,
             "tracking": len(sur) - len(sd), "trail_n": len(trl), "trail_win": round(float(np.mean([x["trail"] > 0 for x in trl])), 3) if trl else None,
             "trail_avg": round(float(np.mean([x["trail"] for x in trl])), 2) if trl else None, "bt_hit": sg_oos.get("app_hit", sg_oos.get("hit")), "bt_fin": sg_oos.get("app_fin", sg_oos.get("fin")),
             "since": sur[0]["date"] if sur else None}
    surge["drift"] = bool(surge["done"] >= 20 and surge["bt_hit"] is not None and surge["rate"] is not None and surge["rate"] < surge["bt_hit"] - 0.12)
    return {"treasure": {"all": agg(done), "signal": signal, "by_source": by_source, "tracking": sum(1 for x in tre if x.get("status") == "追蹤"), "by_tier": by_tier, "by_month": by_month,
                         "since": tre[0]["date"] if tre else None, "roles": dict(ROLE), "expect_AAplus": list(EXPECT_AAPLUS), "note_tiers": NOTE_TIERS,
                         "def": "結案採回測定義：21 交易日、峰值取收盤、停損 −8% 取盤中低點 (先後順序)、相對大盤任一日 ≥ +4pt；與 App 本機帳本 (30 日曆天) 不同。"
                                "avg_fin = 結案日報酬 (含停損提早結案)；avg_fin21 = 第 21 個交易日報酬 (同回測口徑)",
                         "note_src": "published = 真實發布 (掃描當天記下)；backfill = 事後回填 (以 10-04 模型重算、等級門檻為樣本內)，真實發布結案 <15 筆前成績以回填為主"},
            "surge": surge}


def alerts(sc: dict, st: dict) -> list[dict]:
    out = []
    D = sc.get("date") or ""
    for r in [x for x in sc.get("treasure") or [] if x["tier"] in ("A+", "A")][:4]:
        # pr2：p 在等級內無排序力 → 寫「模型分」不寫「命中機率」；預期值 55~57% (回測 A 55% / A+ 64%、2026 回填 OOS 重算 57%)
        out.append({"level": "high", "kind": "treasure", "code": r["code"], "date": D, "tier": r["tier"], "p": r["p"],
                    "msg": f"💎 挖寶 {r['tier']} ({D[5:]} 收盤)：{r['code']} {r['name']} 模型分 {r['p']:.2f} — 歷史統計：訊號日收盤起算 21 個交易日，A/A+ 命中約 55~57% (回測 55%/64%，2026 回填重算 57%)；大盤月線下的日子整體較佳。非買賣建議。"})
    for r in (sc.get("surge") or [])[:3]:
        if r.get("ps") is not None and r["ps"] >= ((((st or {}).get("surge") or {}).get("th_top10")) or 0.3977):
            out.append({"level": "mid", "kind": "surge", "code": r["code"], "date": D, "msg": f"🚀 飆股雷達 ({D[5:]})：{r['code']} {r['name']} 20 日內先漲 20% 機率 {round(r['ps'] * 100)}% (回測前 3 名約 33%；高風險)"})
    for t, g in ((st.get("treasure") or {}).get("by_tier") or {}).items():
        if g.get("role") == "signal" and g.get("drift") and g.get("published"):   # pr2：只對訊號級、以真實發布結案判定；B/B+ 不再警示
            gp = g["published"]
            out.append({"level": "mid", "kind": "drift", "code": "", "date": D, "msg": f"⚠ 挖寶 {t} 級真實發布命中 {round(gp['rate'] * 100)}% ({gp['n']} 筆結案)，低於預期 55~57% 達 15pt 以上：請檢視模型"})
    if (st.get("surge") or {}).get("drift"):
        s = st["surge"]; out.append({"level": "mid", "kind": "drift", "code": "", "date": D, "msg": f"⚠ 飆股雷達上線飆股率 {round(s['rate'] * 100)}% (回測 {round(s['bt_hit'] * 100)}%，{s['done']} 筆)：實盤失準"})
    return out


def backfill(prev: dict, tm: dict, dates: list[str], mk: dict | None = None) -> dict:
    """回填：對過去的交易日逐日掃描並記入帳本 (紀錄標 backfill=True；用 D 日以前的日 K，與當日掃描口徑相同)。
    已在 scans 內的日期跳過。回填的推薦不是「當時真的發布過」，只用來快速累積上線口徑的追蹤樣本。"""
    have = {x["date"] for x in prev.get("scans") or []}
    mk = mk if mk is not None else twii_hist()
    for D in sorted(dates):
        if D in have:
            continue
        try:
            sc = scan(tm, D)
        except Exception as e:  # noqa: BLE001
            log.warning("backfill %s: %s", D, e); continue
        if sc.get("date") != D:
            continue
        n0 = len(prev["ledger"]["treasure"]); n1 = len(prev["ledger"]["surge"])
        prev = record(prev, sc, tm, scan_src="backfill")
        for x in prev["ledger"]["treasure"][n0:] + prev["ledger"]["surge"][n1:]:
            x["backfill"] = True
        evaluate(prev, mk, max_fetch=10 ** 6, upto=D)       # 逐日結案 → 追蹤上限/60 日不重複 與即時一致 (K 棒已在快取)
        log.info("backfill %s: %d treasure / %d surge", D, len(prev["ledger"]["treasure"]) - n0, len(prev["ledger"]["surge"]) - n1)
    return prev


def build(date: str | None = None, do_scan: bool = True, backfill_days: int = 0) -> dict:
    tm = M.load_json("treasure_model") or {}
    prev = load_prev()
    mk = twii_hist()
    if backfill_days and tm.get("trees"):
        end = dt.datetime.now(config.TZ).date() - dt.timedelta(days=1)
        days = [d for d in sorted(mk) if (end - dt.timedelta(days=backfill_days)).isoformat() <= d <= end.isoformat()]
        prev = backfill(prev, tm, days, mk)
    sc = scan(tm, date) if (do_scan and tm.get("trees")) else {"date": None}
    if sc.get("date"):
        prev = record(prev, sc, tm, scan_src="live")
    n_fetch = evaluate(prev, mk)
    st = stats(prev, tm)
    out = {"asof": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "scan": sc, "ledger": prev["ledger"], "scans": prev["scans"], "stats": st,
           "alerts": alerts(sc, st | {"surge": {**(st.get("surge") or {}), "th_top10": ((tm.get("surge") or {}).get("th_top10"))}}),
           "model": {"trained_at": tm.get("trained_at"), "th_A": tm.get("th_A"), "th_Aplus": tm.get("th_Aplus"), "oos": (tm.get("oos") or {}).get("tiers"), "surge_oos": (tm.get("surge") or {}).get("oos"),
                     "exit": tm.get("exit"), "surge_exit": (tm.get("surge") or {}).get("exit"), "entry": tm.get("entry"),
                     "expect_AAplus": list(EXPECT_AAPLUS), "note_tiers": NOTE_TIERS, "roles": dict(ROLE), "spec": "pr2 2026-10-05"},
           "n_fetch": n_fetch,
           "disclaimer": "雲端每日盤後掃描 (與 App 相同模型)；成績為歷史追蹤統計，不是個人化投資建議。"}
    try:
        LOCAL.parent.mkdir(parents=True, exist_ok=True)
        LOCAL.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.debug("local save: %s", e)
    return out
