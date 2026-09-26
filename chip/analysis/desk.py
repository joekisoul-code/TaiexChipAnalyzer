"""操盤台 (2026-09-25)：0050、00631L、00663L、00981A、2330、權證 065423 的每日盤後判讀。

資料：FinMind 日K (分割以 chips.KNOWN_SPLITS 還原、股利以 TaiwanStockDividendResult 還原)、HiStock 八大行庫個股、
期交所 OpenAPI TXO (taifex_opt)、TWSE 權證條款 (t187ap37_L)、TWSE MIS 最佳買賣價、統一投信 00981A 每日持股、Yahoo TSM / 美元台幣。

研究依據：操盤台研究 t1~t6 (走動式 OOS、扣成本、每條研究線獨立對抗式驗證)，數字整理在 data/models/desk_evidence.json。
結論摘要 (UI 會逐區塊標示證據強度)：
- 均線狀態對報酬沒有預測力；MA5 系規則 (含破五日線停損) 明顯有害；MA60 加緩衝只是回撤控制。
- 高低點/底部頂部模型樣本外無效；分批等回檔平均輸給一次買進 (保險性質)。
- 選擇權 iv 當 sigma 的 k 日高低分位最準 (比 App ATR+EWMA pinball −6.7%) → 回檔低點「機率帶」；Put/Call OI 牆、max pain 沒有支撐壓力效果。
- 價量分布 HVN、前波高低、均線都不是壓力/支撐；60 日極值與整數只有弱效果。
- 八大行庫是跌買漲賣，控制反轉後沒有預測力 (只描述)。
- 停損與資金防守只降回撤、不增報酬。
這裡的區間/價位/規則是「規則化的研究框架」，不是個人化投資建議。

輸出兩個檔：desk.json (App 讀) 與 desk_archive.json (八大行庫個股歸檔、選擇權特徵歷史、00981A 持股快照；跨次聯集保存)。
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import math
import re

import numpy as np
import pandas as pd

from .. import config
from ..http import cached, session
from ..sources import exdiv, finmind, global_markets, histock, taifex_opt, twse
from . import chips, desk_bands, gov8_dist

log = logging.getLogger(__name__)
PAGES_URL = "https://joekisoul-code.github.io/TaiexChipAnalyzer/"
DESK = [
    {"id": "0050", "name": "元大台灣50", "kind": "etf", "zz": (0.05, 0.08), "ladder": "1x", "cond": "0050", "touch": "0050", "limit": 0.10},
    {"id": "00631L", "name": "元大台灣50正2", "kind": "lev2", "zz": (0.10, 0.16), "ladder": "2x", "cond": "SYN2X", "touch": "00631L", "under": "0050", "limit": 0.20},
    {"id": "00663L", "name": "國泰臺灣加權正2", "kind": "lev2", "zz": (0.10, 0.16), "ladder": "2x", "cond": "SYN2X", "touch": "00663L", "under": "TWII", "limit": 0.20},
    {"id": "00981A", "name": "主動統一台股增長", "kind": "active", "zz": (0.08, 0.12), "ladder": None, "cond": None, "touch": None, "limit": 0.10},
    {"id": "2330", "name": "台積電", "kind": "stock", "zz": (0.08, 0.12), "ladder": "2330", "cond": "2330L", "touch": "2330", "limit": 0.10},
]
WARRANT = {"id": "065423", "name": "台股增元大5A購01", "underlying": "00981A"}
GOV8_IDS = ["0050", "00631L", "00663L", "00981A", "2330"]
MAS = (5, 10, 30, 60)
ARCHIVE_MAX = 1500
OPT_HIST_MAX = 3000
LEDGER_MAX_DAYS = 400                       # 機率帶上線帳本保留的發布日數
LEDGER_KS = (1, 5, 20)
QS = ("low10", "low20", "high80", "high90")
Q_NOMINAL = {"low10": 0.10, "low20": 0.20, "high80": 0.20, "high90": 0.10}   # 名目觸及率
ACCOUNT, RISK_R = 10_000_000, 0.01          # 部位大小示例：帳戶 1,000 萬、單筆風險 1%
VOL_TARGETS = (0.15, 0.20)
R_F = 0.017
WEIGHTS_2330 = {"taiex": 41.0, "etf0050": 56.0, "etf00981A": 9.8, "asof": "加權 2026-08-31 (推估 09-24)、0050 2026-09-24、00981A 2026-09-24"}
DISCLAIMER = "研究與規則框架 (走動式回測、扣成本、對抗式驗證)，不是個人化投資建議；區間是機率帶，不是進出場訊號。"


def _r(v, n=2):
    try:
        f = float(v)
        return round(f, n) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _evidence() -> dict:
    try:
        from ..predict import model as _M
        return _M.load_json("desk_evidence") or {}
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------ 已發布 (跨次累積)
def published(name: str = "desk_archive") -> dict:
    try:
        r = session().get(PAGES_URL.rstrip("/") + f"/data/{name}.json", timeout=30)
        if r.ok and r.text.strip().startswith("{"):
            return r.json() or {}
    except Exception as e:  # noqa: BLE001
        log.debug("published %s.json unavailable: %s", name, e)
    return {}


RAW_URL = "https://raw.githubusercontent.com/joekisoul-code/TaiexChipAnalyzer/gh-pages/data/"
LOCAL_ARCHIVE = config.CACHE_DIR / "desk_archive.json"      # Actions 的 data/cache 會被 actions/cache 保留 → 第二份備援


def _merge_archives(a: dict, b: dict) -> dict:
    """兩份歸檔取聯集 (同日期以 a 為準)；避免任一來源讀不到時歷史被洗掉。"""
    a, b = a or {}, b or {}
    out = {"gov8": {}, "opt_hist": [], "opt_last": None, "holdings_00981A": {}, "band_ledger": [], "exdiv": {}, "fresh_log": []}
    for sid in set((a.get("gov8") or {}).keys()) | set((b.get("gov8") or {}).keys()):
        rows = {r["date"]: r for r in ((b.get("gov8") or {}).get(sid) or []) if r.get("date")}
        rows.update({r["date"]: r for r in ((a.get("gov8") or {}).get(sid) or []) if r.get("date")})
        out["gov8"][sid] = [rows[d] for d in sorted(rows)]
    oh = {r["date"]: r for r in (b.get("opt_hist") or []) if r.get("date")}
    oh.update({r["date"]: r for r in (a.get("opt_hist") or []) if r.get("date")})
    out["opt_hist"] = [oh[d] for d in sorted(oh)]
    la, lb = a.get("opt_last") or {}, b.get("opt_last") or {}
    out["opt_last"] = la if str(la.get("date") or "") >= str(lb.get("date") or "") else lb
    bl = {(r.get("date"), r.get("sid"), r.get("k")): r for r in (b.get("band_ledger") or []) if r.get("date")}
    bl.update({(r.get("date"), r.get("sid"), r.get("k")): r for r in (a.get("band_ledger") or []) if r.get("date")})
    out["band_ledger"] = [bl[k] for k in sorted(bl)]
    ea, eb = a.get("exdiv") or {}, b.get("exdiv") or {}
    out["exdiv"] = {**eb, **{k: v for k, v in ea.items() if str((v or {}).get("asof") or "") >= str((eb.get(k) or {}).get("asof") or "")}}
    fl = {(r.get("run"), r.get("src")): r for r in (b.get("fresh_log") or []) + (a.get("fresh_log") or []) if r.get("run")}
    out["fresh_log"] = [fl[k] for k in sorted(fl)][-600:]
    ha, hb = a.get("holdings_00981A") or {}, b.get("holdings_00981A") or {}
    out["holdings_00981A"] = ha if str((ha.get("last") or {}).get("date") or "") >= str((hb.get("last") or {}).get("date") or "") else hb
    return out


def _repo_archive_path():
    from ..predict import model as _M
    return _M.MODEL_DIR / "desk_archive.json"      # 每週 train 模式會 commit 回 main → 第四份備援 (最多落後一週，八大 63 日窗足以補齊)


def load_archive() -> dict:
    """上次發布 (Pages，失敗改 raw.githubusercontent) ∪ 本機快取 (actions/cache) ∪ repo 內每週備份。"""
    pub = published("desk_archive")
    if not pub:
        try:
            r = session().get(RAW_URL + "desk_archive.json", timeout=30)
            if r.ok and r.text.strip().startswith("{"):
                pub = r.json() or {}
        except Exception as e:  # noqa: BLE001
            log.debug("raw desk_archive: %s", e)
    locs = []
    for p in (LOCAL_ARCHIVE, _repo_archive_path()):
        try:
            if p.exists():
                locs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            log.debug("desk_archive %s: %s", p, e)
    arc = pub
    for loc in locs:
        arc = _merge_archives(arc, loc)
    arc = _merge_archives(arc, {}) if not locs else arc
    arc["_loaded"] = bool(pub or locs)
    if not arc["_loaded"]:
        log.warning("desk_archive: 已發布、本機快取、repo 備份都讀不到 → 八大行庫歸檔只剩 HiStock 近 63 日")
    return arc


def save_archive(archive: dict) -> None:
    body = json.dumps({k: v for k, v in archive.items() if not k.startswith("_")}, ensure_ascii=False, default=str)
    for p in (LOCAL_ARCHIVE, _repo_archive_path()):
        try:
            p.write_text(body, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.debug("save desk_archive %s: %s", p, e)


# ------------------------------------------------------------------ 價格
def _fm(getter, *a):
    """FinMind 日K：快取 key 以今天為界、TTL 12h → 清晨抓到的 D−1 會撐到下午。資料落後最後交易日且已過 14:00 時改 10 分鐘 TTL 重抓。"""
    df = getter(*a)
    last_td = _last_trading_day()
    if last_td and df is not None and not df.empty and str(df["date"].iloc[-1])[:10] < last_td:
        try:
            df2 = getter(*a, ttl=config.TTL_INTRADAY)
            if df2 is not None and not df2.empty:
                df = df2
        except Exception as e:  # noqa: BLE001
            log.debug("finmind refetch: %s", e)
    return df


def prices(sid: str, days: int = 1100, split_adjust: bool = True) -> pd.DataFrame:
    """date, open, high, low, close, volume (股)；分割已還原到今日股數尺度 (= 實際可交易價尺度)。"""
    df = _fm(finmind.stock_price, sid, (dt.date.today() - dt.timedelta(days=days)).isoformat())
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] > 0)].copy()
    df["date"] = df["date"].astype(str).str[:10]
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if split_adjust:                                  # 權證價格常有 >2.5 倍的真實漲跌，不做分割偵測
        df, _ = chips.adjust_splits(df, sid)
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def dividends(sid: str, start: str) -> list[dict]:
    try:
        dv = finmind.fetch("TaiwanStockDividendResult", sid, start)
    except Exception as e:  # noqa: BLE001
        log.debug("dividend %s: %s", sid, e)
        return []
    out = []
    for _, r in (dv if dv is not None else pd.DataFrame()).iterrows():
        D, B = float(r.get("stock_and_cache_dividend") or 0), float(r.get("before_price") or 0)
        if D > 0 and B > 0:
            out.append({"date": str(r.get("date"))[:10], "div": D, "before": B})
    return out


def adj_close(px: pd.DataFrame, divs: list[dict]) -> pd.Series:
    """含息還原收盤 (最新一日 = 原始收盤)：除息日前的價格乘 (1 − 股利/除息前價)。"""
    c = px["close"].astype(float).reset_index(drop=True)
    f = pd.Series(1.0, index=c.index)
    dates = px["date"].astype(str).str[:10].reset_index(drop=True)
    for d in divs:
        f[dates < d["date"]] *= (1 - d["div"] / d["before"])
    return c * f


def twii(days: int = 1100) -> pd.DataFrame:
    df = _fm(finmind.taiex_price, (dt.date.today() - dt.timedelta(days=days)).isoformat())
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    df["date"] = df["date"].astype(str).str[:10]
    cols = [c for c in ("date", "high", "low", "close", "volume") if c in df]
    return df[cols].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _lr(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _wilder_atr(h, l, c, n=14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(c: pd.Series, n=14) -> float | None:
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] > 0 else np.inf
    return _r(100 - 100 / (1 + rs), 1)


def _beta_idio(a: pd.Series, m: pd.Series, n: int) -> tuple[float | None, float | None]:
    """a, m：日期對齊的含息收盤。回傳 (OLS β, 殘差日波動 %)。"""
    j = pd.concat([_lr(a), _lr(m)], axis=1).dropna().tail(n)
    if len(j) < max(40, n // 3):
        return None, None
    x, y = j.iloc[:, 1].values, j.iloc[:, 0].values
    b = float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))
    idio = math.sqrt(max(0.0, float(np.var(y, ddof=1) - b * b * np.var(x, ddof=1)))) * 100
    return b, idio


# ------------------------------------------------------------------ 1. 均線 MA5/10/30/60
def ma_block(px: pd.DataFrame, adj: pd.Series, meta: dict, ev: dict) -> dict:
    raw = px["close"].astype(float).reset_index(drop=True)
    if len(raw) < 65:
        return {}
    last = float(raw.iloc[-1])
    out: dict = {"date": str(px["date"].iloc[-1])[:10], "close": _r(last, 2), "ma": {}}
    m_raw = {n: raw.rolling(n).mean() for n in MAS}
    m_adj = {n: adj.rolling(n).mean() for n in MAS}
    for n in MAS:
        out["ma"][str(n)] = {"raw": _r(m_raw[n].iloc[-1], 2), "adj": _r(m_adj[n].iloc[-1], 2),
                             "bias": _r((adj.iloc[-1] / m_adj[n].iloc[-1] - 1) * 100, 2),
                             "slope": int(np.sign(m_adj[n].iloc[-1] - m_adj[n].iloc[-2])),
                             "slope5": _r((m_adj[n].iloc[-1] / m_adj[n].iloc[-6] - 1) * 100, 2),
                             "deduct_next": _r(raw.iloc[-n], 2),                               # 次日收盤高於扣抵價 → 看盤價均線上彎
                             "deduct_next_adj": _r(adj.iloc[-n], 2),                           # 含息口徑 (除息後幾日兩者方向可能不同)
                             "close_eq_next": _r(raw.iloc[-(n - 1):].mean(), 2) if n > 1 else None,   # 次日收盤 = 此價時與均線等高 (看盤價)
                             "deduct_path10": [_r(v, 2) for v in raw.iloc[-n:-n + 10].tolist()] if n > 10 else None}   # 未來 10 日扣抵序列

    def align(mm):
        v = [mm[n].iloc[-1] for n in MAS]
        return "完全多頭" if all(v[i] > v[i + 1] for i in range(3)) else "完全空頭" if all(v[i] < v[i + 1] for i in range(3)) else "混合"
    out["align_raw"], out["align_adj"] = align(m_raw), align(m_adj)
    ca = float(adj.iloc[-1])
    tk = _tick(last, meta.get("kind") != "stock")
    out["at_threshold"] = [n for n in MAS if abs(ca - m_adj[n].iloc[-1]) < tk / 2]     # 半個跳動單位內 = 在門檻上 (不算站上)
    out["above"] = [n for n in MAS if ca > m_adj[n].iloc[-1] and n not in out["at_threshold"]]
    out["n_above"] = len(out["above"])
    crosses = []
    for a_, b_ in ((5, 10), (10, 30), (30, 60)):
        d = (m_adj[a_] - m_adj[b_])
        rec = {"pair": f"{a_}/{b_}", "state": "多" if d.iloc[-1] > 0 else "空"}
        for k in range(1, 121):
            if len(d) > k + 1 and np.sign(d.iloc[-k]) != np.sign(d.iloc[-k - 1]) and d.iloc[-k] != 0:
                rec.update({"last": "黃金交叉" if d.iloc[-k] > 0 else "死亡交叉", "date": str(px["date"].iloc[-k])[:10], "days_ago": k - 1})
                break
        S = lambda n: float(adj.iloc[-(n - 1):].sum()) if n > 1 else 0.0   # noqa: E731   含息價 (今日 = 實際價尺度)，與交叉狀態同口徑
        trig = (a_ * S(b_) - b_ * S(a_)) / (b_ - a_)
        rec["trigger_next"] = _r(trig, 2)
        rec["feasible_next"] = bool(trig > 0 and abs(trig / last - 1) <= meta.get("limit", 0.1))
        crosses.append(rec)
    out["crosses"] = crosses
    out["ex_div_distortion"] = out["align_raw"] != out["align_adj"]
    # 歷史同狀態 (20 日，研究 t1；描述，CI 含 0 → 不顯著)
    key = meta["id"] if meta["id"] in (ev.get("ma_states") or {}) else "0050"
    st_tab = ((ev.get("ma_states") or {}).get(key) or {}).get("states") or {}
    names = [{"完全多頭": "完全多頭排列", "完全空頭": "完全空頭排列", "混合": "混合排列"}[out["align_adj"]], f"站上均線數={out['n_above']}",
             "站上MA60" if 60 in out["above"] else "跌破MA60", "MA30>MA60" if m_adj[30].iloc[-1] > m_adj[60].iloc[-1] else "MA30<MA60"]
    hist = []
    for nm in names:
        s = st_tab.get(nm)
        if s:
            ci = s.get("ci") or [None, None]
            hist.append({"state": nm, **s, "sig": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))})
    out["hist"] = hist
    out["hist_from"] = key if key == meta["id"] else "0050 (00981A 資料不足，借用)"
    out["uncond"] = {k: ((ev.get("ma_states") or {}).get(key) or {}).get(k) for k in ("uncond_mean", "uncond_win")}
    rk = ((ev.get("ma_risk") or {}).get(key) or {}).get("states") or {}
    out["risk"] = rk.get("站上MA60" if 60 in out["above"] else "跌破MA60")
    out["risk_borrowed"] = key != meta["id"]
    out["notes"] = ev.get("ma_notes") or []
    return out


def _tick(p: float, etf: bool) -> float:
    """台股升降單位 (ETF：<50 0.01、其餘 0.05；股票：分級)。"""
    if etf:
        return 0.01 if p < 50 else 0.05
    for lim, t in ((10, .01), (50, .05), (100, .1), (500, .5), (1000, 1.0)):
        if p < lim:
            return t
    return 5.0


# ------------------------------------------------------------------ 2. 回撤控制價位 (停損區) + 3. 資金防守
def overlay_block(meta: dict, px: pd.DataFrame, adj: pd.Series, ev: dict) -> dict:
    sid = meta["id"]
    raw = px["close"].astype(float).reset_index(drop=True)
    k = adj / raw                                     # 含息因子 (今日 = 1)
    ah, al = px["high"].astype(float).reset_index(drop=True) * k, px["low"].astype(float).reset_index(drop=True) * k
    atr = _wilder_atr(ah, al, adj, 14)
    ma20, ma60, ma30 = adj.rolling(20).mean(), adj.rolling(60).mean(), adj.rolling(30).mean()
    c = float(adj.iloc[-1])
    hh22 = float(adj.tail(22).max())
    spec = (ev.get("overlay") or {}).get(sid) or {}
    b = float(spec.get("b") or 0.0)
    S59 = float(adj.iloc[-59:].sum())
    exit_next = S59 * (1 - b) / (59 + b)              # 次日收盤低於此價 → 收盤 < MA60×(1−b)
    reenter_next = S59 / 59                            # 次日收盤高於此價 → 站回 MA60
    # MA60 遲滯狀態機 (t 收盤判定，t+1 開盤執行)
    state, last_exit, last_entry = 1, None, None
    for i in range(60, len(adj)):
        if state == 1 and adj.iloc[i] < ma60.iloc[i] * (1 - b):
            state, last_exit = 0, str(px["date"].iloc[i])[:10]
        elif state == 0 and adj.iloc[i] > ma60.iloc[i]:
            state, last_entry = 1, str(px["date"].iloc[i])[:10]
    trail = 0.15 if meta["kind"] == "lev2" else 0.10
    rule_on = spec.get("rule") in ("a", "a2")
    lv = [
        {"key": "ma20", "name": "MA20 (警戒)", "price": _r(ma20.iloc[-1], 2)},
        {"key": "ma20_98", "name": "MA20×0.98", "price": _r(ma20.iloc[-1] * 0.98, 2)},
        {"key": "ma20_95", "name": "MA20×0.95", "price": _r(ma20.iloc[-1] * 0.95, 2)},
        {"key": "atr3", "name": "22日高−3×ATR", "price": _r(hh22 - 3 * atr.iloc[-1], 2)},
        {"key": "ma60", "name": "MA60", "price": _r(ma60.iloc[-1], 2)},
        {"key": "ma60_exit", "name": f"MA60×(1−{b:.0%}) 次日出場門檻" if b else "MA60 次日出場門檻", "price": _r(exit_next, 2)},
        {"key": "trail", "name": f"22日高回落 {trail:.0%}", "price": _r(hh22 * (1 - trail), 2)},
        {"key": "ma20_97", "name": "MA20×0.97", "price": _r(ma20.iloc[-1] * 0.97, 2)},
        {"key": "ma60_98", "name": "MA60×0.98", "price": _r(ma60.iloc[-1] * 0.98, 2)},
    ]
    for x in lv:
        x["dist"] = _r((x["price"] / c - 1) * 100, 2) if x["price"] else None
    L = {x["key"]: x for x in lv}
    primary = None
    if rule_on:
        primary = {"name": L["ma60_exit"]["name"], "price": L["ma60_exit"]["price"], "dist": L["ma60_exit"]["dist"], "reenter": _r(reenter_next, 2)}
    out = {"levels": lv, "primary": primary, "rule": spec, "rule_on": rule_on,
           "atr14": _r(atr.iloc[-1], 3), "atr_pct": _r(atr.iloc[-1] / c * 100, 2), "hh22": _r(hh22, 2),
           "e_rule": {"state": "多" if ma30.iloc[-1] > ma60.iloc[-1] else "空", "gap": _r(ma30.iloc[-1] - ma60.iloc[-1], 3)}}
    if rule_on:       # 不啟用/只描述的標的不輸出進出狀態 (避免替研究沒選的規則標進出紀錄)
        out.update({"state": "持有" if state else "空手 (等站回 MA60)", "last_exit": last_exit, "last_entry": last_entry})
    # 部位大小示例的停損：有規則用規則；00981A 用研究列的 MA20×0.97 (參考)；2330 預設不啟用 → 不給示例
    if primary:
        out["_stop"] = (primary["price"], primary["name"], False)
    elif sid == "00981A":
        out["_stop"] = (L["ma20_97"]["price"], "MA20×0.97 (參考，無回測)", True)
    return out


def sizing(o: dict, dfn: dict, c: float) -> dict | None:
    """單筆風險 R 的股數，並套研究 D3 上限 min(1, 波動目標 15% 曝險) × 帳戶；停損離現價 < 0.5 ATR 視為過近不給。"""
    st = o.pop("_stop", None)
    if not st:
        return None
    stop, name, ref = st
    atr = o.get("atr14") or 0
    if not stop or c - stop < 0.5 * atr:
        return {"note": f"現價距「{name}」不到 0.5 ATR，部位示例不適用"} if stop else None
    ex = (dfn.get("exposure") or {}).get("15") or {}
    w = ex.get("iv") or ex.get("rv20") or 1.0
    raw = int(ACCOUNT * RISK_R / (c - stop))
    cap = int(min(1.0, w) * ACCOUNT / c)
    sh = min(raw, cap)
    return {"account": ACCOUNT, "risk_r": RISK_R, "stop": stop, "stop_name": name, "reference_only": ref, "shares": sh, "lots": sh // 1000, "odd": sh % 1000,
            "capital_pct": _r(sh * c / ACCOUNT * 100, 1), "capped": raw > cap, "cap_w": _r(w, 2)}


def defense_block(meta: dict, adj: pd.Series, dates: pd.Series, tw: pd.DataFrame, opt: dict, under: pd.Series | None) -> dict:
    lr = _lr(adj)
    rv20, rv60 = float(lr.tail(20).std() * math.sqrt(252)), float(lr.tail(60).std() * math.sqrt(252))
    j = pd.DataFrame({"date": dates.values, "a": adj.values}).merge(tw[["date", "close"]], on="date", how="inner")
    vr = float(_lr(j["a"]).tail(60).std() / _lr(j["close"]).tail(60).std()) if len(j) > 70 else None
    iv21 = opt.get("iv21")
    sig_iv = iv21 * vr if (iv21 and vr) else None
    out = {"rv20": _r(rv20, 4), "rv60": _r(rv60, 4), "vol_ratio60": _r(vr, 3), "sigma_iv": _r(sig_iv, 4), "exposure": {}}
    for T in VOL_TARGETS:
        out["exposure"][f"{int(T * 100)}"] = {"iv": _r(min(1.0, T / sig_iv), 2) if sig_iv else None, "rv20": _r(min(1.0, T / rv20), 2), "rv60": _r(min(1.0, T / rv60), 2)}
    dd = adj / adj.cummax() - 1
    out["dd_3y"] = _r(dd.iloc[-1] * 100, 2)                 # 資料視窗約 3 年 (1100 日曆天)
    out["dd_250"] = _r((adj.iloc[-1] / adj.tail(250).max() - 1) * 100, 2)
    th = (-0.20, -0.30) if meta["kind"] == "lev2" else (-0.10, -0.15)
    pk = float(adj.tail(250).max())
    d_ = float(adj.iloc[-1] / pk - 1)
    out["dd_budget"] = {"level": 1.0 if d_ > th[0] else (0.5 if d_ > th[1] else 0.0), "peak": _r(pk, 2), "half_px": _r(pk * (1 + th[0]), 2), "core_px": _r(pk * (1 + th[1]), 2),
                        "th": [th[0] * 100, th[1] * 100], "reset": "收盤站回 MA60 回滿",
                        "scope": "單一標的近似 (以 250 日高為基準)；組合層請以自己的淨值計", "wf_core": 0.0}
    if meta["kind"] == "lev2" and under is not None:
        u = under.reset_index(drop=True)
        ru = _lr(u)
        su_rv20, su_rv60 = float(ru.tail(20).std() * math.sqrt(252)), float(ru.tail(60).std() * math.sqrt(252))
        su_iv = (iv21 * (1.0 if meta.get("under") == "TWII" else (float(ru.tail(60).std() / _lr(tw["close"]).tail(60).std()) if len(tw) > 70 else 1.0))) if iv21 else None
        rL, rU = adj.pct_change().tail(60), u.pct_change().tail(60)
        n = min(len(rL.dropna()), len(rU.dropna()))
        rL, rU = rL.dropna().tail(n).values, rU.dropna().tail(n).values
        gap60 = float(np.log1p(rL).sum() - 2 * np.log1p(rU).sum())
        theo60 = float((np.log1p(2 * rU) - 2 * np.log1p(rU)).sum())
        out["letf"] = {"under": meta.get("under"), "drag_iv": _r(-(su_iv ** 2) * 100, 2) if su_iv else None, "drag_rv20": _r(-su_rv20 ** 2 * 100, 2), "drag_rv60": _r(-su_rv60 ** 2 * 100, 2),
                       "gap60": _r(gap60 * 100, 2), "theo60": _r(theo60 * 100, 2), "resid60": _r((gap60 - theo60) * 100, 2),
                       "breakeven": {str(h): _r((su_iv or su_rv20) * math.sqrt(h / 252) * 100, 1) for h in (20, 60, 120)}}
    return out


# ------------------------------------------------------------------ 4. 選擇權 iv → k 日高低點機率帶 (回檔低點)
def range_block(meta: dict | None, close: float, beta: float | None, idio: float | None, opt: dict, ev: dict, cal: list[str], live: bool = False) -> dict:
    """live = 今日滾動乘數 (desk_bands.range_live 成功)；否則為凍結於 desk_evidence 的 2026-09-24 乘數。"""
    rg = ev.get("range") or {}
    ivk = opt.get("ivk") or {}
    if not ivk or not close:
        return {}
    sid = meta["id"] if meta else "TWII"
    a = (rg.get("assets") or {}).get(sid) or {}
    calibrated = bool(meta and a.get("m") and all(a["m"].get(k) and all(v is not None for v in a["m"][k].values()) for k in a["m"]))
    if not meta:
        method = "指數 (加權自身高低點，每日滾動 750 日)" if live else "指數 (台指期校準，凍結於 2026-09-24)"
    else:
        method = ("校準 (每日滾動 750 日)" if live else "校準 (凍結於 2026-09-24)") if calibrated else "未校準映射 (資料短，低信心)"
    mver = ("twii_roll750" if live else "tx_frozen_20260924") if not meta else (("roll750" if live else "frozen_20260924") if calibrated else "mapped_tx")
    out = {"close": _r(close, 2), "beta": _r(beta, 3), "idio": _r(idio, 3), "method": method, "mver": mver, "live": bool(live),
           "levels": {}, "ends": {}}
    for k in (1, 2, 3, 5, 10, 20):
        v = ivk.get(str(k))
        if not v:
            continue
        sk = v / math.sqrt(252) * 100
        row = {}
        for q in ("low10", "low20", "high80", "high90"):
            cell = (((rg.get("index") or {}).get(str(k)) or {}).get(q) or {})
            m_idx = cell.get("m") if not meta else cell.get("m_tx", cell.get("m"))       # 未校準映射沿用台指期校準值
            if not meta:
                pct = m_idx * sk if m_idx is not None else None
                m_exp = (((rg.get("index") or {}).get(str(k)) or {}).get(q) or {}).get("m_exp")
                row[q] = {"pct": _r(pct, 2), "px": _r(close * (1 + pct / 100), 0) if pct is not None else None,
                          "pct_cons": _r(m_exp * sk, 2) if m_exp is not None else None}
                continue
            if beta is None or idio is None:
                continue
            sa = sk * math.sqrt(beta * beta + (idio / sk) ** 2)
            if calibrated:
                pct = a["m"][str(k)][q] * sa
            elif m_idx is not None:
                pct = (math.exp(math.log(1 + m_idx * sk / 100) * math.sqrt(beta * beta + (idio / sk) ** 2)) - 1) * 100
            else:
                continue
            row[q] = {"pct": _r(pct, 2), "px": _r(close * (1 + pct / 100), 2)}
        if row.get("low10") and row.get("low20") and row["low10"]["pct"] is not None and row["low20"]["pct"] is not None and row["low10"]["pct"] > row["low20"]["pct"]:
            row["low10"], row["low20"] = row["low20"], row["low10"]
        out["levels"][str(k)] = row
        if len(cal) >= k:
            out["ends"][str(k)] = cal[k - 1]
    return out


# ------------------------------------------------------------------ 5. 高低點 / 波段 / 承接價
def _zigzag(c: pd.Series, dates: pd.Series, th: float, raw: pd.Series | None = None) -> dict:
    """收盤 zigzag (與研究 t2 common.zigzag 同邏輯：同價不更新轉折、先判上漲確認)。幅度用含息價 c；顯示價位用原始收盤 raw。"""
    cv = c.values
    trend, hi_i, lo_i, piv = 0, 0, 0, []
    for i in range(1, len(cv)):
        if trend == 0:
            if cv[i] > cv[hi_i]:
                hi_i = i
            if cv[i] < cv[lo_i]:
                lo_i = i
            if cv[i] >= cv[lo_i] * (1 + th):
                piv.append(("L", lo_i)); trend, hi_i = 1, i
            elif cv[i] <= cv[hi_i] * (1 - th):
                piv.append(("H", hi_i)); trend, lo_i = -1, i
        elif trend == 1:
            if cv[i] > cv[hi_i]:
                hi_i = i
            elif cv[i] <= cv[hi_i] * (1 - th):
                piv.append(("H", hi_i)); trend, lo_i = -1, i
        else:
            if cv[i] < cv[lo_i]:
                lo_i = i
            elif cv[i] >= cv[lo_i] * (1 + th):
                piv.append(("L", lo_i)); trend, hi_i = 1, i
    if not piv:
        return {}
    rp = raw if raw is not None else c
    t, pi = piv[-1]
    ext = hi_i if trend == 1 else lo_i
    conf = cv[ext] * (1 - th) if trend == 1 else cv[ext] * (1 + th)
    return {"th": th, "leg": "上升段" if trend == 1 else "下降段",
            "last_pivot": {"type": "低點" if t == "L" else "高點", "date": str(dates.iloc[pi])[:10], "px": _r(rp.iloc[pi], 2), "px_adj": _r(cv[pi], 2)},
            "extreme": {"date": str(dates.iloc[ext])[:10], "px": _r(rp.iloc[ext], 2), "px_adj": _r(cv[ext], 2)},
            "leg_pct": _r((cv[ext] / cv[pi] - 1) * 100, 2), "leg_days": int(ext - pi), "from_extreme": _r((cv[-1] / cv[ext] - 1) * 100, 2),
            "confirm_px": _r(conf * float(rp.iloc[-1]) / cv[-1], 2)}      # 今日價尺度


def pivot_block(meta: dict, px: pd.DataFrame, adj: pd.Series, ev: dict) -> dict:
    sid = meta["id"]
    dates = px["date"].astype(str).reset_index(drop=True)
    raw = px["close"].astype(float).reset_index(drop=True)
    c = adj
    out: dict = {"hilo": {}}
    for n in (20, 60, 250):
        t = c.tail(n)
        ih, il = int(t.idxmax()), int(t.idxmin())
        out["hilo"][str(n)] = {"hi": _r(raw.iloc[ih], 2), "hi_date": dates.iloc[ih], "hi_dist": _r((c.iloc[-1] / c.iloc[ih] - 1) * 100, 2),
                               "lo": _r(raw.iloc[il], 2), "lo_date": dates.iloc[il], "lo_dist": _r((c.iloc[-1] / c.iloc[il] - 1) * 100, 2)}
    out["rsi14"] = _rsi(c)
    out["bias"] = {str(n): _r((c.iloc[-1] / c.tail(n).mean() - 1) * 100, 2) for n in (20, 60, 200) if len(c) >= n}
    out["zigzag"] = [z for z in (_zigzag(c, dates, th, raw) for th in meta["zz"]) if z]
    zs = ((ev.get("zigzag") or {}).get(sid) or {})
    out["zigzag_stats"] = {th: zs.get(f"{th:.2f}") or zs.get(str(th)) for th in meta["zz"]}
    h60 = float(c.tail(60).max())
    dd60 = c.iloc[-1] / h60 - 1
    out["dd60"] = _r(dd60 * 100, 2)
    if meta.get("ladder"):
        lvls = (ev.get("ladder_levels") or {}).get(meta["ladder"]) or []
        touch = ((ev.get("touch") or {}).get(meta.get("touch")) or {})
        out["ladder"] = []
        for lv in lvls:
            p = h60 * (1 - lv)
            t = touch.get(f"{lv:.2f}") or touch.get(str(lv)) or {}
            out["ladder"].append({"lvl": lv, "price": _r(p, 2), "dist": _r((p / c.iloc[-1] - 1) * 100, 2),
                                  "touch_hist": t if dd60 > -0.02 else None, "touch_anchor": _r(-lv * 100, 1)})   # 基準率是「距現價 −lvl」的收盤觸及率
        out["ladder_cost"] = ((ev.get("ladder") or {}).get(sid) or {})
        out["ladder_cluster_ci"] = ((ev.get("ladder_cluster_ci") or {}).get(sid) or {})
        cd = ((ev.get("cond_dd") or {}).get(meta.get("cond")) or {})
        lv_keys = [float(k) for k in (cd.get("levels") or {})]
        out["cond_min"] = min(lv_keys) if lv_keys else None
        crossed = [d for d in lv_keys if dd60 <= -d]
        if crossed:
            d = max(crossed)
            x = cd["levels"][next(k for k in cd["levels"] if float(k) == d)]
            trig = h60 * (1 - d)
            cur = float(c.iloc[-1])
            w = c.tail(60).reset_index(drop=True)
            ih = int(w.idxmax())
            hit = w.iloc[ih:][w.iloc[ih:] <= trig]
            tq = {q: (_r(trig * (1 + x[f"add_{q}"]), 2) if x.get(f"add_{q}") is not None else None) for q in ("q50", "q80", "q90")}
            deep = {"TWII": 0.20, "0050": 0.20, "2330L": 0.15, "SYN2X": 0.30}.get(meta.get("cond"), 9.0)   # verify_pivots：這些層級以上覆蓋不足
            out["cond_dd"] = {"d": d, "src": meta.get("cond"), "R": cd.get("swing_R"), "period": cd.get("period"), "n": x.get("n"),
                              "undercovered": bool(x.get("undercovered") or x.get("cov80") is None or d >= deep - 1e-9),
                              "trough_q50": tq["q50"], "trough_q80": tq["q80"], "trough_q90": tq["q90"],
                              "passed": [q for q in ("q50", "q80", "q90") if tq[q] is not None and cur <= tq[q]],
                              "beyond_table": bool(dd60 < -max(lv_keys) and tq["q90"] is not None and cur <= tq["q90"]),
                              "days_since_trigger": int(len(w) - 1 - hit.index[0]) if len(hit) else 0,
                              "days_q50": x.get("days_q50"), "rec_days_q50": x.get("rec_days_q50")}
    return out


# ------------------------------------------------------------------ 6. 價位帶 (壓力/支撐參考；研究：多數無額外資訊)
def _profile(h, l, v, center, w):
    jmin = int(np.floor((l.min() - center) / w + 0.5)); jmax = int(np.ceil((h.max() - center) / w - 0.5))
    js = np.arange(jmin, jmax + 1)
    edges = center + (np.r_[js, jmax + 1] - 0.5) * w
    rng = h - l
    pt = rng <= 1e-12
    frac = np.where(pt[:, None], (edges[None, :] >= l[:, None]).astype(float),
                    np.clip((edges[None, :] - l[:, None]) / np.where(pt, 1.0, rng)[:, None], 0, 1))
    return edges, np.diff((v[:, None] * frac).sum(0))


def _share(edges, vol, a, b):
    tot = vol.sum()
    if tot <= 0:
        return 0.0
    s = 0.0
    for i in range(len(vol)):
        ov = max(0.0, min(edges[i + 1], b) - max(edges[i], a))
        if ov > 0:
            s += vol[i] * ov / (edges[i + 1] - edges[i])
    return s / tot


def _value_area(edges, vol, frac=0.7):
    i = int(np.argmax(vol)); lo = hi = i; acc = vol[i]; tot = vol.sum()
    while acc < frac * tot and (lo > 0 or hi < len(vol) - 1):
        a = vol[lo - 1] if lo > 0 else -1; b = vol[hi + 1] if hi < len(vol) - 1 else -1
        if a >= b:
            lo -= 1; acc += a
        else:
            hi += 1; acc += b
    return float(edges[lo]), float(edges[hi + 1]), float((edges[i] + edges[i + 1]) / 2)


def zones_block(px: pd.DataFrame, is_index: bool = False) -> dict:
    d = px.reset_index(drop=True)
    h, l, c = (d[k].astype(float).values for k in ("high", "low", "close"))
    v = d["volume"].astype(float).values if "volume" in d else np.ones(len(d))
    n = len(d); t = n - 1
    if n < 130:
        return {}
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.column_stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=1)
    atr = float(pd.Series(tr).rolling(20).mean().iloc[-1])
    ma20, ma60 = float(pd.Series(c).rolling(20).mean().iloc[-1]), float(pd.Series(c).rolling(60).mean().iloc[-1])
    w, close = 0.25 * atr, float(c[t])
    cands, prof, va = [], {}, {}
    for W in (60, 120):
        a = t - W + 1
        edges, vol = _profile(h[a:t + 1], l[a:t + 1], v[a:t + 1], close, w)
        prof[W] = (edges, vol)
        lo_, hi_, poc = _value_area(edges, vol)
        va[str(W)] = {"VAL": _r(lo_, 2), "VAH": _r(hi_, 2), "POC": _r(poc, 2)}
        mids = (edges[:-1] + edges[1:]) / 2
        sm = np.convolve(vol, np.ones(3) / 3, "same")
        pct = np.searchsorted(np.sort(vol), vol, "left") / max(1, len(vol) - 1)
        for i in range(len(vol)):
            if sm[i] == sm[max(0, i - 2):min(len(vol), i + 3)].max() and pct[i] >= 0.75 and sm[i] > 0:
                cands.append({"px": float(mids[i]), "kind": f"HVN{W}"})
        cands += [{"px": poc, "kind": f"POC{W}"}, {"px": hi_, "kind": f"VAH{W}"}, {"px": lo_, "kind": f"VAL{W}"}]
    rmax = pd.Series(h).rolling(11, center=True).max().values
    rmin = pd.Series(l).rolling(11, center=True).min().values
    for s in range(max(0, t - 120), t - 5 + 1):
        if h[s] == rmax[s]:
            cands.append({"px": float(h[s]), "kind": f"前波高{d['date'].iat[s][5:]}"})
        if l[s] == rmin[s]:
            cands.append({"px": float(l[s]), "kind": f"前波低{d['date'].iat[s][5:]}"})
    hi60x, lo60x = float(h[t - 60:t - 4].max()), float(l[t - 60:t - 4].min())      # 與檢定一致 (b01_panel)：t−60..t−5，排除最近 5 日
    cands += [{"px": hi60x, "kind": "60日高(除近5日)"}, {"px": lo60x, "kind": "60日低(除近5日)"}, {"px": ma20, "kind": "MA20"}, {"px": ma60, "kind": "MA60"}]
    minor = 1000.0 if is_index else 10.0 ** (math.floor(math.log10(close)) - 1)
    major = 5000.0 if is_index else 5 * minor
    for stp, lab in ((minor, "整數"), (major, "大整數")):
        if lab == "整數" and stp < 1.5 * atr:
            continue                                    # 步長太小 (≈格點) 只看大整數
        base = math.floor(close / stp) * stp
        for k in range(-6, 8):
            p = base + k * stp
            if abs(p - close) <= 8 * atr:
                cands.append({"px": p, "kind": f"{lab}{p:g}"})
    CL = 0.5 * atr

    def label(kind: str) -> str:
        return ("HVN" if kind.startswith(("HVN", "POC", "VAH", "VAL")) else "前波" if kind.startswith("前波") else "60日極值" if kind.startswith("60日")
                else "整數" if "整數" in kind else "MA")

    def side(above: bool):
        cs = sorted([x for x in cands if (x["px"] > close + w / 2 if above else x["px"] < close - w / 2)], key=lambda x: x["px"] if above else -x["px"])
        bands = []
        for x in cs:
            if bands and abs(x["px"] - bands[-1]["anchor"]) <= CL:
                bands[-1]["pts"].append(x)
            else:
                bands.append({"anchor": x["px"], "pts": [x]})
        out = []
        for bd in bands[:3]:
            pxs = [p["px"] for p in bd["pts"]]
            lo_, hi_ = min(pxs) - w / 2, max(pxs) + w / 2
            kinds = list(dict.fromkeys(p["kind"] for p in bd["pts"]))
            out.append({"lo": _r(lo_, 2), "hi": _r(hi_, 2), "dist": _r(((lo_ if above else hi_) / close - 1) * 100, 2),
                        "share60": _r(_share(*prof[60], lo_, hi_) * 100, 1), "share120": _r(_share(*prof[120], lo_, hi_) * 100, 1),
                        "kinds": kinds, "labels": list(dict.fromkeys(label(k) for k in kinds))})
        return out
    return {"close": _r(close, 2), "atr20": _r(atr, 3), "value_area": va, "in_va60": bool(va["60"]["VAL"] <= close <= va["60"]["VAH"]),
            "above": side(True), "below": side(False)}


# ------------------------------------------------------------------ 7. 八大行庫 (歸檔 + 部位)
def gov8_archive(sid: str, prev_rows: list[dict] | None) -> list[dict]:
    """HiStock 最新 (優先) ∪ 已發布歸檔。每列：date, lots (張), amt (萬元), banks {行庫: 張}, bank_amt {行庫: 萬元}。"""
    rows: dict[str, dict] = {}
    for r in prev_rows or []:
        if r.get("date"):
            rows[str(r["date"])[:10]] = r
    try:
        g = histock.government_bank_stock(sid)
        for _, x in (g if g is not None else pd.DataFrame()).iterrows():
            d = str(x["date"])[:10]
            banks, bamt = {}, {}
            for b in histock.BANKS:
                v, m = x.get(f"lots_{b}"), x.get(f"money_{b}")
                if v is not None and pd.notna(v):
                    banks[b] = int(round(float(v)))
                if m is not None and pd.notna(m):
                    bamt[b] = round(float(m), 1)
            rows[d] = {"date": d, "lots": int(round(float(x.get("gov8_lots") or 0))), "amt": _r(x.get("gov8_net"), 1), "banks": banks, "bank_amt": bamt}
    except Exception as e:  # noqa: BLE001
        log.warning("HiStock gov8 %s: %s", sid, e)
    return [rows[d] for d in sorted(rows)][-ARCHIVE_MAX:]


def gov8_block(sid: str, arch: list[dict], px: pd.DataFrame, ev: dict) -> dict:
    if not arch:
        return {}
    a = pd.DataFrame(arch)
    # 分割換算：歸檔存原始張數；分割日之前的張數乘倍數，才能與分割還原後的價格同一基準 (金額不需換算)
    split_f = pd.Series(1.0, index=a.index)
    for sd, f in chips.KNOWN_SPLITS.get(sid, {}).items():
        split_f[a["date"] < sd] *= float(f)
    a["lots"] = a["lots"].astype(float) * split_f
    a = a.merge(px[["date", "high", "low", "close", "volume"]], on="date", how="left")
    lots, amt = a["lots"].astype(float), pd.to_numeric(a["amt"], errors="coerce")
    # 隱含成交價 (萬元×10/張 = 元/股)；不在當日高低 ±2% 內改用收盤
    ipx = pd.Series(np.where(lots != 0, amt * 10 / lots.replace(0, np.nan), np.nan))
    okp = (ipx >= a["low"] * 0.98) & (ipx <= a["high"] * 1.02)
    ipx = ipx.where(okp, a["close"])

    def vwap(sub_mask, sign):
        m = sub_mask & (np.sign(lots) == sign) & ipx.notna()
        L = lots[m].abs().sum()
        return _r((ipx[m] * lots[m].abs()).sum() / L, 2) if L > 0 else None
    idx = pd.Series(True, index=a.index)
    tail = lambda n: a.index >= len(a) - n     # noqa: E731
    # FIFO：賣出先沖銷最早買進
    inv, oversold, skipped = [], 0.0, 0
    for q, p in zip(lots, ipx.fillna(a["close"])):
        if pd.isna(p):
            skipped += 1
            continue
        if q > 0:
            inv.append([q, p])
        elif q < 0:
            s = -q
            while s > 0 and inv:
                take = min(s, inv[0][0]); inv[0][0] -= take; s -= take
                if inv[0][0] <= 1e-9:
                    inv.pop(0)
            oversold += s
    rem = sum(x[0] for x in inv)
    fifo = _r(sum(x[0] * x[1] for x in inv) / rem, 2) if rem > 0 else None
    buys = lots[lots > 0].sum()
    close = float(px["close"].iloc[-1]) if len(px) else None
    st = 0
    for v in lots[::-1]:
        if v == 0 or (st and np.sign(v) != np.sign(st)):
            break
        st += int(np.sign(v))
    ret = a["close"].astype(float).pct_change()
    down, up = ret < 0, ret > 0
    banks = {}
    for r, f in zip(arch, split_f):
        for b, v in (r.get("banks") or {}).items():
            banks.setdefault(b, []).append((r["date"], v * f))
    bank_rows = []
    for b, seq in banks.items():
        s = pd.Series([v for _, v in seq], index=[d for d, _ in seq], dtype=float)
        tot_abs = s.abs().sum()
        bank_rows.append({"bank": b, "cum": int(s.sum()), "net5": int(s.tail(5).sum()), "net20": int(s.tail(20).sum()),
                          "one_sided": _r(abs(s.sum()) / tot_abs, 2) if tot_abs else None})
    bank_rows.sort(key=lambda x: -x["cum"])
    vol_lots = a["volume"].astype(float) / 1000
    part = (lots.abs() / vol_lots.replace(0, np.nan)) * 100
    out = {"days": int(len(a)), "from": str(a["date"].iloc[0]), "to": str(a["date"].iloc[-1]), "sufficient": len(a) >= 120,
           "today": int(lots.iloc[-1]), "cum5": int(lots.tail(5).sum()), "cum20": int(lots.tail(20).sum()), "cum60": int(lots.tail(60).sum()),
           "cum_all": int(lots.sum()), "amt_all_yi": _r(amt.sum() / 1e4, 2), "amt20_yi": _r(amt.tail(20).sum() / 1e4, 2), "streak": st,
           "one_sided": _r(abs(lots.sum()) / lots.abs().sum(), 2) if lots.abs().sum() else None,
           "buy_vwap_all": vwap(idx, 1), "buy_vwap_60": vwap(pd.Series(tail(60)), 1), "buy_vwap_20": vwap(pd.Series(tail(20)), 1),
           "sell_vwap_20": vwap(pd.Series(tail(20)), -1),
           "net_cost": _r(amt.sum() * 10 / lots.sum(), 2) if lots.sum() > 0 and amt.sum() > 0 else None,
           "fifo_cost": fifo, "fifo_remaining": int(rem), "fifo_oversold": int(oversold), "fifo_skipped": skipped,
           # 成本可信：要有淨買累積、FIFO 有剩、超賣不多 (研究：ETF 常來回/超賣，成本不能解讀為持倉成本；目前只有 0050、2330 可信)
           "cost_unreliable": bool((not buys) or rem <= 0 or oversold >= rem or oversold > 0.3 * buys or lots.sum() <= 0),
           "dip_buy_rate": _r(float((lots[down] > 0).mean()) * 100, 0) if down.any() else None,
           "rally_sell_rate": _r(float((lots[up] < 0).mean()) * 100, 0) if up.any() else None,
           "corr_same_day": _r(float(pd.concat([lots, ret], axis=1).dropna().corr(method="spearman").iloc[0, 1]), 2) if ret.notna().sum() > 10 else None,
           "participation20": _r(part.tail(20).mean(), 2), "participation_today": _r(part.iloc[-1], 2) if len(part) else None,
           "participation_z60": (_r(((part - part.rolling(60, min_periods=20).mean()) / part.rolling(60, min_periods=20).std()).iloc[-1], 2)
                                 if part.notna().sum() >= 20 else None),
           "banks": bank_rows,
           "bank_daily": [{"date": r["date"], "banks": {b: int(round(v * f)) for b, v in (r.get("banks") or {}).items()}} for r, f in zip(arch[-20:], split_f.tail(20))],
           "daily": [{"date": r["date"], "lots": int(r["lots"]), "close": _r(c, 2)} for r, c in zip(arch[-60:], a["close"].tail(60))]}
    for k in ("buy_vwap_all", "buy_vwap_60", "fifo_cost", "net_cost"):
        if out.get(k) and close:
            out[f"vs_{k}"] = _r((close / out[k] - 1) * 100, 2)
    try:
        d = gov8_dist.stock_dist(a.rename(columns={"lots": "gov8_lots"})[["date", "gov8_lots", "close"]].dropna())
        if d:
            out["dist"] = {k: d.get(k) for k in ("bucket", "costs", "zones", "current", "fifo", "history_days") if k in d}
    except Exception as e:  # noqa: BLE001
        log.debug("gov8 dist %s: %s", sid, e)
    return out


# ------------------------------------------------------------------ 8. 權證 (065423)
def warrant_terms(code: str) -> dict:
    def load():
        r = session().get("https://openapi.twse.com.tw/v1/opendata/t187ap37_L", timeout=90, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        for x in r.json():
            if str(x.get("權證代號")).strip() == code:
                return x
        return {}
    x = cached(f"twse:warrant_terms:{code}", config.TTL_DAILY, load) or {}
    if not x:
        return {}

    def roc(s):
        s = str(s or "").strip()
        return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:7]}" if len(s) == 7 else None
    return {"code": code, "name": x.get("權證簡稱"), "type": x.get("權證類型"), "style": x.get("類別"), "underlying_name": x.get("標的證券/指數"),
            "strike": _r(x.get("最新履約價格(元)/履約指數"), 4), "ratio": _r(float(x.get("最新標的履約配發數量(每仟單位權證)") or 0) / 1000, 5),
            "last_trade": roc(x.get("最後交易日")), "expiry": roc(x.get("履約截止日")), "exercise_from": roc(x.get("履約開始日")),
            "units_k": _r(x.get("發行單位數量(仟單位)"), 0), "note": (x.get("備註") or "").strip(),       # 不截斷 (除息調整紀錄在備註裡)
            "report_date": roc(x.get("出表日期")), "strike_orig": _r(x.get("原始履約價格(元)/履約指數"), 4)}


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(S, K, T, sig, r=R_F):
    """歐式買權價、delta、gamma、vega(每 1 波動點)。"""
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0, 0.0
    sT = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / sT
    d2 = d1 - sT
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2), _N(d1), pdf / (S * sT), S * pdf * math.sqrt(T) / 100


def _iv(c, S, K, T, r=R_F):
    if T <= 0 or c <= max(S - K * math.exp(-r * T), 0) + 1e-9:
        return None
    lo, hi = 0.01, 5.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if _bs(S, K, T, m, r)[0] > c:
            hi = m
        else:
            lo = m
    v = 0.5 * (lo + hi)
    return v if 0.011 < v < 4.99 else None


def warrant_block(wpx: pd.DataFrame, upx: pd.DataFrame, uadj: pd.Series, ev: dict, exd981: dict | None = None) -> dict:
    """認購權證：T 以交易日/252 為主 (研究口徑)、另列日曆日；IV 以 LP 最佳買賣中價反推 (沒有報價才用收盤)。"""
    terms = warrant_terms(WARRANT["id"])
    if not terms or wpx.empty or upx.empty or not terms.get("strike") or not terms.get("ratio"):
        return {"terms": terms} if terms else {}
    W, S = float(wpx["close"].iloc[-1]), float(upx["close"].iloc[-1])
    K, ratio = float(terms["strike"]), float(terms["ratio"])
    d0 = str(wpx["date"].iloc[-1])[:10]
    ev981 = exdiv.rebase(exd981 or {}, d0)
    K_twse, r_twse = K, ratio
    guard = exdiv.terms_guard({**terms, "strike": K, "ratio": ratio}, [], ev981.get("realized") or [], d0, first_trade=terms.get("exercise_from"))
    K, ratio = float(guard["strike"]), float(guard["ratio"])            # 條款檔 (每日 ~05:30 批次) 落後除息日時用公式推算
    try:
        cal = twse.next_trading_days(d0, 45)
    except Exception:  # noqa: BLE001
        cal = []
    n_td = len([d for d in cal if terms.get("expiry") and d <= terms["expiry"]])
    n_lt = len([d for d in cal if terms.get("last_trade") and d <= terms["last_trade"]])
    cal_days = (dt.date.fromisoformat(terms["expiry"]) - dt.date.fromisoformat(d0)).days if terms.get("expiry") else 0
    T, Tc = n_td / 252, max(cal_days, 0) / 365
    bid = ask = q_date = None
    u_date = str(upx["date"].iloc[-1])[:10]
    try:
        q5 = twse.index_realtime((f"tse_{WARRANT['id']}.tw",))
        if q5:
            qd = str(q5[0].get("date") or "")
            q_date = f"{qd[:4]}-{qd[4:6]}-{qd[6:8]}" if len(qd) == 8 else qd[:10]
            if q_date == d0 == u_date:                 # 報價日、權證收盤日、標的收盤日一致才用中價
                bid = float(q5[0].get("bid") or 0) or None
                ask = float(q5[0].get("ask") or 0) or None
    except Exception as e:  # noqa: BLE001
        log.debug("warrant quote: %s", e)
    mid = (bid + ask) / 2 if (bid and ask) else None
    Wp = mid or W
    date_ok = u_date == d0
    iv = _iv(Wp / ratio, S, K, T)
    lr = _lr(uadj)
    rv20, rv60 = float(lr.tail(20).std() * math.sqrt(252)), float(lr.tail(60).std() * math.sqrt(252))
    iv = iv if date_ok else None                        # 權證與標的收盤日不同 → 不算 IV/希臘值
    out = {"terms": terms, "date": d0, "price": _r(W, 3), "bid": bid, "ask": ask, "mid": _r(mid, 3), "quote_date": q_date,
           "iv_src": "LP 中價" if mid else (f"收盤價 (報價日 {q_date})" if q_date and q_date != d0 else "收盤價"),
           "spread_pct": _r((ask - bid) / mid * 100, 2) if mid else None, "under": _r(S, 2), "under_date": str(upx["date"].iloc[-1])[:10],
           "trade_days_to_expiry": n_td, "trade_days_to_last": n_lt, "cal_days": cal_days,
           "moneyness": _r((S / K - 1) * 100, 2), "intrinsic": _r(max(S - K, 0) * ratio, 3), "time_value": _r(Wp - max(S - K, 0) * ratio, 3),
           "time_value_pct": _r((Wp - max(S - K, 0) * ratio) / Wp * 100, 1), "breakeven": _r(K + Wp / ratio, 2), "breakeven_pct": _r((K + Wp / ratio) / S * 100 - 100, 2),
           "breakeven_ask": _r(K + ask / ratio, 2) if ask else None, "iv": _r(iv, 4), "iv_cal": _r(_iv(Wp / ratio, S, K, Tc), 4),
           "rv20": _r(rv20, 4), "rv60": _r(rv60, 4), "vol_premium": _r((iv - rv20) * 100, 1) if iv else None}
    if iv:
        p0, dlt, gam, vega = _bs(S, K, T, iv)
        p1 = _bs(S, K, max(T - 1 / 252, 1e-9), iv)[0]
        out.update({"delta_unit": _r(dlt * ratio, 4), "delta_n": _r(dlt, 3), "gamma_unit": _r(gam * ratio, 5), "vega_unit": _r(vega * ratio, 4),
                    "theta_td": _r((p1 - p0) * ratio, 4), "theta_td_pct": _r((p1 - p0) * ratio / Wp * 100, 2),
                    "leverage": _r(dlt * S * ratio / Wp, 2), "nominal_lev": _r(ratio * S / Wp, 1)})
        out["decay"] = [{"days": k, "value": _r(_bs(S, K, max(T - k / 252, 0), iv)[0] * ratio, 3),
                         "chg": _r((_bs(S, K, max(T - k / 252, 0), iv)[0] * ratio / Wp - 1) * 100, 1)} for k in (1, 5, 10, 15) if k < n_td]
    sell_cost = 0.000855 + 0.001
    buy_px = (ask or Wp) * (1 + 0.000855)
    out["scenario"] = [{"chg": x, "under": _r(S * (1 + x / 100), 2), "payoff": _r(max(S * (1 + x / 100) - K, 0) * ratio, 3),
                        "pnl": _r((max(S * (1 + x / 100) - K, 0) * ratio * (1 - sell_cost) / buy_px - 1) * 100, 1),
                        "mid10": _r(_bs(S * (1 + x / 100), K, max(T - 10 / 252, 1e-9), iv)[0] * ratio, 3) if (iv and n_td > 10) else None}
                       for x in (-10, -6, -4, -2, 0, 2, 4, 6, 8, 10)]
    probs = []
    for nm, sig in (("RV20", rv20), ("RV60", rv60), ("IV", iv)):
        if sig and T > 0:
            sT = sig * math.sqrt(T)
            p_itm = _N((math.log(S / K) - 0.5 * sig * sig * T) / sT)
            be = K + Wp / ratio
            p_be = _N((math.log(S / be) - 0.5 * sig * sig * T) / sT)
            ev_pay = _bs(S, K, T, sig, r=0.0)[0] * ratio
            probs.append({"sigma": nm, "value": _r(sig, 4), "p_itm": _r(p_itm * 100, 1), "p_be": _r(p_be * 100, 1), "ev": _r((ev_pay / Wp - 1) * 100, 1)})
    out["probs"] = probs
    try:   # 隱含波動歷史：逐日套用當時有效的條款 (除息調整紀錄解析自 TWSE 備註)，剩餘交易日用完整日曆計算
        exp = terms.get("expiry")
        adjs = sorted((f"{a}-{b}-{c}", float(k), float(q)) for a, b, c, k, q in re.findall(
            r"(\d{4})/(\d{2})/(\d{2})\s*標的證券除息，調整後履約價格([\d.]+)元，調整後行使比例([\d.]+)", terms.get("note") or ""))
        sched = [("0000-00-00", K_twse, r_twse)]
        if adjs:
            fac = (uadj.reset_index(drop=True) / upx["close"].astype(float).reset_index(drop=True))
            fs = pd.Series(fac.values, index=upx["date"].astype(str).str[:10].values)
            e0 = adjs[0][0]
            f0 = float(fs[fs.index < e0].iloc[-1] / fs[fs.index >= e0].iloc[0]) if (fs.index < e0).any() and (fs.index >= e0).any() else 1.0
            sched = [("0000-00-00", adjs[0][1] / f0, adjs[0][2] * f0)] + adjs     # 首次調整前的原始條款 (由除息因子反推)

        est_from = set()
        for e_, k_, r_ in guard.get("steps") or []:        # 條款檔落後：推算的調整也要進逐日條款 (否則除息當天 IV 會假跳)
            if all(e_ != s_[0] for s_ in sched):
                sched.append((e_, k_, r_))
                est_from.add(e_)
        sched.sort(key=lambda s_: s_[0])

        def terms_at(dd):
            k_, r_ = sched[0][1], sched[0][2]
            for eff, kk, rr in sched:
                if dd >= eff:
                    k_, r_ = kk, rr
            return k_, r_
        j = wpx[["date", "close"]].merge(upx[["date", "close"]].rename(columns={"close": "u"}), on="date").tail(80)
        cal_all = twse.next_trading_days(str(j["date"].iloc[0])[:10], 400) if exp and len(j) else []
        import bisect
        hist = []
        for _, r in j.iterrows():
            dd = str(r["date"])[:10]
            n_ = (bisect.bisect_right(cal_all, exp) - bisect.bisect_right(cal_all, dd)) if exp else 0
            k_, r_ = terms_at(dd)
            hist.append({"date": dd, "w": _r(r["close"], 3), "u": _r(r["u"], 2), "K": _r(k_, 3), "ratio": _r(r_, 4),
                         "iv": _r(_iv(float(r["close"]) / r_, float(r["u"]), k_, n_ / 252), 4)})
        out["iv_hist"] = hist
        out["terms_hist"] = [{"from": e, "strike": _r(k, 3), "ratio": _r(q, 4), **({"estimated": True} if e in est_from else {})} for e, k, q in sched]
    except Exception as e:  # noqa: BLE001
        log.debug("warrant iv hist: %s", e)
    flags = list(guard.get("flags") or [])
    try:
        proj = exdiv.project_terms(K, ratio, ev981.get("upcoming") or [], S, terms.get("last_trade") or "9999-12-31")
        if proj:
            out["terms_projected"] = proj
            flags.append("存續期內標的除息 " + "、".join(f"{x['ex_date']} {x['cash']}" for x in proj) + "：履約價/比例將調整 (除息保護)")
    except Exception as e:  # noqa: BLE001
        log.debug("warrant project_terms: %s", e)
    if guard.get("estimated"):
        out["terms_estimated"] = {"strike": K, "ratio": ratio}
    if not date_ok:
        flags.append(f"權證收盤日 {d0} 與 00981A 收盤日 {u_date} 不同：IV/希臘值暫不計算")
    if n_lt <= 20:
        flags.append(f"距最後交易日 {n_lt} 個交易日：時間價值加速衰減")
    if out.get("vol_premium") is not None and out["vol_premium"] >= 15:
        flags.append(f"隱含波動比 20 日實現波動高 {out['vol_premium']:.0f} 點 (偏貴)")
    if (out.get("time_value_pct") or 0) > 50 and n_td < 25:
        flags.append("時間價值佔一半以上且剩不到 25 個交易日：零漂移下持有到期期望值為負")
    if out.get("spread_pct") and out["spread_pct"] >= 2.3:
        flags.append("買賣價差 ≥2 檔：流動性差")
    out["flags"] = flags
    out["notes"] = (ev.get("special") or {}).get("warrant") or []
    out["settle_note"] = "到期價內自動現金結算；結算價計算方式以發行人公開說明書為準。"
    return out


# ------------------------------------------------------------------ 9. 00981A 主動 ETF
def holdings_00981A() -> dict:
    def load():
        r = session().get("https://www.ezmoney.com.tw/ETF/Fund/Info?fundCode=49YTW", timeout=40, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        m = re.search(r'<div id="DataAsset" data-content="(.*?)" style', r.text, re.S)
        return json.loads(html.unescape(m.group(1))) if m else []
    try:
        j = cached("ezmoney:49YTW", config.TTL_DAILY, load) or []
    except Exception as e:  # noqa: BLE001
        log.warning("00981A holdings: %s", e)
        return {}
    by = {a.get("AssetCode"): a for a in j}
    st = (by.get("ST") or {}).get("Details") or []
    if not st:
        return {}
    date = str(st[0].get("TranDate") or "")[:10]
    hold = sorted([{"code": str(x.get("DetailCode")), "name": str(x.get("DetailName") or "")[:20], "shares": float(x.get("Share") or 0), "w": _r(x.get("NavRate"), 2)}
                   for x in st if re.fullmatch(r"[0-9A-Z]{4,6}", str(x.get("DetailCode") or ""))],        # 外部網站資料：代號格式不符的列丟掉
                  key=lambda x: -(x["w"] or 0))
    nav = (by.get("NAV") or {}).get("Value")
    return {"date": date, "nav_unit": _r((by.get("P_UNIT") or {}).get("Value"), 2), "aum_yi": _r(float(nav) / 1e8, 0) if nav else None,
            "stock_pct": _r(float((by.get("ST") or {}).get("Value") or 0) / float(nav) * 100, 2) if nav else None,
            "fut_pct": _r(float((by.get("GD") or {}).get("Value") or 0) / float(nav) * 100, 2) if nav else None,
            "cash_pct": _r(float((by.get("CASH") or {}).get("Value") or 0) / float(nav) * 100, 2) if nav else None,
            "n": len(hold), "top10_w": _r(sum((x["w"] or 0) for x in hold[:10]), 2), "holdings": hold}


def active_block(adj981: pd.DataFrame, adj50: pd.DataFrame, tw: pd.DataFrame, close: float, divs: list[dict], hold: dict, prev_hold: dict | None, ev: dict) -> dict:
    j = adj981.merge(adj50, on="date", suffixes=("", "_50")).merge(tw[["date", "close"]].rename(columns={"close": "tw"}), on="date")
    r, r50, rtw = j["adj"].pct_change(), j["adj_50"].pct_change(), j["tw"].pct_change()
    k = pd.concat([r, r50, rtw], axis=1).dropna()
    k.columns = ["a", "b", "t"]

    def beta(x, y):
        return float(np.cov(x, y)[0, 1] / np.var(y, ddof=1)) if len(x) > 20 else None
    out = {"n_days": int(len(k)), "beta_0050": _r(beta(k.a, k.b), 3), "beta_twii": _r(beta(k.a, k.t), 3),
           "beta60_0050": _r(beta(k.a.tail(60), k.b.tail(60)), 3), "beta60_twii": _r(beta(k.a.tail(60), k.t.tail(60)), 3),
           "corr60": _r(k.a.tail(60).corr(k.b.tail(60)), 3), "te60": _r((k.a - k.b).tail(60).std() * math.sqrt(252) * 100, 1)}
    dn, up = k[k.b < -0.02], k[k.b > 0.02]
    out["big_down"] = {"n": int(len(dn)), "ratio": _r(dn.a.mean() / dn.b.mean(), 2) if len(dn) else None}
    out["big_up"] = {"n": int(len(up)), "ratio": _r(up.a.mean() / up.b.mean(), 2) if len(up) else None}
    out["dd_from_peak"] = _r((j["adj"].iloc[-1] / j["adj"].max() - 1) * 100, 2)
    out["ret_since_list"] = _r((j["adj"].iloc[-1] / j["adj"].iloc[0] - 1) * 100, 1)
    out["ret_0050_same"] = _r((j["adj_50"].iloc[-1] / j["adj_50"].iloc[0] - 1) * 100, 1)
    out["dividends"] = divs
    if divs:
        out["div_runrate"] = _r(divs[-1]["div"] * 4 / close * 100, 1)
    if hold:
        out["holdings"] = {k: hold.get(k) for k in ("date", "nav_unit", "aum_yi", "stock_pct", "fut_pct", "cash_pct", "n", "top10_w")}
        out["holdings"]["top"] = hold["holdings"][:15]
        cd = str(adj981["date"].iloc[-1])[:10]
        ok = bool(hold.get("nav_unit")) and str(hold.get("date") or "")[:10] == cd
        out["premium"] = _r((close / hold["nav_unit"] - 1) * 100, 2) if ok else None
        if not ok and hold.get("nav_unit"):
            out["premium_note"] = f"淨值日 {hold.get('date')} ≠ 收盤日 {cd}，不計折溢價"
        w2330 = next((x["w"] for x in hold["holdings"] if x["code"] == "2330"), None)
        out["holdings"]["w2330"] = w2330
        if prev_hold and prev_hold.get("holdings") and prev_hold.get("date") != hold.get("date"):
            ps = {x["code"]: x for x in prev_hold["holdings"]}
            ch = []
            for x in hold["holdings"]:
                p = ps.get(x["code"])
                d = x["shares"] - (p["shares"] if p else 0)
                if abs(d) > 0:
                    ch.append({"code": x["code"], "name": x["name"], "chg_shares": int(d), "new": p is None, "w": x["w"]})
            for code, p in ps.items():
                if code not in {x["code"] for x in hold["holdings"]}:
                    ch.append({"code": code, "name": p["name"], "chg_shares": int(-p["shares"]), "out": True, "w": 0})
            ch.sort(key=lambda x: -abs(x["chg_shares"]))
            out["holdings"]["changes"] = {"vs": prev_hold.get("date"), "rows": ch[:12]}
    fl = []
    if (out.get("te60") or 0) > 25:
        fl.append(f"追蹤差 {out['te60']}%/年 > 25%")
    if (out.get("beta60_0050") or 0) > 1.3:
        fl.append(f"60 日 β {out['beta60_0050']} > 1.3")
    if out.get("premium") is not None and abs(out["premium"]) > 1:
        fl.append(f"折溢價 {out['premium']}%")
    out["flags"] = {"items": fl, "basis": "經驗法則 (非研究推導)"}
    out["notes"] = (ev.get("special") or {}).get("00981A") or []
    return out


# ------------------------------------------------------------------ 10. 2330：ADR 溢價、相對強弱
def tsmc_block(px2330: pd.DataFrame, adj2330: pd.Series, adj50: pd.DataFrame, ev: dict, px_long: pd.DataFrame | None = None, w981: tuple | None = None) -> dict:
    w = dict(WEIGHTS_2330)
    if w981 and w981[0] is not None:
        w["etf00981A"], w["asof"] = w981[0], f"加權 2026-08-31 (推估)、0050 2026-09-24、00981A {w981[1]}"
    out: dict = {"weights": w}
    try:
        tsm = global_markets.history_closed("TSM", "20y")
        fx = global_markets.history("TWD=X", "20y")
        now = dt.datetime.now(config.TZ)
        # 美股 d 日收盤在台北 d+1 05:00 前後；未收盤的 bar 不用
        tsm = tsm[pd.to_datetime(tsm["date"]) + pd.Timedelta(hours=29, minutes=30) <= pd.Timestamp(now.replace(tzinfo=None))]
        fx = fx[(fx["close"] > 25) & (fx["close"] < 40)]
        t = tsm[["date", "close"]].rename(columns={"close": "tsm"}).merge(fx[["date", "close"]].rename(columns={"close": "fx"}), on="date", how="left")
        t["fx"] = t["fx"].ffill()
        src = px_long if px_long is not None and len(px_long) > len(px2330) else px2330
        j = pd.DataFrame({"date": src["date"].astype(str).values, "c": src["close"].astype(float).values}).merge(t, on="date", how="inner").dropna()
        j["prem"] = j["tsm"] * j["fx"] / 5 / j["c"] - 1
        p = j["prem"]
        last = j.iloc[-1]
        out["adr"] = {"date": last["date"], "tsm": _r(last["tsm"], 2), "fx": _r(last["fx"], 3), "prem": _r(last["prem"] * 100, 2),
                      "pct_all": _r((p < last["prem"]).mean() * 100, 0), "pct_from": str(j["date"].iloc[0]), "pct_750": _r((p.tail(750) < last["prem"]).mean() * 100, 0),
                      "pct_250": _r((p.tail(250) < last["prem"]).mean() * 100, 0),
                      "z60": _r((last["prem"] - p.tail(60).mean()) / p.tail(60).std(), 2), "z250": _r((last["prem"] - p.tail(250).mean()) / p.tail(250).std(), 2),
                      "mean60": _r(p.tail(60).mean() * 100, 2), "mean250": _r(p.tail(250).mean() * 100, 2),
                      "implied60": _r(last["tsm"] * last["fx"] / 5 / (1 + p.tail(60).mean()), 1), "implied250": _r(last["tsm"] * last["fx"] / 5 / (1 + p.tail(250).mean()), 1)}
        # 開盤跳空估計：只有「已收盤的 TSM 日期 ≥ 2330 最後收盤日」時才是對下一個台股開盤的估計 (連假自動納入休市期間的 TSM 變動)
        d_px, c_last = str(px2330["date"].iloc[-1])[:10], float(px2330["close"].iloc[-1])
        tl = t.dropna().iloc[-1]
        prev = j[j["date"] < d_px]
        out["adr"]["gap_base"] = d_px
        if str(tl["date"]) >= d_px and len(prev):
            prem_now = tl["tsm"] * tl["fx"] / 5 / c_last - 1
            out["adr"]["gap_hat"] = _r(0.306 * (prem_now - prev["prem"].iloc[-1]) * 100, 2)   # 斜率 0.306 (研究)，樣本外 R² 約 0.30；不可交易
            out["adr"]["gap_tsm_date"] = str(tl["date"])
            try:
                out["adr"]["gap_target"] = twse.next_trading_days(d_px, 1)[0]
            except Exception:  # noqa: BLE001
                pass
        else:
            out["adr"]["gap_hat"] = None
            out["adr"]["gap_note"] = f"待 {d_px} 美股收盤 (台北次日約 05:30) 後更新"
    except Exception as e:  # noqa: BLE001
        log.warning("2330 ADR: %s", e)
    try:
        j = pd.DataFrame({"date": px2330["date"].astype(str).values, "a": adj2330.values}).merge(adj50, on="date")
        rs = np.log(j["a"] / j["adj"])
        out["rs_0050"] = {str(n): _r((rs.iloc[-1] - rs.iloc[-1 - n]) * 100, 2) for n in (20, 60, 120, 250) if len(rs) > n}
    except Exception as e:  # noqa: BLE001
        log.debug("2330 RS: %s", e)
    today = dt.datetime.now(config.TZ).date()

    def _rev_day(y: int, m: int) -> dt.date:        # 每月 10 日 (含) 前最後一個交易日 (研究定義)
        d = dt.date(y, m, 10)
        try:
            hol = twse.holidays(y)
        except Exception:  # noqa: BLE001
            hol = set()
        while d.weekday() >= 5 or d.isoformat() in hol:
            d -= dt.timedelta(days=1)
        return d
    nxt_rev = _rev_day(today.year, today.month)
    if nxt_rev < today:
        nxt_rev = _rev_day(today.year + (today.month == 12), today.month % 12 + 1)
    try:   # 月營收 (描述；研究：營收加速度與公布後 5/20 日報酬負相關 −0.17~−0.18、不可交易)
        rv = finmind.fetch("TaiwanStockMonthRevenue", "2330", (today - dt.timedelta(days=900)).isoformat())
        if rv is not None and not rv.empty:
            rv = rv.sort_values(["revenue_year", "revenue_month"]).reset_index(drop=True)
            r_ = rv["revenue"].astype(float)
            yoy = r_ / r_.shift(12) - 1
            mom = r_ / r_.shift(1) - 1
            acc = yoy - yoy.shift(1).rolling(3).mean()
            out["revenue"] = [{"ym": f"{int(x.revenue_year)}-{int(x.revenue_month):02d}", "rev_yi": _r(float(x.revenue) / 1e8, 0),
                               "yoy": _r(yoy.iloc[i] * 100, 1), "mom": _r(mom.iloc[i] * 100, 1), "accel": _r(acc.iloc[i] * 100, 1)}
                              for i, x in enumerate(rv.itertuples()) if i >= len(rv) - 6]
    except Exception as e:  # noqa: BLE001
        log.debug("2330 revenue: %s", e)
    out["events"] = [{"what": "月營收公布 (約)", "date": nxt_rev.isoformat(), "note": "10 日前最後一個交易日盤後；只有隔夜跳空有超額 (2017 起不顯著)"},
                     {"what": "季配除息 (約)", "date": "3/6/9/12 月中旬", "note": "除息日開盤溢價扣成本與股利稅後約 0；填息中位數 2 天"}]
    out["notes"] = (ev.get("special") or {}).get("2330") or []
    return out


# ------------------------------------------------------------------ 11. 選擇權區塊 + 歷史
def opt_history(prev: dict, feat: dict) -> list[dict]:
    rows: dict[str, dict] = {}
    try:
        from ..predict import model as _M
        seed = (_M.load_json("txo_iv_seed") or {}).get("rows") or []
    except Exception:  # noqa: BLE001
        seed = []
    for r in list(prev.get("opt_hist") or []) + seed:       # seed (研究同演算法、已休市修正) 在自己涵蓋的日期優先
        if r.get("date"):
            rows[r["date"]] = r
    if feat.get("date") and not feat.get("calendar_fallback") and not feat.get("_carried"):
        row = {"date": feat["date"], **{k: feat.get(k) for k in ("iv5", "iv21", "iv42", "rr25", "ts_5_21", "F_near", "pwall_dist", "cwall_dist", "mp_dist", "pcr_oi")}}
        row["ivk"] = feat.get("ivk")                          # 日後依規格每日滾動重算乘數用
        row["ss"] = [{"n": x.get("n"), "atm": x.get("atm"), "kind": x.get("kind")} for x in (feat.get("series") or []) if x.get("atm") is not None][:8]
        rows[feat["date"]] = row
    return [rows[d] for d in sorted(rows)][-OPT_HIST_MAX:]


def opt_block(feat: dict, hist: list[dict], last_trading: str | None, ev: dict) -> dict:
    if not feat:
        return {}
    h = pd.DataFrame(hist)
    out = {k: feat.get(k) for k in ("date", "iv5", "iv21", "iv42", "ivk", "rr25", "ts_5_21", "p25_21", "c25_21", "F_near", "pcr_oi", "near", "month", "series", "calendar_fallback")}
    out["carried"] = bool(feat.get("_carried"))
    out["source"] = feat.get("source")
    out["stale"] = bool(last_trading and feat.get("date") and feat["date"] < last_trading)
    for col in ("iv5", "iv21"):
        s = pd.to_numeric(h.get(col), errors="coerce").dropna() if col in h else pd.Series(dtype=float)
        if len(s) > 100 and feat.get(col):
            out[f"{col}_pct_1y"] = _r((s.tail(250) < feat[col]).mean() * 100, 0)
            out[f"{col}_pct_all"] = _r((s < feat[col]).mean() * 100, 0)
    out["hist"] = [{"date": r["date"], "iv5": r.get("iv5"), "iv21": r.get("iv21")} for r in hist[-120:]]
    rg = ev.get("range") or {}
    out["evidence"] = {k: rg.get(k) for k in ("evidence", "exec_note", "walls", "chips")}
    return out


# ------------------------------------------------------------------ 機率帶上線帳本 (真正的樣本外追蹤)
def ledger_add(ledger: list[dict], sid: str, date: str, close: float, rng: dict) -> None:
    """記下 date 收盤發布的 k 日帶 (k ∈ LEDGER_KS)。同 (date, sid, k) 只留最新一筆。"""
    lv = (rng or {}).get("levels") or {}
    ends = (rng or {}).get("ends") or {}
    for k in LEDGER_KS:
        row = lv.get(str(k)) or {}
        levels = {q: (row.get(q) or {}).get("px") for q in QS if (row.get(q) or {}).get("px") is not None}
        if len(levels) < 4 or not ends.get(str(k)):
            continue
        rec = {"date": date, "sid": sid, "k": k, "close": _r(close, 2), "end": ends[str(k)], "levels": levels, "mver": (rng or {}).get("mver") or "unknown"}
        if (rng or {}).get("exdiv"):
            rec["exdiv"] = True
        for i in range(len(ledger) - 1, -1, -1):
            x = ledger[i]
            if x.get("date") == date and x.get("sid") == sid and x.get("k") == k:
                ledger[i] = rec
                break
        else:
            ledger.append(rec)


def ledger_eval(ledger: list[dict], hl: dict[str, pd.DataFrame]) -> dict:
    """hl[sid] = DataFrame(date, high, low)。窗口 (date, end] 全部有價格才評估 (實際高低價是否觸及)。
    回傳 {sid: {mver: {k: {q: {n, hit, rate, nominal}}}, recent: [...]}}；依乘數版本分組 (凍結值與每日滾動值分開統計)。"""
    out: dict = {}
    for x in ledger:
        df = hl.get(x["sid"])
        if df is None or df.empty or str(df["date"].iloc[-1]) < x["end"]:
            continue
        w = df[(df["date"] > x["date"]) & (df["date"] <= x["end"])]
        if w.empty:
            continue
        s = 1.0                                   # 分割換算：歷史價已依分割調整，發布時的價位要同比例換算
        if "close" in df.columns and x.get("close"):
            c0 = df.loc[df["date"] == x["date"], "close"]
            if len(c0) and float(c0.iloc[0]) > 0:
                s = float(c0.iloc[0]) / float(x["close"])
                if abs(s - 1) < 0.02:              # 非分割的小差異 (四捨五入) 不換算
                    s = 1.0
        lo, hi = float(w["low"].min()) / s, float(w["high"].max()) / s
        st = out.setdefault(x["sid"], {}).setdefault(x.get("mver") or "unknown", {}).setdefault(str(x["k"]), {q: {"n": 0, "hit": 0} for q in QS})
        hits = {}
        for q, v in x["levels"].items():
            if v is None or q not in st:
                continue
            h = (lo <= v) if q.startswith("low") else (hi >= v)
            st[q]["n"] += 1
            st[q]["hit"] += int(h)
            hits[q] = h
        out[x["sid"]].setdefault("recent", []).append({"date": x["date"], "k": x["k"], "mver": x.get("mver"), "lo": _r(lo, 2), "hi": _r(hi, 2), "hits": hits})
    for sid, v in out.items():
        for mv, byk in v.items():
            if mv == "recent":
                continue
            for k, st in byk.items():
                for q, c in st.items():
                    c["rate"] = _r(c["hit"] / c["n"], 3) if c["n"] else None
                    c["nominal"] = Q_NOMINAL[q]
        v["recent"] = sorted(v.get("recent") or [], key=lambda r: (r["date"], r["k"]))[-15:]
    return out


# ------------------------------------------------------------------ 組裝
def _last_trading_day() -> str | None:
    try:
        d = dt.date.today()
        hol = twse.holidays(d.year)
        while d.weekday() >= 5 or d.isoformat() in hol:
            d -= dt.timedelta(days=1)
        now = dt.datetime.now(config.TZ)
        if d == now.date() and now.hour < 14:
            d -= dt.timedelta(days=1)
            while d.weekday() >= 5 or d.isoformat() in hol:
                d -= dt.timedelta(days=1)
        return d.isoformat()
    except Exception:  # noqa: BLE001
        return None


def build(prev_archive: dict | None = None) -> tuple[dict, dict]:
    """回傳 (desk, archive)。prev_archive = 上次發布的 desk_archive.json (跨次累積)。"""
    prev_archive = prev_archive if prev_archive is not None else load_archive()
    ev = _evidence()
    last_td = _last_trading_day()
    try:
        feat = taifex_opt.features(expect=last_td)
    except Exception as e:  # noqa: BLE001
        log.warning("TXO features: %s", e)
        feat = {}
    if not feat and prev_archive.get("opt_last"):
        feat = dict(prev_archive["opt_last"]); feat["_carried"] = True
    ohist = opt_history(prev_archive, feat) if feat else list(prev_archive.get("opt_hist") or [])
    try:
        tw = twii((dt.date.today() - dt.date.fromisoformat(desk_bands.TWII_START)).days)   # 固定起點 (機率帶保守版 = 2017-01-03 起擴張)
    except Exception as e:  # noqa: BLE001   加權抓不到：各區塊降級，不中止整個 build
        log.warning("TWII: %s", e)
        tw = pd.DataFrame(columns=["date", "high", "low", "close", "volume"])
    tw_close = float(tw["close"].iloc[-1]) if len(tw) else None
    cal = []
    try:
        cal = twse.next_trading_days(str(tw["date"].iloc[-1]) if len(tw) else dt.date.today().isoformat(), 25)
    except Exception:  # noqa: BLE001
        pass
    desk: dict = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "disclaimer": DISCLAIMER,
                  "evidence_asof": ev.get("asof"), "evidence_source": ev.get("source"),
                  "opt": opt_block(feat, ohist, last_td, ev), "index": {}, "instruments": {}, "framework": {k: ev.get(k) for k in ("risk", "zones_evidence", "gov8_evidence", "overlay")}}
    # 除息事件 (已公告未除息 / 已除息；TWSE 預告表 + 計算結果 + FinMind)：區間扣股利、次日門檻換算、權證除息保護
    try:
        exd = exdiv.load_all([m["id"] for m in DESK], last_td or dt.date.today().isoformat(), exdiv.get_json, archive=(prev_archive.get("exdiv") or {}))
    except Exception as e:  # noqa: BLE001
        log.warning("exdiv: %s", e)
        exd = {}
    # 歸檔以上次的內容為底：任一標的這次抓價/抓 HiStock 失敗都不會把它的舊列洗掉
    prev_gov8 = prev_archive.get("gov8") or {}
    archive = {"gov8": {k: list(v or []) for k, v in prev_gov8.items()}, "opt_hist": ohist,
               "opt_last": ({k: v for k, v in feat.items() if k not in ("series", "_carried")} if feat and not feat.get("_carried") else prev_archive.get("opt_last")),
               "holdings_00981A": dict(prev_archive.get("holdings_00981A") or {}),
               "exdiv": ({sid: {"asof": v.get("asof"), "upcoming": v.get("upcoming")} for sid, v in exd.items() if v.get("upcoming")} or dict(prev_archive.get("exdiv") or {})),
               "fresh_log": list(prev_archive.get("fresh_log") or [])}
    frames: dict[str, dict] = {}
    exd_i: dict[str, dict] = {}              # 各標的依自己資料日重切的除息事件
    for meta in DESK:
        sid = meta["id"]
        try:
            days = 1100
            if prev_gov8.get(sid):                   # 價格視窗涵蓋歸檔起點 (FIFO/隱含價需要當日價)
                days = max(days, (dt.date.today() - dt.date.fromisoformat(prev_gov8[sid][0]["date"])).days + 10)
            px_all = prices(sid, max(days, desk_bands.BAND_DAYS))        # 機率帶校準要約 1,300 交易日；其他區塊切回 days
            if px_all.empty:
                continue
            divs_all = dividends(sid, str(px_all["date"].iloc[0])) if meta["kind"] != "lev2" else []
            if meta["kind"] != "lev2":           # 併入 TWSE 除息計算結果 (除息前一晚即可得；不依賴 FinMind 當晚是否更新)
                have = {d["date"] for d in divs_all}
                for e in (exd.get(sid) or {}).get("realized") or []:
                    if e.get("ex_date") and e["ex_date"] not in have and e.get("before") and e.get("cash") and e["ex_date"] <= str(px_all["date"].iloc[-1]):
                        divs_all.append({"date": e["ex_date"], "div": float(e["cash"]), "before": float(e["before"])})
            adj_all = adj_close(px_all, divs_all)
            keep = (px_all["date"] >= (dt.date.today() - dt.timedelta(days=days)).isoformat()).values
            px, adj = px_all[keep].reset_index(drop=True), adj_all[keep].reset_index(drop=True)      # 含息因子只依賴之後的股利 → 切片與只抓 days 相同
            divs = [d for d in divs_all if d["date"] >= str(px["date"].iloc[0])]
            frames[sid] = {"px": px, "adj": adj, "divs": divs, "px_band": px_all, "adj_band": adj_all}
        except Exception as e:  # noqa: BLE001
            log.warning("desk prices %s: %s", sid, e)
    for meta in DESK:                                  # 價格失敗的標的仍更新八大歸檔 (HiStock 不依賴價格)
        if meta["id"] not in frames:
            try:
                archive["gov8"][meta["id"]] = gov8_archive(meta["id"], prev_gov8.get(meta["id"]))
            except Exception as e:  # noqa: BLE001
                log.warning("desk %s gov8 (no px): %s", meta["id"], e)
    adj50 = pd.DataFrame({"date": frames["0050"]["px"]["date"].astype(str).values, "adj": frames["0050"]["adj"].values}) if "0050" in frames else pd.DataFrame(columns=["date", "adj"])
    tw_adj = tw
    # 機率帶乘數每日滾動重算 (研究 opt_bands)；失敗的標的沿用 desk_evidence 凍結值
    band_mon: dict = {}
    try:
        rg_live, band_mon = desk_bands.range_live(ev, tw, frames, ohist)
        ev = {**ev, "range": rg_live}
    except Exception as e:  # noqa: BLE001
        log.warning("range_live: %s", e)
    live = band_mon.get("live") or {}
    desk["opt"]["band_monitor"] = band_mon
    if tw_close:
        desk["index"] = {"date": str(tw["date"].iloc[-1]), "close": _r(tw_close, 2), "stale": bool(last_td and str(tw["date"].iloc[-1]) < last_td),
                         "range": range_block(None, tw_close, None, None, feat, ev, cal, live=bool(live.get("TWII")))}
        try:
            desk["index"]["zones"] = zones_block(tw.tail(400), is_index=True)
        except Exception as e:  # noqa: BLE001
            log.debug("TWII zones: %s", e)
    # 加權含息近似 (研究 t5 E)：加權價格報酬 × 0050 配息成分；只給 00663L 耗損監控用 (年度層級才可靠)
    tw_tr = tw[["date", "close"]].copy()
    if "0050" in frames and len(tw_tr):
        k50 = pd.DataFrame({"date": adj50["date"].values, "k": frames["0050"]["adj"].values / frames["0050"]["px"]["close"].astype(float).values})
        tw_tr = tw_tr.merge(k50, on="date", how="left")
        tw_tr["k"] = tw_tr["k"].ffill().bfill().fillna(1.0)
        tw_tr["close"] = float(tw_tr["close"].iloc[0]) * ((1 + tw_tr["close"].astype(float).pct_change().fillna(0)) * (1 + tw_tr["k"].pct_change().fillna(0))).cumprod()
    hold = {}
    try:
        hold = holdings_00981A() if "00981A" in frames else {}
    except Exception as e:  # noqa: BLE001
        log.warning("00981A holdings: %s", e)
    for meta in DESK:
        sid = meta["id"]
        if sid not in frames:
            continue
        px, adj, divs = frames[sid]["px"], frames[sid]["adj"], frames[sid]["divs"]
        dates = px["date"].astype(str).reset_index(drop=True)
        close = float(px["close"].iloc[-1])
        rec: dict = {"id": sid, "name": meta["name"], "kind": meta["kind"], "date": dates.iloc[-1], "close": _r(close, 2),
                     "chg": _r((close / float(px["close"].iloc[-2]) - 1) * 100, 2) if len(px) > 1 else None,
                     "stale": bool(last_td and dates.iloc[-1] < last_td)}
        steps = {
            "ma": lambda: ma_block(px, adj, meta, ev),
            "overlay": lambda: overlay_block(meta, px, adj, ev),
            "pivots": lambda: pivot_block(meta, px, adj, ev),
            "zones": lambda: zones_block(px),
        }
        for k, fn in steps.items():
            try:
                rec[k] = fn()
            except Exception as e:  # noqa: BLE001
                log.warning("desk %s %s: %s", sid, k, e)
        try:
            j = pd.DataFrame({"date": dates.values, "a": adj.values}).merge(tw_adj[["date", "close"]], on="date")
            b, idio = _beta_idio(j["a"], j["close"], 120 if sid == "00981A" else 250)
            la = ((ev.get("range") or {}).get("assets") or {}).get(sid) or {}
            if (live.get(sid) and la.get("beta") is not None and la.get("idio") is not None
                    and math.isfinite(la["beta"]) and math.isfinite(la["idio"])):   # 與校準同口徑的 β/σ_idio
                b, idio = la["beta"], la["idio"]
            try:
                cal_s = twse.next_trading_days(dates.iloc[-1], 25)
            except Exception:  # noqa: BLE001
                cal_s = cal
            rec["range"] = range_block(meta, close, b, idio, feat, ev, cal_s, live=bool(live.get(sid)))
            ev_x = exdiv.rebase(exd.get(sid) or {}, str(dates.iloc[-1]))
            exd_i[sid] = ev_x
            ups = [u for u in (ev_x.get("upcoming") or []) if not u.get("suspicious")]
            for u in ev_x.get("upcoming") or []:                       # 合理性：股利 > 收盤 15% 不採用
                D = u.get("cash") or u.get("cash_est")
                if D and D / close > 0.15:
                    u["suspicious"] = True
            ups = [u for u in ups if not u.get("suspicious")]
            rec["exdiv"] = {"upcoming": ev_x.get("upcoming") or [], "last": (ev_x.get("realized") or [])[-1:], "estimate": ev_x.get("estimate"),
                            "board_pending": ev_x.get("board_pending"), "sources": ev_x.get("sources"), "carried_from": ev_x.get("carried_from")}
            if rec.get("range") and ups:
                exdiv.adjust_range(rec["range"], ups, cal_s)          # 窗口含除息：low 扣全額、high 依比例 (看盤價口徑，px_tr 為含息口徑)
            if rec["range"] and feat.get("date") and feat["date"] != rec["date"]:
                rec["range"]["date_mismatch"] = {"opt": feat["date"], "close": rec["date"]}
        except Exception as e:  # noqa: BLE001
            log.warning("desk %s range: %s", sid, e)
        try:   # 次日除息：所有「次日收盤 = X」門檻 (含息價的一次齊次函數) 乘 f = (P−D)/P → 換成除息後看盤價 (精確)
            ev_x = exd_i.get(sid) or exdiv.rebase(exd.get(sid) or {}, str(dates.iloc[-1]))
            ups_ok = [u for u in (ev_x.get("upcoming") or []) if not u.get("suspicious")]
            nxt = twse.next_trading_days(dates.iloc[-1], 1)[0]
            fac, e_ = exdiv.next_day_factor(ups_ok, nxt, close)
            if fac != 1.0 and e_:
                o = rec.get("overlay") or {}
                ref_x = float(adj.iloc[-1]) * fac          # 除息參考價 P−D：次日漲跌幅與距離都以它為基準
                for x in o.get("levels") or []:
                    if x["key"] == "ma60_exit" and x.get("price"):
                        x["price_tr"], x["price"] = x["price"], _r(x["price"] * fac, 2)
                        x["dist"] = _r((x["price"] / ref_x - 1) * 100, 2)
                if o.get("primary"):
                    o["primary"]["price_tr"] = o["primary"].get("price")
                    for k2 in ("price", "reenter"):
                        if o["primary"].get(k2):
                            o["primary"][k2] = _r(o["primary"][k2] * fac, 2)
                    o["primary"]["dist"] = _r((o["primary"]["price"] / ref_x - 1) * 100, 2)
                    st_ = o.get("_stop")
                    if st_ and not st_[2]:
                        o["_stop"] = (o["primary"]["price"], st_[1], st_[2])
                for cx in (rec.get("ma") or {}).get("crosses") or []:
                    if cx.get("trigger_next"):          # feasible_next 不重算：|T·f/(P·f) − 1| = |T/P − 1|
                        cx["trigger_next"] = _r(cx["trigger_next"] * fac, 2)
                for m_ in ((rec.get("ma") or {}).get("ma") or {}).values():
                    if m_.get("deduct_next_adj"):
                        m_["deduct_next_adj"] = _r(m_["deduct_next_adj"] * fac, 2)
                rec["next_exdiv"] = {"date": e_["ex_date"], "cash": e_.get("cash") or e_.get("cash_est"), "f": round(fac, 6), "est": not e_.get("cash") or not e_.get("before"),
                                     "note": "次日門檻 (MA60 出場/站回、交叉觸發、含息扣抵) 已換算成除息後看盤價 (×(P−D)/P)" + ("；金額或前收為估計" if (not e_.get("cash") or not e_.get("before")) else "")}
            if rec.get("ma"):
                rec["ma"]["exdiv"] = exdiv.ma_exdiv(dates.tolist(), px["close"].astype(float).tolist(), adj.tolist(),
                                                    ev_x.get("realized") or [], ups_ok, twse.next_trading_days(dates.iloc[-1], 70))
        except Exception as e:  # noqa: BLE001
            log.debug("desk %s exdiv thresholds: %s", sid, e)
        try:
            under = None
            if meta.get("under") == "0050" and "0050" in frames:
                under = pd.DataFrame({"date": dates.values}).merge(adj50, on="date", how="left")["adj"].ffill()
            elif meta.get("under") == "TWII":
                under = pd.DataFrame({"date": dates.values}).merge(tw_tr[["date", "close"]], on="date", how="left")["close"].ffill()
            rec["defense"] = defense_block(meta, adj, dates, tw_adj, feat, under)
        except Exception as e:  # noqa: BLE001
            log.warning("desk %s defense: %s", sid, e)
        try:
            o = rec.get("overlay") or {}
            sz = sizing(o, rec.get("defense") or {}, float(adj.iloc[-1]) * float((rec.get("next_exdiv") or {}).get("f") or 1.0))
            if sz:
                o["sizing"] = sz
        except Exception as e:  # noqa: BLE001
            log.warning("desk %s sizing: %s", sid, e)
        (rec.get("overlay") or {}).pop("_stop", None)
        try:
            arch = gov8_archive(sid, prev_gov8.get(sid))
            archive["gov8"][sid] = arch
            rec["gov8"] = gov8_block(sid, arch, px, ev)
        except Exception as e:  # noqa: BLE001
            log.warning("desk %s gov8: %s", sid, e)
        try:   # 回撤價位與 k=20 機率帶的相對位置
            lv20 = ((rec.get("range") or {}).get("levels") or {}).get("20") or {}
            for L in (rec.get("pivots") or {}).get("ladder") or []:
                l10, l20 = (lv20.get("low10") or {}).get("px"), (lv20.get("low20") or {}).get("px")
                if l10 and l20:
                    L["vs_band20"] = "高於 20 日 low20 (機率 >20%)" if L["price"] >= l20 else "介於 low10~low20 (10~20%)" if L["price"] >= l10 else "低於 low10 (<10%)"
        except Exception:  # noqa: BLE001
            pass
        desk["instruments"][sid] = rec
    # 00981A 主動 ETF：持股快照只往前推 (日期較舊的回應不覆蓋)
    try:
        if "00981A" in frames:
            f = frames["00981A"]
            H = archive["holdings_00981A"]
            if hold:
                last_d = str((H.get("last") or {}).get("date") or "")
                if hold.get("date", "") > last_d:
                    if H.get("last"):
                        H["prev"] = H["last"]
                    H["last"] = hold
                elif hold.get("date", "") == last_d:
                    H["last"] = hold
            a981 = pd.DataFrame({"date": f["px"]["date"].astype(str).values, "adj": f["adj"].values})
            desk["instruments"]["00981A"]["active"] = active_block(a981, adj50, tw, float(f["px"]["close"].iloc[-1]), f["divs"], hold, H.get("prev"), ev)
    except Exception as e:  # noqa: BLE001
        log.warning("desk 00981A active: %s", e)
    try:
        if "2330" in frames:
            try:
                px_long = prices("2330", 7000)
            except Exception:  # noqa: BLE001
                px_long = None
            w981 = (next((x["w"] for x in (hold.get("holdings") or []) if x["code"] == "2330"), None), hold.get("date")) if hold else None
            desk["instruments"]["2330"]["tsmc"] = tsmc_block(frames["2330"]["px"], frames["2330"]["adj"], adj50, ev, px_long, w981)
    except Exception as e:  # noqa: BLE001
        log.warning("desk 2330: %s", e)
    try:
        if "00981A" in frames:
            wpx = prices(WARRANT["id"], 300, split_adjust=False)
            wb = warrant_block(wpx, frames["00981A"]["px"], frames["00981A"]["adj"], ev, exd.get("00981A"))
            if wb:
                wb["id"], wb["name"], wb["kind"] = WARRANT["id"], WARRANT["name"], "warrant"
                wb["stale"] = bool(last_td and wb.get("date") and wb["date"] < last_td)
                rg = ((desk["instruments"].get("00981A") or {}).get("range") or {}).get("levels") or {}
                if wb.get("iv") and (wb.get("terms") or {}).get("strike"):     # 00981A 機率帶換算成權證價 (IV 不變)
                    te = wb.get("terms_estimated") or {}
                    K, ratio = te.get("strike", wb["terms"]["strike"]), te.get("ratio", wb["terms"]["ratio"])
                    T = wb["trade_days_to_expiry"] / 252
                    maps = {}
                    for k in ("5", "10"):
                        if int(k) < wb["trade_days_to_last"] and rg.get(k):
                            # 有除息保護的權證：價值是含息價路徑的函數 → 用含息口徑 px_tr (沒有除息時就是 px) 配現行條款 (驗證：用扣息價配推算條款會高估 9~11%)
                            pxq = lambda q_: (rg[k].get(q_) or {}).get("px_tr") or (rg[k].get(q_) or {}).get("px")   # noqa: E731
                            maps[k] = {q: {"under": (rg[k].get(q) or {}).get("px"),
                                           "warrant": _r(_bs(pxq(q) or 0, K, max(T - int(k) / 252, 1e-9), wb["iv"])[0] * ratio, 3) if pxq(q) else None}
                                       for q in ("low10", "low20", "high80", "high90")}
                    wb["band_map"] = maps
                desk["instruments"][WARRANT["id"]] = wb
    except Exception as e:  # noqa: BLE001
        log.warning("desk warrant: %s", e)
    desk["order"] = [m["id"] for m in DESK] + [WARRANT["id"]]
    # 機率帶上線帳本：記錄今天發布的 1/5/20 日帶，評估已到期的 (實際高低價)
    try:
        led = list(prev_archive.get("band_ledger") or [])
        if not feat.get("_carried") and not feat.get("calendar_fallback"):
            if desk.get("index", {}).get("range") and not desk["index"].get("stale"):
                ledger_add(led, "TWII", desk["index"]["date"], desk["index"]["close"], desk["index"]["range"])
            for sid, r in desk["instruments"].items():
                if r.get("range") and not r.get("stale") and not (r["range"].get("date_mismatch")):
                    ledger_add(led, sid, r["date"], r["close"], r["range"])
        keep = sorted({x["date"] for x in led})[-LEDGER_MAX_DAYS:]
        led = [x for x in led if x["date"] in set(keep)]
        archive["band_ledger"] = led
        hl = {sid: f["px"][["date", "high", "low", "close"]].astype({"date": str}) for sid, f in frames.items()}
        if len(tw) and {"high", "low", "close"} <= set(tw.columns):
            hl["TWII"] = tw[["date", "high", "low", "close"]].astype({"date": str})
        desk["ledger"] = {"since": led[0]["date"] if led else None, "n_records": len(led), "stats": ledger_eval(led, hl),
                          "note": "上線後逐日記錄當天發布的機率帶，到期後用實際最高/最低價核對 (真正的樣本外)；樣本累積前以回測觸及率為準。"}
    except Exception as e:  # noqa: BLE001
        log.warning("desk ledger: %s", e)
        archive["band_ledger"] = list(prev_archive.get("band_ledger") or [])
    # 資料新鮮度：各區塊內容日期 vs 應有日期 (最後交易日)；fresh_log 累積後可統計各來源實際幾點出現當日資料
    try:
        now_s = dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M")
        I_ = desk["instruments"]
        F = exdiv.freshness
        tw48 = [((v.get("sources") or {}).get("TWT48U")) for v in exd.values()]
        ex_ok = bool(exd) and all(x == "ok" for x in tw48)
        ex_stale = bool(exd) and not ex_ok and all(x in ("ok", "stale") for x in tw48)
        # ADR：美股 d 日收盤在台北 d+1 04:00~05:00 → 最近一個已收盤的美股交易日 = (台北現在 − 29.5 小時) 的日期，遇週末往前 (美股假日會誤報 1 天)
        u_ = (dt.datetime.now(config.TZ) - dt.timedelta(hours=29, minutes=30)).date()
        while u_.weekday() >= 5:
            u_ -= dt.timedelta(days=1)
        adr_expect = u_.isoformat()
        g8_to = {sid: (r.get("gov8") or {}).get("to") for sid, r in I_.items() if r.get("kind") != "warrant" and (r.get("gov8") or {}).get("to")}
        g8_lag = sorted(sid for sid, d_ in g8_to.items() if last_td and d_ < last_td)
        fr = [F("日K (FinMind)", max([r.get("date") or "" for r in I_.values() if r.get("kind") != "warrant"] or [""]) or None, last_td, now_s),
              F("加權指數 (FinMind)", (desk.get("index") or {}).get("date"), last_td, now_s),
              F("台指選擇權 (期交所)", (desk.get("opt") or {}).get("date"), last_td, now_s, carried=bool((desk.get("opt") or {}).get("carried"))),
              F("八大行庫 (HiStock)", min(g8_to.values()) if g8_to else None, last_td, now_s),
              F("00981A 持股 (統一投信)", ((((I_.get("00981A") or {}).get("active") or {}).get("holdings")) or {}).get("date"), last_td, now_s),
              F("權證報價 (TWSE MIS)", (I_.get(WARRANT["id"]) or {}).get("quote_date") or (I_.get(WARRANT["id"]) or {}).get("date"), last_td, now_s),
              F("TSM ADR (Yahoo)", (((I_.get("2330") or {}).get("tsmc") or {}).get("adr") or {}).get("gap_tsm_date") or (((I_.get("2330") or {}).get("tsmc") or {}).get("adr") or {}).get("date"), adr_expect, now_s),
              F("除息公告 (TWSE/FinMind)", last_td if (ex_ok or ex_stale) else None, last_td, now_s, carried=ex_stale)]
        if g8_lag:
            fr[3]["lagging"] = g8_lag
        if (desk.get("opt") or {}).get("source"):
            fr[2]["src"] = desk["opt"]["source"]
        desk["freshness"] = fr
        archive["fresh_log"] = (archive.get("fresh_log") or [])[-560:] + [{"run": now_s, "src": x["name"], "date": x["date"]} for x in fr]
    except Exception as e:  # noqa: BLE001
        log.debug("freshness: %s", e)
    desk["data_asof"] = {"last_td": last_td, "twii": (desk.get("index") or {}).get("date"), "opt": (desk.get("opt") or {}).get("date"),
                         "stale": [sid for sid, r in desk["instruments"].items() if r.get("stale")] + (["TWII"] if (desk.get("index") or {}).get("stale") else [])
                         + (["TXO"] if (desk.get("opt") or {}).get("stale") else [])}
    # 縮水保護：任一歸檔比上次少 (扣掉上限截斷) → 保留上次的列
    shrunk = []
    for sid, old in prev_gov8.items():
        new = archive["gov8"].get(sid) or []
        if len(new) < min(len(old or []), ARCHIVE_MAX):
            archive["gov8"][sid] = list(old)[-ARCHIVE_MAX:]
            shrunk.append(sid)
    if len(archive["opt_hist"]) < min(len(prev_archive.get("opt_hist") or []), OPT_HIST_MAX):
        archive["opt_hist"] = list(prev_archive.get("opt_hist") or [])[-OPT_HIST_MAX:]
        shrunk.append("opt_hist")
    if shrunk:
        log.warning("desk_archive 縮水保護：%s 保留上次內容", shrunk)
    if len(archive.get("band_ledger") or []) < len(prev_archive.get("band_ledger") or []) and len({x["date"] for x in prev_archive.get("band_ledger") or []}) < LEDGER_MAX_DAYS:
        archive["band_ledger"] = list(prev_archive.get("band_ledger") or [])
        shrunk.append("band_ledger")
    desk["archive_stats"] = {"gov8": {sid: len(v) for sid, v in archive["gov8"].items()}, "opt_hist": len(archive["opt_hist"]),
                             "band_ledger": len(archive.get("band_ledger") or []), "loaded": prev_archive.get("_loaded", True)}
    save_archive(archive)
    return desk, archive
