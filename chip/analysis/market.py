"""大盤籌碼評分模型 (v2)。

流程：build_frame() 合併所有來源 → add_features() 衍生指標 → score_frame() 每日各因子分數、
市場狀態、狀態自適應綜合分 → assess() 最新一日判讀 (理由、信心度、轉折、進場建議)
→ intraday_hint() 盤中即時提示 → evaluate()/suggest_weights() 歷史驗證。

v2 改進：
- 市場狀態機 (多頭/空頭/盤整，依月線季線結構) 決定各因子權重
- 外資「現貨 + 期貨」多空一致性；期貨部位改用一年百分位 (相對) 而非固定口數
- 外資極端賣超/買超的高潮反轉修正
- 融資改為「融資變化 × 指數變化」四象限 (接刀 / 追價 / 沉澱 / 斷頭)
- 八大行庫以連續買超天數判斷護盤
- 量價加入 5 日量能型態 (量縮整理)、趨勢加入月線斜率與 20 日新高低
- 綜合分 3 日 EMA 平滑、5 日動能、多空轉折偵測、信心度 (因子一致性 + 資料完整度)
- 進場建議依市場狀態使用不同規則
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

import numpy as np
import pandas as pd

from .. import config, store
from ..sources import finmind, histock, taifex, twse, wantgoo
from .common import Factor, composite, fmt, regime, streak, zscore

log = logging.getLogger(__name__)

# ----------------------------------------------------------------- 權重
# v3 (2010~2026 長歷史驗證後)：外資期貨部位是唯一逐年穩定有效的因子 → 加重；
# 外資現貨與指數趨勢在 10~20 日視野幾乎無預測力甚至反向 → 降權；新增「超跌回歸」因子 (月線負乖離)。
WEIGHTS = {
    "foreign": 1.5, "trust": 1.5, "dealer": 0.5, "fut_foreign": 3.5, "gov8": 1.5,
    "margin": 1.5, "maint": 1.5, "short": 1.0, "sbl": 1.0, "pcr": 1.0, "large": 1.0,
    "volume": 1.0, "trend": 1.0, "reversion": 1.5, "global": 1.5, "fx_flow": 1.0,
}
# 依市場狀態調整：空頭時外資 (現貨+期貨) 的 IC 最高、官股護盤/維持率重要；多頭時趨勢因子預測力最差
STATE_WEIGHTS = {
    "多頭": {**WEIGHTS, "trend": 0.75, "margin": 2.0, "gov8": 1.0, "pcr": 0.8, "maint": 1.0},
    "空頭": {**WEIGHTS, "foreign": 2.0, "fut_foreign": 4.0, "gov8": 2.5, "maint": 2.0, "pcr": 1.5, "sbl": 1.5, "reversion": 2.0},
    "盤整": dict(WEIGHTS),
}
NAMES = {
    "foreign": "外資現貨買賣超", "trust": "投信買賣超", "dealer": "自營商買賣超",
    "fut_foreign": "外資台指期部位 (含現貨一致性)", "gov8": "八大行庫買賣超", "margin": "融資量價象限",
    "maint": "大盤融資維持率", "short": "融券餘額變化", "sbl": "借券賣出餘額變化",
    "pcr": "選擇權 Put/Call 比", "large": "大額交易人(特定法人)淨部位", "volume": "量價關係", "trend": "指數趨勢",
    "reversion": "超跌回歸 (月線負乖離)", "global": "國際盤 (VIX/費半/韓股)", "fx_flow": "匯率資金流 (台幣/油價)",
}
FACTOR_KEYS = list(WEIGHTS)
TUNED_PATH = config.DATA_DIR / "tuned_weights.json"


def tuned_multipliers() -> dict:
    """由 `python cli.py optimize` 依長歷史 IC 產生的權重倍數 (data/tuned_weights.json)。CHIP_USE_TUNED=0 可停用。"""
    if os.getenv("CHIP_USE_TUNED", "1") != "1" or not TUNED_PATH.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(TUNED_PATH.read_text(encoding="utf-8")).items() if k in WEIGHTS}
    except Exception:  # noqa: BLE001
        return {}


def effective_weights(state: str) -> dict:
    base = STATE_WEIGHTS.get(state, WEIGHTS)
    mult = tuned_multipliers()
    return {k: round(w * mult.get(k, 1.0), 3) for k, w in base.items()}


# ============================================================ 資料合併
def _safe(fn, meta: dict, name: str, default=None):
    try:
        r = fn()
        if isinstance(r, pd.DataFrame):
            meta[name] = {"status": "ok", "rows": len(r), "latest": str(r["date"].max()) if not r.empty and "date" in r else ""}
        elif isinstance(r, dict):
            meta[name] = {"status": "ok", "latest": str(r.get("date", ""))}
        elif r is None:
            meta[name] = {"status": "empty"}
        else:
            meta[name] = {"status": "ok"}
        return r
    except Exception as e:  # noqa: BLE001
        log.warning("%s failed: %s", name, e)
        meta[name] = {"status": f"fail: {type(e).__name__}: {str(e)[:80]}"}
        return default


def build_frame(use_wantgoo: bool = True) -> tuple[pd.DataFrame, dict]:
    """合併各來源成以 date 為索引的日資料。回傳 (df, meta)。"""
    meta: dict = {}
    price = _safe(finmind.taiex_price, meta, "FinMind 加權指數", pd.DataFrame())
    inst = _safe(finmind.total_institutional, meta, "FinMind 三大法人", pd.DataFrame())
    margin = _safe(finmind.total_margin, meta, "FinMind 融資融券", pd.DataFrame())
    fut = _safe(finmind.tx_futures_institutional, meta, "FinMind 期貨法人", pd.DataFrame())
    from . import gov8 as gov8mod
    gov8 = _safe(lambda: gov8mod.market_history(), meta, "HiStock 八大行庫 (+SQLite 累積)", pd.DataFrame())   # HiStock 近半年 ∪ SQLite ∪ 已發布
    pcr = _safe(taifex.put_call_ratio, meta, "TAIFEX P/C ratio", pd.DataFrame())
    large = _safe(taifex.large_traders_tx, meta, "TAIFEX 大額交易人")
    fut_latest = _safe(taifex.futures_institutional_latest, meta, "TAIFEX 期貨法人(最新)", pd.DataFrame())
    sbl_today = _safe(lambda: {k: v for k, v in (twse.sbl_balance() or {}).items() if k != "stocks"}, meta, "TWSE 借券賣出餘額")
    twse_inst = _safe(twse.institutional_daily, meta, "TWSE 三大法人(當日)")
    twse_margin = _safe(twse.margin_daily, meta, "TWSE 融資融券(當日)")
    twse_mkt = _safe(twse.market_daily, meta, "TWSE 市場成交(當月)", pd.DataFrame())
    wg = _safe(wantgoo.fetch_market, meta, "玩股網 (Playwright)") if use_wantgoo else None
    from ..sources import global_markets as gm
    glob = _safe(lambda: gm.aligned_features(price["date"]) if not price.empty else pd.DataFrame(), meta, "Yahoo 國際市場", pd.DataFrame())
    if wg is None and use_wantgoo:
        meta["玩股網 (Playwright)"] = {"status": "unavailable (未安裝 playwright 或抓取失敗)"}

    # --- 累積只提供最新一日的指標到 SQLite
    if isinstance(pcr, pd.DataFrame) and not pcr.empty:
        store.upsert_frame(pcr, ["pcr_oi", "pcr_vol"])
    if isinstance(large, dict) and large.get("date"):
        store.upsert_metrics(large["date"], {k: v for k, v in large.items() if k != "date"})
    if isinstance(sbl_today, dict) and sbl_today.get("date"):
        store.upsert_metrics(sbl_today["date"], {"sbl_today": sbl_today["sbl_today"], "sbl_change": sbl_today["sbl_change"]})
    if wg:
        if not wg["margin_table"].empty:
            store.upsert_frame(wg["margin_table"], ["maint_ratio", "short_margin_ratio"])
        if not wg["sbl"].empty:
            store.upsert_frame(wg["sbl"], ["sbl_chg_lots", "sbl_chg_amt"])
        if not wg["banks"].empty:
            store.upsert_frame(wg["banks"], ["gov8_lots"])
        h = wg["headline"]
        if h.get("date"):
            store.upsert_metrics(h["date"], {k: v for k, v in h.items() if k != "date"})
    hist = store.load_metrics(["pcr_oi", "pcr_vol", "top10_spec_net", "top10_all_net", "oi_market",
                               "sbl_today", "sbl_change", "maint_ratio", "short_margin_ratio",
                               "sbl_chg_lots", "sbl_chg_amt"])

    if price.empty:
        price = twse_mkt[["date", "close", "change", "amount"]] if not twse_mkt.empty else pd.DataFrame(columns=["date", "close"])
    df = price.copy()
    for part in (inst, margin, fut, gov8[["date", "gov8_net"]] if not gov8.empty else gov8, hist):
        if isinstance(part, pd.DataFrame) and not part.empty:
            df = df.merge(part, on="date", how="outer")
    if isinstance(glob, pd.DataFrame) and not glob.empty:
        df = df.merge(glob, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    def _patch(rec: dict | None, cols: dict):
        if not rec or not rec.get("date"):
            return
        d = rec["date"]
        if d not in set(df["date"]):
            df.loc[len(df)] = {"date": d}
        idx = df.index[df["date"] == d][0]
        for src, dst in cols.items():
            if rec.get(src) is not None and (dst not in df or pd.isna(df.at[idx, dst])):
                df.at[idx, dst] = rec[src]

    _patch(twse_inst, {"foreign": "foreign", "trust": "trust", "dealer_self": "dealer_self",
                       "dealer_hedge": "dealer_hedge", "dealer": "dealer", "total": "total"})
    _patch(twse_margin, {"margin_lots": "margin_lots", "short_lots": "short_lots", "margin_amt": "margin_amt"})
    if not twse_mkt.empty:
        for _, r in twse_mkt.iterrows():
            _patch(r.to_dict(), {"close": "close", "amount": "amount", "change": "change"})
    if isinstance(fut_latest, pd.DataFrame) and not fut_latest.empty:
        _patch(fut_latest.iloc[0].to_dict(), {c: c for c in fut_latest.columns if c != "date"})
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["close"].notna()].reset_index(drop=True)
    meta["_built_at"] = dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S")
    meta["_wantgoo"] = wg
    meta["_gov8_rank"] = _safe(histock.government_banks_ranking, meta, "HiStock 八大行庫排行")
    return add_features(df), meta


# ============================================================ 衍生指標
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("open", "high", "low", "foreign", "trust", "dealer", "total", "margin_amt", "margin_lots", "short_lots", "gov8_net",
              "fut_foreign_net_oi", "pcr_oi", "amount", "maint_ratio", "sbl_chg_lots", "sbl_change",
              "top10_spec_net", "oi_market"):
        if c not in d:
            d[c] = np.nan
    for c in ("pcr_oi", "top10_spec_net", "oi_market", "maint_ratio", "fut_foreign_net_oi"):
        d[c] = d[c].ffill(limit=3)
    c = d["close"].astype(float)
    d["ret1"] = c.pct_change() * 100
    d["ret5"] = c.pct_change(5) * 100
    d["ret20"] = c.pct_change(20) * 100
    d["ma5"], d["ma20"], d["ma60"] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    d["bias20"] = (c / d["ma20"] - 1) * 100
    d["ma20_slope"] = (d["ma20"] / d["ma20"].shift(5) - 1) * 100
    d["hi20"] = c.rolling(20).max()
    d["lo20"] = c.rolling(20).min()
    d["amount_ma20"] = d["amount"].rolling(20).mean()
    d["vol_ratio"] = d["amount"] / d["amount_ma20"]
    d["amount_5d_ratio"] = d["amount"].rolling(5).mean() / d["amount_ma20"]
    for who in ("foreign", "trust", "dealer", "total"):
        d[f"{who}_5d"] = d[who].rolling(5, min_periods=1).sum()
        d[f"{who}_20d"] = d[who].rolling(20, min_periods=1).sum()
    d["foreign_streak"] = streak(d["foreign"])
    d["trust_streak"] = streak(d["trust"])
    d["gov8_streak"] = streak(d["gov8_net"])
    d["margin_chg1"] = d["margin_amt"].diff()
    d["margin_chg5"] = d["margin_amt"].diff(5)
    d["margin_pct20"] = d["margin_amt"].pct_change(20) * 100
    d["margin_div20"] = d["margin_pct20"] - d["ret20"]
    d["short_chg5"] = d["short_lots"].diff(5)
    d["short_ratio"] = d["short_lots"] / d["margin_lots"] * 100
    d["fut_foreign_chg1"] = d["fut_foreign_net_oi"].diff()
    d["fut_foreign_chg5"] = d["fut_foreign_net_oi"].diff(5)
    d["fut_foreign_pct"] = d["fut_foreign_net_oi"].rolling(250, min_periods=60).rank(pct=True)
    # 現貨 5 日淨買 與 期貨 5 日淨部位變化 同向 = 一致 (+1 多 / -1 空)，反向 = 避險/中性 0
    d["foreign_consistency"] = np.where((d["foreign_5d"] > 0) & (d["fut_foreign_chg5"] > 0), 1,
                                        np.where((d["foreign_5d"] < 0) & (d["fut_foreign_chg5"] < 0), -1, 0))
    d["gov8_5d"] = d["gov8_net"].rolling(5, min_periods=1).sum()
    d["gov8_20d"] = d["gov8_net"].rolling(20, min_periods=1).sum()
    sbl = d["sbl_chg_lots"].where(d["sbl_chg_lots"].notna(), d["sbl_change"])
    d["sbl_chg"] = sbl
    d["sbl_chg5"] = sbl.rolling(5, min_periods=1).sum()
    d["large_ratio"] = d["top10_spec_net"] / d["oi_market"]
    for gcol in ("g_vix_level", "g_vix_chg", "g_sox_r1", "g_sox_r20", "g_kospi_r1", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1", "g_sp500_hi20", "g_btc_r20",
                 "g_copper_gold_r20", "g_natgas_r20", "g_usdjpy_r5", "g_curve_10y_3m", "g_bdry_r20",
                 "g_usdtwd_r20", "g_usdtwd_r60", "g_usdtwd_streak", "g_oil_r60"):
        if gcol not in d:
            d[gcol] = np.nan
    d["state"] = np.select([(c > d["ma60"]) & (d["ma20"] > d["ma60"]), (c < d["ma60"]) & (d["ma20"] < d["ma60"])],
                           ["多頭", "空頭"], "盤整")
    d.loc[d["ma60"].isna(), "state"] = "盤整"
    return d


# ============================================================ 因子評分 (向量化)
def _sel(conds, vals, default, index, na_mask=None) -> pd.Series:
    s = pd.Series(np.select(conds, vals, default), index=index).astype(float)
    if na_mask is not None:
        s[na_mask] = np.nan
    return s


def score_frame(d: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=d.index)
    z = lambda col, w=60: zscore(d[col], w)  # noqa: E731

    # 外資現貨：當日 + 5 日 z；極端高潮反轉修正；連買 ≥5 日後歷史報酬顯著落後 (t≈-5) → 追價修正
    z1, z5 = z("foreign"), z("foreign_5d")
    rev = np.where((z1 < -2.2) & (d["ret1"] < -1.5), 0.5, np.where((z1 > 2.2) & (d["bias20"] > 5), -0.5, 0))
    rev = rev + np.where(d["foreign_streak"] >= 5, -0.5, 0)
    s["f_foreign"] = (0.5 * z1 + 0.5 * z5 + rev).clip(-2, 2)
    s["f_trust"] = z("trust_5d").clip(-2, 2)
    s["f_dealer"] = z("dealer").clip(-2, 2)

    # 外資期貨：一年百分位 (相對水準) 50% + 日變化 30% + 現貨/期貨一致性 20%
    level = ((d["fut_foreign_pct"] - 0.5) * 4).clip(-2, 2)
    chg = z("fut_foreign_chg1").clip(-2, 2)
    cons = d["foreign_consistency"] * 2
    s["f_fut_foreign"] = (0.5 * level + 0.3 * chg.fillna(0) + 0.2 * cons).clip(-2, 2)
    s.loc[level.isna(), "f_fut_foreign"] = np.nan

    # 八大行庫：5 日 z + 連續護盤/調節
    g = z("gov8_5d").clip(-2, 2)
    protect = (d["gov8_streak"] >= 3) & (d["ret5"] < 0)
    trim = (d["gov8_streak"] <= -3) & (d["ret5"] > 0)
    g = g.where(~protect, np.maximum(g, 1.5)).where(~trim, np.minimum(g, -1.0))
    s["f_gov8"] = g

    # 融資四象限 (v3：長歷史顯示「斷頭清洗」後 10 日 +1.5%/勝率 66% 是最強訊號 → +1.5/+2；
    # 追價過熱與散戶接刀對未來報酬幾乎無影響 → 由 -2 降為 -1；籌碼沉澱無顯著效果 → +1)
    m20, r20 = d["margin_pct20"], d["ret20"]
    s["f_margin"] = _sel(
        [(m20 < -4) & (r20 < -3), (m20 > 2) & (r20 < -2), (m20 - r20 > 4), (m20 - r20 > 1.5),
         (m20 - r20 < -1.5) & (r20 > 0), (m20 - r20 < -1.5)],
        [np.where(d["ret5"] > 0, 2, 1.5), -1, -1, -0.5, 1, 0.5], 0, d.index, m20.isna() | r20.isna())

    r = d["maint_ratio"]
    s["f_maint"] = _sel([r >= 170, r >= 160, r >= 150], [1, 0, -1], -2, d.index, r.isna())

    zs = z("short_chg5")
    s["f_short"] = pd.Series(np.where(d["ret5"] > 0, 0.5 * zs, -0.5 * zs), index=d.index).clip(-2, 2)
    s.loc[zs.isna(), "f_short"] = np.nan
    s["f_sbl"] = (-z("sbl_chg5")).clip(-2, 2)

    p = d["pcr_oi"]
    s["f_pcr"] = _sel([p >= 120, p >= 110, p > 90, p > 80], [2, 1, 0, -1], -2, d.index, p.isna())
    lr = d["large_ratio"]
    s["f_large"] = _sel([lr > 0.10, lr > 0.02, lr > -0.02, lr > -0.10], [2, 1, 0, -1], -2, d.index, lr.isna())

    # 量價：單日型態 + 5 日量能型態 (v3：「量縮整理」歷史上 10 日 -2%/勝率 32% → 改為 -0.5)
    r1, vr, a5 = d["ret1"], d["vol_ratio"], d["amount_5d_ratio"]
    s["f_volume"] = _sel(
        [(vr > 1.8) & (r1 < -1.0), (r1 > 0.3) & (vr > 1.1), (r1 < -0.3) & (vr > 1.2),
         (a5 < 0.8) & (d["ret5"].abs() < 2) & (d["close"] > d["ma20"]), (r1 < -0.3) & (vr < 0.8), (r1 > 0.3) & (vr < 0.8)],
        [-2, 1.5, -1.5, -0.5, 0.5, 0], 0, d.index, vr.isna() | r1.isna())

    # 趨勢：均線結構 + 月線斜率 + 20 日新高/新低 (v3：正乖離 >6% 歷史上仍續漲 → 不再扣分；負乖離交給回歸因子)
    c, ma20, ma60, b = d["close"], d["ma20"], d["ma60"], d["bias20"]
    tr = np.select([(c > ma20) & (c > ma60) & (ma20 > ma60), (c > ma20) & (c > ma60), c > ma20, (c < ma20) & (c > ma60)],
                   [2, 1.5, 1, -1], -2).astype(float)
    tr = tr + np.where(d["ma20_slope"] > 0.5, 0.5, np.where(d["ma20_slope"] < -0.5, -0.5, 0))
    tr = tr + np.where(c >= d["hi20"], 0.5, np.where(c <= d["lo20"], -0.5, 0))
    s["f_trend"] = pd.Series(tr, index=d.index).clip(-2, 2)
    s.loc[ma60.isna(), "f_trend"] = np.nan

    # 超跌回歸：月線負乖離 <-6% 後 10 日 +3.5%/勝率 75% (t≈4)，單邊因子 (只在負乖離時加分)
    s["f_reversion"] = pd.Series(np.where(b < 0, np.clip(-b / 3.0, 0, 2), 0.0), index=d.index)
    s.loc[b.isna(), "f_reversion"] = np.nan

    # 國際盤 (2007~2026 研究)：VIX 水準對台股未來 5~20 日報酬 IC 0.08~0.15、94% 年份為正 (高恐慌後偏強、VIX<13 自滿後偏弱)；
    # 費半 20 日 >15% 後台股 20 日勝率 74% (t=3.5)；KOSPI 前日 <-2.5% 後 5 日 +1.2% (t=2.5)。前晚美股漲跌只決定跳空、不預測之後。
    # 跨市場研究 (原物料/外匯/利率 × 台美韓日，2007~) 補充：銅金比 20 日 >+8% 後台股 5 日 +0.49% (t=3.9)、
    # 天然氣 20 日 >+25% 後 5 日 -0.26% (t=-1.9，IC -0.13 逐年 88% 為負)、日圓 5 日急升 (套利平倉) 後 5 日 +0.61% (t=2.6)、
    # 殖利率曲線倒掛期間 20 日 +1.70% vs 0.89 (t=2.6)、乾散貨 20 日 >+15% 後 +0.32% (t=2.6)
    vix, sox20, kospi = d["g_vix_level"], d["g_sox_r20"], d["g_kospi_r1"]
    cg, ng, jpy, curve, bdry = d["g_copper_gold_r20"], d["g_natgas_r20"], d["g_usdjpy_r5"], d["g_curve_10y_3m"], d["g_bdry_r20"]
    gl = pd.Series(np.select([vix >= 30, vix >= 22, vix >= 16, vix >= 13], [1.5, 0.75, 0.25, 0], -1.0), index=d.index).astype(float)
    gl = gl + np.where(sox20 > 15, 0.75, np.where(sox20 > 8, 0.25, np.where(sox20 < -10, -0.5, 0)))
    gl = gl + np.where(kospi < -2.5, 0.5, 0)
    gl = gl + np.where(cg > 8, 0.5, np.where(cg < -8, -0.25, 0)).astype(float) * cg.notna()
    gl = gl + np.where(ng > 25, -0.5, 0).astype(float) * ng.notna()
    gl = gl + np.where(jpy < -2, 0.5, np.where(jpy > 2, -0.25, 0)).astype(float) * jpy.notna()
    gl = gl + np.where(curve < 0, 0.25, 0).astype(float) * curve.notna()
    gl = gl + np.where(bdry > 15, 0.25, 0).astype(float) * bdry.notna()
    s["f_global"] = gl.clip(-2, 2)
    s.loc[vix.isna(), "f_global"] = np.nan

    # 匯率資金流 (2007~ 驗證)：美元/台幣 20 日變化與台股同期 20 日 Spearman -0.39 (20 年 18 年為負)，是同步指標；
    # 領先訊號：美元/台幣連漲 ≥5 日 → 之後 20/60 日 -2.67%/-5.57%、勝率 40% (n=106)；台幣 60 日升 >3% → 20 日 +1.45%、勝率 68%；
    # 台幣 20 日貶 >2% 且外資 20 日淨賣 >500 億 → 60 日 +0.98% (基準 +2.78%)。油價：60 日跌 >25% → 60 日 +9.0%、勝率 67%；60 日漲 >25% → 勝率 57%。
    tw20, tw60, streak_, oil60 = d["g_usdtwd_r20"], d["g_usdtwd_r60"], d["g_usdtwd_streak"], d["g_oil_r60"]
    fx = pd.Series(0.0, index=d.index)
    fx = fx + np.where(streak_ >= 5, -1.5, np.where(streak_ >= 4, -0.5, 0)).astype(float) * streak_.notna()
    fx = fx + np.where(tw60 < -3, 0.5, 0).astype(float) * tw60.notna()
    fx = fx + np.where((tw20 > 2) & (d["foreign_20d"] < -500), -0.5, 0).astype(float) * tw20.notna()
    fx = fx + np.where(oil60 < -25, 0.5, np.where(oil60 > 25, -0.25, 0)).astype(float) * oil60.notna()
    s["f_fx_flow"] = fx.clip(-2, 2)
    s.loc[tw20.isna(), "f_fx_flow"] = np.nan

    # 狀態自適應綜合分 (含 tuned 權重倍數)
    comp = pd.Series(np.nan, index=d.index)
    for state in STATE_WEIGHTS:
        w = effective_weights(state)
        mask = d["state"] == state
        if not mask.any():
            continue
        num = pd.Series(0.0, index=d.index)
        den = pd.Series(0.0, index=d.index)
        for k in FACTOR_KEYS:
            col = s[f"f_{k}"]
            num += col.fillna(0) * w[k]
            den += col.notna() * w[k]
        comp[mask] = (num / (2 * den.replace(0, np.nan)) * 100)[mask]
    s["composite"] = comp.round(1)
    s["composite_smooth"] = comp.ewm(span=3, adjust=False).mean().round(1)
    s["composite_chg5"] = (s["composite_smooth"] - s["composite_smooth"].shift(5)).round(1)
    return pd.concat([d, s], axis=1)


# ============================================================ 最新判讀
def _row_factors(r: pd.Series, weights: dict) -> list[Factor]:
    f: list[Factor] = []

    def add(key, value, comment, tags=None):
        sc = r.get(f"f_{key}")
        f.append(Factor(key, NAMES[key], float(sc) if pd.notna(sc) else 0.0, weights[key], value, comment,
                        available=pd.notna(sc), tags=tags or []))

    st = int(r.get("foreign_streak") or 0)
    tags = ["外資連賣"] if st <= -3 else ["外資連買"] if st >= 3 else []
    cm = (f"外資連{'買' if st > 0 else '賣'}{abs(st)}日，" if st else "") + \
         ("大額賣超，籌碼面最大壓力來源" if r["foreign"] < -300 else "大額買超，資金回流" if r["foreign"] > 300 else "買賣力道溫和")
    if r["foreign"] < -300 and r["ret1"] < -1.5:
        cm += "；單日極端賣超伴隨重挫，常見於賣壓高潮 (反轉觀察)"
        tags.append("賣壓高潮")
    add("foreign", f"今 {fmt(r['foreign'])} 億｜5日 {fmt(r['foreign_5d'])}｜20日 {fmt(r['foreign_20d'])}", cm, tags)
    add("trust", f"今 {fmt(r['trust'])} 億｜5日 {fmt(r['trust_5d'])}",
        "投信持續加碼，中小型/ETF 資金有撐" if r["trust_5d"] > 50 else "投信偏賣，作帳/贖回壓力" if r["trust_5d"] < -50 else "投信動作不大")
    add("dealer", f"今 {fmt(r['dealer'])} 億｜5日 {fmt(r['dealer_5d'])}",
        "自營商避險賣壓重 (權證/ETF 對沖)" if r["dealer"] < -150 else "自營商偏多" if r["dealer"] > 100 else "自營商中性")
    oi, pct, cons = r.get("fut_foreign_net_oi"), r.get("fut_foreign_pct"), int(r.get("foreign_consistency") or 0)
    cons_txt = {1: "現貨買+期貨加多 → 外資多方一致", -1: "現貨賣+期貨加空 → 外資空方一致", 0: "現貨與期貨方向不同 → 偏避險/中性"}[cons]
    lvl_txt = ("淨部位在一年最空的 20% 區間" if pd.notna(pct) and pct < 0.2 else "淨部位在一年最多的 20% 區間" if pd.notna(pct) and pct > 0.8 else "淨部位在一年中段")
    add("fut_foreign", f"淨 {fmt(oi, 0)} 口 (一年百分位 {fmt((pct or np.nan) * 100, 0, '%', sign=False)})｜日變 {fmt(r.get('fut_foreign_chg1'), 0)}｜5日變 {fmt(r.get('fut_foreign_chg5'), 0)}",
        f"{lvl_txt}；{cons_txt}", ["外資期貨空單極端"] if pd.notna(pct) and pct < 0.1 else ["外資多空一致偏多"] if cons == 1 else ["外資多空一致偏空"] if cons == -1 else [])
    g5, gs = r.get("gov8_5d"), int(r.get("gov8_streak") or 0)
    tags, cm = [], "八大行庫動作不明顯"
    if gs >= 3 and r["ret5"] < 0:
        cm, tags = f"指數下跌但官股連續買超 {gs} 日 → 護盤/國家隊進場跡象", ["護盤"]
    elif pd.notna(g5) and g5 > 30:
        cm = "官股順勢買進"
    elif gs <= -3 and r["ret5"] > 0:
        cm = f"指數上漲官股連賣 {abs(gs)} 日 → 官股高檔調節"
    elif pd.notna(g5) and g5 < -30:
        cm = "官股減碼"
    add("gov8", f"今 {fmt(r.get('gov8_net'))} 億｜5日 {fmt(g5)}｜20日 {fmt(r.get('gov8_20d'))}｜連{'買' if gs > 0 else '賣'} {abs(gs)} 日", cm, tags)
    m20, r20 = r.get("margin_pct20"), r.get("ret20")
    tags = []
    if pd.isna(m20) or pd.isna(r20):
        cm = "資料不足"
    elif m20 > 2 and r20 < -2:
        cm, tags = "指數下跌融資卻增加 → 散戶逆勢接刀，籌碼最差象限", ["散戶接刀"]
    elif m20 - r20 > 4:
        cm, tags = "融資增速遠超指數 → 散戶追價、籌碼凌亂，拉回風險高", ["融資過熱"]
    elif m20 - r20 > 1.5:
        cm = "融資增加快於指數，籌碼略顯浮動"
    elif m20 < -4 and r20 < -3:
        cm, tags = "融資大減且指數大跌 → 斷頭/去槓桿" + ("，近 5 日已止穩" if r["ret5"] > 0 else "，尚未止穩"), ["融資斷頭清洗"]
    elif m20 - r20 < -1.5 and r20 > 0:
        cm = "指數上漲融資相對減少 → 籌碼沉澱，法人主導的健康上漲"
    elif m20 - r20 < -1.5:
        cm = "融資相對減少，籌碼趨於沉澱"
    else:
        cm = "融資與指數同步，籌碼健康"
    add("margin", f"餘額 {fmt(r.get('margin_amt'), 0, sign=False)} 億｜日變 {fmt(r.get('margin_chg1'))}｜20日 {fmt(m20)}% vs 指數 {fmt(r20)}%", cm, tags)
    mr = r.get("maint_ratio")
    add("maint", f"{fmt(mr, 2, '%', sign=False)}",
        "維持率偏低，接近斷頭區，需留意追繳賣壓" if pd.notna(mr) and mr < 160 else
        "維持率健康" if pd.notna(mr) and mr >= 170 else "維持率尚可" if pd.notna(mr) else "需玩股網資料",
        ["維持率警戒"] if pd.notna(mr) and mr < 160 else [])
    add("short", f"餘額 {fmt(r.get('short_lots'), 0, sign=False)} 張｜5日 {fmt(r.get('short_chg5'), 0)}｜券資比 {fmt(r.get('short_ratio'), 2, '%', sign=False)}",
        "上漲中融券增加 → 軋空動能" if r["ret5"] > 0 and (r.get("short_chg5") or 0) > 0 else
        "下跌中融券增加 → 空方追空" if r["ret5"] < 0 and (r.get("short_chg5") or 0) > 0 else "融券回補/變化不大")
    add("sbl", f"5日 {fmt(r.get('sbl_chg5'), 0)} 張｜今 {fmt(r.get('sbl_chg'), 0)}",
        "借券賣出餘額上升 → 法人放空增加" if (r.get("sbl_chg5") or 0) > 0 else "借券賣出回補 → 空方壓力減輕")
    p = r.get("pcr_oi")
    add("pcr", f"OI {fmt(p, 2, '%', sign=False)}",
        "P/C 比偏高 → 市場偏空避險，反向偏多" if pd.notna(p) and p >= 110 else
        "P/C 比偏低 → 過度樂觀/偏空訊號" if pd.notna(p) and p <= 85 else "P/C 比中性",
        ["PCR極端"] if pd.notna(p) and (p >= 120 or p <= 75) else [])
    add("large", f"前十大特定法人淨 {fmt(r.get('top10_spec_net'), 0)} 口 ({fmt((r.get('large_ratio') or np.nan) * 100, 1, '%')} 市場OI)",
        "大額法人淨多" if (r.get("top10_spec_net") or 0) > 0 else "大額法人淨空")
    fv = r.get("f_volume")
    add("volume", f"漲跌 {fmt(r.get('ret1'), 2, '%')}｜成交 {fmt(r.get('amount'), 0, ' 億', sign=False)} ({fmt(r.get('vol_ratio'), 2, 'x 20日均', sign=False)})｜5日量能 {fmt(r.get('amount_5d_ratio'), 2, 'x', sign=False)}",
        {1.5: "價漲量增，健康", 0.5: "量縮整理／價跌量縮，賣壓減輕", -1.5: "價跌量增，賣壓沉重", -2: "爆量長黑，出貨訊號"}.get(fv, "量價中性"),
        ["爆量長黑"] if fv == -2 else [])
    ft = r.get("f_trend")
    slope = r.get("ma20_slope")
    add("trend", f"收 {fmt(r['close'], 0, sign=False)}｜MA20 {fmt(r.get('ma20'), 0, sign=False)} (斜率 {fmt(slope, 2, '%')})｜MA60 {fmt(r.get('ma60'), 0, sign=False)}｜乖離20 {fmt(r.get('bias20'), 2, '%')}",
        ("多頭排列，站上月季線" if (ft or 0) >= 1.5 else "站上月線" if (ft or 0) >= 1 else "跌破月線，季線之上整理" if (ft or 0) >= -1 else "月季線之下，空方格局")
        + ("，創 20 日新高" if r["close"] >= r.get("hi20", np.inf) else "，創 20 日新低" if r["close"] <= r.get("lo20", -np.inf) else "")
        + "（趨勢在 10~20 日視野歷史上偏反指標，權重已降低）",
        ["乖離過大"] if (r.get("bias20") or 0) > 6 else [])
    b20 = r.get("bias20")
    add("reversion", f"月線乖離 {fmt(b20, 2, '%')}",
        "負乖離超過 6%，歷史上 10 日反彈機率 75%、平均 +3.5%" if pd.notna(b20) and b20 < -6 else
        "負乖離中等，有反彈空間" if pd.notna(b20) and b20 < -3 else "無超跌訊號",
        ["超跌"] if pd.notna(b20) and b20 < -6 else [])
    vix, sox20, kospi, sox1 = r.get("g_vix_level"), r.get("g_sox_r20"), r.get("g_kospi_r1"), r.get("g_sox_r1")
    gtxt = []
    if pd.notna(vix):
        gtxt.append("VIX 恐慌區 (>30)，歷史上之後台股偏強" if vix >= 30 else "VIX 偏高，風險溢酬有利後市" if vix >= 22 else
                    "VIX 極低 (<13)，市場自滿，歷史上之後 5 日顯著偏弱" if vix < 13 else "VIX 正常")
    if pd.notna(sox20):
        gtxt.append("費半 20 日強勢 (>15%)，動能外溢台股" if sox20 > 15 else "費半 20 日弱勢 (<-10%)" if sox20 < -10 else "費半 20 日中性")
    if pd.notna(kospi) and kospi < -2.5:
        gtxt.append("KOSPI 前日重挫，歷史上台股 5 日反彈")
    if pd.notna(sox1):
        gtxt.append(f"前晚費半 {sox1:+.2f}% 主要反映在今日跳空")
    cg, ng, jpy, curve = r.get("g_copper_gold_r20"), r.get("g_natgas_r20"), r.get("g_usdjpy_r5"), r.get("g_curve_10y_3m")
    if pd.notna(cg) and abs(cg) > 8:
        gtxt.append(f"銅金比 20 日 {cg:+.1f}% → {'景氣偏強，四大市場歷史皆偏多' if cg > 0 else '避險升溫'}")
    if pd.notna(ng) and ng > 25:
        gtxt.append(f"天然氣 20 日 {ng:+.0f}%，能源通膨壓力，歷史上之後偏弱")
    if pd.notna(jpy) and jpy < -2:
        gtxt.append(f"日圓 5 日急升 {jpy:+.1f}% (套利平倉衝擊)，歷史上衝擊後 5 日反彈")
    if pd.notna(curve) and curve < 0:
        gtxt.append("美債殖利率曲線倒掛期間，歷史上台股 20 日反而偏強")
    add("global", f"VIX {fmt(vix, 1, sign=False)}｜費半 20 日 {fmt(sox20, 1, '%')}｜前晚費半 {fmt(sox1, 2, '%')}｜KOSPI 前日 {fmt(kospi, 2, '%')}｜銅金比 20 日 {fmt(cg, 1, '%')}｜天然氣 20 日 {fmt(ng, 0, '%')}｜美元/日圓 5 日 {fmt(jpy, 1, '%')}",
        "；".join(gtxt) or "無國際資料", ["VIX恐慌"] if pd.notna(vix) and vix >= 30 else ["VIX自滿"] if pd.notna(vix) and vix < 13 else [])
    tw20, tw60, stk, oil60 = r.get("g_usdtwd_r20"), r.get("g_usdtwd_r60"), r.get("g_usdtwd_streak"), r.get("g_oil_r60")
    ftxt, ftags = [], []
    if pd.notna(stk) and stk >= 5:
        ftxt.append(f"美元/台幣連漲 {int(stk)} 日 (台幣持續貶值、資金外流)，歷史上之後 20 日 -2.7%、勝率 40%")
        ftags.append("台幣連貶")
    elif pd.notna(tw20) and tw20 > 2:
        ftxt.append(f"台幣 20 日貶 {tw20:.1f}%，同期外資 20 日 {fmt(r.get('foreign_20d'), 0)} 億" + ("，貶值與外資賣超同步 → 資金外流" if (r.get("foreign_20d") or 0) < -500 else "，但外資未大賣，貶值已反映"))
    elif pd.notna(tw60) and tw60 < -3:
        ftxt.append(f"台幣 60 日升值 {abs(tw60):.1f}%，資金流入，歷史上 20 日 +1.45%、勝率 68%")
    if pd.notna(oil60):
        ftxt.append(f"油價 60 日 {oil60:+.0f}%" + ("，大跌後 60 日歷史 +9%" if oil60 < -25 else "，大漲後 60 日偏弱 (通膨壓力)" if oil60 > 25 else "，中性 (油價與台股同期為正相關，非負相關)"))
    add("fx_flow", f"美元/台幣 20 日 {fmt(tw20, 2, '%')}｜60 日 {fmt(tw60, 2, '%')}｜連漲 {fmt(stk, 0, ' 日', sign=False)}｜油價 60 日 {fmt(oil60, 0, '%')}",
        "；".join(ftxt) or "匯率與油價無明顯訊號", ftags)
    return f


def _turning(scored: pd.DataFrame) -> str | None:
    sm = scored["composite_smooth"].dropna().tail(4)
    if len(sm) < 4:
        return None
    last, prev = sm.iloc[-1], sm.iloc[:-1]
    if last > -10 and prev.min() <= -15:
        return "籌碼由空轉多中 (平滑分自 -15 以下回升)"
    if last < 10 and prev.max() >= 15:
        return "籌碼由多轉空中 (平滑分自 +15 以上回落)"
    return None


def assess(scored: pd.DataFrame) -> dict:
    """最新一日判讀。"""
    r = scored.iloc[-1]
    state = r.get("state", "盤整")
    weights = effective_weights(state)
    factors = _row_factors(r, weights)
    comp = float(r["composite"]) if pd.notna(r["composite"]) else composite(factors)
    smooth = float(r["composite_smooth"]) if pd.notna(r.get("composite_smooth")) else comp
    mom = float(r["composite_chg5"]) if pd.notna(r.get("composite_chg5")) else 0.0
    reg = regime(smooth)

    # 信心度：因子方向一致性 + 資料完整度
    avail = [f for f in factors if f.available]
    sign = 1 if smooth > 0 else -1
    agree = [f for f in avail if abs(f.score) >= 0.25 and np.sign(f.score) == sign]
    disagree = [f for f in avail if abs(f.score) >= 0.25 and np.sign(f.score) == -sign]
    agree_ratio = len(agree) / max(1, len(agree) + len(disagree))
    coverage = len(avail) / len(factors)
    confidence = "高" if (agree_ratio >= 0.65 and coverage >= 0.75 and abs(smooth) >= 10) else "中" if (agree_ratio >= 0.5 and coverage >= 0.6) else "低"

    contrib = sorted(avail, key=lambda f: f.contribution)
    reasons_neg = [f for f in contrib if f.contribution < -0.4][:3]
    reasons_pos = [f for f in reversed(contrib) if f.contribution > 0.4][:3]

    def ok(cond):
        return bool(cond) if cond == cond else None

    checklist = [
        ("指數趨勢在多方 (站上月線)", ok((r.get("f_trend") or 0) >= 1)),
        ("外資現貨未持續大賣 (5日 > -300 億)", ok(r["foreign_5d"] > -300)),
        ("外資期貨未同步加空 (現貨期貨非空方一致)", ok(int(r.get("foreign_consistency") or 0) != -1)),
        ("融資未過熱/接刀 (融資象限分 ≥ -1)", ok((r.get("f_margin") if pd.notna(r.get("f_margin")) else 0) >= -1)),
        ("融資維持率安全 (≥ 165%)", ok(r.get("maint_ratio") >= 165) if pd.notna(r.get("maint_ratio")) else None),
        ("量價配合 (非價跌量增/爆量長黑)", ok((r.get("f_volume") if pd.notna(r.get("f_volume")) else 0) >= 0)),
        ("八大行庫未連續調節 (非連賣 3 日以上且指數漲)", ok(not ((r.get("gov8_streak") or 0) <= -3 and r["ret5"] > 0))),
        ("P/C 比非極端偏空 (OI > 85%)", ok(r.get("pcr_oi") > 85) if pd.notna(r.get("pcr_oi")) else None),
        ("籌碼動能未惡化 (平滑分 5 日變化 > -15)", ok(mom > -15)),
    ]
    passed = sum(1 for _, v in checklist if v)
    total = sum(1 for _, v in checklist if v is not None)

    bottom = []
    if (r.get("gov8_streak") or 0) >= 3 and r["ret5"] < 0:
        bottom.append("八大行庫連續逆勢護盤")
    if pd.notna(r.get("margin_pct20")) and r["margin_pct20"] < -5:
        bottom.append("融資 20 日大減 (去槓桿)")
    if pd.notna(r.get("maint_ratio")) and r["maint_ratio"] < 160:
        bottom.append("維持率進入警戒區 (斷頭潮後常見底部)")
    if pd.notna(r.get("pcr_oi")) and r["pcr_oi"] >= 115:
        bottom.append("P/C 比偏高 (市場過度避險)")
    if pd.notna(r.get("bias20")) and r["bias20"] < -6:
        bottom.append("月線負乖離過大 (超跌)")
    if (r.get("foreign_streak") or 0) <= -5:
        bottom.append("外資連賣 5 日以上 (賣壓可能接近尾聲)")
    if pd.notna(r.get("margin_pct20")) and r["margin_pct20"] < -4 and r["ret20"] < -3:
        bottom.append("融資斷頭清洗 (歷史 10 日 +1.5%、勝率 66%)")
    if any("賣壓高潮" in f.tags for f in factors):
        bottom.append("外資單日極端賣超 + 重挫 (賣壓高潮)")

    top_risk = []
    if any("融資過熱" in f.tags or "散戶接刀" in f.tags for f in factors):
        top_risk.append("融資追價過熱／散戶接刀")
    if pd.notna(r.get("bias20")) and r["bias20"] > 8:
        top_risk.append("月線正乖離極大 (>8%，歷史上 >6% 仍多續漲，僅提示)")
    if (r.get("gov8_streak") or 0) <= -3 and r["ret5"] > 0:
        top_risk.append("官股連續高檔調節")
    if pd.notna(r.get("pcr_oi")) and r["pcr_oi"] <= 80:
        top_risk.append("P/C 比過低 (過度樂觀)")
    if r.get("f_volume") == -2:
        top_risk.append("爆量長黑")
    if int(r.get("foreign_consistency") or 0) == -1:
        top_risk.append("外資現貨期貨同步偏空")
    if pd.notna(r.get("fut_foreign_pct")) and r["fut_foreign_pct"] > 0.9 and pd.notna(r.get("bias20")) and r["bias20"] > 4:
        top_risk.append("外資期貨多單極端且乖離大 (追高風險)")

    turning = _turning(scored)
    stabilizing = r["ret1"] > 0 and mom > 5
    if state == "多頭":
        if smooth >= 20 and passed >= max(5, total - 2):
            action, detail = "可進場（多頭順勢）", "多頭結構 + 籌碼偏多且訊號一致，順勢做多；跌破月線或外資轉連賣則減碼。"
        elif smooth >= 0 and passed >= 5 and (r.get("bias20") or 0) < 2 and r["foreign_5d"] > -300:
            action, detail = "多頭拉回，可分批布局", "多頭結構未破、乖離收斂，籌碼中性偏多，逢回分批建立部位。"
        elif smooth <= -20 or len(top_risk) >= 2:
            action, detail = "多頭但籌碼轉弱，減碼觀望", "趨勢仍在但籌碼面明顯惡化或高檔風險累積，先降低部位等待籌碼修復。"
        else:
            action, detail = "多頭震盪，持股續抱、不追高", "結構偏多但籌碼訊號不一致，既有部位續抱，新資金等拉回。"
    elif state == "空頭":
        if smooth >= 25 and r["close"] > r["ma20"] and passed >= 6:
            action, detail = "空頭反轉確認，可試單進場", "空頭結構下籌碼明顯轉多且站回月線，屬反轉初期，可控部位進場。"
        elif len(bottom) >= 3 and (stabilizing or turning):
            action, detail = "空頭末端，小部位分批試單", "多項底部訊號集中且籌碼動能回升，僅適合小額試單、嚴設停損。"
        else:
            action, detail = "空頭格局，不宜進場", "趨勢與籌碼皆偏空，等待底部訊號累積與籌碼轉折。"
    else:
        if smooth >= 20 and passed >= 6:
            action, detail = "盤整偏多，可小量進場", "區間整理但籌碼偏多，小量參與，突破季線再加碼。"
        elif smooth <= -20:
            action, detail = "盤整偏空，觀望／減碼", "籌碼偏空，區間下緣不破前不進場。"
        elif len(bottom) >= 3:
            action, detail = "底部訊號浮現，分批試單（小部位）", "逆勢訊號集中，可分批布局但需嚴設停損。"
        else:
            action, detail = "觀望，等待訊號一致", "多空因子互相抵銷，沒有明確優勢，保留現金等待。"
    if confidence == "低" and action.startswith(("可", "多頭拉回")):
        detail += "（信心度低：因子方向分歧或資料不全，部位再減半）"

    level = "70–100%" if smooth >= 40 else "50–70%" if smooth >= 15 else "30–50%" if smooth > -15 else "10–30%" if smooth > -40 else "0–10%"
    if state == "空頭":
        level = {"70–100%": "50–70%", "50–70%": "30–50%", "30–50%": "20–30%"}.get(level, level)
    return {
        "date": str(r["date"]), "close": float(r["close"]), "ret1": float(r["ret1"]) if pd.notna(r["ret1"]) else None,
        "ret20": float(r["ret20"]) if pd.notna(r.get("ret20")) else None,
        "composite": comp, "composite_smooth": smooth, "momentum": mom, "regime": reg, "state": state,
        "confidence": confidence, "agree_ratio": round(agree_ratio, 2), "coverage": round(coverage, 2),
        "turning": turning, "action": action, "detail": detail, "position": level,
        "reasons_pos": reasons_pos, "reasons_neg": reasons_neg,
        "factors": factors, "checklist": checklist, "passed": passed, "total": total,
        "bottom_signals": bottom, "top_risks": top_risk,
    }


# ============================================================ 盤中即時提示
def intraday_hint(scored: pd.DataFrame, quote: dict | None) -> dict | None:
    """用即時指數推估「若以現價收盤」的趨勢/量價變化。籌碼資料盤中仍是前一交易日。"""
    if not quote or quote.get("last") is None:
        return None
    r = scored.iloc[-1]
    last = float(quote["last"])
    qdate = str(quote.get("date", ""))
    same_day = qdate[:4] + "-" + qdate[4:6] + "-" + qdate[6:8] == str(r["date"]) if len(qdate) == 8 else False
    closes = scored["close"].astype(float)
    if same_day:
        ma5 = r["ma5"]
        ma20 = r["ma20"]
        prev_close = closes.iloc[-2] if len(closes) > 1 else last
    else:
        ma5 = (closes.tail(4).sum() + last) / 5
        ma20 = (closes.tail(19).sum() + last) / 20
        prev_close = closes.iloc[-1]
    ret = (last / prev_close - 1) * 100
    bias = (last / ma20 - 1) * 100
    msgs = []
    if same_day:
        msgs.append("已收盤，籌碼資料與指數同日。")
    else:
        msgs.append("盤中：籌碼資料為前一交易日，以下為以現價推估。")
    if last > ma20 and r["close"] <= r["ma20"]:
        msgs.append("盤中站回月線 (趨勢因子將轉正)。")
    elif last < ma20 and r["close"] >= r["ma20"]:
        msgs.append("盤中跌破月線 (趨勢因子將轉負，留意收盤是否收回)。")
    elif last > ma20:
        msgs.append("維持在月線之上。")
    else:
        msgs.append("仍在月線之下。")
    if last > ma5:
        msgs.append("站上 5 日線，短線偏強。")
    if ret <= -2:
        msgs.append("單日重挫 2% 以上：若外資盤後仍大賣，屬賣壓高潮觀察點；不建議盤中接刀。")
    elif ret >= 2:
        msgs.append("單日大漲 2% 以上：留意是否伴隨量增與外資回補，否則提防一日行情。")
    if bias > 6:
        msgs.append("月線乖離過大，不宜追價。")
    elif bias < -6:
        msgs.append("月線負乖離過大，短線超跌。")
    return {"last": last, "ret": round(ret, 2), "ma5": round(ma5, 2), "ma20": round(ma20, 2), "bias20": round(bias, 2),
            "same_day": same_day, "time": quote.get("time"), "messages": msgs}


# ============================================================ 歷史驗證
def evaluate(scored: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> dict:
    """綜合分 (平滑) vs 未來報酬：相關係數、各分區平均報酬與勝率、對照全體基準。"""
    d = scored.dropna(subset=["composite_smooth"]).copy()
    out = {"corr": {}, "buckets": {}, "baseline": {}}
    if len(d) < 40:
        return out
    bins = [-101, -40, -15, 15, 40, 101]
    labels = ["空方(<-40)", "偏空(-40~-15)", "中性(-15~15)", "偏多(15~40)", "強勢多方(>40)"]
    d["bucket"] = pd.cut(d["composite_smooth"], bins=bins, labels=labels)
    for h in horizons:
        fwd = (d["close"].shift(-h) / d["close"] - 1) * 100
        d[f"fwd{h}"] = fwd
        valid = d.dropna(subset=[f"fwd{h}"])
        out["corr"][h] = round(float(valid["composite_smooth"].rank().corr(valid[f"fwd{h}"].rank())), 3) if len(valid) > 10 else None
        out["baseline"][h] = {"平均報酬%": round(float(valid[f"fwd{h}"].mean()), 2), "勝率%": round(float((valid[f"fwd{h}"] > 0).mean() * 100), 1)}
        g = valid.groupby("bucket", observed=True)[f"fwd{h}"]
        out["buckets"][h] = pd.DataFrame({"樣本數": g.size(), "平均報酬%": g.mean().round(2),
                                          "勝率%": (g.apply(lambda x: (x > 0).mean() * 100)).round(1)})
    out["frame"] = d
    return out


def suggest_weights(scored: pd.DataFrame, horizon: int = 10, train_ratio: float = 0.7) -> dict:
    """資料驅動的權重參考：用前 train_ratio 的資料做 ridge 迴歸 (因子分 → 未來報酬)，
    在後段樣本外比較「預設權重」與「擬合權重」的排序相關。僅供參考，樣本少易過擬合。"""
    d = scored.copy()
    d["fwd"] = (d["close"].shift(-horizon) / d["close"] - 1) * 100
    cols = [f"f_{k}" for k in FACTOR_KEYS]
    d = d.dropna(subset=["fwd", "composite"])
    if len(d) < 80:
        return {}
    X = d[cols].fillna(0).to_numpy(dtype=float)
    y = d["fwd"].to_numpy(dtype=float)
    n = int(len(d) * train_ratio)
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    lam = 5.0
    beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(X.shape[1]), Xtr.T @ (ytr - ytr.mean()))
    fitted_te = Xte @ beta
    default_te = d["composite"].to_numpy()[n:]

    def rank_corr(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    coef = pd.DataFrame({"因子": [NAMES[k] for k in FACTOR_KEYS], "預設權重": [WEIGHTS[k] for k in FACTOR_KEYS],
                         "擬合係數": np.round(beta, 3)})
    coef["方向"] = np.where(coef["擬合係數"] > 0.05, "同向 (加分有效)", np.where(coef["擬合係數"] < -0.05, "反向 (歷史上反指標)", "無明顯關係"))
    return {"horizon": horizon, "n_train": n, "n_test": len(d) - n, "coef": coef,
            "oos_corr_default": round(rank_corr(default_te, yte), 3), "oos_corr_fitted": round(rank_corr(fitted_te, yte), 3),
            "is_corr_default": round(rank_corr(d["composite"].to_numpy()[:n], ytr), 3), "is_corr_fitted": round(rank_corr(Xtr @ beta, ytr), 3)}


def run(use_wantgoo: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    df, meta = build_frame(use_wantgoo)
    scored = score_frame(df)
    return scored, assess(scored), meta
