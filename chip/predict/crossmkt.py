"""跨市場關係 × 主力籌碼 深化研究 (2026-09-22)：每次發布用 2010~ 全歷史重算 (數秒)，輸出「今日讀數 + 有效條件 + 領先落後表」到 forecast.json `deep`。

研究結論 (逐年一致性為準)：
- 各國市場：對「隔日」最有用的是同日已知的亞股收盤 —— 恆生同日 (IC 0.09、15/17 年) > KOSPI 同日 (0.07、14/17)；前晚美股只對隔日有微弱訊息 (S&P/費半 IC 0.04)，
  前一日亞股、上證完全無用。對 5~20 日最強的是 VIX 水準 (fwd5 IC 0.10、16/17 年；fwd20 0.17、15/17)：VIX>30 之後 20 日 +2.3% vs 基準 +1.0%，7/7 年。
  背離補漲：台股 5 日落後 KOSPI >3% → 之後 5 日上漲率 64% vs 58%、+0.57pt，13 年 77% 一致；領先 >3% 沒有對稱效果。KOSPI 同日 >2% → 隔日漲 65% (8 年 88%)、<-2% → 45%。
  同日連動：KOSPI 相關 0.66 (逐年 0.47~0.79)、日經 0.56、恆生 0.51、前晚 S&P 0.42。「全球綜合 z」不比單一指標好 (IC 0.04)。
- 主力籌碼 (大盤)：外資期貨 5 日增減 z 對 5~20 日 IC 0.05~0.06；外資現貨 20 日 z 是反指標 (fwd20 IC −0.07)；融資本身無預測力；
  「聰明錢−散戶差」(外資現貨 5 日 z + 外資期貨 5 日 z 之半 − 融資 5 日 z) 對 5/10/20 日 IC 0.044/0.046/0.049，9/9、8/9、8/9 年為正；
  聰明錢 z<-1 且散戶加碼 → 5 日 +0.05% vs +0.24% (9/9 年劣於基準)；外資現貨賣但期貨增倉 (背離) → 5 日 +1.28% (n=70，4 年 75%)。
- 大戶 (追蹤股)：TDCC 週資料目前只有最新一週的快照 (每週累積中)，尚無法統計；顯示最新大戶比與 4 週變化即可。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)
YEARS_MIN = 6


def _z(s: pd.Series, w: int = 250) -> pd.Series:
    s = s.astype(float)
    return (s - s.rolling(w, min_periods=60).mean()) / s.rolling(w, min_periods=60).std()


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """供 short_term 使用的跨市場/主力特徵 (只用 D 日收盤前已知資訊)。"""
    c = d["close"].astype(float)
    r5 = (c / c.shift(5) - 1) * 100
    d["rel_kospi5"] = r5 - pd.to_numeric(d.get("g_kospi_r5"), errors="coerce")
    d["rel_sp5"] = r5 - pd.to_numeric(d.get("g_sp500_r5"), errors="coerce")
    d["smart_spread"] = (pd.to_numeric(d.get("foreign_z5"), errors="coerce") + pd.to_numeric(d.get("fut_foreign_chg5_z"), errors="coerce")) / 2 - _z(pd.to_numeric(d.get("margin_chg5_pct"), errors="coerce"))
    d["hsi_r5"] = pd.to_numeric(d.get("g_hsi_r5"), errors="coerce")
    # 聰明錢 v2 (2026-09-22 優化研究 smart_study.py)：外資期貨 5 日增減 z + 融資 5 日反向 z + 投信 20 日 z − 外資現貨 20 日 z (反指標) + 選擇權外資買賣權差 z
    #   逐年 IC 5/10/20 日 0.080/0.088/0.096 (8/9 年為正)，前 20% 之後上漲 64/67/68% vs 後 20% 58/60/64%；現行 v1 只有 0.028/0.037/0.044。
    #   單一成分：TXO 外資買賣權差 0.08~0.10、外資期貨 10 日增減 0.06~0.10、小台散戶反向 0.05~0.07；外資現貨 5/10/20 日皆為反指標 (IC −0.03~−0.06)；Ridge 走動式學權重反而 IC≈0 (過擬合)。
    num = lambda k: pd.to_numeric(d.get(k), errors="coerce") if k in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    fut_oi = num("fut_foreign_net_oi")
    d["fut_chg10_z"] = _z(fut_oi.diff(10))
    tr20 = num("trust").rolling(20, min_periods=10).sum() if "trust" in d else num("trust_z20")
    d["smart_core"] = num("fut_foreign_chg5_z") - _z(num("margin_chg5_pct")) + (_z(tr20) if "trust" in d else num("trust_z20")) - num("foreign_z20")
    d["smart2"] = d["smart_core"] + num("txo_f_cp_diff_z").fillna(0.0)
    d["mtx_retail_inv"] = -_z(num("mtx_retail_ratio"))
    return d


def _ic_years(x: pd.Series, y: pd.Series, year: pd.Series) -> dict | None:
    dd = pd.concat([x, y, year], axis=1).dropna(); dd.columns = ["x", "y", "year"]
    if len(dd) < 200:
        return None
    yr = dd.groupby("year").apply(lambda g: g["x"].rank().corr(g["y"].rank()) if len(g) >= 60 else np.nan, include_groups=False).dropna()
    return {"ic": round(float(dd["x"].rank().corr(dd["y"].rank())), 3), "pos_years": f"{int((yr > 0).sum())}/{len(yr)}", "n": int(len(dd))}


def _cond(m: pd.DataFrame, mask: pd.Series, col: str) -> dict | None:
    d = m[mask.fillna(False).astype(bool)]
    x, base = d[col].dropna(), m[col].dropna()
    if len(x) < 30:
        return None
    sign = np.sign(x.mean() - base.mean())
    yr = []
    for y, g in d.groupby("year"):
        xy = g[col].dropna(); by = m.loc[m["year"] == y, col].dropna()
        if len(xy) >= 6 and len(by):
            yr.append(np.sign(xy.mean() - by.mean()) == sign)
    return {"n": int(len(x)), "up": round(float((x > 0).mean()), 3), "base_up": round(float((base > 0).mean()), 3), "mean": round(float(x.mean()), 2), "base_mean": round(float(base.mean()), 2),
            "excess": round(float(x.mean() - base.mean()), 2), "consist": round(float(np.mean(yr)), 2) if yr else None, "years": len(yr),
            "valid": bool(yr and len(yr) >= YEARS_MIN and float(np.mean(yr)) >= 0.7 and abs(x.mean() - base.mean()) >= 0.1), "direction": "偏多" if sign > 0 else "偏空"}


# 聰明錢 v2 × 5/10/20 日模型叫牌 共識 (OOS 2019~2026，市場模型走動式預測 + 滾動門檻；smart_study / smart_tier2)：
#   5 日：偏多叫牌 62.3% (基準 60.6%) → +聰明錢>0.5 65.3% (6/7 年贏基準)；偏多但聰明錢<-0.5 58.6% (輸基準)；偏空叫牌 41.1% (基準 39.4%) → +聰明錢<-0.5 44.5% (6/8)；中性區但聰明錢>1 → 64.7%
#   10 日：偏多 64.9% (63.7%) → +聰明錢>0.5 68.1%、>1 70.7%；偏多但聰明錢<-0.5 60.3%；偏空 37.3% (36.3%) → +聰明錢<-0.5 38.6%；中性區但聰明錢>1 → 70.4% (6/8)
#   20 日：偏多叫牌 62.1% 反而低於基準 66.3%；僅聰明錢>1 → 71.6% (6/8 年)；偏空 35.5% (33.7%) → +聰明錢<-0.5 40.9% (5/7)；偏空但聰明錢>0.5 只有 25.6%
SMART_TIERS = {
    5: {"up_confirm": (0.653, 0.606, "6/7"), "up_strong": (0.668, 0.606, "6/7"), "up_conflict": (0.586, 0.606, "3/7"), "dn_confirm": (0.445, 0.394, "6/8"), "dn_conflict": (0.368, 0.394, "3/8"), "neutral_smart": (0.647, 0.606, "5/8"), "up_plain": (0.623, 0.606, "6/8"), "dn_plain": (0.411, 0.394, "4/8")},
    10: {"up_confirm": (0.681, 0.637, "5/8"), "up_strong": (0.707, 0.637, "5/8"), "up_conflict": (0.603, 0.637, "5/7"), "dn_confirm": (0.386, 0.363, "5/8"), "dn_conflict": (0.396, 0.363, "4/7"), "neutral_smart": (0.704, 0.637, "6/8"), "up_plain": (0.649, 0.637, "6/8"), "dn_plain": (0.373, 0.363, "4/8")},
    20: {"up_confirm": (0.626, 0.663, "2/6"), "up_strong": (0.652, 0.663, "3/6"), "up_conflict": (0.598, 0.663, "4/6"), "dn_confirm": (0.409, 0.337, "5/7"), "dn_conflict": (0.256, 0.337, "2/5"), "neutral_smart": (0.744, 0.663, "5/8"), "up_plain": (0.621, 0.663, "5/8"), "dn_plain": (0.355, 0.337, "3/7"), "smart_only_up": (0.716, 0.663, "6/8")},
}


def smart_overlay(h: int, call: str | None, smart2: float | None) -> dict:
    """依聰明錢 v2 調整 5/10/20 日叫牌：回傳 {call_smart, smart_tier, smart_hit, smart_ref, smart_years, smart_note}。
    規則：偏多需聰明錢不反向 (>0.5 為確認、<-0.5 降中性)；偏空需聰明錢確認 (<-0.5) 否則降中性；模型中性但聰明錢 >1 → 偏多 (聰明錢)；
    20 日模型偏多本身輸基準 → 只信聰明錢 >1。"""
    T = SMART_TIERS.get(int(h))
    if not T or smart2 is None or not np.isfinite(smart2):
        return {}
    call = call or "中性"
    def out(cs, tier, key, note):
        hit, ref, yrs = T[key]
        return {"call_smart": cs, "smart_tier": tier, "smart_hit": hit, "smart_ref": ref, "smart_years": yrs, "smart_note": f"{note}；OOS 命中 {hit:.0%} (基準 {ref:.0%}，{yrs} 年贏基準)"}
    if h == 20:
        if smart2 > 1:
            return out("偏多", "高", "smart_only_up", "聰明錢 v2 >1 (20 日只信聰明錢，模型偏多本身輸基準)")
        if call == "偏空" and smart2 < -0.5:
            return out("偏空", "中", "dn_confirm", "模型偏空 + 聰明錢確認")
        if call == "偏空":
            return out("中性", "低", "dn_conflict", "模型偏空但聰明錢未確認 → 中性")
        if call == "偏多":
            return out("中性", "低", "up_plain", "20 日模型偏多歷史輸基準 → 中性")
        return {"call_smart": "中性", "smart_tier": "", "smart_note": "聰明錢中性"}
    if call == "偏多":
        if smart2 > 1: return out("偏多", "高", "up_strong", "模型偏多 + 聰明錢 v2 >1 強確認")
        if smart2 > 0.5: return out("偏多", "高", "up_confirm", "模型偏多 + 聰明錢確認")
        if smart2 < -0.5: return out("中性", "低", "up_conflict", "模型偏多但聰明錢反向 (歷史輸基準) → 中性")
        return out("偏多", "中", "up_plain", "模型偏多，聰明錢中性")
    if call == "偏空":
        if smart2 < -0.5: return out("偏空", "中", "dn_confirm", "模型偏空 + 聰明錢確認")
        if smart2 > 0.5: return out("中性", "低", "dn_conflict", "模型偏空但聰明錢偏多 (歷史輸基準) → 中性")
        return out("偏空", "低", "dn_plain", "模型偏空，聰明錢中性 (優勢很小)")
    if smart2 > 1:
        return out("偏多", "中", "neutral_smart", "模型中性但聰明錢 v2 >1")
    return {"call_smart": "中性", "smart_tier": "", "smart_note": "模型中性、聰明錢中性"}


def build(matrix: pd.DataFrame) -> dict:
    m = matrix.copy()
    m = add_features(m)
    c = m["close"].astype(float)
    for h in (1, 5, 20):
        m[f"fwd{h}"] = (c.shift(-h) / c - 1) * 100
    m["year"] = m["date"].astype(str).str[:4].astype(int)
    g = lambda k: pd.to_numeric(m.get(k), errors="coerce")  # noqa: E731
    lead = {}
    for name, col in {"恆生 (同日)": "hsi_r0", "KOSPI (同日)": "kospi_r0", "日經 (同日)": "nikkei_r0", "S&P500 (前晚)": "g_sp500_r1", "費半 (前晚)": "g_sox_r1", "台積 ADR (前晚)": "g_tsm_adr_r1",
                      "VIX 水準": "g_vix_level", "美債 10 年 5 日": "g_us10y_r5", "外資期貨 5 日增減 z": "fut_foreign_chg5_z", "外資現貨 20 日 z": "foreign_z20", "聰明錢−散戶差": "smart_spread", "聰明錢 v2": "smart2", "外資期貨 10 日增減 z": "fut_chg10_z", "選擇權外資買賣權差 z": "txo_f_cp_diff_z", "小台散戶反向": "mtx_retail_inv", "投信 20 日 z": "trust_z20"}.items():
        if col in m:
            lead[name] = {f"fwd{h}": _ic_years(g(col), m[f"fwd{h}"], m["year"]) for h in (1, 5, 20)}
    conds = {
        "台股 5 日落後 KOSPI >3% (補漲)": (g("rel_kospi5") < -3, "fwd5", "跨市場"), "KOSPI 同日 >2%": (g("kospi_r0") > 2, "fwd1", "跨市場"), "KOSPI 同日 <-2%": (g("kospi_r0") < -2, "fwd1", "跨市場"),
        "恆生同日 >2%": (g("hsi_r0") > 2, "fwd1", "跨市場"), "恆生同日 <-2%": (g("hsi_r0") < -2, "fwd1", "跨市場"), "VIX >30 (20 日)": (g("g_vix_level") > 30, "fwd20", "跨市場"),
        "VIX 5 日升 >30% (5 日)": (g("g_vix_r5") > 30, "fwd5", "跨市場"), "費半前晚 >3%": (g("g_sox_r1") > 3, "fwd1", "跨市場"), "EWT 前晚 >2%": (g("g_ewt_r1") > 2, "fwd1", "跨市場"), "S&P 前晚 <-2%": (g("g_sp500_r1") < -2, "fwd1", "跨市場"),
        "聰明錢 z<-1 且散戶加碼 (5 日)": (g("smart_spread") < -1, "fwd5", "主力"), "聰明錢 z>1 且散戶減碼 (5 日)": (g("smart_spread") > 1, "fwd5", "主力"),
        "聰明錢 v2 >1.5 (5 日)": (g("smart2") > 1.5, "fwd5", "主力"), "聰明錢 v2 >1.5 (20 日)": (g("smart2") > 1.5, "fwd20", "主力"), "聰明錢 v2 <-1.5 (5 日)": (g("smart2") < -1.5, "fwd5", "主力"), "聰明錢 v2 <-1.5 (20 日)": (g("smart2") < -1.5, "fwd20", "主力"),
        "選擇權外資買賣權差 z>1 (10 日)": (g("txo_f_cp_diff_z") > 1, "fwd20", "主力"), "選擇權外資買賣權差 z<-1 (10 日)": (g("txo_f_cp_diff_z") < -1, "fwd20", "主力"),
        "外資現貨賣、期貨增倉 (背離,5 日)": ((g("foreign_z5") < -1) & (g("fut_foreign_chg5_z") > 1), "fwd5", "主力"), "外資現貨買、期貨減倉 (背離,5 日)": ((g("foreign_z5") > 1) & (g("fut_foreign_chg5_z") < -1), "fwd5", "主力"),
        "外資現貨 20 日極端買超 z>1.5 (20 日,反指標)": (g("foreign_z20") > 1.5, "fwd20", "主力"), "外資現貨 20 日極端賣超 z<-1.5 (20 日)": (g("foreign_z20") < -1.5, "fwd20", "主力"),
        "外資期貨 5 日大增倉 z>1.5 (5 日)": (g("fut_foreign_chg5_z") > 1.5, "fwd5", "主力"), "外資期貨 5 日大減倉 z<-1.5 (5 日)": (g("fut_foreign_chg5_z") < -1.5, "fwd5", "主力"),
    }
    findings, active = [], []
    for name, (mask, col, cat) in conds.items():
        st = _cond(m, mask, col)
        if not st:
            continue
        rec = {"name": name, "cat": cat, "horizon": col, **st}
        findings.append(rec)
        if bool(mask.fillna(False).astype(bool).iloc[-1]):
            active.append(rec)
    findings.sort(key=lambda r: (not r["valid"], -abs(r["excess"])))
    last = m.iloc[-1]
    f = lambda k, n=2: (round(float(last[k]), n) if k in m and pd.notna(last[k]) else None)  # noqa: E731
    today = {"date": str(last["date"]), "smart2": f("smart2"), "smart_core": f("smart_core"), "txo_cp_z": f("txo_f_cp_diff_z"), "fut_chg10_z": f("fut_chg10_z"), "mtx_retail_inv": f("mtx_retail_inv"), "rel_kospi5": f("rel_kospi5"), "rel_sp5": f("rel_sp5"), "smart_spread": f("smart_spread"), "vix": f("g_vix_level", 1), "kospi_r0": f("kospi_r0"), "hsi_r0": f("hsi_r0"),
             "nikkei_r0": f("nikkei_r0"), "sox_r1": f("g_sox_r1"), "sp500_r1": f("g_sp500_r1"), "foreign_z5": f("foreign_z5"), "foreign_z20": f("foreign_z20"), "fut_chg5_z": f("fut_foreign_chg5_z"), "margin_chg5_pct": f("margin_chg5_pct")}
    corr = {}
    r1 = c.pct_change() * 100
    for name, col in {"KOSPI": "kospi_r0", "日經": "nikkei_r0", "恆生": "hsi_r0", "S&P500 前晚": "g_sp500_r1", "費半 前晚": "g_sox_r1"}.items():
        if col in m:
            dd = pd.concat([g(col), r1, m["year"]], axis=1).dropna(); dd.columns = ["x", "y", "year"]
            recent = dd[dd["year"] >= dd["year"].max() - 1]
            roll = dd.tail(60)
            corr[name] = {"all": round(float(dd["x"].corr(dd["y"])), 2), "recent2y": round(float(recent["x"].corr(recent["y"])), 2) if len(recent) > 60 else None, "last60": round(float(roll["x"].corr(roll["y"])), 2) if len(roll) > 30 else None}
    s2 = today.get("smart2")
    smart2_txt = ("" if s2 is None else f"聰明錢 v2 {s2:+.2f}：" + ("強力偏多 (前 20% 區，歷史 5/10/20 日上漲 64/67/68%)" if s2 > 1.5 else "偏多" if s2 > 0.5 else "偏空 (後 20% 區，歷史 58/60/64%)" if s2 < -1.5 else "偏空" if s2 < -0.5 else "中性"))
    smart_txt = ("聰明錢 (外資現貨+期貨) 明顯強於散戶 (融資)" if (today["smart_spread"] or 0) > 1 else "散戶加碼快於聰明錢 (歷史上 5 日偏弱，9/9 年)" if (today["smart_spread"] or 0) < -1 else "聰明錢與散戶差距不大")
    return {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "today": today, "active": active, "findings": findings, "lead_lag": lead, "corr": corr,
            "text": {"smart": smart_txt, "smart2": smart2_txt,
                     "kospi": (f"台股 5 日相對 KOSPI {today['rel_kospi5']:+.1f}%" + ("，落後 >3% → 歷史 5 日補漲 64% (13 年 77% 一致)" if (today["rel_kospi5"] or 0) < -3 else "，領先 >3% 無對稱效果 (不算訊號)" if (today["rel_kospi5"] or 0) > 3 else "")) if today["rel_kospi5"] is not None else "",
                     "vix": (f"VIX {today['vix']}" + ("：>30 恐慌區，歷史 20 日 +2.3% vs +1.0% (7/7 年)" if (today["vix"] or 0) > 30 else "：<13 自滿區" if (today["vix"] or 99) < 13 else "")) if today["vix"] is not None else ""},
            "note": "各國市場對台股「隔日」最有用的是同日亞股收盤 (恆生 > KOSPI)，前晚美股只有微弱訊息，前一日亞股/上證無用；5~20 日最強是 VIX 水準。主力籌碼：外資期貨增減與聰明錢−散戶差對 5~20 日穩定 (8~9/9 年)，外資現貨 20 日極端買超反而是反指標，融資本身無預測力。大戶週資料累積中。"}
