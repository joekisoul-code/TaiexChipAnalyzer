"""預測邏輯總表 (2026-09-24)：把系統掌握的「所有資訊」逐一與歷史漲跌比對，找出哪些資訊在哪個視野真的有預測力、方向為何、逐年是否穩定。

方法 (每個資訊 × 視野 1/3/5/10/20 日，2010~)：
- 走動式分位：每年用「之前所有年份」的 20%/80% 分位當門檻，把當年的日子分成 低檔 / 中間 / 高檔 三組 → 高檔組與低檔組的未來上漲率差 (diff)、平均報酬差、
  逐年一致性 (每年高檔 vs 低檔的上漲率差是否與整體同號，年樣本 ≥10)、rank IC (走動式年度平均)。
- 有效 = 年數 ≥8、一致性 ≥0.65、|上漲率差| ≥6pt。方向 = 高檔組上漲率較高 → 「越高越漲」，否則「越高越跌」(反指標)。
- 依資訊群組 (夜盤/亞股/美股/外資現貨/期貨/選擇權/散戶/融資/八大行庫/技術/量能/波動/聰明錢/匯率/商品/日曆) 彙整最強一項，並產生每個視野的邏輯敘述與「無效清單」。
輸出 data/models/infomap.json；forecast.json `infomap` 帶精簡版 (各視野前 8 名 + 今日讀數所在檔位)。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
HORIZONS = (1, 3, 5, 10, 20)
FIRST_YEAR = 2013
MIN_YEARS, MIN_CONS, MIN_DIFF = 8, 0.65, 0.06
GROUPS = {
    "夜盤台指期": ["night_chg_pct"],
    "亞股同日": ["hsi_r0", "kospi_r0", "nikkei_r0", "kospi_rel0"],
    "美股前晚": ["g_sox_r1", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1", "g_vix_r1", "g_vix_level", "g_vix_term"],
    "外資現貨": ["foreign_z1", "foreign_z5", "foreign_z20", "foreign_streak", "foreign_consistency"],
    "投信/自營": ["trust_z5", "trust_z20", "trust_streak", "dealer_z1", "dealer_5d"],
    "外資期貨": ["fut_foreign_chg1_z", "fut_foreign_chg5_z", "fut_chg10_z", "fut_foreign_pct", "fut_foreign_net_oi"],
    "選擇權": ["txo_f_cp_diff_z", "txo_d_cp_diff_z", "txo_f_call_z", "txo_f_put_z", "pcr_oi"],
    "小台散戶": ["mtx_retail_inv", "mtx_retail_ratio", "mtx_retail_chg5", "mtx_foreign_net_z"],
    "期現價差/未平倉": ["tx_basis_pct", "tx_basis_chg1", "tx_oi_chg1_z", "tx_vol_z", "top10_spec_net"],
    "融資/借券/當沖": ["margin_chg5_pct", "margin_pct20", "margin_div20", "sbl_chg5", "short_chg5_z", "short_ratio", "maint_ratio"],
    "八大行庫": ["gov8_5d", "gov8_20d", "gov8_net", "gov8_streak"],
    "技術面": ["bias5", "bias20", "bias60", "ret1", "ret5", "ret20", "ret60", "lo20_dist", "hi20_dist", "streak", "rsi_", "ma20_slope", "clv", "gap_open", "range_pct"],
    "量能": ["vol_ratio", "amount_5d_ratio"],
    "波動": ["vola20", "vola_ratio", "g_vix_chg"],
    "籌碼綜合/聰明錢": ["composite_smooth", "composite_chg5", "smart2", "smart_core", "smart_spread"],
    "匯率": ["g_usdtwd_r1", "g_usdtwd_r5", "g_usdtwd_r20", "g_usdtwd_streak", "g_dxy_r5", "g_usdjpy_r5", "g_usdkrw_r5"],
    "商品/利率": ["g_oil_r5", "g_copper_r5", "g_gold_r5", "g_copper_gold_r20", "g_us10y_r5", "g_curve_10y_3m"],
    "日曆": ["dow", "days_to_settle", "days_to_month_end", "days_to_quarter_end", "month"],
}


def _names() -> dict:
    from . import logic as LG, short_term as ST
    n = dict(ST.NAMES); n.update(LG._names())
    n.update({"kospi_rel0": "台股 vs KOSPI 同日", "g_nasdaq_r1": "那斯達克前晚%", "g_tsm_adr_r1": "台積電 ADR 前晚%", "g_vix_term": "VIX 期限結構", "foreign_z1": "外資當日 z", "foreign_streak": "外資連買賣天數", "foreign_consistency": "外資買賣一致性",
              "trust_z20": "投信 20 日 z", "trust_streak": "投信連買賣天數", "dealer_z1": "自營當日 z", "dealer_5d": "自營 5 日", "fut_foreign_pct": "外資期貨淨多百分位", "fut_foreign_net_oi": "外資期貨淨未平倉",
              "txo_d_cp_diff_z": "自營選擇權 call−put z", "txo_f_call_z": "外資 call z", "txo_f_put_z": "外資 put z", "mtx_retail_ratio": "小台散戶多空比", "mtx_retail_chg5": "小台散戶 5 日變化", "mtx_foreign_net_z": "小台外資淨額 z",
              "tx_basis_chg1": "價差日變化", "tx_oi_chg1_z": "期貨未平倉變化 z", "tx_vol_z": "期貨量 z", "top10_spec_net": "前十大特定法人淨額", "margin_pct20": "融資 20 日變化%", "margin_div20": "融資-股價背離", "sbl_chg5": "借券 5 日變化", "short_chg5_z": "融券 5 日 z", "short_ratio": "券資比", "maint_ratio": "維持率",
              "gov8_5d": "八大行庫 5 日", "gov8_20d": "八大行庫 20 日", "gov8_net": "八大行庫當日", "gov8_streak": "八大行庫連買賣", "ret60": "60 日漲跌%", "rsi_": "RSI", "clv": "收盤在振幅位置", "g_vix_chg": "VIX 變化", "composite_chg5": "籌碼綜合 5 日變化", "smart_core": "聰明錢核心", "smart_spread": "聰明錢−散戶差",
              "g_usdtwd_r1": "美元/台幣 1 日%", "g_usdtwd_r20": "美元/台幣 20 日%", "g_usdtwd_streak": "台幣連貶(升)天數", "g_dxy_r5": "美元指數 5 日%", "g_usdjpy_r5": "美元/日圓 5 日%", "g_usdkrw_r5": "美元/韓元 5 日%",
              "g_oil_r5": "原油 5 日%", "g_copper_r5": "銅 5 日%", "g_gold_r5": "黃金 5 日%", "g_copper_gold_r20": "銅金比 20 日", "g_us10y_r5": "美 10 年債殖利率 5 日", "g_curve_10y_3m": "美債殖利率曲線 10y−3m", "days_to_month_end": "距月底", "days_to_quarter_end": "距季底", "month": "月份"})
    return n


def _frame(scored: pd.DataFrame, night) -> pd.DataFrame:
    from . import crossmkt as XM, patterns as P, short_term as ST
    m = ST.build_matrix(scored, night)
    m = XM.add_features(m)
    d = P._prep(m)
    d["year"] = d["date"].astype(str).str[:4].astype(int)
    return d


def _study(d: pd.DataFrame, f: str, h: int) -> dict | None:
    tgt = f"fwd{h}"
    s = d[[f, tgt, "year"]].dropna()
    if len(s) < 600 or s[f].nunique() < 8:
        return None
    s = s.copy(); s["up"] = (s[tgt] > 0).astype(float)
    grp = pd.Series(np.nan, index=s.index)
    for yv in sorted(s["year"].unique()):
        if yv < FIRST_YEAR:
            continue
        prev = s[s["year"] < yv][f]
        if len(prev) < 400:
            continue
        lo, hi = prev.quantile(0.2), prev.quantile(0.8)
        if hi <= lo:
            continue
        m = s["year"] == yv
        grp[m & (s[f] >= hi)] = 1; grp[m & (s[f] <= lo)] = -1; grp[m & (s[f] > lo) & (s[f] < hi)] = 0
    ev = grp.notna()
    if ev.sum() < 300:
        return None
    e = s[ev].copy(); e["g"] = grp[ev]
    hi_, lo_ = e[e["g"] == 1], e[e["g"] == -1]
    if len(hi_) < 60 or len(lo_) < 60:
        return None
    diff = float(hi_["up"].mean() - lo_["up"].mean()); rdiff = float(hi_[tgt].mean() - lo_[tgt].mean())
    yr = []
    for yv, g in e.groupby("year"):
        a, b = g[g["g"] == 1], g[g["g"] == -1]
        if len(a) >= 10 and len(b) >= 10:
            yr.append(float(a["up"].mean() - b["up"].mean()))
    cons = float(np.mean([(v > 0) == (diff > 0) for v in yr])) if yr else None
    ic = []
    for yv, g in e.groupby("year"):
        if len(g) >= 60:
            ic.append(float(g[f].rank().corr(g[tgt].rank())))
    sd = float(np.sqrt(hi_["up"].var() / len(hi_) + lo_["up"].var() / len(lo_))) or 1e-9
    valid = bool(len(yr) >= MIN_YEARS and cons is not None and cons >= MIN_CONS and abs(diff) >= MIN_DIFF)
    return {"n": int(len(e)), "years": len(yr), "up_hi": round(float(hi_["up"].mean()), 3), "up_lo": round(float(lo_["up"].mean()), 3), "base": round(float(e["up"].mean()), 3), "diff": round(diff, 3), "ret_diff": round(rdiff, 3),
            "t": round(diff / sd, 2), "cons": round(cons, 2) if cons is not None else None, "ic": round(float(np.mean(ic)), 3) if ic else None, "ic_pos_years": f"{sum(1 for v in ic if v > 0)}/{len(ic)}" if ic else None,
            "valid": valid, "dir": ("越高越漲" if diff > 0 else "越高越跌 (反指標)") if valid else "無效"}


def train(scored: pd.DataFrame, night=None, write: bool = True, verbose: bool = True) -> dict:
    d = _frame(scored, night)
    names = _names()
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "criteria": {"min_years": MIN_YEARS, "min_cons": MIN_CONS, "min_diff": MIN_DIFF}, "features": {}, "groups": {}, "horizons": {}}
    for grp, fs in GROUPS.items():
        for f in fs:
            if f not in d.columns:
                continue
            for h in HORIZONS:
                r = _study(d, f, h)
                if r:
                    out["features"].setdefault(f, {"name": names.get(f, f), "group": grp, "h": {}})["h"][str(h)] = r
    for h in HORIZONS:
        rows = []
        for f, v in out["features"].items():
            r = v["h"].get(str(h))
            if r:
                rows.append({"f": f, "name": v["name"], "group": v["group"], **r})
        valid = sorted([r for r in rows if r["valid"]], key=lambda x: -abs(x["diff"]))
        invalid_groups = sorted({r["group"] for r in rows} - {r["group"] for r in valid})
        out["horizons"][str(h)] = {"n_tested": len(rows), "n_valid": len(valid), "valid": valid[:15], "invalid_groups": invalid_groups,
                                   "text": (f"{h} 日：" + ("、".join(f"{r['name']}{'↑' if r['diff'] > 0 else '↓'} (高檔 {r['up_hi']:.0%} vs 低檔 {r['up_lo']:.0%}，一致 {r['cons']:.0%})" for r in valid[:6]) if valid else "沒有任何單一資訊通過驗證")
                                            + (f"；無效群組：{'、'.join(invalid_groups)}" if invalid_groups else ""))}
    for grp in GROUPS:
        best = {}
        for h in HORIZONS:
            cands = [(f, out["features"][f]["h"][str(h)]) for f in GROUPS[grp] if f in out["features"] and str(h) in out["features"][f]["h"]]
            if cands:
                f, r = max(cands, key=lambda x: abs(x[1]["diff"]) * (1.0 if x[1]["valid"] else 0.3))
                best[str(h)] = {"f": f, "name": names.get(f, f), "diff": r["diff"], "valid": r["valid"], "dir": r["dir"], "cons": r["cons"]}
        out["groups"][grp] = {"best": best, "any_valid": any(v["valid"] for v in best.values())}
    if verbose:
        for h in HORIZONS:
            print("  " + out["horizons"][str(h)]["text"][:300])
        print("  群組有效：", [g for g, v in out["groups"].items() if v["any_valid"]], "無效：", [g for g, v in out["groups"].items() if not v["any_valid"]])
    if write:
        M.save_json("infomap", out)
    return out


def build(scored: pd.DataFrame, night=None) -> dict | None:
    """今日各有效資訊的讀數落在哪一檔 (高檔/中間/低檔，門檻用全歷史 20/80 分位) → 每個視野的多空票數與敘述。"""
    st = M.load_json("infomap")
    if not st:
        return None
    d = _frame(scored, night)
    row = d.iloc[-1]
    out = {"date": str(row["date"])[:10], "trained_at": st.get("trained_at"), "h": {}, "groups": {g: v["any_valid"] for g, v in (st.get("groups") or {}).items()}}
    for h in HORIZONS:
        H = (st.get("horizons") or {}).get(str(h)) or {}
        items = []
        bull = bear = 0
        for r in H.get("valid") or []:
            f = r["f"]
            if f not in d.columns:
                continue
            v = row.get(f)
            if v is None or v != v:
                continue
            lo, hi = d[f].quantile(0.2), d[f].quantile(0.8)
            tier = "高檔" if v >= hi else "低檔" if v <= lo else "中間"
            sig = 0
            if tier != "中間":
                sig = (1 if r["diff"] > 0 else -1) * (1 if tier == "高檔" else -1)
            bull += sig > 0; bear += sig < 0
            items.append({"f": f, "name": r["name"], "group": r["group"], "value": round(float(v), 2), "tier": tier, "sig": sig, "up_hi": r["up_hi"], "up_lo": r["up_lo"], "cons": r["cons"], "dir": r["dir"],
                          "note": f"{r['name']} {float(v):+.2f} 在{tier}" + (f" → 歷史{'偏多' if sig > 0 else '偏空'} ({(r['up_hi'] if tier == '高檔' else r['up_lo']):.0%})" if sig else " (中間檔無訊號)")})
        out["h"][str(h)] = {"n_valid": H.get("n_valid", 0), "n_tested": H.get("n_tested", 0), "bull": bull, "bear": bear, "items": items, "text": H.get("text"), "invalid_groups": H.get("invalid_groups")}
    return out
