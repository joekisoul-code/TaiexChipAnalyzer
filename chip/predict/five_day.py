"""後五日 (交易日，不含休市) 方向模組 (2026-09-24)。

研究 (scratch five_study.py，2014~ 走動式，基準 5 日上漲 59%)：
- 現行短線模型 h5 (+asia|lgb) 分位叫牌整體 54.7%：多方 60% ≈ 基準、空方只 42% (五日空單輸給基準)。加 5 日專用特徵 (+asia+extra) 57%，空方仍 42%。
- 驗證有效訊號投票 (美 10 年債 5 日↓、VIX 期限結構↑、籌碼綜合分↑、波動/60 日↑、籌碼綜合 5 日變化↑、美元指數 5 日↑)：
  淨票 +1 → 63%、+2 → 64%、+3 → 70% (n=162)、+4 → 66%；−2 → 54%、−3 → 52%、−4 → 41% (n=64)。
- 模型未叫牌且淨票 +1 → 72% (n=129)；模型叫牌與投票同向 59%、反向 54%。
結論：五日方向的可用邏輯是「不對稱」的 — 只有多方可預測 (訊號共識 ≥+2 → 64~70%)，空方沒有穩定邏輯 (只在淨票 ≤−4 才 59% 下跌、樣本少)。
規則：net = 6 票淨票 + 模型票 (模型 30/70 分位叫牌 ±1)；net ≥ +2 → 偏多 (走動式 65.4%，覆蓋 26%，逐年最低 51%，2026 年 66%)；空方停用 (淨票 ≤−4 命中 47%)；其餘中性並顯示同淨票歷史上漲率。
train() 走動式重算 by_net 統計存 data/models/five_day.json；build() 只查表。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
VOTE_F = {"g_us10y_r5": -1, "g_vix_term": 1, "composite_smooth": 1, "vola_ratio": 1, "composite_chg5": 1, "g_dxy_r5": 1}
NAMES = {"g_us10y_r5": "美 10 年債殖利率 5 日 (反向)", "g_vix_term": "VIX 期限結構", "composite_smooth": "籌碼綜合分", "vola_ratio": "波動/60 日均", "composite_chg5": "籌碼綜合 5 日變化", "g_dxy_r5": "美元指數 5 日", "model": "五日模型 (+asia+extra)"}
EXTRA = ["g_vix_term", "g_us10y_r5", "g_dxy_r5", "vola_ratio", "composite_chg5", "g_curve_10y_3m", "g_vix_level", "fut_foreign_pct", "smart2", "g_copper_gold_r20", "g_usdtwd_r5", "foreign_z20", "gov8_20d"]
FIRST_YEAR = 2014
STRONG_NET = 4   # 2026-09-24：淨票 ≥+4 為「強」：走動式 by_net 合計 72% (n≈162)，逐年最低約 47% (年份樣本少、起伏大)
BULL_NET, BEAR_NET = 2, -99   # 空方停用：淨票 ≤−4 走動式命中只 47% (n=74)，五日空單沒有穩定邏輯


def _frame(scored, night):
    from . import crossmkt as XM, patterns as P, short_term as ST
    m = XM.add_features(ST.build_matrix(scored, night))
    d = P._prep(m)
    d["year"] = d["date"].astype(str).str[:4].astype(int)
    return d


def _feats(d):
    from . import short_term as ST
    return list(dict.fromkeys(ST.FEATURE_SETS["+asia"] + [f for f in EXTRA if f in d.columns]))


def train(scored: pd.DataFrame, night=None, write: bool = True, verbose: bool = True) -> dict:
    from . import short_term as ST
    d = _frame(scored, night)
    feats = _feats(d)
    v = d[["date", "year", "fwd5"] + list(VOTE_F)].dropna(subset=["fwd5"]).copy()
    for f, sgn in VOTE_F.items():
        s = pd.Series(0.0, index=v.index)
        for y in sorted(v["year"].unique()):
            if y < FIRST_YEAR:
                continue
            prev = v[v["year"] < y][f].dropna()
            if len(prev) < 300:
                continue
            lo, hi = prev.quantile(0.2), prev.quantile(0.8); mm = v["year"] == y
            s[mm & (v[f] >= hi)] = sgn; s[mm & (v[f] <= lo)] = -sgn
        v["v_" + f] = s
    v["net_sig"] = v[[c for c in v.columns if c.startswith("v_")]].sum(axis=1)
    # 模型票 (走動式分位 30/70)
    o = ST._wf(d, feats, "fwd5", 5, FIRST_YEAR, lambda: ST.LgbModel())
    o["call"] = 0.0
    yrs = sorted(o["year"].unique())
    for y in yrs[2:]:
        prev = o[o["year"] < y]["pred"]; lo, hi = prev.quantile(0.3), prev.quantile(0.7); mm = o["year"] == y
        o.loc[mm & (o["pred"] >= hi), "call"] = 1; o.loc[mm & (o["pred"] <= lo), "call"] = -1
    v = v.merge(o[["date", "pred", "call"]], on="date", how="left").fillna({"call": 0})
    v = v[v["year"] >= yrs[2]]
    v["net"] = v["net_sig"] + v["call"]
    v["up"] = (v["fwd5"] > 0).astype(float)
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "features": feats, "votes": list(VOTE_F), "bull_net": BULL_NET, "bear_net": BEAR_NET,
           "base_up": round(float(v["up"].mean()), 3), "n": int(len(v)), "by_net": {}, "thresholds": {}}
    for k, g in v.groupby("net"):
        if len(g) >= 20:
            yr = [gg["up"].mean() for _, gg in g.groupby("year") if len(gg) >= 5]
            out["by_net"][str(int(k))] = {"n": int(len(g)), "up": round(float(g["up"].mean()), 3), "yr_min": round(float(min(yr)), 2) if yr else None, "yr_max": round(float(max(yr)), 2) if yr else None, "years": len(yr)}
    for name, mm, side in (("bull", v["net"] >= BULL_NET, 1), ("bear", v["net"] <= BEAR_NET, -1), ("neutral", (v["net"] > BEAR_NET) & (v["net"] < BULL_NET), 0)):
        g = v[mm]
        hit = (g["up"] == 1) if side > 0 else (g["up"] == 0) if side < 0 else None
        yr = [(gg["up"] == (1 if side > 0 else 0)).mean() for _, gg in g.groupby("year") if len(gg) >= 5] if side else []
        out[name] = {"n": int(len(g)), "cov": round(len(g) / len(v), 3), "hit": round(float(hit.mean()), 3) if side and len(g) else None, "up": round(float(g["up"].mean()), 3) if len(g) else None,
                     "yr_min": round(float(min(yr)), 2) if yr else None, "years": len(yr)}
    # 今年 (最新一年) 成績
    ly = v[v["year"] == v["year"].max()]
    out["latest_year"] = {"year": int(v["year"].max()), "n": int(len(ly)), "bull_hit": round(float(ly.loc[ly["net"] >= BULL_NET, "up"].mean()), 3) if (ly["net"] >= BULL_NET).any() else None, "n_bull": int((ly["net"] >= BULL_NET).sum())}
    # 今日用的門檻 (全歷史分位) 與最終模型
    for f in VOTE_F:
        s = d[f].dropna(); out["thresholds"][f] = {"lo": round(float(s.quantile(0.2)), 4), "hi": round(float(s.quantile(0.8)), 4)}
    out["thresholds"]["model"] = {"lo": round(float(o["pred"].quantile(0.3)), 4), "hi": round(float(o["pred"].quantile(0.7)), 4)}
    if write:
        dd = d.dropna(subset=["fwd5"])
        mdl = ST.LgbModel().fit(dd[feats], dd["fwd5"])
        M.save("five_day_lgb", {"models": mdl.models, "psd": mdl.psd, "features": feats, "trained_at": out["trained_at"]})
        M.save_json("five_day", out)
    if verbose:
        print(f"  five_day: 基準 {out['base_up']}；偏多 (net≥{BULL_NET}) 命中 {out['bull']['hit']} 覆蓋 {out['bull']['cov']} 年最低 {out['bull']['yr_min']}；偏空 (net≤{BEAR_NET}) 命中 {out['bear']['hit']} 覆蓋 {out['bear']['cov']}；中性上漲率 {out['neutral']['up']}；今年偏多 {out['latest_year']}")
        print("  by_net:", {k: (x['up'], x['n']) for k, x in out["by_net"].items()})
    return out


def build(scored: pd.DataFrame, night=None) -> dict | None:
    from . import model as M2, short_term as ST
    st = M.load_json("five_day")
    if not st:
        return None
    d = _frame(scored, night)
    row = d.iloc[-1]
    votes = []
    net = 0
    for f, sgn in VOTE_F.items():
        th = st["thresholds"].get(f) or {}
        val = row.get(f)
        if val is None or val != val or not th:
            votes.append({"key": f, "name": NAMES[f], "s": 0, "dir": "—", "note": "無資料"}); continue
        s = sgn if val >= th["hi"] else -sgn if val <= th["lo"] else 0
        net += s
        votes.append({"key": f, "name": NAMES[f], "s": s, "dir": "多" if s > 0 else "空" if s < 0 else "—", "value": round(float(val), 3), "note": f"{NAMES[f]} {float(val):+.2f} 在{'高檔' if val >= th['hi'] else '低檔' if val <= th['lo'] else '中間'}"})
    ms = 0
    b = M2.load("five_day_lgb")
    if b:
        try:
            x = row[b["features"]].to_frame().T.astype(float)
            pred = float(np.mean([mm.predict(x)[0] for mm in b["models"]]) / (b.get("psd") or 1.0))
            th = st["thresholds"]["model"]
            ms = 1 if pred >= th["hi"] else -1 if pred <= th["lo"] else 0
            votes.append({"key": "model", "name": NAMES["model"], "s": ms, "dir": "多" if ms > 0 else "空" if ms < 0 else "—", "value": round(pred, 3), "note": f"五日模型分數 {pred:+.2f} ({'前 30%' if ms > 0 else '後 30%' if ms < 0 else '中間'})"})
            net += ms
        except Exception as e:  # noqa: BLE001
            log.warning("five_day model: %s", e)
    bn = (st.get("by_net") or {}).get(str(int(net))) or {}
    if net >= BULL_NET:
        call, stt = "偏多", st.get("bull") or {}
    elif net <= BEAR_NET:
        call, stt = "偏空", st.get("bear") or {}
    else:
        call, stt = "中性", st.get("neutral") or {}
    text = (f"後五日 (交易日)：{call}。訊號淨票 {net:+d} (多 {sum(1 for v in votes if v['s'] > 0)} / 空 {sum(1 for v in votes if v['s'] < 0)})"
            + (f"；同淨票歷史五日上漲率 {bn['up']:.0%} (n={bn['n']}，年 {bn['yr_min']:.0%}~{bn['yr_max']:.0%})" if bn else "")
            + (f"；規則整體：{'偏多' if call == '偏多' else '偏空'}叫牌樣本外命中 {stt['hit']:.0%} (覆蓋 {stt['cov']:.0%}，逐年最低 {stt['yr_min']:.0%})" if call != "中性" and stt.get("hit") else (f"；基準五日上漲 {st['base_up']:.0%}" if call == "中性" else ""))
            + ("；研究顯示五日空方無穩定邏輯，不建議放空" if call != "偏空" and net < 0 else ""))
    strength, strong = "", None
    if call == "偏多" and net >= STRONG_NET:
        rows_ = [v for k, v in (st.get("by_net") or {}).items() if int(k) >= STRONG_NET and v.get("n")]
        n_ = sum(v["n"] for v in rows_)
        if n_ >= 60:
            strong = {"n": n_, "hit": round(sum(v["n"] * v["up"] for v in rows_) / n_, 3), "yr_min": min(v["yr_min"] for v in rows_ if v.get("yr_min") is not None)}
            strength = "強"
            text += f"；淨票 ≥+{STRONG_NET} 為強訊號：歷史五日上漲 {strong['hit']:.0%} (n={n_}，最差年份約 {strong['yr_min']:.0%})"
    return {"date": str(row["date"])[:10], "call": call, "strength": strength, "strong": strong, "net": int(net), "votes": votes, "by_net": bn or None, "rule": stt, "base_up": st.get("base_up"), "latest_year": st.get("latest_year"), "text": text}
