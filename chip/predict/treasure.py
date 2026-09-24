"""挖寶雷達模型 (2026-09-24)：學習「推薦後 30 天 (21 交易日) 是否命中」並輸出前端可執行的樹模型 JSON。

命中定義與 App (learning.js v3) 相同：21 交易日內收盤峰值 ≥ +6% 或相對大盤 ≥ +4pt，先跌破 −8% (盤中低點) 而峰值 <6% 不算，結案收盤仍 >0。
樣本：當日成交值前 ~230 檔上市櫃個股 (排除 ETF)，2019~ 日 K；候選條件同 App 掃描器 (成交值 >5 千萬、當日漲幅 <9.3%)。
研究 (scratch tr_study.py，2022~ 走動式)：現行評分每日前 6 檔命中 40.1% ≈ 隨機 39.2%；
LGB 分數 ≥ 前一年 90 分位 且在掃描前 40 內 → 44.0% (結案均 +5.6% vs 現行 +4.1%)，逐年 2022 40/34、2024 41/37、2025 50/43、2026 48/41 皆勝現行。
A 級強化 (2026-09-24 第二輪，scratch tr_a/tr_b/tr_c.py)：高分 (≥ 前一年 90 分位) 只在「大盤低於月線」時才有效 —
  高分 ∧ 大盤月線乖離 <0 → 51.8% (n=330，逐年 50~53%，結案均 +7.5%)；高分 ∧ 大盤在月線上 → 42.1% (≈ B 級 40%)。
  閘門穩健：乖離門檻 +1%~−1% 皆 50.8~51.8%。提高分位門檻 (95/97) 或加市場寬度/候選池相對特徵，逐年不穩定，未採用。
  → 分級：A = 高分 ∧ 大盤在月線下；B+ = 高分 ∧ 大盤在月線上；B = 其餘。
  強勢市場另訓專屬模型/加條件：2024 年都跌到 23~34%，無穩定優勢，不採用。
  出場 (tr_g.py)：A 級持有 21 交易日 +7.5% (勝率 67%) 遠勝 +6% 停利 (+1.7%) 或移動停利 (+3.2%) → 建議持有滿 30 天。
  進場 (tr_h.py)：推薦當天收盤 +7.5% > 隔天開盤 +6.8% > 掛低 1~3% 等回檔 (+4.3~4.6%，錯過最強的股票) → 當天就進場。
主要特徵：20 日波動 (高 → 易達 +6%)、距 20 日低點 (遠 → 好)、20/60 日漲幅 (跌多 → 好，均值回歸)、距 60 日高點 (深 → 好)。
前端 (learning.js treasureModel) 用 dump 的樹直接算分；特徵由 Yahoo 6 個月日 K + 今日快照在前端計算。
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
H, TGT, STOP, REL = 21, 6.0, -8.0, 4.0
FEATS = ["pct", "amp", "lval", "b5", "b10", "b20", "b60", "align", "ret5", "ret20", "ret60", "dd_hi20", "dd_hi60", "lo20_dist", "clv", "uw", "lw",
         "vol_ratio", "vola20", "streak", "lag", "rs20", "m_ret1", "m_bias20"]
Q_APLUS = 0.97     # A+：分數 ≥ 前一年 97 分位 ∧ 大盤月線下 (tr_k.py：58.2%、+11.3%、勝率 76%，n=165；逐年 61/54/63/45%)
GATE_MBIAS = 0.0   # A 級閘門：大盤月線乖離 < 0 (大盤在月線下)
PARAMS = dict(n_estimators=160, learning_rate=0.04, num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def features(g: pd.DataFrame, mk: pd.DataFrame, label: bool = True) -> pd.DataFrame:
    """g: 單一標的 date/open/high/low/close/volume/amount；mk: date/m_close/m_ret1/m_bias20/m_ret20。前端 learning.js tmFeatures 為同款實作。"""
    g = g.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g[g["close"] > 0].reset_index(drop=True).merge(mk, on="date", how="left")
    c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
    g["pct"] = (c / c.shift(1) - 1) * 100; g["amp"] = (h - l) / l * 100; g["lval"] = np.log10(g["amount"].clip(lower=1))
    for n in (5, 10, 20, 60):
        g[f"ma{n}"] = c.rolling(n).mean()
    g["b5"], g["b10"], g["b20"], g["b60"] = [(c / g[f"ma{n}"] - 1) * 100 for n in (5, 10, 20, 60)]
    g["align"] = np.where((g["ma5"] >= g["ma10"]) & (g["ma10"] >= g["ma20"]), 1, np.where((g["ma5"] <= g["ma10"]) & (g["ma10"] <= g["ma20"]), -1, 0))
    g["ret5"], g["ret20"], g["ret60"] = c.pct_change(5) * 100, c.pct_change(20) * 100, c.pct_change(60) * 100
    g["dd_hi20"] = (c / h.rolling(20).max() - 1) * 100; g["dd_hi60"] = (c / h.rolling(60).max() - 1) * 100; g["lo20_dist"] = (c / l.rolling(20).min() - 1) * 100
    rng = (h - l).replace(0, np.nan); g["clv"] = ((c - l) / rng).clip(0, 1).fillna(.5)
    g["uw"] = (h - np.maximum(o, c)) / c * 100; g["lw"] = (np.minimum(o, c) - l) / c * 100
    g["vol_ratio"] = v / v.rolling(20).mean(); g["vola20"] = g["pct"].rolling(20).std()
    s = np.sign(g["pct"].fillna(0)).values; st = np.zeros(len(g))
    for i in range(1, len(g)):
        st[i] = st[i - 1] + s[i] if s[i] != 0 and (st[i - 1] == 0 or np.sign(st[i - 1]) == s[i]) else s[i]
    g["streak"] = st; g["lag"] = g["m_ret1"] - g["pct"]; g["rs20"] = g["ret20"] - g["m_ret20"]
    if label:
        C, L, Mk, n = c.values, l.values, g["m_close"].values, len(g)
        hit = np.full(n, np.nan); fin = np.full(n, np.nan)
        for i in range(n - H):
            cr = (C[i + 1:i + H + 1] / C[i] - 1) * 100; lr = (L[i + 1:i + H + 1] / C[i] - 1) * 100
            mr = (Mk[i + 1:i + H + 1] / Mk[i] - 1) * 100 if np.isfinite(Mk[i]) else np.zeros(H)
            cm = np.maximum.accumulate(cr)
            stop = bool(np.any((lr <= STOP) & (cm < TGT)))
            reached = (not stop) and bool(np.any((cr >= TGT) | ((cr - mr) >= REL)))
            hit[i] = float(reached and cr[-1] > 0); fin[i] = cr[-1]
        g["hit"] = hit; g["fin21"] = fin
    return g


def _dump_trees(booster) -> list:
    """LightGBM 樹 → 精簡陣列：每棵樹為節點表 [feat, thr, left, right, default_left] 與葉值；前端逐棵走訪加總 raw score。"""
    js = booster.dump_model()
    out = []
    for t in js["tree_info"]:
        nodes, leaves = [], []
        def walk(n):
            if "leaf_index" in n or "leaf_value" in n and "split_feature" not in n:
                leaves.append(round(float(n["leaf_value"]), 9)); return -len(leaves)   # 負數 = 葉 (-(k+1))
            idx = len(nodes); nodes.append(None)
            l_ = walk(n["left_child"]); r_ = walk(n["right_child"])
            nodes[idx] = [int(n["split_feature"]), float(n["threshold"]), l_, r_, 1 if n.get("default_left", True) else 0]
            return idx
        root = walk(t["tree_structure"])
        out.append({"n": nodes, "v": leaves, "r": root})
    return out


def build_panel(n_twse: int = 170, n_tpex: int = 60, start: str = "2019-01-01") -> pd.DataFrame:
    """樣本：今日成交值前 n_twse 檔上市 + n_tpex 檔上櫃個股 (排除 00 開頭 ETF)，FinMind 日 K。"""
    import time
    import requests
    from ..sources import finmind
    codes = []
    try:
        j = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30).json()
        rows = sorted(((r["Code"], float(r.get("TradeValue") or 0)) for r in j if r.get("Code", "").isdigit() and len(r["Code"]) == 4 and not r["Code"].startswith("00")), key=lambda x: -x[1])
        codes += [c for c, _ in rows[:n_twse]]
    except Exception as e:  # noqa: BLE001
        log.warning("twse list: %s", e)
    try:
        j = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", timeout=30).json()
        rows = []
        for r in j:
            c = r.get("SecuritiesCompanyCode", "")
            try:
                v = float(str(r.get("TransactionAmount", "0")).replace(",", ""))
            except ValueError:
                v = 0
            if c.isdigit() and len(c) == 4 and not c.startswith("00"):
                rows.append((c, v))
        codes += [c for c, _ in sorted(rows, key=lambda x: -x[1])[:n_tpex]]
    except Exception as e:  # noqa: BLE001
        log.warning("tpex list: %s", e)
    if len(codes) < 50:   # GitHub Actions 連不到 TWSE/TPEx openapi (2026-09-24 實測) → 用上次訓練存下的股票清單
        prev = (M.load_json("treasure_model") or {}).get("universe") or []
        if prev:
            log.warning("treasure universe: openapi 失敗，改用模型檔內的 %d 檔清單", len(prev))
            codes = list(prev)
        else:
            from ..realtime import LARGE_CAPS
            codes = list(LARGE_CAPS)
    frames = []
    for c in dict.fromkeys(codes):
        for _ in range(3):
            try:
                p = finmind.stock_price(c, start)
                if not p.empty:
                    frames.append(p[["date", "open", "high", "low", "close", "volume", "amount"]].assign(code=c))
                break
            except Exception as e:  # noqa: BLE001
                if any(k in str(e) for k in ("402", "429", "limit")):
                    time.sleep(65)
                else:
                    break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train(panel: pd.DataFrame | None = None, write: bool = True, verbose: bool = True) -> dict:
    """panel: code/date/open/high/low/close/volume/amount (多檔)。"""
    import lightgbm as lgb
    from ..analysis import backtest
    if panel is None:
        panel = build_panel()
    mk = backtest.load_long("2018-01-01")[["date", "close"]].copy(); mk["date"] = mk["date"].astype(str)
    mk["m_close"] = pd.to_numeric(mk["close"]); mk["m_ret1"] = mk["m_close"].pct_change() * 100
    mk["m_bias20"] = (mk["m_close"] / mk["m_close"].rolling(20).mean() - 1) * 100; mk["m_ret20"] = mk["m_close"].pct_change(20) * 100
    mk = mk[["date", "m_close", "m_ret1", "m_bias20", "m_ret20"]]
    panel = panel.copy(); panel["date"] = panel["date"].astype(str)
    D = pd.concat([features(g, mk).assign(code=code) for code, g in panel.groupby("code")], ignore_index=True)
    D["year"] = D["date"].str[:4].astype(int)
    D = D[(D["year"] >= 2020) & D["hit"].notna() & (D["amount"] > 5e7) & (D["pct"] < 9.3)].dropna(subset=FEATS)
    # 走動式 (年)
    D["p"] = np.nan
    for y in range(2022, int(D["year"].max()) + 1):
        tr = (D["year"] < y) & (D["date"] < f"{y - 1}-12-01"); te = D["year"] == y
        m = lgb.LGBMClassifier(**PARAMS).fit(D.loc[tr, FEATS], D.loc[tr, "hit"])
        D.loc[te, "p"] = m.predict_proba(D.loc[te, FEATS])[:, 1]
    E = D[D["p"].notna()].copy()
    # 掃描器 power (無 PBR/PER) → 前 40 候選池
    mom = np.where(E["pct"] >= 0, np.minimum(E["pct"], 7) * 1.6, E["pct"] * 0.6)
    E["screen"] = mom + E["lval"] * 2 + np.minimum(E["amp"], 8) * .8 + np.where(E["pct"] < E["m_ret1"], np.minimum(E["m_ret1"] - E["pct"], 5) * 1.2, 0) + np.where((E["pct"] >= 3) & (E["pct"] < 9), (E["pct"] - 3) * .6, 0)
    res = {"tiers": {}, "by_year": {}}
    rows = []; last = {}
    for y in sorted(E["year"].unique()):
        prev = E[E["year"] == y - 1]["p"]; _b = prev if len(prev) else E[E["year"] == y]["p"]; th = float(_b.quantile(0.9)); th2 = float(_b.quantile(Q_APLUS))
        for d, g in E[E["year"] == y].sort_values("date").groupby("date"):
            gg = g.nlargest(40, "screen").sort_values("p", ascending=False); n = 0
            for r in gg.itertuples():
                lp = last.get(r.code)
                if lp is not None and (pd.Timestamp(d) - pd.Timestamp(lp)).days < 30:
                    continue
                rows.append((y, bool(r.p >= th and r.m_bias20 < GATE_MBIAS), bool(r.p >= th and r.m_bias20 >= GATE_MBIAS), bool(r.p >= th2 and r.m_bias20 < GATE_MBIAS), r.hit, r.fin21)); last[r.code] = d; n += 1
                if n >= 6:
                    break
    x = pd.DataFrame(rows, columns=["year", "A", "Bp", "Ap", "hit", "fin"])
    for tier, g in (("A", x[x["A"]]), ("A+", x[x["Ap"]]), ("A-", x[x["A"] & ~x["Ap"]]), ("B+", x[x["Bp"]]), ("B", x[~x["A"] & ~x["Bp"]]), ("all", x)):
        res["tiers"][tier] = {"n": int(len(g)), "hit": round(float(g["hit"].mean()), 3), "fin": round(float(g["fin"].mean()), 2),
                              "win": round(float((g["fin"] > 0).mean()), 3), "med": round(float(g["fin"].median()), 2), "q10": round(float(g["fin"].quantile(0.1)), 2), "q90": round(float(g["fin"].quantile(0.9)), 2),
                              "worst_year": round(float(g.groupby("year")["fin"].mean().min()), 2) if len(g) else None}
    for y, g in x.groupby("year"):
        res["by_year"][int(y)] = {"A+": round(float(g[g["Ap"]]["hit"].mean()), 3) if g["Ap"].any() else None, "A+fin": round(float(g[g["Ap"]]["fin"].mean()), 2) if g["Ap"].any() else None, "A": round(float(g[g["A"]]["hit"].mean()), 3) if g["A"].any() else None, "nA": int(g["A"].sum()), "B+": round(float(g[g["Bp"]]["hit"].mean()), 3) if g["Bp"].any() else None, "B": round(float(g[~g["A"] & ~g["Bp"]]["hit"].mean()), 3) if (~g["A"] & ~g["Bp"]).any() else None}
    E["dec"] = pd.qcut(E["p"], 10, labels=False); res["deciles"] = E.groupby("dec")["hit"].mean().round(3).tolist()
    res["base_hit"] = round(float(E["hit"].mean()), 3)
    # 最終模型 + 門檻 (最近一年分數的 90 分位)
    fm = lgb.LGBMClassifier(**PARAMS).fit(D[FEATS], D["hit"])
    ly = D[D["year"] >= int(D["year"].max()) - 1]
    _pl = fm.predict_proba(ly[FEATS])[:, 1]
    th_final = float(np.quantile(_pl, 0.9)); th_plus = float(np.quantile(_pl, Q_APLUS))
    imp = fm.booster_.feature_importance("gain"); imp = imp / imp.sum()
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "features": FEATS, "init": float(fm.booster_.dump_model().get("average_output", 0) or 0),
           "trees": _dump_trees(fm.booster_), "th_A": round(th_final, 4), "th_Aplus": round(th_plus, 4), "gate": {"m_bias20_lt": GATE_MBIAS, "note": "A 級需大盤在月線下；大盤在月線上的高分股標 B+"}, "exit": {"note": "出場研究 (scratch tr_g.py，2022~ 走動式 A 級 n=330)：持有 21 交易日平均 +7.5%、勝率 67%、最差年 +5.1%；+6% 停利/−8% 停損只剩 +1.7%，+10%/−8% +2.5%，+6% 後移動停利 +3.2% → 建議持有滿 30 天，不提早停利；B 級持有 21 日 +3.3%、最差年 −1.6%", "A_hold": 7.48, "A_tp6": 1.68, "A_trail": 3.24},
           "entry": {"note": "進場研究 (scratch tr_h.py，A 級 n=330，出場固定第 21 交易日)：推薦當天收盤 +7.5%；隔天開盤 +6.8%；掛低 1/2/3% 等 3 天成交率 69/57/45%、整體 +4.6/+4.4/+4.3% (沒成交的那批若當天買平均 +9~14%) → 推薦當天就進場，不要等回檔", "A_close": 7.48, "A_open": 6.84, "A_lim1": 4.55, "A_lim2": 4.41, "fill_lim2": 0.57}, "n_rows": int(len(D)), "n_stocks": int(D["code"].nunique()),
           "universe": sorted(D["code"].unique().tolist()), "importance": sorted(({"f": f, "w": round(float(w), 3)} for f, w in zip(FEATS, imp)), key=lambda z: -z["w"])[:10],
           "oos": res, "def": {"H": H, "target": TGT, "stop": STOP, "rel": REL}}
    # 自檢：JSON 樹與 LightGBM 預測一致
    raw = np.array([_eval(out, r) for r in D[FEATS].tail(200).values])
    ref = fm.predict_proba(D[FEATS].tail(200))[:, 1]
    out["selfcheck_maxdiff"] = round(float(np.max(np.abs(1 / (1 + np.exp(-raw)) - ref))), 6)
    if verbose:
        print(f"  treasure: 樣本 {out['n_rows']} 列 {out['n_stocks']} 檔；走動式 A+ {res['tiers']['A+']}、A 級 {res['tiers']['A']}、B+ 級 {res['tiers']['B+']}、B 級 {res['tiers']['B']}、全部 {res['tiers']['all']}；十分位 {res['deciles']}；門檻 {out['th_A']}；樹 {len(out['trees'])} 棵；自檢差 {out['selfcheck_maxdiff']}")
        print("  逐年:", res["by_year"])
    if write:
        M.save_json("treasure_model", out)
    return out


def _eval(model: dict, x) -> float:
    s = model.get("init", 0.0)
    for t in model["trees"]:
        k = t["r"]
        if k < 0:
            s += t["v"][-k - 1]; continue
        while k >= 0:
            f, thr, l_, r_, dl = t["n"][k]
            v = x[f]
            k = (l_ if dl else r_) if (v is None or v != v) else (l_ if v <= thr else r_)
        s += t["v"][-k - 1]
    return s
