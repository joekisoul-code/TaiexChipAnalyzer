"""挖寶雷達 A 級研究 (雲端用，需 FINMIND_TOKEN)：法人買賣超 / 月營收 是否提高 A 級命中。
用法：python tools/treasure_research.py  → 結果印出並寫 research_out/treasure_research.json (workflow research 模式上傳為 artifact)。
比較 (同一批股票、2022~ 逐年走動式、每日掃描前 40 取前 6、30 天去重)：
  V0 = 現行 24 特徵；V3 = V0 + 外資/投信/合計 5、20 日買賣超 (÷20 日均量) + 營收年增 (單月、近 3 月)。
  分級：A = 分數 ≥ 前一年 90 分位 ∧ 大盤月線乖離 <0；B+ = 高分 ∧ 大盤在月線上；B = 其餘。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chip.analysis import backtest  # noqa: E402
from chip.predict import treasure as T  # noqa: E402
from chip.sources import finmind  # noqa: E402

OUT = Path("research_out"); OUT.mkdir(exist_ok=True)
N_TWSE, N_TPEX = int(os.getenv("TR_N_TWSE", "120")), int(os.getenv("TR_N_TPEX", "30"))
t0 = time.time()
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)


def get(fn, c):
    for k in range(4):
        try:
            return fn(c)
        except Exception as e:  # noqa: BLE001
            m = str(e)
            if any(x in m for x in ("402", "429", "upper", "limit")):
                say("rate limit", c, m[:60]); time.sleep(90 * (k + 1))
            else:
                return pd.DataFrame()
    return pd.DataFrame()


P = T.build_panel(N_TWSE, N_TPEX)
codes = P["code"].unique().tolist(); say("panel", len(codes), "stocks", len(P), "rows", round(time.time() - t0), "s")
inst, rev = [], []
for i, c in enumerate(codes):
    a = get(lambda s: finmind.stock_institutional(s, "2019-01-01"), c)
    if not a.empty: inst.append(a.assign(code=c))
    b = get(lambda s: finmind.fetch("TaiwanStockMonthRevenue", s, "2018-01-01"), c)
    if not b.empty: rev.append(b.assign(code=c))
    if i % 25 == 0: say(i, c, len(inst), len(rev), round(time.time() - t0), "s")
inst = pd.concat(inst) if inst else pd.DataFrame(); rev = pd.concat(rev) if rev else pd.DataFrame()
say("inst", inst["code"].nunique() if len(inst) else 0, "rev", rev["code"].nunique() if len(rev) else 0)

import lightgbm as lgb  # noqa: E402
mk = backtest.load_long("2018-01-01")[["date", "close"]].copy(); mk["date"] = mk["date"].astype(str)
mk["m_close"] = pd.to_numeric(mk["close"]); mk["m_ret1"] = mk["m_close"].pct_change() * 100
mk["m_bias20"] = (mk["m_close"] / mk["m_close"].rolling(20).mean() - 1) * 100; mk["m_ret20"] = mk["m_close"].pct_change(20) * 100
mk = mk[["date", "m_close", "m_ret1", "m_bias20", "m_ret20"]]
P["date"] = P["date"].astype(str)
D = pd.concat([T.features(g, mk).assign(code=c) for c, g in P.groupby("code")], ignore_index=True)
D["year"] = D["date"].str[:4].astype(int)
D = D[(D["year"] >= 2020) & D["hit"].notna() & (D["amount"] > 5e7) & (D["pct"] < 9.3)].dropna(subset=T.FEATS)
XF = []
if len(inst):
    inst["date"] = inst["date"].astype(str); inst = inst.sort_values(["code", "date"])
    for c in ("foreign", "trust", "total"):
        inst[c + "5"] = inst.groupby("code")[c].transform(lambda s: s.rolling(5).sum()); inst[c + "20"] = inst.groupby("code")[c].transform(lambda s: s.rolling(20).sum())
    D = D.merge(inst[["code", "date", "foreign5", "foreign20", "trust5", "trust20", "total5"]], on=["code", "date"], how="left")
    D = D.sort_values(["code", "date"])
    vol20 = D.groupby("code")["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean()) / 1000
    for c in ("foreign5", "foreign20", "trust5", "trust20", "total5"):
        D[c + "_r"] = D[c] / vol20; XF.append(c + "_r")
if len(rev):
    rev["date"] = rev["date"].astype(str); rev = rev.sort_values(["code", "date"])
    rev["yoy"] = rev.groupby("code")["revenue"].pct_change(12) * 100
    rev["yoy3"] = rev.groupby("code")["revenue"].transform(lambda s: s.rolling(3).sum().pct_change(12) * 100)
    rev["avail"] = (pd.to_datetime(rev["date"]) + pd.Timedelta(days=10)).dt.strftime("%Y-%m-%d")   # FinMind date = 次月 1 日；10 日前公布
    D = D.sort_values("date")
    D = pd.merge_asof(D, rev[["code", "avail", "yoy", "yoy3"]].dropna(subset=["avail"]).sort_values("avail").rename(columns={"avail": "date"}), on="date", by="code", direction="backward")
    XF += ["yoy", "yoy3"]
D = D.sort_values(["date", "code"]).reset_index(drop=True)
mom = np.where(D["pct"] >= 0, np.minimum(D["pct"], 7) * 1.6, D["pct"] * 0.6)
D["screen"] = mom + D["lval"] * 2 + np.minimum(D["amp"], 8) * .8 + np.where(D["pct"] < D["m_ret1"], np.minimum(D["m_ret1"] - D["pct"], 5) * 1.2, 0) + np.where((D["pct"] >= 3) & (D["pct"] < 9), (D["pct"] - 3) * .6, 0)
D["pool"] = D.groupby("date")["screen"].rank(ascending=False) <= 40
say("rows", len(D), "extra feats", XF, "missing", {f: round(float(D[f].isna().mean()), 3) for f in XF})


def wf(feats):
    p = pd.Series(np.nan, index=D.index)
    for y in range(2022, int(D["year"].max()) + 1):
        tr = (D["year"] < y) & (D["date"] < f"{y - 1}-12-01"); te = D["year"] == y
        m = lgb.LGBMClassifier(**T.PARAMS).fit(D.loc[tr, feats], D.loc[tr, "hit"]); p[te] = m.predict_proba(D.loc[te, feats])[:, 1]
    return p


def evalA(p, name, q=0.9):
    E = D.assign(p=p)[p.notna() & D["pool"]]
    rows, last = [], {}
    for y in sorted(E["year"].unique()):
        prev = E[E["year"] == y - 1]["p"]; th = float((prev if len(prev) else E[E["year"] == y]["p"]).quantile(q))
        for d, g in E[E["year"] == y].groupby("date"):
            n = 0
            for r in g.sort_values("p", ascending=False).itertuples():
                lp = last.get(r.code)
                if lp is not None and (pd.Timestamp(d) - pd.Timestamp(lp)).days < 30: continue
                hi = r.p >= th
                rows.append((y, "A" if hi and r.m_bias20 < 0 else "B+" if hi else "B", r.hit, r.fin21)); last[r.code] = d; n += 1
                if n >= 6: break
    x = pd.DataFrame(rows, columns=["year", "tier", "hit", "fin"])
    res = {t: {"n": int((x["tier"] == t).sum()), "hit": round(float(x[x["tier"] == t]["hit"].mean()), 3) if (x["tier"] == t).any() else None,
               "fin": round(float(x[x["tier"] == t]["fin"].mean()), 2) if (x["tier"] == t).any() else None} for t in ("A", "B+", "B")}
    res["A_by_year"] = {int(y): round(float(g["hit"].mean()), 3) for y, g in x[x["tier"] == "A"].groupby("year")}
    say(f"  {name:<34} A {res['A']} | B+ {res['B+']} | B {res['B']} | A 逐年 {res['A_by_year']}")
    return res


out = {"n_stocks": len(codes), "rows": int(len(D)), "extra": XF, "results": {}}
out["results"]["V0"] = evalA(wf(T.FEATS), "V0 現行")
if XF:
    out["results"]["V3"] = evalA(wf(T.FEATS + XF), "V3 +法人/營收")
    inst_f = [f for f in XF if f.endswith("_r")]; rev_f = [f for f in XF if f.startswith("yoy")]
    if inst_f: out["results"]["V3i"] = evalA(wf(T.FEATS + inst_f), "V3i +法人")
    if rev_f: out["results"]["V3r"] = evalA(wf(T.FEATS + rev_f), "V3r +營收")
    for f in XF:
        q = pd.qcut(D[f].rank(method="first"), 5, labels=False); out.setdefault("single", {})[f] = D.groupby(q)["hit"].mean().round(3).tolist()
    say("  單因子五分位命中:", out.get("single"))
    imp = lgb.LGBMClassifier(**T.PARAMS).fit(D[T.FEATS + XF], D["hit"]).booster_.feature_importance("gain"); imp = imp / imp.sum()
    out["importance"] = sorted(((f, round(float(w), 3)) for f, w in zip(T.FEATS + XF, imp)), key=lambda z: -z[1])[:15]
    say("  重要度:", out["importance"])
out["log"] = log; out["seconds"] = round(time.time() - t0)
(OUT / "treasure_research.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
say("done", out["seconds"], "s")
