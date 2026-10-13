"""小時模型資料來源研究 (正式學習器 LightGBM，不寫模型、不發布)：python tools/intraday_research.py
GitHub Actions workflow_dispatch mode=research_intraday 執行，結果上傳 artifact (research_out/intraday_research.{json,md})。

比較 (同一套逐季 walk-forward、到各目標的盤中 rank IC / 方向命中)：
  yahoo   — 現行舊來源 Yahoo ^TWII 60m (約 2 年，開盤 = 09:00:05 失真第一筆)
  twse    — TWSE 每 5 秒指數 2016~ (data/intraday_marks.json，開盤 09:05)
  early   — TWSE 早盤模型 (09:15 / 09:30 另訓)
共同月份 (yahoo 有測試的月份) 上做配對 bootstrap ΔIC；twse / early 另列 2018~ 逐年 IC。

prev_og 實驗 (2026-10-14，結果鍵 "prev_og")：twse / early 基準 vs 加入前一交易日隔夜跳空 prev_og (+ 前一日開盤→13:30 prev_oc)
(I.OG_FEATS / I.FEATURES_OG，正式 FEATURES 不變)，同一套 2018~ 逐季 walk-forward，對每個目標做配對 bootstrap ΔIC (日為單位)、
逐年 IC 與正年數、命中/基準。採用門檻：ΔIC 95% CI 不含 0 且逐年 ≥ 2/3 為正，才把 prev_og 加進正式 FEATURES。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.analysis import backtest  # noqa: E402
from chip.predict import intraday as I  # noqa: E402
from chip.predict.features import market_matrix  # noqa: E402
from chip.sources import finmind, twse  # noqa: E402

OUT = Path(os.getenv("RESEARCH_OUT", "research_out"))
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def ic(g: pd.DataFrame) -> float:
    return float(g["pred"].rank().corr(g["actual"].rank()))


def summarize(oos: pd.DataFrame) -> dict:
    out = {}
    for m, g in oos.groupby("mark"):
        out[m] = {"n": int(len(g)), "ic": round(ic(g), 3), "hit": round(float(((g.pred > 0) == (g.actual > 0)).mean()) * 100, 1),
                  "base": round(float((g.actual > 0).mean()) * 100, 1)}
    return out


def yearly(oos: pd.DataFrame) -> dict:
    o = oos[oos["mark"] != "pre"].copy()
    o["y"] = o["date"].str[:4]
    return {y: {m: round(ic(gg), 3) for m, gg in g.groupby("mark") if len(gg) >= 30} for y, g in o.groupby("y")}


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000) -> dict:
    """b − a 的 rank IC 差 (同日配對，日為單位 bootstrap)。"""
    out = {}
    for m in sorted(set(a["mark"]) & set(b["mark"])):
        if m == "pre":
            continue
        j = a[a.mark == m][["date", "pred", "actual"]].merge(b[b.mark == m][["date", "pred", "actual"]], on="date", suffixes=("_a", "_b"))
        if len(j) < 60:
            continue
        idx = np.arange(len(j))
        d = []
        for _ in range(n_boot):
            s = j.iloc[rng.choice(idx, len(idx))]
            d.append(s.pred_b.rank().corr(s.actual_b.rank()) - s.pred_a.rank().corr(s.actual_a.rank()))
        d = np.array(d)
        out[m] = {"n": int(len(j)), "d_ic": round(float(d.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])],
                  "p_gt0": round(float((d > 0).mean()), 3)}
    return out


def yearly_delta(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """逐年 (b − a) rank IC，每個盤中 mark 列正年數 / 年數與逐年差。"""
    ya, yb = yearly(a), yearly(b)
    out = {}
    for m in sorted(set(a["mark"]) & set(b["mark"]) - {"pre"}):
        by = {y: round(yb[y][m] - ya[y][m], 3) for y in sorted(set(ya) & set(yb)) if m in ya[y] and m in yb[y]}
        out[m] = {"pos": int(sum(v > 0 for v in by.values())), "years": len(by), "by_year": by}
    return out


def og_block(base: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> dict:
    """基準 vs 各加特徵版本：摘要、配對 ΔIC、逐年 ΔIC。"""
    out = {"base": summarize(base)}
    for k, o in variants.items():
        out[k] = {"summary": summarize(o), "delta": paired(base, o), "yearly_delta": yearly_delta(base, o),
                  "intra_ic": I._intra_ic(o), "intra_ic_base": I._intra_ic(base)}
    return out


def main() -> None:
    t0 = time.time()
    mat = market_matrix(backtest.load_long("2010-01-01"))
    night = finmind.tx_night_history("2023-01-01")
    ctx = {"daily": I._daily_features(mat), "night_map": dict(zip(night["date"], night["night_chg_pct"])) if not night.empty else {}}
    tw = twse.intraday_marks()
    hy = I.build_history("yahoo", _ctx=ctx)
    ht = I.build_history("twse", marks=tw, _ctx=ctx)
    he = I.build_history("twse", marks_list=I.EARLY_MARKS, marks=tw, _ctx=ctx)
    he = he[he["mark"] != "pre"].reset_index(drop=True)
    # prev_og 實驗：同一批列再帶 prev_og / prev_oc 欄 (基準列與日期完全相同 → 可配對)
    hto = I.build_history("twse", marks=tw, _ctx=ctx, extra_feats=tuple(I.OG_FEATS))
    heo = I.build_history("twse", marks_list=I.EARLY_MARKS, marks=tw, _ctx=ctx, extra_feats=tuple(I.OG_FEATS))
    heo = heo[heo["mark"] != "pre"].reset_index(drop=True)
    F_OG = I.FEATURES + ["prev_og"]
    res = {"days": {"yahoo": int(hy["date"].nunique()), "twse": int(ht["date"].nunique()),
                    "twse_prev_og_nonnull": int(hto.dropna(subset=["prev_og"])["date"].nunique())},
           "targets": {}, "prev_og": {"features": {"og": F_OG, "og_oc": I.FEATURES_OG}, "wf_from": I.WF_FROM, "targets": {}}}
    for tm in I.TARGET_MARKS:
        tgt = f"to_{tm}"
        oy = I._walk_forward_months(hy, tgt, quarterly=True)
        ot = I._walk_forward_months(ht, tgt, from_ym=I.WF_FROM, quarterly=True)
        oe = I._walk_forward_months(he, tgt, from_ym=I.WF_FROM, quarterly=True)
        common = set(oy["year"])
        r = {"common_months": len(common),
             "yahoo": summarize(oy), "twse_common": summarize(ot[ot["year"].isin(common)]), "twse_2018": summarize(ot),
             "early_common": summarize(oe[oe["year"].isin(common)]), "early_2018": summarize(oe),
             "delta_twse_vs_yahoo": paired(oy, ot[ot["year"].isin(common)]),
             "twse_yearly": yearly(ot), "early_yearly": yearly(oe),
             "intra_ic_common": {"yahoo": I._intra_ic(oy, common), "twse": I._intra_ic(ot, common)}}
        res["targets"][tm] = r
        print(tm, json.dumps(r["intra_ic_common"]), json.dumps(r["delta_twse_vs_yahoo"]), flush=True)
        # prev_og：twse 基準 vs +prev_og / +prev_og+prev_oc；early 基準 vs +prev_og (走同一套 2018~ 逐季 walk-forward)
        oto = I._walk_forward_months(hto, tgt, from_ym=I.WF_FROM, quarterly=True, features=F_OG)
        otoc = I._walk_forward_months(hto, tgt, from_ym=I.WF_FROM, quarterly=True, features=I.FEATURES_OG)
        oeo = I._walk_forward_months(heo, tgt, from_ym=I.WF_FROM, quarterly=True, features=F_OG)
        g = {"twse": og_block(ot, {"og": oto, "og_oc": otoc}), "early": og_block(oe, {"og": oeo})}
        res["prev_og"]["targets"][tm] = g
        print(f"  prev_og {tm}: twse intra IC {g['twse']['og']['intra_ic_base']} → +og {g['twse']['og']['intra_ic']} / +og+oc {g['twse']['og_oc']['intra_ic']};"
              f" early {g['early']['og']['intra_ic_base']} → +og {g['early']['og']['intra_ic']}", flush=True)
        for grp, var in (("twse", "og"), ("twse", "og_oc"), ("early", "og")):
            for m, dv in g[grp][var]["delta"].items():
                yd = g[grp][var]["yearly_delta"].get(m, {})
                print(f"    {grp}+{var} {m}: ΔIC {dv['d_ic']} {dv['ci95']} P>0 {dv['p_gt0']} 年正 {yd.get('pos')}/{yd.get('years')}", flush=True)
    # prev_og 總結：各目標盤中 ΔIC 平均、CI 不含 0 且逐年 ≥ 2/3 為正的 mark 數 (採用門檻)
    verdict = {}
    for grp, var in (("twse", "og"), ("twse", "og_oc"), ("early", "og")):
        cells = [(tm, m, g[grp][var]["delta"][m], g[grp][var]["yearly_delta"].get(m, {}))
                 for tm, g in res["prev_og"]["targets"].items() for m in g[grp][var]["delta"]]
        passed = [f"{tm}@{m}" for tm, m, dv, yd in cells if dv["ci95"][0] > 0 and yd.get("years") and yd["pos"] >= 2 / 3 * yd["years"]]
        verdict[f"{grp}+{var}"] = {"mean_d_ic": round(float(np.mean([c[2]["d_ic"] for c in cells])), 4) if cells else None,
                                  "cells": len(cells), "pass_cells": len(passed), "passed": passed}
    res["prev_og"]["verdict"] = verdict
    res["sec"] = round(time.time() - t0)
    (OUT / "intraday_research.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # 精簡 markdown 報告
    L = [f"# 小時模型資料來源研究 (LightGBM)\n\n樣本日：Yahoo {res['days']['yahoo']} / TWSE {res['days']['twse']}；耗時 {res['sec']} 秒\n"]
    for tm, r in res["targets"].items():
        L.append(f"\n## 到 {tm}  (共同測試月份 {r['common_months']})\n")
        L.append("| 時點 | Yahoo IC | TWSE IC (共同) | ΔIC [95%] | P(Δ>0) | TWSE IC (2018~) | 命中/基準 (2018~) |\n|---|---|---|---|---|---|---|")
        for m in sorted(set(r["yahoo"]) | set(r["twse_2018"])):
            if m == "pre":
                continue
            y, c, l, d = r["yahoo"].get(m, {}), r["twse_common"].get(m, {}), r["twse_2018"].get(m, {}), r["delta_twse_vs_yahoo"].get(m, {})
            L.append(f"| {m} | {y.get('ic', '—')} | {c.get('ic', '—')} | {d.get('d_ic', '—')} {d.get('ci95', '')} | {d.get('p_gt0', '—')} | {l.get('ic', '—')} | {l.get('hit', '—')}/{l.get('base', '—')} |")
        for m, v in r["early_2018"].items():
            L.append(f"| 早盤 {m} | — | {r['early_common'].get(m, {}).get('ic', '—')} | — | — | {v['ic']} | {v['hit']}/{v['base']} |")
    # prev_og 段落
    L.append(f"\n# 前一日隔夜跳空 prev_og 實驗 (TWSE {I.WF_FROM}~ 逐季 walk-forward；prev_og 非空日 {res['days']['twse_prev_og_nonnull']})\n")
    L.append("採用門檻：ΔIC 95% CI 不含 0 (下界 > 0) 且逐年 ΔIC ≥ 2/3 年為正。\n")
    L.append("| 組 | 平均 ΔIC | 通過格數/總格數 | 通過 (目標@時點) |\n|---|---|---|---|")
    for k, v in verdict.items():
        L.append(f"| {k} | {v['mean_d_ic']} | {v['pass_cells']}/{v['cells']} | {', '.join(v['passed']) or '—'} |")
    for tm, g in res["prev_og"]["targets"].items():
        L.append(f"\n## 到 {tm}\n")
        L.append("| 組 | 時點 | 基準 IC | +特徵 IC | ΔIC [95%] | P(Δ>0) | 逐年正/年數 | 命中 基準→+特徵 / 多頭基準 |\n|---|---|---|---|---|---|---|---|")
        for grp, var in (("twse", "og"), ("twse", "og_oc"), ("early", "og")):
            blk = g[grp][var]
            for m, dv in blk["delta"].items():
                b0, b1, yd = g[grp]["base"].get(m, {}), blk["summary"].get(m, {}), blk["yearly_delta"].get(m, {})
                L.append(f"| {grp}+{var} | {m} | {b0.get('ic', '—')} | {b1.get('ic', '—')} | {dv['d_ic']} {dv['ci95']} | {dv['p_gt0']} "
                         f"| {yd.get('pos', '—')}/{yd.get('years', '—')} | {b0.get('hit', '—')}→{b1.get('hit', '—')} / {b1.get('base', '—')} |")
    (OUT / "intraday_research.md").write_text("\n".join(L), encoding="utf-8")
    print("prev_og verdict", json.dumps(verdict, ensure_ascii=False), flush=True)
    print("done", res["sec"], "s")


if __name__ == "__main__":
    main()
