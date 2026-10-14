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

separate_training 實驗 (2026-10-14，結果鍵 "separate_training")：盤中時間點與 pre 列分開訓練。本機研究 (numpy GBDT 代理、無 night_chg_pct)：
正式混訓形式的 to_13:30 盤中 rank-IC .068，只用 10:00~13:00 盤中列訓練 .140 (+.072)，但零參數基準 ret_open (開盤→現價%，不訓練) .136，
分開訓練對它只 +.004 (CI 含 0)。此處以正式 LightGBM、正式 FEATURES (含 night_chg_pct，FinMind 2023~) 在 twse 來源 2018~ 逐季下比較三組：
  (a) mixed      — 正式混訓形式 (pre + 盤中列同一模型；= 上面 twse 基準列，共用資料)
  (b) intra_only — 同 PARAMS / FEATURES，只用 INTRA_MARKS (10:00~13:00) 列訓練 (_walk_forward_months train_marks)
  (c) ret_open   — 零參數：ret_open 值直接當預測分數 (_walk_forward_months score_feature)
對每個有盤中列的目標 (11:00~13:30) 分 10:00/11:00/12:00/13:00 時點列 IC、(b)−(a) 與 (b)−(c) 的日配對 bootstrap ΔIC 95% CI、逐年正年數、命中/基準。
採用門檻 (每格 目標@時點)：(b)−(a) CI 下界 > 0 且逐年 ≥ 2/3 為正，且 (b)−(c) CI 下界 > 0 → adopt_separate (值得改成分開訓練)；
只過前者 → ret_open_only (用 ret_open 單變量即可，不值得改模型)；其他 → no_gain。
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


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """平均名次 (同 pandas rank(method='average'))。"""
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt + 1
    return (start + (cnt - 1) / 2.0)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _avg_rank(x), _avg_rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000) -> dict:
    """b − a 的 rank IC 差 (同日配對，日為單位 bootstrap)。
    2026-10-14：內層改 numpy (平均名次 + Pearson = pandas rank().corr())，抽樣序列與舊版相同 → 數字不變 (含同值名次)、約 2.5 倍快。"""
    out = {}
    for m in sorted(set(a["mark"]) & set(b["mark"])):
        if m == "pre":
            continue
        j = a[a.mark == m][["date", "pred", "actual"]].merge(b[b.mark == m][["date", "pred", "actual"]], on="date", suffixes=("_a", "_b"))
        if len(j) < 60:
            continue
        idx = np.arange(len(j))
        A = j[["pred_a", "actual_a", "pred_b", "actual_b"]].to_numpy(dtype=float)
        d = []
        for _ in range(n_boot):
            s = A[rng.choice(idx, len(idx))]
            d.append(_spearman(s[:, 2], s[:, 3]) - _spearman(s[:, 0], s[:, 1]))
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


SEP_FEATURE = "ret_open"


def sep_cell_verdict(d_mixed: dict, y_mixed: dict, d_ref: dict | None) -> dict:
    """單格 (目標@時點) 判定：(b)−(a) CI 下界 > 0 且逐年 ≥ 2/3 為正 = 分開訓練贏混訓；(b)−(c) CI 下界 > 0 = 贏 ret_open 零參數。"""
    beats_mixed = bool(d_mixed["ci95"][0] > 0 and y_mixed.get("years") and y_mixed["pos"] >= 2 / 3 * y_mixed["years"])
    beats_ref = bool(d_ref and d_ref["ci95"][0] > 0)
    return {"sep_beats_mixed": beats_mixed, "sep_beats_ret_open": beats_ref,
            "verdict": "adopt_separate" if (beats_mixed and beats_ref) else "ret_open_only" if beats_mixed else "no_gain"}


def sep_block(mixed: pd.DataFrame, sep: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """分開訓練實驗單一目標：(a) mixed 正式混訓、(b) sep 只用盤中列訓練、(c) ref 零參數 ret_open。
    摘要 / 盤中 IC 平均、(b)−(a)、(b)−(c)、(c)−(a) 的配對 ΔIC 與逐年 ΔIC、每個時點的判定。"""
    r3 = lambda v: round(v, 3) if v is not None else None  # noqa: E731
    out = {"summary": {"mixed": summarize(mixed), "intra_only": summarize(sep), SEP_FEATURE: summarize(ref)},
           "intra_ic": {"mixed": r3(I._intra_ic(mixed)), "intra_only": r3(I._intra_ic(sep)), SEP_FEATURE: r3(I._intra_ic(ref))},
           "delta_vs_mixed": paired(mixed, sep), "yearly_vs_mixed": yearly_delta(mixed, sep),
           "delta_vs_ret_open": paired(ref, sep), "yearly_vs_ret_open": yearly_delta(ref, sep),
           "delta_ret_open_vs_mixed": paired(mixed, ref), "yearly_ret_open_vs_mixed": yearly_delta(mixed, ref)}
    out["cells"] = {m: sep_cell_verdict(dv, out["yearly_vs_mixed"].get(m, {}), out["delta_vs_ret_open"].get(m))
                    for m, dv in out["delta_vs_mixed"].items()}
    return out


SEP_TEXT = {"adopt_separate": "分開訓練同時顯著贏混訓與 ret_open 零參數 → 值得把盤中模型改成只用盤中列訓練 (pre 另訓)，並重做校準表",
            "ret_open_only": "分開訓練贏混訓，但贏不了 ret_open 單變量 → 盤中段用 ret_open 單變量即可，不值得改模型",
            "no_gain": "分開訓練對混訓沒有穩定 (CI 下界 > 0 且逐年 ≥ 2/3) 的 ΔIC → 維持現狀"}


def sep_verdict(targets: dict) -> dict:
    """所有格的總結：各判定格數、整體 verdict (≥ 2/3 格 adopt → adopt_separate；≥ 2/3 格 adopt 或 ret_open_only → ret_open_only；否則 no_gain)。"""
    cells = [(tm, m, c) for tm, g in targets.items() for m, c in g["cells"].items()]
    n = len(cells)
    by = {k: [f"{tm}@{m}" for tm, m, c in cells if c["verdict"] == k] for k in ("adopt_separate", "ret_open_only", "no_gain")}
    overall = ("adopt_separate" if n and len(by["adopt_separate"]) >= 2 / 3 * n else
               "ret_open_only" if n and len(by["adopt_separate"]) + len(by["ret_open_only"]) >= 2 / 3 * n else "no_gain")
    dm = [g["delta_vs_mixed"][m]["d_ic"] for tm, g in targets.items() for m in g["delta_vs_mixed"]]
    dr = [g["delta_vs_ret_open"][m]["d_ic"] for tm, g in targets.items() for m in g["delta_vs_ret_open"]]
    return {"cells": n, **by, "overall": overall, "text": SEP_TEXT[overall],
            "mean_d_ic_vs_mixed": round(float(np.mean(dm)), 4) if dm else None,
            "mean_d_ic_vs_ret_open": round(float(np.mean(dr)), 4) if dr else None}


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
                    "twse_prev_og_nonnull": int(hto.dropna(subset=["prev_og"])["date"].nunique()),
                    "twse_night_chg_nonnull": int(ht.dropna(subset=["night_chg_pct"])["date"].nunique())},
           "targets": {}, "prev_og": {"features": {"og": F_OG, "og_oc": I.FEATURES_OG}, "wf_from": I.WF_FROM, "targets": {}},
           # 分開訓練實驗：正式 FEATURES (含 night_chg_pct)、PARAMS，twse 來源 2018~ 逐季；(a) 共用上面的 twse 基準 ot
           "separate_training": {"features": I.FEATURES, "params": I.PARAMS, "wf_from": I.WF_FROM, "train_marks": I.INTRA_MARKS,
                                 "score_feature": SEP_FEATURE, "groups": {"mixed": "正式混訓 (pre + 盤中列)", "intra_only": "只用 10:00~13:00 盤中列訓練",
                                                                           SEP_FEATURE: "零參數：ret_open 值當分數"}, "targets": {}}}
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
        # separate_training：(a) = ot (正式混訓，共用)；(b) 只用盤中列訓練 (多一次 LightGBM walk-forward)；(c) ret_open 零參數 (不訓練)
        if any(m < tm for m in I.INTRA_MARKS):   # 到 10:00 沒有盤中預測列 → 略過
            osep = I._walk_forward_months(ht, tgt, from_ym=I.WF_FROM, quarterly=True, train_marks=I.INTRA_MARKS)
            oref = I._walk_forward_months(ht, tgt, from_ym=I.WF_FROM, quarterly=True, train_marks=I.INTRA_MARKS, score_feature=SEP_FEATURE)
            s = sep_block(ot, osep, oref)
            res["separate_training"]["targets"][tm] = s
            print(f"  separate {tm}: intra IC mixed {s['intra_ic']['mixed']} / intra_only {s['intra_ic']['intra_only']} / ret_open {s['intra_ic'][SEP_FEATURE]}", flush=True)
            for m, c in s["cells"].items():
                d1, d2, y1 = s["delta_vs_mixed"][m], s["delta_vs_ret_open"].get(m, {}), s["yearly_vs_mixed"].get(m, {})
                print(f"    {m}: (b)−(a) {d1['d_ic']} {d1['ci95']} 年正 {y1.get('pos')}/{y1.get('years')}; (b)−(c) {d2.get('d_ic')} {d2.get('ci95')} → {c['verdict']}", flush=True)
    res["separate_training"]["verdict"] = sep_verdict(res["separate_training"]["targets"])
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
    # separate_training 段落
    S, sv = res["separate_training"], res["separate_training"]["verdict"]
    L.append(f"\n# 盤中時間點與 pre 列分開訓練 separate_training (TWSE {I.WF_FROM}~ 逐季 walk-forward；正式 FEATURES 含 night_chg_pct，非空日 {res['days']['twse_night_chg_nonnull']})\n")
    L.append("(a) mixed = 正式混訓 (pre + 盤中列同一模型)；(b) intra_only = 只用 10:00~13:00 盤中列訓練；(c) ret_open = 零參數 (開盤至今% 直接當分數)。\n")
    L.append("採用門檻 (每格 目標@時點)：(b)−(a) ΔIC 95% CI 下界 > 0 且逐年 ≥ 2/3 為正，**且** (b)−(c) CI 下界 > 0 → adopt_separate；只過前者 → ret_open_only；其他 → no_gain。\n")
    L.append(f"**整體判定：{sv['overall']}** — {sv['text']}  (格數 {sv['cells']}：adopt {len(sv['adopt_separate'])} / ret_open_only {len(sv['ret_open_only'])} / no_gain {len(sv['no_gain'])}；"
             f"平均 ΔIC (b)−(a) {sv['mean_d_ic_vs_mixed']}、(b)−(c) {sv['mean_d_ic_vs_ret_open']})\n")
    L.append("| 目標 | 盤中 IC mixed | intra_only | ret_open | adopt | ret_open_only | no_gain |\n|---|---|---|---|---|---|---|")
    for tm, s in S["targets"].items():
        vs = [c["verdict"] for c in s["cells"].values()]
        L.append(f"| {tm} | {s['intra_ic']['mixed']} | {s['intra_ic']['intra_only']} | {s['intra_ic'][SEP_FEATURE]} | "
                 f"{vs.count('adopt_separate')} | {vs.count('ret_open_only')} | {vs.count('no_gain')} |")
    for tm, s in S["targets"].items():
        L.append(f"\n## 到 {tm}\n")
        L.append("| 時點 | IC mixed | IC intra_only | IC ret_open | (b)−(a) ΔIC [95%] | 逐年正/年數 | (b)−(c) ΔIC [95%] | 逐年正/年數 | (c)−(a) ΔIC [95%] | 命中 mixed→intra_only / ret_open / 多頭基準 | 判定 |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for m, c in s["cells"].items():
            a0, b0, r0 = s["summary"]["mixed"].get(m, {}), s["summary"]["intra_only"].get(m, {}), s["summary"][SEP_FEATURE].get(m, {})
            d1, y1 = s["delta_vs_mixed"][m], s["yearly_vs_mixed"].get(m, {})
            d2, y2 = s["delta_vs_ret_open"].get(m, {}), s["yearly_vs_ret_open"].get(m, {})
            d3 = s["delta_ret_open_vs_mixed"].get(m, {})
            L.append(f"| {m} | {a0.get('ic', '—')} | {b0.get('ic', '—')} | {r0.get('ic', '—')} | {d1['d_ic']} {d1['ci95']} | {y1.get('pos', '—')}/{y1.get('years', '—')} "
                     f"| {d2.get('d_ic', '—')} {d2.get('ci95', '')} | {y2.get('pos', '—')}/{y2.get('years', '—')} | {d3.get('d_ic', '—')} {d3.get('ci95', '')} "
                     f"| {a0.get('hit', '—')}→{b0.get('hit', '—')} / {r0.get('hit', '—')} / {b0.get('base', '—')} | {c['verdict']} |")
    (OUT / "intraday_research.md").write_text("\n".join(L), encoding="utf-8")
    print("prev_og verdict", json.dumps(verdict, ensure_ascii=False), flush=True)
    print("separate_training verdict", json.dumps({k: v for k, v in sv.items() if k != "text"}, ensure_ascii=False), flush=True)
    print("done", res["sec"], "s")


if __name__ == "__main__":
    main()
