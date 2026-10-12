"""小時模型資料來源研究 (正式學習器 LightGBM，不寫模型、不發布)：python tools/intraday_research.py
GitHub Actions workflow_dispatch mode=research_intraday 執行，結果上傳 artifact (research_out/intraday_research.{json,md})。

比較 (同一套逐季 walk-forward、到各目標的盤中 rank IC / 方向命中)：
  yahoo   — 現行舊來源 Yahoo ^TWII 60m (約 2 年，開盤 = 09:00:05 失真第一筆)
  twse    — TWSE 每 5 秒指數 2016~ (data/intraday_marks.json，開盤 09:05)
  early   — TWSE 早盤模型 (09:15 / 09:30 另訓)
共同月份 (yahoo 有測試的月份) 上做配對 bootstrap ΔIC；twse / early 另列 2018~ 逐年 IC。
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
    res = {"days": {"yahoo": int(hy["date"].nunique()), "twse": int(ht["date"].nunique())}, "targets": {}}
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
    (OUT / "intraday_research.md").write_text("\n".join(L), encoding="utf-8")
    print("done", res["sec"], "s")


if __name__ == "__main__":
    main()
