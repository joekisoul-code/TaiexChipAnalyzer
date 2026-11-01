"""2026-11-02 precheck 回補 / 守住 / 續走機率改平滑 logistic + build 文字格式化崩潰修正：

- train：gap.smooth.act (實際表，β=0、半衰期 5 年、全部日子) / gap.smooth.est (估計表，β=0.5、不加權、只用估計非平盤日)，各 3 目標 × 6 係數
- _irls 與獨立的梯度下降解一致 (ridge、截距不罰)
- build 盤中 (+0.8% 實際開盤)：stats.p_fill = round(sigmoid(X·coef), 3)，原格值在 p_cell；stats_actual 維持格值
- build 開盤前估計平盤 (±0.15%)：格值不變、無 p_smooth；β>0 但 σ 無效 → 格值不變
- 舊 precheck.json 無 smooth → 與改版前相同
- 格的 dip_fail_q50 = None (守住率高的格) 不再讓 build 崩潰 (11-02 前整個 precheck 消失)

    python tests/test_precheck_smooth.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import model as M  # noqa: E402
from chip.predict import precheck as PC  # noqa: E402
from chip.predict import range_levels as RL  # noqa: E402
from chip.predict import short_term as ST  # noqa: E402


def market(seed=5):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2014-01-02", "2026-10-30"); n = len(days)
    night = rng.normal(0, 0.8, n)
    o = np.empty(n); h = np.empty(n); lo = np.empty(n); c = np.empty(n); prev = 10000.0
    for i in range(n):
        gap = 0.5 * night[i] + rng.normal(0, 0.35)
        o[i] = prev * (1 + gap / 100)
        c[i] = o[i] * (1 + (rng.normal(0.02, 0.8) - 0.3 * gap) / 100)     # 跳空越大越容易回補 (桶內有梯度)
        h[i] = max(o[i], c[i]) * (1 + abs(rng.normal(0, 0.4)) / 100); lo[i] = min(o[i], c[i]) * (1 - abs(rng.normal(0, 0.4)) / 100)
        prev = c[i]
    ds = [d.strftime("%Y-%m-%d") for d in days]
    nh = pd.DataFrame({"date": ds, "night_chg_pct": night}); nh = nh[nh["date"] >= "2017-01-03"].reset_index(drop=True)
    return pd.DataFrame({"date": ds, "open": o, "high": h, "low": lo, "close": c}), nh


def sig(z):
    return 1 / (1 + np.exp(-z))


def build(st, sc, live_open=None, night=None):
    orig_load, orig_nf, orig_us = M.load_json, RL._night_final, PC._us_add
    try:
        M.load_json = lambda name, *a, **k: copy.deepcopy(st) if name == "precheck" else None
        RL._night_final = lambda snap_, nd_, scored_: (night, None)
        PC._us_add = lambda *a, **k: None
        prev = float(sc["close"].iloc[-1])
        snap = {"phase": "open", "taiex": {"open": live_open, "prev": prev}} if live_open else {"phase": "closed"}
        return PC.build(sc, snap, {"intraday": True if live_open else None, "next_days": [{"n": 1, "date": "2026-11-02"}]})
    finally:
        M.load_json, RL._night_final, PC._us_add = orig_load, orig_nf, orig_us


def main():
    n = 0
    sc, nh = market()
    orig_nh = ST._night_hist
    try:
        ST._night_hist = lambda *a, **k: nh
        st = PC.train(sc, write=False, verbose=False)
    finally:
        ST._night_hist = orig_nh

    # 1. train 寫出兩個平滑模型
    sm = st["gap"]["smooth"]
    a, e = sm["act"], sm["est"]
    assert a["beta"] == 0.0 and a["hl_years"] == 5.0 and not a["nonflat_only"] and e["beta"] == 0.5 and e["hl_years"] is None and e["nonflat_only"], sm
    assert all(len(m["coef"][t]) == 6 for m in (a, e) for t in ("fill", "hold", "cont")) and a["n"] > e["n"] > 200 and a["zcap"] > 0, sm
    assert a["n"] == len(sc) - 20, "實際表平滑用到全部日子 (前 20 日沒有月線)"
    n += 2

    # 2. _irls = 獨立解：梯度 X'w(y−p) − λb (截距不罰) 在解處為 0
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(800), rng.normal(size=(800, 3))]); y = (rng.random(800) < sig(X @ [0.3, 1.0, -0.5, 0.0])).astype(float); w = rng.uniform(0.2, 1.0, 800)
    b = PC._irls(X, y, w, lam=1.0)
    g = X.T @ (w * (y - sig(X @ b))) - np.r_[0.0, b[1:]]
    assert np.max(np.abs(g)) < 1e-6 and abs(b[1] - 1.0) < 0.3, (g, b)
    n += 1

    # 3. 盤中 +0.8% → 平滑值；原格值在 p_cell；stats_actual 不變；text 用平滑值
    prev = float(sc["close"].iloc[-1]); bull = prev >= float(sc["close"].rolling(20).mean().iloc[-1])
    gp = build(st, sc, live_open=prev * 1.008)["gap"]
    gap = (prev * 1.008 / prev - 1) * 100
    z = min(gap, a["zcap"]); xv = np.array([1, 1, float(bull), float(bull), z, np.sqrt(z)])
    for t, k in (("fill", "p_fill"), ("hold", "p_hold"), ("cont", "p_cont")):
        assert gp["stats"][k] == round(float(sig(xv @ np.array(a["coef"][t]))), 3), (k, gp["stats"])
    key = f"5|{'bull' if bull else 'bear'}"
    assert gp["stats"]["p_cell"] == {k: st["gap"]["cells"][key][k] for k in ("p_fill", "p_hold", "p_cont")} and gp["stats_actual"]["p_fill"] == st["gap"]["cells"][key]["p_fill"]
    assert gp["stats"]["p_smooth"]["beta"] == 0.0 and f"回補跳空機率 {gp['stats']['p_fill']:.0%}" in gp["text"], gp["text"]
    n += 3

    # 4. 開盤前估計平盤 → 格值、無 p_smooth；估計開低 → 估計表模型 (z 用 σ^0.5)
    bu = st["gap"]["night_beta"]["beta_recent"]
    g0 = build(st, sc, night=0.05 / bu)["gap"]
    assert abs(g0["est"]) < 0.15 and "p_smooth" not in g0["stats"] and g0["stats"]["p_fill"] == g0["stats_est"]["p_fill"], g0["stats"]
    g1 = build(st, sc, night=-0.8 / bu)["gap"]
    s_now = float(RL.sigma_series(sc).iloc[-1]); z1 = min(0.8 / s_now ** 0.5, e["zcap"])
    x1 = np.array([1, 0, float(bull), 0, z1, np.sqrt(z1)])
    assert g1["table"] == "est" and g1["stats"]["p_hold"] == round(float(sig(x1 @ np.array(e["coef"]["hold"]))), 3) and g1["stats"]["p_smooth"]["sigma"] == round(s_now, 3), g1["stats"]
    n += 2

    # 5. β>0 且 σ 無效 → 原樣；舊 json 無 smooth → 與改版前相同 (p_* = 格值，無 p_cell)
    cell = {"p_fill": 0.4, "p_hold": 0.7, "p_cont": 0.6}
    assert PC.smooth_cell(cell, e, -0.8, True, 1, None) is cell and PC.smooth_cell(cell, e, -0.8, True, 1, float("nan")) is cell and PC.smooth_cell(cell, None, 0.8, True, 5, 1.0) is cell
    old = copy.deepcopy(st); old["gap"].pop("smooth")
    go = build(old, sc, live_open=prev * 1.008)["gap"]
    assert "p_cell" not in go["stats"] and go["stats"]["p_fill"] == st["gap"]["cells"][key]["p_fill"]
    n += 2

    # 6. 格的 dip_fail_q50 = None (守住率 >0.9) → build 照常產出 (11-02 前 TypeError → 整個 precheck 消失)
    bad = copy.deepcopy(st)
    for c in list(bad["gap"]["cells"].values()) + list(bad["gap"]["cells_est"].values()):
        c["dip_fail_q50"] = None; c.pop("sq", None)
    gb = build(bad, sc, live_open=prev * 1.008)["gap"]
    assert gb and "未守住日中位" not in gb["text"] and "守住日的日內回檔中位" in gb["text"], gb
    gb2 = build(bad, sc, night=-0.8 / bu)["gap"]
    assert gb2 and gb2["table"] == "est"
    n += 2
    print(f"test_precheck_smooth: {n} checks passed")


if __name__ == "__main__":
    main()
