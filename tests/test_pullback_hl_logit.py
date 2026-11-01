"""2026-11-02 盤中「今日低 / 高點已出現」機率改平滑 logistic (pullback._hl_logit，存 hour_low_table.logit / hour_high_table.logit)：

- _hl_rows：連續值 rbz / obz (σ 單位) 與分箱 rb / ob 同算式、帶日期 d
- _hl_logit：IRLS 解 = 梯度為 0 (ridge λ=1、截距不罰)；各時間點 c / rb[4] / ob[4] 與設計矩陣一致；單調上限 (p 隨 rb 不降、隨 ob 不升)
- _hl_table(marks, side, marks_all)：格 / 分位只用近 3 年 marks (與 11-02 前相同)，logit 用 marks_all；marks_all=None → 無 logit
- hl_logit_p：缺值 (None / NaN / 負) → None (App hlLogitP 同規則；null 不可被當成 0)

    python tests/test_pullback_hl_logit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import pullback as P  # noqa: E402

MK = ("09:15", "09:30", "10:00", "11:00", "12:00", "13:00", "13:30")


def synth(n_days=1500, seed=4):
    """隨機漫步日內路徑 (每時間點 12 步) → marks (09:05 開盤、累積高低) 與 σ 表。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=n_days).strftime("%Y-%m-%d")
    marks, sig = {}, []
    for d in days:
        s = float(rng.uniform(0.7, 1.6)); prev = 100.0; op = prev * (1 + rng.normal(0, 0.4 * s) / 100)
        path = op * np.exp(np.cumsum(rng.normal(0, s / 100 / np.sqrt(len(MK) * 12), len(MK) * 12)))
        r = {"prev_close": prev, "open": op, "hi": {}, "lo": {}}
        for i, mk in enumerate(MK):
            seg = np.r_[op, path[: (i + 1) * 12]]
            r[mk], r["lo"][mk], r["hi"][mk] = float(seg[-1]), float(seg.min()), float(seg.max())
        marks[d] = r; sig.append(s)
    m = pd.DataFrame({"date": list(days), "sigma_range": np.r_[sig[1:], sig[-1]]})   # 前一日 σ = 當日路徑的 σ (shift(1) 之後)
    return marks, m


def main():
    n = 0
    marks, m = synth()
    rows = P._hl_rows(m, marks, "lo")
    r0 = rows.iloc[123]; rec = marks[r0["d"]]; s = dict(zip(m["date"], m["sigma_range"].shift(1)))[r0["d"]]
    rbz = (rec[r0["mark"]] / rec["lo"][r0["mark"]] - 1) * 100 / s; obz = (rec["hi"][r0["mark"]] / rec[r0["mark"]] - 1) * 100 / s
    assert abs(r0["rbz"] - rbz) < 1e-12 and abs(r0["obz"] - obz) < 1e-12 and r0["rb"] == P._rb_bin(rec[r0["mark"]], rec["lo"][r0["mark"]], s) and r0["ob"] == P._ob_bin(obz), r0
    n += 1

    # IRLS 解：梯度 X'(y−p) − λb (截距不罰) ≈ 0；JSON 係數 = 設計矩陣 (同一列重算 p 一致)
    lg = P._hl_logit(rows)
    r = rows.dropna(subset=["rbz", "obz"]); r = r[r["mark"].isin(P.HL_MARKS)]
    X = P._hl_design(r["mark"].to_numpy(), r["rbz"].to_numpy(float), r["obz"].to_numpy(float))
    y = r["done"].to_numpy(float)
    # 由 JSON 還原成設計矩陣係數向量
    nm = len(P.HL_MARKS) - 1
    b = np.zeros(X.shape[1]); b[0] = lg["marks"][P.HL_MARKS[0]]["c"]
    for j, mk in enumerate(P.HL_MARKS[1:], 1):
        b[j] = lg["marks"][mk]["c"] - b[0]
    i_rb, i_ob, i_int = 1 + nm, 1 + nm + 4, 1 + nm + 8
    b[i_rb:i_rb + 4] = lg["marks"][P.HL_MARKS[0]]["rb"]; b[i_ob:i_ob + 4] = lg["marks"][P.HL_MARKS[0]]["ob"]
    for j, mk in enumerate(P.HL_MARKS[1:], 1):
        b[i_int + 2 * (j - 1)] = lg["marks"][mk]["rb"][0] - b[i_rb]; b[i_int + 2 * (j - 1) + 1] = lg["marks"][mk]["ob"][0] - b[i_ob]
    p = 1 / (1 + np.exp(-(X @ b)))
    # 獨立解 (梯度下降到收斂，ridge λ=1、截距不罰) → 與 JSON 係數一致 (JSON 四位小數)
    lam = np.r_[0.0, np.ones(X.shape[1] - 1)]
    bb = np.zeros(X.shape[1])
    for _ in range(200):
        pp = 1 / (1 + np.exp(-(X @ bb)))
        H = X.T @ (X * (pp * (1 - pp))[:, None]) + np.diag(lam)
        bb = bb + np.linalg.lstsq(H, X.T @ (y - pp) - lam * bb, rcond=None)[0]
    assert np.max(np.abs(np.round(bb, 4) - b)) < 2e-4, np.max(np.abs(np.round(bb, 4) - b))
    ix = np.arange(0, len(r), 97)
    pj = np.array([P.hl_logit_p(lg, mk, rz, oz) for mk, rz, oz in zip(r["mark"].to_numpy()[ix], r["rbz"].to_numpy(float)[ix], r["obz"].to_numpy(float)[ix])])
    capped = np.array([min(rz, 3.0, lg["marks"][mk]["rb_max"]) == min(rz, 3.0) and min(oz, 3.0, lg["marks"][mk]["ob_max"]) == min(oz, 3.0)
                       for mk, rz, oz in zip(r["mark"].to_numpy()[ix], r["rbz"].to_numpy(float)[ix], r["obz"].to_numpy(float)[ix])])
    assert np.max(np.abs(pj[capped] - p[ix][capped])) < 5e-3, np.max(np.abs(pj[capped] - p[ix][capped]))
    assert lg["n_days"] == len(marks) - 1 and lg["fit_from"] == sorted(marks)[1], (lg["n_days"], lg["fit_from"])   # 第一天沒有前一日 σ
    n += 3

    # 單調：p 隨 rb 不降、隨 ob 不升 (含上限之外)
    for mk in P.HL_MARKS:
        ps_r = [P.hl_logit_p(lg, mk, x, 0.4) for x in np.linspace(0, 4, 81)]
        ps_o = [P.hl_logit_p(lg, mk, 0.3, x) for x in np.linspace(0, 4, 81)]
        assert all(b_ >= a_ - 1e-12 for a_, b_ in zip(ps_r, ps_r[1:])) and all(b_ <= a_ + 1e-12 for a_, b_ in zip(ps_o, ps_o[1:])), mk
    n += 1

    # rb / ob 截在 3σ (訓練設計與 App 算式同)
    one = np.array(["10:00"])
    assert np.allclose(P._hl_design(one, np.array([5.0]), np.array([4.0])), P._hl_design(one, np.array([3.0]), np.array([3.0])))
    assert P.hl_logit_p(lg, "11:00", 5.0, 4.0) == P.hl_logit_p(lg, "11:00", 3.0, 3.0)
    n += 1

    # 缺值 → None (App：null 不可當 0 → 退回表)
    assert all(P.hl_logit_p(lg, "10:00", a_, b_) is None for a_, b_ in ((None, 0.5), (0.3, None), (float("nan"), 0.5), (0.3, -0.1))) and P.hl_logit_p(lg, "09:15", 0.3, 0.5) is None
    n += 1

    # _hl_table：格只用近 3 年視窗；logit 用全部；marks_all=None → 無 logit 且格與有 logit 時相同
    win = dict(list(sorted(marks.items()))[-700:])
    t1 = P._hour_low_table(m, win, marks)
    t0 = P._hour_low_table(m, win)
    assert "logit" in t1 and "logit" not in t0 and all(t1[k] == t0[k] for k in ("cells", "marks", "cells_rb", "marks_rb", "marks_rbob"))
    assert t1["n_days"] == 700 and t1["logit"]["n_days"] == len(marks) - 1
    th = P._hour_high_table(m, win, marks)
    assert th["logit"]["marks"]["13:00"]["c"] != t1["logit"]["marks"]["13:00"]["c"]
    n += 2
    print(f"test_pullback_hl_logit: {n} checks passed")


if __name__ == "__main__":
    main()
