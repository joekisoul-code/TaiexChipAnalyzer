"""預判邏輯 (2026-09-23)：兩組可驗證的「明天會怎麼走」條件統計，皆用 2010~ 日 K，逐年一致性檢查，只顯示通過驗證的結論。

A. 開盤跳空預判 (gap)：跳空幅度 (開盤 vs 前收) × 多空狀態 (前收 vs 月線) → 當日回補機率、續走機率 (收盤高於開盤/低於開盤)、
   日內 (開→收) 平均與中位、收盤仍守住跳空的機率、當日振幅。今日估計跳空：盤中用實際開盤；夜盤收後用 β × 夜盤；否則不預判。
B. 日曆效應 (calendar)：結算日/結算前一日/後一日、月初第一個交易日/月底最後交易日、季底、長假前/後、週一/週五 →
   隔天收盤漲跌的上漲率、平均、n、逐年一致性 (各年同號比例)、t 值；valid = n≥40 且 一致性≥0.65 且 |t|≥2。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
GAP_EDGES = [-1.5, -0.5, -0.15, 0.15, 0.5, 1.5]
GAP_LABELS = ["大幅開低 (≤-1.5%)", "開低 (-1.5~-0.5%)", "小幅開低 (-0.5~-0.15%)", "平盤附近 (±0.15%)", "小幅開高 (0.15~0.5%)", "開高 (0.5~1.5%)", "大幅開高 (>1.5%)"]
EVENTS = {"settle": "期貨結算日", "pre_settle": "結算前一日", "post_settle": "結算後一日", "month_first": "月初第一個交易日", "month_last": "月底最後交易日", "quarter_last": "季底最後交易日",
          "pre_holiday": "長假前 (休市 ≥3 天)", "post_holiday": "長假後第一天", "monday": "週一", "friday": "週五"}


def _third_wed(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 1)
    off = (2 - d.weekday()) % 7
    return d + dt.timedelta(days=off + 14)


def _events_for(dates: list[str]) -> pd.DataFrame:
    """dates = 交易日序列 (升冪，需含前後日以判斷月初/月底/長假)。回傳每個日期的事件旗標。"""
    ds = [dt.date.fromisoformat(str(x)[:10]) for x in dates]
    rows = []
    for i, d in enumerate(ds):
        prev_ = ds[i - 1] if i > 0 else None; nxt = ds[i + 1] if i + 1 < len(ds) else None
        tw = _third_wed(d.year, d.month)
        settle = d == tw or (d > tw and (prev_ is None or prev_ < tw))   # 結算日休市時順延到下一交易日
        r = {"date": d.isoformat(), "settle": settle,
             "pre_settle": bool(nxt and (nxt == _third_wed(nxt.year, nxt.month) or (nxt > _third_wed(nxt.year, nxt.month) and d < _third_wed(nxt.year, nxt.month)))),
             "post_settle": bool(prev_ and (prev_ == _third_wed(prev_.year, prev_.month) or (prev_ > _third_wed(prev_.year, prev_.month) and (i < 2 or ds[i - 2] < _third_wed(prev_.year, prev_.month))))),
             "month_first": bool(prev_ and prev_.month != d.month), "month_last": bool(nxt and nxt.month != d.month),
             "quarter_last": bool(nxt and nxt.month != d.month and d.month in (3, 6, 9, 12)),
             "pre_holiday": bool(nxt and (nxt - d).days >= 4), "post_holiday": bool(prev_ and (d - prev_).days >= 4),
             "monday": d.weekday() == 0, "friday": d.weekday() == 4}
        rows.append(r)
    return pd.DataFrame(rows)


def train(scored: pd.DataFrame, write: bool = True, verbose: bool = True) -> dict:
    d = scored[["date", "open", "high", "low", "close"]].copy().reset_index(drop=True)
    d["date"] = d["date"].astype(str).str[:10]
    for c_ in ("open", "high", "low", "close"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    d["prev"] = d["close"].shift(1); d["ma20"] = d["close"].rolling(20).mean().shift(1)
    d["gap"] = (d["open"] / d["prev"] - 1) * 100
    d["intra"] = (d["close"] / d["open"] - 1) * 100
    d["ret"] = (d["close"] / d["prev"] - 1) * 100
    d["rng"] = (d["high"] - d["low"]) / d["prev"] * 100
    d["year"] = d["date"].str[:4].astype(int)
    d["bull"] = d["prev"] >= d["ma20"]
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "gap": {"edges": GAP_EDGES, "labels": GAP_LABELS, "cells": {}}, "calendar": {}}
    g = d.dropna(subset=["gap", "intra", "ma20"]).copy()
    g["bucket"] = np.searchsorted(GAP_EDGES, g["gap"].values, side="right")
    up = g["gap"] > 0
    g["fill"] = np.where(up, g["low"] <= g["prev"], g["high"] >= g["prev"])
    g["cont"] = np.where(up, g["close"] > g["open"], np.where(g["gap"] < 0, g["close"] < g["open"], g["close"] > g["open"]))
    g["hold"] = np.where(up, g["close"] > g["prev"], np.where(g["gap"] < 0, g["close"] < g["prev"], g["close"] > g["prev"]))
    g["dip"] = (g["low"] / g["open"] - 1) * 100; g["pop"] = (g["high"] / g["open"] - 1) * 100
    for (b, bull), gg in g.groupby(["bucket", "bull"]):
        if len(gg) < 20:
            continue
        yr = gg.groupby("year")["cont"].mean(); yr = yr[gg.groupby("year").size() >= 5]
        cont = float(gg["cont"].mean())
        hy = gg.groupby("year")["hold"].agg(["mean", "size"]); hy = hy[hy["size"] >= 8]
        hd = gg[gg["hold"] == True]  # noqa: E712
        extra_ = {"hold_yr_min": round(float(hy["mean"].min()), 2) if len(hy) else None, "hold_yr_max": round(float(hy["mean"].max()), 2) if len(hy) else None, "hold_years": int(len(hy)),
                  "hold_recent": round(float(gg[gg["year"] >= gg["year"].max() - 2]["hold"].mean()), 3), "n_recent": int((gg["year"] >= gg["year"].max() - 2).sum()),
                  "dip_hold_q20": round(float(hd["dip"].quantile(0.2)), 2) if len(hd) >= 10 else None, "dip_hold_q50": round(float(hd["dip"].median()), 2) if len(hd) >= 10 else None,
                  "pop_hold_q50": round(float(hd["pop"].median()), 2) if len(hd) >= 10 else None,
                  "dip_fail_q50": round(float(gg[gg["hold"] == False]["dip"].median()), 2) if (gg["hold"] == False).sum() >= 10 else None}  # noqa: E712
        out["gap"]["cells"][f"{int(b)}|{'bull' if bull else 'bear'}"] = {
            "label": GAP_LABELS[int(b)], "regime": "多頭 (前收在月線上)" if bull else "空頭 (前收在月線下)", "n": int(len(gg)), "p_fill": round(float(gg["fill"].mean()), 3), "p_cont": round(cont, 3),
            "p_hold": round(float(gg["hold"].mean()), 3), "intra_mean": round(float(gg["intra"].mean()), 3), "intra_med": round(float(gg["intra"].median()), 3), "intra_p20": round(float(gg["intra"].quantile(0.2)), 2), "intra_p80": round(float(gg["intra"].quantile(0.8)), 2),
            "ret_mean": round(float(gg["ret"].mean()), 3), "p_up_close": round(float((gg["ret"] > 0).mean()), 3), "range_mean": round(float(gg["rng"].mean()), 2),
            "cont_yr_cons": round(float(((yr >= 0.5) == (cont >= 0.5)).mean()), 2) if len(yr) else None, "years": int(len(yr)), **extra_}
    # 夜盤 → 跳空 β (2020~ 有夜盤資料的日子)
    try:
        from . import short_term as ST
        nh = ST._night_hist()
        if nh is not None and not nh.empty:
            m = d.merge(nh[["date", "night_chg_pct"]], on="date", how="inner").dropna(subset=["gap", "night_chg_pct"])
            m = m[m["night_chg_pct"].abs() <= 8]
            if len(m) >= 100:
                x, y = m["night_chg_pct"].values, m["gap"].values
                beta = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
                resid = y - beta * x
                mr = m.tail(250); xr, yr_ = mr["night_chg_pct"].values, mr["gap"].values
                beta_r = float(np.dot(xr - xr.mean(), yr_ - yr_.mean()) / np.dot(xr - xr.mean(), xr - xr.mean())) if len(mr) >= 120 else beta
                by_year = {int(yv): round(float(np.dot(gg["night_chg_pct"] - gg["night_chg_pct"].mean(), gg["gap"] - gg["gap"].mean()) / np.dot(gg["night_chg_pct"] - gg["night_chg_pct"].mean(), gg["night_chg_pct"] - gg["night_chg_pct"].mean())), 3) for yv, gg in m.groupby(m["date"].str[:4]) if len(gg) >= 60}
                big = np.abs(x) > 0.5
                out["gap"]["night_beta"] = {"beta": round(beta, 3), "beta_recent": round(beta_r, 3), "n": int(len(m)), "resid_sd": round(float(resid.std()), 3), "r2": round(float(1 - resid.var() / y.var()), 3),
                                            "by_year": by_year, "dir_agree_big": round(float((np.sign(x[big]) == np.sign(y[big])).mean()), 3), "n_big": int(big.sum())}
    except Exception as e:  # noqa: BLE001
        log.warning("night beta: %s", e)
    # 日曆效應
    ev = _events_for(d["date"].tolist())
    e = d.merge(ev, on="date", how="left").dropna(subset=["ret"])
    base_up, base_mean = float((e["ret"] > 0).mean()), float(e["ret"].mean())
    out["calendar"]["base"] = {"p_up": round(base_up, 3), "mean": round(base_mean, 3), "n": int(len(e))}
    for key, name in EVENTS.items():
        s = e[e[key] == True]  # noqa: E712
        if len(s) < 20:
            continue
        mean = float(s["ret"].mean()); sd = float(s["ret"].std()) or 1.0; t = (mean - base_mean) / (sd / np.sqrt(len(s)))
        yr = s.groupby("year")["ret"].mean(); yr = yr[s.groupby("year").size() >= 3]
        cons = float(((yr - base_mean > 0) == (mean - base_mean > 0)).mean()) if len(yr) else None
        valid = bool(len(s) >= 40 and cons is not None and cons >= 0.65 and abs(t) >= 2)
        out["calendar"][key] = {"name": name, "n": int(len(s)), "p_up": round(float((s["ret"] > 0).mean()), 3), "mean": round(mean, 3), "t": round(float(t), 2), "yr_cons": round(cons, 2) if cons is not None else None, "years": int(len(yr)),
                                "valid": valid, "dir": ("偏多" if mean > base_mean else "偏空") if valid else "無效",
                                "range_mean": round(float(s["rng"].mean()), 2)}
        if verbose:
            print(f"  日曆 {name}: n={len(s)} 上漲率 {out['calendar'][key]['p_up']} 平均 {mean:+.3f}% (基準 {base_mean:+.3f}) t={t:+.2f} 一致 {cons} → {'有效 ' + out['calendar'][key]['dir'] if valid else '無效'}")
    if verbose:
        nb = out["gap"].get("night_beta"); print(f"  跳空 β(夜盤→跳空) {nb}")
        for k, c in list(out["gap"]["cells"].items())[:14]:
            print(f"  跳空 {c['label']} {c['regime'][:2]}: n={c['n']} 回補 {c['p_fill']} 續走 {c['p_cont']} (年一致 {c['cont_yr_cons']}) 守住 {c['p_hold']} 日內 {c['intra_mean']:+.2f}% 收漲 {c['p_up_close']}")
    if write:
        M.save_json("precheck", out)
    return out


def build(scored: pd.DataFrame, snap: dict | None, fc: dict | None) -> dict | None:
    st = M.load_json("precheck")
    if not st:
        return None
    snap = snap or {}; fc = fc or {}
    d = scored[["date", "close"]].copy(); c = pd.to_numeric(d["close"], errors="coerce")
    last_close = float(c.iloc[-1]); ma20 = float(c.rolling(20).mean().iloc[-1]); bull = last_close >= ma20
    out = {"date": str(d["date"].iloc[-1])[:10], "gap": None, "calendar": []}
    # --- A. 跳空 ---
    est = src = None
    tx = snap.get("taiex") or {}
    live = bool(fc.get("intraday"))
    if live and tx.get("open") and tx.get("prev"):
        est, src = (float(tx["open"]) / float(tx["prev"]) - 1) * 100, "實際開盤"
    else:
        nd = fc.get("next_days") or []
        try:
            from . import range_levels as RL
            nv, why = RL._night_final(snap, nd, scored)
        except Exception:  # noqa: BLE001
            nv, why = None, None
        nb = (st.get("gap") or {}).get("night_beta") or {}
        bu = nb.get("beta_recent") or nb.get("beta")   # 2026-09-24：β 逐年 0.31~0.68 (2026 只 0.31)，改用近 250 日 β
        if nv is not None and bu is not None:
            est, src = bu * float(np.clip(nv, -8, 8)), f"夜盤 {nv:+.2f}% × 近一年 β {bu} (全期 {nb.get('beta')}，R² {nb.get('r2')})"
        else:
            tn = snap.get("tx_night") or {}
            if tn.get("change_pct") is not None and bu is not None and snap.get("phase") == "night":
                est, src = bu * float(np.clip(float(tn["change_pct"]), -8, 8)), f"夜盤進行中 {float(tn['change_pct']):+.2f}% × 近一年 β {bu} (暫定)"
    if est is not None:
        b = int(np.searchsorted(GAP_EDGES, est, side="right"))
        cell = (st["gap"]["cells"] or {}).get(f"{b}|{'bull' if bull else 'bear'}") or {}
        if cell:
            up = est > 0.15; dn = est < -0.15
            txt = (f"預估開盤{'跳空' if (up or dn) else '平盤附近'} {est:+.2f}% ({src})，{cell['regime']}下歷史同狀況 {cell['n']} 次："
                   + (f"當日回補跳空機率 {cell['p_fill']:.0%}、收盤高於開盤 {cell['p_cont']:.0%}、收盤守住跳空 {cell['p_hold']:.0%}" if up else
                      f"當日回補跳空 (反彈到前收) 機率 {cell['p_fill']:.0%}、收盤低於開盤 (續跌) {cell['p_cont']:.0%}、收盤仍低於前收 {cell['p_hold']:.0%}" if dn else
                      f"收盤高於開盤 {cell['p_cont']:.0%}")
                   + f"；日內 (開→收) 平均 {cell['intra_mean']:+.2f}% (兩成~八成 {cell['intra_p20']:+.2f}~{cell['intra_p80']:+.2f}%)、收盤上漲率 {cell['p_up_close']:.0%}、平均振幅 {cell['range_mean']:.2f}%")
            if cell.get("dip_hold_q50") is not None and (up or dn):
                txt += f"；守住日的日內回檔中位 {cell['dip_hold_q50']:+.2f}%、兩成 {cell['dip_hold_q20']:+.2f}% (未守住日中位 {cell['dip_fail_q50']:+.2f}%)"
            if cell.get("hold_years"):
                txt += f"；守住率逐年 {cell['hold_yr_min']:.0%}~{cell['hold_yr_max']:.0%} ({cell['hold_years']} 年)，近三年 {cell['hold_recent']:.0%} (n={cell['n_recent']})"
            if up and cell["p_fill"] >= 0.5:
                txt += " → 開高後多半會回測前收，不追開盤價，等回補再看是否守住"
            elif up and cell["p_hold"] >= 0.6:
                txt += f" → 開高守住機率高，開盤價下 {abs(cell.get('dip_hold_q50') or 0.2):.1f}~{abs(cell.get('dip_hold_q20') or 0.5):.1f}% 的小回檔即為進場點，跌破開盤 1.3% 以上視為守不住"
            elif dn and cell["p_fill"] >= 0.5:
                txt += " → 開低多半會反彈到前收附近，開盤殺低是短線買點"
            elif dn and cell["p_hold"] >= 0.6:
                txt += " → 開低後續弱，反彈到前收附近是減碼點"
            out["gap"] = {"est": round(est, 2), "source": src, "bucket": cell["label"], "regime": cell["regime"], "stats": cell, "text": txt, "live": live}
    # --- B. 日曆效應 (以預測目標日 = next_days[0].date；盤中則為今日) ---
    try:
        cal = fc.get("calendar") or []
        nd0 = (fc.get("next_days") or [{}])[0].get("date")
        target = str(nd0) if nd0 else (cal[0] if cal else None)
        if target:
            hist_dates = [str(x)[:10] for x in scored["date"].tolist()[-10:]]
            seq = sorted(set(hist_dates + [str(x) for x in cal[:10]] + [target]))
            ev = _events_for(seq).set_index("date").loc[target].to_dict()
            for key, name in EVENTS.items():
                if ev.get(key):
                    s = (st.get("calendar") or {}).get(key)
                    if s:
                        out["calendar"].append({"key": key, **s, "text": f"{name}：2010~ {s['n']} 次上漲率 {s['p_up']:.0%}、平均 {s['mean']:+.2f}% (基準 {st['calendar']['base']['p_up']:.0%} / {st['calendar']['base']['mean']:+.2f}%)，逐年一致 {s['yr_cons']}" + ("，通過驗證 → " + s["dir"] if s["valid"] else "，未通過驗證 (僅供參考)")})
            out["target"] = target
    except Exception as e:  # noqa: BLE001
        log.warning("calendar build: %s", e)
    out["base"] = (st.get("calendar") or {}).get("base")
    return out
