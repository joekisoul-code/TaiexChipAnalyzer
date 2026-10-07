"""預判邏輯 (2026-09-23)：兩組可驗證的「明天會怎麼走」條件統計，皆用 2010~ 日 K，逐年一致性檢查，只顯示通過驗證的結論。

A. 開盤跳空預判 (gap)：跳空幅度 (開盤 vs 前收) × 多空狀態 (前收 vs 月線) → 當日回補機率、續走機率 (收盤高於開盤/低於開盤)、
   日內 (開→收) 平均與中位、收盤仍守住跳空的機率、當日振幅。今日估計跳空：盤中用實際開盤；夜盤收後用 β × 夜盤；否則不預判。
   復市日 (2026-09-28 r2m reopen_us_move (i))：夜盤只涵蓋前一交易日晚上的美股 → 再加「其餘休市期間美股日費半累積 × β_gap」(gap.us_add)；
   美股尚未全部收盤時維持夜盤估計並標「暫定」(gap.provisional='us_pending'，learn 記 prov、判斷總結不計票)。
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


def _cell(gg: pd.DataFrame, b, bull) -> dict:
    """一個 (跳空桶, 多空) 格的統計；gg 需有 fill/cont/hold/dip/pop/intra/ret/rng/year 欄 (方向定義由呼叫端決定)。"""
    yr = gg.groupby("year")["cont"].mean(); yr = yr[gg.groupby("year").size() >= 5]
    cont = float(gg["cont"].mean())
    hy = gg.groupby("year")["hold"].agg(["mean", "size"]); hy = hy[hy["size"] >= 8]
    hd = gg[gg["hold"] == True]  # noqa: E712
    extra_ = {"hold_yr_min": round(float(hy["mean"].min()), 2) if len(hy) else None, "hold_yr_max": round(float(hy["mean"].max()), 2) if len(hy) else None, "hold_years": int(len(hy)),
              "hold_recent": round(float(gg[gg["year"] >= gg["year"].max() - 2]["hold"].mean()), 3), "n_recent": int((gg["year"] >= gg["year"].max() - 2).sum()),
              "dip_hold_q20": round(float(hd["dip"].quantile(0.2)), 2) if len(hd) >= 10 else None, "dip_hold_q50": round(float(hd["dip"].median()), 2) if len(hd) >= 10 else None,
              "pop_hold_q50": round(float(hd["pop"].median()), 2) if len(hd) >= 10 else None,
              "dip_fail_q50": round(float(gg[gg["hold"] == False]["dip"].median()), 2) if (gg["hold"] == False).sum() >= 10 else None}  # noqa: E712
    return {"label": GAP_LABELS[int(b)], "regime": "多頭 (前收在月線上)" if bull else "空頭 (前收在月線下)", "n": int(len(gg)), "p_fill": round(float(gg["fill"].mean()), 3), "p_cont": round(cont, 3),
            "p_hold": round(float(gg["hold"].mean()), 3), "intra_mean": round(float(gg["intra"].mean()), 3), "intra_med": round(float(gg["intra"].median()), 3), "intra_p20": round(float(gg["intra"].quantile(0.2)), 2), "intra_p80": round(float(gg["intra"].quantile(0.8)), 2),
            "ret_mean": round(float(gg["ret"].mean()), 3), "p_up_close": round(float((gg["ret"] > 0).mean()), 3), "range_mean": round(float(gg["rng"].mean()), 2),
            "cont_yr_cons": round(float(((yr >= 0.5) == (cont >= 0.5)).mean()), 2) if len(yr) else None, "years": int(len(yr)), **extra_}


EST_BETA_WIN = 250      # r6 估計跳空表：滾動 β 視窗 (與 build 的「近一年 β」同長)，至少 120 列，往後移一日 (只用前一日以前的資料)
EST_BETA_MIN = 120


def _est_cells(m: pd.DataFrame) -> tuple[dict, dict]:
    """r6 (2026-10-07 honesty gap_flat_fill)：開盤前 App 用「β × 夜盤」的估計跳空查表，舊表卻以實際跳空分組 →
    估計平盤 (±0.15%) 日顯示回補 79%，實際只有 66% (2017-11~2026-10，n=756，逐年皆低於 79%)。
    這裡以時點正確的估計跳空分組：β_t = 前 250 個有夜盤日 (至少 120) 的 cov/var，往後移一日；est = β_t × clip(夜盤, ±8)。
    方向定義：估計平盤格沿用實際跳空方向 (與實際表同)；估計開高/開低格以估計方向 (回補 = 觸及前收、守住 = 收盤仍在前收同側)。
    m = d 與夜盤合併後的列 (需 date/open/high/low/close/prev/gap/intra/ret/rng/year/bull/ma20/night_chg_pct，|夜盤| ≤ 8)。"""
    m = m.dropna(subset=["gap", "intra", "ma20", "night_chg_pct"]).sort_values("date").reset_index(drop=True)
    x, y = m["night_chg_pct"].astype(float), m["gap"].astype(float)
    cov = x.rolling(EST_BETA_WIN, min_periods=EST_BETA_MIN).cov(y).shift(1)
    var = x.rolling(EST_BETA_WIN, min_periods=EST_BETA_MIN).var().shift(1)
    m["beta_pit"] = cov / var
    m = m.dropna(subset=["beta_pit"]).reset_index(drop=True)
    meta = {"n": int(len(m)), "start": str(m["date"].iloc[0]) if len(m) else None, "end": str(m["date"].iloc[-1]) if len(m) else None,
            "beta": f"滾動 {EST_BETA_WIN} 日 (至少 {EST_BETA_MIN}) cov/var，往後移一日", "night_clip": 8}
    if not len(m):
        return {}, meta
    m["est"] = m["beta_pit"] * m["night_chg_pct"].clip(-8, 8)
    m["bucket"] = np.searchsorted(GAP_EDGES, m["est"].values, side="right")
    flat = m["bucket"] == 3
    s_act = np.sign(m["gap"]); s_est = np.where(m["bucket"] > 3, 1.0, -1.0)
    up_a = m["gap"] > 0
    fill_a = np.where(up_a, m["low"] <= m["prev"], m["high"] >= m["prev"])
    # r6 review：估計平盤格 build() 文字寫「收盤高於開盤 p_cont」→ p_cont 直接用 close>open
    # (實際表 flat 格的 p_cont 是「延續微小實際跳空方向」，空頭 54% vs 真正收盤高於開盤 45%，標籤不符；實際表維持不動)
    cont_a = (m["close"] > m["open"]).values
    hold_a = np.where(up_a, m["close"] > m["prev"], np.where(m["gap"] < 0, m["close"] < m["prev"], m["close"] > m["prev"]))
    fill_e = np.where(s_est > 0, m["low"] <= m["prev"], m["high"] >= m["prev"])
    cont_e = np.where(s_est > 0, m["close"] > m["open"], m["close"] < m["open"])
    hold_e = np.where(s_est > 0, m["close"] > m["prev"], m["close"] < m["prev"])
    m["fill"] = np.where(flat, fill_a, fill_e); m["cont"] = np.where(flat, cont_a, cont_e); m["hold"] = np.where(flat, hold_a, hold_e)
    m["dip"] = (m["low"] / m["open"] - 1) * 100; m["pop"] = (m["high"] / m["open"] - 1) * 100
    meta["flat_share"] = round(float(flat.mean()), 3)
    meta["sign_agree"] = round(float((s_act[~flat] == s_est[~flat]).mean()), 3) if (~flat).any() else None
    cells = {}
    for (b, bull), gg in m.groupby(["bucket", "bull"]):
        if len(gg) < 20:
            continue
        c = _cell(gg, b, bull)
        c["basis"] = "est"
        c["same_bucket_actual"] = round(float((np.searchsorted(GAP_EDGES, gg["gap"].values, side="right") == int(b)).mean()), 3)
        cells[f"{int(b)}|{'bull' if bull else 'bear'}"] = c
    return cells, meta


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
        out["gap"]["cells"][f"{int(b)}|{'bull' if bull else 'bear'}"] = _cell(gg, b, bull)
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
                # r6：開盤前 (估計跳空) 用的時點正確估計跳空分組表；盤中 (實際開盤) 仍用上面的 cells
                try:
                    ce, cm = _est_cells(m)
                    if ce:
                        out["gap"]["cells_est"], out["gap"]["est_meta"] = ce, cm
                except Exception as e:  # noqa: BLE001
                    log.warning("precheck est cells: %s", e)
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
        if out["gap"].get("cells_est"):
            print(f"  估計跳空表 {out['gap'].get('est_meta')}")
            for k, c in out["gap"]["cells_est"].items():
                a = out["gap"]["cells"].get(k) or {}
                print(f"  [估計] {c['label']} {c['regime'][:2]}: n={c['n']} 回補 {c['p_fill']} (實際表 {a.get('p_fill')}) 守住 {c['p_hold']} (實際表 {a.get('p_hold')})")
    if write:
        M.save_json("precheck", out)
    return out


def _us_add(scored: pd.DataFrame, nd: list[dict], us_px, now=None) -> dict | None:
    """復市日 (r2m reopen_us_move (i))：夜盤未涵蓋的美股交易日 (休市期間扣掉 prev 當晚那一個) 的費半累積 × β_gap (TAIEX 開盤跳空 vs
    前一美股日 SOX，前 750 個正常日、無截距；SOX 不齊改 TSM ADR 與其 β)。沒有未涵蓋美股日 → None (與舊版相同)。
    us_px = {"sox","tsm","px"(長 TAIEX date/open/close)} 或回傳該 dict 的函式；None → events.reopen_inputs()。"""
    from . import events as EV
    if not nd:
        return None
    td = str(nd[0]["date"])[:10]
    prev = str(scored["date"].iloc[-1])[:10]
    unc = [u for u in EV.us_sessions(prev, td) if u != prev]
    if not unc:
        return None
    rule = ((EV.load_params().get("gap_rules") or {}).get("precheck_us_add") or {})
    if not rule.get("enabled"):
        return None
    px = us_px() if callable(us_px) else (us_px if us_px is not None else EV.reopen_inputs(prev, td))
    px = px or {}
    mv = EV.us_move(prev, td, px.get("sox"), px.get("tsm"), sessions=unc, now=now or px.get("now"))
    out = {"status": mv["status"], "n_unc": len(unc), "sessions": unc, "src": mv["src"], "sum_pct": None, "beta": None, "beta_n": None, "add_pct": None}
    if mv["status"] != "ready":
        return out
    base = px.get("px") if px.get("px") is not None else scored
    b, n_b, _ = EV.gap_beta(base[["date", "open", "close"]], mv["xret"], td)
    out.update(sum_pct=round(mv["sum_logret"] * 100, 2), beta_n=n_b)
    if b is None or not np.isfinite(b):
        out["status"] = "no_beta"            # 正常日 < 250 → 維持夜盤估計 (資料已齊，非暫定)
        return out
    out.update(beta=round(float(b), 3), add_pct=round(float(b) * mv["sum_logret"] * 100, 3))
    return out


def build(scored: pd.DataFrame, snap: dict | None, fc: dict | None, us_px=None, now=None) -> dict | None:
    """us_px / now：復市日未涵蓋美股的輸入 (測試注入用；省略 → events.reopen_inputs() 與現在時間)。"""
    st = M.load_json("precheck")
    if not st:
        return None
    snap = snap or {}; fc = fc or {}
    d = scored[["date", "close"]].copy(); c = pd.to_numeric(d["close"], errors="coerce")
    last_close = float(c.iloc[-1]); ma20 = float(c.rolling(20).mean().iloc[-1]); bull = last_close >= ma20
    out = {"date": str(d["date"].iloc[-1])[:10], "gap": None, "calendar": []}
    # --- A. 跳空 ---
    est = src = None
    ua, prov = None, None
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
            # 復市日 (2026-09-28 reopen_us_move (i))：完整且對齊的夜盤只涵蓋 prev 當晚的美股 → 加上其餘美股日費半 × β_gap (MSE 1.201 → 0.429，n=67)
            try:
                ua = _us_add(scored, nd, us_px, now)
            except Exception as e:  # noqa: BLE001
                log.warning("precheck us_add: %s", e)
                ua = None
            if ua is not None:
                ua["est_night_only"] = round(est, 3)
                lbl = "費半" if ua.get("src") == "SOX" else "台積電ADR"
                if ua["status"] == "ready":
                    ua["text"] = f"＋夜盤未涵蓋的 {ua['n_unc']} 個美股交易日{lbl}累積 {ua['sum_pct']:+.2f}% × β {ua['beta']:.2f} = {ua['add_pct']:+.2f}%"
                    est, src = est + ua["add_pct"], src + ua["text"]
                elif ua["status"] == "no_beta":
                    ua["text"] = f"夜盤未涵蓋的 {ua['n_unc']} 個美股交易日：正常日樣本不足 ({ua.get('beta_n')} < 250)，只用夜盤估計"
                else:   # 美股未全部收盤 / 資料缺 → 夜盤估計 + 暫定 (learn 記 prov，復市日清晨 ready 後才以 final 入帳)
                    ua["text"] = f"夜盤未涵蓋的 {ua['n_unc']} 個美股交易日 ({'、'.join(u[5:] for u in ua['sessions'])}) 尚未全部收盤"
                    src, prov = src + " (暫定：休市期間美股尚未全部收盤)", "us_pending"
        else:
            tn = snap.get("tx_night") or {}
            if tn.get("change_pct") is not None and bu is not None and snap.get("phase") == "night":
                est, src = bu * float(np.clip(float(tn["change_pct"]), -8, 8)), f"夜盤進行中 {float(tn['change_pct']):+.2f}% × 近一年 β {bu} (暫定)"
    if est is not None:
        b = int(np.searchsorted(GAP_EDGES, est, side="right"))
        key = f"{b}|{'bull' if bull else 'bear'}"
        cell_act = (st["gap"]["cells"] or {}).get(key) or {}
        cell_est = (st["gap"].get("cells_est") or {}).get(key) or {}
        # r6 (honesty gap_flat_fill)：開盤前 (估計跳空) 用估計跳空分組表；盤中實際開盤已知 → 實際跳空表。估計表缺 (舊 precheck.json / 格 n<20) → 實際表
        estimated = src != "實際開盤"
        table = "est" if (estimated and cell_est) else "actual"
        cell = cell_est if table == "est" else cell_act
        if cell:
            up = est > 0.15; dn = est < -0.15
            basis = "，以夜盤估計跳空分組" if table == "est" else ""
            txt = (f"預估開盤{'跳空' if (up or dn) else '平盤附近'} {est:+.2f}% ({src})，{cell['regime']}下歷史同狀況 {cell['n']} 次{basis}："
                   + (f"當日回補跳空機率 {cell['p_fill']:.0%}、收盤高於開盤 {cell['p_cont']:.0%}、收盤守住跳空 {cell['p_hold']:.0%}" if up else
                      f"當日回補跳空 (反彈到前收) 機率 {cell['p_fill']:.0%}、收盤低於開盤 (續跌) {cell['p_cont']:.0%}、收盤仍低於前收 {cell['p_hold']:.0%}" if dn else
                      (f"當日回補 (回到前收) {cell['p_fill']:.0%}、收盤高於開盤 {cell['p_cont']:.0%}" if table == "est" else f"收盤延續開盤方向 {cell['p_cont']:.0%}"))   # r6：實際表平盤格 p_cont = 延續微小實際跳空方向，不是「收盤高於開盤」
                   + f"；日內 (開→收) 平均 {cell['intra_mean']:+.2f}% (兩成~八成 {cell['intra_p20']:+.2f}~{cell['intra_p80']:+.2f}%)、收盤上漲率 {cell['p_up_close']:.0%}、平均振幅 {cell['range_mean']:.2f}%")
            if cell.get("dip_hold_q50") is not None and (up or dn):
                txt += f"；守住日的日內回檔中位 {cell['dip_hold_q50']:+.2f}%、兩成 {cell['dip_hold_q20']:+.2f}% (未守住日中位 {cell['dip_fail_q50']:+.2f}%)"
            if cell.get("hold_years"):
                txt += f"；守住率逐年 {cell['hold_yr_min']:.0%}~{cell['hold_yr_max']:.0%} ({cell['hold_years']} 年)，近三年 {cell['hold_recent']:.0%} (n={cell['n_recent']})"
            # r6：中性描述 (原「不追開盤價 / 進場點 / 短線買點 / 減碼點」)
            if up and cell["p_fill"] >= 0.5:
                txt += " → 開高後多半會回測前收，回測後能否守住前收為觀察點"
            elif up and cell["p_hold"] >= 0.6:
                txt += f" → 開高守住機率高；守住日開盤價下 {abs(cell.get('dip_hold_q50') or 0.2):.1f}~{abs(cell.get('dip_hold_q20') or 0.5):.1f}% 為常見回檔深度 (下緣參考)，跌破開盤 1.3% 以上歷史上多半守不住"
            elif dn and cell["p_fill"] >= 0.5:
                txt += " → 開低多半會反彈到前收附近"
            elif dn and cell["p_hold"] >= 0.6:
                txt += " → 開低後續偏弱，前收附近為上緣參考 (壓力)"
            out["gap"] = {"est": round(est, 2), "source": src, "bucket": cell["label"], "regime": cell["regime"], "stats": cell, "text": txt, "live": live}
            # r6：兩張表都發布，table 說明 stats 用的是哪一張
            out["gap"]["table"] = table
            out["gap"]["table_note"] = ("開盤前以夜盤估計跳空：用「估計跳空分組」歷史統計 (時點正確 β" + (f"，{(st['gap'].get('est_meta') or {}).get('start')}~" if (st['gap'].get('est_meta') or {}).get('start') else "") + ")"
                                        if table == "est" else
                                        ("盤中實際開盤已知：用實際跳空分組歷史統計 (2010~)" if not estimated else
                                         "估計跳空分組表無此格 (樣本不足或未訓練)，暫用實際跳空分組統計 (2010~)" + ("；平盤估計的回補率可能高估" if not (up or dn) else "")))
            if cell_act:
                out["gap"]["stats_actual"] = cell_act
            if cell_est:
                out["gap"]["stats_est"] = cell_est
            if ua is not None:
                out["gap"]["us_add"] = ua
            if prov:
                out["gap"]["provisional"] = prov
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
