"""大盤走勢預測：訓練 (2010~ 長歷史)、逐年驗證、即時預測 (含盤中以現價推估)。"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .. import config
from ..analysis import backtest, market
from . import model as M
from ..sources import twse
from .features import FEATURE_NAMES, MARKET_FEATURES, as_if_close, market_matrix

log = logging.getLogger(__name__)
HORIZONS = (1, 2, 3, 5, 10, 20)
DAY_HORIZONS = (1, 2, 3)
FIRST_TEST_YEAR = 2014


def train(write: bool = True) -> dict:
    scored = backtest.load_long()
    mat = market_matrix(scored)
    results = {}
    for h in HORIZONS:
        oos = M.walk_forward(mat, MARKET_FEATURES, f"fwd{h}", h, FIRST_TEST_YEAR)
        met = M.metrics(oos)
        results[h] = met
        if write:
            d = mat.dropna(subset=[f"fwd{h}"])
            models = M.fit_ensemble(d[MARKET_FEATURES], d[f"fwd{h}"])
            M.save(f"market_h{h}", {"models": models, "features": MARKET_FEATURES, "horizon": h,
                                    "trained_at": dt.datetime.now(config.TZ).isoformat(), "train_end": str(d["date"].max()),
                                    "n_train": int(len(d)), "metrics": met, "calibration": met.get("calibration")})
            oos.to_csv(M.MODEL_DIR / f"market_h{h}_oos.csv", index=False)
    if write:
        M.save_json("market_metrics", {str(h): {k: v for k, v in m.items() if k != "calibration"} for h, m in results.items()})
    return results


def _predict_row(bundle: dict, row: pd.DataFrame) -> dict:
    x = row[bundle["features"]].astype(float)
    pred = float(M.predict_ensemble(bundle["models"], x)[0])
    cal = M.apply_calibration(bundle["calibration"], pred)
    return {"pred": round(pred, 2), **cal, "drivers": M.explain(bundle["models"], x, FEATURE_NAMES)}


def forecast(scored: pd.DataFrame, snapshot: dict | None = None) -> dict:
    bundles = {h: M.load(f"market_h{h}") for h in HORIZONS}
    if any(b is None for b in bundles.values()):
        return {"error": "尚未訓練模型，請執行 python cli.py train"}
    mat = market_matrix(scored)
    last = mat.iloc[[-1]]
    out = {"date": str(last["date"].iloc[0]), "close": float(last["close"].iloc[0]), "trained_at": bundles[10]["trained_at"],
           "train_end": bundles[10]["train_end"], "horizons": {h: _predict_row(b, last) for h, b in bundles.items()}, "intraday": None}
    idx = (snapshot or {}).get("taiex") or {}
    if snapshot and snapshot.get("phase") in ("open", "pre", "post") and idx.get("last") and not snapshot.get("same_day", False):
        try:
            d2 = market.score_frame(market.add_features(as_if_close(scored, float(idx["last"]), snapshot.get("amount_projected"))))
            d2 = d2.loc[:, ~d2.columns.duplicated(keep="last")]      # scored 已含 f_*/composite，score_frame 以 concat 併入會重複 → 留最新一份
            m2 = market_matrix(d2).iloc[[-1]]
            out["intraday"] = {"price": float(idx["last"]), "chg_pct": idx.get("chg_pct"), "time": idx.get("time"),
                               "horizons": {h: _predict_row(b, m2) for h, b in bundles.items()}}
        except Exception as e:  # noqa: BLE001
            log.warning("intraday forecast failed: %s", e)
    out["metrics"] = {h: {k: v for k, v in b.get("metrics", {}).items() if k != "calibration"} for h, b in bundles.items()}
    # 隔天 / 後天 / 第三天 (跳過休市)：預測收盤水準 = 基準價 × (1 + 同分位歷史平均%)
    base_px, base_date = out["close"], out["date"]
    src = out["horizons"]
    if out["intraday"]:
        base_px, src = out["intraday"]["price"], out["intraday"]["horizons"]
        base_date = dt.date.today().isoformat()
    try:
        days = twse.next_trading_days(base_date, 3)
    except Exception:  # noqa: BLE001
        days = [f"T+{i}" for i in (1, 2, 3)]
    out["next_days"] = []
    for i, h in enumerate(DAY_HORIZONS):
        r = src[h]
        out["next_days"].append({"n": h, "date": days[i], "label": ["隔天", "後天", "第三天"][i], "p_up": r["p_up"], "base_hit": r["base_hit"],
                                 "pred": r["pred"], "hist_mean": r["hist_mean"], "q20": r["q20"], "q80": r["q80"], "bin": r["bin"],
                                 "level": round(base_px * (1 + (r["hist_mean"] or 0) / 100), 0),
                                 "level_lo": round(base_px * (1 + (r["q20"] or 0) / 100), 0), "level_hi": round(base_px * (1 + (r["q80"] or 0) / 100), 0),
                                 "drivers": r["drivers"]})
    # 未來 45 個交易日日曆 (跳過週末與 TWSE 休市日)：供前端畫預測路徑，避免把假日當交易日
    try:
        out["calendar"] = twse.next_trading_days(out["date"], 45)
    except Exception:  # noqa: BLE001
        out["calendar"] = []
    out["summary"] = summarize(out)
    return out


# ============================================================ 近五日精修 (v3)
# 日模型 1~5 日 IC 僅 0.02~0.04；短天期真正有預測力的是「前晚夜盤台指期」(隔日開盤跳空，樣本外 IC≈0.6) 與
# 已在 2010~ 逐條驗證的買賣點規則。這裡把兩者疊到 ML 輸出上：
#   1) 夜盤跳空 β：以歷史「隔日開盤跳空 vs 夜盤漲跌」回歸的斜率，把預期跳空併入各日水準 (收盤後/夜盤/開盤前才適用)
#   2) 隔天改用小時模型 13:30 目標 (含夜盤，命中 74% vs 基準 57%)
#   3) 5 日視野：近 3 日觸發且驗證有效 (✓) 的規則，用其歷史 5 日超額報酬做覆蓋 (上限 ±1.5%)
_GAP_CACHE: dict = {}


def night_gap_beta() -> tuple[float, float, int]:
    """隔日開盤跳空 % 對 夜盤台指期漲跌 % 的回歸斜率 (近 3 年)。回傳 (beta, corr, n)。"""
    key = dt.date.today().isoformat()
    if key in _GAP_CACHE:
        return _GAP_CACHE[key]
    from ..sources import finmind
    start = (dt.date.today() - dt.timedelta(days=3 * 365)).isoformat()
    beta, corr, n = 0.85, 0.0, 0
    try:
        # FinMind after_market 的 date = 該夜盤「準備的隔一交易日」→ 對齊該日開盤相對前一日收盤的跳空
        px = finmind.taiex_price(start)[["date", "open", "close"]].sort_values("date").reset_index(drop=True)
        px["gap"] = (px["open"] / px["close"].shift(1) - 1) * 100
        night = finmind.tx_night_history(start)
        d = px.merge(night[["date", "night_chg_pct"]], on="date", how="inner").dropna(subset=["gap", "night_chg_pct"])
        d = d[(d["gap"].abs() < 8) & (d["night_chg_pct"].abs() < 8)]
        if len(d) >= 60:
            x, y = d["night_chg_pct"].astype(float), d["gap"].astype(float)
            beta = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
            corr = float(x.corr(y))
            n = int(len(d))
    except Exception as e:  # noqa: BLE001
        log.warning("night_gap_beta: %s", e)
    _GAP_CACHE[key] = (round(beta, 3), round(corr, 3), n)
    return _GAP_CACHE[key]


def refine_short_term(fc: dict, hourly: dict | None, snapshot: dict | None, signals_res: dict | None, scored: pd.DataFrame | None = None, learn_summary: dict | None = None) -> dict:
    if not fc or fc.get("error") or not fc.get("next_days"):
        return fc
    fc = dict(fc)
    nd = [dict(x) for x in fc["next_days"]]
    hz = {k: dict(v) for k, v in fc["horizons"].items()}
    notes: list[str] = []
    live = bool(fc.get("intraday"))
    base_px = fc["intraday"]["price"] if live else fc["close"]
    snap = snapshot or {}
    # 0) 短線模型 v2 (1/2/3/5 日：短線特徵 + LGB/Ridge 集成 + 夜盤變體 + 叫牌檔位)：取代日模型的分位校準值
    st = {}
    if scored is not None and not live:
        try:
            from . import short_term
            st = short_term.forecast(scored, snap)
        except Exception as e:  # noqa: BLE001
            log.warning("short_term.forecast: %s", e)
    if st:
        for x in nd:
            r = st.get(x["n"])
            if not r:
                continue
            x["daily_model_v1"] = {k: x.get(k) for k in ("p_up", "level", "hist_mean")}
            x.update({k: r[k] for k in ("p_up", "hist_mean", "q20", "q80", "bin", "base_hit", "call", "call_strength", "call_hit", "tier_up_hit", "tier_dn_hit", "call_cov", "variant", "drivers")})
            x["level"] = round(base_px * (1 + (r["hist_mean"] or 0) / 100))
            x["level_lo"] = round(base_px * (1 + (r["q20"] or 0) / 100))
            x["level_hi"] = round(base_px * (1 + (r["q80"] or 0) / 100))
            x["source"] = r["note"]
            x["caveat"] = r.get("caveat") or ""      # 叫牌可信度提醒 (v2.2)：獨立鍵，小時模型覆寫 source 後仍保留
        if 5 in st and 5 in hz:
            r = st[5]
            hz[5]["daily_model_v1"] = {k: hz[5].get(k) for k in ("p_up", "hist_mean")}
            hz[5].update({k: r[k] for k in ("p_up", "hist_mean", "q20", "q80", "bin", "base_hit", "call", "call_strength", "call_hit", "tier_up_hit", "tier_dn_hit", "call_cov", "variant", "drivers")})
            hz[5]["source"] = r["note"]
            hz[5]["caveat"] = r.get("caveat") or ""
        # 信心分層 (2026-09-22)：模型 × 夜盤 × 規律 × 季線 共識 → 高/中/低，附走動式 OOS 命中
        try:
            from . import confidence as CF, patterns as PT
            pat = PT.load() or {}
            rs = 0.0
            for a in pat.get("today") or []:
                v = (a.get("h") or {}).get("1")
                if a.get("valid") and v and v.get("valid"):
                    rs += 1 if v["direction"] == "偏多" else -1
            bull = None
            try:
                bull = bool(float(scored["close"].iloc[-1]) >= float(scored["ma60"].iloc[-1]))
            except Exception:  # noqa: BLE001
                pass
            tn0 = snap.get("tx_night") or {}
            night_v = float(tn0["change_pct"]) if tn0.get("change_pct") is not None and snap.get("phase") in ("night", "closed", "pre") else None
            hsi_v = kospi_v = None
            try:
                from . import short_term as _ST2
                _row = _ST2.build_matrix(scored.tail(60).reset_index(drop=True), None).iloc[-1]
                hsi_v = float(_row["hsi_r0"]) if "hsi_r0" in _row and _row["hsi_r0"] == _row["hsi_r0"] else None
                kospi_v = float(_row["kospi_r0"]) if "kospi_r0" in _row and _row["kospi_r0"] == _row["kospi_r0"] else None
            except Exception:  # noqa: BLE001
                pass
            for x in nd:
                if not x.get("call"):
                    continue
                tier = CF.label(x["call"], x.get("call_strength") or "", x.get("variant") or "base", night_v, rs, bull, hsi_v, kospi_v)
                if tier:
                    st_ = CF.stats_for(int(x["n"]), x.get("variant") or "base", tier) or {}
                    x["conf_tier"], x["conf_hit"], x["conf_cov"], x["conf_yr_min"] = tier, st_.get("hit"), st_.get("cov"), st_.get("yr_min")
                    x["conf_note"] = f"信心{tier}：模型{'強' if x.get('call_strength') else ''}叫牌" + ("、夜盤同向" if night_v is not None and ((night_v > 0) == (x['call'] == '偏多')) else "") + (f"、規律{'同向' if (rs > 0) == (x['call'] == '偏多') and rs != 0 else '不反向' if rs == 0 else '反向'}" ) + ((f"、恆生{'同向' if (hsi_v > 0) == (x['call'] == '偏多') else '反向'}/KOSPI{'同向' if (kospi_v > 0) == (x['call'] == '偏多') else '反向'}") if hsi_v is not None and kospi_v is not None else "") + (f"；OOS 命中 {st_['hit']:.0%} (覆蓋 {st_['cov']:.0%}，逐年最低 {st_['yr_min']:.0%})" if st_.get("hit") else "")
        except Exception as e:  # noqa: BLE001
            log.warning("confidence: %s", e)
        calls = "、".join(f"{x['label']} {x.get('call')}{x.get('call_strength') or ''} ({(x.get('call_hit') or 0):.0%})" for x in nd if x.get("call"))
    # 聰明錢 v2 覆蓋 (2026-09-22)：5/10/20 日視野依聰明錢確認/否決 (見 crossmkt.SMART_TIERS)
    try:
        from . import crossmkt as XM
        pat_deep = fc.get("deep") or {}
        s2 = ((pat_deep.get("today") or {}).get("smart2"))
        if s2 is None and scored is not None:
            try:
                from . import short_term as _ST
                s2 = float(XM.add_features(_ST.build_matrix(scored.tail(400).reset_index(drop=True), None))["smart2"].iloc[-1])
            except Exception:  # noqa: BLE001
                s2 = None
        for h in (5, 10, 20):
            r = hz.get(h) or hz.get(str(h))
            if not r:
                continue
            call0 = r.get("call") or ("偏多" if (r.get("p_up") or 0) >= (r.get("base_hit") or 0.5) + 0.03 else "偏空" if (r.get("p_up") or 0) <= (r.get("base_hit") or 0.5) - 0.03 else "中性")
            ov = XM.smart_overlay(h, call0, s2)
            if ov:
                r["call_model"] = call0
                r.update(ov)
                r["smart2"] = round(float(s2), 2)
        if s2 is not None:
            notes.append(f"聰明錢 v2 {s2:+.2f}：5/10/20 日叫牌依聰明錢確認或否決 (" + "、".join(f"{h} 日 {(hz.get(h) or hz.get(str(h)) or {}).get('call_smart', '—')}" for h in (5, 10, 20)) + ")")
    except Exception as e:  # noqa: BLE001
        log.warning("smart overlay: %s", e) + (f"、5 日 {st[5]['call']}{st[5].get('call_strength') or ''} ({(st[5].get('call_hit') or 0):.0%})" if 5 in st else "")
        notes.append(f"短線模型 v2 ({'含夜盤' if st[min(st)]['variant'] == 'night' else '不含夜盤'})：叫牌 {calls}；括號為該檔位樣本外命中率")
    # 0b) 路徑型買賣點 (拉回買 buy_at / 反彈賣 sell_at / 停損 stop / 目標 target；level_lo/level_hi 覆寫為 20%/80% 路徑分位)
    #     夜盤模式只在夜盤結束且日期對齊時套用 (range_levels._night_final)，因此下方跳空與小時模型都不得再改 level_lo/level_hi
    if scored is not None:
        try:
            from . import range_levels as RLV
            sf = 1.0
            if learn_summary:
                try:
                    from . import learn as LRN
                    sf = LRN.touch_factor(learn_summary, 1)
                except Exception:  # noqa: BLE001
                    sf = 1.0
            rn = RLV.attach_to_next_days(nd, scored, snap, base_px, live=live, sigma_factor=sf)
            if rn:
                notes.append(rn)
        except Exception as e:  # noqa: BLE001
            log.warning("range_levels: %s", e)
    # 1) 夜盤跳空
    tn = snap.get("tx_night")
    gap = None
    if tn and tn.get("change_pct") is not None and not live and snap.get("phase") in ("night", "closed", "pre"):
        beta, corr, n = night_gap_beta()
        gap = beta * float(tn["change_pct"]) / 100
        for x in nd:
            hm = (x.get("hist_mean") or 0) / 100
            x["level_raw"] = x["level"]
            x["level"] = round(base_px * (1 + gap) * (1 + hm))
            if x.get("range_mode") is None:      # 路徑型水準已含 (或刻意不含) 夜盤位移，再乘跳空會重複計算
                x["level_lo"] = round(base_px * (1 + gap) * (1 + (x.get("q20") or 0) / 100))
                x["level_hi"] = round(base_px * (1 + gap) * (1 + (x.get("q80") or 0) / 100))
            x["gap_adj"] = round(gap * 100, 2)
        for h, r in hz.items():
            r["gap_adj"] = round(gap * 100, 2)
        notes.append(f"夜盤台指 {tn['change_pct']:+.2f}% → 預期跳空 {gap * 100:+.2f}% (歷史 β={beta:.2f}、r={corr:.2f}、n={n})，已併入各日水準")
    # 2) 隔天用小時模型 13:30 目標
    tg = (hourly or {}).get("targets") or {}
    if hourly and not hourly.get("live") and "13:30" in tg and nd and hourly.get("day") == nd[0]["date"]:
        t = tg["13:30"]
        keep = {k: nd[0].get(k) for k in ("p_up", "level", "hist_mean", "caveat")}
        hk = ("p_up", "base_hit", "pred", "hist_mean", "q20", "q80", "level") if nd[0].get("range_mode") else ("p_up", "base_hit", "pred", "hist_mean", "q20", "q80", "level", "level_lo", "level_hi")
        nd[0].update({k: t.get(k) for k in hk if t.get(k) is not None})      # 路徑型 level_lo/level_hi 不被小時模型收盤分位覆寫
        nd[0]["source"] = "小時模型 13:30 目標 (含前晚夜盤跳空，樣本外 IC≈0.6、命中 74% vs 基準 57%)"
        nd[0]["daily_model"] = keep
        notes.append(f"隔天改用小時模型收盤目標 {t.get('level'):,.0f} (上漲率 {t.get('p_up', 0):.0%})；日模型原值 {keep['level']:,.0f} ({(keep['p_up'] or 0):.0%})")
    elif nd and not nd[0].get("source"):
        nd[0]["source"] = "日模型 (1 日 IC 僅 0.02~0.04，可預測性低)"
    # 3) 5 日規則覆蓋
    cur = (signals_res or {}).get("current") or {}
    ev = (signals_res or {}).get("evaluation")
    try:
        rows = ev.to_dict("records") if hasattr(ev, "to_dict") else (ev or [])
        stats = {r["訊號"]: r for r in rows}
        base5 = (stats.get("全體基準") or {}).get("5日均報酬%") or 0.0
        excess, used = 0.0, []
        for s in cur.get("buy_signals", []):
            r = stats.get(s["name"])
            if r and s.get("valid") == "✓" and r.get("5日均報酬%") is not None:
                excess += r["5日均報酬%"] - base5
                used.append(f"{s['name']} ({r['5日均報酬%'] - base5:+.2f}%)")
        for s in cur.get("sell_signals", []):
            r = stats.get(s["name"])
            if r and s.get("valid") == "✓" and r.get("5日均報酬%") is not None:
                excess += r["5日均報酬%"] - base5
                used.append(f"{s['name']} ({r['5日均報酬%'] - base5:+.2f}%)")
        excess = max(-1.5, min(1.5, excess))
        if used and abs(excess) >= 0.1 and 5 in hz:
            r5 = hz[5]
            r5["hist_mean_raw"], r5["p_up_raw"] = r5.get("hist_mean"), r5.get("p_up")
            r5["hist_mean"] = round((r5.get("hist_mean") or 0) + excess, 2)
            r5["p_up"] = round(max(0.2, min(0.85, (r5.get("p_up") or 0.5) + excess * 0.06)), 3)
            r5["rule_overlay"] = round(excess, 2)
            notes.append(f"5 日視野加上驗證有效規則的歷史 5 日超額 {excess:+.2f}%：" + "、".join(used))
    except Exception as e:  # noqa: BLE001
        log.debug("rule overlay: %s", e)
    # 4) 7 個交易日趨勢閘門：down → 不推薦買點 (僅賣點/觀望)；up → 不推薦賣點；以最後一個已收盤列計算 (盤中不重算)
    try:
        from . import trend7
        g = trend7.trend7_gate(scored) if scored is not None else trend7._empty("無 scored 資料")
    except Exception as e:  # noqa: BLE001
        log.warning("trend7: %s", e)
        g = {"available": False, "state": "flat", "confidence": "", "p_down": None, "exp_ret7": None, "reasons": [], "text": f"trend7 失敗 ({e})",
             "drivers_toward_down": [], "drivers_toward_up": [], "oos": {}, "horizon_note": "7 個交易日"}
    fc["trend7"] = g
    if g.get("available"):
        if g.get("state") == "down":
            notes.append(f"7 個交易日趨勢閘門 [{g.get('confidence')}]：偏下 → 不推薦買點，僅賣點/觀望 (" + "；".join(g.get("reasons") or []) + ")")
        elif g.get("state") == "up":
            notes.append("7 個交易日趨勢閘門：偏上 → 不推薦賣點 (" + "；".join(g.get("reasons") or []) + ")")
    fc["next_days"], fc["horizons"], fc["short_term_notes"] = nd, hz, notes
    fc["summary"] = summarize(fc)
    return fc


def _tone(r: dict) -> str:
    diff = (r["p_up"] or 0) - r["base_hit"]
    return "偏多" if diff >= 0.04 else "偏空" if diff <= -0.04 else "中性"


def summarize(fc: dict) -> str:
    h10, h5, h20 = fc["horizons"][10], fc["horizons"][5], fc["horizons"][20]
    txt = ""
    for nd in fc.get("next_days", []):
        txt += f"{nd['label']} {nd['date']}：{_tone(nd)}，上漲率 {nd['p_up']:.0%} (基準 {nd['base_hit']:.0%})，預估收盤 {nd['level']:,.0f} (區間 {nd['level_lo']:,.0f}~{nd['level_hi']:,.0f})。"
    txt += (f" 未來 10 日：模型期望報酬 {h10['pred']:+.2f}%，落在歷史第 {h10['bin'] + 1}/5 分位；同分位過去實際上漲率 {h10['p_up']:.0%}"
           f"（基準 {h10['base_hit']:.0%}）、平均 {h10['hist_mean']:+.2f}%、20~80% 區間 {h10['q20']:+.2f}% ~ {h10['q80']:+.2f}% → {_tone(h10)}。"
           f" 5 日 {_tone(h5)} ({h5['p_up']:.0%})；20 日 {_tone(h20)} ({h20['p_up']:.0%}，平均 {h20['hist_mean']:+.2f}%)。")
    pos = "、".join(r["name"] for r in h10["drivers"]["positive"][:3])
    neg = "、".join(r["name"] for r in h10["drivers"]["negative"][:3])
    txt += f" 推升：{pos or '無'}；壓抑：{neg or '無'}。"
    if fc.get("intraday"):
        i = fc["intraday"]["horizons"][10]
        txt += f" 盤中以現價 {fc['intraday']['price']:,.0f} 推估：10 日 {_tone(i)}，上漲率 {i['p_up']:.0%}。"
    return txt
