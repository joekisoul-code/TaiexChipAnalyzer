"""判斷總結 (2026-09-23)：把各獨立層集成成一句可執行的判斷，並附「共識票數 → 走動式 OOS 命中率」。

層 (投票者)：短線模型叫牌 (錨) ＋ 夜盤台指期、歷史規律庫、恆生同日、KOSPI 同日、20 日趨勢 (收盤 vs MA20)、聰明錢 v2 ＝ 6 票；
另有只顯示不計票的即時層：7 日趨勢閘門、線上自學 (Platt 校準/降級/水準偏誤)、盤中小時模型 (現在→13:30)、盤中即時評分 (廣度/價差/量能)。

研究 (scratch consensus_study.py，走動式、門檻用前幾年分位，無洩漏)：
  淨票 (同向 − 反向) 越高命中越高；train() 重算並存 data/models/verdict.json，build() 只查表貼標。
規則：模型無叫牌 → 中性 (顯示其他票的多空比)；有叫牌 → 淨票 ≥ 3 → 「高共識」、≤ -3 → 「分歧，觀望」(歷史上此時叫牌反而低於 50%)、其餘 「一般」。
研究結果 (不含夜盤 1 日，n=1696，全部 56.2%)：淨票 3/4/5 → 58.6/66.7/68.2%；-3/-4 → 46.7/42.1%；含夜盤 1 日全部 85%，各淨票 79~89% 無單調關係。
單票：規律庫同向 60.9% vs 反向 46.6% 最有用；恆生/KOSPI 同日 58 vs 52；月線趨勢 59 vs 53；聰明錢 v2 57 vs 57 (1 日無用，留作 5 日)。
第三輪 (2026-09-24 vote_study)：
- 加權票 (走動式線性機率) 命中 54.7% 不如等權淨票 55.5% → 維持等權。
- 模型未叫牌日 (n=917)：5 票淨多 ≥4 → 隔天上漲 70.5% (n=61，年最低 45%)、+3 → 59.4%；淨空票對下跌無預測力 (−2 → 上漲 52%)。
  → 未叫牌時只在「淨多 ≥4」給「偏多‧訊號共識」(call_action=偏多)，+3 只加註；空方共識不叫牌。3 日視野同型態更穩 (+2~+5 → 63~68%)。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
HORIZONS = (1, 2, 3)
VOTES = ["night", "rule_score", "hsi_r0", "kospi_r0", "tr20", "s2"]
NAMES = {"night": "夜盤台指期", "rule_score": "歷史規律庫", "hsi_r0": "恆生同日", "kospi_r0": "KOSPI 同日", "tr20": "20 日趨勢", "s2": "聰明錢 v2"}
HIGH_NET = 3    # 不含夜盤 1 日：淨票 ≥3 → 61~68%、≤-3 → 42~47% (反指標)、中間 ≈ 基準 55%；含夜盤變體共識幾乎不加分 (夜盤本身已 85%)
LOW_NET = -3


def _prep_frame(matrix: pd.DataFrame, pat: dict) -> pd.DataFrame:
    from . import confidence as CF, crossmkt as XM, patterns as P, short_term as ST
    d = P._prep(XM.add_features(matrix))
    d["rule_score"] = CF.rule_score_series(d, pat)
    d["night"] = d[ST.NIGHT_FEATURE] if ST.NIGHT_FEATURE in d else np.nan
    d["tr20"] = np.sign(d["close"] - d["close"].rolling(20).mean())
    d["s2"] = np.sign(d["smart2"]) if "smart2" in d else 0.0
    return d


def train(matrix: pd.DataFrame, pat: dict, write: bool = True, verbose: bool = True) -> dict:
    """走動式：每個 (h, variant) 的叫牌依淨票分組的 OOS 命中率/覆蓋/逐年最低。matrix = short_term.build_matrix(scored, night)。"""
    from . import short_term as ST
    d = _prep_frame(matrix, pat)
    chosen = M.load_json("short_term_metrics") or {}
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "votes": VOTES, "high_net": HIGH_NET, "low_net": LOW_NET, "tiers": {}}
    for variant in ("base", "night"):
        for h in HORIZONS:
            key = ((chosen.get(str(h)) or {}).get(variant) or {}).get("chosen") or ("+asia|ens" if variant == "night" else "+asia+chip+px|ens")
            sname, mk = key.split("|")
            feats = list(ST.FEATURE_SETS[sname]) + ([ST.NIGHT_FEATURE] if variant == "night" else [])
            dd = d.dropna(subset=[ST.NIGHT_FEATURE]) if variant == "night" else d
            fy = ST.FIRST_TEST_YEAR_NIGHT if variant == "night" else ST.FIRST_TEST_YEAR
            outs = []
            if mk in ("lgb", "ens"):
                outs.append(ST._wf(dd, feats, f"fwd{h}", h, fy, lambda: ST.LgbModel()))
            if mk in ("ridge", "ens"):
                outs.append(ST._wf(dd, feats, f"fwd{h}", h, fy, lambda: ST.RidgeModel()))
            if not outs or any(x.empty for x in outs):
                continue
            o = outs[0] if len(outs) == 1 else ST._combine(outs[0], outs[1])
            o = o.merge(d[["date"] + VOTES], on="date", how="left").sort_values("date").reset_index(drop=True)
            o["call"] = 0
            yrs = sorted(o["year"].unique())
            for y in yrs[2:]:
                prev = o[o["year"] < y]["pred"]
                lo, hi = prev.quantile(ST.TIER), prev.quantile(1 - ST.TIER)
                m = o["year"] == y
                o.loc[m & (o["pred"] >= hi), "call"] = 1; o.loc[m & (o["pred"] <= lo), "call"] = -1
            o0 = o
            o = o[(o["year"] >= yrs[2]) & (o["call"] != 0)].copy()
            vs = VOTES if variant == "night" else [v for v in VOTES if v != "night"]
            sg = o[vs].apply(np.sign).fillna(0)
            o["net"] = (sg.mul(o["call"], axis=0) > 0).sum(axis=1) - (sg.mul(o["call"], axis=0) < 0).sum(axis=1)
            o["hit"] = (np.sign(o["actual"]) == o["call"]).astype(float)
            res = {"n": int(len(o)), "hit_all": round(float(o["hit"].mean()), 3), "model": key, "by_net": {}, "bucket": {}}
            for k, g in o.groupby("net"):
                yr = [gg["hit"].mean() for _, gg in g.groupby("year") if len(gg) >= 5]
                res["by_net"][str(int(k))] = {"n": int(len(g)), "hit": round(float(g["hit"].mean()), 3), "cov": round(len(g) / len(o), 3), "yr_min": round(float(min(yr)), 3) if yr else None}
            for name, msk in (("高共識", o["net"] >= HIGH_NET), ("一般", (o["net"] > LOW_NET) & (o["net"] < HIGH_NET)), ("分歧", o["net"] <= LOW_NET)):
                g = o[msk]
                yr = [gg["hit"].mean() for _, gg in g.groupby("year") if len(gg) >= 5]
                res["bucket"][name] = {"n": int(len(g)), "hit": round(float(g["hit"].mean()), 3) if len(g) else None, "cov": round(len(g) / len(o), 3), "yr_min": round(float(min(yr)), 3) if yr else None}
            # 未叫牌日：其他票的淨多票 (多 − 空) → 上漲率 (只有 base 變體有意義；night 變體叫牌覆蓋高)
            nc = o0[(o0["year"] >= yrs[2]) & (o0["call"] == 0)].copy()
            if len(nc):
                sgn = nc[vs].apply(np.sign).fillna(0)
                nc["net_abs"] = sgn.sum(axis=1); nc["up"] = (nc["actual"] > 0).astype(float)
                res["nocall"] = {"n": int(len(nc)), "base_up": round(float(nc["up"].mean()), 3), "by_net": {}}
                for k_, g in nc.groupby("net_abs"):
                    if len(g) >= 30:
                        yr = [gg["up"].mean() for _, gg in g.groupby("year") if len(gg) >= 5]
                        res["nocall"]["by_net"][str(int(k_))] = {"n": int(len(g)), "up": round(float(g["up"].mean()), 3), "yr_min": round(float(min(yr)), 3) if yr else None}
            out["tiers"][f"{h}_{variant}"] = res
            if verbose:
                print(f"  verdict h{h} {variant:<5} all {res['hit_all']}  " + "  ".join(f"{b} {v['hit']} (覆蓋 {v['cov']}, 年最低 {v['yr_min']})" for b, v in res["bucket"].items()))
    if write:
        M.save_json("verdict", out)
    return out


def _sgn(v) -> int:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0
    if v != v:
        return 0
    return 1 if v > 0 else -1 if v < 0 else 0


def _dir(s: int) -> str:
    return "多" if s > 0 else "空" if s < 0 else "—"


def build(fc: dict, hourly: dict | None, snap: dict | None, scored: pd.DataFrame | None, learn_summary: dict | None = None) -> dict | None:
    """依 refine_short_term / learn.adjust_forecast 之後的 fc 產生判斷總結 (以隔天 next_days[0] 為錨)。"""
    if not fc or fc.get("error") or not fc.get("next_days"):
        return None
    x = fc["next_days"][0]
    snap = snap or {}
    today = fc.get("date")
    call = x.get("call") or "中性"
    call_model = x.get("call_model") or call
    cs = 1 if call == "偏多" else -1 if call == "偏空" else 0
    votes: list[dict] = []
    # --- 計票層 ---
    tn = snap.get("tx_night") or {}
    night_v = float(tn["change_pct"]) if tn.get("change_pct") is not None and snap.get("phase") in ("night", "closed", "pre") else None
    if x.get("variant") == "night" or night_v is not None:
        s = _sgn(night_v) if night_v is not None and abs(night_v) >= 0.1 else 0
        votes.append({"key": "night", "name": NAMES["night"], "dir": _dir(s), "s": s, "note": f"夜盤 {night_v:+.2f}%" if night_v is not None else "夜盤未開/無資料"})
    P = fc.get("patterns") or {}
    s = _sgn(P.get("score1")) if P.get("direction1") in ("偏多", "偏空") else 0
    votes.append({"key": "rule_score", "name": NAMES["rule_score"], "dir": _dir(s), "s": s, "note": (f"有效規律合計 {P.get('direction1')} {float(P.get('score1') or 0):+.2f}%" if s else "今日無有效規律方向")})
    T = (fc.get("deep") or {}).get("today") or {}
    for k in ("hsi_r0", "kospi_r0"):
        v = T.get(k)
        s = _sgn(v) if v is not None and abs(float(v)) >= 0.1 else 0
        votes.append({"key": k, "name": NAMES[k], "dir": _dir(s), "s": s, "note": (f"{float(v):+.2f}%" if v is not None else "無資料")})
    tr = None
    try:
        if scored is not None and len(scored) >= 20:
            c = scored["close"].astype(float)
            tr = float(c.iloc[-1] / c.tail(20).mean() - 1) * 100
    except Exception:  # noqa: BLE001
        tr = None
    s = _sgn(tr)
    votes.append({"key": "tr20", "name": NAMES["tr20"], "dir": _dir(s), "s": s, "note": (f"收盤距月線 {tr:+.2f}%" if tr is not None else "無資料")})
    s2 = T.get("smart2")
    s = _sgn(s2) if s2 is not None and abs(float(s2)) >= 0.2 else 0
    votes.append({"key": "s2", "name": NAMES["s2"], "dir": _dir(s), "s": s, "note": (f"聰明錢 v2 {float(s2):+.2f}" if s2 is not None else "無資料 (期貨/選擇權籌碼未齊)")})
    agree = sum(1 for v in votes if cs and v["s"] * cs > 0)
    disagree = sum(1 for v in votes if cs and v["s"] * cs < 0)
    bull = sum(1 for v in votes if v["s"] > 0); bear = sum(1 for v in votes if v["s"] < 0)
    net = agree - disagree
    # --- 只顯示不計票的即時層 ---
    extra: list[dict] = []
    g = fc.get("trend7") or {}
    if g.get("available"):
        gs = 1 if g.get("state") == "up" else -1 if g.get("state") == "down" else 0
        extra.append({"key": "trend7", "name": "7 日趨勢閘門", "dir": _dir(gs), "s": gs, "note": f"{g.get('state')} {g.get('confidence') or ''}".strip() + (f" (7 日下跌率 {float(g['p_down']):.0%})" if g.get("p_down") is not None else "")})
    ln = []
    if x.get("call_degraded"):
        ln.append(f"近期失準已降級 (原 {call_model})")
    if x.get("p_up_adj") is not None:
        ln.append(f"校準上漲率 {float(x['p_up_adj']):.0%}")
    if x.get("recent_hit") is not None and x.get("recent_n"):
        ln.append(f"近期命中 {float(x['recent_hit']):.0%}/{int(x['recent_n'])} 次")
    if x.get("level_adj_pct"):
        ln.append(f"水準偏誤修正 {float(x['level_adj_pct']):+.2f}%")
    if ln:
        extra.append({"key": "learn", "name": "線上自學", "dir": "—", "s": 0, "note": "、".join(ln)})
    hr = hourly or {}
    t13 = (hr.get("targets") or {}).get("13:30") if hr.get("live") else None
    if t13 and t13.get("p_up") is not None:
        p, bh = float(t13["p_up"]), float(t13.get("base_hit") or 0.5)
        hs = 1 if p >= bh + 0.03 else -1 if p <= bh - 0.03 else 0
        extra.append({"key": "hourly", "name": f"盤中小時模型 {hr.get('mark')}→13:30", "dir": _dir(hs), "s": hs, "note": f"預估收盤 {float(t13.get('level') or 0):,.0f}、上漲率 {p:.0%} (基準 {bh:.0%})"})
    L1 = ((fc.get("logic") or {}).get("h") or {}).get("1") or {}
    if L1.get("rule"):
        r_, o_ = L1["rule"], L1.get("oos") or {}
        ls = 1 if r_.get("dir") == "偏多" else -1 if r_.get("dir") == "偏空" else 0
        edge = (o_.get("hit") or 0) - (o_.get("base") or 0)
        extra.append({"key": "logic", "name": "漲跌邏輯 (決策規則)", "dir": _dir(ls), "s": ls,
                      "note": f"{r_.get('text')} → {r_.get('dir')} (歷史 {r_.get('n')} 次上漲率 {float(r_.get('up_rate') or 0):.0%}；最近 8 次上漲 {float(L1.get('recent_up') or 0):.0%})"
                              + (f"；此變體樣本外命中 {o_['hit']:.0%} vs 基準 {o_['base']:.0%}" + ("，無優勢僅供觀察" if edge < 0.03 else "") if o_.get("hit") else "")})
    gp = (fc.get("precheck") or {}).get("gap") or {}
    if gp.get("stats"):
        s_ = gp["stats"]; est = float(gp.get("est") or 0)
        gs = 1 if (est > 0.15 and s_.get("p_hold", 0) >= 0.6) else -1 if (est < -0.15 and s_.get("p_hold", 0) >= 0.6) else 0
        extra.append({"key": "gap", "name": "開盤跳空預判", "dir": _dir(gs), "s": gs, "note": f"預估跳空 {est:+.2f}% ({gp.get('source')})：{gp.get('bucket')}‧回補 {s_['p_fill']:.0%}‧守住 {s_['p_hold']:.0%}‧收盤上漲率 {s_['p_up_close']:.0%} (n={s_['n']})"})
    for c_ in (fc.get("precheck") or {}).get("calendar") or []:
        if c_.get("valid"):
            extra.append({"key": "cal_" + c_["key"], "name": "日曆效應", "dir": c_["dir"][-1] if c_["dir"] in ("偏多", "偏空") else "—", "s": 1 if c_["dir"] == "偏多" else -1 if c_["dir"] == "偏空" else 0, "note": c_.get("text")})
    fv = fc.get("five") or {}
    if fv.get("call"):
        fs = 1 if fv["call"] == "偏多" else -1 if fv["call"] == "偏空" else 0
        extra.append({"key": "five", "name": "後五日 (交易日)", "dir": _dir(fs), "s": fs, "note": fv.get("text", "")})
    sc = snap.get("score") or {}
    if sc.get("score") is not None and snap.get("phase") == "day":
        ss = _sgn(float(sc["score"]) - 0) if abs(float(sc["score"])) >= 15 else 0
        extra.append({"key": "rtscore", "name": "盤中即時評分", "dir": _dir(ss), "s": ss, "note": f"{float(sc['score']):+.0f} {sc.get('label') or ''}" + ("；" + "、".join(p_["text"] for p_ in (sc.get("parts") or [])[:4]) if sc.get("parts") else "")})
    # --- 結論 ---
    st = M.load_json("verdict") or {}
    tiers = (st.get("tiers") or {}).get(f"1_{x.get('variant') or 'base'}") or {}
    bucket = "高共識" if net >= HIGH_NET else "分歧" if net <= LOW_NET else "一般"
    oos = (tiers.get("bucket") or {}).get(bucket) or {}
    conf_hit = x.get("conf_hit")
    call_action = call
    if cs == 0:
        verdict = "中性"
        head = f"模型未叫牌 (上漲率 {float(x.get('p_up') or 0):.0%}，基準 {float(x.get('base_hit') or 0.5):.0%})；其他訊號 多 {bull} / 空 {bear}"
        net_abs = bull - bear
        nc = ((st.get("tiers") or {}).get("1_base") or {}).get("nocall") or {}
        ncs = (nc.get("by_net") or {}).get(str(net_abs)) or {}
        if net_abs >= 4 and (ncs.get("up") or 0) >= 0.62:   # 第三輪研究：未叫牌日 淨多 ≥4 → 70% (n=61)
            verdict, call_action = "偏多‧訊號共識", "偏多"
            oos = {"n": ncs.get("n"), "hit": ncs.get("up"), "cov": None, "yr_min": ncs.get("yr_min")}
            head += f" → 模型無叫牌但 {len(votes)} 票中淨多 {net_abs}：歷史同狀況隔天上漲 {ncs['up']:.0%} (n={ncs['n']}，年最低 {ncs.get('yr_min') or 0:.0%})，給偏多"
        elif net_abs >= 3 and ncs.get("up"):
            head += f" → 訊號偏多 (歷史同狀況上漲 {ncs['up']:.0%}，n={ncs['n']})，略偏多但不足以叫牌、不追高"
        elif bear - bull >= 3:
            head += " → 訊號偏空，但研究顯示空方共識對下跌無預測力，不放空"
    else:
        verdict = f"{call}{'‧高共識' if bucket == '高共識' else '‧分歧' if bucket == '分歧' else ''}"
        head = f"模型{call}{x.get('call_strength') or ''}，其他 {len(votes)} 票同向 {agree}、反向 {disagree} (淨 {net:+d}) → {bucket}"
        if oos.get("hit"):
            head += f"；同狀況歷史 OOS 命中 {oos['hit']:.0%} (覆蓋 {oos['cov']:.0%}，逐年最低 {oos['yr_min']:.0%})" if oos.get("yr_min") is not None else f"；同狀況歷史 OOS 命中 {oos['hit']:.0%}"
        if x.get("conf_tier"):
            head += f"；信心分層 {x['conf_tier']}" + (f" ({float(conf_hit):.0%})" if conf_hit else "")
    # 行動：結合 7 日閘門 與 路徑買賣點
    gate = g.get("state") if g.get("available") else None
    act = []
    ba, sa, stp, tg = x.get("buy_at"), x.get("sell_at"), x.get("stop"), x.get("target")
    if call_action == "偏多" and cs == 0:
        cs = 1   # 訊號共識偏多：行動比照偏多 (下方買點邏輯)
    if cs > 0:
        if gate == "down":
            act.append("7 日閘門偏下：不推薦買點，偏多僅短打或觀望")
        elif ba:
            act.append(f"拉回 {float(ba):,.0f} 附近承接" + (f"，停損 {float(stp):,.0f}" if stp else "") + (f"，目標 {float(tg):,.0f}" if tg else ""))
        if bucket == "分歧":
            act.append("訊號分歧：縮小部位、不追價")
    elif cs < 0:
        if gate == "up":
            act.append("7 日閘門偏上：不推薦賣點，偏空僅減碼不放空")
        elif sa:
            act.append(f"反彈 {float(sa):,.0f} 附近減碼" + (f"，停損 {float(stp):,.0f}" if stp else ""))
        if bucket == "分歧":
            act.append("訊號分歧：不追空")
    else:
        if ba and sa:
            act.append(f"區間操作：{float(ba):,.0f} 承接 / {float(sa):,.0f} 減碼" + (f"，停損 {float(stp):,.0f}" if stp else ""))
        if gate == "down":
            act.append("7 日閘門偏下：不推薦買點")
        elif gate == "up":
            act.append("7 日閘門偏上：不推薦賣點")
    for e in extra:
        if e["key"] == "hourly" and e["s"] and cs and e["s"] != cs:
            act.append(f"盤中小時模型與日模型相反 ({e['note']})：以盤中為準縮小部位")
        if e["key"] == "rtscore" and e["s"] and cs and e["s"] != cs:
            act.append("盤中即時評分與叫牌相反：等盤中訊號轉向再動作")
    return {"date": today, "target": x.get("date"), "verdict": verdict, "call": call, "call_action": call_action, "net": net, "agree": agree, "disagree": disagree, "bull": bull, "bear": bear,
            "bucket": bucket if cs else None, "oos": (oos or None) if cs else None, "votes": votes, "extra": extra, "head": head, "action": "；".join(act) or "照常依買賣點操作",
            "text": f"判斷總結：{verdict}。{head}。{'；'.join(act) if act else ''}".rstrip("。") + "。"}
