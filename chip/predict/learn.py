"""線上自學 (2026-09-22)：把每天發布的預測記成帳本，用真實走勢逐日對帳，再依「近期實際命中率」調整下一次預測。

離線 walk-forward 只能保證「歷史上」有效；市場狀態一變，命中率會先於重訓反映在近期帳本上。這個模組做三件事：
1. 帳本 (ledger)：每次發布 forecast.json / watchlist 個股預測時，記下 as_of 日、目標日、叫牌、p_up、買賣點水準、trend7 狀態；
   帳本存在已發布的 Pages `learn.json` 裡 (與 gov8.json 相同的「拉回上次 ∪ 本次」累積機制)，SQLite snapshots 另存一份備援。
2. 對帳 (evaluate)：目標日收盤已知後填入實際報酬、方向是否命中、買點/賣點/停損/目標是否被觸及、Brier。
3. 自適應 (adjust)：
   - 方向：近 N 次 (含權重：越近越重) 叫牌命中率 vs 模型長期 call_hit；顯著落後 (差 >5pt 且 n>=20) → 該視野降級為「中性 (近期失準)」，
     並用近期資料重新校準 p_up (Platt 縮放，樣本少時收縮回原值)。
   - 買賣點：近 60 次買點觸及率 vs 目標 20% → 水準乘數在 0.85~1.35 間調整 (觸及太多 → 放寬，太少 → 收窄)。
   - 個股：每檔追蹤股 5/10/20 日相對大盤方向的近期命中率；連續落後 → 提示「模型對此股近期不準」。
所有調整都只用「已對帳」的紀錄，不看未來；調整值同時寫進 forecast.json (p_up_adj / recent_hit / learn_note) 與 learn.json。
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os

import numpy as np
import pandas as pd

from .. import config, store
from ..http import session

log = logging.getLogger(__name__)
PAGES_URL = os.getenv("CHIP_PAGES_URL", "https://joekisoul-code.github.io/TaiexChipAnalyzer/")
MAX_LEDGER = 3000
RECENT_N = (20, 60)
HALF_LIFE = 30          # 近期命中率的指數衰減半衰期 (以叫牌次數計)
MIN_N_ADJ = 20          # 至少 20 筆已對帳才做調整
DEGRADE_GAP = 0.05      # 近期命中低於長期 5 個百分點以上 → 降級
TOUCH_TARGET = 0.20


# ------------------------------------------------------------------ 已發布帳本
def published() -> dict:
    try:
        r = session().get(PAGES_URL.rstrip("/") + "/data/learn.json", timeout=20)
        if r.ok and r.text.strip().startswith("{"):
            return r.json() or {}
    except Exception as e:  # noqa: BLE001
        log.debug("published learn.json unavailable: %s", e)
    return {}


def _key(r: dict) -> str:
    return f"{r.get('kind')}|{r.get('sid') or 'TAIEX'}|{r.get('as_of')}|{r.get('target')}|{r.get('h')}|{r.get('mode')}"


def _merge(prev_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """同 key 以「盤後正式版」優先於「盤中近似版」，其餘保留最早一筆 (預測一旦記下就不改，才是誠實對帳)。"""
    m: dict[str, dict] = {}
    for r in prev_rows:
        m[_key(r)] = r
    for r in new_rows:
        k = _key(r)
        old = m.get(k)
        if old is None:
            m[k] = r
        elif old.get("live") and not r.get("live"):
            m[k] = {**r, "realized": old.get("realized")}
    rows = sorted(m.values(), key=lambda x: (x.get("as_of", ""), x.get("kind", ""), x.get("sid", ""), x.get("h", 0)))
    return rows[-MAX_LEDGER:]


# ------------------------------------------------------------------ 記錄
def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:  # noqa: BLE001
        return None


def records_from_forecast(fc: dict, snap: dict | None) -> list[dict]:
    """從 forecast.json 的 forecast 物件抽出要對帳的預測。"""
    if not fc or fc.get("error"):
        return []
    live = bool(fc.get("intraday"))
    as_of = str(fc.get("date"))
    base = _num((fc.get("intraday") or {}).get("price")) if live else _num(fc.get("close"))
    if live:   # 盤中：基準是「今天」的即時價，對帳日從今天起算 (forecast.date 仍是前一收盤日)
        as_of = str((snap or {}).get("ts") or "")[:10] or dt.date.today().isoformat()
    mode = "live" if live else ((snap or {}).get("phase") or "closed")
    rows = []
    for x in fc.get("next_days") or []:
        rows.append({"kind": "mkt", "sid": "TAIEX", "as_of": as_of, "target": str(x.get("date")), "h": int(x.get("n") or 0), "mode": "live" if live else "close",
                     "phase": mode, "live": live, "base": base, "p_up": _num(x.get("p_up")), "base_hit": _num(x.get("base_hit")), "call": x.get("call") or "中性",
                     "strength": x.get("call_strength") or "", "call_hit": _num(x.get("call_hit")), "variant": x.get("variant"), "level": _num(x.get("level")),
                     "buy_at": _num(x.get("buy_at")), "sell_at": _num(x.get("sell_at")), "stop": _num(x.get("stop")), "target_px": _num(x.get("target")),
                     "range_mode": x.get("range_mode"), "trend7": (fc.get("trend7") or {}).get("state"), "realized": None})
    for h in (5, 10, 20):
        r = (fc.get("horizons") or {}).get(h) or (fc.get("horizons") or {}).get(str(h))
        if not r:
            continue
        rows.append({"kind": "mkt", "sid": "TAIEX", "as_of": as_of, "target": None, "h": h, "mode": "live" if live else "close", "phase": mode, "live": live, "base": base,
                     "p_up": _num(r.get("p_up")), "base_hit": _num(r.get("base_hit")), "call": r.get("call") or ("偏多" if (_num(r.get("p_up")) or 0) >= (_num(r.get("base_hit")) or 0.5) + 0.03 else "偏空" if (_num(r.get("p_up")) or 0) <= (_num(r.get("base_hit")) or 0.5) - 0.03 else "中性"),
                     "strength": r.get("call_strength") or "", "call_hit": _num(r.get("call_hit")), "variant": r.get("variant") or "daily", "level": None,
                     "buy_at": None, "sell_at": None, "stop": None, "target_px": None, "range_mode": None, "trend7": (fc.get("trend7") or {}).get("state"), "realized": None})
    return rows


def records_from_hourly(hr: dict, snap: dict | None) -> list[dict]:
    """盤中每次發布把小時模型「現在 → 13:30 收盤」的即時叫牌記下 (kind='hr')，收盤後對帳 → 即時修正的真實命中率。"""
    if not hr or hr.get("error") or not hr.get("live"):
        return []
    t = (hr.get("targets") or {}).get("13:30")
    if not t or t.get("p_up") is None:
        return []
    p = _num(t.get("p_up")); bh = _num(t.get("base_hit")) or 0.5
    call = "偏多" if p >= bh + 0.03 else "偏空" if p <= bh - 0.03 else "中性"
    mark = str(hr.get("mark") or "")
    return [{"kind": "hr", "sid": "TAIEX", "as_of": str(hr.get("day")), "target": str(hr.get("day")), "h": 0, "mode": "live:" + mark, "phase": "open", "live": False,
             "base": _num(hr.get("price")), "p_up": p, "base_hit": bh, "call": call, "strength": "", "call_hit": None, "variant": "hourly", "mark": mark,
             "level": _num(t.get("level")), "buy_at": None, "sell_at": None, "stop": None, "target_px": None, "range_mode": None, "trend7": None, "realized": None}]


def records_from_stock(sid: str, sf: dict) -> list[dict]:
    if not sf or sf.get("error"):
        return []
    rows = []
    for h, r in (sf.get("horizons") or {}).items():
        p = _num(r.get("p_up")); bh = _num(r.get("base_hit")) or 0.5
        call = "偏多" if p is not None and p >= bh + 0.03 else "偏空" if p is not None and p <= bh - 0.03 else "中性"
        rows.append({"kind": "stk", "sid": str(sid), "as_of": str(sf.get("date")), "target": None, "h": int(h), "mode": "close", "phase": "closed", "live": False,
                     "base": _num(sf.get("close")), "p_up": p, "base_hit": bh, "call": call, "strength": "", "call_hit": None, "variant": "stock", "level": None,
                     "buy_at": None, "sell_at": None, "stop": None, "target_px": None, "range_mode": None, "trend7": None, "pred": _num(r.get("pred")), "realized": None})
    return rows


# ------------------------------------------------------------------ 對帳
def _price_map(frame: pd.DataFrame) -> tuple[list[str], dict]:
    d = frame.dropna(subset=["close"]).copy()
    d["date"] = d["date"].astype(str).str[:10]
    dates = d["date"].tolist()
    px = {r.date: {"close": float(r.close), "high": float(r.high) if "high" in d and pd.notna(r.high) else float(r.close),
                   "low": float(r.low) if "low" in d and pd.notna(r.low) else float(r.close)} for r in d.itertuples()}
    return dates, px


def evaluate(rows: list[dict], frames: dict[str, pd.DataFrame], market_close: dict[str, float] | None = None) -> int:
    """填入 realized；frames: {'TAIEX': scored, sid: stock price frame}。個股用「相對大盤」報酬 (與 stock_forecast 的 xfwd 一致)。回傳新對帳筆數。"""
    done = 0
    cache = {}
    for r in rows:
        if r.get("realized") or r.get("p_up") is None:
            continue
        fr = frames.get(r.get("sid") or "TAIEX")
        if fr is None or fr.empty:
            continue
        if r["sid"] not in cache:
            cache[r["sid"]] = _price_map(fr)
        dates, px = cache[r["sid"]]
        as_of = str(r["as_of"])[:10]
        if as_of not in px:
            continue
        i0 = dates.index(as_of)
        h = int(r.get("h") or 0)
        if h == 0 and r.get("kind") == "hr":   # 小時模型：當日收盤 vs 記帳時的現價 (呼叫端保證 scored 已含當日收盤)
            t_date = as_of
            base = float(r.get("base") or px[as_of]["close"])
            y = (px[t_date]["close"] / base - 1) * 100
            call = r.get("call") or "中性"
            r["realized"] = {"date": t_date, "ret": round(y, 3), "rel": None, "up": bool(y > 0), "hit": None if call == "中性" else bool((y > 0) if call == "偏多" else (y < 0)),
                             "brier": round((float(r["p_up"]) - (1.0 if y > 0 else 0.0)) ** 2, 4)}
            done += 1
            continue
        if h <= 0:
            continue
        t_date = str(r.get("target") or "")[:10]
        if t_date and t_date in px:
            pass
        elif t_date and t_date > dates[-1]:
            continue                      # 目標日尚未到
        elif i0 + h < len(dates):
            t_date = dates[i0 + h]
        else:
            continue
        base = float(r.get("base") or px[as_of]["close"])
        ret = (px[t_date]["close"] / base - 1) * 100
        rel = None
        if r.get("kind") == "stk" and market_close:
            m0, m1 = market_close.get(as_of), market_close.get(t_date)
            if m0 and m1:
                rel = ret - (m1 / m0 - 1) * 100
        y = rel if rel is not None else ret
        call = r.get("call") or "中性"
        hit = None if call == "中性" else bool((y > 0) if call == "偏多" else (y < 0))
        p = float(r["p_up"])
        rz = {"date": t_date, "ret": round(ret, 3), "rel": round(rel, 3) if rel is not None else None, "up": bool(y > 0), "hit": hit,
              "brier": round((p - (1.0 if y > 0 else 0.0)) ** 2, 4)}
        if r.get("buy_at") and h <= 3:
            lo = min(px[dates[j]]["low"] for j in range(i0 + 1, i0 + h + 1))
            hi = max(px[dates[j]]["high"] for j in range(i0 + 1, i0 + h + 1))
            rz.update({"path_low": round(lo, 2), "path_high": round(hi, 2), "buy_touch": bool(lo <= r["buy_at"]), "sell_touch": bool(r.get("sell_at") and hi >= r["sell_at"]),
                       "stop_hit": bool(r.get("stop") and lo <= r["stop"]), "target_hit": bool(r.get("target_px") and hi >= r["target_px"])})
        r["realized"] = rz
        done += 1
    return done


# ------------------------------------------------------------------ 統計與自適應
def _ewm_hit(hits: list[bool], half_life: int = HALF_LIFE) -> float | None:
    if not hits:
        return None
    lam = 0.5 ** (1.0 / half_life)
    w = np.array([lam ** k for k in range(len(hits))][::-1])
    return float(np.dot(w, np.array(hits, dtype=float)) / w.sum())


def _platt(ps: list[float], ys: list[int]) -> tuple[float, float]:
    """簡單 Platt 縮放 (logit 上做 a·x+b)，梯度下降幾百步；樣本少時外層再收縮。"""
    x = np.array([math.log(max(1e-4, p) / max(1e-4, 1 - p)) for p in ps]); y = np.array(ys, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(400):
        z = 1 / (1 + np.exp(-(a * x + b)))
        ga, gb = float(np.mean((z - y) * x)) + 0.02 * (a - 1), float(np.mean(z - y))
        a -= 0.3 * ga; b -= 0.3 * gb
    return float(np.clip(a, 0.3, 2.0)), float(np.clip(b, -1.0, 1.0))


def _apply_platt(a: float, b: float, p: float) -> float:
    x = math.log(max(1e-4, p) / max(1e-4, 1 - p))
    return 1 / (1 + math.exp(-(a * x + b)))


def _group_stats(rs: list[dict]) -> dict:
    ev = [r for r in rs if r.get("realized")]
    calls = [r for r in ev if r["realized"].get("hit") is not None]
    hits = [bool(r["realized"]["hit"]) for r in calls]
    out = {"n_eval": len(ev), "n_calls": len(calls), "n_pending": len([r for r in rs if not r.get("realized")]),
           "hit_all": round(float(np.mean(hits)), 3) if hits else None,
           "hit_ewm": round(_ewm_hit(hits), 3) if hits else None,
           "brier": round(float(np.mean([r["realized"]["brier"] for r in ev])), 4) if ev else None,
           "model_hit": round(float(np.nanmean([r["call_hit"] for r in calls if r.get("call_hit") is not None])), 3) if any(r.get("call_hit") is not None for r in calls) else None,
           "base_hit": round(float(np.nanmean([r["base_hit"] for r in ev if r.get("base_hit") is not None])), 3) if ev else None,
           "up_rate": round(float(np.mean([r["realized"]["up"] for r in ev])), 3) if ev else None}
    for n in RECENT_N:
        hh = hits[-n:]
        out[f"hit{n}"] = round(float(np.mean(hh)), 3) if len(hh) >= 5 else None
        out[f"n{n}"] = len(hh)
    # 自適應
    adj = {"factor": 1.0, "platt": None, "degrade": False, "note": ""}
    if len(ev) >= MIN_N_ADJ:
        a, b = _platt([float(r["p_up"]) for r in ev], [1 if r["realized"]["up"] else 0 for r in ev])
        k = min(1.0, (len(ev) - MIN_N_ADJ) / 60.0)        # 樣本越多越信近期校準；20 筆 → 0，80 筆 → 1
        adj["platt"] = [round(1 + (a - 1) * k, 3), round(b * k, 3), round(k, 2)]
    ref = out["model_hit"] or out["base_hit"]
    if out["n_calls"] >= MIN_N_ADJ and out["hit_ewm"] is not None and ref is not None:
        gap = out["hit_ewm"] - ref
        if gap <= -DEGRADE_GAP:
            adj["degrade"] = True
            adj["note"] = f"近期命中 {out['hit_ewm']:.0%} 低於模型長期 {ref:.0%}，此視野暫改中性"
        elif gap >= DEGRADE_GAP:
            adj["note"] = f"近期命中 {out['hit_ewm']:.0%} 高於長期 {ref:.0%}"
    out["adjust"] = adj
    return out


def _touch_stats(rs: list[dict]) -> dict:
    ev = [r for r in rs if r.get("realized") and "buy_touch" in r["realized"]]
    if not ev:
        return {"n": 0}
    ev = ev[-60:]
    b = float(np.mean([r["realized"]["buy_touch"] for r in ev])); s = float(np.mean([r["realized"]["sell_touch"] for r in ev]))
    # 觸及率 vs 目標 20%：太常碰到 → 水準太近 → 放寬 (乘數 >1)；太少 → 收窄。以 sqrt 緩和，限制 0.85~1.35
    fac = float(np.clip(math.sqrt(max(0.05, (b + s) / 2) / TOUCH_TARGET), 0.85, 1.35)) if len(ev) >= 20 else 1.0
    return {"n": len(ev), "buy_touch": round(b, 3), "sell_touch": round(s, 3), "stop_hit": round(float(np.mean([r["realized"]["stop_hit"] for r in ev])), 3),
            "target_hit": round(float(np.mean([r["realized"]["target_hit"] for r in ev])), 3), "sigma_factor": round(fac, 3),
            "note": ("買賣點近期觸及率 %.0f%%/%.0f%% (目標 20%%) → 水準乘數 ×%.2f" % (b * 100, s * 100, fac)) if len(ev) >= 20 else "樣本未達 20 次，未調整"}


def summarize(rows: list[dict]) -> dict:
    mk = [r for r in rows if r.get("kind") == "mkt" and not r.get("live")]
    out = {"market": {"by_h": {}, "touch": {}, "trend7": None, "recent": []}, "stocks": {}}
    for h in (1, 2, 3, 5, 10, 20):
        rs = [r for r in mk if r.get("h") == h]
        if rs:
            g = _group_stats(rs)
            g["by_variant"] = {v: _group_stats([r for r in rs if r.get("variant") == v]) for v in sorted({r.get("variant") or "" for r in rs}) if v}
            out["market"]["by_h"][str(h)] = g
    for h in (1, 2, 3):
        out["market"]["touch"][str(h)] = _touch_stats([r for r in mk if r.get("h") == h])
    t7 = [r for r in mk if r.get("h") == 5 and r.get("trend7") and r.get("realized")]
    if t7:
        st = {}
        for s in ("down", "up", "flat"):
            xs = [r["realized"]["ret"] for r in t7 if r["trend7"] == s]
            if xs:
                st[s] = {"n": len(xs), "mean5": round(float(np.mean(xs)), 2), "pneg": round(float(np.mean([x < 0 for x in xs])), 2)}
        out["market"]["trend7"] = st
    # 小時模型即時叫牌 (盤中 → 13:30)：依時間點統計
    hr_rows = [r for r in rows if r.get("kind") == "hr"]
    if hr_rows:
        out["market"]["hourly"] = {"all": _group_stats(hr_rows), "by_mark": {mk_: _group_stats([r for r in hr_rows if r.get("mark") == mk_]) for mk_ in sorted({r.get("mark") or "" for r in hr_rows}) if mk_}}
    # 水準偏誤 (即時修正)：近期「預估收盤 vs 實際收盤」的帶號誤差 (相對基準價 %)，指數衰減；下次預測依此微調水準
    lb = {}
    for h in (1, 2, 3):
        rs = [r for r in mk if r.get("h") == h and r.get("realized") and r.get("level") and r.get("base")]
        errs = [((r["realized"]["ret"] / 100 + 1) * r["base"] - r["level"]) / r["base"] * 100 for r in rs]
        if len(errs) >= 10:
            lam = 0.5 ** (1 / 20); w = np.array([lam ** k for k in range(len(errs))][::-1])
            bias = float(np.dot(w, np.array(errs)) / w.sum()); mae = float(np.mean(np.abs(errs[-60:])))
            k = min(1.0, (len(errs) - 10) / 40)   # 10 筆開始、50 筆全信
            side = "低" if bias > 0 else "高"
            lb[str(h)] = {"n": len(errs), "bias": round(bias, 3), "mae": round(mae, 3), "adj": round(bias * k * 0.5, 3), "note": f"近期預估收盤平均偏{side} {abs(bias):.2f}%，下次水準修正 {bias * k * 0.5:+.2f}%"}
    out["market"]["level_bias"] = lb
    rec = [r for r in mk if r.get("h") in (1, 2, 3, 5) and r.get("call") != "中性"][-40:]
    out["market"]["recent"] = [{"as_of": r["as_of"], "h": r["h"], "target": r.get("target") or (r["realized"] or {}).get("date"), "call": r["call"] + (r.get("strength") or ""),
                                "p_up": r["p_up"], "ret": (r["realized"] or {}).get("ret"), "hit": (r["realized"] or {}).get("hit"), "variant": r.get("variant")} for r in rec][::-1]
    for sid in sorted({r["sid"] for r in rows if r.get("kind") == "stk"}):
        rs = [r for r in rows if r.get("kind") == "stk" and r["sid"] == sid]
        o = {"by_h": {str(h): _group_stats([r for r in rs if r.get("h") == h]) for h in (5, 10, 20) if any(r.get("h") == h for r in rs)}}
        o["recent"] = [{"as_of": r["as_of"], "h": r["h"], "call": r["call"], "p_up": r["p_up"], "rel": (r["realized"] or {}).get("rel"), "hit": (r["realized"] or {}).get("hit")} for r in rs if r.get("call") != "中性"][-15:][::-1]
        out["stocks"][sid] = o
    return out


# ------------------------------------------------------------------ 套用到本次預測
def adjust_forecast(fc: dict, summary: dict) -> dict:
    """依近期對帳結果把 p_up_adj / recent_hit / learn_note 寫進 next_days 與 horizons；降級時 call 改中性 (原 call 保留在 call_model)。"""
    if not fc or fc.get("error"):
        return fc
    byh = (summary.get("market") or {}).get("by_h") or {}
    def apply(x: dict, h: int):
        g = byh.get(str(h))
        if not g:
            return
        # 2026-09-23：依變體對帳。帳本多為不含夜盤 (15:40 正式版)，含夜盤叫牌 (歷史 85%) 不能被不含夜盤的失準降級；
        # 同變體對帳 ≥ MIN_N_ADJ 才用其校準/降級，否則只顯示近期命中並註明變體。
        v = x.get("variant") or ""
        gv = (g.get("by_variant") or {}).get(v) or {}
        if v and gv.get("n_calls", 0) >= MIN_N_ADJ:
            g = gv
        elif v and v not in ("base", "daily", ""):   # 同變體對帳不足 (含 n_calls=0)
            x["recent_hit"] = g.get("hit_ewm"); x["recent_n"] = g.get("n_calls")
            x["learn_note"] = f"近期對帳為不含夜盤紀錄 (命中 {g['hit_ewm']:.0%}，n={g['n_calls']})，不套用到含夜盤叫牌" if g.get("n_calls") else ""
            return
        adj = g.get("adjust") or {}
        p = _num(x.get("p_up"))
        if p is not None and adj.get("platt"):
            a, b, k = adj["platt"]
            x["p_up_adj"] = round(_apply_platt(a, b, p), 3)
        x["recent_hit"] = g.get("hit_ewm"); x["recent_n"] = g.get("n_calls"); x["recent_hit20"] = g.get("hit20")
        note = adj.get("note") or ""
        if adj.get("degrade") and x.get("call") not in (None, "中性"):
            x["call_model"] = x.get("call"); x["call_strength_model"] = x.get("call_strength") or ""
            x["call"] = "中性"; x["call_strength"] = ""
            x["call_degraded"] = True
        if g.get("n_calls"):
            note = (note + "；" if note else "") + f"近期實際命中 {g['hit_ewm']:.0%} (n={g['n_calls']})"
        x["learn_note"] = note
    for x in fc.get("next_days") or []:
        apply(x, int(x.get("n") or 0))
    for h, r in (fc.get("horizons") or {}).items():
        if isinstance(r, dict):
            apply(r, int(h))
    lb = (summary.get("market") or {}).get("level_bias") or {}
    for x in fc.get("next_days") or []:
        b = lb.get(str(x.get("n")))
        if b and b.get("adj") and _num(x.get("level")) and _num(fc.get("close")):
            base_px = _num((fc.get("intraday") or {}).get("price")) or _num(fc.get("close"))
            x["level_model"] = x["level"]
            x["level"] = round(x["level"] + base_px * b["adj"] / 100)
            x["level_adj_pct"] = b["adj"]; x["level_bias_note"] = b["note"]
    hr = (summary.get("market") or {}).get("hourly") or {}
    if hr:
        fc["hourly_learn"] = {"all": {k: hr["all"].get(k) for k in ("n_calls", "hit_ewm", "hit20", "hit_all", "base_hit", "brier")}, "by_mark": {m: {k: v.get(k) for k in ("n_calls", "hit_ewm", "hit_all")} for m, v in (hr.get("by_mark") or {}).items()}}
    tt = (summary.get("market") or {}).get("touch") or {}
    fc["learn"] = {"touch_sigma_factor": {h: v.get("sigma_factor") for h, v in tt.items()}, "note": "依近期對帳自適應：p_up_adj=近期 Platt 校準、call_degraded=近期失準改中性"}
    return fc


def touch_factor(summary: dict, h: int) -> float:
    try:
        return float((summary["market"]["touch"][str(h)] or {}).get("sigma_factor") or 1.0)
    except Exception:  # noqa: BLE001
        return 1.0


# ------------------------------------------------------------------ 主流程
BACKFILL_DAYS = 160


def backfill(scored: pd.DataFrame, days: int = BACKFILL_DAYS) -> list[dict]:
    """帳本回填 (2026-09-23)：用「今年之前資料訓練」的短線模型 (不含夜盤變體，與 15:40 正式版同) 對今年最近 days 個交易日做樣本外預測，
    以與 records_from_forecast 相同欄位記入帳本 (backfill=True、mode=close)，讓 Platt 校準 / 失準降級 / 水準偏誤 立刻有 ≥20 筆對帳，不必等數週。
    已存在的真實紀錄優先 (_merge 保留先前)；只回填 kind=mkt h=1/2/3/5。"""
    try:
        from . import model as M, short_term as ST
        try:   # 雲端 market.run 的 scored 只有近幾年 → 用長歷史重建，才有 ≥400 列可訓練 (實際發生：Actions 回填 0 筆)
            from ..analysis import backtest as _bt
            long = _bt.load_long("2010-01-01")
            if len(long) > len(scored):
                scored = long
        except Exception as e:  # noqa: BLE001
            log.warning("backfill load_long: %s", e)
        mat = ST.build_matrix(scored, None)
        year = int(str(mat["date"].iloc[-1])[:4])
        rows = []
        for h in (1, 2, 3, 5):
            b = M.load(f"st_h{h}_base")
            if not b:
                continue
            sname, mk = b["chosen"].split("|")
            feats = list(ST.FEATURE_SETS[sname])
            outs = []
            if mk in ("lgb", "ens"):
                outs.append(ST._wf(mat, feats, f"fwd{h}", h, year, lambda: ST.LgbModel()))
            if mk in ("ridge", "ens"):
                outs.append(ST._wf(mat, feats, f"fwd{h}", h, year, lambda: ST.RidgeModel()))
            if not outs or any(x.empty for x in outs):
                continue
            o = outs[0] if len(outs) == 1 else ST._combine(outs[0], outs[1])
            o = o.tail(days)
            rep, t = b["report"], b["report"]["tiers"]
            dates = mat["date"].astype(str).tolist(); closes = mat["close"].astype(float).tolist()
            idx = {d_: i for i, d_ in enumerate(dates)}
            for r in o.itertuples():
                i = idx.get(str(r.date))
                if i is None or i + h >= len(dates):
                    continue
                pred = float(r.pred); cal = M.apply_calibration(rep["calibration"], pred)
                call, call_hit, strength = "中性", t.get("mid_up"), ""
                if t.get("up_on") and pred >= t["edge_hi"]:
                    call, call_hit = "偏多", t["up_hit"]
                    if t.get("strong_hi") is not None and pred >= t["strong_hi"] and (t.get("up_hit_strong") or 0) >= t["up_hit"]:
                        strength, call_hit = "強", t["up_hit_strong"]
                elif t.get("dn_on") and pred <= t["edge_lo"]:
                    call, call_hit = "偏空", t["dn_hit"]
                    if t.get("strong_lo") is not None and pred <= t["strong_lo"] and (t.get("dn_hit_strong") or 0) >= t["dn_hit"]:
                        strength, call_hit = "強", t["dn_hit_strong"]
                base_px = closes[i]
                rows.append({"kind": "mkt", "sid": "TAIEX", "as_of": dates[i], "target": dates[i + h] if h <= 3 else None, "h": h, "mode": "close", "phase": "closed", "live": False,
                             "base": base_px, "p_up": cal["p_up"], "base_hit": cal["base_hit"], "call": call, "strength": strength, "call_hit": round(float(call_hit), 3) if call_hit is not None else None,
                             "variant": "base", "level": round(base_px * (1 + (cal["hist_mean"] or 0) / 100)) if h <= 3 else None, "buy_at": None, "sell_at": None, "stop": None, "target_px": None,
                             "range_mode": None, "trend7": None, "realized": None, "backfill": True})
        print(f"  learn backfill: {len(rows)} rows")
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"  learn backfill failed: {e}")
        return []


def run(fc: dict, snap: dict | None, scored: pd.DataFrame, stock_forecasts: dict[str, dict] | None = None, stock_frames: dict[str, pd.DataFrame] | None = None,
        prev: dict | None = None, hourly: dict | None = None) -> dict:
    prev = prev if prev is not None else published()
    rows = list(prev.get("ledger") or [])
    new = records_from_forecast(fc, snap) + records_from_hourly(hourly, snap)
    for sid, sf in (stock_forecasts or {}).items():
        new += records_from_stock(sid, sf)
    rows = _merge(rows, new)
    n_eval_mkt = sum(1 for r in rows if r.get("kind") == "mkt" and r.get("h") == 1 and r.get("realized"))
    if n_eval_mkt < MIN_N_ADJ and not any(r.get("backfill") for r in rows):   # 首次：回填今年樣本外預測，讓自學層立刻啟動
        bf = backfill(scored)
        rows = _merge(rows, bf)
        log.info("ledger backfill: %d rows", len(bf))
    frames = {"TAIEX": scored, **(stock_frames or {})}
    mclose = {str(r.date)[:10]: float(r.close) for r in scored.dropna(subset=["close"]).itertuples()}
    n_eval = evaluate(rows, frames, mclose)
    summ = summarize(rows)
    try:
        store.save_snapshot(dt.datetime.now(config.TZ).strftime("%Y-%m-%d"), "pred_ledger", {"n": len(rows), "rows": rows[-400:]})
    except Exception as e:  # noqa: BLE001
        log.debug("ledger snapshot: %s", e)
    return {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "n_ledger": len(rows), "n_new": len(new), "n_evaluated_now": n_eval, "n_backfill": sum(1 for r in rows if r.get("backfill")),
            "market": summ["market"], "stocks": summ["stocks"], "ledger": rows,
            "method": {"half_life": HALF_LIFE, "min_n_adj": MIN_N_ADJ, "degrade_gap": DEGRADE_GAP, "touch_target": TOUCH_TARGET,
                       "desc": "帳本只記發布當下的預測，目標日收盤後對帳；近期命中以指數衰減加權 (半衰期 30 次)；p_up 以近期 Platt 校準 (樣本 20→80 筆逐步信任)；"
                               "近期命中低於長期 5pt 以上的視野降為中性；買賣點水準依近 60 次觸及率調整乘數 (0.85~1.35)。個股為相對大盤方向。"}}
