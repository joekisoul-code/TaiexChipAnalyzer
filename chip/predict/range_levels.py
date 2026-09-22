"""路徑型買賣點水準 (拉回 X 買 / 反彈 Y 賣)：未來 k 日 (k=1,2,3) 路徑最低 / 最高 相對今日收盤的 10/20/80/90 分位。

研究結論 (2010~ 日 OHLC，2014~2026 逐年擴張視窗樣本外，見 research/range)：
- 收盤報酬 q20/q80 當成低/高點時，盤中路徑「觸及率」為 34~38% 而非 20%，且逐年在 13%~68% 間漂移。
- 波動縮放：sigma_t = 0.5×(ATR14/close% + EWMA(0.94) 日波動%)，level = m_q × sigma_t
  → q20/q80 觸及率 19~22%、q10/q90 9.5~12%，逐年平均誤差 2~4 個百分點 (個別年份 15~29%)。
  相對「路徑分位但不做波動縮放」的公平基準，賣方 (high) pinball 低 11~20%、買方 (low) 只低 ~3%；
  真正穩健的好處是 (a) 用路徑目標取代收盤分位 (b) 逐年校準穩定 (c) 夜盤位移。
- 夜盤已知 (夜盤收盤後~開盤前)：level = beta_k×夜盤% + m'_q×sigma_t，k=1 pinball 再降 25~31%，k=2/3 降 12~18% (7/7 年)。
  beta 以「完整夜盤收盤」擬合 → 只在夜盤結束後 (phase closed/pre) 使用；夜盤進行中改用 base 並標示近似。
- LightGBM 分位迴歸僅再降 1~4% 且覆蓋率偏淺不穩 → 不採用；狀態變數條件乘數增益 <2% → 不採用。
- 可操作性 (OOS)：掛在 low20 等拉回、k 日收盤出場平均為負 (12/13 年)，水準是「校準過的機率帶」不是 alpha 訊號。
  前端文案：「約兩成機率觸及 (以加權指數計，逐年 15~29%)」；台指期觸及會比指數多 15~20% (基差平均 -0.2%)。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)
DEFAULT_PATH = config.DATA_DIR / "models" / "range_levels.json"

KS = (1, 2, 3)
QUANTS = {"low10": 0.10, "low20": 0.20, "high80": 0.80, "high90": 0.90}
EWMA_LAMBDA = 0.94
ATR_N = 14
NIGHT_CLIP = 8.0        # 夜盤 |%| 超過此值視為異常 (fit 時剔除，預測時裁切)
TOUCH_PROB = 0.2        # buy_at / sell_at 的校準觸及機率 (q20 / q80)
COVERAGE_YEARS = 5      # JSON 內附最近 N 年逐年觸及率 (監控用)
_CACHE: dict = {}


# ------------------------------------------------------------------ sigma / targets
def sigma_series(frame: pd.DataFrame) -> pd.Series:
    """sigma_t (%) = 0.5 × (ATR14/close% + EWMA(0.94) 日報酬波動%)；只用到 t 日收盤前資訊。frame 需有 open/high/low/close (依日期排序)。"""
    c = pd.to_numeric(frame["close"], errors="coerce").astype(float).ffill()
    pc = c.shift(1)
    # 2026-09-22：雲端 (Actions) 的 scored 最新幾列 high/low 可能缺值 (某些價格來源只有收盤) → 以 max/min(收盤, 前收, 開盤) 補，
    # 否則 ATR14 會整段變 NaN，路徑型買賣點水準就不會輸出 (實際發生：09-14~09-21 線上 forecast.json 都沒有 buy_at)。
    o = pd.to_numeric(frame["open"], errors="coerce").astype(float) if "open" in frame else c
    hi_fb, lo_fb = pd.concat([c, pc, o], axis=1).max(axis=1), pd.concat([c, pc, o], axis=1).min(axis=1)
    h = pd.to_numeric(frame["high"], errors="coerce").astype(float) if "high" in frame else hi_fb
    l = pd.to_numeric(frame["low"], errors="coerce").astype(float) if "low" in frame else lo_fb
    h, l = h.fillna(hi_fb), l.fillna(lo_fb)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N, min_periods=max(5, ATR_N // 2)).mean() / c * 100
    r = np.nan_to_num((c.pct_change() * 100).values, nan=0.0, posinf=0.0, neginf=0.0)
    v2 = np.zeros(len(r))
    v2[0] = float(np.var(r[: min(60, len(r))])) if len(r) > 1 else 1.0
    for i in range(1, len(r)):
        v2[i] = EWMA_LAMBDA * v2[i - 1] + (1 - EWMA_LAMBDA) * r[i] ** 2
    ew = pd.Series(np.sqrt(v2), index=frame.index)
    return (0.5 * (atr + ew)).fillna(ew)   # ATR 仍缺時退回 EWMA-only


def path_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """未來 1..k 日路徑最低 / 最高 相對今日收盤 (%)。"""
    c, h, l = frame["close"].astype(float), frame["high"].astype(float), frame["low"].astype(float)
    out = pd.DataFrame(index=frame.index)
    for k in KS:
        # skipna=False：最後 k-1 列未來路徑不完整 → NaN (預設 skipna 會用部分路徑當目標，讓擬合與觸及率多算 1~2 列)
        out[f"pathLow{k}"] = (pd.concat([l.shift(-j) for j in range(1, k + 1)], axis=1).min(axis=1, skipna=False) / c - 1) * 100
        out[f"pathHigh{k}"] = (pd.concat([h.shift(-j) for j in range(1, k + 1)], axis=1).max(axis=1, skipna=False) / c - 1) * 100
    return out


def _night_aligned(d: pd.DataFrame, night: pd.DataFrame | None) -> pd.Series:
    """列 D 對應「D 收盤後、D+1 開盤前」的夜盤：FinMind after_market 的 date = 該夜盤準備的隔一交易日 (與 short_term.build_matrix 相同)。"""
    if night is None or night.empty:
        return pd.Series(np.nan, index=d.index)
    nm = dict(zip(night["date"].astype(str), night["night_chg_pct"].astype(float)))
    return d["date"].astype(str).shift(-1).map(nm)


# ------------------------------------------------------------------ fit
def _coverage_table(d: pd.DataFrame, sig: pd.Series, tg: pd.DataFrame, nv: pd.Series, params: dict) -> dict:
    """逐年 (最近 COVERAGE_YEARS 年) 與最近 250 列的實際觸及率；base 與 night 兩種模式。乘數為全樣本擬合 → 屬 in-sample 監控表，
    用途是看 regime 漂移 (研究：2024/2025 low20 只有 15~16%、2026 high80 28%)，不是 OOS 評估 (OOS 見 research/range)。"""
    years = sorted(d["date"].astype(str).str[:4].unique())[-COVERAGE_YEARS:]
    yr = d["date"].astype(str).str[:4]

    def cov(mask: pd.Series, mode: str) -> dict:
        row = {}
        for k in KS:
            m = params[mode].get(str(k))
            if not m:
                continue
            for name in QUANTS:
                side = "low" if name.startswith("low") else "high"
                tgt = tg[f"pathLow{k}" if side == "low" else f"pathHigh{k}"]
                if mode == "night":
                    lv = m[f"beta_{side}"] * nv.clip(-NIGHT_CLIP, NIGHT_CLIP) + m[name] * sig
                    ok = mask & tgt.notna() & sig.notna() & nv.notna()
                else:
                    lv = m[name] * sig
                    ok = mask & tgt.notna() & sig.notna()
                hit = (tgt[ok] <= lv[ok]) if side == "low" else (tgt[ok] >= lv[ok])
                row[f"k{k}_{name}"] = round(float(hit.mean()), 3) if ok.sum() else None
            row[f"k{k}_n"] = int(((mask & tg[f"pathLow{k}"].notna() & sig.notna()) & (nv.notna() if mode == "night" else True)).sum())
        return row

    out: dict = {"target": {"low10": 0.10, "low20": 0.20, "high80": 0.80, "high90": 0.90, "touch": "low20/high80 目標 0.20；low10/high90 目標 0.10"},
                 "by_year": {}, "last250": {}}
    for y in years:
        out["by_year"][y] = {"base": cov(yr == y, "base")}
        if params.get("night"):
            out["by_year"][y]["night"] = cov(yr == y, "night")
    last = pd.Series(False, index=d.index)
    last.iloc[-250:] = True
    out["last250"] = {"base": cov(last, "base")}
    if params.get("night"):
        out["last250"]["night"] = cov(last, "night")
    return out


def fit_multipliers(scored: pd.DataFrame, night: pd.DataFrame | None = None, path: Path | str | None = DEFAULT_PATH, verbose: bool = False) -> dict:
    """以全部歷史擬合乘數並存 JSON (含最近 5 年逐年觸及率監控表)。scored = backtest.load_long() 輸出 (date/open/high/low/close)；
    night = finmind.tx_night_history() (date = 該夜盤準備的隔一交易日)。重新訓練時機：每次 `cli.py train` 一併執行 (數秒)。"""
    d = scored.sort_values("date").reset_index(drop=True)
    sig = sigma_series(d)
    tg = path_targets(d)
    nv = _night_aligned(d, night)
    out: dict = {"fitted_at": dt.datetime.now(config.TZ).isoformat(timespec="minutes"), "start": str(d["date"].iloc[0]), "end": str(d["date"].iloc[-1]),
                 "n": int(len(d)), "sigma": f"0.5*(ATR{ATR_N}/close% + EWMA({EWMA_LAMBDA}) vol%)", "night_clip": NIGHT_CLIP, "touch_prob": TOUCH_PROB,
                 "base": {}, "night": {}}
    for k in KS:
        m = {}
        for name, q in QUANTS.items():
            tgt = tg[f"pathLow{k}" if name.startswith("low") else f"pathHigh{k}"]
            ratio = (tgt / sig).dropna()
            m[name] = round(float(np.quantile(ratio, q)), 4)
        m["n"] = int((tg[f"pathLow{k}"] / sig).notna().sum())
        out["base"][str(k)] = m
    if nv.notna().any():
        for k in KS:
            m = {}
            for side, tcol in (("low", f"pathLow{k}"), ("high", f"pathHigh{k}")):
                z = pd.DataFrame({"y": tg[tcol], "x": nv, "s": sig}).dropna()
                z = z[z["x"].abs() < NIGHT_CLIP]
                if len(z) < 300:
                    continue
                beta = float(np.cov(z["x"], z["y"])[0, 1] / np.var(z["x"]))
                m[f"beta_{side}"] = round(beta, 4)
                resid = (z["y"] - beta * z["x"]) / z["s"]
                for name, q in QUANTS.items():
                    if name.startswith(side):
                        m[name] = round(float(np.quantile(resid, q)), 4)
                m[f"n_{side}"] = int(len(z))
            if "beta_low" in m and "beta_high" in m:
                out["night"][str(k)] = m
        if out["night"]:
            nd = nv.dropna()
            out["night_start"] = str(d.loc[nd.index[0], "date"])
    try:
        out["coverage"] = _coverage_table(d, sig, tg, nv, out)
    except Exception as e:  # noqa: BLE001
        log.warning("range_levels coverage table: %s", e)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _CACHE.clear()
    if verbose:
        print(format_coverage(out))
    return out


def format_coverage(p: dict) -> str:
    """給 cli.py train 印出的逐年觸及率摘要 (目標 low20/high80 0.20、low10/high90 0.10)。"""
    cv = p.get("coverage") or {}
    lines = [f"range_levels 乘數 {p.get('start')}~{p.get('end')} n={p.get('n')}；夜盤 n={p.get('night', {}).get('1', {}).get('n_low')}；"
             f"base k=1 low20 {p['base']['1']['low20']:+.3f}σ high80 {p['base']['1']['high80']:+.3f}σ"]
    for y, r in (cv.get("by_year") or {}).items():
        b, n = r.get("base", {}), r.get("night", {})
        lines.append(f"  {y}: base low20/high80 k1 {b.get('k1_low20')}/{b.get('k1_high80')} k3 {b.get('k3_low20')}/{b.get('k3_high80')} (n={b.get('k1_n')})"
                     + (f"｜night k1 {n.get('k1_low20')}/{n.get('k1_high80')} (n={n.get('k1_n')})" if n else ""))
    l250 = (cv.get("last250") or {}).get("base", {})
    if l250:
        lines.append(f"  最近 250 日: base low20/high80 k1 {l250.get('k1_low20')}/{l250.get('k1_high80')}, low10/high90 {l250.get('k1_low10')}/{l250.get('k1_high90')} (目標 0.20 / 0.10)")
    return "\n".join(lines)


def load_multipliers(path: Path | str | None = DEFAULT_PATH) -> dict | None:
    key = str(path)
    if key not in _CACHE:
        p = Path(path)
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return _CACHE[key]


# ------------------------------------------------------------------ predict
def range_levels(scored_row_or_frame, k: int, night_ret: float | None = None, base_px: float | None = None,
                 params: dict | None = None, path: Path | str | None = DEFAULT_PATH) -> dict:
    """回傳未來 k 日 (k=1,2,3) 路徑低/高點的 10/20/80/90 分位 (%)，以及換算成指數點位的 buy_at / sell_at / stop / target。

    scored_row_or_frame: 含 open/high/low/close 的 DataFrame (取最後一列，並用其歷史算 sigma；建議 >= 100 列)，
                         或已含 'sigma_range' 的 Series/dict (一列)。
    night_ret: 「完整」夜盤台指期漲跌% (夜盤收盤後~隔日開盤前才有；None = 不用夜盤)。|x| 裁切在 ±8%。
    base_px:   換算點位的基準價，預設用該列收盤。
    """
    p = params or load_multipliers(path)
    if not p:
        raise RuntimeError("range_levels.json 不存在，請先執行 fit_multipliers()")
    if isinstance(scored_row_or_frame, pd.DataFrame):
        fr = scored_row_or_frame
        sigma = float(sigma_series(fr).iloc[-1])
        close = float(fr["close"].iloc[-1])
    else:
        row = scored_row_or_frame
        sigma = float(row["sigma_range"])
        close = float(row["close"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma 無效 ({sigma})，frame 列數不足或含 NaN")
    base_px = float(base_px or close)
    k = int(k)
    m = p["base"][str(k)]
    mode = "base"
    x = None
    if night_ret is not None and np.isfinite(float(night_ret)) and str(k) in p.get("night", {}) and "beta_low" in p["night"][str(k)]:
        nm = p["night"][str(k)]
        x = float(np.clip(float(night_ret), -NIGHT_CLIP, NIGHT_CLIP))     # |夜盤| 裁切 ±8% (擬合時已剔除，極端日覆蓋率會較差)
        lv = {n: (nm[f"beta_{'low' if n.startswith('low') else 'high'}"] * x + nm[n] * sigma) for n in QUANTS}
        mode = "night"
    else:
        lv = {n: m[n] * sigma for n in QUANTS}
    # 夜盤大漲時 low20 可為正 (跳空後回測仍在今收之上)，不強制 <=0，但保證排序
    lv["low10"] = min(lv["low10"], lv["low20"])
    lv["high90"] = max(lv["high90"], lv["high80"])
    px = lambda v: int(round(base_px * (1 + v / 100)))  # noqa: E731
    return {"k": k, "mode": mode, "sigma": round(sigma, 3), "night_ret": None if mode != "night" else round(x, 2),
            "low10": round(lv["low10"], 2), "low20": round(lv["low20"], 2), "high80": round(lv["high80"], 2), "high90": round(lv["high90"], 2),
            "buy_at": px(lv["low20"]), "stop": px(lv["low10"]), "sell_at": px(lv["high80"]), "target": px(lv["high90"]),
            "level_lo": px(lv["low20"]), "level_hi": px(lv["high80"]),
            "note": f"{k} 日路徑約兩成機率跌到 {px(lv['low20']):,} (一成: {px(lv['low10']):,})、約兩成機率漲到 {px(lv['high80']):,} (一成: {px(lv['high90']):,})；"
                    f"sigma {sigma:.2f}%{'，含夜盤 ' + format(x, '+.2f') + '%' if mode == 'night' else ''}"}


# ------------------------------------------------------------------ integration (market_forecast.refine_short_term)
def _night_final(snap: dict, nd: list[dict], scored: pd.DataFrame) -> tuple[float | None, str | None]:
    """判斷可否使用夜盤模式。回傳 (完整夜盤漲跌% 或 None, 不採用的原因)。

    條件 (驗證者要求)：
    1) 夜盤已收盤：phase 為 'closed'/'pre' (05:00 之後) 或 tx_night 帶 final 旗標；phase 'night' 為進行中 → 不用。
    2) 日期對齊：scored 最後一列必須是 next_days[0]['date'] 的前一個交易日，且該夜盤準備的交易日 (快照日起算的第一個交易日)
       必須等於 next_days[0]['date']；若今日日 K 尚未進資料 (scored 停在 D-1 而夜盤是 D→D+1) 則回退 base。
    """
    tn = snap.get("tx_night") or {}
    if tn.get("change_pct") is None:
        return None, None
    phase = snap.get("phase")
    final = bool(tn.get("final") or tn.get("is_final"))
    if phase == "night" and not final:
        return None, "夜盤進行中 (未收盤)，不套夜盤位移"
    if phase not in ("closed", "pre") and not final:
        return None, f"phase={phase} 非夜盤結束後"
    if not nd:
        return None, "next_days 為空"
    try:
        from ..sources import twse
        last = str(scored["date"].iloc[-1])[:10]
        nxt = twse.next_trading_days(last, 1)[0]
        if nxt != str(nd[0]["date"]):
            return None, f"資料末日 {last} 的下一交易日 {nxt} ≠ next_days[0] {nd[0]['date']}"
        ts = str(snap.get("ts") or dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"))[:10]
        expect = twse.next_trading_days((dt.date.fromisoformat(ts) - dt.timedelta(days=1)).isoformat(), 1)[0]
        if expect != str(nd[0]["date"]):
            return None, f"夜盤準備的交易日 {expect} ≠ next_days[0] {nd[0]['date']} (今日日 K 可能尚未進資料)"
    except Exception as e:  # noqa: BLE001
        return None, f"日期對齊檢查失敗 ({e})"
    return float(tn["change_pct"]), None


def attach_to_next_days(nd: list[dict], scored: pd.DataFrame, snapshot: dict | None, base_px: float, live: bool = False, sigma_factor: float = 1.0,
                        path: Path | str | None = DEFAULT_PATH) -> str | None:
    """把路徑型水準寫入 next_days (就地修改)。回傳給 short_term_notes 的一句說明；未擬合乘數時回 None。

    新增鍵：buy_at (=low20 點位)、sell_at (=high80)、stop (=low10)、target (=high90)、range_sigma (%/日)、
            range_mode ('base' | 'night' | 'base_intraday_approx' | 'night_intraday_approx')、touch_prob (0.2)、
            path_low10/path_low20/path_high80/path_high90 (%)、range_note、close_q_lo/close_q_hi (舊收盤分位點位備查)；
            level_lo/level_hi 覆寫為 buy_at/sell_at。
    夜盤模式只在夜盤已結束且日期對齊時使用 (見 _night_final)；夜盤進行中 (phase 'night') 用 base 並標 '_intraday_approx'。
    盤中 (live=True) 以現價當基準價，sigma 仍用前一收盤的歷史 → 近似，標 '_intraday_approx'。"""
    if not nd or not load_multipliers(path):
        return None
    snap = snapshot or {}
    night, why = (None, "盤中以現價推估") if live else _night_final(snap, nd, scored)
    approx = live or (snap.get("phase") == "night" and night is None)     # 夜盤進行中 (無 final 旗標) 屬近似
    fr = scored.tail(400)
    sf = float(sigma_factor) if sigma_factor and np.isfinite(sigma_factor) else 1.0    # 線上自學：近期觸及率校準 (learn.touch_factor)
    for x in nd:
        r = range_levels(fr, x["n"], night_ret=night, base_px=base_px, path=path)
        if abs(sf - 1.0) > 1e-6:   # 依乘數放寬/收窄四個水準 (相對基準價的距離)
            bp = float(base_px)
            for k_ in ("buy_at", "sell_at", "stop", "target", "level_lo", "level_hi"):
                if r.get(k_) is not None:
                    r[k_] = round(bp + (float(r[k_]) - bp) * sf)
            r["note"] = (r.get("note") or "") + f"；自學乘數 ×{sf:.2f}"
        x["close_q_lo"], x["close_q_hi"] = x.get("level_lo"), x.get("level_hi")       # 保留舊值 (收盤報酬分位) 供除錯
        x["level_lo"], x["level_hi"] = r["level_lo"], r["level_hi"]                    # 既有鍵 → 路徑型 20%/80%
        x["buy_at"], x["sell_at"], x["stop"], x["target"] = r["buy_at"], r["sell_at"], r["stop"], r["target"]
        x["path_low10"], x["path_low20"], x["path_high80"], x["path_high90"] = r["low10"], r["low20"], r["high80"], r["high90"]
        x["range_sigma"], x["range_mode"], x["touch_prob"] = r["sigma"], r["mode"] + ("_intraday_approx" if approx else ""), TOUCH_PROB
        x["range_note"] = r["note"] + (f"；{why}" if why and not live else "") + ("；盤中近似值" if live else "")
    return (f"買賣點改用路徑型水準 (sigma {nd[0]['range_sigma']:.2f}%/日"
            + ("，含完整夜盤 " + format(night, "+.2f") + "%" if night is not None else ("，" + why if why else "，不含夜盤")) + ")："
            + "；".join(f"{x['label']} 拉回 {x['buy_at']:,} 買 (停損 {x['stop']:,})、反彈 {x['sell_at']:,} 賣 (目標 {x['target']:,})" for x in nd)
            + "。約兩成機率觸及 (以加權指數計，2014~ 樣本外逐年 15~29%)，為校準機率帶而非方向訊號；台指期觸及會比指數多約 15~20%")
