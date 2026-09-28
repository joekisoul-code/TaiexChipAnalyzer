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
  前端文案：「約兩成機率觸及 (以加權指數計)」；台指期觸及會比指數多 15~20% (基差平均 -0.2%)。
- 2026-09-28 (r2m rl_ivk_sigma)：sigma 改用 TXO ATM ivk_k/√252 (選擇權隱含波動，k 日期限) → 發布分位 pinball 0.3331 → 0.3115 (×0.935，
  7/7 年；2020~ 走動式 n=1633)，夜盤高點 ×0.940、夜盤低點 ×0.942 (事前登錄 X2)。乘數另存 base_iv / night_iv (2017~ 有 ivk 的列)，
  上線 ivk 必須當日或前一交易日 (lag-2 已無優勢)、非 calendar_fallback；缺/過期 → 完整 ATR 路徑 (σ 與乘數不混用)。
  休市後首日 (base_event) 所有 k 維持 ATR (X3：IV 無增益，復市中心移也是在 ATR 帶上驗證)。總開關 iv_monitor.iv_enabled。
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
TOUCH_PROB_REOPEN = 0.15  # 休市後首日 k=1 中心移後的歷史觸及 (low20 .144 / high80 .154，2014~ 104 次)
COVERAGE_YEARS = 5      # JSON 內附最近 N 年逐年觸及率 (監控用)
# TXO IV sigma (2026-09-28 r2m rl_ivk_sigma)
IV_KS = (1, 2, 3)
IV_START = "2017-01-01"
IV_MAX_LAG_TD = 1       # ivk 可落後的交易日數 (lag-1 ×0.958 仍有效；lag-2 ×0.9875 [−0.012, +0.004] 失效)
IV_FORMULA = "ivk_k/sqrt(252)*100 (TXO ATM，T=n/252 交易日)"
IV_ALERT_LAST250 = {"k1_high90": [0.05, 0.15], "k3_low20": [0.13, 0.27], "k1_high80": [0.13, 0.27]}
IV_ROLLBACK_LAST250 = {"k1_high90": 0.17, "k3_low20": 0.30}    # 連續兩次週訓超過 → 自動 iv_enabled=false
IV_ROLLBACK_TEXT = ("若最近 250 日 base_iv k1_high90 > 0.17 或 k3_low20 > 0.30 連續兩次週訓，或 learn 帳本 range_sigma_src=txo_iv 的 1 日 buy/sell 觸及 "
                    "(n≥120) 落在 [0.12,0.28] 之外 → 將 iv_enabled 設 false")
IV_TOUCH_TEXT = "sigma 來源 TXO 選擇權 IV；q20/q80 樣本外觸及約 21~22%、q10/q90 約 11~12% (2020~，pinball 約 −6.5%)"
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


# ------------------------------------------------------------------ TXO IV sigma
def sigma_iv_frame(dates, ivk_hist) -> pd.DataFrame:
    """每個日期的 sigma_iv (%/日)：欄 k=1..3 = ivk_k/√252×100，依日期字串對齊，缺值 / ≤0 → NaN。
    ivk_hist = [{date, ivk:{"1":..}}]，同一日期以後出現者優先 (desk_bands.ivk_frame) → 依「種子在前、歸檔在後」組。"""
    from ..analysis import desk_bands as DB
    ds = pd.Series(list(dates)).astype(str).str[:10]
    idx = dates.index if isinstance(dates, pd.Series) else pd.RangeIndex(len(ds))
    iv = DB.ivk_frame(ivk_hist)
    out = pd.DataFrame(index=idx)
    for k in IV_KS:
        v = iv[k].reindex(ds.values).values.astype(float) if k in iv.columns and len(iv) else np.full(len(ds), np.nan)
        s = v / np.sqrt(252) * 100
        out[k] = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    return out


def ivk_history() -> list[dict]:
    """擬合用 ivk 歷史：data/models/txo_ivk_seed.json (研究同演算法、休市修正) 在前 + 操盤台歸檔 opt_hist (有 ivk 的列) 在後 → 同日歸檔優先。
    歸檔已由 desk.opt_history 剔除 calendar_fallback 列；讀不到歸檔 → 只用種子 (擬合的 iv_end 會停在種子末日)。
    tools/ivk_consistency.py 列出的無效日 (data/models/ivk_invalid.json) 剔除。"""
    from ..analysis import desk_bands as DB
    hist = DB._ivk_seed()
    try:
        from ..analysis import desk as _dk
        hist = hist + [r for r in (_dk.load_archive().get("opt_hist") or []) if r.get("ivk") and not r.get("calendar_fallback")]
    except Exception as e:  # noqa: BLE001
        log.warning("ivk_history archive: %s", e)
    try:
        from . import model as _M
        bad = set((_M.load_json("ivk_invalid") or {}).get("dates") or [])
        if bad:
            hist = [r for r in hist if str(r.get("date"))[:10] not in bad]
    except Exception:  # noqa: BLE001
        pass
    return hist


def iv_enabled(p: dict | None) -> bool:
    """range_levels.json 有 base_iv 且 iv_monitor.iv_enabled 為 true (總開關；回滾時設 false)。"""
    return bool(p and p.get("base_iv") and (p.get("iv_monitor") or {}).get("iv_enabled"))


def _ivk_valid(ivk_live: dict | None, last_date: str, p: dict | None = None) -> dict:
    """上線 ivk 的有效性 (spec_fix 2/3)：非空 dict 且有 date、非 calendar_fallback、落後 ≤ IV_MAX_LAG_TD 個交易日 (twse 交易日曆，
    ivk.date == last → 0；next_trading_days(ivk.date, 1)[0] == last → 1；其餘 (含 ivk 比資料新) 無效)、range_levels.json 有 base_iv 且 iv_enabled。
    各 k 的 ivk 值 (有限且 >0) 由呼叫端逐一檢查。回傳 {ok, why, date, stale_td, ivk}。"""
    out = {"ok": False, "why": None, "date": None, "stale_td": None, "ivk": {}}
    if not iv_enabled(p):
        out["why"] = "IV 未啟用" if (p or {}).get("base_iv") else "range_levels.json 尚無 IV 乘數"
        return out
    if not isinstance(ivk_live, dict) or not ivk_live or not ivk_live.get("date"):
        out["why"] = "ivk 缺"
        return out
    if ivk_live.get("calendar_fallback"):
        out["why"] = "ivk 以週末日曆推算 (休市表缺)"
        return out
    d, last = str(ivk_live["date"])[:10], str(last_date)[:10]
    out["date"] = d
    if d == last:
        st = 0
    else:
        try:
            from ..sources import twse
            st = 1 if (d < last and twse.next_trading_days(d, 1)[0] == last) else None
        except Exception as e:  # noqa: BLE001
            out["why"] = f"交易日曆檢查失敗 ({e})"
            return out
    if st is None or st > IV_MAX_LAG_TD:
        out["why"] = f"ivk {d} 與資料日 {last} 不符 (過期或超前)"
        return out
    out.update(ok=True, stale_td=st, ivk=dict(ivk_live.get("ivk") or {}))
    return out


def _ivk_sigma(ivv: dict, k: int) -> float | None:
    """單一 k 的上線 sigma_iv (%/日)；ivk 值 None / 0 / 負 / 非數 → None。"""
    if not ivv.get("ok"):
        return None
    try:
        v = float((ivv.get("ivk") or {}).get(str(int(k))))
    except (TypeError, ValueError):
        return None
    return v / np.sqrt(252) * 100 if np.isfinite(v) and v > 0 else None


# ------------------------------------------------------------------ fit
def _coverage_table(d: pd.DataFrame, sig: pd.Series, tg: pd.DataFrame, nv: pd.Series, params: dict, siv: pd.DataFrame | None = None) -> dict:
    """逐年 (最近 COVERAGE_YEARS 年) 與最近 250 列的實際觸及率；base 與 night 兩種模式 (有 siv 與 base_iv/night_iv 時另加 IV 兩種)。
    乘數為全樣本擬合 → 屬 in-sample 監控表，用途是看 regime 漂移 (研究：2024/2025 low20 只有 15~16%、2026 high80 28%)，不是 OOS 評估。"""
    years = sorted(d["date"].astype(str).str[:4].unique())[-COVERAGE_YEARS:]
    yr = d["date"].astype(str).str[:4]

    def cov(mask: pd.Series, mode: str) -> dict:
        nmode = mode in ("night", "night_iv")
        row = {}
        for k in KS:
            m = params[mode].get(str(k))
            if not m:
                continue
            s = sig if mode in ("base", "night") else siv[k]
            for name in QUANTS:
                side = "low" if name.startswith("low") else "high"
                tgt = tg[f"pathLow{k}" if side == "low" else f"pathHigh{k}"]
                if nmode:
                    lv = m[f"beta_{side}"] * nv.clip(-NIGHT_CLIP, NIGHT_CLIP) + m[name] * s
                    ok = mask & tgt.notna() & s.notna() & nv.notna()
                else:
                    lv = m[name] * s
                    ok = mask & tgt.notna() & s.notna()
                hit = (tgt[ok] <= lv[ok]) if side == "low" else (tgt[ok] >= lv[ok])
                row[f"k{k}_{name}"] = round(float(hit.mean()), 3) if ok.sum() else None
            row[f"k{k}_n"] = int(((mask & tg[f"pathLow{k}"].notna() & s.notna()) & (nv.notna() if nmode else True)).sum())
        return row

    iv_modes = [m for m in ("base_iv", "night_iv") if siv is not None and params.get(m)]
    out: dict = {"target": {"low10": 0.10, "low20": 0.20, "high80": 0.80, "high90": 0.90, "touch": "low20/high80 目標 0.20；low10/high90 目標 0.10"},
                 "by_year": {}, "last250": {}}
    for y in years:
        out["by_year"][y] = {"base": cov(yr == y, "base")}
        if params.get("night"):
            out["by_year"][y]["night"] = cov(yr == y, "night")
        for m in iv_modes:
            out["by_year"][y][m] = cov(yr == y, m)
    last = pd.Series(False, index=d.index)
    last.iloc[-250:] = True
    out["last250"] = {"base": cov(last, "base")}
    if params.get("night"):
        out["last250"]["night"] = cov(last, "night")
    for m in iv_modes:
        out["last250"][m] = cov(last, m)
    return out


def _fit_iv(d: pd.DataFrame, tg: pd.DataFrame, nv: pd.Series, siv: pd.DataFrame) -> dict:
    """base_iv / night_iv 乘數：只用 date ≥ IV_START、sigma_iv_k > 0、目標非 NaN 的列 (與 base / night 同式，sigma 換成 sigma_iv_k)。"""
    out: dict = {"base_iv": {}, "night_iv": {}}
    for k in IV_KS:
        s = siv[k]
        m = {}
        for name, q in QUANTS.items():
            tgt = tg[f"pathLow{k}" if name.startswith("low") else f"pathHigh{k}"]
            ratio = (tgt / s).dropna()
            if len(ratio) < 250:
                m = {}
                break
            m[name] = round(float(np.quantile(ratio, q)), 4)
        if m:
            m["n"] = int((tg[f"pathLow{k}"] / s).notna().sum())
            out["base_iv"][str(k)] = m
        if not nv.notna().any():
            continue
        m = {}
        for side, tcol in (("low", f"pathLow{k}"), ("high", f"pathHigh{k}")):
            z = pd.DataFrame({"y": tg[tcol], "x": nv, "s": s}).dropna()
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
            out["night_iv"][str(k)] = m
    return out


def _iv_monitor(cov: dict, prev: dict | None) -> dict:
    """iv_monitor：最近 250 日 base_iv 觸及警示 + 回滾規則。前一版 iv_enabled=false (手動或自動) 會保留；
    最近 250 日超過回滾門檻連續兩次擬合 (週訓) → 自動設 false。"""
    l250 = ((cov or {}).get("last250") or {}).get("base_iv") or {}
    pm = (prev or {}).get("iv_monitor") or {}
    alerts = [f"{key} {l250[key]:.3f} ∉ [{lo}, {hi}]" for key, (lo, hi) in IV_ALERT_LAST250.items() if l250.get(key) is not None and not lo <= l250[key] <= hi]
    breach = [key for key, thr in IV_ROLLBACK_LAST250.items() if l250.get(key) is not None and l250[key] > thr]
    streak = int(pm.get("breach_streak") or 0) + 1 if breach else 0
    enabled = bool(pm.get("iv_enabled", True))
    out = {"alert_last250": IV_ALERT_LAST250, "rollback": IV_ROLLBACK_TEXT, "iv_enabled": enabled,
           "last250": {key: l250.get(key) for key in IV_ALERT_LAST250}, "alerts": alerts, "breach": breach, "breach_streak": streak}
    if pm.get("disabled_by") and not enabled:
        out["disabled_by"] = pm["disabled_by"]
    if enabled and streak >= 2:
        out["iv_enabled"] = False
        out["disabled_by"] = f"auto {dt.date.today().isoformat()}：最近 250 日 {'/'.join(breach)} 連續 {streak} 次擬合超過回滾門檻"
    return out


def fit_multipliers(scored: pd.DataFrame, night: pd.DataFrame | None = None, path: Path | str | None = DEFAULT_PATH, verbose: bool = False,
                    ivk_hist: list[dict] | None = None) -> dict:
    """以全部歷史擬合乘數並存 JSON (含最近 5 年逐年觸及率監控表)。scored = backtest.load_long() 輸出 (date/open/high/low/close)；
    night = finmind.tx_night_history() (date = 該夜盤準備的隔一交易日)。重新訓練時機：每次 `cli.py train` 一併執行 (數秒)。
    ivk_hist (ivk_history()：種子在前、歸檔在後) → 另存 base_iv / night_iv / iv_monitor；base / night 與不帶 ivk_hist 時逐位元相同
    (pullback.train 讀的是 ATR 的 base low20/low10)。"""
    prev = None
    if path and ivk_hist:
        try:
            prev = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None
        except Exception:  # noqa: BLE001
            prev = None
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
    siv = None
    if ivk_hist:
        try:
            siv = sigma_iv_frame(d["date"], ivk_hist)
            siv.loc[d["date"].astype(str).str[:10] < IV_START, :] = np.nan
            fi = _fit_iv(d, tg, nv, siv)
            if fi["base_iv"]:
                out["base_iv"] = fi["base_iv"]
                if fi["night_iv"]:
                    out["night_iv"] = fi["night_iv"]
                okd = d.loc[siv[1].notna(), "date"].astype(str)
                out.update(sigma_iv=IV_FORMULA, iv_start=str(okd.iloc[0]), iv_end=str(okd.iloc[-1]), n_iv=int(siv[1].notna().sum()))
            else:
                siv = None
        except Exception as e:  # noqa: BLE001
            log.warning("range_levels IV multipliers: %s", e)
            siv = None
    try:
        out["coverage"] = _coverage_table(d, sig, tg, nv, out, siv)
    except Exception as e:  # noqa: BLE001
        log.warning("range_levels coverage table: %s", e)
    if out.get("base_iv"):
        out["iv_monitor"] = _iv_monitor(out.get("coverage") or {}, prev)
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
    if p.get("base_iv"):
        li = (cv.get("last250") or {}).get("base_iv", {})
        mon = p.get("iv_monitor") or {}
        stale = f" ⚠ ivk 歷史只到 {p.get('iv_end')} (資料到 {p.get('end')})" if str(p.get("iv_end") or "") < str(p.get("end") or "") else ""
        lines.append(f"  IV sigma ({p.get('sigma_iv')}): n_iv={p.get('n_iv')} {p.get('iv_start')}~{p.get('iv_end')}{stale}；base_iv k1 low20 {p['base_iv'].get('1', {}).get('low20')}σ "
                     f"high80 {p['base_iv'].get('1', {}).get('high80')}σ；night_iv {'有' if p.get('night_iv') else '無'}")
        lines.append(f"  最近 250 日 base_iv: k1 low20/high80 {li.get('k1_low20')}/{li.get('k1_high80')}, k1 high90 {li.get('k1_high90')}, k3 low20 {li.get('k3_low20')}；"
                     f"iv_enabled={mon.get('iv_enabled')}" + (f"；警示 {mon.get('alerts')}" if mon.get("alerts") else "") + (f"；{mon.get('disabled_by')}" if mon.get("disabled_by") else ""))
    elif p.get("n_iv") is None:
        lines.append("  IV sigma：未提供 ivk 歷史 (只擬合 ATR+EWMA)")
    return "\n".join(lines)


def load_multipliers(path: Path | str | None = DEFAULT_PATH) -> dict | None:
    key = str(path)
    if key not in _CACHE:
        p = Path(path)
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return _CACHE[key]


# ------------------------------------------------------------------ predict
def range_levels(scored_row_or_frame, k: int, night_ret: float | None = None, base_px: float | None = None,
                 params: dict | None = None, path: Path | str | None = DEFAULT_PATH, sigma_iv: float | None = None) -> dict:
    """回傳未來 k 日 (k=1,2,3) 路徑低/高點的 10/20/80/90 分位 (%)，以及換算成指數點位的 buy_at / sell_at / stop / target。

    scored_row_or_frame: 含 open/high/low/close 的 DataFrame (取最後一列，並用其歷史算 sigma；建議 >= 100 列)，
                         或已含 'sigma_range' 的 Series/dict (一列)。
    night_ret: 「完整」夜盤台指期漲跌% (夜盤收盤後~隔日開盤前才有；None = 不用夜盤)。|x| 裁切在 ±8%。
    base_px:   換算點位的基準價，預設用該列收盤。
    sigma_iv:  TXO IV sigma (%/日 = ivk_k/√252×100)。同時滿足 iv_enabled、sigma_iv 有限且 >0、有 base_iv[k] (夜盤模式另需 night_iv[k]
               含 beta_low/beta_high) 才走 IV 路徑 (IV σ × IV 乘數)；否則一律完整 ATR 路徑 (ATR σ × base/night 乘數)，兩套不混用。
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
    sigma_atr = sigma
    night_ok = night_ret is not None and np.isfinite(float(night_ret)) and str(k) in p.get("night", {}) and "beta_low" in p["night"][str(k)]
    use_iv = bool(iv_enabled(p) and sigma_iv is not None and np.isfinite(float(sigma_iv)) and float(sigma_iv) > 0 and str(k) in (p.get("base_iv") or {}))
    if use_iv and night_ok:
        niv = (p.get("night_iv") or {}).get(str(k)) or {}
        use_iv = "beta_low" in niv and "beta_high" in niv
    if use_iv:
        sigma = float(sigma_iv)
    m = (p["base_iv"] if use_iv else p["base"])[str(k)]
    mode = "base"
    x = None
    if night_ok:
        nm = (p["night_iv"] if use_iv else p["night"])[str(k)]
        x = float(np.clip(float(night_ret), -NIGHT_CLIP, NIGHT_CLIP))     # |夜盤| 裁切 ±8% (擬合時已剔除，極端日覆蓋率會較差)
        lv = {n: (nm[f"beta_{'low' if n.startswith('low') else 'high'}"] * x + nm[n] * sigma) for n in QUANTS}
        mode = "night"
    else:
        lv = {n: m[n] * sigma for n in QUANTS}
    # 夜盤大漲時 low20 可為正 (跳空後回測仍在今收之上)，不強制 <=0，但保證排序
    lv["low10"] = min(lv["low10"], lv["low20"])
    lv["high90"] = max(lv["high90"], lv["high80"])
    px = lambda v: int(round(base_px * (1 + v / 100)))  # noqa: E731
    src_txt = "；sigma 來源 TXO 選擇權 IV" if use_iv else ("；sigma 來源 ATR+EWMA" if iv_enabled(p) else "")     # IV 未設定時文字與舊版相同
    return {"k": k, "mode": mode, "sigma": round(sigma, 3), "night_ret": None if mode != "night" else round(x, 2),
            "low10": round(lv["low10"], 2), "low20": round(lv["low20"], 2), "high80": round(lv["high80"], 2), "high90": round(lv["high90"], 2),
            "buy_at": px(lv["low20"]), "stop": px(lv["low10"]), "sell_at": px(lv["high80"]), "target": px(lv["high90"]),
            "level_lo": px(lv["low20"]), "level_hi": px(lv["high80"]),
            "sigma_src": "txo_iv" if use_iv else "atr_ewma", "sigma_atr": round(sigma_atr, 3),
            "note": f"{k} 日路徑約兩成機率跌到 {px(lv['low20']):,} (一成: {px(lv['low10']):,})、約兩成機率漲到 {px(lv['high80']):,} (一成: {px(lv['high90']):,})；"
                    f"sigma {sigma:.2f}%{'，含夜盤 ' + format(x, '+.2f') + '%' if mode == 'night' else ''}{src_txt}"}


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


def _reopen_shift(scored: pd.DataFrame, prev: str, td: str, us_px) -> dict:
    """休市後首日 1 日帶中心移 (r2m reopen_us_move (ii) CAND2)：c = β_cc × 休市期間全部美股交易日費半累積 log 報酬 (%)，
    β_cc = TAIEX 收盤對收盤 vs 前一美股日 SOX (前 750 個正常日、無截距；SOX 不齊改 TSM ADR 與其 β)；寬度 ×width_ratio (event_params)。
    us_px = {"sox", "tsm", "px"(長 TAIEX date/open/close，可省略 → 用 scored)} 或回傳該 dict 的函式；取不到 / 美股未全收 / β 不足 → 不移 (BASE)。"""
    from . import events as EV
    P = EV.load_params()
    rule = (P.get("band_rules") or {}).get("post_closure_centre_shift") or {}
    out: dict = {"status": "disabled", "sessions": [], "src": None}
    try:
        wr = float(rule.get("width_ratio"))
    except (TypeError, ValueError):
        wr = float("nan")
    if not rule.get("enabled") or not np.isfinite(wr) or wr <= 0:
        return out
    px = us_px() if callable(us_px) else (us_px or {})
    px = px or {}
    mv = EV.us_move(prev, td, px.get("sox"), px.get("tsm"), now=px.get("now"))
    out.update(status=mv["status"], sessions=mv["sessions"], src=mv["src"])
    if mv["status"] != "ready":
        return out
    base = px.get("px") if px.get("px") is not None else scored
    b, n, _ = EV.gap_beta(base[["date", "open", "close"]], mv["xret"], td, target="cc")
    if b is None or not np.isfinite(b):
        out.update(status="no_beta", beta_n=n)
        return out
    c = b * mv["sum_logret"] * 100
    out.update(status="ready", beta=round(float(b), 4), beta_n=int(n), c=round(float(c), 4), wr=wr, sum_pct=round(float(mv["sum_logret"]) * 100, 2))
    return out


def attach_to_next_days(nd: list[dict], scored: pd.DataFrame, snapshot: dict | None, base_px: float, live: bool = False, sigma_factor: float = 1.0,
                        path: Path | str | None = DEFAULT_PATH, ivk_live: dict | None = None, us_px=None) -> str | None:
    """把路徑型水準寫入 next_days (就地修改)。回傳給 short_term_notes 的一句說明；未擬合乘數時回 None。

    新增鍵：buy_at (=low20 點位)、sell_at (=high80)、stop (=low10)、target (=high90)、range_sigma (%/日，實際使用的 sigma)、
            range_mode ('base' | 'night' | 'base_event' | '..._intraday_approx' | '..._night_pending')、touch_prob (0.2；休市後首日中心移的 k=1 為 0.15)、
            path_low10/path_low20/path_high80/path_high90 (%)、range_note、close_q_lo/close_q_hi (舊收盤分位點位備查)；
            range_sigma_src ('txo_iv' | 'atr_ewma')、range_sigma_atr、ivk_date、ivk_stale_td (IV 生效時)、range_sigma_fallback (IV 已設定但未用的原因)；
            休市後首日 k=1 另有 event_shift_pct / event_shift_src / event_shift_beta / event_width_ratio / event_shift_sessions / event_shift_status。
            level_lo/level_hi 覆寫為 buy_at/sell_at。
    夜盤模式只在夜盤已結束且日期對齊時使用 (見 _night_final)；夜盤進行中 (phase 'night') 用 base 並標 '_intraday_approx'。
    盤中 (live=True) 以現價當基準價，sigma 仍用前一收盤的歷史 → 近似，標 '_intraday_approx'。
    ivk_live = taifex_opt.features() (失敗為 {})：有效 (_ivk_valid) 時各 k 用 IV sigma；休市後首日 (base_event) 所有 k 維持 ATR (X3)。
    us_px：休市後首日中心移所需的美股 / 長 TAIEX (見 _reopen_shift)；None → 不移 (與舊版相同)。"""
    p = load_multipliers(path)
    if not nd or not p:
        return None
    snap = snapshot or {}
    night, why = (None, "盤中以現價推估") if live else _night_final(snap, nd, scored)
    approx = live or (snap.get("phase") == "night" and night is None)     # 夜盤進行中 (無 final 旗標) 屬近似
    fr = scored.tail(400)
    sf = float(sigma_factor) if sigma_factor and np.isfinite(sigma_factor) else 1.0    # 線上自學：近期觸及率校準 (learn.touch_factor)
    # 事件 (2026-09-27, chip/predict/events)：休市後首日 k=1 寬度 × √n_US。研究驗證的帶不含美股資訊 → n_US ≥ 2 時退回 base
    # ('base_event')，不與夜盤 β 混用 (夜盤最多只涵蓋前一交易日晚上那 1 個美股日)。只在驗收 A 通過 (event_params) 時套用。
    ev = None
    try:
        from . import events as EV
        prev = dt.datetime.now(config.TZ).date().isoformat() if live else str(scored["date"].iloc[-1])[:10]
        cal = EV.refresh_calendar(write=False)
        ev = EV.session_factor(str(nd[0]["date"]), {"used": night is not None}, prev=prev, cal=cal)
        ev["enabled"] = EV.range_levels_enabled()
        ev["tags_by_date"] = {str(x["date"]): EV.session_tags(str(x["date"]), cal) for x in nd}
    except Exception as e:  # noqa: BLE001
        log.warning("events session_factor: %s", e)
    ev_on = bool(ev and ev.get("enabled") and ev["range_k1"] > 1 and int(nd[0].get("n") or 0) == 1)
    night_dropped = bool(ev_on and night is not None)
    why_kn = why
    if night_dropped:          # √n_US 只放寬 k=1；k≥2 只是一起改 base 模式 (不含夜盤)，不能寫成「× √n_US」
        why = f"休市期間 {ev['n_us']} 個美股交易日 (夜盤只涵蓋其中 1 個) → 改用 base 模式 × √n_US"
        why_kn = "休市後首日不含夜盤模式 (改用 base 模式)；k≥2 不放寬"
        night = None
    # 休市後首日：事件帶是在 sf=1 下驗證的 (touch_factor 由正常日估出) → k1~3 都不套自學乘數
    sf_note = ev_on and abs(sf - 1.0) > 1e-6
    if ev_on:
        sf = 1.0
    # TXO IV sigma：ivk 有效 (當日或前一交易日、非 calendar_fallback、iv_enabled) 時各 k 用 IV；休市後首日所有 k 維持 ATR (X3)
    ivv = _ivk_valid(ivk_live, str(scored["date"].iloc[-1])[:10], p)
    iv_cfg = iv_enabled(p)
    # 休市後首日 1 日帶中心移 (reopen_us_move (ii))：非盤中、有注入美股資料時才算
    cs = None
    if ev_on and not live and us_px is not None:
        try:
            cs = _reopen_shift(scored, str(ev.get("prev_td") or scored["date"].iloc[-1])[:10], str(nd[0]["date"]), us_px)
        except Exception as e:  # noqa: BLE001
            log.warning("reopen centre shift: %s", e)
            cs = {"status": "error", "sessions": [], "src": None}
    shift_on = bool(cs and cs.get("status") == "ready")
    for x in nd:
        siv = None if ev_on else _ivk_sigma(ivv, int(x["n"]))
        r = range_levels(fr, x["n"], night_ret=night, base_px=base_px, path=path, sigma_iv=siv)
        ef = float(ev["range_k1"]) if (ev_on and int(x["n"]) == 1) else 1.0
        tp = TOUCH_PROB
        if ef != 1.0:              # 四個分位對稱放寬 (相對基準價的距離 × factor)；有中心移時 = 放寬 × width_ratio + c
            for q in QUANTS:
                r[q] = round(r[q] * ef * cs["wr"] + cs["c"], 2) if shift_on else round(r[q] * ef, 2)
            px_ = lambda v: int(round(float(base_px) * (1 + v / 100)))  # noqa: E731
            r.update(buy_at=px_(r["low20"]), stop=px_(r["low10"]), sell_at=px_(r["high80"]), target=px_(r["high90"]),
                     level_lo=px_(r["low20"]), level_hi=px_(r["high80"]), mode="base_event")
            r["note"] = (f"{r['k']} 日路徑約兩成機率跌到 {r['buy_at']:,} (一成: {r['stop']:,})、約兩成機率漲到 {r['sell_at']:,} (一成: {r['target']:,})；"
                         f"sigma {r['sigma']:.2f}%；{ev['why']}")
            if shift_on:
                tp = TOUCH_PROB_REOPEN
                r["note"] += (f"；休市 {len(cs['sessions'])} 個美股交易日{'費半' if cs['src'] == 'SOX' else '台積電 ADR'}累積 {cs['sum_pct']:+.2f}% × β_cc {cs['beta']:.2f}"
                              f" → 1 日帶中心移 {cs['c']:+.2f}%，寬度 ×√n_US×{cs['wr']:g}；休市後首日歷史觸及約 15%")
                x.update(event_shift_pct=round(cs["c"], 2), event_shift_src=cs["src"], event_shift_beta=cs["beta"], event_width_ratio=cs["wr"])
            elif cs and cs.get("status") == "pending_us":
                r["note"] += "；休市期間美股尚未全部收盤，1 日帶中心移待美股收盤後更新"
            if cs is not None:
                x.update(event_shift_status=cs.get("status"), event_shift_sessions=list(cs.get("sessions") or []))
        elif int(x["n"]) == 1 and ev and not ev_on and float(ev.get("range_k1") or 1) > 1 and r["sigma_src"] == "txo_iv":
            # spec_fix 7：events 的 range_levels_path 被停用時，n_US≥2 的 k1 落到「IV 不乘 √n_US」(相對 PROD×√ 約 1.10，仍優於 ATR 不乘 √ 的約 1.16)
            r["note"] += f"；休市後首日 (事件放寬停用)：IV 已含休市期間 {ev.get('n_us')} 個美股交易日的預期波動，未另乘 √n_US"
        x["event_tags"] = list(((ev or {}).get("tags_by_date") or {}).get(str(x["date"])) or [])
        x["event_range_factor"] = round(ef, 4)
        if abs(sf - 1.0) > 1e-6:   # 依乘數放寬/收窄四個水準 (相對基準價的距離)
            bp = float(base_px)
            for k_ in ("buy_at", "sell_at", "stop", "target", "level_lo", "level_hi"):
                if r.get(k_) is not None:
                    r[k_] = round(bp + (float(r[k_]) - bp) * sf)
            r["note"] = (r.get("note") or "") + f"；自學乘數 ×{sf:.2f}"
        elif sf_note:
            r["note"] = (r.get("note") or "") + "；休市後首日不套自學乘數"
        x["close_q_lo"], x["close_q_hi"] = x.get("level_lo"), x.get("level_hi")       # 保留舊值 (收盤報酬分位) 供除錯
        x["level_lo"], x["level_hi"] = r["level_lo"], r["level_hi"]                    # 既有鍵 → 路徑型 20%/80%
        x["buy_at"], x["sell_at"], x["stop"], x["target"] = r["buy_at"], r["sell_at"], r["stop"], r["target"]
        x["path_low10"], x["path_low20"], x["path_high80"], x["path_high90"] = r["low10"], r["low20"], r["high80"], r["high90"]
        suffix = "_intraday_approx" if live else ("_night_pending" if approx else "")   # 2026-09-24：夜盤進行中另標 (舊版與盤中同標「盤中近似」)
        x["range_sigma"], x["range_mode"], x["touch_prob"] = r["sigma"], r["mode"] + suffix, tp
        x["range_sigma_src"], x["range_sigma_atr"] = r["sigma_src"], r["sigma_atr"]
        used_iv = r["sigma_src"] == "txo_iv"
        x["ivk_date"], x["ivk_stale_td"] = (ivv["date"], ivv["stale_td"]) if used_iv else (None, None)
        if iv_cfg and not used_iv:
            x["range_sigma_fallback"] = ("休市後首日維持 ATR+EWMA" if ev_on else (ivv.get("why") or f"ivk k={x['n']} 缺值"))
        w_ = why if int(x["n"]) == 1 else why_kn
        x["range_note"] = r["note"] + (f"；{w_}" if w_ and not live else "") + ("；盤中近似值" if live else "")
    if ev_on:   # 路徑單調：k 日路徑極值不可能比 1 日窄 → k=2/3 至少與放寬後的 k=1 一樣寬 (k≥2 本身不放寬)
        k1 = next((x for x in nd if int(x["n"]) == 1), None)
        for x in nd:
            if k1 is None or int(x["n"]) == 1:
                continue
            bp = float(base_px)
            chg = False
            for q, side in (("path_low10", min), ("path_low20", min), ("path_high80", max), ("path_high90", max)):
                v = side(x[q], k1[q])
                if v != x[q]:
                    x[q], chg = v, True
            if chg:
                pxs = lambda v: int(round(bp * (1 + v * sf / 100)))  # noqa: E731   (含自學乘數，與上面一致；休市後首日 sf=1)
                x["stop"], x["buy_at"] = pxs(x["path_low10"]), pxs(x["path_low20"])
                x["sell_at"], x["target"] = pxs(x["path_high80"]), pxs(x["path_high90"])
                x["level_lo"], x["level_hi"] = x["buy_at"], x["sell_at"]
                x["range_note"] = (x.get("range_note") or "") + "；已依路徑單調性不窄於放寬後的 1 日帶"
    any_iv = any(x.get("range_sigma_src") == "txo_iv" for x in nd)
    if any_iv:
        touch_txt = f"。{IV_TOUCH_TEXT}，為校準機率帶而非方向訊號；台指期觸及會比指數多約 15~20%"
    else:
        fb = ("休市後首日維持 ATR+EWMA" if ev_on else "IV 缺/過期 → ATR+EWMA") if iv_cfg else None
        touch_txt = "。約兩成機率觸及 (以加權指數計" + (f"；{fb}" if fb else "") + ")，為校準機率帶而非方向訊號；台指期觸及會比指數多約 15~20%"
    ev_txt = ""
    if ev_on:
        ev_txt = (f"。隔天為休市後首日：{ev['why']}，" + ("1 日帶依休市期間美股平移並調寬 (2014~ 104 次：pinball −31%)" if shift_on else
                                                         "1 日帶已放寬 (驗收回測 2014+ 116 次休市：pinball −11%)"
                                                         + ("；休市期間美股尚未全部收盤，中心移待美股收盤後更新" if (cs and cs.get("status") == "pending_us") else "")))
    return (f"買賣點改用路徑型水準 (sigma {nd[0]['range_sigma']:.2f}%/日" + ("，選擇權 IV" if nd[0].get("range_sigma_src") == "txo_iv" else "")
            + ("，含完整夜盤 " + format(night, "+.2f") + "%" if night is not None else
               (("，休市後首日不含夜盤 (夜盤只涵蓋 1 個美股交易日，改用 base 模式)；"
                 + (f"1 日帶 ×√n_US×{cs['wr']:g} 並依休市期間美股平移 {cs['c']:+.2f}%，k≥2 不放寬" if shift_on else "只有 1 日帶 × √n_US，k≥2 不放寬"))
                if night_dropped else
                ("，" + why if why else "，不含夜盤"))) + ")："
            + "；".join(f"{x['label']} 拉回 {x['buy_at']:,} 買 (停損 {x['stop']:,})、反彈 {x['sell_at']:,} 賣 (目標 {x['target']:,})" for x in nd)
            + touch_txt + ev_txt)
