"""操盤台：選擇權機率帶「每日滾動重算乘數 + 觸及率監控」(研究 opt_bands，2026-09-26；獨立驗證逐位重現)。

與研究 t3 s01/s02 的「滾動 750 日、只用已實現目標」一致：
  t 日乘數 = quantile(ratio[s], τ)，s ∈ [t−k+1−750, t−k]；ratio = 未來 k 日路徑低(高)點 % ÷ sigma[s]。
  指數：sigma = ivk_k/√252×100，改用加權指數自身高低點 (相對加權收盤) 校準 (以台指期校準量 TWII 觸及率偏低 1~2pp)；
  標的：sigma_asset = sigma_k·√(β² + (σ_idio/sigma_k)²)，β/σ_idio 逐日滾動 250 日。
改進研究 (視窗 500/1000/擴張、ACI、狀態條件、sigma 混合) 沒有一個顯著勝過基準 → 維持滾動 750。
監控：最近 250 個已實現日的觸及率；k 依賴門檻 (完全校準下的抽樣區間) 當警示，規格門檻只當參考 (k≥10 規格門檻誤警 20~40%)。
"""
from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RANGE_KS = (1, 2, 3, 5, 10, 20)
RANGE_QS = {"low10": 0.10, "low20": 0.20, "high80": 0.80, "high90": 0.90}
RANGE_NOMINAL = {"low10": 0.10, "low20": 0.20, "high80": 0.20, "high90": 0.10}          # 名目觸及率
RANGE_ALERT = {"low10": (0.05, 0.16), "low20": (0.12, 0.28), "high80": (0.12, 0.28), "high90": (0.05, 0.16)}   # 規格門檻
# k 依賴門檻：完全校準下 (i.i.d. 模擬，路徑重疊) 滾動 250 日觸及率的 2.5%~97.5% 區間；k≤3 沿用規格 (誤警率 ≤ 4%)。
# 規格門檻在完全校準下的誤警率：k=5 7~11%、k=10 20~24%、k=20 36~42% (幾乎都是抽樣雜訊)。
RANGE_ALERT_K = {5: {"low10": (0.04, 0.175), "low20": (0.115, 0.30), "high80": (0.115, 0.30), "high90": (0.04, 0.175)},
                 10: {"low10": (0.02, 0.205), "low20": (0.085, 0.34), "high80": (0.085, 0.34), "high90": (0.02, 0.205)},
                 20: {"low10": (0.0, 0.25), "low20": (0.045, 0.40), "high80": (0.045, 0.40), "high90": (0.0, 0.25)}}
BAND_WIN, BAND_MIN_N = 750, 250
BAND_DAYS = 2200                             # FinMind 日曆天：標的 (滾動 750 + β 250 + 監控 250 ≈ 1,300 交易日，留餘裕)
TWII_START = "2016-12-01"                   # 加權固定起點 (保守版 m_exp = 2017-01-03 起擴張，每日重算)
INDEX_IDS = ("TWII", "TX")


def ivk_frame(ivk_hist) -> pd.DataFrame:
    """[{date, ivk:{"1":v,..}}] (ivk 種子 ∪ 歸檔 opt_hist；同日後出現者優先) → index=date 字串、欄=k (int)、值=年化 ivk。"""
    rows = {}
    for r in ivk_hist or []:
        d, iv = str((r or {}).get("date") or "")[:10], (r or {}).get("ivk") or {}
        if d and iv:
            rows[d] = [iv.get(str(k)) if iv.get(str(k)) is not None else np.nan for k in RANGE_KS]
    if not rows:
        return pd.DataFrame(columns=list(RANGE_KS), dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(RANGE_KS), dtype=float).sort_index()


def rolling_beta_idio(adj: pd.Series, mkt: pd.Series, w: int = 250) -> pd.DataFrame:
    """逐日 β 與殘差日波動 % (研究 s02 rolling_beta)：各自日曆的對數報酬 → 取交集 → 滾動 w 日 (至少 0.8w 筆) OLS。
    adj：標的含息收盤、mkt：加權指數收盤，皆以日期字串為索引。回傳 index=日期、欄 beta / idio。"""
    x = pd.concat([np.log(adj.astype(float)).diff().rename("a"), np.log(mkt.astype(float)).diff().rename("m")], axis=1).sort_index().dropna()
    mp = int(w * 0.8)
    cov = x["a"].rolling(w, min_periods=mp).cov(x["m"])
    var = x["m"].rolling(w, min_periods=mp).var()
    beta = cov / var
    idio = np.sqrt((x["a"].rolling(w, min_periods=mp).var() - beta ** 2 * var).clip(lower=0)) * 100
    return pd.DataFrame({"beta": beta, "idio": idio})


def _path_ext(c: np.ndarray, h: np.ndarray, l: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """未來 1..k 日路徑最低/最高 相對當日收盤 (%)；路徑不完整或含缺值 → NaN (研究 path_targets)。"""
    n = len(c)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    if n > k:
        sw = np.lib.stride_tricks.sliding_window_view
        lo[:n - k] = sw(l[1:], k).min(axis=1)
        hi[:n - k] = sw(h[1:], k).max(axis=1)
    return (lo / c - 1) * 100, (hi / c - 1) * 100


def band_frame(close_df: pd.DataFrame, hi_lo_df: pd.DataFrame, ivk_hist, sid: str, beta_idio_fn=None) -> pd.DataFrame:
    """每個 (價格日 ∩ ivk 日) 一列：sig{k} (日 %)、pl{k}/ph{k} (未來 k 日路徑低/高 %)、iv{k}。
    close_df：date, close[, adj]；hi_lo_df：date, high, low (原始價；有 adj 時以 adj/close 換成含息價)。
    sid ∈ INDEX_IDS → 指數 sigma；其他 → sigma_asset，β/σ_idio 由 beta_idio_fn(含息收盤 Series，index=日期) 回傳 (index=日期、欄 beta/idio)。"""
    c = close_df.copy()
    c["date"] = c["date"].astype(str).str[:10]
    h = hi_lo_df[["date", "high", "low"]].copy()
    h["date"] = h["date"].astype(str).str[:10]
    p = c.merge(h, on="date", how="inner").drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    close = p["close"].astype(float)
    adj = p["adj"].astype(float) if "adj" in p else close
    f = adj / close
    a_c, a_h, a_l = adj.values, (p["high"].astype(float) * f).values, (p["low"].astype(float) * f).values
    X = pd.DataFrame(index=pd.Index(p["date"].values, name="date"))
    for k in RANGE_KS:
        X[f"pl{k}"], X[f"ph{k}"] = _path_ext(a_c, a_h, a_l, k)
    iv = ivk_frame(ivk_hist)
    X = X.join(iv.rename(columns={k: f"iv{k}" for k in RANGE_KS}), how="inner")
    is_index = sid in INDEX_IDS
    if not is_index:
        if beta_idio_fn is None:
            raise ValueError("asset needs beta_idio_fn")
        bi = beta_idio_fn(pd.Series(a_c, index=p["date"].values))
        X = X.join(bi[["beta", "idio"]], how="left")
    for k in RANGE_KS:
        sk = X[f"iv{k}"] / math.sqrt(252) * 100
        X[f"sig{k}"] = sk if is_index else sk * np.sqrt(X["beta"] ** 2 + (X["idio"] / sk) ** 2)
    return X


def _wf_mult(ratio: np.ndarray, tau: float, k: int, pos, window: int | None, min_n: int, require_full: bool = False) -> np.ndarray:
    """走動式乘數 (研究 wf_daily)：位置 t 只用 ratio[max(0, t−k+1−window) : t−k+1] 的有限值 (目標在 t 收盤前已實現)。
    require_full：窗口 (window 列) 沒被有效 ratio 填滿 (容許 ≤10 個缺值) → NaN；避免價格歷史被截短 (或 β 暖機) 時悄悄算出不同定義的乘數。"""
    out = np.full(len(pos), np.nan)
    for j, t in enumerate(pos):
        hi = int(t) - k + 1
        if hi <= 0 or (require_full and window is not None and hi < window):
            continue
        x = ratio[(0 if window is None else max(0, hi - window)):hi]
        x = x[np.isfinite(x)]
        if require_full and window is not None and len(x) < window - 10:
            continue
        if len(x) >= min_n:
            out[j] = np.quantile(x, tau)
    return out


def calib_multipliers(close_df, hi_lo_df, ivk_hist, sid, beta_idio_fn=None, window=BAND_WIN, min_n=BAND_MIN_N, frame=None, require_full=False) -> dict:
    """最新一日 (聯集最後一列) 的滾動乘數 {k: {q: m}} (字串 k，4 位小數；樣本不足 → None)。window=None → 擴張 (保守版 m_exp)。
    frame：可傳入 band_frame 的結果以免重算。"""
    X = frame if frame is not None else band_frame(close_df, hi_lo_df, ivk_hist, sid, beta_idio_fn)
    out = {}
    t = len(X) - 1
    for k in RANGE_KS:
        out[str(k)] = {}
        for q, tau in RANGE_QS.items():
            y = X[f"pl{k}" if q.startswith("low") else f"ph{k}"].values
            m = _wf_mult(y / X[f"sig{k}"].values, tau, k, [t], window, min_n, require_full)[0] if t >= 0 else np.nan
            out[str(k)][q] = round(float(m), 4) if np.isfinite(m) else None
    return out


def coverage(close_df, hi_lo_df, ivk_hist, sid, beta_idio_fn=None, window=BAND_WIN, min_n=BAND_MIN_N, last=250, frame=None, require_full=False) -> dict:
    """最近 last 個「已實現」日 (s+k ≤ 今日) 的實際觸及率：水準 = s 日走動式乘數 × s 日 sigma (與上線一致，無前視)。
    alert = 規格門檻 (low20/high80 ∉ [0.12,0.28]、low10/high90 ∉ [0.05,0.16])；alert_k = k 依賴門檻 (k≥5 放寬，見 RANGE_ALERT_K)。"""
    X = frame if frame is not None else band_frame(close_df, hi_lo_df, ivk_hist, sid, beta_idio_fn)
    n = len(X)
    res, alerts, alerts_k = {}, [], []
    for k in RANGE_KS:
        res[str(k)] = {}
        pos = np.arange(max(0, n - k - last), max(0, n - k))       # 目標已實現的最後 last 列
        for q, tau in RANGE_QS.items():
            y = X[f"pl{k}" if q.startswith("low") else f"ph{k}"].values
            sg = X[f"sig{k}"].values
            lv = _wf_mult(y / sg, tau, k, pos, window, min_n, require_full) * sg[pos]
            yy = y[pos]
            ok = np.isfinite(lv) & np.isfinite(yy)
            hit = (yy[ok] <= lv[ok]) if q.startswith("low") else (yy[ok] >= lv[ok])
            rate = float(hit.mean()) if ok.sum() else None
            enough = rate is not None and ok.sum() >= 0.8 * last
            lo_, hi_ = RANGE_ALERT[q]
            klo, khi = RANGE_ALERT_K[k][q] if k in RANGE_ALERT_K else RANGE_ALERT[q]
            flag, flag_k = bool(enough and not (lo_ <= rate <= hi_)), bool(enough and not (klo <= rate <= khi))
            res[str(k)][q] = {"rate": round(rate, 3) if rate is not None else None, "n": int(ok.sum()), "nominal": RANGE_NOMINAL[q],
                              "alert": flag, "alert_k": flag_k}
            msg = f"k={k} {q} 近 {int(ok.sum())} 日觸及率 {rate:.0%} (名目 {RANGE_NOMINAL[q]:.0%}" if enough else ""
            if flag:
                alerts.append(msg + f"，規格警戒 {lo_:.0%}~{hi_:.0%})")
            if flag_k:
                alerts_k.append(msg + f"，k 依賴警戒 {klo:.1%}~{khi:.1%})")
    return {"last": last, "asof": str(X.index[-1]) if n else None, "from_k20": str(X.index[max(0, n - RANGE_KS[-1] - last)]) if n else None,
            "levels": res, "alerts": alerts, "alerts_k": alerts_k}


# ------------------------------------------------------------------ 接到 build()：每日重算 → 覆蓋 ev["range"] 的乘數 (失敗的格子沿用凍結值)
def _ivk_seed() -> list[dict]:
    """data/models/txo_ivk_seed.json (= 研究 opt_bands/ivk_seed.json；2017-01-03~2026-09-24，休市修正後)。"""
    try:
        from ..predict import model as _M
        return (_M.load_json("txo_ivk_seed") or {}).get("rows") or []
    except Exception:  # noqa: BLE001
        return []


def range_live(ev: dict, tw: pd.DataFrame, frames: dict, ohist: list[dict]) -> tuple[dict, dict]:
    """回傳 (range 證據副本：今日滾動乘數 (成功才整組覆蓋；指數改加權自身校準、原台指期值留 m_tx)，監控)。
    tw：date, high, low, close (FinMind TAIEX，自 TWII_START 起)；frames：{sid: {px, adj, px_band?, adj_band?}}；ohist：歸檔 opt_hist (含 ivk)。"""
    rg = json.loads(json.dumps(ev.get("range") or {}))
    mon: dict = {"asof": None, "coverage": {}, "live": {}, "m_exp_live": False, "window": BAND_WIN, "min_n": BAND_MIN_N,
                 "thresholds_k": {str(k): v for k, v in RANGE_ALERT_K.items()}, "thresholds_spec": RANGE_ALERT}
    for k_, row in (rg.get("index") or {}).items():          # 先把凍結的台指期乘數留在 m_tx (00981A 未校準映射沿用)
        for q, cell in row.items():
            cell.setdefault("m_tx", cell.get("m"))
    hist = _ivk_seed() + [r for r in (ohist or []) if r.get("ivk")]
    if not hist or tw is None or len(tw) < BAND_WIN + 30:
        return rg, mon
    t_ = tw.copy()
    t_["date"] = t_["date"].astype(str).str[:10]
    try:
        X = band_frame(t_[["date", "close"]], t_[["date", "high", "low"]], hist, "TWII")
        m = calib_multipliers(None, None, None, "TWII", frame=X, require_full=True)
        ok = all(v is not None for row in m.values() for v in row.values())
        if ok:                                                # 全有或全無：不混用台指期與加權兩種校準
            for k, row in m.items():
                for q, v in row.items():
                    rg["index"][k][q]["m"] = v
        mon["live"]["TWII"] = ok
        if ok and X.index[0] <= "2017-01-03":                 # 保守版：2017-01-03 起擴張 (固定起點)
            me = calib_multipliers(None, None, None, "TWII", frame=X, window=None)
            if all(v is not None for row in me.values() for v in row.values()):
                for k, row in me.items():
                    for q, v in row.items():
                        rg["index"][k][q]["m_exp"] = v
                mon["m_exp_live"] = True
        mon["asof"] = str(X.index[-1])
        mon["coverage"]["TWII"] = coverage(None, None, None, "TWII", frame=X, require_full=True)
    except Exception as e:  # noqa: BLE001
        log.warning("range_live TWII: %s", e)
    mkt = pd.Series(t_["close"].astype(float).values, index=t_["date"].values)
    for sid, f in (frames or {}).items():
        if sid == "00981A":                     # 資料短 (研究：≥500 個已實現日再校準)；維持未校準映射 (用 m_tx)
            continue
        try:
            px = f.get("px_band") if f.get("px_band") is not None else f["px"]
            adj = f.get("adj_band") if f.get("adj_band") is not None else f["adj"]
            C = pd.DataFrame({"date": px["date"].astype(str).str[:10].values, "close": px["close"].astype(float).values, "adj": np.asarray(adj, float)})
            H = pd.DataFrame({"date": C["date"].values, "high": px["high"].astype(float).values, "low": px["low"].astype(float).values})
            X = band_frame(C, H, hist, sid, lambda s: rolling_beta_idio(s, mkt, 250))
            m = calib_multipliers(None, None, None, sid, frame=X, require_full=True)
            ok = all(v is not None for row in m.values() for v in row.values())
            a = rg.setdefault("assets", {}).setdefault(sid, {})
            if ok:
                a["m"], a["method"] = m, "calibrated (每日滾動 750)"
                bi = X[["beta", "idio"]].replace([np.inf, -np.inf], np.nan).dropna()   # 加權晚一天時最後一列為 NaN → 取最後有限值
                if len(bi):
                    a["beta"], a["idio"] = round(float(bi["beta"].iloc[-1]), 4), round(float(bi["idio"].iloc[-1]), 4)
                else:
                    a.pop("beta", None)
                    a.pop("idio", None)
                mon["coverage"][sid] = coverage(None, None, None, sid, frame=X, require_full=True)
            mon["live"][sid] = ok                   # False → 沿用 desk_evidence 凍結乘數 (價格歷史不足或 ivk 缺)
        except Exception as e:  # noqa: BLE001
            log.warning("range_live %s: %s", sid, e)
    return rg, mon
