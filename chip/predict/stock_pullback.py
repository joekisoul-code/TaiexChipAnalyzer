"""個股回落模型 + 回落判斷 (2026-09-23)：跨股票共用 (pooled) 的「路徑最低點分位」與「回落止跌機率」模型，並輸出前端可離線套用的查表。

- 樣本：realtime.LARGE_CAPS (40 檔) + chips.WATCHLIST + 加權指數，2018~ 日 OHLCV (FinMind)；特徵只用價量 (前端對任何股票都能算) + 大盤當日/乖離。
- 目標：yLow_k = 未來 k 日路徑最低 / sigma (k=1,2,3)；hold3 = 未來 3 日最低不破今日低點 (−0.3%) = 「今日低點就是回落低點」。
- 模型：LightGBM quantile (0.20 買點 / 0.10 停損) 與 LGB 回歸 0/1 (止跌機率)，2021~ 逐年走動式 (pooled)；
  基準 = 訓練年份合併分位 (常數乘數) 與止跌基準率。
- 查表 (前端用)：dd_hi20 × 連漲跌 × K 棒位置 (clv) × 量能 4 維 36 格 → q20/q10 乘數、止跌率、n；也走動式驗證 (前幾年建表、當年套用)。
- 支撐止跌統計 (pooled)：5 日線/月線/季線/昨低/20 日低 的回測後止跌率。

r2m 2026-09-28 (final_spec §3.2、§6、§7)：
- B2 價格清洗 _clean_price：丟 OHLC ≤0 的列 + chips.adjust_splits (FinMind 為未還原價；00685L 1:25、00631L 1:22、2327 1:4、6669 1:3)。
  訓練 (build_panel) 與上線 (build) 都清洗；未清洗時 00685L σ 高估 39%、bias60 −57.7 (應 +11.4)。原始 frame 保留給 ivmap (需未調整的 change 欄)。
- 3a 除息扣除 (build)：k 日窗口含已公告除息日 → 買點/停損扣現金股利 (看盤價口徑，含息口徑另存 *_tr)；除息窗口 pinball20 −62% (259 次)。
- 3b 夜盤變體：特徵 + night_chg_pct + beta250×night (列 D 對應 D 收盤後的夜盤 = range_levels._night_aligned；絕不用 date==D 合併)；
  另存 client 公式 F (low = b·β250·night + m·σ)。k1 pinball 約 −8% (未清洗面板)；gate 照舊 (改善 ≥3% 且觸及 0.15~0.27)。
- 3c IV 映射 σ (弱但安全)：ivr = ln(√(β²σ_iv,k² + idio²)/σ)，β/idio 由原始 frame 的除權息修正報酬滾動 250 日；ivr 可算且 use_iv 才用 IV 模型，否則 FEATS 模型。
"""
from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
KS = (1, 2, 3)
Q_BUY, Q_STOP = 0.20, 0.10
START = "2018-01-01"
FIRST_TEST_YEAR = 2021
HOLD_TOL = 0.003
FEATS = ["sigma", "bias5", "bias20", "bias60", "dd_hi20", "lo20_dist", "days_since_hi20", "clv", "lower_wick", "upper_wick", "range_pct", "ret1", "ret5", "streak",
         "vol_ratio", "vola_ratio", "m_ret1", "m_bias20", "dow"]
NAMES = {"sigma": "波動 σ", "bias5": "5 日線乖離%", "bias20": "月線乖離%", "bias60": "季線乖離%", "dd_hi20": "距 20 日高點%", "lo20_dist": "距 20 日低點%", "days_since_hi20": "高點後天數",
         "clv": "收盤在當日振幅位置", "lower_wick": "下影線%", "upper_wick": "上影線%", "range_pct": "振幅%", "ret1": "今日漲跌%", "ret5": "5 日漲跌%", "streak": "連漲(跌)天數",
         "vol_ratio": "量能/20 日均", "vola_ratio": "波動/60 日均", "m_ret1": "大盤今日%", "m_bias20": "大盤月線乖離%", "dow": "星期"}
LGB_Q = dict(M.PARAMS, n_estimators=150, min_child_samples=200)
SEEDS = (1, 2, 3)
BUCKETS = {"dd_hi20": [-6.0, -2.0], "streak": [-2.5, -0.5], "clv": [0.35], "vol_ratio": [0.8]}   # 邊界 → 格 0..len
SUPPORTS = {"ma5": "5 日線", "ma20": "月線", "ma60": "季線", "prev_low": "昨日低點", "lo20": "20 日低點"}
NEAR, HOLD_S = 0.003, 0.005
OHLCV = ["date", "open", "high", "low", "close", "volume"]
# 3b 夜盤變體
NIGHT_FEATS = FEATS + ["night_chg_pct", "bnight"]
NIGHT_CLIP = 8.0
BETA_N, BETA_MIN = 250, 60
NIGHT_GUARD_CORR = 0.5          # 對齊守門：TAIEX 列 corr(night, 隔日開盤跳空) 必須 > 0.5 (錯位 date==D 約 −0.07)
# 3c IV 映射
IV_FEATS = FEATS + ["ivr"]
IV_W, IV_MINP = 250, 200
IV_FFILL = 5
IV_ADJ_FLAG_MAX = 0.25          # 除權息/分割旗標列占比上限：超過 = 餵進已分割調整的 frame (change 未縮放) → 拒算
# 3a 除息
EXDIV_MAX_YIELD = 0.15
EXDIV_OOS = {"events": 259, "windows": 1554, "pinball20_rel": -0.62, "ci95": [-0.66, -0.58], "touch20": [0.687, 0.340], "years": "2021-2026",
             "note": "2330/ETF ≈0 或略差 (2330 +6% n.s.)；效益集中在高殖利率年配股；扣除後除息窗口觸及仍約 34% (k1 約 38%，目標 20%)"}
POST_CLOSURE_NOTE = "休市後首日：夜盤只涵蓋一晚，歷史觸及約 30%，非 20%"


def _clean_price(p: pd.DataFrame, sid: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """B2：丟 OHLC ≤0 的列 (FinMind 停牌/無成交列) + 還原分割 (chips.adjust_splits，KNOWN_SPLITS 優先)。回傳 (清洗後, 原始, 分割事件)；原始 frame 給 ivmap。"""
    from ..analysis import chips
    raw = p.copy()
    p = p[~(p[["open", "high", "low", "close"]] <= 0).any(axis=1)].reset_index(drop=True)
    p, splits = chips.adjust_splits(p, sid)
    return p, raw, splits


def features_from_ohlc(df: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: date/open/high/low/close/volume (單一標的、日期升冪)。前端 JS 有同款實作 (prediction.js pullbackFeatures)。"""
    from . import range_levels as RL
    d = df.copy().reset_index(drop=True)
    for c_ in ("open", "high", "low", "close", "volume"):
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    d["sigma"] = RL.sigma_series(d).values
    d["ma5"], d["ma20"], d["ma60"] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    d["bias5"], d["bias20"], d["bias60"] = (c / d["ma5"] - 1) * 100, (c / d["ma20"] - 1) * 100, (c / d["ma60"] - 1) * 100
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    d["hi20"], d["lo20"] = hi20, lo20
    d["dd_hi20"], d["lo20_dist"] = (c / hi20 - 1) * 100, (c / lo20 - 1) * 100
    # 高點後天數：20 日窗內最高 high 距今幾天
    d["days_since_hi20"] = h.rolling(20).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    rng = (h - l).replace(0, np.nan)
    d["clv"] = ((c - l) / rng).clip(0, 1).fillna(0.5)
    d["lower_wick"] = (np.minimum(o, c) - l) / c * 100
    d["upper_wick"] = (h - np.maximum(o, c)) / c * 100
    d["range_pct"] = (h - l) / c * 100
    d["ret1"] = c.pct_change() * 100; d["ret5"] = c.pct_change(5) * 100
    sgn = np.sign(d["ret1"].fillna(0))
    st = np.zeros(len(d))
    for i in range(1, len(d)):
        st[i] = st[i - 1] + sgn[i] if sgn[i] != 0 and (st[i - 1] == 0 or np.sign(st[i - 1]) == sgn[i]) else sgn[i]
    d["streak"] = st
    d["vol_ratio"] = v / v.rolling(20).mean()
    d["vola_ratio"] = d["sigma"] / d["sigma"].rolling(60).mean()
    d["prev_low"] = l
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
    if mkt is not None:
        mm = mkt[["date", "m_ret1", "m_bias20"]]
        d = d.merge(mm, on="date", how="left")
    else:
        d["m_ret1"] = np.nan; d["m_bias20"] = np.nan
    # beta250 (3b)：個股 ret1 對大盤 m_ret1 的滾動 250 日 OLS 斜率 (至少 60 列；任一缺值的列遮掉；±11% 截尾)，缺 → 1
    r_, m_ = d["ret1"].clip(-11, 11), d["m_ret1"].clip(-11, 11)
    ok_ = r_.notna() & m_.notna()
    r_, m_ = r_.where(ok_), m_.where(ok_)
    d["beta250"] = (r_.rolling(BETA_N, min_periods=BETA_MIN).cov(m_) / m_.rolling(BETA_N, min_periods=BETA_MIN).var()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for k in KS:
        pl = pd.concat([l.shift(-j) for j in range(1, k + 1)], axis=1).min(axis=1, skipna=False)
        d[f"pathLow{k}"] = (pl / c - 1) * 100
        d[f"yLow{k}"] = d[f"pathLow{k}"] / d["sigma"]
    lo3 = pd.concat([l.shift(-j) for j in (1, 2, 3)], axis=1).min(axis=1, skipna=False)
    d["hold3"] = (lo3 >= l * (1 - HOLD_TOL)).astype(float).where(lo3.notna())
    d["c3"] = c.shift(-3)
    return d


def _market_env() -> pd.DataFrame:
    from ..analysis import backtest
    m = backtest.load_long("2010-01-01")[["date", "open", "high", "low", "close", "volume"]].copy()
    m["date"] = m["date"].astype(str)
    c = pd.to_numeric(m["close"], errors="coerce")
    return pd.DataFrame({"date": m["date"], "m_ret1": c.pct_change() * 100, "m_bias20": (c / c.rolling(20).mean() - 1) * 100}), m


def build_panel(universe: list[str] | None = None, raw_out: dict | None = None, clean_out: dict | None = None) -> pd.DataFrame:
    """raw_out (選用) 收集各檔原始 FinMind frame (ivmap 用)；clean_out 收集清洗紀錄 {sid: {zero_rows, splits}}。"""
    from ..sources import finmind
    from ..analysis import chips
    from ..realtime import LARGE_CAPS
    env, midx = _market_env()
    uni = universe or sorted(set(LARGE_CAPS) | set(chips.WATCHLIST))
    frames = []
    for sid in uni:
        try:
            p = finmind.stock_price(sid, START)
            if p.empty or len(p) < 120:
                continue
            p, raw, splits = _clean_price(p, sid)           # B2：零價列 + 分割
            if raw_out is not None:
                raw_out[sid] = raw
            if clean_out is not None:
                clean_out[sid] = {"zero_rows": int(len(raw) - len(p)), "splits": splits}
            f = features_from_ohlc(p[OHLCV], env); f["stock_id"] = sid
            frames.append(f)
        except Exception as e:  # noqa: BLE001
            log.warning("stock_pullback %s: %s", sid, e)
    mi = midx[midx["date"] >= START]
    f = features_from_ohlc(mi, env); f["stock_id"] = "TAIEX"; f["beta250"] = 1.0; frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    for k in KS:   # 除權息/異常
        panel.loc[panel[f"pathLow{k}"] < -30, [f"pathLow{k}", f"yLow{k}"]] = np.nan
    panel["year"] = panel["date"].str[:4].astype(int)
    return panel.sort_values(["date", "stock_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ 3b 夜盤對齊
def _night_by_date(night: pd.DataFrame | None, mkt_dates) -> dict:
    """{交易日 D: D 收盤後、下一交易日開盤前的夜盤漲跌%}；與 range_levels._night_aligned 同一函式 (FinMind after_market date = 夜盤準備的隔一交易日)。"""
    from . import range_levels as RL
    tw = pd.DataFrame({"date": sorted(set(pd.Series(list(mkt_dates)).astype(str).str[:10]))})
    tw["night"] = RL._night_aligned(tw, night).values
    return dict(zip(tw["date"], tw["night"].astype(float)))


def _attach_night(panel: pd.DataFrame, night: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """列 D → night[date == next_td(D)] (日曆 = _market_env 的 TWII 日期)；bnight = beta250 × night。絕不可用 date==D 合併 (那是 B1 錯位)。"""
    if night is None:
        from . import short_term
        night = short_term._night_hist()
    if mkt is None:
        mkt = _market_env()[1]
    nm = _night_by_date(night, mkt["date"])
    p = panel.copy()
    p["night_chg_pct"] = p["date"].astype(str).map(nm).astype(float)
    p["bnight"] = p["beta250"] * p["night_chg_pct"]
    return p


def _night_guard(panel: pd.DataFrame) -> float:
    """回歸守門：TAIEX 列 corr(night_chg_pct, 隔日開盤跳空) 必須 > NIGHT_GUARD_CORR，否則 raise (避免錯位合併默默訓練)。"""
    t = panel[panel["stock_id"] == "TAIEX"].sort_values("date")
    gap = (pd.to_numeric(t["open"], errors="coerce").shift(-1) / pd.to_numeric(t["close"], errors="coerce") - 1) * 100
    ok = t["night_chg_pct"].notna() & gap.notna() & (t["night_chg_pct"].abs() <= NIGHT_CLIP)
    if ok.sum() < 100:
        raise ValueError(f"夜盤對齊守門：可比對列不足 ({int(ok.sum())})")
    c = float(np.corrcoef(t.loc[ok, "night_chg_pct"], gap[ok])[0, 1])
    if not (c > NIGHT_GUARD_CORR):
        raise ValueError(f"夜盤對齊守門失敗：corr(night, 隔日跳空) = {c:.3f} ≤ {NIGHT_GUARD_CORR} (疑似 date==D 錯位)")
    return c


def _formula_fit(bn, pl, sg) -> tuple[float, float, float]:
    """公式 F：b = cov(bnight, pathLow)/var(bnight)；m20/m10 = (pathLow − b·bnight)/σ 的 0.2/0.1 分位 (研究 s2_wf 同式)。"""
    x, y, s = np.asarray(bn, float), np.asarray(pl, float), np.asarray(sg, float)
    b = float(np.cov(x, y)[0, 1] / np.var(x))
    r = (y - b * x) / s
    return b, float(np.quantile(r, Q_BUY)), float(np.quantile(r, Q_STOP))


# ------------------------------------------------------------------ 3c IV 映射
def ivmap(price_raw: pd.DataFrame, twii_close) -> pd.DataFrame:
    """個股 β / 殘差日波動 idio (%)：必須吃 FinMind 原始 frame (未分割調整、含 change 欄；spec_fix 2)。
    報酬：丟 OHLC ≤0 列；close − change ≠ 前收 的日子 (除權息/分割/參考價調整) 用 ln(close/(close−change)) (change≠0)，否則 NaN；
    與 desk_bands.rolling_beta_idio 同款 OLS (250 日、至少 200 筆)，向前補最多 5 列。twii_close：DataFrame(date, close) 或以日期為索引的 Series。
    餵進已分割調整的 frame (change 未縮放) 時旗標列暴增 → raise ValueError。"""
    if price_raw is None or "change" not in price_raw:
        raise ValueError("ivmap 需要 FinMind 原始 frame (含 change 欄)")
    p = price_raw.copy()
    p["date"] = p["date"].astype(str).str[:10]
    p = p[~(p[["open", "high", "low", "close"]] <= 0).any(axis=1)].sort_values("date").reset_index(drop=True)
    c, ch = pd.to_numeric(p["close"], errors="coerce"), pd.to_numeric(p["change"], errors="coerce")
    prev = c.shift(1)
    ref = (c - ch).where(lambda s: s > 0)
    flag = ((ref - prev).abs() > np.maximum(0.01, 1e-4 * prev)).fillna(False)
    if len(p) >= 60 and float(flag.mean()) > IV_ADJ_FLAG_MAX:
        raise ValueError(f"ivmap：除權息旗標列 {float(flag.mean()):.0%} 過多，疑似已分割調整的價格 (change 未縮放)，請用原始 frame")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c / prev)
        r = r.where(~flag, np.where(ch != 0, np.log(c / ref), np.nan))
    ra = pd.Series(r.replace([np.inf, -np.inf], np.nan).values, index=p["date"].values)
    if isinstance(twii_close, pd.DataFrame):
        tw = pd.Series(pd.to_numeric(twii_close["close"], errors="coerce").values, index=twii_close["date"].astype(str).str[:10].values)
    else:
        tw = pd.Series(np.asarray(twii_close, float), index=pd.Index(twii_close.index).astype(str).str[:10])
    tw = tw[~tw.index.duplicated(keep="last")].sort_index()
    rm = np.log(tw.where(tw > 0)).diff()
    x = pd.concat([ra[~ra.index.duplicated(keep="last")].rename("a"), rm.rename("m")], axis=1).sort_index().dropna()
    cov = x["a"].rolling(IV_W, min_periods=IV_MINP).cov(x["m"])
    var = x["m"].rolling(IV_W, min_periods=IV_MINP).var()
    beta = cov / var
    idio = np.sqrt((x["a"].rolling(IV_W, min_periods=IV_MINP).var() - beta ** 2 * var).clip(lower=0)) * 100
    bi = pd.DataFrame({"beta": beta, "idio": idio}).reindex(pd.Index(p["date"].values).drop_duplicates()).ffill(limit=IV_FFILL).reindex(p["date"].values)
    return pd.DataFrame({"date": p["date"].values, "beta": bi["beta"].values, "idio": bi["idio"].values})


_IVK_HIST: list | None = None


def _ivk_hist() -> list[dict]:
    """ivk 歷史 = txo_ivk_seed (種子) + 操盤台歸檔 opt_hist (有 ivk、非 calendar_fallback)，剔除 ivk_invalid；同一執行內快取。"""
    global _IVK_HIST
    if _IVK_HIST is None:
        try:
            from . import range_levels as RL
            _IVK_HIST = RL.ivk_history()
        except Exception as e:  # noqa: BLE001
            log.warning("stock_pullback ivk history: %s", e)
            _IVK_HIST = []
    return _IVK_HIST


def _sig_iv(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f / math.sqrt(252) * 100 if math.isfinite(f) and f > 0 else None


def _iv_columns(panel: pd.DataFrame, raw: dict, mkt: pd.DataFrame, hist: list[dict]) -> pd.DataFrame:
    """訓練面板加上 beta/idio (各檔原始 frame 的 ivmap；TAIEX β=1、idio=0) 與 sigiv{k} (同日 ivk)、sigivL{k} (前一交易日 ivk，晚 1 日對照)。"""
    from ..analysis import desk_bands as DB
    parts = []
    for sid, g in panel.groupby("stock_id", sort=False):
        b = pd.DataFrame({"date": g["date"].astype(str).values, "stock_id": sid})
        if sid == "TAIEX":
            b["beta"], b["idio"] = 1.0, 0.0
        else:
            try:
                m = ivmap(raw[sid], mkt).drop_duplicates("date", keep="last")
                b = b.merge(m, on="date", how="left")
            except Exception as e:  # noqa: BLE001
                log.warning("ivmap %s: %s", sid, e)
                b["beta"], b["idio"] = np.nan, np.nan
        parts.append(b)
    p = panel.merge(pd.concat(parts, ignore_index=True), on=["date", "stock_id"], how="left", validate="one_to_one")
    iv = DB.ivk_frame(hist)
    cal = sorted(set(mkt["date"].astype(str).str[:10]))
    prev = dict(zip(cal[1:], cal[:-1]))
    for k in KS:
        s = (iv[k] / math.sqrt(252) * 100) if k in iv.columns and len(iv) else pd.Series(dtype=float)
        s = s.where(s > 0)
        m_ = dict(zip(s.index.astype(str), s.values))
        p[f"sigiv{k}"] = p["date"].map(m_).astype(float)
        p[f"sigivL{k}"] = p["date"].map(prev).map(m_).astype(float)
    return p


def _ivr(beta, idio, sig_iv, sigma):
    """ivr = ln(√(β²σ_iv² + idio²)/σ)；任一無法計算 → NaN。"""
    sm = np.sqrt(np.asarray(beta, float) ** 2 * np.asarray(sig_iv, float) ** 2 + np.asarray(idio, float) ** 2)
    s = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((sm > 0) & (s > 0), np.log(sm / s), np.nan)
    return sm, r


def _pick_ivk(d0: str, prev_td: str | None, ivk_live: dict | None = None, hist: list | None = None) -> dict | None:
    """上線 ivk：歷史 (種子+歸檔) + ivk_live (同日以 live 為準)；日期須 ≤ 價格日 且 ≥ 前一交易日 (晚 1 日已驗證仍有效)。回傳 {date, stale_td, ivk} 或 None。"""
    rows = list(hist if hist is not None else _ivk_hist())
    if isinstance(ivk_live, dict) and ivk_live.get("date") and ivk_live.get("ivk") and not ivk_live.get("calendar_fallback"):
        rows.append({"date": str(ivk_live["date"])[:10], "ivk": ivk_live["ivk"]})
    best = None
    for r in rows:
        dd = str((r or {}).get("date") or "")[:10]
        if not dd or dd > d0 or (dd != d0 and (not prev_td or dd < prev_td)):
            continue
        if best is None or dd >= best["date"]:
            best = {"date": dd, "ivk": dict((r or {}).get("ivk") or {})}
    if best:
        best["stale_td"] = 0 if best["date"] == d0 else 1
    return best


# ------------------------------------------------------------------ 模型
def _qfit(X, y, alpha):
    import lightgbm as lgb
    return [lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=s, **LGB_Q).fit(X, y) for s in SEEDS]


def _rfit(X, y):
    import lightgbm as lgb
    return [lgb.LGBMRegressor(random_state=s, **LGB_Q).fit(X, y) for s in SEEDS]


def _pred(ms, X):
    return np.mean([m_.predict(X) for m_ in ms], axis=0)


def _pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def cell_of(row) -> str:
    idx = []
    for f, edges in BUCKETS.items():
        v = row[f] if not isinstance(row, pd.DataFrame) else row[f].iloc[0]
        v = float(v) if v == v else (edges[0] + 0.0)   # NaN → 中間格
        idx.append(str(int(np.searchsorted(edges, v, side="right"))))
    return "|".join(idx)


def _table(d: pd.DataFrame) -> dict:
    cells = d.apply(cell_of, axis=1) if len(d) else pd.Series(dtype=str)
    out = {"buckets": BUCKETS, "cells": {}, "all": {}}
    def stats(g):
        r = {"n": int(len(g))}
        for k in KS:
            yy = g[f"yLow{k}"].dropna()
            r[f"q20_{k}"] = round(float(yy.quantile(Q_BUY)), 3) if len(yy) >= 30 else None
            r[f"q10_{k}"] = round(float(yy.quantile(Q_STOP)), 3) if len(yy) >= 30 else None
        hh = g["hold3"].dropna(); r["hold3"] = round(float(hh.mean()), 3) if len(hh) >= 30 else None
        return r
    out["all"] = stats(d)
    for c_, g in d.groupby(cells):
        out["cells"][c_] = stats(g)
    return out


def _apply_table(tbl: dict, d: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    cells = d.apply(cell_of, axis=1)
    q20 = np.array([((tbl["cells"].get(c_) or {}).get(f"q20_{k}") or tbl["all"][f"q20_{k}"]) for c_ in cells], float)
    hold = np.array([((tbl["cells"].get(c_) or {}).get("hold3") or tbl["all"]["hold3"]) for c_ in cells], float)
    return q20, hold


def _support_stats(panel: pd.DataFrame) -> dict:
    out = {}
    c = panel["close"]; lo3 = panel.groupby("stock_id")["low"].transform(lambda s: pd.concat([s.shift(-j) for j in (1, 2, 3)], axis=1).min(axis=1, skipna=False))
    c3 = panel["c3"]; r3 = (c3 / c - 1) * 100
    for key, name in SUPPORTS.items():
        s = panel[key]
        cand = (s < c) & (s > c * 0.97) & lo3.notna() & s.notna()
        tested = cand & (lo3 <= s * (1 + NEAR)); held = tested & (lo3 >= s * (1 - HOLD_S)) & (c3 > s)
        n_t = int(tested.sum())
        out[key] = {"name": name, "n_tested": n_t, "hold_rate": round(float(held[tested].mean()), 3) if n_t else None,
                    "bounce3_after_hold": round(float(r3[held].mean()), 2) if held.any() else None, "break3_after_fail": round(float(r3[tested & ~held].mean()), 2) if (tested & ~held).any() else None}
        for st_, msk in (("bull", c >= panel["ma20"]), ("bear", c < panel["ma20"])):
            t2 = tested & msk
            out[key][f"hold_{st_}"] = round(float(held[t2].mean()), 3) if t2.sum() >= 50 else None
    return out


def _lvl_stats(yv, sg, q20, q10, years) -> dict:
    t20, t10 = yv <= q20, yv <= q10
    byy = pd.DataFrame({"t": t20, "y": years}).groupby("y")["t"].mean()
    return {"pinball20": round(_pinball(yv * sg, q20 * sg, Q_BUY), 4), "pinball10": round(_pinball(yv * sg, q10 * sg, Q_STOP), 4),
            "touch20": round(float(t20.mean()), 3), "touch10": round(float(t10.mean()), 3),
            "touch_yr_min": round(float(byy.min()), 3), "touch_yr_max": round(float(byy.max()), 3), "mae_low": round(float(np.mean(np.abs(yv - q20) * sg)), 3)}


def _train_night(d: pd.DataFrame, k: int, pm: pd.Series, ps: pd.Series, cal: list[str], trained_at: str, write: bool, keep: dict | None) -> tuple[dict, dict]:
    """3b：夜盤變體走動式 (2021~，測試列 = 夜盤已知且 |night|≤8；訓練保留夜盤缺值列、剔除 |night|>8) + 公式 F。回傳 (k 結果, 公式參數)。"""
    y, yr, sg, nv = d[f"yLow{k}"], d["year"], d["sigma"], d["night_chg_pct"]
    okN = nv.isna() | (nv.abs() <= NIGHT_CLIP)
    test = (yr >= FIRST_TEST_YEAR) & nv.notna() & (nv.abs() <= NIGHT_CLIP)
    XN = d[NIGHT_FEATS]
    n20, n10, f20, f10 = (pd.Series(np.nan, index=d.index) for _ in range(4))
    for yv in range(FIRST_TEST_YEAR, int(yr.max()) + 1):
        tr, te = yr < yv, test & (yr == yv)
        if tr.sum() < 3000 or not te.any():
            continue
        trN = tr & okN
        a20 = _pred(_qfit(XN[trN], y[trN], Q_BUY), XN[te])
        n20[te] = a20; n10[te] = np.minimum(_pred(_qfit(XN[trN], y[trN], Q_STOP), XN[te]), a20)
        pd_ = [c_ for c_ in cal if c_ < f"{yv}-01-01"]
        cut = pd_[-k] if len(pd_) >= k else None           # 目標跨入測試年的最後 k 個交易日不進公式訓練
        trF = tr & nv.notna() & (nv.abs() < NIGHT_CLIP) & d[f"pathLow{k}"].notna() & ((d["date"] < cut) if cut else True)
        b, m20, m10 = _formula_fit(d.loc[trF, "bnight"], d.loc[trF, f"pathLow{k}"], sg[trF])
        x_, s_ = d.loc[te, "bnight"].values, sg[te].values
        f20[te] = (b * x_ + m20 * s_) / s_; f10[te] = np.minimum((b * x_ + m10 * s_) / s_, f20[te].values)
    ev = n20.notna() & pm.notna() & ps.notna()
    yv_, sg_, yrs = y[ev].values, sg[ev].values, yr[ev].values
    res = {"n_oos": int(ev.sum()), "n_stocks": int(d.loc[ev, "stock_id"].nunique()),
           "night": _lvl_stats(yv_, sg_, n20[ev].values, n10[ev].values, yrs),
           "base": _lvl_stats(yv_, sg_, pm[ev].values, ps[ev].values, yrs),
           "formula": _lvl_stats(yv_, sg_, f20[ev].values, f10[ev].values, yrs)}
    pb = res["base"]["pinball20"]
    res["improve_pinball"] = round(1 - res["night"]["pinball20"] / pb, 4) if pb else None
    res["improve_pinball10"] = round(1 - res["night"]["pinball10"] / res["base"]["pinball10"], 4) if res["base"]["pinball10"] else None
    res["formula"]["improve_vs_base"] = round(1 - res["formula"]["pinball20"] / pb, 4) if pb else None
    e = d.loc[ev, ["stock_id", "year"]].assign(y=yv_ * sg_, n=n20[ev].values * sg_, b=pm[ev].values * sg_, tn=yv_ <= n20[ev].values, tb=yv_ <= pm[ev].values)
    pin_y = [(_pinball(g["y"].values, g["n"].values, Q_BUY), _pinball(g["y"].values, g["b"].values, Q_BUY)) for _, g in e.groupby("year")]
    res["yrs_better"] = f"{sum(1 for a, b_ in pin_y if a < b_)}/{len(pin_y)}"
    res["by_stock"] = {}
    for sid, g in e.groupby("stock_id"):
        if len(g) < 60:
            continue
        pbs = _pinball(g["y"].values, g["b"].values, Q_BUY)
        res["by_stock"][sid] = {"n": int(len(g)), "touch20": round(float(g["tn"].mean()), 3), "touch20_base": round(float(g["tb"].mean()), 3),
                                "improve": round(1 - _pinball(g["y"].values, g["n"].values, Q_BUY) / pbs, 3) if pbs else None}
    imp = [v["improve"] for v in res["by_stock"].values() if v["improve"] is not None]
    res["by_stock_summary"] = {"n_stocks": len(imp), "improve_median": round(float(np.median(imp)), 3) if imp else None, "n_negative": int(sum(1 for v in imp if v < 0))}
    res["use_model"] = bool(res["improve_pinball"] is not None and res["improve_pinball"] >= 0.03 and 0.15 <= res["night"]["touch20"] <= 0.27)
    okF = nv.notna() & (nv.abs() < NIGHT_CLIP) & d[f"pathLow{k}"].notna()
    b, m20, m10 = _formula_fit(d.loc[okF, "bnight"], d.loc[okF, f"pathLow{k}"], sg[okF])
    form = {"b": round(b, 4), "m20": round(m20, 4), "m10": round(m10, 4), "n": int(okF.sum()),
            "oos": {"pinball20": res["formula"]["pinball20"], "touch20": res["formula"]["touch20"], "improve_vs_base": res["formula"]["improve_vs_base"]}}
    if write:
        M.save(f"stock_pullback_k{k}_night", {"buy": _qfit(XN[okN], y[okN], Q_BUY), "stop": _qfit(XN[okN], y[okN], Q_STOP), "features": NIGHT_FEATS, "trained_at": trained_at})
    if keep is not None:
        keep.update({"n20": n20, "n10": n10, "f20": f20, "f10": f10})
    return res, form


def _train_iv(d: pd.DataFrame, k: int, pm: pd.Series, trained_at: str, write: bool, keep: dict | None) -> dict:
    """3c：FEATS + ivr 的走動式；improve_iv 與「清洗後 FEATS 模型」在同一批 (ivr 可算) 列比較 (spec_fix 4)；另存 ivk 晚 1 日版本。"""
    y, yr, sg = d[f"yLow{k}"], d["year"], d["sigma"]
    _, ivr = _ivr(d["beta"], d["idio"], d[f"sigiv{k}"], sg)
    _, ivrL = _ivr(d["beta"], d["idio"], d[f"sigivL{k}"], sg)
    X, XL = d[FEATS].assign(ivr=ivr), d[FEATS].assign(ivr=ivrL)
    pi, piL = pd.Series(np.nan, index=d.index), pd.Series(np.nan, index=d.index)
    for yv in range(FIRST_TEST_YEAR, int(yr.max()) + 1):
        tr, te = yr < yv, yr == yv
        if tr.sum() < 3000 or not te.any():
            continue
        pi[te] = _pred(_qfit(X[tr], y[tr], Q_BUY), X[te])
        piL[te] = _pred(_qfit(XL[tr], y[tr], Q_BUY), XL[te])
    res = {}
    for tag, pp, rr in (("", pi, ivr), ("_lag1", piL, ivrL)):
        ev = pp.notna() & pm.notna() & np.isfinite(rr)
        yv_, sg_ = y[ev].values, sg[ev].values
        a, b = _pinball(yv_ * sg_, pm[ev].values * sg_, Q_BUY), _pinball(yv_ * sg_, pp[ev].values * sg_, Q_BUY)
        res[f"improve_iv{tag}"] = round(1 - b / a, 4) if a else None
        if not tag:
            res.update(n_oos=int(ev.sum()), pinball20_model=round(a, 4), pinball20_iv=round(b, 4), touch20_iv=round(float((yv_ <= pp[ev].values).mean()), 3),
                       touch20_model=round(float((yv_ <= pm[ev].values).mean()), 3))
            e = pd.DataFrame({"yr": yr[ev].values, "y": yv_ * sg_, "m": pm[ev].values * sg_, "i": pp[ev].values * sg_})
            py = [(_pinball(g["y"].values, g["i"].values, Q_BUY), _pinball(g["y"].values, g["m"].values, Q_BUY)) for _, g in e.groupby("yr")]
            res["yrs_better"] = f"{sum(1 for a_, b_ in py if a_ < b_)}/{len(py)}"
    res["use_iv"] = bool(res.get("improve_iv") is not None and res["improve_iv"] > 0 and 0.15 <= res["touch20_iv"] <= 0.26)
    if write:
        M.save(f"stock_pullback_iv_k{k}", {"buy": _qfit(X, y, Q_BUY), "stop": _qfit(X, y, Q_STOP), "features": IV_FEATS, "trained_at": trained_at})
    if keep is not None:
        keep.update({"pi": pi, "ivr": pd.Series(ivr, index=d.index)})
    return res


def train(write: bool = True, verbose: bool = True, oos_out: dict | None = None) -> dict:
    """oos_out (選用，驗證用)：收集每個 k 的逐列 OOS (base/夜盤/公式/IV) DataFrame。"""
    raw, cleaned = {}, {}
    panel = build_panel(raw_out=raw, clean_out=cleaned)
    _, midx = _market_env()
    cal = sorted(set(midx["date"].astype(str)))
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "features": FEATS, "n_rows": int(len(panel)), "n_stocks": int(panel["stock_id"].nunique()),
           "k": {}, "hold3": {}, "table_oos": {}, "supports": _support_stats(panel),
           "data_clean": cleaned}
    try:   # 3b：夜盤對齊 + 守門
        panel = _attach_night(panel, mkt=midx)
        out["night"] = {"aligned": True, "align_corr": round(_night_guard(panel), 3), "features": NIGHT_FEATS, "clip": NIGHT_CLIP, "first_test_year": FIRST_TEST_YEAR, "k": {}, "formula": {}}
    except Exception as e:  # noqa: BLE001
        log.warning("stock_pullback night: %s", e)
        out["night"] = {"error": str(e)[:200]}
    try:   # 3c：ivmap (原始 frame) + ivk 歷史
        hist = _ivk_hist()
        panel = _iv_columns(panel, raw, midx, hist)
        ivf = [str(r.get("date"))[:10] for r in hist if r.get("ivk")]
        out["ivmap"] = {"formula": "ivr = ln(√(β²·σ_iv,k² + idio²)/σ)，σ_iv,k = ivk_k/√252×100；β/idio 滾動 250 日 (原始報酬、除權息修正)",
                        "ivk_from": min(ivf) if ivf else None, "ivk_to": max(ivf) if ivf else None}
    except Exception as e:  # noqa: BLE001
        log.warning("stock_pullback ivmap: %s", e)
        out["ivmap"] = {"error": str(e)[:200]}
    night_on, iv_on = "error" not in out["night"], "error" not in out["ivmap"]
    # --- 分位模型 (走動式) ---
    for k in KS:
        d = panel.dropna(subset=[f"yLow{k}", "sigma"]); d = d[d["sigma"] > 0]
        X, y, yy_ = d[FEATS], d[f"yLow{k}"], d["year"]
        pm = pd.Series(np.nan, index=d.index); pc = pd.Series(np.nan, index=d.index); pt = pd.Series(np.nan, index=d.index); ps = pd.Series(np.nan, index=d.index)
        for yv in range(FIRST_TEST_YEAR, int(yy_.max()) + 1):
            tr, te = yy_ < yv, yy_ == yv
            if tr.sum() < 3000 or not te.any():
                continue
            pm[te] = _pred(_qfit(X[tr], y[tr], Q_BUY), X[te])
            if night_on or oos_out is not None:   # 停損 OOS (夜盤變體比較 pinball10 用)；min(q10, q20) 同 build()
                ps[te] = np.minimum(_pred(_qfit(X[tr], y[tr], Q_STOP), X[te]), pm[te].values)
            pc[te] = float(y[tr].quantile(Q_BUY))
            pt[te], _ = _apply_table(_table(d[tr]), d[te], k)
        ev = pm.notna()
        yv_, sg = y[ev].values, d.loc[ev, "sigma"].values
        res = {"n_oos": int(ev.sum())}
        for name, q in (("model", pm[ev].values), ("const", pc[ev].values), ("table", pt[ev].values)):
            touch = yv_ <= q
            byy = pd.DataFrame({"t": touch, "y": yy_[ev].values}).groupby("y")["t"].mean()
            res[name] = {"pinball20": round(_pinball(yv_ * sg, q * sg, Q_BUY), 4), "touch20": round(float(touch.mean()), 3), "touch_yr_min": round(float(byy.min()), 3), "touch_yr_max": round(float(byy.max()), 3),
                         "mae_low": round(float(np.mean(np.abs(yv_ - q) * sg)), 3)}
        res["improve_model"] = round(1 - res["model"]["pinball20"] / res["const"]["pinball20"], 3)
        if k == 1:   # 逐股 OOS 改善 (驗證：是否所有股票都受益)
            bs = []
            for sid, gi in d[ev].assign(pm=pm[ev].values, pc=pc[ev].values).groupby("stock_id"):
                if len(gi) < 150:
                    continue
                yy2, sg2 = gi[f"yLow{k}"].values, gi["sigma"].values
                bs.append(round(1 - _pinball(yy2 * sg2, gi["pm"].values * sg2, Q_BUY) / _pinball(yy2 * sg2, gi["pc"].values * sg2, Q_BUY), 3))
            res["by_stock"] = {"n_stocks": len(bs), "improve_median": round(float(np.median(bs)), 3) if bs else None, "improve_min": round(float(min(bs)), 3) if bs else None, "n_negative": int(sum(1 for b_ in bs if b_ < 0))}
        res["improve_table"] = round(1 - res["table"]["pinball20"] / res["const"]["pinball20"], 3)
        res["use_model"] = bool(res["improve_model"] >= 0.02 and 0.15 <= res["model"]["touch20"] <= 0.26)
        out["k"][str(k)] = res
        if write:
            M.save(f"stock_pullback_k{k}", {"buy": _qfit(X, y, Q_BUY), "stop": _qfit(X, y, Q_STOP), "features": FEATS, "trained_at": out["trained_at"]})
        if verbose:
            print(f"  stock_pullback k{k}: pinball 模型 {res['model']['pinball20']} / 查表 {res['table']['pinball20']} / 常數 {res['const']['pinball20']} (改善 模型 {res['improve_model']}、查表 {res['improve_table']})；觸及率 模型 {res['model']['touch20']} 查表 {res['table']['touch20']} 常數 {res['const']['touch20']}；誤差 {res['model']['mae_low']} vs {res['const']['mae_low']}%")
        keep = {} if oos_out is not None else None
        if night_on:
            try:
                rn, fm = _train_night(d, k, pm, ps, cal, out["trained_at"], write, keep)
                out["night"]["k"][str(k)], out["night"]["formula"][str(k)] = rn, fm
                if verbose:
                    print(f"  stock_pullback 夜盤 k{k}: pinball20 {rn['night']['pinball20']} vs 基準 {rn['base']['pinball20']} (改善 {rn['improve_pinball']}、{rn['yrs_better']} 年) 觸及 {rn['night']['touch20']} → use_model {rn['use_model']}；"
                          f"公式 F b {fm['b']} m20 {fm['m20']} m10 {fm['m10']} (改善 {fm['oos']['improve_vs_base']})")
            except Exception as e:  # noqa: BLE001
                log.warning("stock_pullback night k%s: %s", k, e)
                out["night"]["k"][str(k)] = {"error": str(e)[:200], "use_model": False}
        if iv_on:
            try:
                ri = _train_iv(d, k, pm, out["trained_at"], write, keep)
                out["ivmap"][str(k)] = ri
                if verbose:
                    print(f"  stock_pullback IV 映射 k{k}: pinball20 {ri['pinball20_iv']} vs FEATS {ri['pinball20_model']} (改善 {ri['improve_iv']}、晚 1 日 {ri['improve_iv_lag1']}、{ri['yrs_better']} 年) 觸及 {ri['touch20_iv']} → use_iv {ri['use_iv']}")
            except Exception as e:  # noqa: BLE001
                log.warning("stock_pullback ivmap k%s: %s", k, e)
                out["ivmap"][str(k)] = {"error": str(e)[:200], "use_iv": False}
        if oos_out is not None:
            o = d.loc[ev, ["date", "stock_id", "year", "close", "sigma", f"pathLow{k}"] + [c_ for c_ in ("night_chg_pct", "beta250") if c_ in d]].rename(columns={f"pathLow{k}": "y_pct"})
            o["pm"], o["ps"], o["pc"] = pm[ev].values, ps[ev].values, pc[ev].values
            for key_, s_ in (keep or {}).items():
                o[key_] = s_[ev].values
            oos_out[k] = o.reset_index(drop=True)
    # --- 止跌機率 (走動式) ---
    d = panel.dropna(subset=["hold3"]); X, y, yy_ = d[FEATS], d["hold3"], d["year"]
    pm = pd.Series(np.nan, index=d.index); pt = pd.Series(np.nan, index=d.index); pb = pd.Series(np.nan, index=d.index)
    for yv in range(FIRST_TEST_YEAR, int(yy_.max()) + 1):
        tr, te = yy_ < yv, yy_ == yv
        if tr.sum() < 3000 or not te.any():
            continue
        pm[te] = np.clip(_pred(_rfit(X[tr], y[tr]), X[te]), 0, 1); pb[te] = float(y[tr].mean())
        _, pt[te] = _apply_table(_table(d[tr]), d[te], 1)
    ev = pm.notna(); yv_ = y[ev].values
    def tiers(p):
        hi, lo = np.quantile(p, 0.7), np.quantile(p, 0.3)
        return {"top30_hold": round(float(yv_[p >= hi].mean()), 3), "bot30_hold": round(float(yv_[p <= lo].mean()), 3), "brier": round(float(np.mean((p - yv_) ** 2)), 4)}
    out["hold3"] = {"n_oos": int(ev.sum()), "base": round(float(yv_.mean()), 3), "model": tiers(pm[ev].values), "table": tiers(pt[ev].values), "const_brier": round(float(np.mean((pb[ev].values - yv_) ** 2)), 4)}
    cal_ = pd.DataFrame({"p": pm[ev].values, "y": yv_, "year": yy_[ev].values}); cal_["bin"] = pd.cut(cal_["p"], [0, .3, .4, .5, .6, .7, 1.0])
    out["hold3"]["calibration"] = [{"bin": str(b_), "rate": round(float(g_["y"].mean()), 3), "n": int(len(g_))} for b_, g_ in cal_.groupby("bin", observed=True)]
    out["hold3"]["by_year"] = {int(yv): {"top30": round(float(g_.loc[g_["p"] >= g_["p"].quantile(.7), "y"].mean()), 3), "bot30": round(float(g_.loc[g_["p"] <= g_["p"].quantile(.3), "y"].mean()), 3)} for yv, g_ in cal_.groupby("year")}
    bs2 = []
    for sid, g_ in d[ev].assign(p=pm[ev].values).groupby("stock_id"):
        if len(g_) >= 150:
            bs2.append(round(float(g_.loc[g_["p"] >= g_["p"].quantile(.7), "hold3"].mean() - g_.loc[g_["p"] <= g_["p"].quantile(.3), "hold3"].mean()), 3))
    out["hold3"]["by_stock"] = {"n_stocks": len(bs2), "gap_median": round(float(np.median(bs2)), 3) if bs2 else None, "gap_min": round(float(min(bs2)), 3) if bs2 else None, "n_negative": int(sum(1 for b_ in bs2 if b_ < 0))}
    fb = _rfit(X, y)
    imp = np.mean([mm.feature_importances_ for mm in fb], axis=0); imp = imp / imp.sum()
    out["hold3"]["importance"] = sorted(({"f": f, "name": NAMES.get(f, f), "w": round(float(w), 3)} for f, w in zip(FEATS, imp) if w > 0.03), key=lambda x: -x["w"])
    if write:
        M.save("stock_pullback_hold3", {"models": fb, "features": FEATS, "trained_at": out["trained_at"]})
    if verbose:
        h3 = out["hold3"]; print(f"  止跌機率 hold3: 基準 {h3['base']}；模型 高 30% {h3['model']['top30_hold']} / 低 30% {h3['model']['bot30_hold']} (brier {h3['model']['brier']} vs 常數 {h3['const_brier']})；查表 {h3['table']['top30_hold']} / {h3['table']['bot30_hold']}；主要特徵 " + "、".join(f"{x['name']} {x['w']}" for x in h3["importance"][:5]))
        for key, s in out["supports"].items():
            print(f"  個股支撐 {s['name']}: 回測 {s['n_tested']} 止跌率 {s['hold_rate']} (多頭 {s.get('hold_bull')} / 空頭 {s.get('hold_bear')})，止跌後 3 日 {s['bounce3_after_hold']}%，跌破後 {s['break3_after_fail']}%")
        dc = {s: v for s, v in cleaned.items() if v["zero_rows"] or v["splits"]}
        print("  價格清洗 (零價列/分割)：" + "、".join(f"{s} {v['zero_rows']} 列{' 分割 ' + ','.join(str(x['date']) + '×' + str(x['factor']) for x in v['splits']) if v['splits'] else ''}" for s, v in dc.items()))
    out["table"] = _table(panel.dropna(subset=["yLow1"]))   # 全樣本查表 (前端離線用)
    if write:
        M.save_json("stock_pullback", out)
    return out


# ------------------------------------------------------------------ 上線 (單一標的今日)
def _night_gate(d0: str, night_ret, night_for) -> tuple[bool, str | None]:
    """目標日檢查 (spec_fix 1)：night_for 已給、個股資料日的下一交易日 == night_for、night_ret 有限 → 可用夜盤變體 (再看各 k 的 use_model)。"""
    if night_for is None or night_ret is None:
        return False, None
    try:
        v = float(night_ret)
    except (TypeError, ValueError):
        return False, "夜盤漲跌無效"
    if not math.isfinite(v):
        return False, "夜盤漲跌無效"
    try:
        from ..sources import twse
        nx = twse.next_trading_days(d0, 1)[0]
    except Exception as e:  # noqa: BLE001
        return False, f"交易日曆檢查失敗 ({e})"
    if nx != str(night_for)[:10]:
        return False, f"個股資料日 {d0} 的下一交易日 {nx} ≠ 夜盤目標日 {str(night_for)[:10]} (個股日 K 可能尚未更新)"
    return True, None


def _iv_live(raw: pd.DataFrame, midx: pd.DataFrame, d0: str, ivk_live: dict | None) -> dict:
    """上線 IV 映射輸入：β/idio (原始 frame 的 ivmap，取價格日 d0 那列) + ivk (≤ d0 且 ≥ 前一交易日)。失敗 → {"why": ...}。"""
    try:
        m = ivmap(raw, midx)
        r = m[m["date"] == d0]
        beta, idio = (float(r["beta"].iloc[-1]), float(r["idio"].iloc[-1])) if len(r) else (float("nan"), float("nan"))
    except Exception as e:  # noqa: BLE001
        return {"why": f"β/idio 無法計算 ({str(e)[:60]})"}
    if not (math.isfinite(beta) and math.isfinite(idio)):
        return {"why": f"報酬不足 {IV_MINP} 筆，β/idio 無法計算"}
    prev = max((x for x in midx["date"].astype(str).str[:10] if x < d0), default=None)
    pk = _pick_ivk(d0, prev, ivk_live)
    if not pk:
        return {"beta": beta, "idio": idio, "why": "ivk 缺或過期 (>1 交易日)"}
    return {"beta": beta, "idio": idio, "ivk_date": pk["date"], "stale_td": pk["stale_td"], "ivk": pk["ivk"]}


def _exdiv_live(sid: str, exdiv_ev: dict | None, d0: str, close: float) -> tuple[dict, list[dict], list[str]]:
    """3a：除息事件 → (info, 可扣事件 (ex_date 已對到交易日), 交易日曆 cal[0..2])。取不到 → status unknown、不扣。"""
    from ..sources import exdiv as XD
    try:
        ev0 = exdiv_ev if exdiv_ev is not None else XD.load_all([sid], d0, XD.get_json).get(sid, {})
        ev = XD.rebase(ev0, d0) if ev0 else {}
    except Exception as e:  # noqa: BLE001
        return {"status": "unknown", "why": str(e)[:80]}, [], []
    if not ev:
        return {"status": "unknown"}, [], []
    try:
        from ..sources import twse
        cal = twse.next_trading_days(d0, 3)
    except Exception:  # noqa: BLE001
        from . import events as EV
        cal = [EV.next_td(d0, j) for j in (1, 2, 3)]
    ups = []
    for u in ev.get("upcoming") or []:
        cash = u.get("cash") or u.get("cash_est")
        ex = str(u.get("ex_date") or "")[:10]
        if u.get("suspicious") or not cash or not close or cash / close > EXDIV_MAX_YIELD or not ex or ex <= d0:
            continue
        eff = next((c_ for c_ in cal if c_ >= ex), None)      # 颱風改期：預告日不是交易日 → 對到下一交易日 (spec_fix 6)
        if eff is None:
            continue
        kind = str(u.get("kind") or "")
        ups.append({**u, "ex_date": eff, "ex_date_announced": ex if eff != ex else None, "stock_part": bool("權" in kind or (u.get("stock") or 0) > 0)})
    srcs = ev.get("sources") or {}
    live = any(str(srcs.get(x, "")) in ("ok", "stale") for x in ("TWT48U", "TWT49U", "FinMind")) or not srcs or ev.get("carried_from")
    info = {"status": "ok" if live else "unknown", "upcoming": [{k_: u.get(k_) for k_ in ("ex_date", "cash", "cash_est", "status", "suspicious", "kind")} for u in (ev.get("upcoming") or [])[:3]],
            "sources": ev.get("sources")}
    return info, ups, cal


def _iv_block(ivx: dict, k: int, sg: float, ri: dict) -> dict | None:
    """每個 k 的 IV 映射說明：{sigma_ivmap, beta, idio, ivk, ivk_date, stale_td, ivr, used, why, oos}；IV 模型未訓練 → None。"""
    if not ivx:
        return None
    if "beta" not in ivx:
        return {"used": False, "why": ivx.get("why") or "β/idio 無法計算"}
    v = (ivx.get("ivk") or {}).get(str(k))
    iv = {"beta": round(ivx["beta"], 3), "idio": round(ivx["idio"], 3), "ivk": v, "ivk_date": ivx.get("ivk_date"), "stale_td": ivx.get("stale_td"), "used": False}
    if ri:
        iv["oos"] = {x_: ri.get(x_) for x_ in ("improve_iv", "improve_iv_lag1", "pinball20_iv", "pinball20_model", "touch20_iv", "yrs_better", "use_iv")}
    s_iv = _sig_iv(v)
    if s_iv is None:
        iv["why"] = ivx.get("why") or f"ivk_{k} 缺"
        return iv
    smv, r_ = _ivr([ivx["beta"]], [ivx["idio"]], [s_iv], [sg])
    iv["sigma_ivmap"] = round(float(smv[0]), 3)
    if math.isfinite(r_[0]):
        iv["ivr"], iv["ivr_raw"] = round(float(r_[0]), 4), float(r_[0])
        if not ri.get("use_iv"):
            iv["why"] = "use_iv=False (IV 版未勝過 FEATS 模型)"
    else:
        iv["why"] = "ivr 無法計算"
    return iv


def build(stock_id: str, price: pd.DataFrame | None = None, base_px: float | None = None, exdiv_ev: dict | None = None,
          night_ret: float | None = None, night_for: str | None = None, ivk_live: dict | None = None) -> dict | None:
    """單一標的今日：模型買點/停損 (k=1..3)、止跌機率、回落脈絡、支撐 (pooled 止跌率)。price 需含 date/open/high/low/close/volume (FinMind 原始列含 change 時才有 IV 映射)。
    exdiv_ev：exdiv.load_all 的單檔事件 (None → 自行抓；{} → 視為取不到)；night_ret/night_for：完整夜盤漲跌% 與其目標交易日；ivk_live：taifex_opt.features()。"""
    st = M.load_json("stock_pullback")
    if not st:
        return None
    if price is None:
        from ..sources import finmind
        price = finmind.stock_price(stock_id, "2024-01-01")
    if price is None or price.empty or len(price) < 70:
        return None
    price, raw, splits = _clean_price(price, stock_id)            # B2
    if len(price) < 70:
        return None
    env, midx = _market_env()
    d = features_from_ohlc(price[OHLCV], env)
    row = d.iloc[[-1]].copy()
    close = float(row["close"].iloc[0]); px_base = float(base_px or close); sg = float(row["sigma"].iloc[0])
    if not (sg > 0):
        return None
    d0 = str(row["date"].iloc[0])[:10]
    beta = float(row["beta250"].iloc[0])
    out = {"sid": stock_id, "date": d0, "sigma": round(sg, 3), "k": {}, "stock": True, "close": round(close, 2), "beta250": round(beta, 3), "variant": "base",
           "context": {f: (round(float(row[f].iloc[0]), 2) if pd.notna(row[f].iloc[0]) else None) for f in ("dd_hi20", "days_since_hi20", "streak", "clv", "lower_wick", "vol_ratio", "bias20", "lo20_dist")}, "cell": cell_of(row)}
    if len(raw) != len(price) or splits:
        out["data_clean"] = {"zero_rows": int(len(raw) - len(price)), "splits": splits}
    use_n, n_why = _night_gate(d0, night_ret, night_for)
    nv = float(np.clip(float(night_ret), -NIGHT_CLIP, NIGHT_CLIP)) if use_n else None
    ivx = _iv_live(raw, midx, d0, ivk_live) if (st.get("ivmap") or {}).get("1") else {}      # IV 映射尚未訓練 → 不輸出 iv 區塊 (FEATS 模型)
    xinfo, xups, xcal = _exdiv_live(stock_id, exdiv_ev, d0, close)
    out["exdiv"] = xinfo
    nst, ivst = (st.get("night") or {}).get("k") or {}, st.get("ivmap") or {}
    from ..sources import exdiv as XD
    for k in KS:
        b = M.load(f"stock_pullback_k{k}"); r = (st.get("k") or {}).get(str(k)) or {}
        if not b:
            continue
        rn, ri = nst.get(str(k)) or {}, ivst.get(str(k)) or {}
        bn = M.load(f"stock_pullback_k{k}_night") if (use_n and rn.get("use_model")) else None
        iv, ivr = _iv_block(ivx, k, sg, ri), None
        if iv is not None and iv.get("ivr") is not None:
            ivr = iv["ivr_raw"]
        kk: dict = {"variant": "base"}
        if bn:   # 3b 夜盤變體：水準以收盤 D 為基準；不加 ivr (兩者未聯合驗證)
            xr = row.copy(); xr["night_chg_pct"] = nv; xr["bnight"] = beta * nv
            q20 = float(_pred(bn["buy"], xr[bn["features"]])[0]) * sg; q10 = min(float(_pred(bn["stop"], xr[bn["features"]])[0]) * sg, q20)
            px0 = close
            kk.update(variant="night", use_model=True,
                      oos={"improve_pinball": rn.get("improve_pinball"), "model": rn.get("night"), "sigma": rn.get("base"), "base": rn.get("base"),
                           "stock": (rn.get("by_stock") or {}).get(stock_id), "by_stock": rn.get("by_stock_summary"), "yrs_better": rn.get("yrs_better")})
            if iv is not None:
                iv["why"] = "夜盤變體不加 IV 映射 (未聯合驗證)"
        else:
            px0, bm, X = px_base, b, row
            if ivr is not None and ri.get("use_iv"):     # 3c 閘門：ivr 可算且 use_iv → IV 模型；否則 FEATS 模型 (不依賴 LGB 缺值分支)
                bi = M.load(f"stock_pullback_iv_k{k}")
                if bi:
                    bm, X, iv["used"] = bi, row.assign(ivr=ivr), True
                    iv.pop("why", None)
                else:
                    iv["why"] = "IV 模型檔缺"
            q20 = float(_pred(bm["buy"], X[bm["features"]])[0]) * sg; q10 = min(float(_pred(bm["stop"], X[bm["features"]])[0]) * sg, q20)
            kk.update(use_model=bool(r.get("use_model")), oos={"improve_pinball": r.get("improve_model"), "model": r.get("model"), "sigma": r.get("const"), "by_stock": r.get("by_stock")})
        if iv is not None:
            iv.pop("ivr_raw", None)
            kk["iv"] = iv
        # 3a 除息扣除 (看盤價口徑)；未公告金額 (cash_est) 只在 k=1 扣 (spec_fix 5)
        hits = XD.exdiv_in_window(xups, xcal, k) if xups else []
        use = [(j, e) for j, e in hits if e.get("cash") or k == 1]
        if use:
            cash_t = sum(float(e.get("cash") or e.get("cash_est")) for _, e in use)
            ded = cash_t / close * 100
            kk["buy_model_tr"], kk["stop_model_tr"] = round(px0 * (1 + q20 / 100), 2), round(px0 * (1 + q10 / 100), 2)
            kk["low20_pct_tr"], kk["low10_pct_tr"] = round(q20, 2), round(q10, 2)
            q20 -= ded; q10 -= ded
            e0 = use[0][1]
            kk["exdiv_adj"] = {"ex_date": e0["ex_date"], "cash": round(cash_t, 4), "pct": round(ded, 3), "est": any(not e.get("cash") for _, e in use),
                               "stock_part": any(e.get("stock_part") for _, e in use), "day": use[0][0]}
            if e0.get("ex_date_announced"):
                kk["exdiv_adj"]["ex_date_announced"] = e0["ex_date_announced"]
        elif hits:
            kk["exdiv_skip"] = {"ex_date": hits[0][1]["ex_date"], "why": "除息金額未公告 (以上次配息估計)：只在 k=1 扣除"}
        kk.update(buy_model=round(px0 * (1 + q20 / 100), 2), stop_model=round(px0 * (1 + q10 / 100), 2), low20_pct=round(q20, 2), low10_pct=round(q10, 2))
        out["k"][str(k)] = kk
    if any(v.get("variant") == "night" for v in out["k"].values()):
        out["variant"], out["night_ret"], out["night_for"] = "night", round(float(night_ret), 3), str(night_for)[:10]
        try:   # 休市後首日 (spec_fix 3)：不丟夜盤變體，但加註觸及偏高
            from . import events as EV
            if EV.n_us_between(d0, str(night_for)[:10]) >= 2:
                out["night_note"] = POST_CLOSURE_NOTE
        except Exception:  # noqa: BLE001
            pass
    elif night_ret is not None and night_for is not None:
        out["night_skip"] = n_why or "夜盤模型未達上線門檻 (use_model=False)"
    hb = M.load("stock_pullback_hold3")
    if hb:
        p = float(np.clip(_pred(hb["models"], row[hb["features"]])[0], 0, 1))
        h3 = st.get("hold3") or {}
        cal_ = next((c for c in (h3.get("calibration") or []) if _in_bin(c["bin"], p)), None)
        out["hold3"] = {"p": round(p, 3), "base": h3.get("base"), "top30": (h3.get("model") or {}).get("top30_hold"), "bot30": (h3.get("model") or {}).get("bot30_hold"),
                        "cal_rate": cal_["rate"] if cal_ else None, "cal_n": cal_["n"] if cal_ else None, "by_year": h3.get("by_year"), "by_stock": h3.get("by_stock"),
                        "label": "回落可能已到低點" if p >= (h3.get("base") or 0.5) + 0.08 else "回落可能未完" if p <= (h3.get("base") or 0.5) - 0.08 else "不明顯"}
    out["supports"] = []
    bull = bool(close >= float(row["ma20"].iloc[0])) if pd.notna(row["ma20"].iloc[0]) else None
    for key, name in SUPPORTS.items():
        s = float(row[key].iloc[0]) if pd.notna(row[key].iloc[0]) else None
        if s is None or not (s < px_base and s > px_base * 0.93):
            continue
        ss = (st.get("supports") or {}).get(key) or {}
        out["supports"].append({"key": key, "name": name, "level": round(s, 2), "dist_pct": round((s / px_base - 1) * 100, 2), "hold_rate": ss.get("hold_rate"),
                                "hold_regime": (ss.get("hold_bull") if bull else ss.get("hold_bear")) if bull is not None else None, "regime": "多頭" if bull else "空頭", "bounce3": ss.get("bounce3_after_hold"), "break3": ss.get("break3_after_fail"), "n": ss.get("n_tested")})
    out["supports"].sort(key=lambda x: -x["level"])
    return out


def _in_bin(b: str, p: float) -> bool:
    try:
        lo, hi = b.strip("()[]").split(",")
        return float(lo) < p <= float(hi)
    except Exception:  # noqa: BLE001
        return False


def client_table() -> dict | None:
    """給前端 (watchlist.json) 的離線查表：sigma 乘數 + 止跌率 + 支撐統計 + 驗證摘要 (+ 除息 OOS、夜盤公式 F、IV 映射 OOS)。"""
    st = M.load_json("stock_pullback")
    if not st:
        return None
    out = {"table": st.get("table"), "supports": st.get("supports"), "hold3_base": (st.get("hold3") or {}).get("base"),
           "oos": {"k": {k: {"table": v.get("table"), "const": v.get("const"), "improve_table": v.get("improve_table")} for k, v in (st.get("k") or {}).items()},
                   "hold3_table": (st.get("hold3") or {}).get("table"), "exdiv": EXDIV_OOS},
           "trained_at": st.get("trained_at")}
    nt = st.get("night") or {}
    if nt.get("formula"):   # 公式 F (清單外個股 / 伺服器沒有夜盤水準時)：low_k = b·β250·clip(night,±8) + m·σ
        def _f_ok(k, v):   # 伺服器夜盤閘門 (use_model) 與公式 F 自身事前門檻 (改善 ≥3%、觸及 0.15~0.27) 都要過，前端才可套公式
            o = (v or {}).get("oos") or {}
            return bool(((nt.get("k") or {}).get(k) or {}).get("use_model")) and (o.get("improve_vs_base") or 0) >= 0.03 and 0.15 <= (o.get("touch20") or 0) <= 0.27
        out["night_formula"] = {k: {x: v.get(x) for x in ("b", "m20", "m10")} for k, v in nt["formula"].items() if _f_ok(k, v)}
        out["night_clip"] = nt.get("clip", NIGHT_CLIP)
    if nt.get("k"):
        out["oos"]["night"] = {k: {"use_model": v.get("use_model"), "improve_pinball": v.get("improve_pinball"), "yrs_better": v.get("yrs_better"),
                                   "night": {x: (v.get("night") or {}).get(x) for x in ("pinball20", "touch20")}, "base": {x: (v.get("base") or {}).get(x) for x in ("pinball20", "touch20")},
                                   "formula": (nt.get("formula") or {}).get(k, {}).get("oos"),
                                   "by_stock_touch20": {s: b_.get("touch20") for s, b_ in (v.get("by_stock") or {}).items()}}
                               for k, v in nt["k"].items() if isinstance(v, dict) and "error" not in v}
    iv = st.get("ivmap") or {}
    if any(k in iv for k in ("1", "2", "3")):
        out["ivmap"] = {k: {x: (iv.get(k) or {}).get(x) for x in ("pinball20_model", "pinball20_iv", "improve_iv", "improve_iv_lag1", "touch20_iv", "yrs_better", "use_iv")} for k in ("1", "2", "3") if k in iv}
    return out
