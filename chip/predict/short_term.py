"""前五日預測模組 v2 (視野 1/2/3/5 日)：目標是提高「方向命中率」，而不是只做期望報酬排序。

作法 (全部 2014~ 逐年擴張視窗樣本外驗證，用 OOS 結果選模型與門檻，不用訓練集內表現)：
1. 短線專用特徵：只留與 1~5 日報酬有關的 (當日/前兩日漲跌、跳空、振幅、收盤位置、近 5 日上漲天數、乖離、量能、
   外資當日/5 日、外資期貨變化、結算週、前晚美股/費半/ADR/韓股、VIX)，去掉 20~60 日慢變數 → 少雜訊、少過擬合。
2. 兩種模型 + 集成：淺層 LightGBM 多種子 (非線性) 與 Ridge (線性、穩定)，各自標準化後平均 → 通常比單一模型更穩。
3. 夜盤變體：加入「前晚夜盤台指期漲跌」(2017-05 起有資料，隔日開盤跳空 r≈0.71) 另訓一組，收盤後~開盤前用它。
4. 命中率提升的關鍵是「有把握才叫方向」：以 OOS 預測分位切三檔 (前 30% 偏多 / 後 30% 偏空 / 中間中性)，
   只有該檔位 OOS 命中率高於基準 3 個百分點以上才啟用，並把該檔位的歷史命中率一起輸出，讓使用者知道這次叫牌的可信度。
5. 每個視野在 {lgb, ridge, ens} 中依 OOS「叫牌命中率」自動選最佳；夜盤變體另選。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from ..analysis import backtest
from ..analysis.common import zscore
from ..sources import finmind
from . import model as M
from .features import FEATURE_NAMES, market_matrix

log = logging.getLogger(__name__)
HORIZONS = (1, 2, 3, 5)
FIRST_TEST_YEAR = 2014
FIRST_TEST_YEAR_NIGHT = 2020
ST_FEATURES = [
    "ret1", "ret1_l1", "ret1_l2", "ret5", "bias5", "bias20", "gap_open", "range_pct", "clv", "up5",
    "hi20_dist", "lo20_dist", "vola20", "vola_ratio", "vol_ratio", "amount_5d_ratio",
    "foreign_z1", "foreign_z5", "foreign_streak", "trust_z5", "dealer_z1",
    "fut_foreign_pct", "fut_foreign_chg1_z", "fut_foreign_chg5_z", "foreign_consistency", "margin_chg5_pct", "short_chg5_z",
    "f_fut_foreign", "f_reversion", "composite_smooth", "composite_chg5", "state_bull", "state_bear",
    "dow", "days_to_settle", "settle_week", "days_to_month_end",
    "g_vix_level", "g_vix_chg", "g_sox_r1", "g_sox_r5", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1",
    "g_kospi_r1", "g_kospi_r5", "g_usdtwd_r5", "g_dxy_r1", "g_us10y_r5", "g_vix_term",
]
NIGHT_FEATURE = "night_chg_pct"
# v2.1 新資料源 (2026-09-13 消融研究後加入)：
#   亞股同日收盤 (韓/日/港，台股收盤後 1 小時內已知) → 無夜盤時 1~3 日 IC 0.05→0.09
#   小台散戶多空比、選擇權外資/自營 買賣權淨部位、台指期近月期現價差、期貨 OI/量 (2018-06~) → 無夜盤時 3/5 日偏多檔命中 59%→63%
#   美股同夜 (ADR/費半) 對夜盤變體無增益 (夜盤已反映) → 不加
ASIA_FEATURES = ["kospi_r0", "nikkei_r0", "hsi_r0"]
CHIP_FEATURES = ["mtx_retail_ratio", "mtx_retail_chg5", "mtx_foreign_net_z", "txo_f_call_z", "txo_f_put_z", "txo_f_cp_diff_z", "txo_d_cp_diff_z",
                 "tx_basis_pct", "tx_basis_chg1", "tx_oi_chg1_z", "tx_vol_z"]
# v2.2 價格型態特徵 (2026-09-13 R3 方向研究)：2010~ 條件統計顯示 1 日方向為「短線動能」(韓股同日/振幅/跳空/K 棒) 而非均值回歸。
#   加入第四組特徵集後，1 日不含夜盤 OOS 叫牌命中 0.547→0.558 (IC 0.100→0.116；滾動門檻 0.553→0.563；種子重抽 +0.4~+1.6pp，10/10 為正)，
#   2/3 日與夜盤變體在 ±1pp 雜訊內無差異 (由 _pick 自動保留原特徵集；h2 base 可能無害地翻選 +px|ridge 0.601 vs 0.602)。
#   注意：增益來自 ~300 個邊際檔位日的更替，屬「小但方向一致」的改善；2025~26 未見增益。盤中以未收盤 K 棒計算時屬暫定值。
PX_FEATURES = ["streak", "range_ratio", "body", "upper_wick", "lower_wick", "gap_filled", "ret1_x_clv", "kospi_rel0", "new_hi20", "new_lo20"]
# v2.3 跨市場/主力特徵 (2026-09-22 深化研究)：台股 5 日相對 KOSPI/S&P (落後 >3% 歷史補漲 64%，13 年 77%)、聰明錢−散戶差 (5~20 日 IC 0.045~0.049，8~9/9 年)、恆生 5 日
XM_FEATURES = ["rel_kospi5", "rel_sp5", "smart_spread", "hsi_r5", "smart2", "smart_core", "fut_chg10_z", "mtx_retail_inv"]
FEATURE_SETS = {"short": ST_FEATURES, "+asia": ST_FEATURES + ASIA_FEATURES, "+asia+chip": ST_FEATURES + ASIA_FEATURES + CHIP_FEATURES,
                "+asia+chip+px": ST_FEATURES + ASIA_FEATURES + CHIP_FEATURES + PX_FEATURES,
                "+asia+chip+px+xm": ST_FEATURES + ASIA_FEATURES + CHIP_FEATURES + PX_FEATURES + XM_FEATURES}
NAMES = {**FEATURE_NAMES, "rel_kospi5": "台股 5 日相對 KOSPI", "rel_sp5": "台股 5 日相對 S&P", "smart_spread": "聰明錢−散戶差", "hsi_r5": "恆生 5 日", "smart2": "聰明錢 v2", "smart_core": "聰明錢核心", "fut_chg10_z": "外資期貨 10 日增減 z", "mtx_retail_inv": "小台散戶反向", "ret1_l1": "前 1 日漲跌%", "ret1_l2": "前 2 日漲跌%", "gap_open": "今日開盤跳空%", "range_pct": "今日振幅%",
         "clv": "收盤在當日區間位置", "up5": "近 5 日上漲天數", NIGHT_FEATURE: "前晚夜盤台指期%",
         "kospi_r0": "KOSPI 今日%", "nikkei_r0": "日經今日%", "hsi_r0": "恆生今日%",
         "mtx_retail_ratio": "小台散戶多空比", "mtx_retail_chg5": "小台散戶多空比 5 日變化", "mtx_foreign_net_z": "小台外資淨部位 z",
         "txo_f_call_z": "選擇權外資買權淨 z", "txo_f_put_z": "選擇權外資賣權淨 z", "txo_f_cp_diff_z": "選擇權外資買減賣權 z", "txo_d_cp_diff_z": "選擇權自營買減賣權 z",
         "tx_basis_pct": "台指期近月期現價差%", "tx_basis_chg1": "期現價差日變化", "tx_oi_chg1_z": "台指期 OI 日變化 z", "tx_vol_z": "台指期成交量 z",
         "streak": "連漲/連跌天數 (帶號)", "range_ratio": "今日振幅/20 日均振幅", "body": "K 棒實體%", "upper_wick": "上影線%", "lower_wick": "下影線%",
         "gap_filled": "今日跳空已回補", "ret1_x_clv": "漲跌×收盤位置", "kospi_rel0": "台股相對韓股同日強弱", "new_hi20": "創 20 日新高", "new_lo20": "創 20 日新低"}
LGB_PARAMS = dict(M.PARAMS, n_estimators=120, min_child_samples=150)
RIDGE_ALPHA = 30.0
TIER = 0.30          # 前/後 30% 才叫方向 (一般)
TIER_STRONG = 0.15   # 前/後 15% 為「強」叫牌 (另報命中率)
MIN_EDGE = 0.03      # 檔位命中率需高於基準 3 個百分點才啟用


# ------------------------------------------------------------------ 特徵
def build_matrix(scored: pd.DataFrame, night: pd.DataFrame | None = None) -> pd.DataFrame:
    d = market_matrix(scored)
    c = d["close"].astype(float)
    prev = c.shift(1)
    d["ret1_l1"] = d["ret1"].shift(1)
    d["ret1_l2"] = d["ret1"].shift(2)
    o, h, l = d["open"].astype(float), d["high"].astype(float), d["low"].astype(float)
    d["gap_open"] = (o / prev - 1) * 100
    d["range_pct"] = (h - l) / prev * 100
    d["clv"] = np.where((h - l) > 0, ((c - l) - (h - c)) / (h - l).replace(0, np.nan), 0.0)
    d["up5"] = (d["ret1"] > 0).astype(int).rolling(5).sum()
    # 夜盤：FinMind after_market 的 date = 該夜盤準備的「隔一交易日」→ 列 D 要用 date == D 的下一列日期 的夜盤
    d[NIGHT_FEATURE] = np.nan
    if night is not None and not night.empty:
        nm = dict(zip(night["date"].astype(str), night["night_chg_pct"].astype(float)))
        nxt = d["date"].astype(str).shift(-1)
        d[NIGHT_FEATURE] = nxt.map(nm)
    try:
        d = _add_extra_features(d)
    except Exception as e:  # noqa: BLE001
        log.warning("extra features: %s", e)
    for col in ST_FEATURES + ASIA_FEATURES + CHIP_FEATURES + PX_FEATURES + XM_FEATURES:
        if col not in d:
            d[col] = np.nan
    return d


def _asof_return(dates: pd.Series, sym: str) -> pd.Series:
    """該市場「同一天 D」的日報酬% (亞股同日收盤，台股收盤後已知)。
    2026-09-24 改為嚴格同日：該市場 D 日沒有資料 (休市、或 Yahoo 快取尚未更新) → NaN，交給模型的缺值處理與投票的「無資料」。
    舊版用「<= D 最後一筆」會把過期值 (例如上週五的 +1.18%) 靜默填到今天，實際發生於 09-24 早上 (恆生/KOSPI 三天同值)，
    污染模型特徵、判斷總結投票、信心分層與預測邏輯總表。"""
    from ..sources import global_markets as gm
    h = gm.history(sym).sort_values("date")
    h["r"] = h["close"].astype(float).pct_change() * 100
    m = dict(zip(h["date"].astype(str), h["r"]))
    return pd.Series([m.get(str(dte), np.nan) for dte in dates], index=dates.index, dtype=float)


def asia_last_dates() -> dict:
    """各亞股資料最後日期 (前端顯示資料新鮮度用)。"""
    from ..sources import global_markets as gm
    out = {}
    for col, sym in (("kospi_r0", "^KS11"), ("nikkei_r0", "^N225"), ("hsi_r0", "^HSI")):
        try:
            h = gm.history(sym)
            out[col] = str(h["date"].max())[:10] if h is not None and len(h) else None
        except Exception:  # noqa: BLE001
            out[col] = None
    return out


def _add_price_pattern_features(d: pd.DataFrame) -> pd.DataFrame:
    """價格型態特徵 (PX_FEATURES)：全部只用 D 日收盤前資訊；kospi_rel0 用台股收盤後已知的韓股同日收盤。

    只依賴 OHLC 與 build_matrix 已算好的 ret1/gap_open/range_pct/clv/hi20/lo20，不需任何網路資料；
    kospi_r0 缺值 (Yahoo 快取未更新) 時 kospi_rel0 為 NaN → LGB 原生處理、Ridge 以訓練中位數補值。
    """
    c, o = d["close"].astype(float), d["open"].astype(float)
    h, l = d["high"].astype(float), d["low"].astype(float)
    prev = c.shift(1)
    r1 = d["ret1"].astype(float)
    sg = np.sign(r1)
    d["streak"] = (sg.groupby((sg != sg.shift()).cumsum()).cumcount() + 1) * sg          # 連漲 +n / 連跌 -n
    d["range_ratio"] = d["range_pct"] / d["range_pct"].rolling(20).mean()
    d["body"] = (c - o) / prev * 100
    d["upper_wick"] = (h - np.maximum(c, o)) / prev * 100
    d["lower_wick"] = (np.minimum(c, o) - l) / prev * 100
    d["gap_filled"] = (((d["gap_open"] > 0) & (l <= prev)) | ((d["gap_open"] < 0) & (h >= prev))).astype(int)
    d["ret1_x_clv"] = r1 * d["clv"]
    d["new_hi20"] = (c >= d["hi20"]).astype(int)   # hi20 含今日 → 收盤即知
    d["new_lo20"] = (c <= d["lo20"]).astype(int)
    d["kospi_rel0"] = (d["kospi_r0"].astype(float) - r1) if "kospi_r0" in d else np.nan
    return d


def _add_extra_features(d: pd.DataFrame) -> pd.DataFrame:
    dates = d["date"].astype(str)
    c = d["close"].astype(float)
    for col, sym in (("kospi_r0", "^KS11"), ("nikkei_r0", "^N225"), ("hsi_r0", "^HSI")):
        try:
            d[col] = _asof_return(dates, sym)
        except Exception as e:  # noqa: BLE001
            log.debug("%s: %s", col, e)
    # 價格型態 (放在 FinMind 籌碼抓取之前：不依賴網路，籌碼資料失敗時仍可算出)
    try:
        d = _add_price_pattern_features(d)
    except Exception as e:  # noqa: BLE001
        log.warning("price pattern features: %s", e)
    try:   # v2.3 跨市場/主力特徵 (XM_FEATURES)
        from . import crossmkt as _XM
        d = _XM.add_features(d)
    except Exception as e:  # noqa: BLE001
        log.warning("crossmkt features: %s", e)
    by = lambda s: pd.Series([s.get(x, np.nan) for x in dates], index=d.index, dtype=float)  # noqa: E731
    # 小台 MTX：散戶淨部位 = -(三大法人淨部位)，除以近月全部 OI
    mtx = finmind.fetch("TaiwanFuturesInstitutionalInvestors", "MTX", "2018-01-01")
    if not mtx.empty:
        mtx["net"] = mtx["long_open_interest_balance_volume"] - mtx["short_open_interest_balance_volume"]
        inst_net = mtx.groupby("date")["net"].sum()
        mfor = mtx[mtx["institutional_investors"].astype(str).str.contains("外資")].groupby("date")["net"].sum()
        mtxd = finmind.fetch("TaiwanFuturesDaily", "MTX", "2018-01-01")
        if not mtxd.empty:
            reg = mtxd[(mtxd["trading_session"] == "position") & (~mtxd["contract_date"].astype(str).str.contains("/"))]
            tot = reg.groupby("date")["open_interest"].sum()
            ratio = (-inst_net).reindex(tot.index) / tot.replace(0, np.nan)
            d["mtx_retail_ratio"] = by(ratio.to_dict())
            d["mtx_retail_chg5"] = d["mtx_retail_ratio"].diff(5)
        d["mtx_foreign_net_z"] = zscore(by(mfor.to_dict()), 60)
    # 選擇權 TXO：外資 / 自營 買權、賣權 淨部位
    txo = finmind.fetch("TaiwanOptionInstitutionalInvestors", "TXO", "2018-01-01")
    if not txo.empty:
        txo["net"] = txo["long_open_interest_balance_volume"] - txo["short_open_interest_balance_volume"]
        def opt(inv, cp):
            m = txo[txo["institutional_investors"].astype(str).str.contains(inv) & (txo["call_put"] == cp)]
            return by(m.groupby("date")["net"].sum().to_dict())
        fc, fp, dc, dp = opt("外資", "買權"), opt("外資", "賣權"), opt("自營", "買權"), opt("自營", "賣權")
        d["txo_f_call_z"], d["txo_f_put_z"] = zscore(fc, 60), zscore(fp, 60)
        d["txo_f_cp_diff_z"], d["txo_d_cp_diff_z"] = zscore(fc - fp, 60), zscore(dc - dp, 60)
    # 台指期日盤：近月期現價差、全部 OI 日變化、成交量
    txd = finmind.fetch("TaiwanFuturesDaily", "TX", "2010-01-01")
    if not txd.empty:
        txr = txd[(txd["trading_session"] == "position") & (~txd["contract_date"].astype(str).str.contains("/"))].sort_values(["date", "contract_date"])
        near = txr.groupby("date").first()
        d["tx_basis_pct"] = (by(near["close"].to_dict()) / c - 1) * 100
        d["tx_basis_chg1"] = d["tx_basis_pct"].diff()
        d["tx_oi_chg1_z"] = zscore(by(txr.groupby("date")["open_interest"].sum().to_dict()).diff(), 60)
        d["tx_vol_z"] = zscore(by(txr.groupby("date")["volume"].sum().to_dict()), 60)
    return d


def _night_hist() -> pd.DataFrame:
    try:
        return finmind.tx_night_history("2017-01-01")
    except Exception as e:  # noqa: BLE001
        log.warning("night history: %s", e)
        return pd.DataFrame()


# ------------------------------------------------------------------ 模型
class RidgeModel:
    """標準化 + 中位數補值 + Ridge；預測值除以訓練期預測標準差，方便與 LGB 平均。"""

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.linear_model import Ridge
        self.med = X.median().fillna(0.0)          # 整欄缺值 (早期無資料) 的中位數為 NaN → 補 0
        Xf = X.fillna(self.med).fillna(0.0)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1.0).fillna(1.0)
        Z = (Xf - self.mu) / self.sd
        self.m = Ridge(alpha=self.alpha).fit(Z.values, y.values)
        p = self.m.predict(Z.values)
        self.psd = float(np.std(p)) or 1.0
        self.coef = dict(zip(X.columns, self.m.coef_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = ((X.fillna(self.med).fillna(0.0) - self.mu) / self.sd).fillna(0.0)
        return self.m.predict(Z.values)

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X) / self.psd


class LgbModel:
    def __init__(self, params: dict | None = None):
        self.params = params or LGB_PARAMS

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.models = M.fit_ensemble(X, y, self.params)
        p = M.predict_ensemble(self.models, X)
        self.psd = float(np.std(p)) or 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return M.predict_ensemble(self.models, X)

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X) / self.psd


def _wf(d: pd.DataFrame, feats: list[str], target: str, h: int, first_year: int, make, min_train: int = 400) -> pd.DataFrame:
    d = d.dropna(subset=[target]).copy()
    d["year"] = d["date"].str[:4].astype(int)
    rows = []
    for y in sorted(d["year"].unique()):
        if y < first_year:
            continue
        test = d[d["year"] == y]
        train = d[d["year"] < y].iloc[:-h]
        if len(train) < min_train or test.empty:
            continue
        m = make().fit(train[feats], train[target])
        rows.append(pd.DataFrame({"date": test["date"].values, "year": y, "pred": m.predict_std(test[feats]), "actual": test[target].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ 命中率評估與叫牌門檻
def tier_stats(oos: pd.DataFrame) -> dict:
    """三檔叫牌：pred 前 30% 叫偏多、後 30% 叫偏空。回傳各檔 OOS 命中率、覆蓋率與是否啟用。"""
    if oos.empty:
        return {}
    lo, hi = float(oos["pred"].quantile(TIER)), float(oos["pred"].quantile(1 - TIER))
    up, dn, mid = oos[oos["pred"] >= hi], oos[oos["pred"] <= lo], oos[(oos["pred"] > lo) & (oos["pred"] < hi)]
    base_up = float((oos["actual"] > 0).mean())
    up_hit = float((up["actual"] > 0).mean()) if len(up) else np.nan
    dn_hit = float((dn["actual"] < 0).mean()) if len(dn) else np.nan
    up_on = bool(up_hit >= base_up + MIN_EDGE)
    dn_on = bool(dn_hit >= (1 - base_up) + MIN_EDGE)
    calls = (len(up) if up_on else 0) + (len(dn) if dn_on else 0)
    hits = (float((up["actual"] > 0).sum()) if up_on else 0) + (float((dn["actual"] < 0).sum()) if dn_on else 0)
    # 強叫牌 (前/後 15%)
    slo, shi = float(oos["pred"].quantile(TIER_STRONG)), float(oos["pred"].quantile(1 - TIER_STRONG))
    sup, sdn = oos[oos["pred"] >= shi], oos[oos["pred"] <= slo]
    return {"edge_lo": round(lo, 4), "edge_hi": round(hi, 4), "base_up": round(base_up, 3),
            "strong_lo": round(slo, 4), "strong_hi": round(shi, 4),
            "up_hit_strong": round(float((sup["actual"] > 0).mean()), 3) if len(sup) else None, "dn_hit_strong": round(float((sdn["actual"] < 0).mean()), 3) if len(sdn) else None,
            "up_hit": round(up_hit, 3), "up_n": int(len(up)), "up_on": up_on, "up_mean": round(float(up["actual"].mean()), 2) if len(up) else None,
            "dn_hit": round(dn_hit, 3), "dn_n": int(len(dn)), "dn_on": dn_on, "dn_mean": round(float(dn["actual"].mean()), 2) if len(dn) else None,
            "mid_up": round(float((mid["actual"] > 0).mean()), 3) if len(mid) else None,
            "call_hit": round(hits / calls, 3) if calls else None, "call_cov": round(calls / len(oos), 3),
            "by_year": _tier_by_year(oos, lo, hi, up_on, dn_on)}


def _tier_by_year(oos: pd.DataFrame, lo: float, hi: float, up_on: bool, dn_on: bool) -> list[dict]:
    out = []
    for y, g in oos.groupby("year"):
        up, dn = g[g["pred"] >= hi], g[g["pred"] <= lo]
        calls = (len(up) if up_on else 0) + (len(dn) if dn_on else 0)
        hits = (float((up["actual"] > 0).sum()) if up_on else 0) + (float((dn["actual"] < 0).sum()) if dn_on else 0)
        out.append({"year": int(y), "n": int(len(g)), "base_up": round(float((g["actual"] > 0).mean()), 3),
                    "call_hit": round(hits / calls, 3) if calls else None, "calls": int(calls)})
    return out


def _combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b[["date", "pred"]], on="date", suffixes=("", "_b"))
    m["pred"] = (m["pred"] + m["pred_b"]) / 2
    return m.drop(columns=["pred_b"])


def _report(oos: pd.DataFrame) -> dict:
    met = M.metrics(oos)
    ts = tier_stats(oos)
    return {"n": met.get("n"), "rank_ic": met.get("rank_ic"), "ic_positive_years": met.get("ic_positive_years"),
            "base_hit": met.get("base_hit"), "bin_hit": met.get("bin_hit"), "tiers": {k: v for k, v in ts.items() if k != "by_year"},
            "tier_by_year": ts.get("by_year"), "calibration": met.get("calibration")}


def _pick(cands: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    """依 OOS 叫牌命中率 (次序 IC) 選最佳。"""
    best, best_key, reps = None, None, {}
    for k, oos in cands.items():
        if oos.empty:
            continue
        rep = _report(oos)
        reps[k] = rep
        score = ((rep["tiers"].get("call_hit") or 0), rep.get("rank_ic") or 0)
        if best is None or score > best:
            best, best_key = score, k
    return best_key, reps


# ------------------------------------------------------------------ 訓練
def train(write: bool = True, verbose: bool = True) -> dict:
    scored = backtest.load_long()
    night = _night_hist()
    mat = build_matrix(scored, night)
    results = {}
    mn = mat.dropna(subset=[NIGHT_FEATURE])
    for h in HORIZONS:
        tgt = f"fwd{h}"
        results[h] = {}
        for variant in ("base", "night"):
            dd = mat if variant == "base" else mn
            first = FIRST_TEST_YEAR if variant == "base" else FIRST_TEST_YEAR_NIGHT
            # 特徵集 × 模型 全部跑 OOS，依叫牌命中率 (次序 IC) 選
            cands, featmap = {}, {}
            for sname, feats in FEATURE_SETS.items():
                f = feats + ([NIGHT_FEATURE] if variant == "night" else [])
                oos_l = _wf(dd, f, tgt, h, first, lambda: LgbModel())
                oos_r = _wf(dd, f, tgt, h, first, lambda: RidgeModel())
                cands[f"{sname}|lgb"], cands[f"{sname}|ridge"] = oos_l, oos_r
                cands[f"{sname}|ens"] = _combine(oos_l, oos_r) if not oos_l.empty and not oos_r.empty else pd.DataFrame()
                for mk in ("lgb", "ridge", "ens"):
                    featmap[f"{sname}|{mk}"] = f
            key, reps = _pick(cands)
            results[h][variant] = {"chosen": key, "reports": reps}
            if verbose and key:
                t = reps[key]["tiers"]
                print(f"h{h} {variant:<5} {key:<16} IC {reps[key]['rank_ic']} ({reps[key]['ic_positive_years']}) 基準 {t['base_up']} 叫牌命中 {t['call_hit']} 覆蓋 {t['call_cov']} "
                      f"偏多 {t['up_hit']}{'✓' if t['up_on'] else '✗'}(強 {t['up_hit_strong']}) 偏空 {t['dn_hit']}{'✓' if t['dn_on'] else '✗'}(強 {t['dn_hit_strong']})")
            if write and key:
                sname, mk = key.split("|")
                feats = featmap[key]
                d = dd.dropna(subset=[tgt])
                bundle = {"horizon": h, "variant": variant, "chosen": key, "features": feats, "trained_at": dt.datetime.now(config.TZ).isoformat(),
                          "train_end": str(d["date"].max()), "n_train": int(len(d)), "report": reps[key],
                          "lgb": LgbModel().fit(d[feats], d[tgt]) if mk in ("lgb", "ens") else None,
                          "ridge": RidgeModel().fit(d[feats], d[tgt]) if mk in ("ridge", "ens") else None}
                M.save(f"st_h{h}_{variant}", bundle)
    if write:
        M.save_json("short_term_metrics", {str(h): {v: {"chosen": results[h][v]["chosen"], **{kk: vv for kk, vv in (results[h][v]["reports"].get(results[h][v]["chosen"]) or {}).items() if kk != "calibration"}}
                                                    for v in ("base", "night")} for h in HORIZONS})
    return results


# ------------------------------------------------------------------ 預測
def _predict_bundle(b: dict, row: pd.DataFrame) -> tuple[float, dict]:
    X = row[b["features"]].astype(float)
    preds, drivers = [], {}
    if b.get("lgb") is not None:
        preds.append(float(b["lgb"].predict_std(X)[0]))
        drivers = M.explain(b["lgb"].models, X, NAMES)
    if b.get("ridge") is not None:
        preds.append(float(b["ridge"].predict_std(X)[0]))
        if not drivers:
            Z = ((X.fillna(b["ridge"].med).fillna(0.0) - b["ridge"].mu) / b["ridge"].sd).fillna(0.0).iloc[0]
            contrib = sorted(((f, float(Z[f] * b["ridge"].coef[f])) for f in b["features"]), key=lambda x: x[1])
            drivers = {"negative": [{"feature": f, "name": NAMES.get(f, f), "value": float(X.iloc[0][f]) if pd.notna(X.iloc[0][f]) else None, "contrib": c} for f, c in contrib if c < 0][:6],
                       "positive": [{"feature": f, "name": NAMES.get(f, f), "value": float(X.iloc[0][f]) if pd.notna(X.iloc[0][f]) else None, "contrib": c} for f, c in reversed(contrib) if c > 0][:6], "bias": 0.0}
    return float(np.mean(preds)), drivers


def forecast(scored: pd.DataFrame, snapshot: dict | None = None) -> dict:
    """回傳 {h: {pred, p_up, hist_mean, q20, q80, call, call_hit, base_hit, variant, drivers}}；模型未訓練回 {}。"""
    snap = snapshot or {}
    tn = snap.get("tx_night") or {}
    use_night = tn.get("change_pct") is not None and snap.get("phase") in ("night", "closed", "pre")
    mat = build_matrix(scored.tail(400).reset_index(drop=True), None)   # 只需最後一列；400 列足夠算 60 日 z
    row = mat.iloc[[-1]].copy()
    if use_night:
        row[NIGHT_FEATURE] = float(tn["change_pct"])
    out = {}
    for h in HORIZONS:
        b = M.load(f"st_h{h}_night") if use_night else None
        variant = "night" if b else "base"
        b = b or M.load(f"st_h{h}_base")
        if not b:
            continue
        pred, drivers = _predict_bundle(b, row)
        rep = b["report"]
        cal = M.apply_calibration(rep["calibration"], pred)
        t = rep["tiers"]
        call, call_hit, strength = "中性", t.get("mid_up"), ""
        if t.get("up_on") and pred >= t["edge_hi"]:
            call, call_hit = "偏多", t["up_hit"]
            if t.get("strong_hi") is not None and pred >= t["strong_hi"] and (t.get("up_hit_strong") or 0) >= t["up_hit"]:
                strength, call_hit = "強", t["up_hit_strong"]
        elif t.get("dn_on") and pred <= t["edge_lo"]:
            call, call_hit = "偏空", t["dn_hit"]
            if t.get("strong_lo") is not None and pred <= t["strong_lo"] and (t.get("dn_hit_strong") or 0) >= t["dn_hit"]:
                strength, call_hit = "強", t["dn_hit_strong"]
        caveat = _honesty_note(h, variant, snap.get("phase"))
        out[h] = {"pred_std": round(pred, 3), "p_up": cal["p_up"], "hist_mean": cal["hist_mean"], "q20": cal["q20"], "q80": cal["q80"], "bin": cal["bin"],
                  "base_hit": cal["base_hit"], "call": call, "call_strength": strength, "call_hit": round(call_hit, 3) if call_hit is not None else None,
                  "tier_up_hit": t.get("up_hit"), "tier_dn_hit": t.get("dn_hit"), "call_cov": t.get("call_cov"),
                  "variant": variant, "model": b["chosen"], "rank_ic": rep.get("rank_ic"), "drivers": drivers,
                  "caveat": caveat,
                  "note": f"短線模型 v2 ({variant == 'night' and '含夜盤' or '不含夜盤'}‧{b['chosen']}‧OOS IC {rep.get('rank_ic')}‧叫牌命中 {t.get('call_hit')} 覆蓋 {t.get('call_cov')})"
                          + (f"‧{caveat}" if caveat else "")}
    return out


def _honesty_note(h: int, variant: str, phase: str | None = None) -> str:
    """叫牌可信度的誠實提醒 (v2.2 驗證者要求)：
    - 1 日不含夜盤：OOS 命中僅約 56% (滾動門檻 0.563；基準 55%)，避免使用者高估盤後 1 日叫牌把握。
    - 1 日含夜盤：81% 為收盤到收盤，主要來自隔夜跳空 (夜盤 05:00 收盤後才可知)；隔日開盤進場的方向命中約 67%。
    - 盤中 (phase=open) 以未收盤 K 棒與韓股盤中值計算 → 暫定。
    """
    parts: list[str] = []
    if h == 1 and variant == "base":
        parts.append("OOS 命中約 56% (基準 55%)")
    elif h == 1 and variant == "night":
        parts.append("81% 為收盤到收盤,主要來自隔夜跳空;開盤進場約 67%")
    elif variant == "night":
        parts.append("命中為收盤到收盤,含隔夜跳空")
    if phase == "open":
        parts.append("盤中以未收盤 K 棒計算,屬暫定")
    return "；".join(parts)


def load_metrics() -> dict | None:
    return M.load_json("short_term_metrics")
