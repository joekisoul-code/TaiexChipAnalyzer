"""當日小時級預測：以近 2 年的加權指數小時 K (Yahoo ^TWII) + 前一日籌碼特徵 + 前晚夜盤台指期，
預測「從目前時間點到 10:00 / 11:00 / 12:00 / 13:00 / 13:30 收盤」的報酬，逐月 walk-forward 驗證後做分位校準。

時間點 (mark)：pre(開盤前，以前收為基準) → 09:00(開盤價) → 10:00 → 11:00 → 12:00 → 13:00。
盤中即時：用 TWSE 1 分鐘分時把現價套到「最近已過的時間點」特徵 (ret_1h 以現價對一小時前價計算)。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from ..analysis import backtest
from ..sources import finmind, twse, yahoo
from . import model as M
from .features import market_matrix

log = logging.getLogger(__name__)
# 注意：加權指數 09:00:05 的第一筆「開盤價」是用大量尚未成交股票的前收計算的失真值，開盤後幾分鐘才反映真實跳空；
# 因此不設 09:00 時間點 (會讓模型「預測」到失真開盤的修正而高估準確度)。10:00 以後的時間點才是可信的盤中預測。
MARKS = ["pre", "10:00", "11:00", "12:00", "13:00"]
TARGET_MARKS = ["10:00", "11:00", "12:00", "13:00", "13:30"]
MARK_IDX = {m: i for i, m in enumerate(MARKS)}
FEATURES = ["mark_idx", "gap_pct", "ret_open", "ret_prev", "hi_pos", "range_pct", "ret_1h", "night_chg_pct",
            "prev_ret1", "prev_ret5", "prev_bias20", "prev_vola20", "prev_composite_smooth", "prev_fut_foreign_pct", "prev_foreign_z5",
            "prev_state_bull", "prev_state_bear", "dow", "days_to_settle",
            "g_sox_r1", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1", "g_vix_level", "g_vix_chg", "g_kospi_r1"]
GLOBAL_COLS = ["g_sox_r1", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1", "g_vix_level", "g_vix_chg", "g_kospi_r1"]
NAMES = {"mark_idx": "時間點", "gap_pct": "開盤跳空%", "ret_open": "開盤至今%", "ret_prev": "前收至今%", "hi_pos": "位於當日區間位置",
         "range_pct": "當日振幅%", "ret_1h": "近 1 小時%", "night_chg_pct": "前晚夜盤台指期%",
         "prev_ret1": "前一日漲跌%", "prev_ret5": "前 5 日漲跌%", "prev_bias20": "前日月線乖離", "prev_vola20": "20 日波動率",
         "prev_composite_smooth": "籌碼綜合分", "prev_fut_foreign_pct": "外資期貨百分位", "prev_foreign_z5": "外資 5 日 z",
         "prev_state_bull": "多頭狀態", "prev_state_bear": "空頭狀態", "dow": "星期", "days_to_settle": "距結算日",
         "g_sox_r1": "前晚費半%", "g_sp500_r1": "前晚 S&P500%", "g_nasdaq_r1": "前晚 Nasdaq%", "g_tsm_adr_r1": "前晚台積電 ADR%",
         "g_vix_level": "VIX 水準", "g_vix_chg": "VIX 變化%", "g_kospi_r1": "KOSPI 前日%"}
PARAMS = dict(n_estimators=120, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=20.0, verbose=-1)


# ------------------------------------------------------------------ 特徵
def mark_rows(date: str, rec: dict, daily: dict | None, night_chg: float | None, live_mark: str | None = None) -> list[dict]:
    """rec: {'prev_close','open','10:00',...,'13:30','hi':{},'lo':{}}。訓練時 live_mark=None (所有 mark 皆產生，含目標)；
    即時時只產生 live_mark 一列 (目標 NaN)。"""
    prev_close, open_px = rec.get("prev_close"), rec.get("open")
    if not prev_close:
        return []
    rows = []
    marks = [live_mark] if live_mark else MARKS
    for mark in marks:
        if mark == "pre":
            px = prev_close
            feat = {"gap_pct": np.nan, "ret_open": np.nan, "ret_prev": 0.0, "hi_pos": np.nan, "range_pct": np.nan, "ret_1h": np.nan}
        else:
            px = rec.get(mark)
            if px is None or open_px is None:
                continue
            hi, lo = rec["hi"].get(mark, px), rec["lo"].get(mark, px)
            prev_mark = MARKS[MARK_IDX[mark] - 1]
            p1h = open_px if prev_mark == "pre" else rec.get(prev_mark)
            feat = {"gap_pct": (open_px / prev_close - 1) * 100, "ret_open": (px / open_px - 1) * 100, "ret_prev": (px / prev_close - 1) * 100,
                    "hi_pos": (px - lo) / (hi - lo) if hi > lo else 0.5, "range_pct": (hi / lo - 1) * 100,
                    "ret_1h": (px / p1h - 1) * 100 if p1h else np.nan}
        row = {"date": date, "mark": mark, "mark_idx": MARK_IDX[mark], "price": px, "prev_close": prev_close, "night_chg_pct": night_chg, **feat}
        if daily:
            row.update(daily)
        cur_t = "00:00" if mark == "pre" else mark
        for tm in TARGET_MARKS:
            tp = rec.get(tm)
            row[f"to_{tm}"] = (tp / px - 1) * 100 if (tp and tm > cur_t and not live_mark) else np.nan
        rows.append(row)
    return rows


def _daily_features(mat: pd.DataFrame) -> dict[str, dict]:
    """以「前一交易日」的日資料作為當日特徵。{date: {...}}"""
    cols = {"ret1": "prev_ret1", "ret5": "prev_ret5", "bias20": "prev_bias20", "vola20": "prev_vola20",
            "composite_smooth": "prev_composite_smooth", "fut_foreign_pct": "prev_fut_foreign_pct", "foreign_z5": "prev_foreign_z5",
            "state_bull": "prev_state_bull", "state_bear": "prev_state_bear"}
    out = {}
    dates = list(mat["date"])
    for i in range(1, len(dates)):
        prev, cur = mat.iloc[i - 1], mat.iloc[i]
        out[dates[i]] = {dst: float(prev[src]) if pd.notna(prev[src]) else np.nan for src, dst in cols.items()}
        out[dates[i]]["dow"] = pd.Timestamp(dates[i]).weekday()
        out[dates[i]]["days_to_settle"] = int(cur["days_to_settle"])
        for g in GLOBAL_COLS:   # 當日開盤前已知的國際盤 (aligned_features 已對齊到 D)
            out[dates[i]][g] = float(cur[g]) if g in cur and pd.notna(cur[g]) else np.nan
    return out


def _daily_for_next(mat: pd.DataFrame, day: str) -> dict:
    """最新一列作為「day」的前一日特徵。"""
    prev = mat.iloc[-1]
    out = {"prev_ret1": prev["ret1"], "prev_ret5": prev["ret5"], "prev_bias20": prev["bias20"], "prev_vola20": prev["vola20"],
           "prev_composite_smooth": prev["composite_smooth"], "prev_fut_foreign_pct": prev["fut_foreign_pct"], "prev_foreign_z5": prev["foreign_z5"],
           "prev_state_bull": prev["state_bull"], "prev_state_bear": prev["state_bear"], "dow": pd.Timestamp(day).weekday(),
           "days_to_settle": int(prev["days_to_settle"])}
    try:
        from ..sources import global_markets as gm
        latest = gm.latest_features()
        for g in GLOBAL_COLS:
            out[g] = latest.get(g, np.nan)
    except Exception:  # noqa: BLE001
        for g in GLOBAL_COLS:
            out[g] = np.nan
    return out


def build_history() -> pd.DataFrame:
    mat = market_matrix(backtest.load_long("2010-01-01"))
    daily = _daily_features(mat)
    night = finmind.tx_night_history("2023-01-01")
    night_map = dict(zip(night["date"], night["night_chg_pct"])) if not night.empty else {}
    marks = yahoo.daily_marks()
    rows = []
    for d, rec in marks.items():
        if d not in daily:
            continue
        rows += mark_rows(d, rec, daily[d], night_map.get(d))   # FinMind 夜盤 date = 該夜盤準備的交易日
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 訓練 / 驗證
def _walk_forward_months(df: pd.DataFrame, target: str, min_train_days: int = 120) -> pd.DataFrame:
    d = df.dropna(subset=[target]).copy()
    d["ym"] = d["date"].str[:7]
    rows = []
    for ym in sorted(d["ym"].unique()):
        train, test = d[d["ym"] < ym], d[d["ym"] == ym]
        if train["date"].nunique() < min_train_days or test.empty:
            continue
        models = M.fit_ensemble(train[FEATURES], train[target], PARAMS)
        rows.append(pd.DataFrame({"date": test["date"].values, "year": test["ym"].values, "mark": test["mark"].values,
                                  "pred": M.predict_ensemble(models, test[FEATURES]), "actual": test[target].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def train(write: bool = True) -> dict:
    hist = build_history()
    results = {}
    bundle = {"features": FEATURES, "trained_at": dt.datetime.now(config.TZ).isoformat(), "days": int(hist["date"].nunique()),
              "train_end": str(hist["date"].max()), "models": {}, "calibration": {}, "metrics": {}}
    for tm in TARGET_MARKS:
        target = f"to_{tm}"
        oos = _walk_forward_months(hist, target)
        met = M.metrics(oos) if not oos.empty else {}
        if met:
            met["hit_by_mark"] = {m: round(float(((g["pred"] > 0) == (g["actual"] > 0)).mean() * 100), 1) for m, g in oos.groupby("mark")}
            met["base_by_mark"] = {m: round(float((g["actual"] > 0).mean() * 100), 1) for m, g in oos.groupby("mark")}
            met["ic_by_mark"] = {m: round(float(g["pred"].rank().corr(g["actual"].rank())), 3) for m, g in oos.groupby("mark")}
        results[tm] = met
        d = hist.dropna(subset=[target])
        bundle["models"][tm] = M.fit_ensemble(d[FEATURES], d[target], PARAMS)
        # 校準分開做：開盤前 (含跳空，可預測性高) 與盤中 (10:00 後，可預測性低) 的預測值分布完全不同
        cal = {}
        if not oos.empty:
            for grp, sub in (("pre", oos[oos["mark"] == "pre"]), ("intra", oos[oos["mark"] != "pre"])):
                if len(sub) >= 50:
                    cal[grp] = M.calibrate(sub)
        bundle["calibration"][tm] = cal or None
        bundle["metrics"][tm] = {k: v for k, v in met.items() if k != "calibration"}
    if write:
        M.save("intraday", bundle)
        M.save_json("intraday_metrics", bundle["metrics"])
    return results


# ------------------------------------------------------------------ 即時預測
def _live_rec(snapshot: dict) -> tuple[dict, str] | None:
    """由即時 1 分鐘分時建立 rec (各時間點價、累積高低) 與目前 mark。"""
    bars = (snapshot.get("intraday") or {}).get("bars") or []
    idx = snapshot.get("taiex") or {}
    if not bars or not idx.get("prev"):
        return None
    now_t = dt.datetime.now(config.TZ).strftime("%H:%M")
    bars = [b for b in bars if b.get("close") and "09:00" <= b["time"] <= min(now_t, "13:30")]
    if not bars:
        return None
    px_at = lambda t: next((b["close"] for b in reversed(bars) if b["time"] <= t), None)  # noqa: E731
    # 開盤價：取 09:05 (避開 09:00:05 失真的第一筆)，不足則取第一筆
    open_px = px_at("09:05") or bars[0]["close"]
    rec = {"prev_close": idx["prev"], "open": open_px, "hi": {}, "lo": {}}
    cur = "10:00"                      # 10:00 之前也套用 10:00 模型 (以現價視為 10:00 時間點，屬近似)
    for m in MARKS[2:]:
        if now_t >= m:
            cur = m
    last = idx.get("last") or bars[-1]["close"]
    closes = [b["close"] for b in bars]
    rec[cur] = last
    rec["hi"][cur], rec["lo"][cur] = max(closes + [last]), min(closes + [last])
    # ret_1h：一小時前的實際價 (不足一小時則用開盤價)
    h, mi = int(now_t[:2]), int(now_t[3:])
    p1h = px_at(f"{h - 1:02d}:{mi:02d}") if h - 1 >= 9 else None
    prev_mark = MARKS[MARK_IDX[cur] - 1]
    if prev_mark != "pre":
        rec[prev_mark] = p1h or open_px
    else:
        rec["open"] = p1h or open_px    # 10:00 的 ret_1h 以 open 計
    return rec, cur


def forecast(scored: pd.DataFrame, snapshot: dict | None) -> dict:
    bundle = M.load("intraday")
    if not bundle:
        return {"error": "尚未訓練小時模型，請執行 python cli.py train --intraday"}
    now = dt.datetime.now(config.TZ)
    phase = (snapshot or {}).get("phase", "closed")
    mat = market_matrix(scored)
    last_date = str(mat["date"].iloc[-1])
    live = phase == "open" and snapshot is not None and not snapshot.get("same_day", False)
    rec_mark = _live_rec(snapshot) if live else None
    if rec_mark:
        rec, mark = rec_mark
        day = now.date().isoformat()
    else:
        live = False
        day = twse.next_trading_days(last_date, 1)[0]
        rec, mark = {"prev_close": float(mat["close"].iloc[-1]), "open": None, "hi": {}, "lo": {}}, "pre"
    from .short_term import night_final as _nf
    night = ((snapshot or {}).get("tx_night") or {}).get("change_pct") if _nf(snapshot) else None   # 2026-09-24：夜盤收盤後才用
    rows = mark_rows(day, rec, _daily_for_next(mat, day), night, live_mark=mark)
    if not rows:
        return {"error": "無法建立當前時間點特徵"}
    row = rows[0]
    x = pd.DataFrame([row])[FEATURES].astype(float)
    px = float(row["price"])
    out = {"day": day, "mark": mark, "price": px, "prev_close": float(row["prev_close"]), "live": live, "night_chg_pct": night,
           "basis_date": last_date, "trained_at": bundle["trained_at"], "days": bundle["days"], "targets": {}}
    now_t = now.strftime("%H:%M")
    cur_t = "00:00" if mark == "pre" else (now_t if live else mark)
    out["note"] = ("10:00 前以現價套用 10:00 時間點模型 (近似)" if live and now_t < "10:00" else
                   "開盤前預測含跳空：主要來自夜盤台指期，可預測性高但多屬『開盤已反映』的部分" if mark == "pre" else "")
    for tm in TARGET_MARKS:
        if tm <= cur_t:
            continue
        cal_all = bundle["calibration"].get(tm) or {}
        cal = cal_all.get("pre" if mark == "pre" else "intra")
        pred = float(M.predict_ensemble(bundle["models"][tm], x)[0])
        c = M.apply_calibration(cal, pred) if cal else {}
        mean_ = c.get("hist_mean") if c.get("hist_mean") is not None else pred
        out["targets"][tm] = {"pred": round(pred, 2), **c,
                              "level": round(px * (1 + mean_ / 100), 0),
                              "level_lo": round(px * (1 + (c.get("q20") if c.get("q20") is not None else pred) / 100), 0),
                              "level_hi": round(px * (1 + (c.get("q80") if c.get("q80") is not None else pred) / 100), 0),
                              "drivers": M.explain(bundle["models"][tm], x, NAMES, top=4), "metrics": bundle["metrics"].get(tm, {})}
    return out
