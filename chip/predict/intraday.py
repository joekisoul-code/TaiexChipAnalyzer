"""當日小時級預測：加權指數盤中時間點 (2026-10-12 起：TWSE 每 5 秒指數 2016~，開盤價取 09:05；舊來源 Yahoo ^TWII 60m 約 2 年作為對照)
+ 前一日籌碼特徵 + 前晚夜盤台指期，
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
# 早盤模型 (2026-10-12)：09:15 / 09:30 另外訓練 (和 10:00 後混訓會稀釋後段；研究：2018~ 逐年 walk-forward IC 0.06~0.08、9 年 7~8 年為正，訊號弱)
EARLY_MARKS = ["pre", "09:15", "09:30"]
TARGET_MARKS = ["10:00", "11:00", "12:00", "13:00", "13:30"]
INTRA_MARKS = MARKS[1:]   # 盤中時間點 (去 pre)：研究用「只用盤中列訓練」(_walk_forward_months train_marks)；正式 train() 仍 pre + 盤中混訓
MARK_IDX = {m: i for i, m in enumerate(MARKS)}
WF_FROM = "2018-01"   # TWSE 來源的 walk-forward 起始月 (2016~2017 只當訓練；控制每週重訓時間)
FEATURES = ["mark_idx", "gap_pct", "ret_open", "ret_prev", "hi_pos", "range_pct", "ret_1h", "night_chg_pct",
            "prev_ret1", "prev_ret5", "prev_bias20", "prev_vola20", "prev_composite_smooth", "prev_fut_foreign_pct", "prev_foreign_z5",
            "prev_state_bull", "prev_state_bear", "dow", "days_to_settle",
            "g_sox_r1", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1", "g_vix_level", "g_vix_chg", "g_kospi_r1"]
GLOBAL_COLS = ["g_sox_r1", "g_sp500_r1", "g_nasdaq_r1", "g_tsm_adr_r1", "g_vix_level", "g_vix_chg", "g_kospi_r1"]
# 研究用可選特徵 (2026-10-14，未上線)：前一交易日的隔夜跳空 prev_og (前日 open / 前前日收 −1) 與前一日開盤→13:30 prev_oc。
# 跳空研究 (ridge/logit) 顯示已知當日跳空下再加 og_D 對盤中段 ΔIC +0.02~0.04；需以正式 LightGBM 驗證 (tools/intraday_research.py) 才改 FEATURES。
OG_FEATS = ["prev_og", "prev_oc"]
FEATURES_OG = FEATURES + OG_FEATS
NAMES = {"mark_idx": "時間點", "gap_pct": "開盤跳空%", "ret_open": "開盤至今%", "ret_prev": "前收至今%", "hi_pos": "位於當日區間位置",
         "range_pct": "當日振幅%", "ret_1h": "近 1 小時%", "night_chg_pct": "前晚夜盤台指期%",
         "prev_ret1": "前一日漲跌%", "prev_ret5": "前 5 日漲跌%", "prev_bias20": "前日月線乖離", "prev_vola20": "20 日波動率",
         "prev_composite_smooth": "籌碼綜合分", "prev_fut_foreign_pct": "外資期貨百分位", "prev_foreign_z5": "外資 5 日 z",
         "prev_state_bull": "多頭狀態", "prev_state_bear": "空頭狀態", "dow": "星期", "days_to_settle": "距結算日",
         "g_sox_r1": "前晚費半%", "g_sp500_r1": "前晚 S&P500%", "g_nasdaq_r1": "前晚 Nasdaq%", "g_tsm_adr_r1": "前晚台積電 ADR%",
         "g_vix_level": "VIX 水準", "g_vix_chg": "VIX 變化%", "g_kospi_r1": "KOSPI 前日%",
         "prev_og": "前一日隔夜跳空%", "prev_oc": "前一日開盤→收盤%"}
PARAMS = dict(n_estimators=120, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=20.0, verbose=-1)
# 2026-11-02 延續模型 (研究 hit2 A + 反方獨立重算)：10:00~13:00「現在 → 13:30 收盤」改零參數延續分數
# z = 開盤 (09:05) 至今% ÷ 前一日 σ (range_levels.sigma_series)；各時間點自己的 5 分位表 (2018~，每週重訓) 給上漲率 / 平均 / 兩成~八成 (y/σ，發布時 × σ)，叫牌規則不變。
# 走動式 2019-01~2026-10 (每月只用之前的列建表)：rank IC .10~.15 (LightGBM .04~.08；11:00~13:00 逐年 9/9、9/9、8/9 年較好)；
# 叫牌命中 52.6% → 57.5% (+4.8 [2.3, 7.4])，叫牌比例約 59% → 57%；20~80% 區間覆蓋 40/50/59/73% → 60/61/60/62%。
# 2024~26 淨正確叫牌數持平 (−0.4 [−5.5, 4.6])：改善是命中率與區間誠實，不是叫更多。09:15 / 09:30 早盤、開盤前、其他目標仍用 LightGBM。
CONT_MARKS = ["10:00", "11:00", "12:00", "13:00"]
CONT_TARGET = "13:30"
CONT_FROM = "2018-01-01"
CONT_WF_FROM = "2019-01"
CONT_MIN = 250
# 11-04 (研究 hit3 F + 反方獨立重算) 短目標 (現在 → 11:00 / 12:00 / 13:00) 也用延續表 (同 z、各 (時點, 目標) 一張表)：
# 20~80% 區間 6 格覆蓋 56~61% (LightGBM 39~68%)、interval score 6/6 格較好；叫牌只有 10:00 / 11:00 → 13:00 兩格經 Bonferroni 仍有效
# (edge +5.3 / +5.4，8/8、6/8 年)，其他格不給上漲率 (10→12、11→12 不顯著；10→11、12→13 無力)；離目標 < 30 分鐘也不給 (1 分鐘實盤近目標接近擲硬幣)
CONT_SHORT = {"11:00": ["10:00"], "12:00": ["10:00", "11:00"], "13:00": ["10:00", "11:00", "12:00"]}
CONT_SHORT_CALL = {("10:00", "13:00"), ("11:00", "13:00")}
CONT_SHORT_MUTE_MIN = 30   # 保險：目前兩個叫牌格 (10:00/11:00 → 13:00) 在時點用表期間離目標都 ≥ 61 分，實際不會觸發


# ------------------------------------------------------------------ 特徵
def mark_rows(date: str, rec: dict, daily: dict | None, night_chg: float | None, live_mark: str | None = None,
              marks_list: list[str] | None = None) -> list[dict]:
    """rec: {'prev_close','open','10:00',...,'13:30','hi':{},'lo':{}}。訓練時 live_mark=None (所有 mark 皆產生，含目標)；
    即時時只產生 live_mark 一列 (目標 NaN)。"""
    prev_close, open_px = rec.get("prev_close"), rec.get("open")
    if not prev_close:
        return []
    rows = []
    ML = marks_list or MARKS
    MI = {m: i for i, m in enumerate(ML)}
    marks = [live_mark] if live_mark else ML
    for mark in marks:
        if mark == "pre":
            px = prev_close
            feat = {"gap_pct": np.nan, "ret_open": np.nan, "ret_prev": 0.0, "hi_pos": np.nan, "range_pct": np.nan, "ret_1h": np.nan}
        else:
            px = rec.get(mark)
            if px is None or open_px is None:
                continue
            hi, lo = rec["hi"].get(mark, px), rec["lo"].get(mark, px)
            prev_mark = ML[MI[mark] - 1]
            p1h = open_px if prev_mark == "pre" else rec.get(prev_mark)
            feat = {"gap_pct": (open_px / prev_close - 1) * 100, "ret_open": (px / open_px - 1) * 100, "ret_prev": (px / prev_close - 1) * 100,
                    "hi_pos": (px - lo) / (hi - lo) if hi > lo else 0.5, "range_pct": (hi / lo - 1) * 100,
                    "ret_1h": (px / p1h - 1) * 100 if p1h else np.nan}
        row = {"date": date, "mark": mark, "mark_idx": MI[mark], "price": px, "prev_close": prev_close, "night_chg_pct": night_chg, **feat}
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


def _marks_source(source: str) -> dict[str, dict]:
    if source == "twse":
        return twse.intraday_marks()
    return yahoo.daily_marks()


def og_features(marks: dict) -> dict[str, dict]:
    """由 marks dict 算每日的「前一交易日」隔夜跳空 / 日內特徵 (研究用，見 OG_FEATS)：{date: {prev_og, prev_oc}}。
    前一交易日 = marks 中前一個 (日期排序) 有資料的日子；第一天或前一日缺 open / prev_close / 13:30 → NaN。
    prev_og = 前日 open / 前日 prev_close −1 ×100 (開盤定義與 marks 來源相同：TWSE 09:05)；prev_oc = 前日 13:30 / 前日 open −1 ×100。"""
    out = {}
    prev = None
    for d in sorted(marks):
        rec = marks[d]
        og = oc = np.nan
        if prev:
            o, pc, c = prev.get("open"), prev.get("prev_close"), prev.get("13:30")
            if o and pc:
                og = (o / pc - 1) * 100
            if o and c:
                oc = (c / o - 1) * 100
        out[d] = {"prev_og": og, "prev_oc": oc}
        prev = rec
    return out


def build_history(source: str = "twse", marks_list: list[str] | None = None, marks: dict | None = None, _ctx: dict | None = None,
                  extra_feats: tuple[str, ...] = ()) -> pd.DataFrame:
    """source: twse (2016~，開盤 09:05) / yahoo (舊，約 2 年)。_ctx 可傳入已算好的 daily / night_map 省時。
    extra_feats: 研究用可選欄位 (OG_FEATS 的子集，見 og_features)；預設空 → 輸出欄位與正式路徑完全相同。"""
    if _ctx is None:
        mat = market_matrix(backtest.load_long("2010-01-01"))
        night = finmind.tx_night_history("2023-01-01")
        _ctx = {"daily": _daily_features(mat), "night_map": dict(zip(night["date"], night["night_chg_pct"])) if not night.empty else {}}
    daily, night_map = _ctx["daily"], _ctx["night_map"]
    marks = marks if marks is not None else _marks_source(source)
    extra = [f for f in extra_feats if f in OG_FEATS]
    og = og_features(marks) if extra else {}
    rows = []
    for d, rec in marks.items():
        if d not in daily:
            continue
        day_rows = mark_rows(d, rec, daily[d], night_map.get(d), marks_list=marks_list)   # FinMind 夜盤 date = 該夜盤準備的交易日
        if extra:
            add = {f: og.get(d, {}).get(f, np.nan) for f in extra}
            for r in day_rows:
                r.update(add)
        rows += day_rows
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 訓練 / 驗證
def _walk_forward_months(df: pd.DataFrame, target: str, min_train_days: int = 120, from_ym: str | None = None,
                         quarterly: bool = False, features: list[str] | None = None,
                         train_marks: list[str] | None = None, score_feature: str | None = None) -> pd.DataFrame:
    """逐月 (或逐季：quarterly=True，每季重訓一次，測整季) walk-forward。year 欄 = 測試月份 (metrics 依此分組)。
    features: 研究用特徵清單 (例如 FEATURES_OG)；預設正式 FEATURES。
    train_marks (研究用，2026-10-14)：只保留這些 mark 的列 (訓練與測試列同一組；例如 INTRA_MARKS = 盤中列不含 pre，模型只服務這些時間點)。
      預設 None = 全部列 (正式 pre + 盤中混訓形式)。
    score_feature (研究用零參數基準)：給定時不訓練，直接以該欄位值當預測分數 (例如 ret_open = 開盤至今%，正號 = 延續)；
      切分、列與訓練版完全相同 (可同日配對)，該欄缺值的列剔除 (pre 列的 ret_open 為 NaN)。"""
    feats = features or FEATURES
    d = df.dropna(subset=[target] + ([score_feature] if score_feature else [])).copy()
    if train_marks is not None:
        d = d[d["mark"].isin(train_marks)]
    d["ym"] = d["date"].str[:7]
    d["blk"] = d["ym"].map(lambda ym: f"{ym[:4]}Q{(int(ym[5:7]) - 1) // 3 + 1}") if quarterly else d["ym"]
    rows = []
    for blk in sorted(d.loc[(d["ym"] >= from_ym) if from_ym else d.index == d.index, "blk"].unique()):
        test = d[(d["blk"] == blk) & ((d["ym"] >= from_ym) if from_ym else True)]
        train = d[d["date"] < test["date"].min()]
        if train["date"].nunique() < min_train_days or test.empty:
            continue
        if score_feature:
            pred = test[score_feature].to_numpy(dtype=float)
        else:
            models = M.fit_ensemble(train[feats], train[target], PARAMS)
            pred = M.predict_ensemble(models, test[feats])
        rows.append(pd.DataFrame({"date": test["date"].values, "year": test["ym"].values, "mark": test["mark"].values,
                                  "pred": pred, "actual": test[target].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _intra_ic(oos: pd.DataFrame, months: set | None = None) -> float | None:
    """盤中 (非 pre) 各 mark 的 rank IC 平均；months 限定比較的月份。"""
    if oos is None or oos.empty:
        return None
    o = oos[oos["mark"] != "pre"]
    if months is not None:
        o = o[o["year"].isin(months)]
    ics = [g["pred"].rank().corr(g["actual"].rank()) for _, g in o.groupby("mark") if len(g) >= 30]
    return float(np.mean(ics)) if ics else None


def _metrics(oos: pd.DataFrame) -> dict:
    met = M.metrics(oos) if not oos.empty else {}
    if met:
        met["hit_by_mark"] = {m: round(float(((g["pred"] > 0) == (g["actual"] > 0)).mean() * 100), 1) for m, g in oos.groupby("mark")}
        met["base_by_mark"] = {m: round(float((g["actual"] > 0).mean() * 100), 1) for m, g in oos.groupby("mark")}
        met["ic_by_mark"] = {m: round(float(g["pred"].rank().corr(g["actual"].rank())), 3) for m, g in oos.groupby("mark")}
    return met


def _cont_table(z: np.ndarray, yz: np.ndarray) -> dict:
    """延續模型一個時間點的表：z 的 5 分位切點 (M.calibrate 同法) + 每區 y/σ 的上漲率 / 平均 / 兩成 / 八成。"""
    edges = pd.Series(z).quantile(np.linspace(0, 1, M.N_BINS + 1)[1:-1]).to_numpy(dtype=float)
    b = np.digitize(z, edges); up = yz > 0
    bins = [{"p_up": round(float(up[b == k].mean()), 3), "mean_z": float(yz[b == k].mean()), "q20_z": float(np.quantile(yz[b == k], 0.2)),
             "q80_z": float(np.quantile(yz[b == k], 0.8)), "n": int((b == k).sum())} for k in range(M.N_BINS)]
    # 11-04 v_z = 該時點全部列 y/σ 的穩健變異 ((q80−q20)/1.6832)²：整點之間依時間在兩時點間線性內插 → 區間縮放 (_cont_k)
    v_z = float(((np.quantile(yz, 0.8) - np.quantile(yz, 0.2)) / 1.6832) ** 2)
    return {"z_edges": edges.tolist(), "bins": bins, "base_hit": round(float(up.mean()), 3), "base_mean_z": float(yz.mean()), "n": int(len(z)), "v_z": v_z}


def _hm_min(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:5])


def _cont_k(marks: dict, mark: str, now_t: str) -> float:
    """11-04 13:30 目標整點之間的區間縮放 k = √(V(t)/V(mark))，V 在 mark → 下一時點之間依時間線性內插；13:00 之後 (收盤集合競價，變異不再縮) 與舊表 → 1。
    歷史下一時點測試 (2019~2026)：覆蓋 72.4% → 61.1%、interval score 較好 (7/7 年)；46 天 1 分鐘實盤 67.7% → 62.9%。
    反方：效果與「整點後依時間縮到約 0.7 倍」的常數版相當 (差異不顯著)，不是新發現的結構。"""
    if mark not in CONT_MARKS or mark == CONT_MARKS[-1] or now_t <= mark:
        return 1.0
    nxt = CONT_MARKS[CONT_MARKS.index(mark) + 1]
    v0, v1 = (marks.get(mark) or {}).get("v_z"), (marks.get(nxt) or {}).get("v_z")
    if not v0 or not v1:
        return 1.0
    w = min(max((_hm_min(now_t) - _hm_min(mark)) / (_hm_min(nxt) - _hm_min(mark)), 0.0), 1.0)
    return float(np.sqrt(max(v0 + (v1 - v0) * w, 0.0) / v0))


def _cont_apply(T: dict, z: float, sig: float) -> dict:
    b = int(np.digitize([z], T["z_edges"])[0]); B = T["bins"][b]
    return {"bin": b, "p_up": B["p_up"], "base_hit": T["base_hit"], "n": B["n"],
            "mean": B["mean_z"] * sig, "q20": B["q20_z"] * sig, "q80": B["q80_z"] * sig}


def _cont_calls(R: pd.DataFrame) -> dict:
    """叫牌 (p_up vs base_hit ±0.03，與 learn / verdict 同規則) 的比例、命中、淨正確 (每 100 個時間點)、區間覆蓋。"""
    up = R["y"].to_numpy() > 0
    pu, bh = R["p_up"].to_numpy(float), R["base_hit"].to_numpy(float)
    call = np.where(pu >= bh + 0.03 - 1e-9, 1, np.where(pu <= bh - 0.03 + 1e-9, -1, 0)); nz = call != 0
    ok = ((call == 1) & up) | ((call == -1) & ~up)
    y = R["y"].to_numpy(float)
    return {"n": int(len(R)), "coverage": round(float(nz.mean() * 100), 1), "call_hit": round(float(ok[nz].mean() * 100), 1) if nz.any() else None,
            "edge": round(float((ok[nz].sum() - (~ok[nz]).sum()) / len(R) * 100), 1),
            "band_cov": round(float(np.mean((y >= R["q20"].to_numpy(float)) & (y <= R["q80"].to_numpy(float))) * 100), 1)}


def _cont_wf(Z: pd.DataFrame) -> dict:
    """走動式：每個測試月 (CONT_WF_FROM~) 只用之前的列 (CONT_FROM~) 建各時間點表 → 叫牌統計 (全期 + 近兩年)。"""
    out = []
    for ym in sorted(Z.loc[Z["date"] >= CONT_WF_FROM, "date"].str[:7].unique()):
        te, hist = Z[Z["date"].str[:7] == ym], Z[Z["date"] < ym + "-01"]
        for mk, g in te.groupby("mark"):
            h = hist[hist["mark"] == mk]
            if len(h) < CONT_MIN:
                continue
            T = _cont_table(h["z"].to_numpy(float), h["yz"].to_numpy(float))
            ap = [_cont_apply(T, z, s) for z, s in zip(g["z"].to_numpy(float), g["s"].to_numpy(float))]
            out.append(pd.DataFrame({"date": g["date"].values, "mark": mk, "y": g["y"].values, "p_up": [a["p_up"] for a in ap], "base_hit": T["base_hit"],
                                     "q20": [a["q20"] for a in ap], "q80": [a["q80"] for a in ap]}))
    if not out:
        return {}
    R = pd.concat(out, ignore_index=True)
    rec_from = f"{int(R['date'].max()[:4]) - 1}-01-01"
    res = {mk: {**_cont_calls(g), "recent": _cont_calls(g[g["date"] >= rec_from]) if (g["date"] >= rec_from).any() else None} for mk, g in R.groupby("mark")}
    res["all"] = {**_cont_calls(R), "recent": _cont_calls(R[R["date"] >= rec_from]), "from": str(R["date"].min()), "to": str(R["date"].max()), "recent_from": rec_from}
    return res


def _cont_build(hist: pd.DataFrame, sp: dict, target: str, marks_list: list[str]) -> tuple[dict, pd.DataFrame]:
    tc = f"to_{target}"
    Z = hist[hist["mark"].isin(marks_list) & (hist["date"] >= CONT_FROM)].dropna(subset=["ret_open", tc])[["date", "mark", "ret_open", tc]].copy()
    Z["s"] = pd.to_numeric(Z["date"].map(sp), errors="coerce")
    Z = Z[np.isfinite(Z["s"]) & (Z["s"] > 0)].rename(columns={tc: "y"}).sort_values(["date", "mark"]).reset_index(drop=True)
    Z["z"] = Z["ret_open"] / Z["s"]; Z["yz"] = Z["y"] / Z["s"]
    marks = {}
    for mk, g in Z.groupby("mark"):
        if len(g) < CONT_MIN:
            continue
        T = _cont_table(g["z"].to_numpy(float), g["yz"].to_numpy(float))
        T["z_edges"] = [round(e, 4) for e in T["z_edges"]]
        T["base_mean_z"] = round(T["base_mean_z"], 4)
        T["v_z"] = round(T["v_z"], 5)
        for B in T["bins"]:
            for k in ("mean_z", "q20_z", "q80_z"):
                B[k] = round(B[k], 4)
        T["ic"] = round(float(g["ret_open"].rank().corr(g["y"].rank())), 3)
        marks[mk] = T
    return marks, Z


def cont_tables(hist: pd.DataFrame, sp: dict) -> dict | None:
    """延續模型的表 (train 每週重算，存進 bundle['cont'])。hist = build_history('twse') 的列；sp = {date: 前一日 σ}。
    11-04：另有 short = {目標: {時點: 表}} (現在 → 11:00 / 12:00 / 13:00) 與 short_metrics。"""
    marks, Z = _cont_build(hist, sp, CONT_TARGET, CONT_MARKS)
    if not marks:
        return None
    out = {"marks": marks, "score": "ret_open / sigma_prev", "open_at": "09:05", "target": CONT_TARGET, "sigma": "range_levels.sigma_series 前一日",
           "from": str(Z["date"].min()), "to": str(Z["date"].max()), "metrics": _cont_wf(Z), "short": {}, "short_metrics": {},
           "short_call": sorted(f"{a}>{b}" for a, b in CONT_SHORT_CALL)}
    for tm, mks in CONT_SHORT.items():
        try:
            sm, Zs = _cont_build(hist, sp, tm, mks)
            if sm:
                out["short"][tm] = sm
                out["short_metrics"][tm] = _cont_wf(Zs)
        except Exception as e:  # noqa: BLE001
            log.warning("cont short %s: %s", tm, e)
    return out


def train(write: bool = True) -> dict:
    long_ = backtest.load_long("2010-01-01")
    mat = market_matrix(long_)
    night = finmind.tx_night_history("2023-01-01")
    ctx = {"daily": _daily_features(mat), "night_map": dict(zip(night["date"], night["night_chg_pct"])) if not night.empty else {}}
    # 兩個來源都跑同一套逐季 walk-forward，在共同月份比盤中 IC，選較好的 (平手偏好 TWSE：開盤定義與盤中即時一致)
    try:
        tw = twse.intraday_marks()
    except Exception as e:  # noqa: BLE001
        log.warning("TWSE intraday marks unavailable: %s", e)
        tw = {}
    hists = {}
    if len(tw) >= 500:
        hists["twse"] = build_history("twse", marks=tw, _ctx=ctx)
    try:
        hists["yahoo"] = build_history("yahoo", _ctx=ctx)
    except Exception as e:  # noqa: BLE001
        log.warning("Yahoo hourly unavailable: %s", e)
    if not hists:
        raise RuntimeError("小時模型：TWSE 與 Yahoo 盤中資料都取不到")
    oos_all = {src: {tm: _walk_forward_months(h, f"to_{tm}", from_ym=WF_FROM if src == "twse" else None, quarterly=True)
                     for tm in TARGET_MARKS} for src, h in hists.items()}
    compare = {}
    if len(hists) == 2:
        common = None
        for src in hists:
            ms = set().union(*[set(o["year"]) for o in oos_all[src].values() if not o.empty])
            common = ms if common is None else common & ms
        for src in hists:
            v = [_intra_ic(oos_all[src][tm], common) for tm in TARGET_MARKS]
            v = [x for x in v if x is not None]
            compare[src] = round(float(np.mean(v)), 4) if v else None
        compare["months"] = len(common or [])
        src = "twse" if (compare.get("twse") is not None and (compare.get("yahoo") is None or compare["twse"] >= compare["yahoo"] - 0.005)) else "yahoo"
    else:
        src = next(iter(hists))
    hist = hists[src]
    results = {}
    bundle = {"features": FEATURES, "trained_at": dt.datetime.now(config.TZ).isoformat(), "days": int(hist["date"].nunique()),
              "train_end": str(hist["date"].max()), "source": src, "source_compare": compare, "models": {}, "calibration": {}, "metrics": {}}
    for tm in TARGET_MARKS:
        target = f"to_{tm}"
        oos = oos_all[src][tm]
        met = _metrics(oos)
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
    # 早盤模型 (只在 TWSE 來源有 09:15 / 09:30)：只用 09:15 / 09:30 兩列訓練
    if src == "twse":
        he = build_history("twse", marks_list=EARLY_MARKS, marks=tw, _ctx=ctx)
        he = he[he["mark"] != "pre"].reset_index(drop=True)
        early = {"marks": EARLY_MARKS[1:], "models": {}, "calibration": {}, "metrics": {}}
        for tm in TARGET_MARKS:
            target = f"to_{tm}"
            oos = _walk_forward_months(he, target, from_ym=WF_FROM, quarterly=True)
            met = _metrics(oos)
            d = he.dropna(subset=[target])
            early["models"][tm] = M.fit_ensemble(d[FEATURES], d[target], PARAMS)
            early["calibration"][tm] = {"intra": M.calibrate(oos)} if len(oos) >= 50 else None
            early["metrics"][tm] = {k: v for k, v in met.items() if k != "calibration"}
            results[f"early→{tm}"] = met
        bundle["early"] = early
        try:   # 11-02 延續模型 (10:00~13:00 → 13:30)
            from . import range_levels as RL
            lf = long_.reset_index(drop=True)
            sp = dict(zip(lf["date"].astype(str).str[:10], RL.sigma_series(lf).shift(1).values))
            cont = cont_tables(hist, sp)
            if cont:
                bundle["cont"] = cont
                results["_cont"] = cont["metrics"]
        except Exception as e:  # noqa: BLE001
            log.warning("intraday cont tables: %s", e)
    results["_source"] = {"chosen": src, **compare}
    if write:
        M.save("intraday", bundle)
        M.save_json("intraday_metrics", {**bundle["metrics"], "_source": results["_source"],
                                         "_early": (bundle.get("early") or {}).get("metrics"), "_cont": bundle.get("cont")})
        if tw:
            twse.save_intraday_marks(tw)   # 補抓到的新交易日寫回 (train 模式會 commit 進 git)
    return results


# ------------------------------------------------------------------ 即時預測
def _live_rec(snapshot: dict, early: bool = False) -> tuple[dict, str] | None:
    """由即時 1 分鐘分時建立 rec (各時間點價、累積高低) 與目前 mark。early=True 且 09:15~10:00 → 早盤 mark (09:15 / 09:30)。"""
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
    rec = {"prev_close": idx["prev"], "open": open_px, "open_0905": open_px, "hi": {}, "lo": {}}
    last = idx.get("last") or bars[-1]["close"]
    closes = [b["close"] for b in bars if b["time"] >= "09:05"] or [b["close"] for b in bars]   # 高低從 09:05 起算 (與 TWSE 訓練資料同口徑)
    if early and "09:15" <= now_t < "10:00":
        cur = "09:30" if now_t >= "09:30" else "09:15"
        rec[cur] = last
        rec["hi"][cur], rec["lo"][cur] = max(closes + [last]), min(closes + [last])
        if cur == "09:30":
            rec["09:15"] = px_at("09:15") or open_px      # 09:30 的 ret_1h 以 09:15 價計 (與訓練相同)
        return rec, cur
    cur = "10:00"                      # 10:00 之前 (無早盤模型時) 也套用 10:00 模型 (以現價視為 10:00 時間點，屬近似)
    for m in MARKS[2:]:
        if now_t >= m:
            cur = m
    rec[cur] = last
    rec["hi"][cur], rec["lo"][cur] = max(closes + [last]), min(closes + [last])
    # ret_1h：一小時前的實際價 (不足一小時則用開盤價)
    h, mi = int(now_t[:2]), int(now_t[3:])
    p1h = px_at(f"{h - 1:02d}:{mi:02d}") if h - 1 >= 9 else None
    prev_mark = MARKS[MARK_IDX[cur] - 1]
    if prev_mark != "pre":
        rec[prev_mark] = p1h or open_px
    # 10:00 的 ret_1h 在訓練就是以 09:05 開盤計 (mark_rows：前一時間點 = pre → 開盤價)，open 不再改。
    # 11-02 前這裡在 10:00~10:59 把 open 換成一小時前的價 (10:00 時是失真的 09:00 那筆) → gap_pct / ret_open 與訓練口徑不同
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
    early_b = bundle.get("early") if isinstance(bundle.get("early"), dict) else None
    rec_mark = _live_rec(snapshot, early=bool(early_b)) if live else None
    use_early = bool(rec_mark and early_b and rec_mark[1] in early_b.get("marks", []))
    if rec_mark:
        rec, mark = rec_mark
        day = now.date().isoformat()
    else:
        live = False
        day = twse.next_trading_days(last_date, 1)[0]
        rec, mark = {"prev_close": float(mat["close"].iloc[-1]), "open": None, "hi": {}, "lo": {}}, "pre"
    from .short_term import night_final as _nf
    night = ((snapshot or {}).get("tx_night") or {}).get("change_pct") if _nf(snapshot) else None   # 2026-09-24：夜盤收盤後才用
    rows = mark_rows(day, rec, _daily_for_next(mat, day), night, live_mark=mark, marks_list=EARLY_MARKS if use_early else None)
    if not rows:
        return {"error": "無法建立當前時間點特徵"}
    row = rows[0]
    x = pd.DataFrame([row])[FEATURES].astype(float)
    px = float(row["price"])
    out = {"day": day, "mark": mark, "price": px, "prev_close": float(row["prev_close"]), "live": live, "night_chg_pct": night,
           "basis_date": last_date, "trained_at": bundle["trained_at"], "days": bundle["days"], "targets": {},
           "source": bundle.get("source", "yahoo"), "model": "early" if use_early else "main"}
    now_t = now.strftime("%H:%M")
    cur_t = "00:00" if mark == "pre" else (now_t if live else mark)
    out["note"] = ("早盤模型 (09:15 / 09:30 時間點，2018~ 逐年驗證訊號弱：IC 約 0.06~0.08)" if use_early else
                   "10:00 前以現價套用 10:00 時間點模型 (近似)" if live and now_t < "10:00" else
                   "開盤前預測含跳空：主要來自夜盤台指期，可預測性高但多屬『開盤已反映』的部分" if mark == "pre" else "")
    for tm in TARGET_MARKS:
        if tm <= cur_t:
            continue
        B = early_b if use_early else bundle
        cal_all = B["calibration"].get(tm) or {}
        cal = cal_all.get("pre" if mark == "pre" else "intra")
        pred = float(M.predict_ensemble(B["models"][tm], x)[0])
        c = M.apply_calibration(cal, pred) if cal else {}
        mean_ = c.get("hist_mean") if c.get("hist_mean") is not None else pred
        out["targets"][tm] = {"pred": round(pred, 2), **c,
                              "level": round(px * (1 + mean_ / 100), 0),
                              "level_lo": round(px * (1 + (c.get("q20") if c.get("q20") is not None else pred) / 100), 0),
                              "level_hi": round(px * (1 + (c.get("q80") if c.get("q80") is not None else pred) / 100), 0),
                              "drivers": M.explain(B["models"][tm], x, NAMES, top=4), "metrics": B["metrics"].get(tm, {})}
    # 11-02 延續模型：10:00 起 (09:00~09:14 雖標 10:00 但不算) 的 13:30 目標改由 cont 表給 (欄位同 LightGBM，App / learn / verdict 照讀)
    cont = bundle.get("cont") if isinstance(bundle.get("cont"), dict) else None
    if live and not use_early and now_t >= "10:00" and mark in CONT_MARKS and cont and mark in (cont.get("marks") or {}) and CONT_TARGET in out["targets"]:
        try:
            from . import range_levels as RL
            sc = scored.reset_index(drop=True)
            ds = sc["date"].astype(str).str[:10].tolist()
            i = len(ds) - 1
            while i >= 0 and ds[i] >= day:   # σ 用前一交易日 (scored 若已含今日列則往前一筆)
                i -= 1
            sig = float(RL.sigma_series(sc).iloc[i]) if i >= 0 else float("nan")
            o905 = float(rec.get("open_0905") or rec.get("open") or 0)
            if np.isfinite(sig) and sig > 0 and o905 > 0:
                T = cont["marks"][mark]
                ro = (px / o905 - 1) * 100; z = ro / sig
                r = _cont_apply(T, z, sig)
                k = _cont_k(cont["marks"], mark, now_t)   # 11-04 整點之間區間縮放 (上漲率 / 平均不變)
                hm = r["mean"]; q20, q80 = hm + (r["q20"] - hm) * k, hm + (r["q80"] - hm) * k
                old = out["targets"][CONT_TARGET]
                drv = {"feature": "ret_open", "name": NAMES["ret_open"], "value": round(ro, 3), "contrib": round(hm, 3)}
                out["targets"][CONT_TARGET] = {
                    "pred": round(hm, 2), "bin": r["bin"], "p_up": r["p_up"], "hist_mean": round(hm, 2), "q20": round(q20, 2), "q80": round(q80, 2), "n": r["n"],
                    "base_hit": r["base_hit"], "base_mean": round(T["base_mean_z"] * sig, 2),
                    "level": round(px * (1 + hm / 100), 0), "level_lo": round(px * (1 + q20 / 100), 0), "level_hi": round(px * (1 + q80 / 100), 0),
                    "drivers": {"positive": [drv] if hm > 0 else [], "negative": [drv] if hm < 0 else [], "bias": 0.0},
                    "metrics": {**((cont.get("metrics") or {}).get(mark) or {}), "ic": T.get("ic")}, "model": "cont", "band_k": round(k, 3),
                    "lgbm": {f: old.get(f) for f in ("pred", "p_up", "base_hit", "level", "level_lo", "level_hi")}}
                out["cont"] = {"mark": mark, "open_0905": round(o905, 2), "ret_open": round(ro, 3), "sigma_prev": round(sig, 3), "sigma_date": ds[i], "z": round(z, 3), "band_k": round(k, 3),
                               "table_from": cont.get("from"), "table_to": cont.get("to")}
                out["note"] = (f"13:30 收盤預估用延續模型：09:05 開盤至今 {ro:+.2f}% ÷ 前一日波動 {sig:.2f}% = {z:+.2f}，"
                               f"查 {mark} 時間點歷史同分位 (2018~，n={r['n']})")
                # 11-04 短目標：區間全用延續表 (整點之間依剩餘時間 √ 縮放)；上漲率只給 10:00 / 11:00 → 13:00 兩格，且離目標 ≥ 30 分鐘
                for tm, ST_ in (cont.get("short") or {}).items():
                    if tm not in out["targets"] or mark not in ST_:
                        continue
                    T2 = ST_[mark]; r2 = _cont_apply(T2, z, sig); h2 = r2["mean"]
                    rem, full = _hm_min(tm) - _hm_min(now_t), _hm_min(tm) - _hm_min(mark)
                    k2 = float(np.sqrt(max(rem, 0) / full)) if full > 0 and now_t > mark else 1.0
                    lo2, hi2 = h2 + (r2["q20"] - h2) * k2, h2 + (r2["q80"] - h2) * k2
                    ok_call = (mark, tm) in CONT_SHORT_CALL and rem >= CONT_SHORT_MUTE_MIN
                    o2 = out["targets"][tm]
                    out["targets"][tm] = {
                        "pred": round(h2, 2), "bin": r2["bin"], "p_up": r2["p_up"] if ok_call else None, "p_up_table": r2["p_up"], "call_ok": ok_call,
                        "hist_mean": round(h2, 2), "q20": round(lo2, 2), "q80": round(hi2, 2), "n": r2["n"], "base_hit": r2["base_hit"] if ok_call else None,
                        "level": round(px * (1 + h2 / 100), 0), "level_lo": round(px * (1 + lo2 / 100), 0), "level_hi": round(px * (1 + hi2 / 100), 0),
                        "drivers": {"positive": [], "negative": [], "bias": 0.0}, "model": "cont", "band_k": round(k2, 3), "ic": T2.get("ic"),
                        "metrics": ((cont.get("short_metrics") or {}).get(tm) or {}).get(mark) or {},
                        "lgbm": {f: o2.get(f) for f in ("pred", "p_up", "base_hit", "level", "level_lo", "level_hi")}}
                if cont.get("short"):
                    out["note"] += "；11:00 / 12:00 / 13:00 也用延續表 (上漲率只在 10:00、11:00 → 13:00 且離目標 ≥ 30 分鐘時給)"
        except Exception as e:  # noqa: BLE001
            log.warning("intraday cont forecast: %s", e)
    out["asof_t"] = now_t if live else None
    return out
