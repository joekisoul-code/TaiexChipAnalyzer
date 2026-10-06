"""潛力股雷達 研究 5 共用工具 (10-07)。只讀資料；不寫專案 data/ 或模型檔。

資料：twse_10y.parquet (全上市 1,084 檔 Yahoo 日 K，清單 = 2026-10-02 仍上市 → 期間下市者不在內)、twii_10y.parquet、universe170.parquet (正式模型的 170 檔)。
特徵/標籤：完全使用正式程式 chip.predict.treasure.features (FEATS / SURGE_FEATS / hit / fin21 / surge / fin20)。
壞資料：單日 |漲跌| > 11% (減資/分割未還原) 之後 61 根與之前 21 根剔除 (同 radar_study2/3)。
候選條件 (同正式)：amount > 5 千萬、當日漲幅 < 9.3%、SURGE_FEATS 無缺值；year ≥ 2018。

宇宙欄位：
  u170  = 正式模型的固定 170 檔 (2026-09 依「當時」成交值挑的 → 對 2022~2025 回測有後見之明)
  rk    = 時點正確 (PIT) 成交值排名：每月第一個交易日，以「前一交易日為止 120 日平均成交值 (close×volume)」排名 (1 = 最大)，整月沿用
          → in_pit(N) = rk <= N
用法：
  import harness as H
  F = H.load_feat()                         # 全部候選列 (第一次會建 feat_all.parquet，約數分鐘)
  P = H.walk_forward(F, train_mask=F.u170, test_mask=F.u170, cadence="Y")   # 回傳 F 的子集 + p/ps/thA/thAp 欄位
  picks = H.simulate(P, cand_mask=P.u170)   # 正式選股規則 → 推薦列 (radar T/S、tier)
  H.report(picks)                            # A/A+/B/B+/S3 指標 + 逐年 + Wilson
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/TaiexChipAnalyzer")
from chip.predict import treasure as T  # noqa: E402

DIR = Path(__file__).parent
FEAT_PATH = DIR / "feat_all.parquet"
KEEP = list(dict.fromkeys(["date", "code", "year", "open", "high", "low", "close", "volume", "amount", "m_close", "m_ret1", "m_bias20", "m_ret20"]
                          + T.SURGE_FEATS + ["hit", "fin21", "surge", "fin20", "amt20", "rk", "u170"]))


def market() -> pd.DataFrame:
    mk = pd.read_parquet(DIR / "twii_10y.parquet")
    mk["m_close"] = mk["close"].astype(float); mk["m_ret1"] = mk["m_close"].pct_change() * 100
    mk["m_bias20"] = (mk["m_close"] / mk["m_close"].rolling(20).mean() - 1) * 100; mk["m_ret20"] = mk["m_close"].pct_change(20) * 100
    return mk[["date", "m_close", "m_ret1", "m_bias20", "m_ret20"]]


def panel() -> pd.DataFrame:
    return pd.read_parquet(DIR / "twse_10y.parquet")


def _one(args):
    code, g, mk = args
    g = g.sort_values("date").reset_index(drop=True)
    g["amount"] = g["close"] * g["volume"].fillna(0)
    f = T.features(g[["date", "open", "high", "low", "close", "volume", "amount"]], mk)
    jump = (f["close"].pct_change().abs() > 0.11).astype(float)
    bad = (jump.rolling(61, min_periods=1).max() > 0) | (jump[::-1].rolling(22, min_periods=1).max()[::-1] > 0)
    f["amt20"] = f["amount"].rolling(20).mean()
    f = f[~bad.values]
    return f.assign(code=code)


def pit_rank(P: pd.DataFrame) -> pd.DataFrame:
    """每月第一個交易日、以前一交易日為止 120 日平均成交值排名 (全上市)。回傳 code/date/rk (每個交易日)。"""
    P = P[["code", "date", "close", "volume"]].copy()
    P["amt"] = P["close"] * P["volume"].fillna(0)
    P = P.sort_values(["code", "date"])
    P["a120"] = P.groupby("code")["amt"].transform(lambda s: s.rolling(120, min_periods=60).mean().shift(1))
    dates = sorted(P["date"].unique()); ym = pd.Series(dates).str[:7]
    firsts = set(pd.Series(dates)[~ym.duplicated()].tolist())
    R = P[P["date"].isin(firsts)].copy()
    R["rk"] = R.groupby("date")["a120"].rank(ascending=False, method="first")
    R["ym"] = R["date"].str[:7]
    P["ym"] = P["date"].str[:7]
    out = P[["code", "date", "ym"]].merge(R[["code", "ym", "rk"]], on=["code", "ym"], how="left")
    return out[["code", "date", "rk"]]


def build_feat() -> pd.DataFrame:
    P = panel(); mk = market()
    u170 = set(pd.read_parquet(DIR / "universe170.parquet")["code"])
    with ProcessPoolExecutor(12) as ex:
        frames = list(ex.map(_one, [(c, g, mk) for c, g in P.groupby("code")], chunksize=8))
    D = pd.concat(frames, ignore_index=True)
    D = D.merge(pit_rank(P), on=["code", "date"], how="left")
    D["year"] = D["date"].str[:4].astype(int); D["u170"] = D["code"].isin(u170)
    D = D[(D["year"] >= 2018) & (D["amount"] > 5e7) & (D["pct"] < 9.3)].dropna(subset=T.SURGE_FEATS)
    D = D[KEEP].reset_index(drop=True)
    for c in D.columns:
        if D[c].dtype == "float64" and c not in ("m_close", "close", "open", "high", "low"):
            D[c] = D[c].astype("float32")
    D.to_parquet(FEAT_PATH, index=False)
    return D


def load_feat() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        return build_feat()
    return pd.read_parquet(FEAT_PATH)


# ---------------- 走動式訓練 ----------------
def _periods(dates: pd.Series, cadence: str, start: str = "2022-01-01"):
    """cadence: Y (每年) / H (半年) / Q (每季) / M (每月) → [(期間開始, 期間結束(不含))]。"""
    d = pd.to_datetime(pd.Series(sorted(dates[dates >= start].unique())))
    if not len(d):
        return []
    freq = {"Y": "YS", "H": "6MS", "Q": "QS", "M": "MS"}[cadence]
    starts = pd.date_range(pd.Timestamp(start), d.max() + pd.Timedelta(days=1), freq=freq)
    if cadence == "H":
        starts = pd.DatetimeIndex([s for s in pd.date_range(pd.Timestamp(start), d.max() + pd.Timedelta(days=1), freq="MS") if s.month in (1, 7)])
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else d.max() + pd.Timedelta(days=1)
        out.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return out


def walk_forward(F: pd.DataFrame, train_mask, test_mask, cadence: str = "Y", embargo_days: int = 40, window_years: float | None = None,
                 train_start: str = "2018-01-01", params: dict | None = None, sparams: dict | None = None, feats: list | None = None,
                 sfeats: list | None = None, seeds=(None,), weight_fn=None, th_rows_mask=None, verbose: bool = False) -> pd.DataFrame:
    """回傳 test_mask 列 (2022-01 起) 的 p / ps 與當期門檻 thA (0.9) / thAp (Q_APLUS) / thS (飆股 0.9)。
    訓練列 = train_mask ∧ date < 期間開始 − embargo (標籤 21 交易日 ≈ 30 日曆天；40 天含春節保守) ∧ (window_years 內)。
    門檻 (同正式 train)：最終模型對「訓練列最後兩個年份」的樣本內分數取分位 (th_rows_mask 可再限定)。
    seeds：多個種子 → 分數平均 (bagging 研究用)。weight_fn(train_df) → sample_weight。"""
    import lightgbm as lgb
    params = dict(params or T.PARAMS); sparams = dict(sparams or T.SURGE_PARAMS)
    feats = feats or T.FEATS; sfeats = sfeats or T.SURGE_FEATS
    train_mask = pd.Series(train_mask, index=F.index).fillna(False).astype(bool)
    test_mask = pd.Series(test_mask, index=F.index).fillna(False).astype(bool)
    out = []
    for s, e in _periods(F.loc[test_mask, "date"], cadence):
        cut = (pd.Timestamp(s) - pd.Timedelta(days=embargo_days)).strftime("%Y-%m-%d")
        lo = train_start if window_years is None else max(train_start, (pd.Timestamp(s) - pd.Timedelta(days=int(365.25 * window_years))).strftime("%Y-%m-%d"))
        tr = train_mask & (F["date"] < cut) & (F["date"] >= lo)
        te = test_mask & (F["date"] >= s) & (F["date"] < e)
        if not te.any() or tr.sum() < 5000:
            continue
        Tr, Te = F[tr], F[te].copy()
        bh, bs = Tr[Tr["hit"].notna()], Tr[Tr["surge"].notna()]
        ly = int(Tr["year"].max()) - 1
        thr = Tr[(Tr["year"] >= ly)] if th_rows_mask is None else Tr[(Tr["year"] >= ly) & th_rows_mask[tr][Tr["year"] >= ly]]
        p = np.zeros(len(Te)); ps = np.zeros(len(Te)); pth = np.zeros(len(thr)); psth = np.zeros(len(thr))
        for sd in seeds:
            kw = {} if sd is None else {"random_state": int(sd)}
            w = weight_fn(bh) if weight_fn else None; ws = weight_fn(bs) if weight_fn else None
            m = lgb.LGBMClassifier(**params, **kw).fit(bh[feats], bh["hit"], sample_weight=w)
            m2 = lgb.LGBMClassifier(**sparams, **kw).fit(bs[sfeats], bs["surge"], sample_weight=ws)
            p += m.predict_proba(Te[feats])[:, 1]; ps += m2.predict_proba(Te[sfeats])[:, 1]
            pth += m.predict_proba(thr[feats])[:, 1]; psth += m2.predict_proba(thr[sfeats])[:, 1]
        k = len(seeds)
        Te["p"], Te["ps"] = p / k, ps / k
        Te["thA"] = float(np.quantile(pth / k, 0.9)); Te["thAp"] = float(np.quantile(pth / k, T.Q_APLUS)); Te["thS"] = float(np.quantile(psth / k, 0.9))
        Te["period"] = s
        out.append(Te)
        if verbose:
            print("wf", cadence, s, "train", int(tr.sum()), "test", int(te.sum()), flush=True)
    return pd.concat(out) if out else F.iloc[:0].assign(p=np.nan, ps=np.nan)


# ---------------- 選股模擬 (正式規則) ----------------
def screen(E: pd.DataFrame) -> pd.Series:
    """正式掃描器 power (無 PBR/PER)；同 treasure.train / radar_study。"""
    mom = np.where(E["pct"] >= 0, np.minimum(E["pct"], 7) * 1.6, E["pct"] * 0.6)
    return pd.Series(mom + E["lval"] * 2 + np.minimum(E["amp"], 8) * .8 + np.where(E["pct"] < E["m_ret1"], np.minimum(E["m_ret1"] - E["pct"], 5) * 1.2, 0)
                     + np.where((E["pct"] >= 3) & (E["pct"] < 9), (E["pct"] - 3) * .6, 0), index=E.index)


def simulate(P: pd.DataFrame, cand_mask=None, pool: int = 40, cap: int = 6, cool: int = 30, q_gate: float = 0.0, spool: int = 80, stop: int = 3,
             scool: int = 28, pcol: str = "p", pscol: str = "ps", tier_fn=None) -> pd.DataFrame:
    """正式規則：挖寶 = 掃描前 pool 名依 p 排序取 cap 檔 (同檔 cool 天內跳過)；tier：p ≥ thA ∧ 大盤月線乖離 < q_gate → A (≥ thAp → A+)，
    p ≥ thA ∧ 月線上 → B+，其餘 B。飆股 (App 口徑) = 掃描前 spool 名中 ps 前 stop 名 (同檔 scool 天內跳過、不補位)。
    回傳推薦列：radar (T/S)、tier、date、code、year、hit、fin21、surge、fin20、p、ps、m_bias20 等。"""
    E = P if cand_mask is None else P[pd.Series(cand_mask, index=P.index).fillna(False).astype(bool)]
    E = E.assign(screen=screen(E))
    recs = []; lastT = {}; lastS = {}
    for d, g in E.sort_values("date").groupby("date", sort=True):
        n = 0; td = pd.Timestamp(d)
        for r in g.nlargest(pool, "screen").sort_values(pcol, ascending=False).itertuples():
            lp = lastT.get(r.code)
            if lp is not None and (td - lp).days < cool:
                continue
            pv = getattr(r, pcol)
            if tier_fn is not None:
                tier = tier_fn(r)
            else:
                hi = pv >= r.thA; gate = r.m_bias20 < q_gate
                tier = "B" if not hi else ("B+" if not gate else ("A+" if pv >= r.thAp else "A"))
            recs.append(("T", tier, r.Index)); lastT[r.code] = td; n += 1
            if n >= cap:
                break
        for r in g.nlargest(spool, "screen").nlargest(stop, pscol).itertuples():
            lp = lastS.get(r.code)
            if lp is not None and (td - lp).days < scool:
                continue
            recs.append(("S", None, r.Index)); lastS[r.code] = td
    if not recs:
        return pd.DataFrame()
    idx = [i for _, _, i in recs]
    R = E.loc[idx].copy()
    R["radar"] = [a for a, _, _ in recs]; R["tier"] = [t for _, t, _ in recs]
    return R.reset_index().rename(columns={"index": "row"})


# ---------------- 指標 ----------------
def wilson(k: float, n: int, z: float = 1.96):
    if n <= 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def agg(x: pd.DataFrame, ok: str, ret: str) -> dict | None:
    x = x[x[ok].notna() & x[ret].notna()]
    if not len(x):
        return None
    k = float(x[ok].sum()); n = int(len(x)); w = float((x[ret] > 0).sum())
    return {"n": n, "hit": round(k / n, 3), "hit_ci": wilson(k, n), "win": round(w / n, 3), "win_ci": wilson(w, n), "avg": round(float(x[ret].mean()), 2),
            "med": round(float(x[ret].median()), 2), "q10": round(float(x[ret].quantile(.1)), 2),
            "hit_by_year": {int(y): round(float(g[ok].mean()), 3) for y, g in x.groupby("year")},
            "win_by_year": {int(y): round(float((g[ret] > 0).mean()), 3) for y, g in x.groupby("year")},
            "n_by_year": {int(y): int(len(g)) for y, g in x.groupby("year")}}


def report(R: pd.DataFrame) -> dict:
    if R is None or not len(R):
        return {}
    T_ = R[R["radar"] == "T"]; S_ = R[R["radar"] == "S"]
    return {"A": agg(T_[T_["tier"].isin(["A", "A+"])], "hit", "fin21"), "A+": agg(T_[T_["tier"] == "A+"], "hit", "fin21"),
            "B+": agg(T_[T_["tier"] == "B+"], "hit", "fin21"), "B": agg(T_[T_["tier"] == "B"], "hit", "fin21"),
            "T_all": agg(T_, "hit", "fin21"), "S3": agg(S_, "surge", "fin20"), "S3_win": agg(S_.assign(_w=(S_["fin20"] > 0).astype(float)), "_w", "fin20")}


def pool_base(P: pd.DataFrame, cand_mask=None, pool: int = 40, spool: int = 80) -> dict:
    """同日候選池 (掃描前 pool / spool 名) 的平均命中 / 飆股率 = 不靠模型的基準。"""
    E = P if cand_mask is None else P[pd.Series(cand_mask, index=P.index).fillna(False).astype(bool)]
    E = E.assign(screen=screen(E))
    a = E.groupby("date", group_keys=False).apply(lambda g: g.nlargest(pool, "screen"))
    b = E.groupby("date", group_keys=False).apply(lambda g: g.nlargest(spool, "screen"))
    return {"pool_hit": round(float(a["hit"].mean()), 3), "pool_win21": round(float((a["fin21"] > 0).mean()), 3),
            "spool_surge": round(float(b["surge"].mean()), 3), "spool_win20": round(float((b["fin20"] > 0).mean()), 3),
            "pool_hit_by_year": {int(y): round(float(g["hit"].mean()), 3) for y, g in a.groupby("year")},
            "spool_surge_by_year": {int(y): round(float(g["surge"].mean()), 3) for y, g in b.groupby("year")}}


if __name__ == "__main__":
    import time
    t0 = time.time()
    F = build_feat()
    print("feat rows", len(F), "codes", F["code"].nunique(), "u170 rows", int(F["u170"].sum()), "secs", round(time.time() - t0), flush=True)
    print(F.groupby("year").agg(n=("code", "size"), codes=("code", "nunique")).to_string())
    # 宇宙重疊：每年 PIT 前 170 與固定 170 的重疊
    for y in range(2018, 2027):
        g = F[F["year"] == y]
        pit = set(g[g["rk"] <= 170]["code"]); fx = set(g[g["u170"]]["code"])
        print(y, "PIT170 codes", len(pit), "fixed170 codes", len(fx), "overlap", len(pit & fx))
