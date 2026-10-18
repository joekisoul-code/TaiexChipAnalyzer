"""融資行為動態模組 (chip/analysis/margin_dyn.py) 單元測試 — SPEC §3.7 (c)~(j) (+ (b) 恆等式、(l) backfill)。不連網、不需 LightGBM；
資料層函式一律以 monkeypatch 取代，SQLite 用暫存 CHIP_DATA_DIR。

    python tests/test_margin_dyn.py
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="margin_dyn_test_")
os.environ["CHIP_DATA_DIR"] = _TMP                     # 不碰真實 data/chip.sqlite

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip import store  # noqa: E402
from chip.analysis import margin_dyn as M  # noqa: E402
from chip.serialize import clean  # noqa: E402

M.THROTTLE = 0
FETCHERS = ("_fetch_margin_all", "_fetch_closes", "_fetch_tpex", "_fetch_total_flows", "_fetch_total_margin", "_fetch_twse_ms", "_fetch_taiex",
            "_fetch_maint_real", "_fetch_holidays")


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def offline(calls: list | None = None):
    """所有資料層函式 → 記錄並 raise。"""
    calls = calls if calls is not None else []
    olds = {n: getattr(M, n) for n in FETCHERS}

    def mk(name):
        def f(*a, **k):
            calls.append(name)
            raise RuntimeError("offline")
        return f
    for n in FETCHERS:
        setattr(M, n, mk(n))
    try:
        yield calls
    finally:
        for n, v in olds.items():
            setattr(M, n, v)


def _wipe_db():
    with store.conn() as c:
        for t in ("margin_stock_daily", "margin_nodata", "daily_metrics", "snapshots"):
            c.execute(f"DELETE FROM {t}") if c.execute("SELECT name FROM sqlite_master WHERE name=?", (t,)).fetchone() else None


def _dates(start="2001-01-02", end="2026-10-13") -> list[str]:
    return pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()


def _hist(seed=3, start="2001-01-02", end="2026-10-13") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _dates(start, end)
    n = len(dates)
    close = 5000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    amt = 2500 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, n)))
    h = pd.DataFrame({"date": dates, "margin_amt": amt, "margin_lots": amt * 1500, "short_lots": amt * 40, "close": close,
                      "amount": 3000 + rng.normal(0, 200, n), "volume_lots": 6e6 + rng.normal(0, 1e5, n)})
    recent = np.arange(n) > n - 500
    h["buy_lots"] = np.where(recent, 340000 + rng.normal(0, 30000, n), np.nan)
    h["sell_lots"] = np.where(recent, 330000 + rng.normal(0, 30000, n), np.nan)
    h["repay_lots"] = np.where(recent, 8000 + rng.normal(0, 1000, n), np.nan)
    h["maint_twse"] = np.where(np.arange(n) > n - 480, 165 + rng.normal(0, 8, n), np.nan)
    h["maint_real"] = h["maint_twse"] + 0.1
    h["tpex_margin_amt"] = np.where(np.arange(n) > n - 400, 1000 + rng.normal(0, 20, n), np.nan)
    h["br_up"], h["br_dn"], h["br_net"], h["usage_gt60_pct"] = 45.0, 44.0, 1.0, 0.3
    return h


def _code_panel(code: str, dates: list[str], lots0=20000.0, seed=1, limit=100000.0, close0=100.0, **tweak) -> pd.DataFrame:
    """合成單檔面板：恆等式 today = prev + buy − sell − repay 全列成立；tweak：prev_override {date: prev}, note {date_range: 'O'}, zero_lots (dates set)。"""
    rng = np.random.default_rng(seed)
    rows, lots, px = [], lots0, close0
    for d in dates:
        buy, sell, rep = max(0.0, rng.normal(300, 100)), max(0.0, rng.normal(290, 100)), max(0.0, rng.normal(5, 3))
        prev = lots
        if d in tweak.get("prev_override", {}):
            prev = tweak["prev_override"][d]
        if d in tweak.get("zero_lots", set()):
            buy, sell, rep, prev = 0.0, 0.0, 0.0, 0.0
        lots = prev + buy - sell - rep
        px = px * float(np.exp(rng.normal(0, 0.015)))
        if d in tweak.get("price_override", {}):
            px = tweak["price_override"][d]
        rows.append({"date": d, "code": code, "mkt": "twse", "name": "N" + code, "note": tweak.get("note", {}).get(d, ""),
                     "m_buy": buy, "m_sell": sell, "m_repay": rep, "m_prev": prev, "m_today": lots, "m_limit": tweak.get("limit_fn", lambda d: limit)(d),
                     "s_buy": 10.0, "s_sell": 9.0, "s_repay": 1.0, "s_prev": 1500.0, "s_today": 1500.0, "s_limit": 100000.0, "offset": 3.0,
                     "close": px, "vol_lots": 20000.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (b) 恆等式
def test_identity_rows_excluded():
    dates = _dates("2025-01-02", "2025-12-31")
    p = _code_panel("1234", dates)
    F = M.stock_features(p)
    assert not F["identity_err"].any() and F["buy"].notna().all()
    p2 = p.copy()
    p2.loc[100, "m_today"] += 500                       # 違反恆等式
    F2 = M.stock_features(p2)
    assert bool(F2["identity_err"].iloc[100]) and F2["identity_err"].sum() == 1
    assert np.isnan(F2["buy"].iloc[100]) and np.isnan(F2["sell"].iloc[100]), "違反列的流量不進統計"
    assert np.isfinite(F2["lots"].iloc[100])


# ------------------------------------------------------------------ (c) 歷史合併
def test_market_history_merge_keep_first_shrink_and_anomaly():
    _wipe_db()
    h = _hist(end="2026-10-10")
    fm = h[["date", "margin_amt", "margin_lots", "short_lots", "buy_lots", "sell_lots", "repay_lots"]].copy()
    bad_date = fm["date"].iloc[-200]
    true_amt = float(fm.loc[fm["date"] == bad_date, "margin_amt"].iloc[0])
    fm.loc[fm["date"] == bad_date, "margin_amt"] = true_amt * 1.25        # FinMind 金額側單日異常 (張數未動)
    fm = fm.iloc[300:]                                                   # fresh 比 prev 短 (縮水)
    prev_rows = [[d, round(a, 1), 1.0, 1.0, None, None] for d, a in zip(h["date"], h["margin_amt"])]
    prev_rows[-1][1] = 1.0                                               # prev 在重疊日的值不同 → fresh 優先 (keep-first)
    prev = {"history": {"cols": M.HISTORY_COLS, "rows": prev_rows}, "flows": "not a table"}
    store.upsert_metrics(bad_date, {"mg_amt": true_amt, "maint_twse": 170.0})       # SQLite 有 TWSE 值
    ms = {"date": "20261013", "margin_lots": 9430485.0, "margin_amt": 6449.78, "margin_buy_lots": 436812.0, "margin_sell_lots": 452990.0,
          "margin_lots_prev": 9453096.0, "short_lots": 225459.0}
    with patched(M, "_fetch_total_flows", lambda: fm), patched(M, "_fetch_twse_ms", lambda: ms), \
            patched(M, "_fetch_taiex", lambda: h[["date", "close", "amount"]]), patched(M, "_fetch_maint_real", lambda: (_ for _ in ()).throw(RuntimeError())):
        df = M.market_history(prev)
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    assert len(df) == len(h) + 1, "fresh ∪ prev ∪ TWSE 當日：縮水保護 (prev 的舊日期保留)"
    assert df["date"].iloc[-1] == "2026-10-13" and abs(df["margin_amt"].iloc[-1] - 6449.78) < 1e-6
    assert abs(df.loc[df["date"] == bad_date, "margin_amt"].iloc[0] - true_amt) < 1e-6, "FinMind 金額異常日以 TWSE 覆寫"
    assert bool(df.loc[df["date"] == bad_date, "amt_fixed"].iloc[0]) and df["amt_fixed"].sum() == 1
    last_overlap = h["date"].iloc[-1]
    assert abs(df.loc[df["date"] == last_overlap, "margin_amt"].iloc[0] - float(h["margin_amt"].iloc[-1])) < 1e-6, "重疊日 fresh 優先"
    assert df["margin_amt"].notna().all(), "必要欄無洞"
    assert float(df.loc[df["date"] == bad_date, "maint_twse"].iloc[0]) == 170.0, "SQLite 的維持率併入"
    assert float(df.loc[df["date"] == "2026-10-13", "repay_lots"].iloc[0]) == 9453096.0 + 436812.0 - 452990.0 - 9430485.0, "TWSE MS 以恆等式推現償"
    # Tier-1 流量 (buy/sell/repay 2001~) 也持久化到 daily_metrics (FinMind 失敗時 turnover_hot 門檻與統計才不會漂移)
    got = store.load_metrics(["mg_buy_lots", "mg_sell_lots", "mg_repay_lots", "mg_amt"])
    n_flow = int(got["mg_buy_lots"].notna().sum())
    assert n_flow == int(fm["buy_lots"].notna().sum()) + 1 and n_flow > 400 and int(got["mg_amt"].notna().sum()) >= len(fm), (n_flow, len(fm))   # +1 = TWSE MS 當日
    with offline():
        df_off = M.market_history({})
    assert int(df_off["buy_lots"].notna().sum()) == n_flow, "離線時流量可由 SQLite 重建"
    # prev 壞 JSON 容錯：整個 prev 不是 dict 也不會炸
    with offline():
        df2 = M.market_history({"history": [{"x": 1}], "flows": None})
    assert isinstance(df2, pd.DataFrame)


# ------------------------------------------------------------------ (d) 維持率自算
def test_maint_direct():
    day = pd.DataFrame({"code": ["2330", "2317", "0050"], "m_today": [1000.0, 2000.0, 500.0]})
    closes = {"2330": (1000.0, 1), "2317": (100.0, 1), "0050": (200.0, 1)}
    amt = (1000 * 1000 * 1000 + 2000 * 100 * 1000 + 500 * 200 * 1000) / 1e8 / 1.6     # 讓維持率 = 160%
    v, cov, src = M.maint_direct(day, closes, amt, {})
    assert abs(v - 160.0) < 1e-6 and cov == 100.0 and src == "twse", (v, cov, src)
    # 缺價檔以 last_px 補 (age ≤60)；單檔貢獻 >0.5 pt → partial
    v2, cov2, src2 = M.maint_direct(day, {"2330": (1000.0, 1), "0050": (200.0, 1)}, amt, {"2317": (100.0, 30)})
    assert abs(v2 - 160.0) < 1e-6 and src2 == "partial", (v2, src2)
    # age >60 → 不補 → 缺價張數占比 >1% → partial 且數值變小
    v3, cov3, src3 = M.maint_direct(day, {"2330": (1000.0, 1), "0050": (200.0, 1)}, amt, {"2317": (100.0, 61)})
    assert v3 < 160 and cov3 < 99 and src3 == "partial"
    # 補價檔貢獻微小且缺價占比 <1% → 仍 twse
    big = pd.DataFrame({"code": [str(i) for i in range(200)], "m_today": [1000.0] * 199 + [1.0]})
    cl = {str(i): (100.0, 1) for i in range(199)}
    amt2 = 199 * 1000 * 100 * 1000 / 1e8 / 1.6
    v4, cov4, src4 = M.maint_direct(big, cl, amt2, {"199": (100.0, 5)})
    assert src4 == "twse" and abs(v4 - 160.0) < 0.01
    assert M.maint_direct(day, closes, None, {}) == (None, None, None)
    # 無價且估不出價 (無 last_px / next_px) 的高價檔：張數占比 0.5% 但市值占 13% → 不能再以 src='twse' 發布 (審查 P1：6949 停牌 1.4 pt 偏差)
    hi = pd.DataFrame({"code": [str(i) for i in range(200)], "m_today": [1000.0] * 200})
    cl = {str(i): (100.0, 1) for i in range(199)}
    amt3 = (199 * 1000 * 100 * 1000 + 1000 * 3000 * 1000) / 1e8 / 1.6
    v5, cov5, src5 = M.maint_direct(hi, cl, amt3, {})
    assert src5 == "partial" and v5 < 160 and cov5 <= 99.5, (v5, cov5, src5)
    # 同一檔有「之後」的收盤 (停牌 → 復牌，面額變更張數 ×20、價格 ÷20)：以張數比還原估市值 → 貢獻 >0.5 pt → partial；估價 ≈ 3000
    assert abs(M._est_px(1000.0, 150.0, 20000.0) - 3000.0) < 1e-9 and M._est_px(1000.0, 150.0, None) == 150.0 and M._est_px(1000.0, None, 1.0) is None
    v6, cov6, src6 = M.maint_direct(hi, cl, amt3, {}, {"199": (150.0, 20000.0)})
    assert src6 == "partial" and abs(v6 - v5) < 1e-9 and cov6 < cov5, (v6, cov6, src6)        # 估計值只用來判 partial / coverage，不進分子
    # 較舊的 last_px (age >60) 同樣只用來估市值
    v7, _, src7 = M.maint_direct(hi, cl, amt3, {"199": (3000.0, 200, 1000.0)})
    assert src7 == "partial" and abs(v7 - v5) < 1e-9
    # 無價、估不出價、張數合計 < UNEST_LOTS_PARTIAL (1,000 張) 且占比 <1% → 仍 twse
    small = pd.DataFrame({"code": [str(i) for i in range(200)], "m_today": [1000.0] * 199 + [500.0]})
    amt4 = 199 * 1000 * 100 * 1000 / 1e8 / 1.6
    v8, cov8, src8 = M.maint_direct(small, cl, amt4, {})
    assert src8 == "twse" and abs(v8 - 160.0) < 1e-6 and cov8 < 100.0, (v8, cov8, src8)
    # 補價檔 (age ≤60) 帶 lots_then 三元組也可
    v9, _, src9 = M.maint_direct(day, {"2330": (1000.0, 1), "0050": (200.0, 1)}, amt, {"2317": (100.0, 30, 2000.0)})
    assert abs(v9 - 160.0) < 1e-6 and src9 == "partial"
    # check：自算與 real 同日差 >1.5 pt
    h = _hist(end="2026-10-13")
    h["maint_real"] = h["maint_twse"] + 2.0
    view, _ = M.market_view(h, None)
    assert view["maint"]["check"] is True and view["maint"]["src"] == "twse" and view["maint"]["label"] == "risk_descriptive"
    h["maint_real"] = h["maint_twse"] + 0.3
    view, _ = M.market_view(h, None)
    assert view["maint"]["check"] is False


# ------------------------------------------------------------------ (e) 衍生
def test_derived_by_hand():
    h = _hist(end="2026-10-13")
    view, d = M.market_view(h, None)
    a = h["margin_amt"].to_numpy()
    L = d.iloc[-1]
    assert abs(L["chg1"] - (a[-1] - a[-2])) < 1e-9 and abs(L["pct20"] - (a[-1] / a[-21] - 1) * 100) < 1e-9
    assert abs(L["acc5"] - ((a[-1] / a[-6] - 1) * 100 - (a[-6] / a[-11] - 1) * 100)) < 1e-9
    past = a[-251:-1]
    assert abs(L["pct250"] - (past <= a[-1]).mean() * 100) < 1e-6, "250 日百分位不含今日"
    c1 = pd.Series(a).diff()
    assert abs(L["z1"] - (c1.iloc[-1] - c1.tail(60).mean()) / c1.tail(60).std()) < 1e-9
    # streak
    s = 0
    for v in c1.to_numpy()[::-1]:
        if np.sign(v) == np.sign(c1.iloc[-1]) and v != 0:
            s += 1
        else:
            break
    assert abs(L["streak"]) == s
    # 周轉 (不含現償) / 現償占出場 / 新增率
    b, sl, rp, lots = h["buy_lots"].iloc[-1], h["sell_lots"].iloc[-1], h["repay_lots"].iloc[-1], h["margin_lots"].iloc[-2]
    assert abs(L["turnover_ratio"] - (b + sl) / lots * 100) < 1e-9 and abs(L["repay_share"] - rp / (sl + rp) * 100) < 1e-9
    assert abs(L["new_rate"] - b / lots * 100) < 1e-9
    # 槓桿強度只有 5 年百分位
    assert abs(L["lev_turn"] - a[-1] / h["amount"].tail(20).mean()) < 1e-9 and "lev_turn_pct1250" in view and "lev_turn_pct_all" not in view
    assert view["lev_turn_pct1250"] is not None and 0 <= view["lev_turn_pct1250"] <= 100
    # 四象限 ±1%
    q = M._quad([2.0, 2.0, -2.0, -2.0, 0.5, 2.0, np.nan], [2.0, -2.0, -2.0, 2.0, 0.5, 0.5, 1.0])
    assert list(q) == ["資增價漲", "資增價跌", "資減價跌", "資減價漲", "中性", "中性", ""]
    assert view["quad"]["text"] in M.QUADS and view["mode"]["id"] in M.QUADS
    # regime：連續 ≥20 日 pct20 同號；base_rate = 歷史處於 regime 的比例
    sign = np.sign(d["pct20"]).fillna(0)
    run = M._run_length(sign)
    exp = ((run >= 20) & (sign != 0)).mean() * 100
    assert abs(view["regime"]["base_rate"] - round(exp, 0)) < 1e-9
    assert view["regime"]["id"] in ("expansion", "contraction", "none")
    # behav60 對手算
    t = d.tail(60)
    assert view["behav60"]["inc_days_pct"] == round((t["chg1"] > 0).mean() * 100, 0)
    down = t[t["ret1"] < 0]
    assert view["behav60"]["inc_on_down_rate"] == round((down["chg1"] > 0).mean() * 100, 0) and view["behav60"]["down_days"] == len(down)
    # 全市場 Σ餘額/Σ限額 不存在
    js = json.dumps(clean(view), ensure_ascii=False)
    assert "use_pct" not in js and "usage_total" not in js
    assert len(view["weekly"]) == 12 and all(set(w) == {"week", "chg", "days", "ret", "ret_basis"} for w in view["weekly"])   # 10-18 chips-1 審查：加口徑鍵
    assert view["weekly_ret_basis"] == "prev_close" and all(w["ret_basis"] == "prev_close" for w in view["weekly"])


def test_maint_pct2004_gated_by_coverage():
    """「2004 年來」百分位 / 文案只在自算維持率覆蓋期起點 ≤ 2004-06-30 時輸出；只有 2026 年資料 → pct2004 null、pct_hist 仍有、文案改「{maint_from} 以來」。"""
    h = _hist(end="2026-10-13")                                  # maint 只有最後 480 列 (2024-11~)
    view, d = M.market_view(h, None)
    mf = str(h.loc[h["maint_twse"].notna(), "date"].iloc[0])
    assert view["maint"]["pct2004"] is None and view["maint"]["pct_hist"] is not None and view["maint"]["since"] == mf
    assert view["maint"]["days"] == int(h["maint_twse"].notna().sum())
    assert d["maint_pct2004"].isna().all() and d["maint_pct_hist"].notna().sum() > 300
    ev = M.events(d, [])
    t = ev["events"]["maint_lt150"]
    assert "2004 年來" not in t["text"] and t["share_since"] == f"{mf} 以來" and t["hist_from"] == mf and t["share"] is not None
    # 使維持率 <150 成立 → active 文案用「{maint_from} 以來」
    h2 = h.copy()
    h2.loc[h2.index[-1], "maint_twse"] = 140.0
    _, d2 = M.market_view(h2, None)
    t2 = M.events(d2, [])["events"]["maint_lt150"]
    assert t2["active"] and f"{mf} 以來" in t2["text"] and "140.0% 低於 150" in t2["text"] and "2004 年來" not in t2["text"]
    # 覆蓋到 2004-02 → pct2004 輸出且文案「2004 年來」
    h3 = h.copy()
    rng = np.random.default_rng(1)
    full = h3["date"] >= "2004-02-11"
    h3.loc[full, "maint_twse"] = 165 + rng.normal(0, 8, int(full.sum()))
    h3.loc[h3.index[-1], "maint_twse"] = 140.0
    view3, d3 = M.market_view(h3, None)
    assert view3["maint"]["pct2004"] is not None and view3["maint"]["since"] == "2004-02-11" and view3["maint"]["pct2004"] == view3["maint"]["pct_hist"]
    t3 = M.events(d3, [])["events"]["maint_lt150"]
    assert "2004 年來" in t3["text"] and t3["share_since"] == "2004 年來"


# ------------------------------------------------------------------ (f) 事件
def test_events_fwd_from_d_plus_1_and_no_lookahead():
    h = _hist(start="2024-01-02", end="2025-12-31")
    c = np.full(len(h), 100.0)
    c[1] = 110.0
    c[2:] = 121.0
    h["close"] = c
    _, d = M.market_view(h, None)
    assert abs(d["fwd5"].iloc[0] - 10.0) < 1e-9, "fwd 自 D+1 收盤起算 (D 起算會是 21%)"
    assert abs(d["fwd10_d0"].iloc[0] - 21.0) < 1e-9
    # 特徵無未來值：改動最後 40 日的資料，之前的事件遮罩不變 (分位型 ext_* / turnover_hot 為全歷史樣本內定義，另測)
    h1 = _hist(end="2026-10-13")
    _, d1 = M.market_view(h1, None)
    h2 = h1.copy()
    h2.loc[h2.index[-40:], "margin_amt"] *= 1.3
    h2.loc[h2.index[-40:], "close"] *= 0.8
    _, d2 = M.market_view(h2, None)
    m1, m2 = M._event_masks(d1), M._event_masks(d2)
    for k in ("streak_up7", "streak_dn7", "wipeout", "chase", "knife", "hi250", "lo250", "maint_lt150", "deleverage_fast"):
        a, b = m1[k].fillna(False).to_numpy()[:-40], m2[k].fillna(False).to_numpy()[:-40]
        assert (a == b).all(), k


def test_events_matching_bootstrap_labels():
    h = _hist(end="2026-10-13")
    _, d = M.market_view(h, None)
    masks = M._event_masks(d)
    ev_idx, ep, ctrl, keep = M._controls(d, masks["chase"].fillna(False).to_numpy())
    assert len(ev_idx) > 50 and len(keep) > 0 and len(ctrl) == len(keep)
    r20 = d["ret20"].to_numpy()
    years = pd.to_datetime(d["date"]).dt.year.to_numpy()
    ev_all = set(np.flatnonzero(masks["chase"].fillna(False).to_numpy()))
    near = {i + o for i in ev_all for o in range(-3, 4)}
    for j, cs in zip(keep, ctrl):
        i = ev_idx[j]
        assert 1 <= len(cs) <= M.MATCH_K
        assert all(abs(r20[c] - r20[i]) <= M.CALIPER_RET20 + 1e-9 for c in cs), "ret20 卡尺"
        assert all(abs(years[c] - years[i]) <= M.YEAR_DIFF for c in cs)
        assert not any(c in near for c in cs), "對照不在事件日 ±3 日"
    with patched(M, "CALIPER_RET20", 1e-6):
        _, _, _, keep2 = M._controls(d, masks["chase"].fillna(False).to_numpy())
    assert len(keep2) < len(keep), "卡尺收緊 → 找得到對照的事件變少"
    e1 = M.events(d, [])
    e2 = M.events(d, [])
    assert json.dumps(clean(e1), sort_keys=True) == json.dumps(clean(e2), sort_keys=True), "seed 固定可重現"
    assert set(e1["events"]) == set(M.EVENT_DEFS) and len(e1["events"]) == 12 and e1["tests_total"] == 36
    for k, v in e1["events"].items():
        for key in ("n", "episodes", "fwd5", "fwd10", "fwd20", "win10", "excess10", "ci_ep", "ci_year", "years_pos", "years_n", "smd_ret20", "last", "since_days", "label", "evidence", "text"):
            assert key in v, (k, key)
        assert v["label"] in M.LABEL_SET and v["label"] != "predictive"
        assert v["evidence"]["grade"] in ("none", "weak", "weak_upper_bound") and v["evidence"]["tests_total"] == 36
        if v["n_matched"] > 1:
            assert v["ci_ep"] is None or (v["ci_ep"][0] <= v["excess10"] <= v["ci_ep"][1])
            assert v["ci_year"] is None or len(v["ci_year"]) == 2
            assert 0 <= v["years_pos"] <= v["years_n"]
    assert e1["events"]["streak_up7"]["evidence"]["grade"] == "weak" and e1["events"]["wipeout"]["evidence"]["grade"] == "weak_upper_bound"
    assert e1["events"]["wipeout"]["label"] == "risk_descriptive" and e1["events"]["maint_lt150"]["label"] == "risk_descriptive"
    e3 = M.events(d, ["streak_up7"])
    assert e3["events"]["streak_up7"]["label"] == "predictive" and e3["events"]["streak_dn7"]["label"] == "descriptive"
    js = json.dumps(clean(e1), ensure_ascii=False)
    for bad in ("ev_maint_drop10_5d", "maint_proxy", "ema", "EMA", "沉澱", "清洗", "斷頭"):
        assert bad not in js, bad
    assert "2010" in e1["baseline"] and "all" in e1["baseline"] and e1["baseline"]["2010"]["fwd10"] is not None
    # 標籤只能由 validated 給 predictive
    assert M.validated_ids(Path(_TMP) / "nope.json") == []
    p = Path(_TMP) / "v.json"
    p.write_text(json.dumps({"validated": [{"id": "wipeout", "evidence": "x"}, "streak_up7"]}), encoding="utf-8")
    assert M.validated_ids(p) == ["wipeout", "streak_up7"]


def test_events_text_inactive_insufficient_no_hardcoded():
    """未成立事件 text 只給定義句 (不帶今日數值、不與 L 矛盾)；成立事件引用同物件計算值，不硬編碼研究數字；分位門檻樣本不足 → insufficient / null。"""
    h = _hist(end="2026-10-13")                                  # flows 只有最後 ~500 日 → turnover_hot 樣本不足；chg1 6,000+ 日 → ext_* 足夠
    view, d = M.market_view(h, None)
    ev = M.events(d, [])["events"]
    for eid, v in ev.items():
        assert "+1.4%" not in v["text"] and "+0.4~0.5" not in v["text"] and "2003~" not in v["text"], (eid, v["text"])
        if not v["active"]:
            assert "未成立" in v["text"] or v["insufficient"], (eid, v["text"])
            assert not re.search(r"連[增減] -?\d+ 日", v["text"]) and "億，為" not in v["text"] and "% 低於 150" not in v["text"] \
                and not re.search(r"第 \d+ 日", v["text"]), (eid, v["text"])
    th = ev["turnover_hot"]
    assert th["insufficient"] is True and th["n"] == 0 and th["fwd10"] is None and th["active"] is False and "樣本不足" in th["text"] and "2500" in th["text"]
    assert ev["ext_inc"]["insufficient"] is False and ev["ext_inc"]["n"] > 0
    assert M._quantile_insufficient(d) == {"turnover_hot"}
    # 流量全歷史都有 → turnover_hot 不再 insufficient
    h2 = h.copy()
    rng = np.random.default_rng(4)
    h2["buy_lots"], h2["sell_lots"], h2["repay_lots"] = 340000 + rng.normal(0, 30000, len(h2)), 330000 + rng.normal(0, 30000, len(h2)), 8000.0
    _, d2 = M.market_view(h2, None)
    assert M._quantile_insufficient(d2) == set()
    th2 = M.events(d2, [])["events"]["turnover_hot"]
    assert th2["insufficient"] is False and th2["n"] > 0 and "樣本不足" not in th2["text"]
    # 成立事件：streak_up7 / wipeout 文案引用同物件 excess / fwd10 / sub2010
    h3 = h.copy()
    a = h3["margin_amt"].to_numpy().copy()
    a[-8:] = a[-9] * (1 + 0.002 * np.arange(1, 9))             # 連增 8 日
    h3["margin_amt"] = a
    _, d3 = M.market_view(h3, None)
    e3 = M.events(d3, [])["events"]
    s7 = e3["streak_up7"]
    st3 = int(d3.iloc[-1]["streak"])
    assert st3 >= 8 and s7["active"] and f"融資連增 {st3} 日 (≥7)" in s7["text"] and f"{s7['excess5']:+.2f}" in s7["text"] and f"n={s7['n']}" in s7["text"]
    h4 = h.copy()
    c = h4["close"].to_numpy().copy(); a4 = h4["margin_amt"].to_numpy().copy()
    c[-1] = c[-21] * 0.9; a4[-1] = a4[-21] * 0.9
    h4["close"], h4["margin_amt"] = c, a4
    _, d4 = M.market_view(h4, None)
    e4 = M.events(d4, [])["events"]
    w = e4["wipeout"]
    assert w["active"] and f"{w['fwd10']:+.2f}%" in w["text"] and w["sub2010"] is not None and f"{w['sub2010']['fwd10']:+.2f}%" in w["text"] \
        and f"n={w['sub2010']['n']}" in w["text"] and w["sub2010"]["from"] == "2010-01-01" and 0 < w["sub2010"]["n"] <= w["n"]
    assert e4["knife"]["active"] is False and "第 " not in e4["knife"]["text"]
    for v in list(e3.values()) + list(e4.values()):
        assert not M.FORBIDDEN_RE.search(v["text"]), v["text"]


# ------------------------------------------------------------------ (g) 個股
def test_restatements_and_stock_features():
    dates = _dates("2024-06-03", "2025-12-31")
    i_split = dates.index("2024-12-11")
    # 00632R 反分割：prev(D) = today(D−1) × 0.14 → capital，pct20 連續化後不會 −86%
    p = _code_panel("00632R", dates, lots0=600000.0, seed=5)
    p.loc[i_split, "m_prev"] = p.loc[i_split - 1, "m_today"] * 0.14
    p.loc[i_split:, "m_today"] = (p.loc[i_split:, "m_prev"] + p.loc[i_split:, "m_buy"] - p.loc[i_split:, "m_sell"] - p.loc[i_split:, "m_repay"]).cumsum() - \
        (p.loc[i_split:, "m_prev"].cumsum() - p.loc[i_split, "m_prev"]) if False else p.loc[i_split:, "m_today"]
    # 重建 i_split 之後的恆等式鏈
    lots = float(p.loc[i_split, "m_prev"])
    for i in range(i_split, len(p)):
        if i > i_split:
            p.loc[i, "m_prev"] = lots
        lots = p.loc[i, "m_prev"] + p.loc[i, "m_buy"] - p.loc[i, "m_sell"] - p.loc[i, "m_repay"]
        p.loc[i, "m_today"] = lots
    p.loc[i_split:, "close"] = p.loc[i_split:, "close"] / 0.14         # 價格同步 ×7.14
    rs = M.restatements(p)
    assert rs["kind"].iloc[i_split] == "capital" and abs(rs["f"].iloc[i_split] - 0.14) < 1e-9 and (rs["kind"].iloc[:i_split] == "").all()
    assert abs(rs["adj"].iloc[i_split - 1] - 0.14) < 1e-9 and rs["adj"].iloc[i_split] == 1.0, "連續化係數只套用在重述日之前"
    F = M.stock_features(p)
    assert abs(F["pct20"].iloc[i_split + 5]) < 20, F["pct20"].iloc[i_split + 5]
    assert abs(F["chg1"].iloc[i_split] - (p.loc[i_split, "m_today"] - p.loc[i_split, "m_prev"])) < 1e-9, "chg1 用 TWSE prev 欄"
    assert F["kind"].iloc[i_split] == "capital"
    assert np.isnan(F["ret20"].iloc[i_split + 5]) and np.isnan(F["ret5"].iloc[i_split + 2]) and F["quad20"].iloc[i_split + 5] == "", "股本事件日起 ret / quad 視窗重算"
    assert np.isfinite(F["ret20"].iloc[i_split + 20]) and abs(F["ret20"].iloc[i_split + 20]) < 60 and np.isfinite(F["ret5"].iloc[i_split + 5])
    assert np.isfinite(F["ret20"].iloc[i_split - 1]), "事件日之前不受影響"
    assert F["pnl_fifo"].iloc[-1] is not None and abs(F["pnl_fifo"].iloc[-1]) < 60, "FIFO 成本與價格同基準"
    # 00685L 24x
    q = _code_panel("00685L", dates, lots0=3600.0, seed=6)
    q.loc[i_split, "m_prev"] = q.loc[i_split - 1, "m_today"] * 24
    lots = float(q.loc[i_split, "m_prev"])
    for i in range(i_split, len(q)):
        if i > i_split:
            q.loc[i, "m_prev"] = lots
        lots = q.loc[i, "m_prev"] + q.loc[i, "m_buy"] - q.loc[i, "m_sell"] - q.loc[i, "m_repay"]
        q.loc[i, "m_today"] = lots
    rq = M.restatements(q)
    assert rq["kind"].iloc[i_split] == "capital" and abs(rq["f"].iloc[i_split] - 24) < 1e-9
    # adjust：|f−1| 3% 且限額未變 → adjust，不連續化 (|f−1| ≤ 0.5% 不分類)
    a = _code_panel("2330", dates, lots0=50000.0, seed=7)
    a.loc[i_split, "m_prev"] = a.loc[i_split - 1, "m_today"] * 1.03
    lots = float(a.loc[i_split, "m_prev"])
    for i in range(i_split, len(a)):
        if i > i_split:
            a.loc[i, "m_prev"] = lots
        lots = a.loc[i, "m_prev"] + a.loc[i, "m_buy"] - a.loc[i, "m_sell"] - a.loc[i, "m_repay"]
        a.loc[i, "m_today"] = lots
    ra = M.restatements(a)
    assert ra["kind"].iloc[i_split] == "adjust" and (ra["adj"] == 1.0).all()
    a0 = a.copy()
    a0.loc[i_split, "m_prev"] = a0.loc[i_split - 1, "m_today"] * 1.003
    assert M.restatements(a0)["kind"].iloc[i_split] == ""
    # 0.10 < |f−1| ≤ 0.4 且限額同日變 → capital；限額未變 → adjust
    b = a.copy()
    b.loc[i_split, "m_prev"] = b.loc[i_split - 1, "m_today"] * 1.25
    assert M.restatements(b)["kind"].iloc[i_split] == "adjust"
    b.loc[i_split:, "m_limit"] = 125000.0
    assert M.restatements(b)["kind"].iloc[i_split] == "capital"
    # 春節後首日 (2025-02-03，前一交易日 2025-01-24) 全面為負 → settlement 保留原值
    d2 = [d for d in dates if not ("2025-01-25" <= d <= "2025-02-02")]
    s = _code_panel("2603", d2, lots0=30000.0, seed=8)
    i_cny = d2.index("2025-02-03")
    s.loc[i_cny, "m_prev"] = s.loc[i_cny - 1, "m_today"] * 0.98
    lots = float(s.loc[i_cny, "m_prev"])
    for i in range(i_cny, len(s)):
        if i > i_cny:
            s.loc[i, "m_prev"] = lots
        lots = s.loc[i, "m_prev"] + s.loc[i, "m_buy"] - s.loc[i, "m_sell"] - s.loc[i, "m_repay"]
        s.loc[i, "m_today"] = lots
    rs2 = M.restatements(s)
    assert rs2["kind"].iloc[i_cny] == "settlement" and (rs2["adj"] == 1.0).all()
    Fs = M.stock_features(s)
    assert "restated" not in M._flags_at(Fs, i_cny) and M._restated_obj(Fs) is None
    assert "restated" in M._flags_at(M.stock_features(a), i_split + 3) and M._restated_obj(M.stock_features(a)) is None, "adjust 只進 flags"
    Fr = F.iloc[: i_split + 100]                                   # restated 物件只看最近 250 列內的 capital 事件
    assert M._restated_obj(Fr)["kind"] == "capital" and M._restated_obj(Fr)["date"] == "2024-12-11" and abs(M._restated_obj(Fr)["f"] - 0.14) < 1e-6
    assert M._restated_obj(F) is None, "超過 250 列的舊事件不再列出"


def test_stock_halted_recovering_base_fifo_pressure():
    dates = _dates("2024-06-03", "2025-12-31")
    halt = {d: "O" for d in dates[200:230]}
    zero = set(dates[200:230])
    p = _code_panel("1101", dates, lots0=30000.0, seed=9, note=halt, zero_lots=zero)
    F = M.stock_features(p)
    assert F["halted"].iloc[210] and np.isnan(F["util"].iloc[210]) and "halted" in M._flags_at(F, 210)
    assert not F["halted"].iloc[231]
    i_rec = 230                                                  # lots 0 → >0 的第一天
    assert F["lots"].iloc[229] == 0 and F["lots"].iloc[230] > 0
    assert F["recovering"].iloc[i_rec: i_rec + 40].all() and not F["recovering"].iloc[i_rec + 40]
    assert F["pct5"].iloc[i_rec: i_rec + 40].isna().all() and F["pct20"].iloc[i_rec: i_rec + 40].isna().all(), "停資恢復 40 日 pct_k 為 null"
    assert np.isfinite(F["pct20"].iloc[i_rec + 45])
    assert "recovering" in M._flags_at(F, i_rec + 10)
    # 基期 <50 張 → null
    small = _code_panel("9999", dates, lots0=30.0, seed=10)
    small["m_buy"] = small["m_sell"] = small["m_repay"] = 0.0
    small["m_today"] = small["m_prev"] = 30.0
    Fs = M.stock_features(small)
    assert Fs["pct5"].isna().all() and Fs["pct20"].isna().all() and (Fs["chg5"].dropna() == 0).all()
    # FIFO burn-in 120、除息扣股利
    g = _code_panel("2412", dates, lots0=40000.0, seed=11)
    F0 = M.stock_features(g)
    assert F0["cost_fifo"].iloc[:120].isna().all() and F0["cost_fifo"].iloc[120:].notna().all() and F0["burn_in"].iloc[119] and not F0["burn_in"].iloc[120]
    F1 = M.stock_features(g, None, [{"ex_date": dates[150], "cash": 5.0}])
    assert F1["cost_fifo"].iloc[150] < F0["cost_fifo"].iloc[150] - 0.5, "除息日成本扣現金股利"
    assert abs(F0["cost_fifo"].iloc[149] - F1["cost_fifo"].iloc[149]) < 1e-9
    assert F1["ret20"].iloc[160] != F0["ret20"].iloc[160], "報酬以除息調整"
    L = F0.iloc[-1]
    u = min(max(L["util"], 0), 100) / 100
    loss = min(max(-L["pnl_fifo"], 0), 30) / 30
    dec = min(max(-L["pct5"], 0), 15) / 15
    drop = min(max(-L["ret5"], 0), 15) / 15
    assert abs(L["pressure_score"] - (35 * u + 35 * loss + 15 * dec + 15 * drop)) < 1e-9 and not L["pressure_partial"]
    assert abs(L["maint_est"] - L["close"] / (L["cost_fifo"] * 0.6) * 100) < 1e-9
    assert abs(L["util"] - L["lots"] / 100000 * 100) < 1e-9
    assert abs(L["bal_pos250"] - (F0["lots_adj"].iloc[-251:-1] <= L["lots_adj"]).mean() * 100) < 1e-6
    # pressure_score：35% 分項 (pnl_fifo burn-in / util 停資) 缺 → null，不以 0 代入充當完整分數；15% 分項 (pct5 / ret5) 缺 → 以 0 計並標 pressure_partial
    assert F0["pressure_score"].iloc[:120].isna().all() and F0["pressure_score"].iloc[120:].notna().all()
    assert np.isnan(F["pressure_score"].iloc[210]), "停資 (util null) → null"
    assert F0["pressure_partial"].iloc[120:].sum() == 0
    gp = g.copy()
    gp.loc[gp.index[-1], "close"] = np.nan                      # 最後一列無收盤 → pnl NaN → null
    assert np.isnan(M.stock_features(gp)["pressure_score"].iloc[-1])
    Lq = F.iloc[240]                                              # 1101 停資恢復期 (230~269)：pct5 null → partial 且以 0 計；util / pnl 皆有 → 分數仍輸出
    assert np.isnan(Lq["pct5"]) and bool(Lq["pressure_partial"]) and np.isfinite(Lq["pressure_score"]) and np.isfinite(Lq["pnl_fifo"])
    assert abs(Lq["pressure_score"] - (35 * min(max(Lq["util"], 0), 100) / 100 + 35 * min(max(-Lq["pnl_fifo"], 0), 30) / 30 + 15 * min(max(-Lq["ret5"], 0), 15) / 15)) < 1e-9
    # stocks()：60 列切片、欄位、無收盤列不崩潰
    g2 = g.copy()
    g2.loc[g2.index[-3:], "close"] = np.nan
    out = M.stocks(pd.concat([g2, p]), ["2412", "1101", "0000"], None, {"2412": {"realized": [{"ex_date": dates[150], "cash": 5.0}]}})
    assert set(out) == {"2412", "1101"}
    r = out["2412"]
    assert r["cols"] == M.DAILY_COLS and len(r["daily"]) == 60 and all(len(x) == 9 for x in r["daily"])
    assert "no_close" in r["flags"] and r["ret20"] is None and r["mode"]["label"] == "descriptive" and r["quad"]["text"] in M.QUADS
    assert r["cost_fifo"] is not None and r["pressure_score"] is None and r["pressure_partial"] is False and r["restated"] is None, "無收盤 → pnl 缺 → 壓力分數 null"
    assert "融資壓力分數" not in r["mode"]["text"]
    r1 = out["1101"]
    assert r1["pressure_score"] is not None and "融資壓力分數" in r1["mode"]["text"]
    json.dumps(clean(out))


def test_turnover5_excludes_repay():
    """§2.2 個股 5 日周轉 = (buy+sell)/lots，不含現金償還 (與市場級 turnover_ratio 一致)。"""
    dates = _dates("2025-01-01", "2025-01-14")
    rows, lots = [], 10000.0
    for d in dates:
        buy, sell, rep = 100.0, 100.0, 50.0
        rows.append({"date": d, "code": "1", "name": "x", "note": "", "m_buy": buy, "m_sell": sell, "m_repay": rep, "m_prev": lots, "m_today": lots + buy - sell - rep,
                     "m_limit": 100000.0, "s_buy": 0, "s_sell": 0, "s_repay": 0, "s_prev": 0, "s_today": 0, "s_limit": 0, "offset": 0, "close": 100.0, "vol_lots": 1000.0})
        lots = lots + buy - sell - rep
    F = M.stock_features(pd.DataFrame(rows))
    L = F.iloc[-1]
    assert abs(L["turnover5"] - 5 * 200 / L["lots"] * 100) < 1e-9, L["turnover5"]
    assert abs(L["repay_share"] - 50 / 150 * 100) < 1e-9


# ------------------------------------------------------------------ (h) 排行
def test_ranking_thresholds_etf_snapshots_stats():
    dates = _dates("2025-01-02", "2025-12-31")
    frames = [_code_panel("2330", dates, lots0=50000.0, seed=1), _code_panel("2603", dates, lots0=800.0, seed=2),
              _code_panel("1111", dates, lots0=300.0, seed=3),                          # lots <500 → 不進 inc5 / dec5 / div20 / short
              _code_panel("0050", dates, lots0=20000.0, seed=4), _code_panel("00631L", dates, lots0=90000.0, seed=5, limit=100000.0)]  # ETF
    big = _code_panel("3443", dates, lots0=5000.0, seed=6)
    big.loc[big.index[-1], ["m_buy", "m_sell", "m_repay"]] = [2000.0, 0.0, 0.0]
    big.loc[big.index[-1], "m_today"] = big.loc[big.index[-1], "m_prev"] + 2000.0
    big.loc[big.index[-1], "s_today"] = big.loc[big.index[-1], "m_today"] * 0.25                      # 券資比 25%
    big.loc[big.index[-1], "close"] = np.nan                                                          # 無價格列
    frames.append(big)
    panel = pd.concat(frames, ignore_index=True)
    snaps = [(dates[-2], {"inc5": [{"code": "3443", "v": 1}], "dec5": [], "util": [{"code": "00631L", "v": 90}]}),
             (dates[-3], {"inc5": [{"code": "3443", "v": 1}], "util": []}), (dates[-4], {"inc5": [], "util": [{"code": "00631L", "v": 90}]})]
    rk = M.ranking(panel, None, snapshots=snaps, save=False)
    assert rk["date"] == dates[-1] and rk["n_days"] == 4
    codes = lambda b: [r["code"] for r in rk[b]]  # noqa: E731
    assert "3443" in codes("inc5") and "1111" not in codes("inc5") + codes("dec5") + codes("div20") + codes("short")
    assert not any(M._is_etf(c) for c in codes("inc5") + codes("dec5")), "ETF 不混排"
    assert set(codes("etf_inc5") + codes("etf_dec5")) <= {"0050", "00631L"}
    assert all(abs(r["chg5"]) >= 100 and r["lots"] >= 500 for r in rk["inc5"] + rk["dec5"])
    assert codes("util") == ["00631L"] and all(r["util"] >= 60 for r in rk["util"])
    assert "3443" in codes("short") and "2603" in codes("short") and "1111" not in codes("short") and all(r["v"] >= 20 for r in rk["short"])
    assert rk["short"][0]["v"] >= rk["short"][-1]["v"]
    r3443 = next(r for r in rk["inc5"] if r["code"] == "3443")
    assert r3443["days_on"] == 3 and r3443["cum_on"] == 3 and r3443["ret5"] is None and "n/a" in r3443["text"] and "已連續上榜 3 日" in r3443["text"]
    u = rk["util"][0]
    assert u["days_on"] == 2 and u["cum_on"] == 3, (u["days_on"], u["cum_on"])
    assert "偏弱" not in u["text"] and "限額" in u["text"]
    assert rk["stats"]["universe"]["basis"].startswith("同日等權宇宙") and rk["stats"]["inc5"]["universe_ret5_eqw"] == rk["stats"]["universe"]["ret5_eqw"]
    assert all(len(rk[b]) <= 15 for b in ("inc5", "dec5", "etf_inc5", "etf_dec5", "util", "pressure", "div20", "offset", "short"))
    assert rk["label"]["util"] == "risk_descriptive" and rk["label"]["pressure"] == "risk_descriptive"
    # 門檻輸出 (壓力榜 lots ≥500 為 SPEC §2.3 之外的門檻，於 thresholds 註明)；壓力榜只列 pressure_score 非 null 的檔 (3443 無收盤 → pnl 缺 → 不列)
    assert rk["thresholds"] == M.RANK_THRESHOLDS and rk["thresholds"]["pressure_lots_min"] == 500 and rk["thresholds"]["lots_min"] == 500
    assert "3443" not in codes("pressure") and "1111" not in codes("pressure") and all(r["v"] is not None and r["lots"] >= 500 for r in rk["pressure"])
    assert all("pressure_partial" in r and "maint_est" in r for r in rk["pressure"]) and len(rk["pressure"]) >= 1
    # 沒有快照 → days_on = 1 (今日)
    rk2 = M.ranking(panel, None, snapshots=[], save=False)
    assert all(r["days_on"] == 1 and r["cum_on"] == 1 for r in rk2["inc5"])
    json.dumps(clean(rk))


# ------------------------------------------------------------------ (i) 文案
def test_copy_forbidden_words_labels_alerts():
    h = _hist(end="2026-10-13")
    view, d = M.market_view(h, None)
    ev = M.events(d, [])
    view["events"], view["baseline"] = ev["events"], ev["baseline"]
    dates = _dates("2025-01-02", "2025-12-31")
    panel = pd.concat([_code_panel("2330", dates, lots0=50000.0, seed=1), _code_panel("3443", dates, lots0=5000.0, seed=6, limit=6000.0),
                       _code_panel("00631L", dates, lots0=90000.0, seed=5)])
    snaps = [(dates[-k], {"inc5": [{"code": "2330", "v": 1}, {"code": "3443", "v": 1}], "dec5": [{"code": "2330", "v": 1}]}) for k in (2, 3, 4)]
    rk = M.ranking(panel, None, snapshots=snaps, save=False)
    stk = M.stocks(panel, ["2330", "3443", "00631L"])
    stk["3443"]["flags"].append("halted"); stk["3443"]["maint_est"] = 120
    stk["2330"]["restated"] = {"date": stk["2330"]["date"], "f": 0.14, "kind": "capital"}
    view["streak"], view["z1"], view["chg1"] = 8, 2.5, 120.0
    view["quad"]["days"] = 1
    view["maint"]["value"], view["maint"]["real"], view["maint"]["check"] = 145.0, 147.0, True
    view["cross"]["br_net_z60"], view["cross"]["usage_gt80_n"] = -2.4, 12
    view["pct250"], view["dhi250"] = 100, 0.0
    al = M.alerts(view, rk, stk)
    assert 1 <= len(al) <= 12
    texts = al + [view["verdict"]["text"], view["mode"]["text"], view["regime"]["text"]] + [v["text"] for v in ev["events"].values()] + \
        [r["text"] for b in ("inc5", "dec5", "etf_inc5", "etf_dec5", "util", "pressure", "div20", "offset", "short") for r in rk[b]] + \
        [s["mode"]["text"] for s in stk.values()] + list(M.LABELS.values()) + [v["evidence"]["note"] or "" for v in ev["events"].values()] + \
        [v["evidence"]["placebo"] or "" for v in ev["events"].values()] + [ev.get("note") or ""]
    for t in texts:
        assert not M.FORBIDDEN_RE.search(t), t
    assert all(("(描述)" in a or "(風險描述)" in a) for a in al)
    assert view["verdict"]["label"] in M.LABEL_SET and view["mode"]["label"] in M.LABEL_SET and view["regime"]["label"] in M.LABEL_SET
    for v in ev["events"].values():
        for key in ("n", "ci_ep", "ci_year", "years_pos", "smd_ret20", "label"):
            assert key in v
        assert v["evidence"]["grade"] in ("none", "weak", "weak_upper_bound") and v["label"] in M.LABEL_SET
    assert all(lab in M.LABELS for lab in M.LABEL_SET) and set(M.LABEL_SET) == {"predictive", "descriptive", "risk_descriptive"}
    src = (ROOT / "chip" / "analysis" / "margin_dyn.py").read_text(encoding="utf-8")
    body = src.split("FORBIDDEN_RE = re.compile(")[1].split("\n", 2)[2]       # 排除 regex 定義本身
    for w in ("斷頭", "散戶", "追價過熱", "籌碼沉澱", "護盤", "偏弱", "偏強", "買進", "賣出", "抄底", "逃頂", "停損", "持股水位", "續抱", "試單", "承接", "進場", "放空", "做多", "做空"):
        assert w not in body, f"原始碼文案含禁用字 {w}"
    assert "yahoo" not in src.lower() or "絕不用 Yahoo" in src


# ------------------------------------------------------------------ (j) build / size_guard
def test_build_fast_offline_and_full_carry():
    _wipe_db()
    prev = {"meta": {"latest_trading": "2026-10-13", "publish_log": [{"date": "2026-10-13", "first_seen": "x"}]}, "market": {"date": "2026-10-13", "amt": 1.0},
            "history": {"cols": M.HISTORY_COLS, "rows": [["2026-10-13", 6449.8, 9430485, 225459, None, None]]}, "ranking": {"date": "2026-10-13", "n_days": 3, "inc5": [{"code": "x"}]},
            "stocks": {"2330": {"name": "台積電"}}, "alerts": ["舊的"]}
    with offline() as calls:
        out = M.build(None, None, None, prev=prev, fast=True)
    assert calls == [], "fast 不連網"
    assert out["meta"]["mode"] == "fast" and out["meta"]["carried"] is True and set(out["meta"]["carried_parts"]) == set(M.PARTS)
    assert out["market"] == prev["market"] and out["ranking"] == prev["ranking"] and out["history"]["cols"] == M.HISTORY_COLS
    assert out["meta"]["labels"] == M.LABELS and out["meta"]["validated"] == [] and out["meta"]["tests_total"]["module_events"] == 36
    json.dumps(clean(out))
    # fast 沒有 prev → 骨架
    with offline():
        out0 = M.build(None, None, None, prev={}, fast=True)
    assert out0["meta"]["carried"] and out0["history"]["cols"] == M.HISTORY_COLS and out0["stocks"] == {}
    # full：面板 / 排行失敗 → 沿用 prev 對應區塊；市場段正常
    h = _hist(end="2026-10-13")
    scored = pd.DataFrame({"date": h["date"].tail(400), "close": h["close"].tail(400), "amount": 3000.0, "margin_div20": 1.0, "f_margin": 0.0, "margin_pct20": 2.0})

    def boom(*a, **k):
        raise RuntimeError("ranking boom")
    wg = pd.DataFrame({"date": h["date"].tail(1250), "maint_ratio": 170.0 + np.arange(1250) % 7, "margin_amt": 1.0})   # 同事 wantgoo.maint_real 的欄名
    with offline(), patched(M, "_fetch_total_flows", lambda: h[["date", "margin_amt", "margin_lots", "short_lots", "buy_lots", "sell_lots", "repay_lots"]]), \
            patched(M, "_fetch_taiex", lambda: h[["date", "close", "amount"]]), patched(M, "_fetch_holidays", lambda y: set()), patched(M, "ranking", boom), \
            patched(M, "_watch_ids", lambda: ["2330"]), patched(M, "_fetch_maint_real", lambda: wg):
        out = M.build(scored, None, [{"cols": ["code"], "rows": [["3443"]]}], prev=prev, fast=False)
    assert out["meta"]["sources"]["wantgoo"] == {"date": "2026-10-13", "ok": True} and out["market"]["maint"]["real"] is not None
    assert out["market"]["maint"]["real_date"] == "2026-10-13" and out["meta"]["sources"]["maint"]["real_date"] == "2026-10-13"
    assert out["meta"]["mode"] == "full" and out["meta"]["carried"] is False
    assert "ranking" in out["meta"]["carried_parts"] and out["ranking"] == prev["ranking"]
    assert "stocks" in out["meta"]["carried_parts"] and out["stocks"] == prev["stocks"], out["meta"]["carried_parts"]
    assert "market" not in out["meta"]["carried_parts"] and out["market"]["date"] == "2026-10-13" and "events" in out["market"]
    assert out["history"]["cols"] == M.HISTORY_COLS and out["flows"]["cols"] == M.FLOW_COLS and out["derived"]["cols"] == M.DERIVED_COLS
    assert len(out["history"]["rows"]) == len(h) and len(out["derived"]["rows"]) == 400 and 0 < len(out["flows"]["rows"]) <= 520
    assert out["meta"]["sources"]["twse_all"]["ok"] is False and out["meta"]["sources"]["tpex"]["status"] == "fail"
    assert out["meta"]["coverage"]["history_days"] == len(h) and out["meta"]["coverage"]["panel_days"] == 0
    assert out["meta"]["publish_log"][-1]["date"] == "2026-10-13" and len(out["meta"]["publish_log"]) == 1, "publish_log 同日不重複"
    assert out["watch_ids"] == ["2330"] and out["pool_ids"] == ["3443"]
    js = json.dumps(clean(out), ensure_ascii=False)
    assert "NaN" not in js and "Infinity" not in js
    assert out["meta"]["size_guard"]["steps"] == []
    # Tier-1 離線重建：上一輪 full 已把餘額 / 大盤收盤持久化到 daily_metrics → 全部來源失敗仍可從 SQLite ∪ prev 算出 market
    with offline():
        out1 = M.build(scored, None, None, prev=prev, fast=False)
    assert "market" not in out1["meta"]["carried_parts"] and out1["market"]["date"] == "2026-10-13" and len(out1["history"]["rows"]) == len(h)
    # SQLite 也沒有 → 全部段落失敗 → 沿用 prev
    _wipe_db()
    with offline():
        out2 = M.build(scored, None, None, prev=prev, fast=False)
    assert set(out2["meta"]["carried_parts"]) >= {"market", "history", "ranking", "stocks"} and out2["market"] == prev["market"]


def test_size_guard_three_levels():
    def big_out(n_hist=6000, n_stocks=100, n_derived=400):
        return {"meta": {}, "watch_ids": ["2330"], "pool_ids": [f"{i:04d}" for i in range(1, n_stocks)],
                "history": {"cols": M.HISTORY_COLS, "rows": [[f"{2001 + i // 250}-{1 + (i // 21) % 12:02d}-{1 + i % 21:02d}", 1234.5, 1, 1, None, None] for i in range(n_hist)]},
                "derived": {"cols": M.DERIVED_COLS, "rows": [[f"2025-01-{1 + i % 28:02d}"] + [1.0] * 9 for i in range(n_derived)]},
                "stocks": {**{"2330": {"daily": [[1] * 9] * 60}}, **{f"{i:04d}": {"daily": [[1] * 9] * 60} for i in range(1, n_stocks)}}}
    o = M.size_guard(big_out(), limit=10 ** 9)
    assert o["meta"]["size_guard"]["steps"] == []
    size0 = o["meta"]["size_guard"]["bytes"]
    o = M.size_guard(big_out(), limit=size0 - 1)
    assert o["meta"]["size_guard"]["steps"][0] == "history_2010" and all(r[0] >= "2010-01-01" for r in o["history"]["rows"])
    size1 = o["meta"]["size_guard"]["bytes"]
    o = M.size_guard(big_out(), limit=size1 - 1)
    assert o["meta"]["size_guard"]["steps"][:2] == ["history_2010", "stocks_60"] and len(o["stocks"]) == 60 and "2330" in o["stocks"] and len(o["pool_ids"]) == 59
    size2 = o["meta"]["size_guard"]["bytes"]
    o = M.size_guard(big_out(), limit=size2 - 1)
    assert o["meta"]["size_guard"]["steps"] == ["history_2010", "stocks_60", "derived_250"] and len(o["derived"]["rows"]) == 250
    assert o["history"]["cols"] == M.HISTORY_COLS


# ------------------------------------------------------------------ (l) backfill
def test_backfill_budget_skip_nodata_resume_order():
    _wipe_db()
    cal = _dates("2026-09-01", "2026-10-13")
    hol = {"2026-09-28", "2026-10-09"}
    rows_for = lambda d: [{"code": "2330", "name": "x", "m_buy": 1, "m_sell": 1, "m_repay": 0, "m_prev": 100, "m_today": 100, "m_limit": 1000, "s_buy": 0, "s_sell": 0, "s_repay": 0,  # noqa: E731
                           "s_prev": 0, "s_today": 0, "s_limit": 10, "offset": 0, "note": ""}]
    calls = {"all": [], "close": [], "tpex": []}

    def fake_all(ymd):
        calls["all"].append(ymd)
        d = M._iso(ymd)
        if d == "2026-10-02":
            raise M.NoData("沒有符合")
        return {"date": d, "market": [["融資(交易單位)", "1", "1", "0", "100", "100"], ["融券(交易單位)", "0", "0", "0", "0", "0"], ["融資金額(仟元)", "100", "100", "0", "1000000", "1000000"]],
                "stocks": [["2330", "台積電", "1", "1", "0", "100", "100", "1000", "0", "0", "0", "0", "0", "10", "0", ""]]}

    def fake_close(ymd):
        calls["close"].append(ymd)
        return {"2330": (1000.0, 5.0)}

    def fake_tpex(ymd):
        calls["tpex"].append(ymd)
        raise M.NotYet("0 列")
    dropped = []
    for d in cal[:10]:                      # 已有 10 日 (無收盤)
        if d not in hol:
            M._panel_upsert(d, rows_for(d), None)
    with patched(M, "_fetch_margin_all", fake_all), patched(M, "_fetch_closes", fake_close), patched(M, "_fetch_tpex", fake_tpex), patched(M, "_fetch_holidays", lambda y: hol), \
            patched(M, "cache_delete", dropped.append):
        rep = M.backfill(5, today="2026-10-13", panel_from="2026-08-20")
        assert rep["done"] + rep["nodata"] == 5 and len(calls["all"]) == 5 and len(calls["all"]) + len(calls["close"]) + len(calls["tpex"]) <= 15
        assert all(M._iso(x) not in hol for x in calls["all"]) and not any(M._iso(x) < "2026-08-20" for x in calls["all"])
        have = set(M._panel_dates())
        assert not any(M._iso(x) in set(cal[:10]) for x in calls["all"]), "已有日期跳過"
        first = sorted(M._iso(x) for x in calls["all"])
        assert first[0] > cal[9], "forward (最近缺日) 先於 back"
        # 個股列入 SQLite 後即刪整份原始快取 (TWSE ALL；TPEx 這裡 NotYet 未入庫不刪)
        assert sorted(dropped) == sorted(f"twse:MI_MARGN:ALL:{M._ymd(d)}" for d in M._panel_dates() if d > cal[9]), dropped
        assert "timeout" not in rep
        rep2 = M.backfill(40, today="2026-10-13", panel_from="2026-08-20")
        assert rep2["pending"] == 0 and "2026-10-02" in M._nodata("twse_all") and "2026-10-02" not in have
        n_all = len(calls["all"])
        rep3 = M.backfill(40, today="2026-10-13", panel_from="2026-08-20")
        assert len(calls["all"]) == n_all, "NoData / 已有日期不重抓"
        assert rep2["closes"] + rep3["closes"] > rep2["done"] and rep3["pending_closes"] == 0, "種子日 (無收盤) 由 closes 階段補 MI_INDEX"
        assert M._panel_dates_without_close() == []
        # 例外中斷 → 可重入
        _wipe_db()
        state = {"n": 0}

        def flaky(ymd):
            state["n"] += 1
            if state["n"] == 3:
                raise RuntimeError("timeout")
            return fake_all(ymd)
        with patched(M, "_fetch_margin_all", flaky):
            r = M.backfill(10, today="2026-10-13", panel_from="2026-09-20")
            assert r["failed"] == 1 and r["done"] == 2 and r["pending"] > 0
            r = M.backfill(40, today="2026-10-13", panel_from="2026-09-20")
            assert r["failed"] == 0 and r["pending"] == 0
        # 冷啟動 (面板空)：全部缺日由新往舊，第一個抓的是最新日 (SPEC §3.2/§3.4 先補最近缺日再往回；雲端沒有本機 491 日快取)
        _wipe_db()
        calls["all"].clear()
        r = M.backfill(3, today="2026-10-13", panel_from="2026-08-20")
        got = [M._iso(x) for x in calls["all"]]
        assert got == ["2026-10-13", "2026-10-12", "2026-10-08"] and r["done"] == 3 and r["pending"] > 0, got     # 10-09 休市
        assert M._panel_dates() == ["2026-10-08", "2026-10-12", "2026-10-13"]
        r = M.backfill(2, today="2026-10-13", panel_from="2026-08-20")
        assert [M._iso(x) for x in calls["all"]][3:] == ["2026-10-07", "2026-10-06"], "續補仍由新往舊"
        # 時間預算：max_seconds 到 → 停止、剩下回 pending、rep.timeout=True (下次續補)
        calls["all"].clear()
        r = M.backfill(40, today="2026-10-13", panel_from="2026-08-20", max_seconds=0)
        assert r.get("timeout") is True and r["done"] == 0 and calls["all"] == [] and r["pending"] > 0, r
        r = M.backfill(40, today="2026-10-13", panel_from="2026-08-20", max_seconds=600)
        assert "timeout" not in r and r["pending"] == 0
    # 日曆：trading_days 用 TWSE 休市表 + 週末
    with patched(M, "_fetch_holidays", lambda y: hol):
        td = M.trading_days("2026-09-25", "2026-10-12")
    assert "2026-09-28" not in td and "2026-10-09" not in td and "2026-09-26" not in td and "2026-10-12" in td
    _wipe_db()


def test_backfill_closes_no_maint_then_recompute_ffill_and_partial_redo():
    """(l) run_closes 由新往舊補收盤時不算維持率 (ffill 無效)；build 的 _recompute_maint 升冪重算：兩檔其一在 D 無收盤但 D−1 有 → D 的 maint 等於以 D−1 價補算的值；
    coverage<100 / partial 的日子會重算 (補到更早收盤後自動修正)；完全無價檔以之後的收盤估市值判 partial。"""
    _wipe_db()
    cal = _dates("2026-09-01", "2026-09-30")
    hol = set()
    px = {"2330": 1000.0, "2317": 100.0}
    lots = {"2330": 1000.0, "2317": 2000.0}
    D = cal[10]
    for d in cal:
        rows = [{"code": c, "name": c, "m_buy": 0, "m_sell": 0, "m_repay": 0, "m_prev": lots[c], "m_today": lots[c], "m_limit": 100000, "s_buy": 0, "s_sell": 0,
                 "s_repay": 0, "s_prev": 0, "s_today": 0, "s_limit": 10, "offset": 0, "note": ""} for c in px]
        M._panel_upsert(d, rows, None)                                   # 種子：無收盤
    amt_yi = (1000 * 1000 * 1000 + 2000 * 100 * 1000) / 1e8 / 1.6          # 維持率 160%
    for d in cal:
        store.upsert_metrics(d, {"mg_amt": amt_yi})
    closes_calls = []

    def fake_close(ymd):
        d = M._iso(ymd); closes_calls.append(d)
        out = {c: (px[c], 10.0) for c in px}
        if d == D:
            out.pop("2317")                                              # D 日 2317 無成交 → 無收盤
        return out
    with patched(M, "_fetch_closes", fake_close), patched(M, "_fetch_margin_all", lambda y: (_ for _ in ()).throw(RuntimeError("no"))), \
            patched(M, "_fetch_tpex", lambda y: None), patched(M, "_fetch_holidays", lambda y: hol):
        rep = M.backfill(100, today=cal[-1], panel_from=cal[0])
    assert rep["pending_closes"] == 0 and closes_calls[0] == cal[-1] and closes_calls[-1] == cal[0], "由新往舊補收盤"
    got = store.load_metrics(["maint_twse", "maint_cov"])
    assert got.empty or got["maint_twse"].notna().sum() == 0, "run_closes 不算維持率 (交給 _recompute_maint)"
    hist = pd.DataFrame({"date": cal, "margin_amt": amt_yi})
    n = M._recompute_maint(hist)
    assert n == len(cal)
    got = store.load_metrics(["maint_twse", "maint_cov"]).set_index("date")
    assert abs(got.loc[D, "maint_twse"] - 160.0) < 1e-6, "D 日 2317 以 D−1 收盤補價"
    assert got.loc[D, "maint_cov"] == -100.0, "補價檔貢獻 >0.5 pt → partial (負值)"
    assert (got.drop(index=D)["maint_cov"] == 100.0).all() and (abs(got["maint_twse"] - 160.0) < 1e-6).all()
    # 第二次：cov=100 的日子不重算、partial 的 D 會重算 (仍 partial → 值不變)
    n2 = M._recompute_maint(hist)
    assert n2 == 1
    # 無價且無之前收盤 (面板第一天 2317 無收盤) → 用之後收盤估市值 → 貢獻大 → partial；值 = 只含 2330 的部分
    with store.conn() as c:
        c.execute("UPDATE margin_stock_daily SET close=NULL WHERE date=? AND code='2317'", (cal[0],))
        c.execute("DELETE FROM daily_metrics WHERE date=? AND key IN ('maint_twse','maint_cov')", (cal[0],))
    assert M._recompute_maint(hist) == 2                               # cal[0] (無 maint) + D (partial)
    got = store.load_metrics(["maint_twse", "maint_cov"]).set_index("date")
    assert got.loc[cal[0], "maint_cov"] < 0 and abs(got.loc[cal[0], "maint_twse"] - 160.0 * 1000 * 1000 / (1000 * 1000 + 2000 * 100)) < 0.01, got.loc[cal[0]]
    _wipe_db()


if __name__ == "__main__":
    fails = 0
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            n += 1
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'} ({n} tests)")
    sys.exit(1 if fails else 0)
