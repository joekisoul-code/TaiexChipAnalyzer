"""個股回落模型 r2m 整合單元測試 (final_spec §3.2 B2、§6.1 3a 除息、§6.2 3b 夜盤、§7 3c IV 映射)。
不需網路、不寫 data/：大盤/個股價格用合成資料、模型用假預測器 (patch model.load / load_json)、交易日曆用規則函式；
00685L / 00631L 的實際資料檢查只讀專案快取檔 (找不到就略過)。

    python tests/test_stock_pullback_r2m.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

from chip.predict import events as E  # noqa: E402
from chip.predict import stock_pullback as SP  # noqa: E402
from chip.sources import exdiv as XD  # noqa: E402
from chip.sources import twse  # noqa: E402

HOL = {"2026-09-28", "2026-10-09", "2026-10-10"}      # 教師節補假 / 國慶 (合成日曆)


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _ntd(d0: str, n: int = 3, hol=HOL) -> list[str]:
    d, out = dt.date.fromisoformat(d0), []
    while len(out) < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in hol:
            out.append(d.isoformat())
    return out


def _dates(end="2026-09-25", n=420, hol=HOL) -> list[str]:
    d, out = dt.date.fromisoformat(end), []
    while len(out) < n:
        if d.weekday() < 5 and d.isoformat() not in hol:
            out.append(d.isoformat())
        d -= dt.timedelta(days=1)
    return out[::-1]


def _market(dates, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, len(dates)); r[0] = 0
    c = 20000 * np.exp(np.cumsum(r))
    midx = pd.DataFrame({"date": dates, "open": c * (1 - 0.001), "high": c * 1.006, "low": c * 0.994, "close": c, "volume": 1e9})
    cc = pd.Series(c)
    env = pd.DataFrame({"date": dates, "m_ret1": cc.pct_change() * 100, "m_bias20": (cc / cc.rolling(20).mean() - 1) * 100})
    return env, midx, r


def _stock(dates, mret, beta=1.2, idio=0.01, seed=1, split=None, zero=()):
    """FinMind 原始語意：分割前價格為舊尺度；change = 收盤 − 參考價 (分割/除息日的參考價已調整)。"""
    rng = np.random.default_rng(seed)
    r = beta * mret + rng.normal(0, idio, len(dates)); r[0] = 0
    c = 100 * np.exp(np.cumsum(r))
    ref = np.r_[c[0], c[:-1]]
    if split:
        i = dates.index(split[0])
        c[:i] = c[:i] * split[1]
        ref = np.r_[c[0], c[:-1]]
        ref[i] = c[i - 1] / split[1]
    p = pd.DataFrame({"date": dates, "open": c * 0.998, "high": c * 1.01, "low": c * 0.99, "close": c, "change": c - ref, "volume": 1e6})
    for d in zero:
        p.loc[p["date"] == d, ["open", "high", "low", "close", "change"]] = 0.0
    return p


class Lin:
    """假預測器：const + Σ coef·X[f]。"""
    def __init__(self, const, coef=None):
        self.const, self.coef = const, coef or {}

    def predict(self, X):
        v = np.full(len(X), float(self.const))
        for f, w in self.coef.items():
            v = v + w * X[f].astype(float).values
        return v


def _models(night_use=True, iv_use=True, iv_file=True):
    st = {"k": {str(k): {"use_model": True, "improve_model": 0.03} for k in SP.KS}, "hold3": {}, "supports": {},
          "night": {"aligned": True, "k": {str(k): {"use_model": night_use, "improve_pinball": 0.08, "by_stock": {"TEST": {"n": 1300, "touch20": 0.145}}} for k in SP.KS},
                    "formula": {str(k): {"b": 0.9 + 0.05 * k, "m20": -0.9 * k, "m10": -1.2 * k, "oos": {"improve_vs_base": 0.07, "touch20": 0.2}} for k in SP.KS}, "clip": 8.0},
          "ivmap": {str(k): {"use_iv": iv_use, "improve_iv": 0.003} for k in SP.KS}}
    pk = {}
    for k in SP.KS:
        pk[f"stock_pullback_k{k}"] = {"buy": [Lin(-0.8 * k)], "stop": [Lin(-1.2 * k)], "features": SP.FEATS}
        pk[f"stock_pullback_k{k}_night"] = {"buy": [Lin(-0.8 * k, {"bnight": 0.9})], "stop": [Lin(-1.2 * k, {"bnight": 0.9})], "features": SP.NIGHT_FEATS}
        if iv_file:
            pk[f"stock_pullback_iv_k{k}"] = {"buy": [Lin(-0.8 * k, {"ivr": 1.0})], "stop": [Lin(-1.2 * k, {"ivr": 1.0})], "features": SP.IV_FEATS}
    return st, pk


@contextlib.contextmanager
def env_ctx(st, pk, dates, seed=0, ivk_hist=()):
    env, midx, mret = _market(dates, seed)
    with patched(SP.M, "load_json", lambda n: st if n == "stock_pullback" else None), \
            patched(SP.M, "load", lambda n: pk.get(n)), \
            patched(SP, "_market_env", lambda: (env, midx)), \
            patched(twse, "next_trading_days", _ntd), \
            patched(E, "n_us_between", lambda a, b: 2 if "2026-09-28" in _between(a, b) else 1), \
            patched(SP, "_IVK_HIST", list(ivk_hist)):
        yield env, midx, mret


def _between(a, b):
    d, out = dt.date.fromisoformat(a), []
    while d < dt.date.fromisoformat(b):
        out.append(d.isoformat()); d += dt.timedelta(days=1)
    return out


def _run(price, **kw):
    kw.setdefault("exdiv_ev", {})
    return SP.build("TEST", price=price, **kw)


# ------------------------------------------------------------------ B2 價格清洗
def test_clean_price_zero_rows_and_split():
    ds = _dates()
    env, midx, mret = _market(ds)
    raw = _stock(ds, mret, split=(ds[300], 25.0), zero=(ds[100], ds[101]))
    cl, rw, sp = SP._clean_price(raw, "XXXX")
    assert len(rw) == len(raw) and len(cl) == len(raw) - 2, (len(cl), len(raw))
    assert not (cl[["open", "high", "low", "close"]] <= 0).any().any()
    assert sp and sp[0]["date"] == ds[300] and sp[0]["factor"] == 25.0, sp
    r = cl["close"].pct_change().abs()
    assert r.max() < 0.2, r.max()                               # 分割日前後連續
    assert rw["close"].iloc[0] > cl["close"].iloc[0] * 20        # 原始 frame 保留未調整價 (ivmap 用)


def test_known_split_00631L():
    ds = _dates(end="2026-06-30", n=200)
    env, midx, mret = _market(ds)
    raw = _stock(ds, mret, split=("2026-03-31", 21.6))           # 相鄰比值推算為 22；KNOWN_SPLITS 指定 22
    _, _, sp = SP._clean_price(raw, "00631L")
    assert sp and sp[0]["factor"] == 22.0, sp


def _cached_price(sid: str, start: str) -> pd.DataFrame | None:
    """唯讀：直接讀專案 data/cache 的 FinMind 檔 (最近 30 天鍵)，不經 chip.http (不會連網、不寫檔)。"""
    base = ROOT / "data" / "cache"
    d0 = dt.date.today()
    for i in range(0, 31):
        key = f"finmind:data_id={sid}&dataset=TaiwanStockPrice&start_date={start}&d={(d0 - dt.timedelta(days=i)).isoformat()}"
        p = base / (hashlib.sha1(key.encode()).hexdigest() + ".json")
        if p.exists():
            df = pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))["data"])
            return df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume", "spread": "change"})
    return None


def test_real_00685L_cleaning():
    p = _cached_price("00685L", "2024-01-01")
    if p is None:
        print("   (略過：專案快取沒有 00685L)")
        return
    q, raw, sp = SP._clean_price(p, "00685L")
    assert any(s["date"] == "2026-07-07" and s["factor"] == 25.0 for s in sp), sp
    zr = int((raw[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    assert len(raw) - len(q) == zr
    i = int(q.index[q["date"] == "2026-07-07"][0])
    assert abs(q["close"].iloc[i] / q["close"].iloc[i - 1] - 1) < 0.2
    f = SP.features_from_ohlc(q[SP.OHLCV], None).iloc[-1]
    assert 5 < f["bias60"] < 20, f["bias60"]                     # 清洗後約 +11 (未清洗 −57.7)


def test_real_00631L_ivmap_beta():
    p = _cached_price("00631L", "2024-01-01")
    if p is None:
        print("   (略過：專案快取沒有 00631L)")
        return
    ds = sorted(p["date"].astype(str))
    tw = _cached_twii()
    if tw is None:
        print("   (略過：專案快取沒有 TAIEX)")
        return
    b = SP.ivmap(p, tw).dropna().iloc[-1]["beta"]
    assert abs(b - 2.0) <= 0.5, b
    q, _, _ = SP._clean_price(p, "00631L")
    q["change"] = p.set_index("date").loc[q["date"], "change"].values
    try:
        SP.ivmap(q, tw)
        raise AssertionError("分割調整後的 frame 應被偵測")
    except ValueError:
        pass
    assert ds


def _cached_twii():
    for start in ("2024-01-01", "2023-09-28", "2021-01-01", "2018-01-01", "2010-01-01"):
        df = _cached_price("TAIEX", start)
        if df is not None:
            return df[["date", "close"]]
    return None


# ------------------------------------------------------------------ 3c ivmap (合成)
def test_ivmap_raw_vs_adjusted():
    ds = _dates(n=500)
    env, midx, mret = _market(ds, seed=5)
    raw = _stock(ds, mret, beta=2.0, idio=0.004, split=(ds[250], 22.0), seed=7)
    m = SP.ivmap(raw, midx).dropna()
    assert abs(m["beta"].iloc[-1] - 2.0) < 0.3, m["beta"].iloc[-1]
    adj, _ = __import__("chip.analysis.chips", fromlist=["x"]).adjust_splits(raw.drop(columns=["change"]), "XXXX")
    adj["change"] = raw["change"].values                           # change 未縮放 = 規格陷阱
    try:
        SP.ivmap(adj, midx)
        raise AssertionError("已分割調整 frame 應 raise")
    except ValueError:
        pass
    try:
        SP.ivmap(raw.drop(columns=["change"]), midx)
        raise AssertionError("缺 change 欄應 raise")
    except ValueError:
        pass


# ------------------------------------------------------------------ 3a 除息
def _ev(ex, cash=1.0, cash_est=None, **kw):
    return {"upcoming": [{"ex_date": ex, "cash": cash, "cash_est": cash_est, "status": "confirmed" if cash else "date_only", **kw}], "realized": [], "sources": {"TWT48U": "ok"}}


def test_exdiv_k2_window():
    ds = _dates()
    st, pk = _models(night_use=False, iv_use=False)
    with env_ctx(st, pk, ds) as (env, midx, mret):
        p = _stock(ds, mret)
        close = float(p["close"].iloc[-1])
        cal = _ntd(ds[-1], 3)
        base = _run(p, exdiv_ev={})
        assert base["exdiv"]["status"] == "unknown"
        out = _run(p, exdiv_ev=_ev(cal[1], cash=close / 100))      # cash/close = 1% → k2/k3 −1.00
    for k in ("1", "2", "3"):
        d20 = base["k"][k]["low20_pct"] - out["k"][k]["low20_pct"]
        d10 = base["k"][k]["low10_pct"] - out["k"][k]["low10_pct"]
        exp = 0.0 if k == "1" else 1.0
        assert abs(d20 - exp) < 0.011 and abs(d10 - exp) < 0.011, (k, d20, d10)
    assert "exdiv_adj" not in out["k"]["1"]
    a = out["k"]["2"]["exdiv_adj"]
    assert a["ex_date"] == cal[1] and abs(a["pct"] - 1.0) < 1e-6 and a["est"] is False
    assert out["k"]["2"]["buy_model_tr"] == base["k"]["2"]["buy_model"]      # 含息口徑保留
    assert out["exdiv"]["status"] == "ok"


def test_exdiv_over_15pct_not_deducted():
    ds = _dates()
    st, pk = _models(night_use=False, iv_use=False)
    with env_ctx(st, pk, ds) as (env, midx, mret):
        p = _stock(ds, mret)
        close = float(p["close"].iloc[-1])
        base = _run(p, exdiv_ev={})
        out = _run(p, exdiv_ev=_ev(_ntd(ds[-1], 1)[0], cash=close * 0.2))
    for k in ("1", "2", "3"):
        assert out["k"][k]["low20_pct"] == base["k"][k]["low20_pct"] and "exdiv_adj" not in out["k"][k]


def test_exdiv_typhoon_shift():
    ds = _dates(end="2026-09-24")                                    # 週四；09-25 颱風休市 (日曆沒有)，下一交易日 09-29
    st, pk = _models(night_use=False, iv_use=False)
    hol = HOL | {"2026-09-25"}
    with env_ctx(st, pk, ds) as (env, midx, mret), patched(twse, "next_trading_days", lambda d, n=3: _ntd(d, n, hol)):
        p = _stock(ds, mret)
        close = float(p["close"].iloc[-1])
        out = _run(p, exdiv_ev=_ev("2026-09-25", cash=close / 100))
    a = out["k"]["1"]["exdiv_adj"]
    assert a["ex_date"] == "2026-09-29" and a["ex_date_announced"] == "2026-09-25", a


def test_exdiv_unknown_when_fetch_fails():
    ds = _dates()
    st, pk = _models(night_use=False, iv_use=False)

    def boom(*a, **k):
        raise RuntimeError("offline")
    with env_ctx(st, pk, ds) as (env, midx, mret), patched(XD, "load_all", boom):
        p = _stock(ds, mret)
        base = _run(p, exdiv_ev={})
        out = SP.build("TEST", price=p)                                # exdiv_ev=None → 自行抓 → 例外
    assert out["exdiv"]["status"] == "unknown"
    assert all(out["k"][k]["low20_pct"] == base["k"][k]["low20_pct"] for k in ("1", "2", "3"))


def test_exdiv_cash_est_only_k1():
    ds = _dates()
    st, pk = _models(night_use=False, iv_use=False)
    with env_ctx(st, pk, ds) as (env, midx, mret):
        p = _stock(ds, mret)
        close = float(p["close"].iloc[-1])
        base = _run(p, exdiv_ev={})
        out = _run(p, exdiv_ev=_ev(_ntd(ds[-1], 1)[0], cash=None, cash_est=close / 50))
    assert out["k"]["1"]["exdiv_adj"]["est"] is True and abs(base["k"]["1"]["low20_pct"] - out["k"]["1"]["low20_pct"] - 2.0) < 0.011
    for k in ("2", "3"):
        assert out["k"][k]["low20_pct"] == base["k"][k]["low20_pct"] and "exdiv_skip" in out["k"][k]


def test_export_asof_px_defined():
    import export_static as X
    slim = {"2330": {"date": "2026-09-25"}, "00631L": {"date": "2026-09-24"}, "BAD": {"error": "x"}, "N": {"date": None}}
    assert X.watchlist_asof(slim, "2026-09-01") == "2026-09-25"
    assert X.watchlist_asof({"BAD": {"error": "x"}}, "2026-09-01") == "2026-09-01"
    src = (ROOT / "tools" / "export_static.py").read_text(encoding="utf-8")
    i = src.index("asof_px = watchlist_asof(")
    assert i < src.index("_xd.load_all(_ok_ids, asof_px")                   # 用到前已定義 (不再 NameError)


def test_upcoming_market():
    def gj(url, params=None):
        if "TWT48U" in url:
            return {"stat": "OK", "data": [["115年09月29日", "2109", "x", "息", "", "", "", "0.5"], ["115年10月05日", "00939", "x", "息", "", "", "", "待公告"],
                                            ["115年09月20日", "1111", "x", "息", "", "", "", "1"]]}
        if "TWT49U" in url:
            return {"stat": "OK", "data": [["115年09月29日", "2109", "x", "50", "49.52", "0.48", "息"]]}
        raise RuntimeError("no")
    u = XD.upcoming_market("2026-09-25", gj, today="2026-09-27")
    assert u["2109"][0] == {"ex_date": "2026-09-29", "cash": 0.48, "est": False, "kind": "息"}, u["2109"]
    assert u["00939"][0]["cash"] is None and u["00939"][0]["est"] is True
    assert "1111" not in u


# ------------------------------------------------------------------ 3b 夜盤
def _night_panel(n=400, seed=11, misalign=False):
    ds = _dates(n=n)
    rng = np.random.default_rng(seed)
    night = rng.normal(0, 1.0, n)                   # night[i] = ds[i-1] 收盤後、ds[i] 開盤前
    c, o = np.empty(n), np.empty(n)
    x = 20000.0
    for i in range(n):
        oo = x * (1 + (0.8 * night[i] + rng.normal(0, 0.1)) / 100) if i else x
        x = oo * (1 + rng.normal(0, 0.5) / 100)
        o[i], c[i] = oo, x
    midx = pd.DataFrame({"date": ds, "open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c, "volume": 1e9})
    nh = pd.DataFrame({"date": ds[1:], "night_chg_pct": night[1:]})        # FinMind 語意
    panel = pd.DataFrame({"date": ds, "stock_id": "TAIEX", "open": o, "close": c, "beta250": 1.0})
    return panel, nh, midx, dict(zip(ds, night))


def test_attach_night_aligned_and_guard():
    panel, nh, midx, nmap = _night_panel()
    p = SP._attach_night(panel, night=nh, mkt=midx)
    ds = list(midx["date"])
    for i in range(len(ds) - 1):
        assert math.isclose(p["night_chg_pct"].iloc[i], nmap[ds[i + 1]]), i        # 列 D ← next_td(D) 的夜盤
    assert np.isnan(p["night_chg_pct"].iloc[-1])
    assert SP._night_guard(p) > 0.5
    bad = panel.merge(nh, on="date", how="left")                                    # date==D 錯位 (B1)
    bad["bnight"] = bad["night_chg_pct"]
    try:
        SP._night_guard(bad)
        raise AssertionError("date==D 錯位合併應被守門擋下")
    except ValueError:
        pass


def test_beta250_taiex_is_one():
    ds = _dates(n=300)
    env, midx, mret = _market(ds)
    f = SP.features_from_ohlc(midx[SP.OHLCV], env)
    assert np.allclose(f["beta250"].iloc[80:], 1.0), f["beta250"].iloc[80:].describe()
    f2 = SP.features_from_ohlc(midx[SP.OHLCV], None)
    assert (f2["beta250"] == 1.0).all()


def test_night_variant_gates():
    ds = _dates()
    st, pk = _models(night_use=True, iv_use=False)
    with env_ctx(st, pk, ds) as (env, midx, mret):
        p = _stock(ds, mret)
        close = float(p["close"].iloc[-1])
        nf = _ntd(ds[-1], 1)[0]                                            # 09-25 → 09-29 (09-28 休市)
        base = _run(p)
        ok = _run(p, night_ret=-2.0, night_for=nf)
        wrong = _run(p, night_ret=-2.0, night_for=_ntd(ds[-1], 2)[1])
        nan = _run(p, night_ret=float("nan"), night_for=nf)
    assert ok["variant"] == "night" and ok["night_for"] == nf and ok["night_ret"] == -2.0
    assert ok["night_note"].startswith("休市後首日")
    b = ok["beta250"]
    for k in ("1", "2", "3"):
        kk = ok["k"][k]
        assert kk["variant"] == "night"
        exp = (-0.8 * int(k) + 0.9 * b * -2.0) * ok["sigma"]
        assert abs(kk["low20_pct"] - round(exp, 2)) < 0.02, (kk["low20_pct"], exp)
        assert abs(kk["buy_model"] - round(close * (1 + kk["low20_pct"] / 100), 2)) < 0.02 * close / 100 + 0.01   # 以收盤 D 為基準
        assert kk["oos"]["stock"]["touch20"] == 0.145
    for o in (wrong, nan, base):
        assert o["variant"] == "base" and all(o["k"][k]["variant"] == "base" for k in ("1", "2", "3"))
        assert o["k"]["1"]["low20_pct"] == base["k"]["1"]["low20_pct"]
    assert "night_skip" in wrong and "night_skip" in nan
    st2, pk2 = _models(night_use=False, iv_use=False)
    with env_ctx(st2, pk2, ds) as (env, midx, mret):
        off = _run(_stock(ds, mret), night_ret=-2.0, night_for=nf)
    assert off["variant"] == "base" and off["k"]["1"]["low20_pct"] == base["k"]["1"]["low20_pct"]


def test_client_table_formula_matches_train():
    st, pk = _models()
    with patched(SP.M, "load_json", lambda n: st if n == "stock_pullback" else None):
        ct = SP.client_table()
    for k in ("1", "2", "3"):
        f = st["night"]["formula"][k]
        assert ct["night_formula"][k] == {"b": f["b"], "m20": f["m20"], "m10": f["m10"]}
    assert ct["oos"]["exdiv"]["pinball20_rel"] == -0.62 and ct["night_clip"] == 8.0
    # 閘門：k3 伺服器未採用 (use_model False) 或公式 F 自身未過門檻 → 不匯出公式，前端不可套
    st3 = json.loads(json.dumps(st)); st3["night"]["k"]["3"]["use_model"] = False; st3["night"]["formula"]["2"]["oos"] = {"improve_vs_base": 0.015, "touch20": 0.29}
    with patched(SP.M, "load_json", lambda n: st3 if n == "stock_pullback" else None):
        ct3 = SP.client_table()
    assert set(ct3["night_formula"]) == {"1"}, ct3["night_formula"]
    assert ct["oos"]["night"]["1"]["by_stock_touch20"] == {"TEST": 0.145}
    old = {"k": {}, "hold3": {}, "table": {}, "supports": {}}
    with patched(SP.M, "load_json", lambda n: old if n == "stock_pullback" else None):
        ct2 = SP.client_table()
    assert "night_formula" not in ct2 and "ivmap" not in ct2


def test_fast_patch():
    import export_static as X
    wl = {"stocks": {"2330": {"pullback": {"variant": "base", "x": 1}}, "BAD": {"error": "x"}}, "pullback_table": {"t": 1}}
    snap = json.dumps(wl, sort_keys=True)
    calls = []

    def fake_build(sid, **kw):
        calls.append((sid, kw))
        return {"variant": "night", "sid": sid}
    assert X.night_patch_watchlist(wl, None, "2026-09-29", build=fake_build) == 0
    assert json.dumps(wl, sort_keys=True) == snap and not calls              # 沒有完成的夜盤 → 不改
    assert X.night_patch_watchlist(wl, -1.2, "2026-09-29", build=lambda s, **k: {"variant": "base"}) == 0
    assert json.dumps(wl, sort_keys=True) == snap
    assert X.night_patch_watchlist(wl, -1.2, "2026-09-29", exdiv_map={}, build=fake_build) == 1
    assert wl["stocks"]["2330"]["pullback"]["variant"] == "night" and wl["pullback_table"] == {"t": 1}
    assert calls[0][1]["night_ret"] == -1.2 and calls[0][1]["exdiv_ev"] == {}


# ------------------------------------------------------------------ 3c IV 閘門
def test_iv_gate_and_fallbacks():
    ds = _dates(n=420)
    st, pk = _models(night_use=False, iv_use=True)
    with env_ctx(st, pk, ds) as (env, midx, mret):
        p = _stock(ds, mret, beta=1.5)
        d0 = ds[-1]
        live = {"date": d0, "ivk": {"1": 0.25, "2": 0.24, "3": 0.23}}
        feats = _run(p)                                                        # 沒有 ivk → FEATS 模型
        used = _run(p, ivk_live=live)
        lag1 = _run(p, ivk_live={**live, "date": ds[-2]})
        stale2 = _run(p, ivk_live={**live, "date": ds[-3]})
        fb = _run(p, ivk_live={**live, "calendar_fallback": True})
        short = _run(p.iloc[-150:].reset_index(drop=True), ivk_live=live)      # 報酬 <200 (00988A 早期)
    for k in ("1", "2", "3"):
        assert used["k"][k]["iv"]["used"] is True and lag1["k"][k]["iv"]["used"] is True and lag1["k"][k]["iv"]["stale_td"] == 1
        iv = used["k"][k]["iv"]
        exp = math.sqrt(iv["beta"] ** 2 * (float(live["ivk"][k]) / math.sqrt(252) * 100) ** 2 + iv["idio"] ** 2)
        assert abs(iv["sigma_ivmap"] - exp) < 0.01
        assert abs(used["k"][k]["low20_pct"] - round((-0.8 * int(k) + iv["ivr"]) * used["sigma"], 2)) < 0.02
        for o in (feats, stale2, fb):
            assert o["k"][k]["low20_pct"] == feats["k"][k]["low20_pct"] and not (o["k"][k].get("iv") or {}).get("used")
    assert short["k"]["1"]["iv"]["used"] is False
    st2, pk2 = _models(night_use=False, iv_use=False)
    with env_ctx(st2, pk2, ds) as (env, midx, mret):
        off = _run(_stock(ds, mret, beta=1.5), ivk_live={"date": ds[-1], "ivk": {"1": 0.25, "2": 0.24, "3": 0.23}})
        f0 = _run(_stock(ds, mret, beta=1.5))
    for k in ("1", "2", "3"):
        assert off["k"][k]["low20_pct"] == f0["k"][k]["low20_pct"] and off["k"][k]["buy_model"] == f0["k"][k]["buy_model"]
        assert off["k"][k]["iv"]["used"] is False and "use_iv=False" in off["k"][k]["iv"]["why"]


def test_untrained_json_is_base_fallback():
    """舊 stock_pullback.json (沒有 night/ivmap) → 與 FEATS 模型相同、沒有 iv 區塊、variant base。"""
    ds = _dates()
    st, pk = _models()
    st = {k: v for k, v in st.items() if k not in ("night", "ivmap")}
    with env_ctx(st, pk, ds) as (env, midx, mret):
        p = _stock(ds, mret)
        o = _run(p, night_ret=-1.0, night_for=_ntd(ds[-1], 1)[0], ivk_live={"date": ds[-1], "ivk": {"1": 0.2, "2": 0.2, "3": 0.2}})
    for k in ("1", "2", "3"):
        assert o["k"][k]["variant"] == "base" and "iv" not in o["k"][k]
        assert o["k"][k]["low20_pct"] == round(-0.8 * int(k) * o["sigma"], 2)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {n}: {e}")
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
