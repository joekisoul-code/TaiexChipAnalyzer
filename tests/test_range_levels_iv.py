"""range_levels TXO IV sigma 單元測試 (r2m final_spec §4.6)。不需網路：台股日曆用規則表、美股日曆用 NYSE 規則、價格與 ivk 用合成資料。

    python tests/test_range_levels_iv.py
    python -m pytest tests/test_range_levels_iv.py
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip.predict import events as E, learn, range_levels as RL  # noqa: E402
from chip.sources import twse  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="rl_iv_"))
NUM_KEYS = ("buy_at", "sell_at", "stop", "target", "level_lo", "level_hi", "path_low10", "path_low20", "path_high80", "path_high90",
            "range_sigma", "range_mode", "touch_prob", "event_range_factor")


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def offline():
    """台股日曆 = 規則表 (events 與 twse.next_trading_days 共用)；美股 = NYSE 規則；不載入 ^GSPC。"""
    E.reset_calendar_cache()
    twse._HOL_FAIL.clear()
    with patched(E, "tw_holidays", lambda y: (set(E.tw_rule_holidays(y)), "api" if y <= 2026 else "rule")), \
            patched(E, "_load_us_actual", lambda: None), \
            patched(twse, "holidays", lambda y=None: set(E.tw_rule_holidays(y))):
        yield
    E.reset_calendar_cache()


def _frame(end="2026-09-24", seed=5):
    with offline():
        days = E.tw_sessions("2016-06-01", end)
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.011, len(days))
    c = 10000 * np.exp(np.cumsum(r))
    o = c * np.exp(rng.normal(0, 0.003, len(days)))
    h = np.maximum(c, o) * np.exp(np.abs(rng.normal(0, 0.005, len(days))))
    l_ = np.minimum(c, o) * np.exp(-np.abs(rng.normal(0, 0.005, len(days))))
    return pd.DataFrame({"date": days, "open": o, "high": h, "low": l_, "close": c})


def _ivk_hist(fr: pd.DataFrame, seed=6):
    rng = np.random.default_rng(seed)
    rows = []
    for d in fr["date"]:
        if d < "2017-01-03":
            continue
        base = 0.011 * math.sqrt(252) * float(np.exp(rng.normal(0, 0.15)))
        rows.append({"date": d, "ivk": {"1": round(base * 1.05, 5), "2": round(base * 1.02, 5), "3": round(base, 5), "5": round(base, 5)}})
    return rows


def _night(fr: pd.DataFrame, seed=7):
    """FinMind 語意：date = 夜盤準備的隔一交易日 (2019~)。"""
    rng = np.random.default_rng(seed)
    ds = fr["date"][fr["date"] >= "2019-01-02"].tolist()
    return pd.DataFrame({"date": ds, "night_chg_pct": rng.normal(0, 0.8, len(ds))})


FR = _frame()
HIST = _ivk_hist(FR)
NIGHT = _night(FR)
FIT_IV = RL.fit_multipliers(FR, NIGHT, path=None, ivk_hist=HIST)
FIT_ATR = RL.fit_multipliers(FR, NIGHT, path=None)


def _params_path(name: str, mutate=None, iv: bool = True) -> Path:
    p = copy.deepcopy(FIT_IV if iv else FIT_ATR)
    if mutate:
        mutate(p)
    f = TMP / f"{name}.json"
    f.write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")
    RL._CACHE.pop(str(f), None)
    return f


P_IV = _params_path("iv")
P_ATR = _params_path("atr", iv=False)


def _nd(last: str):
    with offline():
        ds = twse.next_trading_days(last, 3)
    return [{"n": i + 1, "date": ds[i], "label": ["隔天", "後天", "第三天"][i]} for i in range(3)]


def _ivk_live(date: str, **kw):
    row = next(r for r in HIST if r["date"] == date)
    return {"date": date, "ivk": dict(row["ivk"]), "calendar_fallback": False, **kw}


def _attach(fr, path, ivk_live=None, night=None, sf=1.0, us_px=None):
    last = str(fr["date"].iloc[-1])
    nd = _nd(last)
    with offline(), patched(RL, "_night_final", lambda snap, nd_, scored: (night, None)):
        hdr = RL.attach_to_next_days(nd, fr, {"phase": "closed"}, float(fr["close"].iloc[-1]), path=path, ivk_live=ivk_live, sigma_factor=sf, us_px=us_px)
    return nd, hdr


def _same_levels(a: list[dict], b: list[dict]) -> None:
    for x, y in zip(a, b):
        for k in NUM_KEYS:
            assert x.get(k) == y.get(k), (x["n"], k, x.get(k), y.get(k))


# ------------------------------------------------------------------ 1. 各種無效輸入 → 完整 ATR 路徑，水準與現行逐位元相同
def test_invalid_inputs_fall_back_to_atr():
    fr = FR[FR["date"] <= "2026-09-22"]            # 一般日 (隔天 09-23，非休市後首日)
    ref, ref_hdr = _attach(fr, P_ATR, None)        # 現行：json 無 base_iv
    assert all(x["range_sigma_src"] == "atr_ewma" for x in ref)
    assert all("sigma 來源" not in x["range_note"] for x in ref)            # IV 未設定 → 文字與舊版相同
    assert "2014~ 樣本外逐年 15~29%" not in ref_hdr and "11~22%" not in ref_hdr
    # 與手算的 ATR 公式一致 (sigma_series × base 乘數)
    sig = float(RL.sigma_series(fr.tail(400)).iloc[-1])
    assert ref[0]["path_low20"] == round(FIT_ATR["base"]["1"]["low20"] * sig, 2)
    good = _ivk_live("2026-09-22")
    cases = {
        "empty": {}, "none": None, "no_date": {"ivk": good["ivk"]},
        "fallback_cal": {**good, "calendar_fallback": True},
        "stale2": _ivk_live("2026-09-18"),           # 09-18 → 09-22 隔 1 個交易日 (09-21) → 落後 2
        "future": _ivk_live("2026-09-23"),
    }
    for name, ivk in cases.items():
        nd, hdr = _attach(fr, P_IV, ivk)
        _same_levels(nd, ref)
        assert all(x["range_sigma_src"] == "atr_ewma" and x.get("range_sigma_fallback") for x in nd), name
        assert "IV 缺/過期 → ATR+EWMA" in hdr, (name, hdr)
    for name, bad in (("k1_none", None), ("k1_zero", 0), ("k1_neg", -0.2), ("k1_str", "x")):
        ivk = copy.deepcopy(good)
        ivk["ivk"]["1"] = bad
        nd, _ = _attach(fr, P_IV, ivk)
        _same_levels(nd[:1], ref[:1])
        assert nd[0]["range_sigma_src"] == "atr_ewma" and nd[1]["range_sigma_src"] == "txo_iv", name
    # json 沒有 base_iv / iv_enabled=false → ATR
    for path in (P_ATR, _params_path("off", lambda p: p["iv_monitor"].update(iv_enabled=False))):
        nd, _ = _attach(fr, path, good)
        _same_levels(nd, ref)
        assert all(x["range_sigma_src"] == "atr_ewma" for x in nd)


def test_stale_counts_trading_days_across_holidays():
    """2026-09-25 (中秋) / 09-28 (教師節) 休市：ivk 09-24 對資料日 09-29 只落後 1 個交易日 (日曆差 5 天) → 有效；09-23 → 落後 2 → 無效。"""
    p = RL.load_multipliers(P_IV)
    with offline():
        v1 = RL._ivk_valid({"date": "2026-09-24", "ivk": {"1": 0.2}}, "2026-09-29", p)
        v2 = RL._ivk_valid({"date": "2026-09-23", "ivk": {"1": 0.2}}, "2026-09-29", p)
        v0 = RL._ivk_valid({"date": "2026-09-29", "ivk": {"1": 0.2}}, "2026-09-29", p)
    assert v1["ok"] and v1["stale_td"] == 1
    assert not v2["ok"]
    assert v0["ok"] and v0["stale_td"] == 0
    assert RL._ivk_sigma(v1, 1) == 0.2 / math.sqrt(252) * 100
    assert RL._ivk_sigma(v1, 2) is None


# ------------------------------------------------------------------ 2. 不混用兩套 sigma
def test_no_mixing_and_iv_levels():
    fr = FR[FR["date"] <= "2026-09-22"]
    good = _ivk_live("2026-09-22")
    nd, hdr = _attach(fr, P_IV, good)
    for x in nd:
        k = str(x["n"])
        s = good["ivk"][k] / math.sqrt(252) * 100
        m = FIT_IV["base_iv"][k]
        assert x["range_sigma_src"] == "txo_iv" and x["ivk_date"] == "2026-09-22" and x["ivk_stale_td"] == 0
        assert x["range_sigma"] == round(s, 3)
        assert x["path_low20"] == round(m["low20"] * s, 2) and x["path_high80"] == round(m["high80"] * s, 2)
        assert x["path_low10"] == round(min(m["low10"], m["low20"]) * s, 2)
        assert x["range_sigma_atr"] == round(float(RL.sigma_series(fr.tail(400)).iloc[-1]), 3)
        assert "sigma 來源 TXO 選擇權 IV" in x["range_note"]
        assert x["touch_prob"] == RL.TOUCH_PROB
    assert RL.IV_TOUCH_TEXT in hdr and "選擇權 IV" in hdr
    # 夜盤模式：有 base_iv 但缺 night_iv → 全部 ATR 夜盤
    p_nonight = _params_path("no_night_iv", lambda p: p.pop("night_iv", None))
    ref_n, _ = _attach(fr, P_ATR, None, night=0.6)
    nd_n, _ = _attach(fr, p_nonight, good, night=0.6)
    _same_levels(nd_n, ref_n)
    assert all(x["range_mode"] == "night" and x["range_sigma_src"] == "atr_ewma" for x in nd_n)
    # 有 night_iv → IV 夜盤 (β×夜盤 + m×σ_iv)
    nd_i, _ = _attach(fr, P_IV, good, night=0.6)
    s1 = good["ivk"]["1"] / math.sqrt(252) * 100
    ni = FIT_IV["night_iv"]["1"]
    assert nd_i[0]["range_mode"] == "night" and nd_i[0]["range_sigma_src"] == "txo_iv"
    assert nd_i[0]["path_high80"] == round(ni["beta_high"] * 0.6 + ni["high80"] * s1, 2)
    assert nd_i[0]["path_low20"] == round(ni["beta_low"] * 0.6 + ni["low20"] * s1, 2)
    # 單一 range_levels 呼叫：IV σ 缺 → ATR σ × ATR 乘數 (不會 IV 乘數 × ATR σ)
    r_atr = RL.range_levels(fr.tail(400), 1, path=P_IV, sigma_iv=None)
    r_ref = RL.range_levels(fr.tail(400), 1, path=P_ATR)
    assert r_atr["low20"] == r_ref["low20"] and r_atr["sigma_src"] == "atr_ewma"


# ------------------------------------------------------------------ 3. 休市後首日：所有 k 維持 ATR；k1 = ATR × √n_US
def test_reopen_day_all_k_atr():
    if not E.range_levels_enabled():
        print("  (skip: event_params range_levels_path 未啟用)")
        return
    fr = FR[FR["date"] <= "2026-09-24"]             # 下一交易日 09-29 (n_US = 3)
    good = _ivk_live("2026-09-24")
    nd, hdr = _attach(fr, P_IV, good, night=0.8)
    ref, _ = _attach(fr, P_ATR, None, night=0.8)
    _same_levels(nd, ref)
    assert all(x["range_sigma_src"] == "atr_ewma" and x["range_sigma_fallback"] == "休市後首日維持 ATR+EWMA" for x in nd)
    assert nd[0]["range_mode"] == "base_event"
    r1 = RL.range_levels(fr.tail(400), 1, base_px=float(fr["close"].iloc[-1]), path=P_ATR)
    assert abs(nd[0]["path_low20"] - round(r1["low20"] * math.sqrt(3), 2)) < 1e-9
    assert "休市後首日維持 ATR+EWMA" in hdr and "只有 1 日帶 × √n_US，k≥2 不放寬" in hdr


# ------------------------------------------------------------------ 4. fit_multipliers：base / night 逐位元相同；IV 區塊與監控
def test_fit_base_night_bit_identical():
    for key in ("base", "night", "night_start", "start", "end", "n", "sigma"):
        assert json.dumps(FIT_IV.get(key), sort_keys=True) == json.dumps(FIT_ATR.get(key), sort_keys=True), key
    assert json.dumps(FIT_IV["coverage"]["last250"]["base"]) == json.dumps(FIT_ATR["coverage"]["last250"]["base"])
    assert json.dumps(FIT_IV["coverage"]["last250"]["night"]) == json.dumps(FIT_ATR["coverage"]["last250"]["night"])
    assert "base_iv" not in FIT_ATR and "iv_monitor" not in FIT_ATR
    assert set(FIT_IV["base_iv"]) == {"1", "2", "3"} and set(FIT_IV["night_iv"]) == {"1", "2", "3"}
    for k in ("1", "2", "3"):
        assert {"low10", "low20", "high80", "high90", "beta_low", "beta_high"} <= set(FIT_IV["night_iv"][k])
    assert FIT_IV["iv_start"] >= RL.IV_START and FIT_IV["iv_end"] == "2026-09-24" and FIT_IV["sigma_iv"] == RL.IV_FORMULA
    assert "base_iv" in FIT_IV["coverage"]["last250"] and "night_iv" in FIT_IV["coverage"]["last250"]
    mon = FIT_IV["iv_monitor"]
    assert mon["iv_enabled"] is True and mon["alert_last250"] == RL.IV_ALERT_LAST250 and "iv_enabled" in mon["rollback"]
    # 2017 以前的 ivk 不參與 (IV_START)
    early = _ivk_hist(FR) + [{"date": "2016-12-30", "ivk": {"1": 9.9, "2": 9.9, "3": 9.9}}]
    f2 = RL.fit_multipliers(FR, NIGHT, path=None, ivk_hist=early)
    assert f2["base_iv"] == FIT_IV["base_iv"]


def test_iv_monitor_rollback_and_persistence():
    cov = {"last250": {"base_iv": {"k1_high90": 0.18, "k3_low20": 0.2, "k1_high80": 0.2}}}
    m1 = RL._iv_monitor(cov, None)
    assert m1["iv_enabled"] and m1["breach_streak"] == 1 and m1["alerts"]
    m2 = RL._iv_monitor(cov, {"iv_monitor": m1})
    assert m2["iv_enabled"] is False and m2["breach_streak"] == 2 and "auto" in m2["disabled_by"]
    ok = {"last250": {"base_iv": {"k1_high90": 0.1, "k3_low20": 0.2, "k1_high80": 0.2}}}
    m3 = RL._iv_monitor(ok, {"iv_monitor": m2})                 # 關閉後保留 (手動改回 true 才恢復)
    assert m3["iv_enabled"] is False and m3["breach_streak"] == 0
    m4 = RL._iv_monitor(ok, {"iv_monitor": {"iv_enabled": True, "breach_streak": 1}})
    assert m4["iv_enabled"] and m4["breach_streak"] == 0
    # 寫檔時讀前一版：前一版 iv_enabled=false 會保留
    f = TMP / "persist.json"
    prev = copy.deepcopy(FIT_IV)
    prev["iv_monitor"]["iv_enabled"] = False
    f.write_text(json.dumps(prev), encoding="utf-8")
    out = RL.fit_multipliers(FR, NIGHT, path=f, ivk_hist=HIST)
    assert out["iv_monitor"]["iv_enabled"] is False
    assert not RL.iv_enabled(RL.load_multipliers(f))


def test_sigma_iv_frame_alignment():
    hist = [{"date": "2026-09-22", "ivk": {"1": 0.2, "2": 0.0, "3": None}}, {"date": "2026-09-22", "ivk": {"1": 0.3, "2": 0.25, "3": -1}}]
    f = RL.sigma_iv_frame(pd.Series(["2026-09-21", "2026-09-22"]), hist)
    assert np.isnan(f.loc[0, 1]) and abs(f.loc[1, 1] - 0.3 / math.sqrt(252) * 100) < 1e-12      # 同日後者優先
    assert abs(f.loc[1, 2] - 0.25 / math.sqrt(252) * 100) < 1e-12 and np.isnan(f.loc[1, 3])        # ≤0 → NaN


# ------------------------------------------------------------------ 5. learn 帳本：欄位名與型別不變，另記 sigma 來源
def test_learn_records_unchanged_types():
    fr = FR[FR["date"] <= "2026-09-22"]
    for path, ivk, src in ((P_ATR, None, "atr_ewma"), (P_IV, _ivk_live("2026-09-22"), "txo_iv")):
        nd, _ = _attach(fr, path, ivk)
        fc = {"date": "2026-09-22", "close": float(fr["close"].iloc[-1]), "next_days": nd, "horizons": {}}
        rows = learn.records_from_forecast(fc, {"phase": "closed"})
        assert len(rows) == 3
        for r in rows:
            for k in ("buy_at", "sell_at", "stop", "target_px"):
                assert isinstance(r[k], float), (k, r[k])
            assert r["range_sigma_src"] == src
        for x in nd:
            for k in ("buy_at", "sell_at", "stop", "target"):
                assert isinstance(x[k], int), (k, x[k])


def test_self_learning_factor_applies_to_iv():
    fr = FR[FR["date"] <= "2026-09-22"]
    good = _ivk_live("2026-09-22")
    a, _ = _attach(fr, P_IV, good, sf=1.0)
    b, _ = _attach(fr, P_IV, good, sf=1.2)
    bp = float(fr["close"].iloc[-1])
    assert b[0]["buy_at"] == round(bp + (a[0]["buy_at"] - bp) * 1.2)


def test_ivk_consistency_tool_flags_calendar_change():
    """tools/ivk_consistency.py：上線 n 依當時日曆；事後多一個臨時休市 (颱風) → n 少 1 → ivk 差 >1% → 無效；日曆不變 → 有效。"""
    sys.path.insert(0, str(ROOT / "tools"))
    import ivk_consistency as IC
    from chip.sources import taifex_opt
    with offline():
        live_cal = lambda d, n: twse.next_trading_days(d, n)     # noqa: E731
        ss = [{"n": 2, "atm": 0.30, "kind": "W"}, {"n": 7, "atm": 0.24, "kind": "W"}, {"n": 17, "atm": 0.22, "kind": "M"}]
        live = {str(k): round(taifex_opt._cm(pd.DataFrame(ss), "atm", k), 5) for k in (1, 2, 3, 5, 10, 20)}
        row = {"date": "2026-09-14", "ivk": live, "ss": ss}
        days = E.tw_sessions("2026-09-01", "2026-09-24")
        ok = IC.check([row], days, live_cal)[0]
        assert not ok["invalid"] and ok["rel_same_n"] < 1e-6 and ok["rel_expost_n"] < 1e-6 and ok["n_changed"] == []
        typhoon = [d for d in days if d != "2026-09-15"]               # 事後 09-15 臨時休市
        bad = IC.check([row], typhoon, live_cal)[0]
        assert bad["invalid"] and bad["n_changed"] == [2, 7, 17], bad


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
