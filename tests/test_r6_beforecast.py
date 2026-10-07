"""r6 (2026-10-07) be-forecast 單元測試：合成資料、不連網、不寫 data/models。

    python tests/test_r6_beforecast.py
    python -m pytest tests/test_r6_beforecast.py

(1) 禁用字：負責模組的字串常數 (不含 docstring/註解) 不含買進/賣出/加碼/減碼/放空/…/承接/停損/持股水位/續抱/試單；
(2) verdict.action 各分支中性用語、n<30 桶/信心分層 hit 為 null (C6 b/c)；confidence.for_display；
(3) precheck：開盤前 (夜盤估計跳空) 用估計跳空分組表、盤中 (實際開盤) 用實際表、舊 precheck.json 無 cells_est → 實際表並註明；
(4) range_levels：touch_last250 (C6 a) 模式對應；IV 回滾改 pinball 相對規則 (F8)，無 pinball → 舊觸及率規則；
(5) realtime.combined_view / short_term._honesty_note 無禁用字。
"""
from __future__ import annotations

import ast
import contextlib
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip import realtime as RT  # noqa: E402
from chip.predict import confidence as CF, model as M, precheck as PC, range_levels as RL, short_term as ST, verdict as VD  # noqa: E402

BANNED = ("買進", "賣出", "加碼", "減碼", "放空", "做多", "做空", "建議買", "建議賣", "抄底", "逃頂", "縮小部位", "進場", "持有",
          "持股水位", "續抱", "試單", "承接", "停損")
OWNED = ("chip/predict/verdict.py", "chip/predict/precheck.py", "chip/predict/short_term.py", "chip/predict/market_forecast.py", "chip/predict/confidence.py",
         "chip/predict/learn.py", "chip/predict/range_levels.py", "chip/predict/es_evening.py", "chip/analysis/market.py", "chip/realtime.py", "tools/export_static.py")


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _clean(s: str) -> list[str]:
    return [w for w in BANNED if w in str(s)]


# ------------------------------------------------------------------ 1. 禁用字 (原始碼字串常數)
def _strings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            b0 = node.body[0]
            if isinstance(b0, ast.Expr) and isinstance(b0.value, ast.Constant) and isinstance(b0.value.value, str):
                doc.add(id(b0.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc:
            yield node.lineno, node.value


def test_owned_sources_have_no_banned_words():
    bad = []
    for f in OWNED:
        for ln, s in _strings(ROOT / f):
            if _clean(s):
                bad.append(f"{f}:{ln} {_clean(s)} {s[:60]!r}")
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------ 2. verdict / confidence
VT = {"tiers": {
    "1_base": {"bucket": {"高共識": {"n": 573, "hit": 0.628, "cov": 0.338, "yr_min": 0.485}, "一般": {"n": 1025, "hit": 0.535, "cov": 0.604, "yr_min": 0.431}, "分歧": {"n": 12, "hit": 0.75, "cov": 0.01, "yr_min": 0.5}},
               "nocall": {"n": 917, "base_up": 0.549, "by_net": {"4": {"n": 61, "up": 0.705, "yr_min": 0.45}, "-3": {"n": 80, "up": 0.52, "yr_min": 0.4}}}},
    "1_night": {"bucket": {"高共識": {"n": 243, "hit": 0.86, "cov": 0.362, "yr_min": 0.738}, "一般": {"n": 399, "hit": 0.855, "cov": 0.595, "yr_min": 0.802}, "分歧": {"n": 29, "hit": 0.897, "cov": 0.043, "yr_min": 0.8}}}}}
CT = {"tiers": {"1_night": {"高": {"n": 350, "cov": 0.304, "hit": 0.903, "yr_min": 0.881}, "低": {"n": 2, "cov": 0.002, "hit": 1.0, "yr_min": None}},
                "1_base": {"高": {"n": 578, "cov": 0.221, "hit": 0.607, "yr_min": 0.508}, "低": {"n": 300, "cov": 0.2, "hit": 0.53, "yr_min": 0.4}}}}


@contextlib.contextmanager
def tables():
    with patched(M, "load_json", lambda name: copy.deepcopy({"verdict": VT, "confidence": CT}.get(name))):
        yield


def _fc(variant="base", call="偏多", strength="", conf_tier=None, conf_hit=None, gate=None, votes_bull=False, **nd_kw):
    nd = {"n": 1, "date": "2026-10-08", "variant": variant, "call": call, "call_strength": strength, "p_up": 0.6, "base_hit": 0.55,
          "level": 49775, "buy_at": 49189, "sell_at": 50298, "stop": 48962, "target": 50466, **nd_kw}
    if conf_tier:
        nd.update(conf_tier=conf_tier, conf_hit=conf_hit)
    d = {"date": "2026-10-07", "close": 49500.0, "next_days": [nd], "horizons": {},
         "patterns": {"direction1": "偏多", "score1": 0.5} if votes_bull else {"direction1": "偏空", "score1": -0.5},
         "deep": {"today": {"hsi_r0": 1.2, "kospi_r0": 0.8, "smart2": 1.5} if votes_bull else {"hsi_r0": -1.2, "kospi_r0": -0.8, "smart2": -1.5}},
         "trend7": {"available": bool(gate), "state": gate or "flat", "confidence": "中"}}
    return d


def _all_text(V: dict) -> str:
    return " ".join(str(V.get(k) or "") for k in ("verdict", "head", "action", "text"))


def test_verdict_actions_neutral_all_branches():
    cases = []
    with tables():
        cases.append(VD.build(_fc(call="偏多"), None, {"phase": "closed"}, None))                                   # 偏多 + 分歧 (n=12 → hit null)
        cases.append(VD.build(_fc(call="偏多", gate="down"), None, {"phase": "closed"}, None))                      # 7 日閘門偏下
        cases.append(VD.build(_fc(call="偏空", votes_bull=True), None, {"phase": "closed"}, None))                  # 偏空 + 上緣
        cases.append(VD.build(_fc(call="偏空", votes_bull=True, gate="up"), None, {"phase": "closed"}, None))       # 偏空 + 閘門偏上
        cases.append(VD.build(_fc(call="中性"), None, {"phase": "closed"}, None))                                   # 中性區間 + 空方共識
        cases.append(VD.build(_fc(call="中性", gate="down"), None, {"phase": "closed"}, None))
        cases.append(VD.build(_fc(call="中性", gate="up", buy_at=None), None, {"phase": "closed"}, None))          # 無區間 → 預設句
        hr = {"live": True, "mark": "10:00", "targets": {"13:30": {"p_up": 0.3, "base_hit": 0.55, "level": 49000}}}
        snap = {"phase": "day", "score": {"score": -40, "label": "弱勢", "parts": []}}
        cases.append(VD.build(_fc(call="偏多", votes_bull=True), hr, snap, None))                                   # 小時模型 / 即時評分相反
    for V in cases:
        assert not _clean(_all_text(V)), (_clean(_all_text(V)), V["action"], V["head"])
    a = [V["action"] for V in cases]
    assert a[0] == "下緣參考 49,189，跌破 48,962 轉弱，上緣 (一成機率) 50,466；訊號分歧：叫牌可信度較低", a[0]
    assert a[1] == "7 日閘門偏下：偏多訊號可信度降低，下緣參考價僅供觀察；訊號分歧：叫牌可信度較低", a[1]
    assert a[2].startswith("上緣參考 (壓力) 50,298，跌破 48,962 轉弱"), a[2]
    assert a[3].startswith("7 日閘門偏上：偏空訊號可信度降低"), a[3]
    assert a[4] == "區間參考：下緣 49,189 / 上緣 50,298，跌破 48,962 轉弱", a[4]
    assert a[5].endswith("7 日閘門偏下：下緣參考價可信度降低") and a[6] == "7 日閘門偏上：上緣參考價可信度降低", (a[5], a[6])
    assert "日模型叫牌可信度降低" in a[7] and "留意盤中訊號是否轉向" in a[7], a[7]
    assert "不叫偏空" in cases[4]["head"] and "不放空" not in cases[4]["head"]
    # 分歧桶 n=12 < 30 → hit/yr_min null、head 寫樣本不足 (不印 75%)
    o = cases[0]["oos"]
    assert o["n"] == 12 and o["hit"] is None and o["yr_min"] is None and o["hit_raw"] == 0.75 and o["small_n"] is True
    assert "樣本不足 (n=12" in cases[0]["head"] and "75%" not in cases[0]["head"]
    # 預設句 (無任何參考價) 亦中性
    with tables():
        V = VD.build(_fc(call="中性", buy_at=None, sell_at=None), None, {"phase": "closed"}, None)
    assert V["action"] == "依上下緣參考價觀察"


def test_conf_tier_small_n_hidden():
    st = CF.for_display({"n": 2, "cov": 0.002, "hit": 1.0, "yr_min": None})
    assert st["hit"] is None and st["hit_raw"] == 1.0 and st["small_n"] is True and st["n"] == 2
    big = {"n": 350, "hit": 0.903, "cov": 0.3, "yr_min": 0.88}
    assert CF.for_display(big) is big and CF.for_display(None) is None and CF.for_display({"hit": 0.5}) == {"hit": 0.5}
    assert CF.for_display({"n": 30, "hit": 0.6})["hit"] == 0.6          # 恰 30 顯示
    snap = {"phase": "closed", "tx_night": {"change_pct": 1.2, "final": True}}
    with tables():
        # 舊 fc 仍帶小樣本 conf_hit=1.0 (夜盤 低 n=2) → verdict 不顯示
        V = VD.build(_fc(variant="night", call="偏多", strength="強", conf_tier="低", conf_hit=1.0), None, snap, None)
        V2 = VD.build(_fc(variant="night", call="偏多", strength="強", conf_tier="高", conf_hit=0.903), None, snap, None)
    assert V["conf_hit"] is None and V["conf_oos"]["hit"] is None and V["conf_oos"]["n"] == 2 and "100%" not in V["head"], V["head"]
    assert V["head"].startswith("模型偏多強；信心分層 低；其他"), V["head"]
    assert V2["conf_hit"] == 0.903 and "(歷史 90%" in V2["head"]
    # 夜盤 分歧 桶 n=29 → hit null
    assert V["oos"]["n"] == 29 and V["oos"]["hit"] is None


# ------------------------------------------------------------------ 3. precheck 估計跳空表
def _gap_market(seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2026-10-06")
    n = len(days)
    night = rng.normal(0, 0.8, n)
    c = np.empty(n); o = np.empty(n); h = np.empty(n); lo = np.empty(n)
    prev = 10000.0
    for i in range(n):
        gap = 0.5 * night[i] + rng.normal(0, 0.35)
        o[i] = prev * (1 + gap / 100)
        c[i] = o[i] * (1 + rng.normal(0.02, 0.8) / 100)
        h[i] = max(o[i], c[i]) * (1 + abs(rng.normal(0, 0.4)) / 100)
        lo[i] = min(o[i], c[i]) * (1 - abs(rng.normal(0, 0.4)) / 100)
        prev = c[i]
    ds = [d.strftime("%Y-%m-%d") for d in days]
    scored = pd.DataFrame({"date": ds, "open": o, "high": h, "low": lo, "close": c})
    nh = pd.DataFrame({"date": ds, "night_chg_pct": night})
    nh = nh[nh["date"] >= "2017-01-03"].reset_index(drop=True)
    return scored, nh


SC, NH = _gap_market()


def _trained():
    with patched(ST, "_night_hist", lambda: NH):
        return PC.train(SC, write=False, verbose=False)


ST_PC = _trained()


def _build(st, night=None, live_open=None, est_override=None):
    nd = [{"n": 1, "date": "2026-10-07"}]
    fc = {"date": "2026-10-06", "close": float(SC["close"].iloc[-1]), "next_days": nd, "intraday": {"price": 1.0} if live_open else None}
    snap = {"phase": "closed"}
    if live_open:
        snap = {"phase": "open", "taiex": {"open": live_open, "prev": float(SC["close"].iloc[-1])}}
    with patched(M, "load_json", lambda name: copy.deepcopy(st) if name == "precheck" else None), \
            patched(RL, "_night_final", lambda snap_, nd_, scored_: (night, None)), patched(PC, "_us_add", lambda *a, **k: None):
        return PC.build(SC, snap, fc)


def test_precheck_est_table_trained():
    g = ST_PC["gap"]
    assert g.get("cells_est") and g.get("est_meta") and g["est_meta"]["n"] > 1000 and g["est_meta"]["start"] >= "2017"
    for k, c in g["cells_est"].items():
        assert c["basis"] == "est" and c["n"] >= 20 and 0 <= c["p_fill"] <= 1 and set(g["cells"][k]) <= set(c), k
    # 估計平盤日混入實際有跳空的日子 → 回補率低於實際平盤表 (研究：真實資料 66% vs 79%)
    f_est, f_act = g["cells_est"]["3|bull"]["p_fill"], g["cells"]["3|bull"]["p_fill"]
    assert f_est < f_act, (f_est, f_act)
    # 實際跳空表與 r6 前同一程式 (重構不改數值)：flat 格的回補 = 實際平盤日的 fill
    assert all("basis" not in c for c in g["cells"].values())


def test_precheck_build_uses_est_preopen_and_actual_intraday():
    bu = ST_PC["gap"]["night_beta"]["beta_recent"]
    out = _build(ST_PC, night=0.02)                                          # 夜盤 → 估計平盤
    gp = out["gap"]
    key = "3|" + ("bull" if gp["regime"].startswith("多頭") else "bear")
    assert abs(gp["est"] - round(bu * 0.02, 2)) < 1e-9 and gp["table"] == "est"
    assert gp["stats"] == ST_PC["gap"]["cells_est"][key] and gp["stats_actual"] == ST_PC["gap"]["cells"][key] and gp["stats_est"] == gp["stats"]
    assert "以夜盤估計跳空分組" in gp["text"] and "估計跳空分組" in gp["table_note"] and f"回補 (回到前收) {gp['stats']['p_fill']:.0%}" in gp["text"]
    # 盤中實際開盤已知 → 實際表
    o = float(SC["close"].iloc[-1]) * 1.0001
    out2 = _build(ST_PC, live_open=o)
    g2 = out2["gap"]
    assert g2["table"] == "actual" and g2["stats"] == ST_PC["gap"]["cells"][key] and g2["source"] == "實際開盤" and "以夜盤估計跳空分組" not in g2["text"]
    assert "盤中實際開盤已知" in g2["table_note"]
    # 舊 precheck.json (無 cells_est) → 開盤前仍用實際表並註明可能高估
    old = copy.deepcopy(ST_PC); old["gap"].pop("cells_est"); old["gap"].pop("est_meta")
    out3 = _build(old, night=0.02)
    g3 = out3["gap"]
    assert g3["table"] == "actual" and g3["stats"] == ST_PC["gap"]["cells"][key] and "stats_est" not in g3 and "暫用實際跳空分組" in g3["table_note"]
    # 大跳空 (開高/開低) 文字中性
    for nv in (2.5, -2.5, 1.0, -1.0, 0.6, -0.6):
        for st in (ST_PC, old):
            gx = _build(st, night=nv)["gap"]
            assert gx and not _clean(gx["text"]) and not _clean(gx["table_note"]), (nv, gx["text"])
    for nv, st in ((0.02, ST_PC), (None, ST_PC)):
        r = _build(st, night=nv)
        assert r["gap"] is None or not _clean(r["gap"]["text"])


# ------------------------------------------------------------------ 4. range_levels
P_COV = {"end": "2026-10-02", "fitted_at": "2026-10-04T08:28+08:00", "base_iv": {"1": {}}, "iv_monitor": {"iv_enabled": True},
         "coverage": {"last250": {
             "base": {"k1_low10": 0.076, "k1_low20": 0.177, "k1_high80": 0.281, "k1_high90": 0.145, "k1_n": 249, "k2_low10": 0.085, "k2_low20": 0.173, "k2_high80": 0.29, "k2_high90": 0.173, "k2_n": 248},
             "base_iv": {"k1_low10": 0.092, "k1_low20": 0.181, "k1_high80": 0.285, "k1_high90": 0.161, "k1_n": 249, "k3_low10": 0.097, "k3_low20": 0.198, "k3_high80": 0.259, "k3_high90": 0.142, "k3_n": 247},
             "night_iv": {"k1_low10": 0.092, "k1_low20": 0.189, "k1_high80": 0.253, "k1_high90": 0.129, "k1_n": 249}}}}


def test_touch_last250_modes():
    t = RL.touch_last250([{"n": 1, "range_mode": "base", "range_sigma_src": "txo_iv"}], p=P_COV)
    assert t["mode"] == "base_iv" and t["src"] == "iv_monitor" and t["n"] == 250 and set(t) >= {"k1", "k3", "n_k", "note", "nominal_touch"} and "k2" not in t
    assert t["k1"] == {"low10": 0.092, "low20": 0.181, "high80": 0.285, "high90": 0.161} and t["n_k"] == {"k1": 249, "k3": 247}
    t2 = RL.touch_last250([{"n": 1, "range_mode": "night", "range_sigma_src": "txo_iv"}], p=P_COV)
    assert t2["mode"] == "night_iv" and t2["k1"]["high80"] == 0.253 and t2["src"] == "coverage.last250.night_iv"
    t3 = RL.touch_last250([{"n": 1, "range_mode": "base_event", "range_sigma_src": "atr_ewma"}], p=P_COV)
    assert t3["mode"] == "base" and t3["k2"]["high80"] == 0.29
    t4 = RL.touch_last250([{"n": 1, "range_mode": "night", "range_sigma_src": "atr_ewma"}], p=P_COV)   # 夜盤 ATR 表缺 → base
    assert t4["mode"] == "base"
    assert RL.touch_last250(None, p=P_COV)["mode"] == "base_iv"                                     # 無 nd → iv_enabled
    assert RL.touch_last250([{"n": 1}], p={"coverage": {}}) is None and RL.touch_last250(None, path=ROOT / "no_such.json") is None


def test_iv_monitor_pinball_rule():
    cov_breach = {"last250": {"base_iv": {"k1_high90": 0.18, "k3_low20": 0.2, "k1_high80": 0.29}}}
    good, bad = {"ratio": 0.94, "iv": 0.41, "atr": 0.44}, {"ratio": 1.05, "iv": 0.46, "atr": 0.44}
    # 觸及率連續超標但 IV pinball 較好 → 不回滾 (只警示)
    m1 = RL._iv_monitor(cov_breach, None, good)
    m2 = RL._iv_monitor(cov_breach, {"iv_monitor": m1}, good)
    assert m2["iv_enabled"] is True and m2["breach_streak"] == 1 and m2["rollback_rule"] == "pinball" and m2["pin_streak"] == 0 and m2["alerts"]
    # r6 review：pinball 主導時觸及率不累積 → 退回觸及率規則 (pin=None) 時要再連續兩次才回滾
    f1 = RL._iv_monitor(cov_breach, {"iv_monitor": m2}, None)
    assert f1["iv_enabled"] is True and f1["breach_streak"] == 1 and f1["rollback_rule"] == "touch"
    f2 = RL._iv_monitor(cov_breach, {"iv_monitor": f1}, None)
    assert f2["iv_enabled"] is False
    # pinball(IV) > 1.02 × ATR 連續兩次 → 回滾
    p1 = RL._iv_monitor({"last250": {"base_iv": {}}}, None, bad)
    assert p1["iv_enabled"] is True and p1["pin_streak"] == 1
    p2 = RL._iv_monitor({"last250": {"base_iv": {}}}, {"iv_monitor": p1}, bad)
    assert p2["iv_enabled"] is False and "pinball" in p2["disabled_by"] and p2["pin_streak"] == 2
    # 中斷 (一次好) → streak 歸零
    p3 = RL._iv_monitor({"last250": {"base_iv": {}}}, {"iv_monitor": p1}, good)
    assert p3["iv_enabled"] is True and p3["pin_streak"] == 0
    # 關閉後保留
    p4 = RL._iv_monitor({"last250": {"base_iv": {}}}, {"iv_monitor": p2}, good)
    assert p4["iv_enabled"] is False and p4["disabled_by"] == p2["disabled_by"]
    # 無 pinball → 舊觸及率規則 (相容)
    o2 = RL._iv_monitor(cov_breach, {"iv_monitor": RL._iv_monitor(cov_breach, None)})
    assert o2["iv_enabled"] is False and o2["rollback_rule"] == "touch" and "pinball_last250" not in o2
    assert "pinball(IV) > 1.02 × pinball(ATR)" in RL.IV_ROLLBACK_TEXT and "iv_enabled" in RL.IV_ROLLBACK_TEXT


def _rl_frame(seed=9):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-05", "2026-09-30")
    r = rng.normal(0, 0.011, len(days))
    c = 10000 * np.exp(np.cumsum(r))
    o = c * np.exp(rng.normal(0, 0.003, len(days)))
    h = np.maximum(c, o) * np.exp(np.abs(rng.normal(0, 0.005, len(days))))
    l_ = np.minimum(c, o) * np.exp(-np.abs(rng.normal(0, 0.005, len(days))))
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in days], "open": o, "high": h, "low": l_, "close": c})


def test_pinball_last250_and_fit():
    d = _rl_frame()
    sig = RL.sigma_series(d)
    tg = RL.path_targets(d)
    # IV σ = 常數 × ATR σ 且訓練列相同 → 乘數等比例、水準相同 → ratio 恰為 1
    siv = pd.DataFrame({k: sig * 1.3 for k in RL.IV_KS})
    with patched(RL, "IV_START", "1900-01-01"):
        pin = RL._pinball_last250(d, sig, tg, siv)
    assert pin and abs(pin["ratio"] - 1.0) < 1e-9 and pin["cells"] == 12 and set(pin["n"]) == {"1", "2", "3"} and pin["n"]["1"] >= 240, pin
    # IV σ 為純雜訊 → 比 ATR 差
    rng = np.random.default_rng(1)
    noisy = pd.DataFrame({k: pd.Series(np.exp(rng.normal(0, 0.6, len(d))), index=d.index) * float(sig.median()) for k in RL.IV_KS})
    pin2 = RL._pinball_last250(d, sig, tg, noisy)
    assert pin2["ratio"] > 1.02, pin2
    assert RL._pinball_last250(d, sig, tg, None) is None and RL._pinball_last250(d.tail(400).reset_index(drop=True), sig.tail(400).reset_index(drop=True), tg.tail(400).reset_index(drop=True), siv.tail(400).reset_index(drop=True)) is None
    # fit_multipliers：有 ivk 歷史 → iv_monitor 走 pinball 規則；base/night 乘數不受影響
    hist = [{"date": dd, "ivk": {"1": round(float(v), 5), "2": round(float(v), 5), "3": round(float(v), 5)}}
            for dd, v in zip(d["date"], 0.011 * np.sqrt(252) * np.exp(rng.normal(0, 0.15, len(d)))) if dd >= "2017-01-03"]
    f_iv = RL.fit_multipliers(d, None, path=None, ivk_hist=hist)
    f_atr = RL.fit_multipliers(d, None, path=None)
    mon = f_iv["iv_monitor"]
    assert mon["rollback_rule"] == "pinball" and mon["pinball_last250"]["ratio"] > 0 and mon["pin_streak"] in (0, 1) and mon["iv_enabled"] is True
    assert f_iv["base"] == f_atr["base"] and "iv_monitor" not in f_atr
    assert "pinball" in RL.format_coverage(f_iv)


# ------------------------------------------------------------------ 5. 其他文字
def test_realtime_and_caveats_neutral():
    for cs in (30, 0, -30):
        for rs in (30, 0, -30):
            for ph in ("open", "closed"):
                t = RT.combined_view(cs, "", rs, "", ph)
                assert t and not _clean(t), t
    for h in (1, 2, 3, 5):
        for v in ("base", "night"):
            for ph in (None, "open"):
                assert not _clean(ST._honesty_note(h, v, ph))
    assert "開盤後才看" in ST._honesty_note(1, "night")


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
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
