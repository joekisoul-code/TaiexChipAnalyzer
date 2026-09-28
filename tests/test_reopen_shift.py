"""復市日美股移動 (r2m reopen_us_move) 單元測試 (final_spec §5.6)：gap_beta(target='cc')、us_move、precheck 未涵蓋美股、
range_levels 休市後首日 1 日帶中心移、fit_centre_shift / save_centre_shift。不需網路 (規則日曆 + 合成價格)。

    python tests/test_reopen_shift.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip import config  # noqa: E402
from chip.predict import events as E, learn, model as M, precheck as PC, range_levels as RL  # noqa: E402
from chip.sources import twse  # noqa: E402

TZ = config.TZ
TMP = Path(tempfile.mkdtemp(prefix="reopen_"))


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
    E.reset_calendar_cache()
    twse._HOL_FAIL.clear()
    with patched(E, "tw_holidays", lambda y: (set(E.tw_rule_holidays(y)), "api" if y <= 2026 else "rule")), \
            patched(E, "_load_us_actual", lambda: None), \
            patched(twse, "holidays", lambda y=None: set(E.tw_rule_holidays(y))):
        yield
    E.reset_calendar_cache()


def _market(start="2019-01-02", end="2026-09-24", b_gap=0.18, b_cc=0.33, seed=11):
    """合成 TAIEX (open/high/low/close) 與 SOX/TSM：開盤跳空 = b_gap × 期間美股累積，收盤 = b_cc × 期間美股累積 + 噪音。美股到 09-28。"""
    rng = np.random.default_rng(seed)
    with offline():
        us = E.us_sessions(start, "2026-09-29")
        tw = E.tw_sessions(start, end)
        sox_r = rng.normal(0, 0.02, len(us))
        xr = dict(zip(us, sox_r))
        rows, c = [], 20000.0
        for i, d in enumerate(tw):
            if i == 0:
                rows.append({"date": d, "open": c, "high": c * 1.005, "low": c * 0.995, "close": c})
                continue
            s = sum(xr[u] for u in E.us_sessions(tw[i - 1], d))
            o = c * math.exp(b_gap * s + rng.normal(0, 0.003))
            c2 = c * math.exp(b_cc * s + rng.normal(0, 0.009))
            rows.append({"date": d, "open": o, "high": max(o, c2) * 1.004, "low": min(o, c2) * 0.996, "close": c2})
            c = c2
    sox = pd.DataFrame({"date": us, "close": 3000 * np.exp(np.cumsum(sox_r))})
    tsm = pd.DataFrame({"date": us, "close": 100 * np.exp(np.cumsum(sox_r * 0.9 + rng.normal(0, 0.005, len(us))))})
    return pd.DataFrame(rows), sox, tsm


PX, SOX, TSM = _market()
AFTER = dt.datetime(2026, 9, 29, 5, 31, tzinfo=TZ)
BEFORE = dt.datetime(2026, 9, 29, 5, 29, tzinfo=TZ)


def _manual_beta(px, xr, before, target, win=750, min_n=250):
    d = px.sort_values("date").reset_index(drop=True)
    rows = []
    for i in range(1, len(d)):
        if d["date"][i] >= before:
            break
        pv, cur = E._d(d["date"][i - 1]), E._d(d["date"][i])
        if any((pv + dt.timedelta(days=j)).weekday() < 5 for j in range(1, (cur - pv).days)):
            continue
        us = E.us_sessions(pv, cur)
        if len(us) != 1 or us[0] not in xr:
            continue
        a, b = float(d["close"][i] if target == "cc" else d["open"][i]), float(d["close"][i - 1])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        y = math.log(a / b)
        if abs(y) > 0.15:
            continue
        rows.append((xr[us[0]], y))
    rows = rows[-win:]
    x, y = np.array(rows).T
    return float((x * y).sum() / (x * x).sum()), len(rows)


# ------------------------------------------------------------------ 1. gap_beta target='cc' + NaN 防護
def test_gap_beta_cc_and_nan_guard():
    with offline():
        xr = E._logret_by_date(SOX)
        b, n, rsd = E.gap_beta(PX[["date", "open", "close"]], xr, "2026-09-29", target="cc")
        bm, nm = _manual_beta(PX, xr, "2026-09-29", "cc")
        assert n == nm == 750 and abs(b - bm) < 1e-12, (b, bm, n, nm)
        assert 0.25 < b < 0.41, b                     # 合成真值 0.33
        bg, ng, _ = E.gap_beta(PX[["date", "open", "close"]], xr, "2026-09-29")
        bgm, _ = _manual_beta(PX, xr, "2026-09-29", "gap")
        assert abs(bg - bgm) < 1e-12 and 0.15 < bg < 0.21
        bad = PX.copy()
        bad.loc[bad.index[-40], "open"] = np.nan
        bad.loc[bad.index[-60], "close"] = np.nan
        b2, _, _ = E.gap_beta(bad[["date", "open", "close"]], xr, "2026-09-29")
        b3, _, _ = E.gap_beta(bad[["date", "open", "close"]], xr, "2026-09-29", target="cc")
        assert b2 is not None and np.isfinite(b2) and b3 is not None and np.isfinite(b3)
        assert E.gap_beta(PX.head(100)[["date", "open", "close"]], xr, "2026-09-29")[0] is None      # 正常日 < 250


# ------------------------------------------------------------------ 2. us_move
def test_us_move_rules():
    with offline():
        ss = E.us_sessions("2026-09-24", "2026-09-29")
        assert ss == ["2026-09-24", "2026-09-25", "2026-09-28"]
        m0 = E.us_move("2026-09-24", "2026-09-29", SOX, TSM, now=BEFORE)
        assert m0["status"] == "pending_us" and m0["sum_logret"] is None
        m1 = E.us_move("2026-09-24", "2026-09-29", SOX, TSM, now=AFTER)
        xr = E._logret_by_date(SOX)
        assert m1["status"] == "ready" and m1["src"] == "SOX" and abs(m1["sum_logret"] - sum(xr[u] for u in ss)) < 1e-15
        sox_miss = SOX[SOX["date"] != "2026-09-25"]
        m2 = E.us_move("2026-09-24", "2026-09-29", sox_miss, TSM, now=AFTER)
        tx = E._logret_by_date(TSM)
        assert m2["status"] == "ready" and m2["src"] == "TSM" and abs(m2["sum_logret"] - sum(tx[u] for u in ss)) < 1e-15
        m3 = E.us_move("2026-09-24", "2026-09-29", sox_miss, TSM[TSM["date"] != "2026-09-28"], now=AFTER)
        assert m3["status"] == "pending_us"
        m4 = E.us_move("2026-09-24", "2026-09-29", None, None, now=AFTER)
        assert m4["status"] == "pending_us"
        assert E.us_move("2026-09-24", "2026-09-29", SOX, TSM, sessions=[], now=AFTER)["status"] == "none"
        mu = E.us_move("2026-09-24", "2026-09-29", SOX, TSM, sessions=["2026-09-25", "2026-09-28"], now=AFTER)
        assert abs(mu["sum_logret"] - (xr["2026-09-25"] + xr["2026-09-28"])) < 1e-15


# ------------------------------------------------------------------ 3. precheck：未涵蓋美股
def _precheck(scored, nd, now, night=0.5, us_px=None):
    snap = {"phase": "closed", "tx_night": {"change_pct": night, "final": True}, "ts": now.strftime("%Y-%m-%d %H:%M:%S")}
    fc = {"date": str(scored["date"].iloc[-1]), "close": float(scored["close"].iloc[-1]), "next_days": nd}
    with offline():
        out = PC.build(scored, snap, fc, us_px=us_px if us_px is not None else {"sox": SOX, "tsm": TSM, "px": PX}, now=now)
    return out, fc


def test_precheck_us_add_ready_and_pending():
    st = M.load_json("precheck")
    if not st:
        print("  (skip: precheck.json 不存在)")
        return
    nb = st["gap"]["night_beta"]
    bu = nb.get("beta_recent") or nb.get("beta")
    sc = PX[PX["date"] <= "2026-09-24"]
    nd = [{"n": 1, "date": "2026-09-29"}, {"n": 2, "date": "2026-09-30"}, {"n": 3, "date": "2026-10-01"}]
    est0 = bu * 0.5
    out, fc = _precheck(sc, nd, AFTER)
    g = out["gap"]
    with offline():
        xr = E._logret_by_date(SOX)
        b, _, _ = E.gap_beta(PX[["date", "open", "close"]], xr, "2026-09-29")
    add = b * (xr["2026-09-25"] + xr["2026-09-28"]) * 100
    assert g["us_add"]["status"] == "ready" and g["us_add"]["n_unc"] == 2 and g["us_add"]["src"] == "SOX"
    assert g["us_add"]["sessions"] == ["2026-09-25", "2026-09-28"]
    assert abs(g["est"] - round(est0 + round(add, 3), 2)) < 1e-9, (g["est"], est0, add)
    assert abs(g["us_add"]["est_night_only"] - round(est0, 3)) < 1e-9 and "暫定" not in g["source"] and "provisional" not in g
    assert "夜盤未涵蓋的 2 個美股交易日費半" in g["source"]
    rec = learn.records_from_precheck({**fc, "precheck": out})
    assert rec and rec[0]["mode"] == "final" and abs(rec[0]["est_gap"] - g["est"]) < 1e-3
    # 美股尚未全部收盤 → 夜盤估計 + 暫定 → learn 記 prov
    out2, fc2 = _precheck(sc, nd, BEFORE)
    g2 = out2["gap"]
    assert g2["est"] == round(est0, 2) and "暫定" in g2["source"] and g2["provisional"] == "us_pending" and g2["us_add"]["status"] == "pending_us"
    rec2 = learn.records_from_precheck({**fc2, "precheck": out2})
    assert rec2 and rec2[0]["mode"] == "prov"
    # 美股資料取不到 → 同暫定
    out3, _ = _precheck(sc, nd, AFTER, us_px={"sox": None, "tsm": None, "px": PX})
    assert out3["gap"]["est"] == round(est0, 2) and out3["gap"]["provisional"] == "us_pending"
    # 正常日 (沒有未涵蓋美股日) → 不加欄位、est 與舊版相同
    sc4 = PX[PX["date"] <= "2026-09-22"]
    out4, _ = _precheck(sc4, [{"n": 1, "date": "2026-09-23"}], dt.datetime(2026, 9, 23, 6, 0, tzinfo=TZ))
    assert "us_add" not in out4["gap"] and out4["gap"]["est"] == round(est0, 2) and "provisional" not in out4["gap"]
    # 正常日樣本不足 (px 很短) → 夜盤估計、非暫定
    out5, _ = _precheck(sc, nd, AFTER, us_px={"sox": SOX, "tsm": TSM, "px": PX.tail(120)})
    assert out5["gap"]["us_add"]["status"] == "no_beta" and out5["gap"]["est"] == round(est0, 2) and "暫定" not in out5["gap"]["source"]


# ------------------------------------------------------------------ 4/5. range_levels 休市後首日中心移
def _attach(sc, us_px=None, sf=1.0):
    with offline():
        nd = [{"n": i + 1, "date": d, "label": ["隔天", "後天", "第三天"][i]} for i, d in enumerate(twse.next_trading_days(str(sc["date"].iloc[-1]), 3))]
        with patched(RL, "_night_final", lambda snap, nd_, scored: (0.8, None)):
            hdr = RL.attach_to_next_days(nd, sc, {"phase": "closed"}, float(sc["close"].iloc[-1]), sigma_factor=sf, us_px=us_px)
    return nd, hdr


def test_range_levels_centre_shift():
    if not RL.load_multipliers() or not E.range_levels_enabled():
        print("  (skip: range_levels.json 不存在或 range_levels_path 未啟用)")
        return
    sc = PX[PX["date"] <= "2026-09-24"]
    base, hdr0 = _attach(sc)                                       # 不注入美股 → 與現行相同 (√n_US 放寬)
    ref = RL.range_levels(sc.tail(400), 1, base_px=float(sc["close"].iloc[-1]))
    assert base[0]["range_mode"] == "base_event" and base[0]["path_low20"] == round(ref["low20"] * math.sqrt(3), 2)
    assert "event_shift_pct" not in base[0] and base[0]["touch_prob"] == 0.2
    pend, _ = _attach(sc, {"sox": SOX, "tsm": TSM, "px": PX, "now": BEFORE})
    for x, y in zip(pend, base):
        assert x["path_low20"] == y["path_low20"] and x["buy_at"] == y["buy_at"]
    assert pend[0]["event_shift_status"] == "pending_us" and "中心移待美股收盤後更新" in pend[0]["range_note"]
    nd, hdr = _attach(sc, {"sox": SOX, "tsm": TSM, "px": PX, "now": AFTER})
    wr = E.load_params()["band_rules"]["post_closure_centre_shift"]["width_ratio"]
    with offline():
        xr = E._logret_by_date(SOX)
        b, _, _ = E.gap_beta(PX[["date", "open", "close"]], xr, "2026-09-29", target="cc")
        c = b * sum(xr[u] for u in E.us_sessions("2026-09-24", "2026-09-29")) * 100
    x1 = nd[0]
    assert x1["range_mode"] == "base_event" and x1["touch_prob"] == 0.15
    for q, key in (("low10", "path_low10"), ("low20", "path_low20"), ("high80", "path_high80"), ("high90", "path_high90")):
        assert abs(x1[key] - round(ref[q] * math.sqrt(3) * wr + c, 2)) < 1e-9, (key, x1[key], ref[q], c)
    assert abs(x1["event_shift_pct"] - round(c, 2)) < 1e-9 and x1["event_width_ratio"] == wr and x1["event_shift_src"] == "SOX"
    assert x1["event_shift_sessions"] == ["2026-09-24", "2026-09-25", "2026-09-28"]
    bp = float(sc["close"].iloc[-1])
    assert x1["buy_at"] == int(round(bp * (1 + x1["path_low20"] / 100)))
    assert "1 日帶中心移" in x1["range_note"] and "休市後首日歷史觸及約 15%" in x1["range_note"]
    for x in nd[1:]:                                                # 單調：k2/k3 不窄於中心移後的 k1
        assert x["path_low20"] <= x1["path_low20"] and x["path_high80"] >= x1["path_high80"] and x["touch_prob"] == 0.2
        assert "event_shift_pct" not in x
    assert "1 日帶依休市期間美股平移並調寬 (2014~ 104 次：pinball −31%)" in hdr
    for t in (hdr, x1["range_note"]):
        for w in E.banned_phrases():
            assert w not in t, (w, t)
    # sigma_factor ≠ 1：休市後首日各 k 與 sf=1 相同
    nd_sf, _ = _attach(sc, {"sox": SOX, "tsm": TSM, "px": PX, "now": AFTER}, sf=1.25)
    for x, y in zip(nd_sf, nd):
        for k in ("buy_at", "sell_at", "stop", "target", "path_low20", "path_high90", "level_lo", "level_hi"):
            assert x[k] == y[k], (x["n"], k)
    assert "休市後首日不套自學乘數" in nd_sf[0]["range_note"]
    base_sf, _ = _attach(sc, None, sf=1.25)
    for x, y in zip(base_sf, base):
        assert x["buy_at"] == y["buy_at"] and x["sell_at"] == y["sell_at"]


def test_centre_shift_disabled_rule_keeps_base():
    if not RL.load_multipliers() or not E.range_levels_enabled():
        print("  (skip)")
        return
    P = json.loads(json.dumps(E.load_params()))
    P["band_rules"]["post_closure_centre_shift"]["enabled"] = False
    sc = PX[PX["date"] <= "2026-09-24"]
    base, _ = _attach(sc)
    with patched(E, "load_params", lambda path=None: P):
        nd, _ = _attach(sc, {"sox": SOX, "tsm": TSM, "px": PX, "now": AFTER})
    for x, y in zip(nd, base):
        assert x["path_low20"] == y["path_low20"] and x["touch_prob"] == y["touch_prob"]
    assert nd[0]["event_shift_status"] == "disabled"


# ------------------------------------------------------------------ 6. fit_centre_shift / save_centre_shift
def test_fit_and_save_centre_shift():
    px, sox, tsm = _market(start="2003-01-02", seed=12)
    with offline():
        fit = E.fit_centre_shift(px, sox, tsm)
    assert fit and fit["n"] >= 40 and 0.3 < fit["width_ratio"] < 1.0, fit        # 合成資料美股可解釋部分 → 殘差比 < 1
    assert fit["date_from"] < "2006-01-01" and fit["walkforward_range_2014plus"]
    tmp = TMP / "event_params.json"
    shutil.copy(E.PARAMS_PATH, tmp)
    assert not E.save_centre_shift({**fit, "n": 20}, path=tmp)                    # 樣本不足 → 不更新
    assert not E.save_centre_shift({**fit, "n": 200, "date_from": "2008-01-02"}, path=tmp)   # 起點太晚 → 不更新
    assert E.save_centre_shift({**fit, "n": 200}, path=tmp)
    P = json.loads(tmp.read_text(encoding="utf-8"))
    r = P["band_rules"]["post_closure_centre_shift"]
    assert r["width_ratio"] == round(fit["width_ratio"], 2) and r["width_ratio_fit"]["n"] == 200 and r["width_ratio_fit"]["raw"] == fit["width_ratio"]
    assert E.load_params(tmp)["schema"] == "event_params/v1"


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
