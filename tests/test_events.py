"""事件預判模組單元測試 (design.md §7 第 1~9 項)。不需網路：台股日曆以規則表 (2026 與 TWSE API 相同) 取代、美股日曆用 NYSE 規則、價格用合成資料。

    python tests/test_events.py          # 純 Python 執行 (印出每項結果)
    python -m pytest tests/test_events.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip import config  # noqa: E402
from chip.predict import events as E  # noqa: E402

TZ = config.TZ


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def offline_calendar():
    """台股日曆 = 規則表 (API 年度的規則已核對等於 TWSE 公告)；美股 = NYSE 規則；不載入 ^GSPC。"""
    E.reset_calendar_cache()
    with patched(E, "tw_holidays", lambda y: (set(E.tw_rule_holidays(y)), "api" if y <= 2026 else "rule")), \
            patched(E, "_load_us_actual", lambda: None):
        yield
    E.reset_calendar_cache()


def _synthetic_prices(start="2022-01-03", end="2026-09-24", beta=0.3, seed=1):
    """合成 0050 (open/close) 與 SOX：0050 跳空 = beta × 前一美股日 SOX log 報酬 + 噪音。"""
    rng = np.random.default_rng(seed)
    us = E.us_sessions(start, "2026-09-29")          # 美股到 09-28 (復市日 09-29 之前)
    sox_r = rng.normal(0, 0.02, len(us))
    sox = pd.DataFrame({"date": us, "close": 3000 * np.exp(np.cumsum(sox_r))})
    xr = dict(zip(us, sox_r))
    tw = E.tw_sessions(start, end)
    rows, c = [], 100.0
    for i, d in enumerate(tw):
        if i == 0:
            rows.append({"date": d, "open": c, "close": c})
            continue
        s = sum(xr.get(u, 0.0) for u in E.us_sessions(tw[i - 1], d))
        o = c * math.exp(beta * s + rng.normal(0, 0.002))
        c = o * math.exp(rng.normal(0, 0.008))
        rows.append({"date": d, "open": o, "close": c})
    return pd.DataFrame(rows), sox


# ------------------------------------------------------------------ 1. 參數
def test_params_schema_and_zero_direction_weight():
    P = E.load_params()
    assert P["schema"] == "event_params/v1"
    assert P["direction"]["vote_weight"] == 0.0 and P["direction"]["votes"] == {}
    for k in ("band_rules", "gap_rules", "iv_expectations", "event_types", "calendar_upcoming"):
        assert k in P
    assert P["band_rules"]["desk_iv_bands"]["multiplier"] == 1.0
    assert P["band_rules"]["post_closure_sqrt_nus"]["k"] == [1]
    assert P["band_rules"]["tsmc_call_2330_k1"]["lambda"] == 1.8
    acc = P["band_rules"]["post_closure_sqrt_nus"]["range_levels_path"]
    assert acc["enabled"] == (acc["acceptance"]["result"] == "pass")       # 開關只能由驗收結果決定
    for key in ("pre_holiday_session_range", "yearend_week_range", "tw_election_T0_hv", "fengguan_iv_k1_widen", "pre_pre_holiday_iv_k1_narrow"):
        assert P["band_rules"][key]["enabled"] is False
    bad = json.loads(json.dumps(P))
    bad["direction"]["vote_weight"] = 0.1
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.json"
        p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        try:
            E.load_params(p)
            raise AssertionError("vote_weight != 0 應該被拒絕")
        except ValueError:
            pass
    E._P_CACHE.clear()


# ------------------------------------------------------------------ 2. n_US
def test_n_us_examples():
    with offline_calendar():
        cases = {"2026-09-29": 3, "2026-10-12": 2, "2026-10-27": 2, "2026-11-27": 0, "2026-12-28": 1, "2027-01-04": 1, "2027-02-11": 8}
        for td, n in cases.items():
            got = E.n_us_between(E.prev_td(td), td)
            assert got == n, (td, got, n)
        assert E.prev_td("2026-09-29") == "2026-09-24" and E.prev_td("2027-02-11") == "2027-02-01"
        assert E.closed_weekdays_between("2027-02-01", "2027-02-11") == 7


def test_rule_holidays_match_twse_2026():
    api_2026 = {"2026-01-01", "2026-02-12", "2026-02-13", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-27",
                "2026-04-03", "2026-04-06", "2026-05-01", "2026-06-19", "2026-09-25", "2026-09-28", "2026-10-09", "2026-10-26", "2026-12-25"}
    assert set(E.tw_rule_holidays(2026)) == api_2026
    r27 = E.tw_rule_holidays(2027)
    assert {"2027-02-02", "2027-02-03", "2027-02-04", "2027-02-05", "2027-02-08", "2027-02-09", "2027-02-10", "2027-03-01"} <= set(r27)
    # review #0：2027 兒童節 04-04 週日 → 補假撞清明 (04-05) → 往後到 04-06 (人事行政總處 116 年行事曆)，不是往前到 04-02
    assert r27.get("2027-04-05") == "清明節" and r27.get("2027-04-06") == "兒童節" and "2027-04-02" not in r27, {k: v for k, v in r27.items() if k[5:7] == "04"}
    with offline_calendar():
        assert E.is_tw_trading("2027-04-02") and not E.is_tw_trading("2027-04-06")
        assert E.prev_td("2027-04-07") == "2027-04-02" and E.n_us_between("2027-04-02", "2027-04-07") == 3
    ny = E.nyse_holidays(2026)
    assert "2026-11-26" in ny and "2026-12-25" in ny and "2026-04-03" in ny and "2026-10-12" not in ny
    assert "2027-12-31" not in E.nyse_holidays(2027)      # 元旦逢週六不補


# ------------------------------------------------------------------ 3. session_factor
def test_session_factor_values():
    with offline_calendar():
        f0 = E.session_factor("2026-11-27")
        assert f0["n_us"] == 0 and f0["range_k1"] == 1.0 and f0["iv"] == 1.0
        f1 = E.session_factor("2026-09-30")
        assert f1["n_us"] == 1 and f1["range_k1"] == 1.0 and f1["iv"] == 1.0
        f3 = E.session_factor("2026-09-29")
        assert f3["n_us"] == 3 and abs(f3["range_k1"] - 1.732) < 1e-3 and f3["iv"] == 1.0 and f3["range_mode"] == "base_event"
        f8 = E.session_factor("2027-02-11")
        assert abs(f8["range_k1"] - math.sqrt(8)) < 1e-3 and "lny_reopen" in f8["tags"]
        assert "us_holiday_no_session" in f0["tags"] and "post_closure_session" in f3["tags"]


# ------------------------------------------------------------------ 4. 夜盤
def test_night_mode_vs_base_event():
    with offline_calendar():
        a = E.session_factor("2026-09-23", {"used": True})          # 一般交易日、夜盤對齊 → 維持 night
        assert a["range_mode"] == "night" and a["range_k1"] == 1.0 and a["n_us_uncovered"] == 0
        b = E.session_factor("2026-09-29", {"used": True})          # 復市日：夜盤只涵蓋 09-24 → 2 個美股日未涵蓋 → base_event × √3
        assert b["range_mode"] == "base_event" and b["n_us_uncovered"] == 2 and abs(b["range_k1"] - math.sqrt(3)) < 1e-3
        c = E.session_factor("2026-09-29", {"used": False})
        assert c["n_us_uncovered"] == 3 and c["range_mode"] == "base_event"


def test_range_levels_integration():
    """range_levels.attach_to_next_days：一般日保留夜盤模式；復市日退回 base 並把 k=1 放寬 √n_US (驗收 A 通過時)。"""
    from chip.predict import range_levels as RL
    if not RL.load_multipliers():
        print("  (skip: range_levels.json 不存在)")
        return
    with offline_calendar():
        rng = np.random.default_rng(3)
        days = E.tw_sessions("2025-01-02", "2026-09-24")
        c = 20000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        fr = pd.DataFrame({"date": days, "open": c, "high": c * 1.006, "low": c * 0.994, "close": c})

        def nd_for(last):
            ds = [E.next_td(last, i) for i in (1, 2, 3)]
            return [{"n": i + 1, "date": ds[i], "label": ["隔天", "後天", "第三天"][i]} for i in range(3)]

        with patched(RL, "_night_final", lambda snap, nd, scored: (0.8, None)):
            base = nd_for("2026-09-24")
            hdr = RL.attach_to_next_days(base, fr, {"phase": "closed"}, float(c[-1]))
            fr2 = fr[fr["date"] <= "2026-09-22"]
            normal = nd_for("2026-09-22")
            RL.attach_to_next_days(normal, fr2, {"phase": "closed"}, float(fr2["close"].iloc[-1]))
        assert normal[0]["range_mode"] == "night" and normal[0]["event_range_factor"] == 1.0
        on = E.range_levels_enabled()
        if on:
            assert base[0]["range_mode"] == "base_event", base[0]["range_mode"]
            assert abs(base[0]["event_range_factor"] - math.sqrt(3)) < 1e-3
            ref = RL.range_levels(fr, 1, night_ret=None, base_px=float(c[-1]))
            assert abs(base[0]["path_low20"] - round(ref["low20"] * math.sqrt(3), 2)) < 0.02
            assert base[1]["event_range_factor"] == 1.0 and base[1]["range_mode"] == "base"      # k≥2 不放寬 (只做路徑單調)
            assert base[1]["path_low20"] <= base[0]["path_low20"] and base[1]["path_high80"] >= base[0]["path_high80"]
            # review #17：只有 k=1 的說明可以寫 × √n_US；k=2/3 未放寬
            assert "√n_US" in base[0]["range_note"], base[0]["range_note"]
            assert "改用 base 模式 × √n_US)" not in hdr and "只有 1 日帶 × √n_US，k≥2 不放寬" in hdr, hdr
            for x in base[1:]:
                assert "√n_US" not in x["range_note"] and "k≥2 不放寬" in x["range_note"], x["range_note"]
        else:
            assert base[0]["event_range_factor"] == 1.0
        assert "post_closure_session" in base[0]["event_tags"]


# ------------------------------------------------------------------ 5. 禁用字
def _build_offline(now=None, last="2026-09-24"):
    px, sox = _synthetic_prices()
    return E.build(today=(now or dt.datetime(2026, 9, 27, 10, tzinfo=TZ)).date(), write_calendar=False,
                   ctx={"last_td": last, "px0050": px, "sox": sox, "tsm": sox, "exdiv_0050": [], "now": now or dt.datetime(2026, 9, 27, 10, tzinfo=TZ),
                        "twii": px[["date", "close"]]})


def test_no_banned_phrases():
    P = E.load_params()
    banned = P["direction"]["banned_phrases_zh"]
    with offline_calendar():
        ev = _build_offline()
    texts = E.all_texts(ev) + [u.get("title_zh", "") for u in ev["upcoming"]] + [u.get("dir_text", "") for u in ev["upcoming"]] \
        + [u.get("notes_zh", "") for u in ev["upcoming"]] + [x.get("notes_zh", "") for x in P["event_types"].values()]
    assert len(ev["upcoming"]) > 20
    for t in texts:
        for w in banned:
            assert w not in t, (w, t)
    for u in ev["upcoming"]:                      # 方向描述一律標未通過驗證 / 無效應
        assert ("未通過驗證" in u["dir_text"]) or ("無可辨識" in u["dir_text"]), u["dir_text"]
        assert u["grade_dir"] in ("無", "弱")
        assert u["date_confidence"] in ("confirmed", "scheduled", "estimated", "low")
    assert ev["direction_vote_weight"] == 0.0


# ------------------------------------------------------------------ 6. 方向模組不讀 events
def test_direction_modules_do_not_read_events():
    for f in ("verdict.py", "short_term.py", "five_day.py"):
        src = (ROOT / "chip" / "predict" / f).read_text(encoding="utf-8")
        assert not re.search(r"\bevents\b|event_tags|event_range_factor|from \. import events|\bEV\.", src), f


# ------------------------------------------------------------------ 7. API 失敗 → 規則 + estimated，build 不拋錯
def test_api_failure_falls_back_to_rules():
    from chip.sources import twse

    def boom(y=None):
        raise RuntimeError("HTTP 402 / timeout (mock)")
    E.reset_calendar_cache()
    with patched(twse, "holidays", boom), patched(E, "_load_us_actual", lambda: None):
        h, src = E.tw_holidays(2026)
        assert src == "rule" and "2026-09-28" in h
        cal = E.refresh_calendar(dt.date(2026, 9, 27), write=False)
        assert cal["health"]["stale"] is True and all(s == "rule" for s in cal["health"]["tw_sources"].values())
        pc = [e for e in cal["events"] if e["event_type"] == "post_closure_session"]
        assert pc and all(e["date_confidence"] == "estimated" for e in pc)
        px, sox = _synthetic_prices()
        ev = E.build(today=dt.date(2026, 9, 27), write_calendar=False,
                     ctx={"last_td": "2026-09-24", "px0050": px, "sox": sox, "tsm": sox, "now": dt.datetime(2026, 9, 27, 10, tzinfo=TZ)})
        assert ev["next_session"]["date"] == "2026-09-29"
    E.reset_calendar_cache()


# ------------------------------------------------------------------ 8. 台積電法說
def test_tsmc_multiplier_only_when_confirmed():
    with offline_calendar():
        before_adr = dt.datetime(2026, 10, 15, 14, 30, tzinfo=TZ)
        t = E.tsmc_2330_k1("2026-10-15", now=before_adr)
        assert t["applied"] and t["factor"] == 1.8 and t["date_confidence"] == "confirmed"
        t2 = E.tsmc_2330_k1("2026-10-15", now=dt.datetime(2026, 10, 16, 7, 0, tzinfo=TZ))     # ADR 已收盤 → 不套
        assert not t2["applied"] and t2["factor"] == 1.0
        t3 = E.tsmc_2330_k1("2027-01-14", now=dt.datetime(2027, 1, 14, 14, 30, tzinfo=TZ))     # 估計日期 → 不套
        assert t3.get("date_confidence") == "estimated" and t3["factor"] == 1.0 and not t3["applied"]
        assert E.tsmc_2330_k1("2026-10-14", now=before_adr)["factor"] == 1.0
        # desk 套用：k=1 四分位 ×1.8、k≥2 不動、原值保留
        from chip.analysis import desk
        rng = {"levels": {"1": {q: {"pct": p, "px": 1000 * (1 + p / 100)} for q, p in (("low10", -2.0), ("low20", -1.2), ("high80", 1.1), ("high90", 1.9))},
                          "5": {"low20": {"pct": -3.0, "px": 970.0}}}}
        with patched(E, "tsmc_2330_k1", lambda d, now=None, P=None: {"call_date": d, "factor": 1.8, "applied": True, "date_confidence": "confirmed"}):
            desk._event_2330(rng, "2026-10-15", 1000.0)
        assert rng["levels"]["1"]["low20"]["pct"] == round(-1.2 * 1.8, 2) and rng["levels"]["1"]["low20"]["pct_base"] == -1.2
        assert rng["levels"]["5"]["low20"]["pct"] == -3.0 and rng["event_factor"] == {"1": 1.8}
        led = []
        desk.ledger_add(led, "2330", "2026-10-15", 1000.0, {**rng, "ends": {"1": "2026-10-16"}}, lambda a, b: ["tsmc_call"], E.load_params())
        assert led and led[0]["event_factor_applied"] == 1.8 and led[0]["event_tags"] == ["tsmc_call"] and "levels_base" in led[0]
        plain = {"levels": {"1": {q: {"pct": -1.0, "px": 990.0 + i} for i, q in enumerate(desk.QS)}}, "ends": {"1": "2026-10-16"}}
        desk.ledger_add(led, "2330", "2026-10-15", 1000.0, plain, None, None)            # 之後未放寬的重算不覆蓋
        assert led[0]["event_factor_applied"] == 1.8 and len(led) == 1


def test_desk_exdiv_uses_band_window():
    """review #1：β_gap 用 px_band (~2,200 天) → 除息清單要用 divs_band，不能只用 1,100 天視窗的 divs。"""
    from chip.analysis import desk
    frames = {"0050": {"divs": [{"date": "2024-01-17"}], "divs_band": [{"date": "2023-01-30"}, {"date": "2023-07-18"}, {"date": "2024-01-17"}]}}
    exd = {"0050": {"upcoming": [{"ex_date": "2026-10-20"}]}}
    got = desk._exdiv_0050(frames, exd)
    assert got == ["2023-01-30", "2023-07-18", "2024-01-17", "2026-10-20"], got
    assert desk._exdiv_0050({"0050": {"divs": [{"date": "2024-01-17"}]}}, {}) == ["2024-01-17"]      # 舊 frames 無 divs_band 仍可用
    src = (ROOT / "chip" / "analysis" / "desk.py").read_text(encoding="utf-8")
    assert '"divs_band": divs_all' in src and "exd50 = _exdiv_0050(frames, exd)" in src


# ------------------------------------------------------------------ 9. 跳空卡
def test_gap_card_pending_and_ready():
    with offline_calendar():
        px, sox = _synthetic_prices(beta=0.3)
        now = dt.datetime(2026, 9, 27, 10, tzinfo=TZ)
        g = E.gap_forecast("2026-09-29", "2026-09-24", px, sox[sox["date"] <= "2026-09-25"], None, (), 1.0, now=now)
        assert g["status"] == "pending_us" and g["est_0050_pct"] is None and g["n_us"] == 3 and g["n_us_closed"] == 2
        g2 = E.gap_forecast("2026-09-29", "2026-09-24", px, sox, None, (), 1.0, now=dt.datetime(2026, 9, 29, 7, tzinfo=TZ))
        assert g2["status"] == "ready" and abs(g2["beta"] - 0.3) < 0.03, g2
        x = math.log(sox.set_index("date").loc["2026-09-28", "close"] / sox.set_index("date").loc["2026-09-23", "close"])
        assert abs(g2["est_0050_pct"] - g2["beta"] * x * 100) < 0.02
        # review #20：± 殘差依來源選 (SOX 1.23 σ60、TSM 1.17 σ60；ge2wd)
        oos = E.load_params()["gap_rules"]["post_closure_gap_forecast"]["oos"]["gap_ge2wd"]["resid_rms_sigma60"]
        assert g2["source"] == "SOX" and g2["resid_rms_sigma60"] == oos["sox"] == 1.23 and g2["resid_sd_pct"] == 1.23, g2
        assert g2["sign_hit_oos"] == 0.89 and g2["r2_oos"] == 0.62
        gt = E.gap_forecast("2026-09-29", "2026-09-24", px, None, sox, (), 1.0, now=dt.datetime(2026, 9, 29, 7, tzinfo=TZ))
        assert gt["source"] == "TSM" and gt["resid_rms_sigma60"] == oos["tsm"] == 1.17 and gt["r2_oos"] == 0.62
        g3 = E.gap_forecast("2026-09-29", "2026-09-24", px, sox, None, ("2026-09-29",), 1.0, now=now)
        assert g3["status"] == "exdiv_excluded"
        g0 = E.gap_forecast("2026-11-27", "2026-11-26", px, sox, None, (), 1.0, now=now)
        assert g0["status"] == "no_us_session" and g0["gap_size_factor"] == 0.76
        assert E.gap_forecast("2026-09-30", "2026-09-29", px, sox, None, (), 1.0, now=now) is None      # 一般日不出卡
        ev = _build_offline()
        ns = ev["next_session"]
        assert ns["date"] == "2026-09-29" and ns["n_us"] == 3 and ns["gap"]["status"] == "pending_us"
        assert ns["iv_expect"]["ivk5_dlog"] == -0.084 and ns["iv_band_factor"] == 1.0
        f = E.for_forecast(ev)
        assert len(f["upcoming"]) == 10 and f["next_session"]["date"] == "2026-09-29"


def test_review_texts_and_badges():
    """review #12/#14/#15/#16：文案與證據徽章不可誇大或與參數矛盾。"""
    P = E.load_params()
    with offline_calendar():
        cal = E.refresh_calendar(dt.date(2026, 9, 27), write=False)
        ups = E.upcoming(cal, "2026-09-28", P, today="2026-09-27")
        by = {}
        for u in ups:
            by.setdefault((u["event_type"], u.get("role")), u)
        # #12 連假前一日 / 年底週：驗收 B 已跑且未通過 → 不可再寫「未做帶寬測試 / 待驗收」
        for k in ("pre_holiday_session", "tw_yearend_last5"):
            vt = by[(k, None)]["vol_text"]
            assert "驗收 B 未通過" in vt and "未做帶寬測試" not in vt and "待驗收" not in vt, vt
            assert by[(k, None)]["band"]["acceptance_B_pass"] is False
        for k in ("pre_holiday_session_range", "yearend_week_range"):
            r = P["band_rules"][k]
            assert r["acceptance_B"]["pass"] is False and "fail(band)" in r["verified_status"] and "untested" not in r["verified_status"]
        # #15 春節 ivk5：dlog −0.216 → 約 19% (不是 22%)
        lny = [u for u in ups if u["event_type"] == "lny_reopen"]
        assert lny and "機械下降約 19%" in lny[0]["vol_text"] and "22%" not in lny[0]["vol_text"], lny[0]["vol_text"]
        assert abs(E.dlog_pct(-0.216) - 19.4) < 0.1 and abs(E.dlog_pct(-0.084) - 8.1) < 0.1
        # #16 前一晚美股休市：全日波動徽章 = 無，跳空另給 grade_gap = 強；選舉 T-1 波動 = 無、T0 = 中
        nu = by[("us_holiday_no_session", None)]
        assert nu["grade_vol"] == "無" and nu.get("grade_gap") == "強", nu
        assert by[("tw_local_election", "T-1")]["grade_vol"] == "無" and by[("tw_local_election", "T0")]["grade_vol"] == "中"
        assert "grade_gap" in E.for_forecast({"upcoming": [nu]})["upcoming"][0]
        # #14 選舉 T0 iv5：≥1.2 → 引用實際案例 (21%、40%)；<1.2 → 幅度不定；不再出現「10~20%」「小幅下降」
        def hist(ratio):
            ds = E.tw_sessions("2026-10-01", "2026-11-27")[-21:]
            return [{"date": d, "iv5": 0.15} for d in ds[:-1]] + [{"date": ds[-1], "iv5": 0.15 * ratio}]
        hi = E.iv_expect("2026-11-27", "2026-11-30", hist(1.5), P, cal)["election_iv5"]
        assert hi["t1_ratio"] == 1.5 and hi["expect_pct_range"] == [-40, -21] and "21%、40%" in hi["text"] and "10~20" not in hi["text"], hi
        lo = E.iv_expect("2026-11-27", "2026-11-30", hist(1.0), P, cal)["election_iv5"]
        assert lo["expect_pct_range"] is None and "幅度不定" in lo["text"] and "小幅下降" not in lo["text"] and "-25%~+16%" in lo["text"], lo
        na = E.iv_expect("2026-11-27", "2026-11-30", None, P, cal)["election_iv5"]
        assert na["t1_ratio"] is None and "平均下降約 13%" in na["text"] and "6/8" in na["text"], na
        for x in (hi, lo, na):
            assert x["mean_pct"] == -13 and x["n"] == 8
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # #19 README 符號命中分類列出 (1 平日 0.79)
    assert "符號命中 0.88~1.0" not in readme and "1 平日 0.79" in readme


def test_election_and_calendar_mapping():
    with offline_calendar():
        cal = E.refresh_calendar(dt.date(2026, 9, 27), write=False)
        by = {(e["event_type"], e.get("role")): e for e in cal["events"]}
        assert by[("tw_local_election", "T0")]["tw_session"] == "2026-11-30"
        assert by[("tw_local_election", "T-1")]["tw_session"] == "2026-11-27"
        assert by[("us_midterm", None)]["tw_session"] == "2026-11-04"
        fomc = [e for e in cal["events"] if e["event_type"] == "fomc"]
        assert fomc[0]["date"] == "2026-10-28" and fomc[0]["tw_session"] == "2026-10-29"
        tsmc = [e for e in cal["events"] if e["event_type"] == "tsmc_call"]
        assert tsmc[0]["date"] == "2026-10-15" and tsmc[0]["tw_session"] == "2026-10-16" and tsmc[0]["date_confidence"] == "confirmed"
        nfp = [e for e in cal["events"] if e["event_type"] == "us_nfp"]
        assert [e["tw_session"] for e in nfp[:3]] == ["2026-10-05", "2026-11-09", "2026-12-07"]
        lny = [e for e in cal["events"] if e["event_type"] == "lny_reopen"]
        assert lny and lny[0]["tw_session"] == "2027-02-11" and lny[0]["date_confidence"] == "estimated" and lny[0]["beyond_horizon"]


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
