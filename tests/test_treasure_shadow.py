"""2026-10-12 影子觀察「嚴格 A」：python tests/test_treasure_shadow.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chip.predict import treasure_live as L  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print("  ✓", name)


TM = {"th_A": 0.50, "th_Aplus": 0.64}            # 內插 0.92 分位 = 0.50 + 0.14 × 2/7 = 0.54
th, approx = L.strict_a_th({}, TM)
check("沒有 th_q92 → 0.90/0.97 分位間內插 (近似)", abs(th - 0.54) < 1e-9 and approx)
check("列上有 th_q92 → 直接用 (非近似)", L.strict_a_th({"th_q92": 0.61}, TM) == (0.61, False))
check("模型有 th_q92 (舊列沒記) → 用模型的，仍標近似", L.strict_a_th({}, {"th_q92": 0.57}) == (0.57, True))
check("列的內插優先於模型", abs(L.strict_a_th({"th_A": 0.4, "th_Aplus": 0.47}, TM)[0] - 0.42) < 1e-9)

mbd = {"2026-07-20": -7.0, "2026-08-03": -1.27}
row = lambda **k: {"tier": "A", "p": 0.58, "date": "2026-07-20", "status": "命中", "cur": 10, "fin21": 10, **k}   # noqa: E731
check("A、乖離 −7 (由當日掃描補)、分數 ≥ 門檻 → 嚴格 A", L.is_strict_a(row(), TM, mbd) is True)
check("乖離 −1.27 (> −1.5) → 不是", L.is_strict_a(row(date="2026-08-03"), TM, mbd) is False)
check("分數低於門檻 → 不是", L.is_strict_a(row(p=0.52), TM, mbd) is False)
check("B 級 → 不是", L.is_strict_a(row(tier="B+"), TM, mbd) is False)
check("缺大盤乖離 (無列值、無掃描) → None", L.is_strict_a(row(date="2026-09-01"), TM, mbd) is None)
check("列上的 mkt_bias20 優先於掃描", L.is_strict_a(row(mkt_bias20=-0.5), TM, mbd) is False)

prev = {"ledger": {"treasure": [row(code="1"), row(code="2", status="未命中", cur=-5, fin21=-5), row(code="3", date="2026-08-03"),
                                row(code="4", status="追蹤", fin21=None), row(code="5", date="2026-09-01")], "surge": []},
        "scans": [{"date": d, "mkt_bias20": v} for d, v in mbd.items()]}
sh = L.stats(prev, TM)["treasure"]["shadow"]["strict_A"]
check("stats：嚴格 A 結案 2 (命中 1)、追蹤 1、資料不足 1、近似門檻 3、正式 A/A+ 5", sh["n"] == 2 and sh["hit"] == 1 and sh["tracking"] == 1 and sh["n_unknown"] == 1 and sh["n_approx_th"] == 3 and sh["n_signal"] == 5)
check("期滿勝率 / 平均 21 日 與規則文字", sh["win21"] == 0.5 and sh["avg_fin21"] == 2.5 and "前 8%" in sh["rule"] and "不是正式訊號" in sh["note"])
check("沒有 A 級 → n=0、不丟例外", L.stats({"ledger": {"treasure": [], "surge": []}, "scans": []}, TM)["treasure"]["shadow"]["strict_A"]["n"] == 0)

# record：新列記下 th_q92
p0 = {"ledger": {"treasure": [], "surge": []}, "scans": []}
sc = {"date": "2026-10-12", "mkt_bias20": -2.0, "mkt_close": 1, "treasure": [{"code": "9", "name": "x", "close": 10, "p": 0.6, "tier": "A", "pct": 1.0}], "surge": []}
r = L.record(p0, sc, {"th_A": 0.5, "th_Aplus": 0.64, "th_q92": 0.55, "trained_at": "t"})["ledger"]["treasure"][0]
check("record：新列帶 th_q92 與 mkt_bias20 → 之後判定不用近似", r.get("th_q92") == 0.55 and r.get("mkt_bias20") == -2.0 and L.strict_a_th(r)[1] is False)

# BE-9：舊列沒記 mkt_bias20 → record() 趁 scans 還在時寫進列本身 (mb_src='scan')；之後 scans 只留 400 天、那天被截掉也分得出嚴格 A
legacy = row(code="7", date="2026-07-20", status="命中")
p1 = {"ledger": {"treasure": [legacy], "surge": []}, "scans": [{"date": "2026-07-20", "mkt_bias20": -7.0}]}
p1 = L.record(p1, sc, TM)
check("record：舊 A 列補上當日掃描的大盤乖離 (mb_src=scan)", legacy.get("mkt_bias20") == -7.0 and legacy.get("mb_src") == "scan")
new = next(x for x in p1["ledger"]["treasure"] if x["code"] == "9")
check("record：列本身已有的大盤乖離不被覆蓋", new["mkt_bias20"] == -2.0 and "mb_src" not in new)
p1["scans"] = [x for x in p1["scans"] if x["date"] != "2026-07-20"]          # 模擬 scans[-400:] 截掉那天
mbd1 = {x["date"]: x.get("mkt_bias20") for x in p1["scans"]}
check("那天的掃描紀錄被截掉後，舊列仍判得出嚴格 A (不是資料不足)", L.is_strict_a(legacy, TM, mbd1) is True and L.stats(p1, TM)["treasure"]["shadow"]["strict_A"]["n_unknown"] == 0)
print(f"\ntreasure shadow: {ok} passed")
