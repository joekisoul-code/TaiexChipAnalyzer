"""2026-10-12 r13：scan() 為 A/A+ 發布 why_d (每項原因的模型分下降，與 why 同順序、四捨五入到 0.001)，App 畫小橫條；B/B+ 沒有。

    python tests/test_treasure_why_d.py

不連網：market_snapshot / twii_hist / valuation / score_one 換成假資料，模型用兩棵樹的小模型 (同 test_treasure_live_ledger._tiny_model)。
"""
import json
import math
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


FEATS = ["ret20", "dd_hi60", "vola20"]
TM = {"features": FEATS, "init": -0.5,
      "trees": [{"r": 0, "n": [[0, 5.0, -1, -2, 1]], "v": [0.0, 1.0]}, {"r": 0, "n": [[1, 0.0, -1, -2, 1]], "v": [0.6, 0.0]}],   # ret20 > 5 加 1.0、dd_hi60 ≤ 0 加 0.6
      "th_A": 0.5, "th_Aplus": 0.64, "gate": {"m_bias20_lt": 0.0}, "universe": ["1101", "2330", "2489", "3008"], "trained_at": "2026-10-11 08:00:00"}
X = {"2489": {"ret20": 12.0, "dd_hi60": -20.0, "vola20": 3.0},    # 兩項都比中位突出 → A+，2 條原因
     "3008": {"ret20": 1.0, "dd_hi60": -3.0, "vola20": 2.0},      # 只有 dd_hi60 突出 → A (p = σ(0.1) ≈ 0.52)，1 條原因
     "2330": {"ret20": 2.0, "dd_hi60": 5.0, "vola20": 3.0},
     "1101": {"ret20": 1.0, "dd_hi60": 6.0, "vola20": 3.0}}
D = "2026-10-12"
ROWS = {c: {"name": "n" + c, "close": 100.0, "open": 99.0, "high": 101.0, "low": 98.0, "volume": 1000, "value": 5e9 - i * 1e8, "pct": 1.0 + i * 0.1, "amp": 2.0}
        for i, c in enumerate(["2489", "3008", "2330", "1101", "9999"])}   # 9999 不在訓練宇宙 → 不進候選


def sig(z):
    return 1 / (1 + math.exp(-z))


def fake_score(code, r, d, mkf, tm):
    x = X[code]
    p = sig(L.T._eval(tm, [x[f] for f in FEATS]))
    tier = "B" if p < tm["th_A"] else ("A+" if p >= tm["th_Aplus"] else "A")
    return {"p": round(p, 4), "tier": tier, "ps": 0.3, "m_bias20": -2.0, "_x": dict(x)}


DATES = ["2026-09-%02d" % d for d in range(1, 31)] + ["2026-10-%02d" % d for d in range(1, 13)]
saved = {k: getattr(L, k) for k in ("market_snapshot", "twii_hist", "valuation", "score_one")}
try:
    L.market_snapshot = lambda date=None: (D, {k: dict(v) for k, v in ROWS.items()})
    L.twii_hist = lambda: {d: 1000.0 for d in DATES}
    L.valuation = lambda: {}
    L.score_one = fake_score
    sc = L.scan(TM, D)
finally:
    for k, v in saved.items():
        setattr(L, k, v)

T = {r["code"]: r for r in sc["treasure"]}
check("候選限定訓練宇宙：4 檔 (9999 不在)、uni_n = 4", set(T) == {"2489", "3008", "2330", "1101"} and sc["uni_n"] == 4)
a = T["2489"]
check("2489 為 A+、原因 2 條、why_d 2 個 (同順序同長度)", a["tier"] == "A+" and len(a["why"]) == 2 and len(a["why_d"]) == 2)
med = {"ret20": 1.5, "dd_hi60": 1.0, "vola20": 3.0}   # 4 檔中位數
ex = L.explain(TM, X["2489"], med)
check("why_d = explain() 的 d 四捨五入到 0.001、與 why 文字一一對應", a["why_d"] == [round(w["d"], 3) for w in ex] and a["why"] == [w["txt"] for w in ex])
p0 = sig(-0.5 + 1.0 + 0.6)
check("數值：換掉 ret20 掉 σ(1.1)−σ(0.1)、換掉 dd_hi60 掉 σ(1.1)−σ(0.5)，由大到小", a["why_d"] == [round(p0 - sig(0.1), 3), round(p0 - sig(0.5), 3)] and a["why_d"][0] > a["why_d"][1] > 0)
check("why_d 數值 (0.225 / 0.128)", a["why_d"] == [0.225, 0.128])
check("why 第一條是 ret20 (近 20 日漲幅)", a["why"][0].startswith("近 20 日漲幅 +12.0%"))
b = T["3008"]
check("3008 為 A、1 條原因 (距 60 日高點)、why_d 1 個 = σ(0.1)−σ(−0.5)", b["tier"] == "A" and len(b["why"]) == 1 and b["why"][0].startswith("距 60 日高點") and b["why_d"] == [round(sig(0.1) - sig(-0.5), 3)])
check("A/A+ 仍帶大盤閘門文字與同分數組 d0", "A 級的必要條件" in a["why_gate"] and a.get("d0") is not None)
check("B 級沒有 why / why_d", all("why_d" not in T[c] and "why" not in T[c] for c in ("2330", "1101")))
check("可序列化為 JSON (發布用)、why_d 都是 float", json.loads(json.dumps(sc, ensure_ascii=False))["treasure"][0]["why_d"] == a["why_d"] and all(isinstance(v, float) for v in a["why_d"]))
print(f"\ntreasure why_d: {ok} passed")
