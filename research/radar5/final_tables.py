"""潛力股雷達 研究 5 → 正式常數 (10-07)：時點正確 (PIT) 前 170 大成交值宇宙重建 RADAR_BT / RADAR_PATH / RADAR_FAIL / 推薦當天 A 級分數校準。
只讀資料；輸出 final/final_constants.json (由 patch 腳本寫進 treasure_live.py)。
- 模型：正式特徵/參數，訓練與候選都只用「當月 PIT 前 170 名」(120 日平均成交值，前一交易日為止)，逐年走動式 (2020 起樣本外，2018 起訓練，40 天隔離)。
- 5 個種子 × 2 條選股路徑 (2020 起連續模擬 / 2022-01 冷啟動) = 10 組推薦 → 顯示數字取 10 組平均 + 範圍 (驗證代理建議：單一種子/路徑的格子噪音 ±3~13pt)。
- 顯示期間 2022-01~2026-09 (冷啟動路徑本來就只有這段；連續路徑截掉 2020~21)。查表另外用 2020~21 的推薦當訓練資料 (X2 驗證：Brier 12/12 改善)。
- 飆股追蹤表改用「仍在追蹤」母體 (S1：舊表含已飆 +20% 結案的推薦 → 後段高估 30~50pt)。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import harness as H  # noqa: E402
from chip.predict import treasure as T  # noqa: E402

OUT = Path(__file__).parent / "final"; OUT.mkdir(exist_ok=True)
t0 = time.time()
SEEDS = [None, 1, 2, 3, 4]
KS = [1, 2, 3, 5, 10, 15]
EDGES = [-1e9, -8, -5, -2, 2, 5, 1e9]
LAB = ["≤−8%", "−8~−5%", "−5~−2%", "−2~+2%", "+2~+5%", "≥+5%"]
DISP0, DISP1 = "2022-01-01", "2026-09-30"

F = H.load_feat()
pit = F["rk"] <= 170
H._periods.__defaults__ = ("2020-01-01",)     # 2020 起樣本外 (2020~21 只當查表訓練資料)
prm = {**T.PARAMS, "n_jobs": 12}; sprm = {**T.SURGE_PARAMS, "n_jobs": 12}

sets = []   # (set_id, seed, path, picks DataFrame)
pool_rows = []
for sd in SEEDS:
    P = H.walk_forward(F, pit, pit, cadence="Y", params=prm, sparams=sprm, seeds=(sd,))
    for path, start in (("cont", "2020-01-01"), ("cold", DISP0)):
        R = H.simulate(P[P["date"] >= start], P[P["date"] >= start]["rk"] <= 170)
        sets.append((f"{sd}-{path}", sd, path, R))
    if sd is None:
        PB = P[(P["date"] >= DISP0) & (P["rk"] <= 170)].copy()
    print("seed", sd, "picks", [len(s[3]) for s in sets[-2:]], round(time.time() - t0), flush=True)
NS = len(sets)


def disp(R):
    return R[(R["date"] >= DISP0) & (R["date"] <= DISP1)]


def metr(x, ok, ret):
    x = x[x[ok].notna() & x[ret].notna()]
    if not len(x):
        return None
    return {"n": len(x), "hit": float(x[ok].mean()), "win": float((x[ret] > 0).mean()), "avg": float(x[ret].mean()), "med": float(x[ret].median()), "q10": float(x[ret].quantile(.1))}


def tiers(R):
    Tt = R[R["radar"] == "T"]; S = R[R["radar"] == "S"]
    keyS = set(zip(S["date"], S["code"]))
    dual = Tt[[k in keyS for k in zip(Tt["date"], Tt["code"])]]
    return {"A": (Tt[Tt["tier"].isin(["A", "A+"])], "hit", "fin21"), "A+": (Tt[Tt["tier"] == "A+"], "hit", "fin21"), "B": (Tt[Tt["tier"].isin(["B", "B+"])], "hit", "fin21"),
            "S3": (S, "surge", "fin20"), "S3h": (S, "hit", "fin21"), "dual": (dual, "hit", "fin21"),
            "A-": (Tt[Tt["tier"] == "A"], "hit", "fin21"), "B+": (Tt[Tt["tier"] == "B+"], "hit", "fin21"), "Bo": (Tt[Tt["tier"] == "B"], "hit", "fin21")}


def summarize(filter_fn):
    acc = {}
    for _, _, _, R in sets:
        for k, (x, ok, ret) in tiers(filter_fn(R)).items():
            m = metr(x, ok, ret)
            if m:
                acc.setdefault(k, []).append(m)
    out = {}
    for k, L in acc.items():
        mean = {f: float(np.mean([m[f] for m in L])) for f in L[0]}
        out[k] = {**{f: round(v, 3 if f in ("hit", "win") else 2) for f, v in mean.items()}, "n": int(round(mean["n"])),
                  "win_rng": [round(min(m["win"] for m in L), 3), round(max(m["win"] for m in L), 3)], "hit_rng": [round(min(m["hit"] for m in L), 3), round(max(m["hit"] for m in L), 3)]}
    return out


def bt_block(S):
    """轉成 RADAR_BT 欄位：A/A+/B/dual 用 hit/win21/fin21；S3 用 surge + 20 日勝率 (沿用鍵名 win21，App btCell 讀它)。"""
    def tr(m, surge_from=None):
        if not m:
            return None
        return {"n": m["n"], "hit": m["hit"], "win21": m["win"], "fin21": m["avg"], "med21": m["med"], "q10": m["q10"], "win_rng": m["win_rng"], "hit_rng": m["hit_rng"]}
    o = {k: tr(S.get(k)) for k in ("A", "A+", "B", "dual")}
    s3 = S.get("S3"); s3h = S.get("S3h") or {}
    if s3:
        o["S3"] = {"n": s3["n"], "surge": s3["hit"], "hit": s3h.get("hit"), "win21": s3["win"], "fin21": s3["avg"], "med21": s3["med"], "q10": s3["q10"], "win_rng": s3["win_rng"], "surge_rng": s3["hit_rng"]}
    return o


full = summarize(disp)
y26 = summarize(lambda R: R[(R["date"] >= "2026-01-01") & (R["date"] <= DISP1)])
same = summarize(lambda R: R[(R["date"] >= "2026-07-01") & (R["date"] <= "2026-09-03")])
# 同日候選池基準 (只算已有標籤的列；pool_base 會把未結案列當輸)
E = PB.assign(screen=H.screen(PB))
a40 = E.groupby("date", group_keys=False).apply(lambda g: g.nlargest(40, "screen"))
a80 = E.groupby("date", group_keys=False).apply(lambda g: g.nlargest(80, "screen"))
a40 = a40[a40["fin21"].notna()]; a80 = a80[a80["fin20"].notna()]
pool = {"n": int(len(a40)), "hit": round(float(a40["hit"].mean()), 3), "win21": round(float((a40["fin21"] > 0).mean()), 3), "fin21": round(float(a40["fin21"].mean()), 2),
        "surge": round(float(a80["surge"].mean()), 3), "win20": round(float((a80["fin20"] > 0).mean()), 3)}

# ---------- 路徑 ----------
Pn = H.panel()
CL, HI, LO, IX = {}, {}, {}, {}
for c, g in Pn.groupby("code"):
    g = g.sort_values("date"); CL[c] = g["close"].to_numpy(float); HI[c] = g["high"].fillna(g["close"]).to_numpy(float); LO[c] = g["low"].fillna(g["close"]).to_numpy(float)
    IX[c] = {d: i for i, d in enumerate(g["date"].tolist())}


def binlab(v):
    for e, l in zip(EDGES[1:], LAB):
        if v <= e:
            return l
    return LAB[-1]


rowsA, rowsS = [], []
for sid, sd, path, R in sets:
    # 查表訓練資料：連續路徑 2020~ 全部；冷啟動路徑 2022~ (兩者都納入、各算一組，n 以組數平均)
    for r in R.itertuples():
        i = IX.get(r.code, {}).get(r.date)
        if i is None:
            continue
        c0 = CL[r.code][i]; n = len(CL[r.code])
        if r.radar == "T" and r.tier in ("A", "A+") and pd.notna(r.hit) and pd.notna(r.fin21):
            for k in KS:
                if i + k < n:
                    rowsA.append((sid, r.code, r.date, k, (CL[r.code][i + k] / c0 - 1) * 100, r.hit, r.fin21, r.year))
        if r.radar == "S" and pd.notna(r.surge) and pd.notna(r.fin20):
            for d in range(1, 20):
                if i + d >= n:
                    break
                hi = (HI[r.code][i + 1:i + d + 1].max() / c0 - 1) * 100; lo = (LO[r.code][i + 1:i + d + 1].min() / c0 - 1) * 100
                if hi >= T.SURGE_UP or lo <= T.SURGE_DN:      # App _eval_surge：碰到 +20% 或 −10% 就結案，不再顯示機率
                    break
                k = max(x for x in KS if x <= d)
                rowsS.append((sid, r.code, r.date, k, d, (CL[r.code][i + d] / c0 - 1) * 100, r.surge, r.fin20, r.year))
A = pd.DataFrame(rowsA, columns=["sid", "code", "date", "k", "cur", "hit", "fin", "year"])
S = pd.DataFrame(rowsS, columns=["sid", "code", "date", "k", "d", "cur", "hit", "fin", "year"])
A["bin"] = A["cur"].map(binlab); S["bin"] = S["cur"].map(binlab)
# 同一組推薦內，cold 與 cont 在 2022~ 大部分重疊 → 以組數平均 n；機率用全部列 (等權每組)
print("path rows", len(A), len(S), round(time.time() - t0), flush=True)


def table(D, unique_picks: bool):
    out = {}
    for k in KS:
        rows = []
        g = D[D["k"] == k]
        for lab in LAB:
            x = g[g["bin"] == lab]
            if not len(x):
                continue
            row = {"bin": lab, "n": int(round((x[["sid", "code", "date"]].drop_duplicates().shape[0] if unique_picks else len(x)) / NS)),
                   "hit": round(float(x["hit"].mean()), 3), "pos": round(float((x["fin"] > 0).mean()), 3), "avg": round(float(x["fin"].mean()), 2)}
            if unique_picks:
                row["nd"] = int(round(len(x) / NS))
            rows.append(row)
        out[str(k)] = rows
    return out


PATH = {"A": table(A, False), "S3": table(S, True)}


def wf_brier(D, okcol="hit"):
    """走動式：用 < y 年的列建表 (各組合併)、測 y 年 (只用 None-cont 一組) → 表 vs 常數 Brier。"""
    res = []
    for y in (2023, 2024, 2025, 2026):
        tr = D[D["year"] < y]; te = D[(D["year"] == y) & (D["sid"] == "None-cont")]
        if len(te) < 30:
            continue
        rate = tr.groupby(["k", "bin"])[okcol].mean(); base = tr[okcol].mean()
        pred = np.array([rate.get((k, b), base) for k, b in zip(te["k"], te["bin"])])
        res.append({"year": y, "table": round(float(((pred - te[okcol]) ** 2).mean()), 4), "base": round(float(((base - te[okcol]) ** 2).mean()), 4), "n": int(len(te))})
    return res


wfA, wfS = wf_brier(A), wf_brier(S)

# ---------- 失敗分析 (顯示期間、10 組合併) ----------
mk = pd.read_parquet(Path(__file__).parent / "twii_10y.parquet")
mk["r"] = mk["close"].pct_change() * 100; mk["m_vola20"] = mk["r"].rolling(20).std(); mk["m_bias60"] = (mk["close"] / mk["close"].rolling(60).mean() - 1) * 100
MK = mk.set_index("date")[["m_vola20", "m_bias60"]]
allp = pd.concat([disp(R).assign(sid=sid) for sid, _, _, R in sets], ignore_index=True)
allp = allp.join(MK, on="date")
r3 = []
for r in allp.itertuples():
    i = IX.get(r.code, {}).get(r.date)
    r3.append((CL[r.code][i + 3] / CL[r.code][i] - 1) * 100 if i is not None and i + 3 < len(CL[r.code]) else np.nan)
allp["r3"] = r3


def early(G, ok, ret):
    G = G[G[ok].notna() & G[ret].notna() & G["r3"].notna()]
    m = G["r3"] <= -5; lose = (G[ok] == 0) & (G[ret] < 0)
    return {"rule": "第 3 天收盤 ≤ −5%", "share": round(float(m.mean()), 3), "hit": round(float(G[m][ok].mean()), 3), "lose": round(float(lose[m].mean()), 3),
            "hold": round(float(G[m][ret].mean()), 2), "exit": round(float(G[m]["r3"].mean()), 2), "n": int(round(m.sum() / NS))}


GA = allp[(allp["radar"] == "T") & allp["tier"].isin(["A", "A+"]) & allp["hit"].notna() & allp["fin21"].notna()]
GS = allp[(allp["radar"] == "S") & allp["surge"].notna() & allp["fin20"].notna()]
loseA = GA[(GA["hit"] == 0) & (GA["fin21"] < 0)]; winA = GA[GA["hit"] == 1]
yrsA = []
for y, g in GA.groupby("year"):
    l, w = g[(g["hit"] == 0) & (g["fin21"] < 0)], g[g["hit"] == 1]
    if len(l) >= 8 and len(w) >= 8:
        yrsA.append(l["m_vola20"].median() < w["m_vola20"].median())
loseS = GS[(GS["surge"] == 0) & (GS["fin20"] < 0)]; winS = GS[GS["surge"] == 1]
yrsS = []
for y, g in GS.groupby("year"):
    l, w = g[(g["surge"] == 0) & (g["fin20"] < 0)], g[g["surge"] == 1]
    if len(l) >= 8 and len(w) >= 8:
        yrsS.append(l["ps"].median() < w["ps"].median())
# 飆股分數走動式濾網：用之前年份的 20 分位排除最低兩成
dh = []
for sid, _, _, R in sets:
    S_ = R[(R["radar"] == "S") & R["surge"].notna()]
    for y in range(2022, 2027):
        past, cur = S_[S_["year"] < y], S_[S_["year"] == y]
        if len(past) < 80 or len(cur) < 20:
            continue
        th = past["ps"].quantile(0.2); keep = cur[cur["ps"] >= th]
        dh.append(keep["surge"].mean() - cur["surge"].mean())
FAIL = {"n": {"A": int(round(len(GA) / NS)), "S3": int(round(len(GS) / NS))},
        "pre_note": "推薦當天看得到的約 60 項訊號，輸家與贏家幾乎分不開 (多數 AUC 0.44~0.56)；模型已經用掉大部分資訊",
        "A_market": {"lose_mvol": round(float(loseA["m_vola20"].median()), 2), "hit_mvol": round(float(winA["m_vola20"].median()), 2),
                     "lose_mb60": round(float(loseA["m_bias60"].median()), 1), "hit_mb60": round(float(winA["m_bias60"].median()), 1), "yrs": f"{sum(yrsA)}/{len(yrsA)}",
                     "note": "描述：A 級失敗多在大盤溫和下跌、波動低時；命中多在急跌恐慌後。走動式濾網年數不足，未採用"},
        "S_ps": {"lose": round(float(loseS["ps"].median()), 3), "hit": round(float(winS["ps"].median()), 3), "yrs": f"{sum(yrsS)}/{len(yrsS)}", "wf_dhit": round(float(np.mean(dh)), 3),
                 "note": "同為前 3 名，飆股分數較高的失敗較少；濾掉最低兩成只多 1~2pt"},
        "early": {"A": early(GA, "hit", "fin21"), "S3": early(GS, "surge", "fin20")}}

# ---------- 推薦當天 A 級分數校準 (z = logit(p) − logit(thA)；PAV 單調) ----------
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
ZA = pd.concat([R[(R["radar"] == "T") & R["tier"].isin(["A", "A+"]) & R["hit"].notna()].assign(sid=sid) for sid, _, _, R in sets], ignore_index=True)
ZA["z"] = lg(ZA["p"].to_numpy(float)) - lg(ZA["thA"].to_numpy(float))


def pav(y, w):
    y, w = list(map(float, y)), list(map(float, w)); blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]; nw = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / nw, nw, a[2] + b[2]]; del blocks[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return out


def zcal(D, nb=5):
    edges = list(np.quantile(D["z"], np.linspace(0, 1, nb + 1)[1:-1]))
    b = np.searchsorted(edges, D["z"].to_numpy(), side="right")
    hit = [float(D["hit"].to_numpy()[b == j].mean()) for j in range(nb)]; n = [int((b == j).sum()) for j in range(nb)]
    return edges, pav(hit, n), n, hit


zc_edges, zc_hit, zc_n, zc_raw = zcal(ZA)
wfz = []
for y in (2023, 2024, 2025, 2026):
    tr, te = ZA[ZA["year"] < y], ZA[(ZA["year"] == y) & (ZA["sid"] == "None-cont")]
    if len(te) < 10 or len(tr) < 100:
        continue
    e, h, _, _ = zcal(tr); pr = np.array(h)[np.searchsorted(e, te["z"].to_numpy(), side="right")]
    base = tr["hit"].mean()
    wfz.append({"year": y, "cal": round(float(((pr - te["hit"]) ** 2).mean()), 4), "const": round(float(((base - te["hit"]) ** 2).mean()), 4), "n": int(len(te))})
DAY0 = {"edges": [round(float(e), 3) for e in zc_edges], "hit": [round(v, 3) for v in zc_hit], "raw": [round(v, 3) for v in zc_raw], "n": [int(round(v / NS)) for v in zc_n],
        "base": round(float(ZA["hit"].mean()), 3), "wf": wfz,
        "note": "z = logit(p) − logit(th_A)；A/A+ 推薦依分數分 5 組的歷史命中 (2020~ 樣本外、10 組推薦平均、單調化)。走動式 Brier 對常數只小幅改善，當參考"}

TIERS = {k: {"n": full[k]["n"], "hit": full[k]["hit"], "fin": full[k]["avg"], "win21": full[k]["win"]} for k in ("A+", "A-", "B+", "Bo") if k in full}
res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "sets": [s[0] for s in sets], "full": full, "tiers": TIERS, "bt": bt_block(full), "y2026": bt_block(y26), "same": bt_block(same), "pool": pool,
       "path": PATH, "wf_path": {"A": wfA, "S3": wfS}, "fail": FAIL, "day0": DAY0, "secs": round(time.time() - t0)}
(OUT / "final_constants.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("bt", "y2026", "same", "pool", "wf_path", "fail", "day0")}, ensure_ascii=False, indent=0)[:6000])
print("done", res["secs"])
