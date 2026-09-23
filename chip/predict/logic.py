"""漲跌邏輯 (2026-09-23)：從歷史資料學出「可讀的決策規則」(淺層決策樹)，逐年走動式驗證，並回看每條規則的歷史實例。

方法：
- 特徵只用可解釋的 ~35 個 (夜盤、亞股同日、美股前晚、VIX、外資/投信/期貨/選擇權籌碼、乖離/動能/波動/量能、結算週期…)。
- 每個視野 (1/3/5 日) 走動式：以「之前所有年份」擬合 depth-3 決策樹 (自製 entropy 樹，葉節點 ≥120 筆，缺值補訓練中位數)，預測當年；只有葉節點上漲率偏離基準 ≥ MARGIN 才算「有方向」。
  → oos: 有方向日的命中率 / 覆蓋 / 逐年最低 (方法整體的誠實成績)。
- 最終規則：用全部歷史擬合的樹，每個葉節點展開成「若 A 且 B 且 C → 偏多/偏空 (n、上漲率、逐年一致性、最近實例)」。
  這些規則的上漲率是樣本內；請看 oos 與逐年一致性 (yr_cons) 判斷可信度。
- 兩個變體：base (不含夜盤，收盤後即可用) / night (含前晚夜盤台指期，夜盤收後才用)。研究顯示夜盤一旦可用就主宰整棵樹 (重要度 0.8~0.96)，
  所以 base 變體才是「籌碼/技術面本身的漲跌邏輯」。
- build(): 今日資料走到哪個葉節點 → 今日適用規則 + 該規則歷史最近 8 次的實際結果 (查看歷史資料)。
- train() 另外輸出走動式 OOS 訊號序列 (data/models/logic_oos.csv)，供 verdict.train 把「漲跌邏輯」當成一票做共識驗證。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
HORIZONS = (1, 3, 5)
MARGIN = 0.07          # 葉節點上漲率須偏離基準 ≥7pt 才算有方向
MIN_LEAF = 120
DEPTH = 3
FIRST_TEST_YEAR = 2013
FEATS = ["night_chg_pct", "hsi_r0", "kospi_r0", "nikkei_r0", "g_sox_r1", "g_sp500_r1", "g_vix_level", "g_vix_r1", "g_usdtwd_r5",
         "foreign_z5", "foreign_z20", "trust_z5", "fut_foreign_chg1_z", "fut_foreign_chg5_z", "txo_f_cp_diff_z", "mtx_retail_inv", "pcr_oi", "tx_basis_pct",
         "bias20", "bias5", "ret1", "ret5", "ret20", "gap_open", "range_pct", "vola20", "vol_ratio", "lo20_dist", "hi20_dist", "ma20_slope", "streak",
         "margin_chg5_pct", "smart2", "composite_smooth", "days_to_settle", "dow", "amount_5d_ratio"]
OPS = {"<=": "≤", ">": ">"}


def _names() -> dict:
    from . import short_term as ST
    n = dict(ST.NAMES)
    n.update({"g_vix_level": "VIX 水準", "g_vix_r1": "VIX 前晚%", "g_usdtwd_r5": "美元/台幣 5 日%", "trust_z5": "投信 5 日 z", "fut_foreign_chg1_z": "外資期貨 1 日增減 z", "fut_foreign_chg5_z": "外資期貨 5 日增減 z",
              "txo_f_cp_diff_z": "外資選擇權 call−put z", "pcr_oi": "選擇權 P/C 未平倉比", "tx_basis_pct": "期現價差%", "bias20": "月線乖離%", "bias5": "5 日線乖離%", "ret1": "今日漲跌%", "ret5": "5 日漲跌%", "ret20": "20 日漲跌%",
              "vola20": "20 日波動率", "vol_ratio": "量能/20 日均", "lo20_dist": "距 20 日低點%", "hi20_dist": "距 20 日高點%", "ma20_slope": "月線斜率", "streak": "連漲(跌)天數", "margin_chg5_pct": "融資 5 日增減%",
              "composite_smooth": "籌碼綜合分", "days_to_settle": "距結算日", "dow": "星期 (0=一)", "amount_5d_ratio": "5 日量能/20 日均", "foreign_z5": "外資 5 日 z", "foreign_z20": "外資 20 日 z", "mtx_retail_inv": "小台散戶反向", "smart2": "聰明錢 v2",
              "night_chg_pct": "前晚夜盤台指期%", "hsi_r0": "恆生今日%", "kospi_r0": "KOSPI 今日%", "nikkei_r0": "日經今日%", "g_sox_r1": "費半前晚%", "g_sp500_r1": "S&P 前晚%", "gap_open": "今日開盤跳空%", "range_pct": "今日振幅%"})
    return n


class _Tree:
    """自製淺層決策樹 (entropy)，不依賴 sklearn (本機 sklearn 的 DLL 被應用程式控制原則封鎖)。缺值以訓練集中位數補。
    介面仿 sklearn：fit / predict_proba / apply / feature_importances_ / tree_ (children_left/right, feature, threshold)。"""

    def __init__(self, max_depth=DEPTH, min_leaf=MIN_LEAF, n_thr=24):
        self.max_depth, self.min_leaf, self.n_thr = max_depth, min_leaf, n_thr

    @staticmethod
    def _ent(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, int)
        self.med_ = np.nanmedian(X, axis=0); self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
        X = np.where(np.isnan(X), self.med_, X)
        self.nodes_ = []   # dict(feature, thr, left, right, n, p)
        self.imp_ = np.zeros(X.shape[1])

        def grow(idx, depth):
            n = len(idx); p = y[idx].mean()
            node = {"feature": -1, "thr": 0.0, "left": -1, "right": -1, "n": n, "p": float(p)}
            self.nodes_.append(node); me = len(self.nodes_) - 1
            if depth >= self.max_depth or n < 2 * self.min_leaf or p in (0.0, 1.0):
                return me
            best = (0.0, None, None)
            H = self._ent(p)
            for f in range(X.shape[1]):
                col = X[idx, f]
                qs = np.unique(np.quantile(col, np.linspace(0.05, 0.95, self.n_thr)))
                for thr in qs:
                    L = col <= thr; nl = L.sum(); nr = n - nl
                    if nl < self.min_leaf or nr < self.min_leaf:
                        continue
                    gain = H - (nl * self._ent(y[idx][L].mean()) + nr * self._ent(y[idx][~L].mean())) / n
                    if gain > best[0]:
                        best = (gain, f, float(thr))
            if best[1] is None:
                return me
            gain, f, thr = best
            self.imp_[f] += gain * n
            node["feature"], node["thr"] = f, thr
            L = X[idx, f] <= thr
            node["left"] = grow(idx[L], depth + 1); node["right"] = grow(idx[~L], depth + 1)
            return me

        grow(np.arange(len(y)), 0)
        self.feature_importances_ = self.imp_ / self.imp_.sum() if self.imp_.sum() > 0 else self.imp_

        class _T:  # 仿 sklearn tree_ 屬性
            pass
        t = _T(); nd = self.nodes_
        t.children_left = np.array([x["left"] for x in nd]); t.children_right = np.array([x["right"] for x in nd])
        t.feature = np.array([x["feature"] for x in nd]); t.threshold = np.array([x["thr"] for x in nd])
        self.tree_ = t
        return self

    def apply(self, X):
        X = np.where(np.isnan(np.asarray(X, float)), self.med_, np.asarray(X, float))
        out = np.zeros(len(X), int)
        for i, row in enumerate(X):
            k = 0
            while self.nodes_[k]["left"] != -1:
                nd = self.nodes_[k]; k = nd["left"] if row[nd["feature"]] <= nd["thr"] else nd["right"]
            out[i] = k
        return out

    def predict_proba(self, X):
        p = np.array([self.nodes_[k]["p"] for k in self.apply(X)])
        return np.c_[1 - p, p]


def _tree():
    return _Tree()


def _frame(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from . import crossmkt as XM
    d = XM.add_features(matrix).copy() if "smart2" not in matrix else matrix.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    if "dow" not in d:
        d["dow"] = d["date"].dt.dayofweek
    feats = [f for f in FEATS if f in d.columns]
    return d, feats


def _leaf_rules(tree, feats: list[str], X: pd.DataFrame, y: pd.Series, fwd: pd.Series, years: pd.Series, dates: pd.Series, names: dict, base: float) -> list[dict]:
    t = tree.tree_
    leaf_of = tree.apply(X.values)
    rules = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            m = leaf_of == node
            n = int(m.sum())
            if n == 0:
                return
            up = float(y[m].mean())
            side = 1 if up >= base + MARGIN else -1 if up <= base - MARGIN else 0
            yr = {}
            for yv, g in y[m].groupby(years[m]):
                if len(g) >= 5:
                    yr[int(yv)] = round(float(g.mean()), 3)
            cons = (np.mean([(v >= base) if side > 0 else (v <= base) for v in yr.values()]) if yr and side else None)
            ex = dates[m].tail(3).dt.strftime("%Y-%m-%d").tolist()
            rules.append({"id": int(node), "conds": conds, "text": " 且 ".join(f"{names.get(c['f'], c['f'])} {OPS[c['op']]} {c['thr']:g}" for c in conds),
                          "n": n, "up_rate": round(up, 3), "mean_fwd": round(float(fwd[m].mean()), 3), "dir": "偏多" if side > 0 else "偏空" if side < 0 else "中性",
                          "years": len(yr), "yr_cons": round(float(cons), 2) if cons is not None else None, "yr_min": min(yr.values()) if yr else None, "yr_max": max(yr.values()) if yr else None, "examples": ex})
            return
        f = feats[t.feature[node]]; thr = float(t.threshold[node])
        walk(t.children_left[node], conds + [{"f": f, "op": "<=", "thr": round(thr, 2)}])
        walk(t.children_right[node], conds + [{"f": f, "op": ">", "thr": round(thr, 2)}])

    walk(0, [])
    return rules


def train(matrix: pd.DataFrame, write: bool = True, verbose: bool = True) -> dict:
    d, feats = _frame(matrix)
    names = _names()
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "margin": MARGIN, "depth": DEPTH, "min_leaf": MIN_LEAF, "features": feats, "h": {}}
    oos_rows = []
    for variant, h in [(v, h) for v in ("base", "night") for h in HORIZONS]:
        tgt = f"fwd{h}"
        vf = [f for f in feats if f != "night_chg_pct"] if variant == "base" else feats
        dd = d.dropna(subset=[tgt] + (["night_chg_pct"] if variant == "night" else [])).reset_index(drop=True)
        X, y, yrs = dd[vf], (dd[tgt] > 0).astype(int), dd["year"]
        preds = pd.Series(np.nan, index=dd.index)
        for yv in sorted(yrs.unique()):
            if yv < FIRST_TEST_YEAR:
                continue
            tr = yrs < yv
            if tr.sum() < 500:
                continue
            m = _tree().fit(X[tr].values, y[tr])
            preds[yrs == yv] = m.predict_proba(X[yrs == yv].values)[:, 1]
        base_prev = y.expanding().mean().shift(1)
        sig = pd.Series(0, index=dd.index)
        sig[(preds >= base_prev + MARGIN)] = 1; sig[(preds <= base_prev - MARGIN)] = -1
        ev = dd[preds.notna()]
        s_ev, y_ev = sig[ev.index], y[ev.index]
        call = ev[s_ev != 0]
        hit = ((np.sign(call[tgt]) == s_ev[call.index])).astype(float)
        yr = [g.mean() for _, g in hit.groupby(yrs[call.index]) if len(g) >= 5]
        oos = {"n_days": int(len(ev)), "n_calls": int(len(call)), "cov": round(len(call) / len(ev), 3) if len(ev) else None, "hit": round(float(hit.mean()), 3) if len(call) else None,
               "base": round(float((ev[tgt] > 0).mean()), 3), "yr_min": round(float(min(yr)), 3) if yr else None, "yr_med": round(float(np.median(yr)), 3) if yr else None, "years": len(yr),
               "up_hit": round(float(hit[s_ev[call.index] > 0].mean()), 3) if (s_ev[call.index] > 0).any() else None, "dn_hit": round(float(hit[s_ev[call.index] < 0].mean()), 3) if (s_ev[call.index] < 0).any() else None}
        for i in call.index:
            oos_rows.append({"date": dd.loc[i, "date"].strftime("%Y-%m-%d"), "h": h, "variant": variant, "sig": int(s_ev[i]), "p": round(float(preds[i]), 3)})
        final = _tree().fit(X.values, y)
        base = float(y.mean())
        rules = _leaf_rules(final, vf, X, y, dd[tgt], yrs, dd["date"], names, base)
        imp = sorted(((vf[i], round(float(v), 3)) for i, v in enumerate(final.feature_importances_) if v > 0), key=lambda x: -x[1])
        out["h"].setdefault(str(h), {})[variant] = {"base": round(base, 3), "oos": oos, "rules": rules, "med": {f: round(float(m_), 4) for f, m_ in zip(vf, final.med_)}, "importance": [{"f": f, "name": names.get(f, f), "w": w} for f, w in imp]}
        if verbose:
            print(f"  logic h{h} {variant:<5}: OOS 有方向日 {oos['n_calls']}/{oos['n_days']} (覆蓋 {oos['cov']}) 命中 {oos['hit']} (基準 {oos['base']}，逐年最低 {oos['yr_min']}，多 {oos['up_hit']} / 空 {oos['dn_hit']})；規則 {len(rules)} 條，主要特徵 " + "、".join(f"{names.get(f, f)} {w}" for f, w in imp[:4]))
    if write:
        M.save_json("logic", out)
        pd.DataFrame(oos_rows).to_csv(config.DATA_DIR / "models" / "logic_oos.csv", index=False)
    return out


def oos_signal(h: int = 1, variant: str = "base") -> pd.Series | None:
    """走動式 OOS 訊號 (date → ±1)，給 verdict.train 用。"""
    p = config.DATA_DIR / "models" / "logic_oos.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[(df["h"] == h) & (df["variant"] == variant)]
    return pd.Series(df["sig"].values, index=pd.to_datetime(df["date"]))


def build(matrix: pd.DataFrame, night_final: bool = False) -> dict | None:
    """今日資料走到哪條規則 + 該規則最近 8 次歷史實例。matrix = short_term.build_matrix(scored, night) (最後一列 = 今日)。"""
    st = M.load_json("logic")
    if not st:
        return None
    d, feats = _frame(matrix)
    names = _names()
    today = d.iloc[-1]
    nv = today.get("night_chg_pct")
    variant = "night" if (nv is not None and nv == nv and night_final) else "base"
    out = {"date": today["date"].strftime("%Y-%m-%d"), "variant": variant, "variant_note": "夜盤已收、含夜盤變體" if variant == "night" else "夜盤未定、不含夜盤變體 (夜盤收後改用含夜盤規則)", "h": {}, "trained_at": st.get("trained_at")}
    for h in HORIZONS:
        H = ((st.get("h") or {}).get(str(h)) or {}).get(variant)
        if not H:
            continue
        med = H.get("med") or {}
        def val(f):
            v = today.get(f)
            return med.get(f) if (v is None or v != v) else float(v)
        match = None
        for r in H["rules"]:
            ok = True
            for c in r["conds"]:
                v = val(c["f"])
                if v is None:
                    ok = False; break
                ok = (v <= c["thr"]) if c["op"] == "<=" else (v > c["thr"])
                if not ok:
                    break
            if ok:
                match = r; break
        if not match:
            out["h"][str(h)] = {"rule": None, "note": "今日有特徵缺值，無法對應規則"}
            continue
        # 歷史實例：符合同一組條件的最近 8 天與其實際 fwd
        m = pd.Series(True, index=d.index)
        for c in match["conds"]:
            m &= (d[c["f"]] <= c["thr"]) if c["op"] == "<=" else (d[c["f"]] > c["thr"])
        hist = d[m & d[f"fwd{h}"].notna()].tail(8)
        ex = [{"date": r_["date"].strftime("%Y-%m-%d"), "fwd": round(float(r_[f"fwd{h}"]), 2)} for _, r_ in hist.iterrows()][::-1]
        vals = {c["f"]: {"name": names.get(c["f"], c["f"]), "value": round(float(val(c["f"])), 2), "op": OPS[c["op"]], "thr": c["thr"], "filled": bool(today.get(c["f"]) != today.get(c["f"]))} for c in match["conds"]}
        out["h"][str(h)] = {"rule": match, "today_values": vals, "recent": ex, "recent_up": round(float(np.mean([e["fwd"] > 0 for e in ex])), 2) if ex else None, "oos": H["oos"], "base": H["base"]}
    return out
