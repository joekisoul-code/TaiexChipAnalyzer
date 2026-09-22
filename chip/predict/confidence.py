"""叫牌信心分層 (2026-09-22)：模型分位 × 夜盤方向 × 歷史規律 × 季線狀態 的「共識」→ 高/中/低 三檔，各檔附走動式 OOS 命中率與覆蓋率。

研究 (scratch conf_study.py，門檻用滾動前幾年分位，無洩漏)：
- 含夜盤：模型前後 15% + 夜盤同向 >0.5% + 規律不反向 → 1 日 89.7% (5 年最低 89%，覆蓋 30%)、2 日 80.7%、3 日 75.2%；一般叫牌 84.9 / 74.6 / 67.1。
- 不含夜盤：模型 30% + 規律庫同向 → 1 日 60.8% (11 年最低 50%，覆蓋 23%)、2 日 59.8%、3 日 59.5%；一般叫牌 56.2 / 56.9 / 55.5。
分層定義 (train() 走動式重算並存 data/models/confidence.json；forecast 時 label() 只做標籤)：
  night: 高 = 強叫牌 且 夜盤同向 |夜盤|>0.5% 且 規律不反向；中 = 叫牌 且 夜盤同向；低 = 其餘叫牌
  base : 高 = 叫牌 且 規律同向；中 = 叫牌 且 恆生同日同向 且 KOSPI 同日同向 (2026-09-22 第二輪：1 日 59.7% 覆蓋 37%，原「規律不反向+季線同向」55.2% 覆蓋 17%)；低 = 其餘叫牌
  第二輪 (conf3_study) 結論：含夜盤的高信心再加恆生同向只多 1pt 但覆蓋少四成、逐年最低反而降 → 不改；不含夜盤改用亞股同日共識當中信心。
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from . import model as M

log = logging.getLogger(__name__)
PATH = config.DATA_DIR / "models" / "confidence.json"
HORIZONS = (1, 2, 3)
NIGHT_MIN = 0.5


def rule_score_series(d: pd.DataFrame, pat: dict) -> pd.Series:
    """有效規律 (1 日) 的帶號計數：偏多 +1、偏空 -1。d 需為 patterns._prep 後的 frame。"""
    from . import patterns as P
    R = P.rule_library(d)
    s = pd.Series(0.0, index=d.index)
    for r in pat.get("rules", []):
        v = r.get("h", {}).get("1")
        if r.get("valid_any") and v and v.get("valid") and r["name"] in R:
            s = s + R[r["name"]][0].fillna(False).astype(float) * (1 if v["direction"] == "偏多" else -1)
    return s


def _tier_masks(o: pd.DataFrame, variant: str) -> dict[str, tuple[pd.Series, pd.Series]]:
    c, rs, bull, nt = o["call"], o["rule_score"], o["bull"], o["night"]
    hs, ko = o["hsi_r0"], o["kospi_r0"]
    if variant == "night":
        return {"高": ((c >= 2) & (nt > NIGHT_MIN) & (rs >= 0), (c <= -2) & (nt < -NIGHT_MIN) & (rs <= 0)),
                "中": ((c >= 1) & (nt > 0), (c <= -1) & (nt < 0)),
                "低": (c >= 1, c <= -1)}
    return {"高": ((c >= 1) & (rs > 0), (c <= -1) & (rs < 0)),
            "中": ((c >= 1) & (hs > 0) & (ko > 0), (c <= -1) & (hs < 0) & (ko < 0)),
            "低": (c >= 1, c <= -1)}


def _exclusive(masks: dict) -> dict:
    """高 > 中 > 低 互斥 (中 = 中且非高；低 = 叫牌且非高非中)。"""
    hu, hd = masks["高"]; mu, md = masks["中"]; lu, ld = masks["低"]
    return {"高": (hu, hd), "中": (mu & ~hu, md & ~hd), "低": (lu & ~hu & ~mu, ld & ~hd & ~md)}


def label(call: str, strength: str, variant: str, night: float | None, rule_score: float, bull: bool | None, hsi: float | None = None, kospi: float | None = None) -> str | None:
    """forecast 時的標籤；沒有叫牌回 None。"""
    if call not in ("偏多", "偏空"):
        return None
    up = call == "偏多"
    rs_ok_strict = (rule_score > 0) if up else (rule_score < 0)
    rs_ok = (rule_score >= 0) if up else (rule_score <= 0)
    if variant == "night":
        if night is None:
            return "低"
        same = (night > 0) if up else (night < 0)
        if strength == "強" and same and abs(night) > NIGHT_MIN and rs_ok:
            return "高"
        return "中" if same else "低"
    if rs_ok_strict:
        return "高"
    if hsi is not None and kospi is not None and ((hsi > 0 and kospi > 0) if up else (hsi < 0 and kospi < 0)):
        return "中"
    return "低"


def train(matrix: pd.DataFrame, pat: dict, write: bool = True, verbose: bool = True) -> dict:
    """走動式重算三檔的 OOS 命中率 (門檻 = 之前年份 OOS 分位)。matrix = short_term.build_matrix(scored, night)。"""
    from . import patterns as P, short_term as ST
    d = P._prep(matrix)
    d["rule_score"] = rule_score_series(d, pat)
    d["bull"] = (d["close"] >= d["ma60"]).astype(int)
    d["night"] = d[ST.NIGHT_FEATURE] if ST.NIGHT_FEATURE in d else np.nan
    chosen = M.load_json("short_term_metrics") or {}
    out = {"trained_at": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "night_min": NIGHT_MIN, "tiers": {}}
    for variant in ("base", "night"):
        for h in HORIZONS:
            key = ((chosen.get(str(h)) or {}).get(variant) or {}).get("chosen") or ("+asia|ens" if variant == "night" else "+asia+chip+px|ens")
            sname, mk = key.split("|")
            feats = list(ST.FEATURE_SETS[sname]) + ([ST.NIGHT_FEATURE] if variant == "night" else [])
            dd = d.dropna(subset=[ST.NIGHT_FEATURE]) if variant == "night" else d
            fy = ST.FIRST_TEST_YEAR_NIGHT if variant == "night" else ST.FIRST_TEST_YEAR
            outs = []
            if mk in ("lgb", "ens"):
                outs.append(ST._wf(dd, feats, f"fwd{h}", h, fy, lambda: ST.LgbModel()))
            if mk in ("ridge", "ens"):
                outs.append(ST._wf(dd, feats, f"fwd{h}", h, fy, lambda: ST.RidgeModel()))
            if not outs or any(x.empty for x in outs):
                continue
            o = outs[0] if len(outs) == 1 else ST._combine(outs[0], outs[1])
            o = o.merge(d[["date", "rule_score", "bull", "night", "hsi_r0", "kospi_r0"]], on="date", how="left").sort_values("date").reset_index(drop=True)
            o["call"] = 0
            yrs = sorted(o["year"].unique())
            for y in yrs[2:]:
                prev = o[o["year"] < y]["pred"]
                lo, hi, slo, shi = prev.quantile(ST.TIER), prev.quantile(1 - ST.TIER), prev.quantile(ST.TIER_STRONG), prev.quantile(1 - ST.TIER_STRONG)
                m = o["year"] == y
                o.loc[m & (o["pred"] >= hi), "call"] = 1; o.loc[m & (o["pred"] <= lo), "call"] = -1
                o.loc[m & (o["pred"] >= shi), "call"] = 2; o.loc[m & (o["pred"] <= slo), "call"] = -2
            o = o[o["year"] >= yrs[2]]
            masks = _exclusive(_tier_masks(o, variant))
            res = {"n_oos": int(len(o)), "base_up": round(float((o["actual"] > 0).mean()), 3), "model": key}
            for tier, (mu, md) in masks.items():
                up, dn = o[mu], o[md]
                n = len(up) + len(dn)
                hits = float((up["actual"] > 0).sum() + (dn["actual"] < 0).sum())
                yr = []
                for y, g in o.groupby("year"):
                    u, dd_ = g[mu.loc[g.index]], g[md.loc[g.index]]
                    nn = len(u) + len(dd_)
                    if nn >= 5:
                        yr.append(round(float((u["actual"] > 0).sum() + (dd_["actual"] < 0).sum()) / nn, 3))
                res[tier] = {"n": int(n), "cov": round(n / len(o), 3), "hit": round(hits / n, 3) if n else None, "yr_min": min(yr) if yr else None, "yr_med": round(float(np.median(yr)), 3) if yr else None, "years": len(yr)}
            out["tiers"][f"{h}_{variant}"] = res
            if verbose:
                print(f"  conf h{h} {variant:<5} " + "  ".join(f"{t} {res[t]['hit']} (覆蓋 {res[t]['cov']}, 年最低 {res[t]['yr_min']})" for t in ("高", "中", "低")))
    if write:
        M.save_json("confidence", out)
    return out


def stats_for(h: int, variant: str, tier: str) -> dict | None:
    c = M.load_json("confidence") or {}
    return ((c.get("tiers") or {}).get(f"{h}_{variant}") or {}).get(tier)
