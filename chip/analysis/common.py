"""評分共用工具。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Factor:
    key: str
    name: str
    score: float            # -2 .. +2
    weight: float
    value: str              # 顯示用數值
    comment: str            # 一句話解讀
    available: bool = True
    tags: list[str] = field(default_factory=list)   # 例如 '護盤', '斷頭', '軋空'

    @property
    def contribution(self) -> float:
        return self.score * self.weight if self.available else 0.0


def clip(x: float, lo: float = -2.0, hi: float = 2.0) -> float:
    if x is None or x != x:
        return 0.0
    return float(max(lo, min(hi, x)))


def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """rolling z-score (以 window 內標準差正規化，避免受市值成長影響)"""
    s = series.astype(float)
    sd = s.rolling(window, min_periods=max(10, window // 3)).std()
    mu = s.rolling(window, min_periods=max(10, window // 3)).mean()
    return (s - mu) / sd.replace(0, np.nan)


def streak(series: pd.Series) -> pd.Series:
    """連續同號天數 (正=連買, 負=連賣)"""
    out = []
    run = 0
    for v in series.fillna(0):
        if v > 0:
            run = run + 1 if run > 0 else 1
        elif v < 0:
            run = run - 1 if run < 0 else -1
        else:
            run = 0
        out.append(run)
    return pd.Series(out, index=series.index)


def composite(factors: list[Factor]) -> float:
    """加權平均 → -100..100"""
    avail = [f for f in factors if f.available]
    if not avail:
        return 0.0
    wsum = sum(abs(f.weight) for f in avail)
    return round(sum(f.contribution for f in avail) / (2 * wsum) * 100, 1)


def regime(score: float) -> str:
    if score >= 40:
        return "強勢多方"
    if score >= 15:
        return "偏多"
    if score > -15:
        return "中性"
    if score > -40:
        return "偏空"
    return "空方"


def fmt(v, digits: int = 1, unit: str = "", sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    s = f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"
    return s + unit
