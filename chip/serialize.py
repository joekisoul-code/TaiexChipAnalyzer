"""把分析結果 (dataclass / DataFrame / numpy) 轉成可 JSON 序列化的純 Python 物件。"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if dataclasses.is_dataclass(o):
        return clean(dataclasses.asdict(o))
    if isinstance(o, pd.DataFrame):
        return clean(o.replace([np.inf, -np.inf], np.nan).to_dict("records"))
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    if isinstance(o, (np.floating, float)):
        return None if (o != o or o in (np.inf, -np.inf)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o
