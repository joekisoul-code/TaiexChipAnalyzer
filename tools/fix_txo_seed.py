"""一次性：把 data/models/txo_iv_seed.json 近期列換成研究 t3 s00 的休市修正值 (features_fixed_recent.parquet)。
原因：共用的 txo_features_daily 在資料截止日前後，未到期序列用了不扣休市日的日曆 → n 高估、IV 低估 (09-17~09-23 iv5 低 2~4 點)。
用法：python tools/fix_txo_seed.py <features_fixed_recent.parquet>
"""
import json
import sys
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parents[1] / "data" / "models" / "txo_iv_seed.json"
fx = pd.read_parquet(sys.argv[1])
fx.index = pd.to_datetime(fx.index).strftime("%Y-%m-%d")
seed = json.loads(P.read_text(encoding="utf-8"))
n = 0
for r in seed["rows"]:
    if r["date"] in fx.index:
        for c in ("iv5", "iv21", "rr25", "ts_5_21", "F_near", "pwall_dist", "cwall_dist", "mp_dist"):
            if c in fx.columns and pd.notna(fx.at[r["date"], c]):
                v = round(float(fx.at[r["date"], c]), 5)
                if r.get(c) != v:
                    r[c] = v
                    n += 1
if "休市修正" not in seed["source"]:
    seed["source"] += "；近期列已套研究 s00 休市修正 (features_fixed_recent)"
P.write_text(json.dumps(seed, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("updated cells:", n)
