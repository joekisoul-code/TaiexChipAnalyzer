import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
from chip.analysis import global_study  # noqa: E402

r = global_study.run()
print("rows", r["report"]["rows"], r["report"]["start"], "~", r["report"]["end"])
print("\n=== 同日效應 (前晚/前日報酬 vs 台股當日) rank corr ===")
print(r["same_day"].to_string(index=False))
print("\n=== 預測力 (rank-IC vs 未來報酬) ===")
print(r["predictive"].to_string(index=False))
print("\n=== 事件研究 ===")
print(r["events"].to_string(index=False))
print("\n=== 滾動相關 (60 日) ===")
print(r["rolling"].to_string(index=False))
