import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
from chip.analysis import cross_market  # noqa: E402

r = cross_market.run()
for line in r["summary"]:
    print(line)
for target, res in r["results"].items():
    print(f"\n\n########## {target}  {res['start']} ~ {res['end']}  n={res['n']}")
    for g, tbl in res["groups"].items():
        print(f"\n--- {g} ---")
        print(tbl.head(12).to_string(index=False))
    print("\n--- 事件研究 ---")
    print(res["events"].to_string(index=False))
