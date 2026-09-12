import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
from chip.analysis import chips  # noqa: E402

ids = sys.argv[1:] or chips.WATCHLIST
res = chips.assess_watchlist(ids)
for sid, a in res.items():
    print(f"\n═══ {sid} {a.get('name', '')} ═══")
    if "error" in a:
        print("ERR", a["error"])
        continue
    print(f"價 {a['price']}  日期 {a['date']}  {a['label']} (分 {a['score']})")
    for n in a["notes"]:
        print("  -", n)
    print(a["costs"].to_string(index=False))
    vp = a["profile60"]
    if vp:
        print(f"  分價量 60 日：POC {vp['poc']} 價值區 {vp['va_lo']}~{vp['va_hi']} 上方 {vp['above_pct']}% 下方 {vp['below_pct']}%")
    print("  大戶:", a["holders"])
    if not a["brokers"].empty:
        print(a["brokers"].head(8).to_string(index=False))
