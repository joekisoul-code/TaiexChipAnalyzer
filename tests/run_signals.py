import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
from chip.analysis import backtest, signals  # noqa: E402

long = backtest.load_long("2010-01-01")
r = signals.run(long)
print(r["evaluation"].to_string(index=False))
print("\ncurrent:", r["current"]["label"], r["current"]["buy_strength"], r["current"]["sell_strength"])
for s in r["current"]["buy_signals"] + r["current"]["sell_signals"]:
    print("  ", s)
print(r["points"].tail(5).to_string(index=False))
