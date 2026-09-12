"""國際市場資料源檢查：python tests/probe_global.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from chip.sources import global_markets as g  # noqa: E402

for k, (sym, name, _) in g.SYMBOLS.items():
    t = time.time()
    try:
        df = g.history(sym)
        print(f"{k:<8} {sym:<10} {name:<14} {len(df):>5} rows {df['date'].min()} ~ {df['date'].max()}  last={df['close'].iloc[-1]:.2f}  {time.time() - t:.1f}s")
    except Exception as e:  # noqa: BLE001
        print(k, sym, "ERR", e)
