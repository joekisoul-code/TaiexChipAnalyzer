"""即時模組煙霧測試：python tests/smoke_realtime.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from chip import realtime  # noqa: E402
from chip.analysis import market  # noqa: E402

intra = realtime.taiex_intraday()
print("intraday:", intra["date"], len(intra["bars"]), "bars; total_amount 億 =", round(intra["total_amount"], 1), "; last bars:", intra["bars"][-2:])
print("tx day:", realtime.futures_quotes(False))
print("tx night:", realtime.futures_quotes(True))
lc = realtime.large_caps()
print("large caps:", len(lc)); print(lc.head(3).to_string())

scored, a, meta = market.run(use_wantgoo=False)
s = realtime.snapshot(scored)
print("\nphase:", s["phase"], "| taiex:", s.get("taiex"), "| vol_pace:", s.get("vol_pace"), "| ma20:", s.get("ma20"))
print("breadth:", s.get("breadth"))
print("score:", json.dumps(s["score"], ensure_ascii=False, indent=1))
print("combined:", realtime.combined_view(a["composite_smooth"], a["regime"], s["score"]["score"], s["score"]["label"], s["phase"]))
fired = set()
print("alerts:", realtime.check_alerts(s, None, fired))
realtime.persist(s)
from chip import store  # noqa: E402
print(store.load_intraday(s["ts"][:10]).tail(2).to_string())
