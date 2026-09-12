"""快速煙霧測試：每個資料源各抓一次並印出摘要。  python tests/smoke_sources.py"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

from chip.sources import finmind, histock, taifex, twse, wantgoo  # noqa: E402


def run(name, fn):
    try:
        r = fn()
        if isinstance(r, pd.DataFrame):
            print(f"[OK] {name}: {len(r)} rows, cols={list(r.columns)[:12]}")
            print(r.tail(3).to_string())
        else:
            s = str(r)
            print(f"[OK] {name}: {s[:400]}")
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
    print()


run("twse.institutional_daily", twse.institutional_daily)
run("twse.margin_daily", twse.margin_daily)
run("twse.market_daily", twse.market_daily)
run("twse.index_realtime", twse.index_realtime)
run("twse.stock_realtime 2330", lambda: twse.stock_realtime("2330"))
run("twse.t86", lambda: twse.t86().head(3))
run("twse.sbl_balance", lambda: {k: v for k, v in twse.sbl_balance().items() if k != "stocks"})
run("twse.margin_stocks", lambda: twse.margin_stocks().head(3))
run("taifex.futures_institutional_latest", taifex.futures_institutional_latest)
run("taifex.put_call_ratio", taifex.put_call_ratio)
run("taifex.large_traders_tx", taifex.large_traders_tx)
run("finmind.taiex_price", finmind.taiex_price)
run("finmind.total_institutional", finmind.total_institutional)
run("finmind.total_margin", finmind.total_margin)
run("finmind.tx_futures_institutional", finmind.tx_futures_institutional)
run("finmind.stock_price 2330", lambda: finmind.stock_price("2330"))
run("finmind.stock_institutional 2330", lambda: finmind.stock_institutional("2330"))
run("finmind.stock_margin 2330", lambda: finmind.stock_margin("2330"))
run("finmind.stock_shareholding 2330", lambda: finmind.stock_shareholding("2330"))
run("histock.government_banks_history", histock.government_banks_history)
run("histock.government_banks_ranking", lambda: histock.government_banks_ranking()["buy"].head(5))
run("histock.government_bank_stock 2330", lambda: histock.government_bank_stock("2330"))
if "--wantgoo" in sys.argv:
    def wg():
        d = wantgoo.fetch_market()
        if d is None:
            return "unavailable"
        print("headline", d["headline"])
        print(d["margin_table"].tail(3).to_string())
        print(d["banks"].tail(3).to_string())
        print(d["sbl"].tail(3).to_string())
        return "done"
    run("wantgoo.fetch_market", wg)
