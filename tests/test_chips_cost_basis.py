"""10-18 chips-2：cost_table「N日成本」淨賣超時其實是出貨均價 → 加 N日口徑 ('buy'/'sell') 與一律買超加權的 N日買超成本、N日出貨均價；
gov8 watchlist 帶出 basis20/60、buy_cost20/60 (與分布區塊 c20/c60 同口徑)、sell_avg20/60。舊鍵 cost20/vs20 不變。

    python tests/test_chips_cost_basis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.analysis import chips as C, gov8 as G  # noqa: E402


def _flows(vals, closes):
    return pd.DataFrame({"date": pd.bdate_range("2026-08-03", periods=len(vals)).strftime("%Y-%m-%d"), "gov8": vals, "close": closes})


def test_net_sell_marks_sell_and_gives_buy_cost():
    # 買 100@30、賣 300@32 → 淨賣；出貨均價 32、買超成本 30
    f = _flows([100, -300] * 30, [30.0, 32.0] * 30)
    r = C.cost_table(f, 31.0).iloc[0]
    assert r["20日淨買(張)"] < 0 and r["20日口徑"] == "sell"
    assert r["20日成本"] == 32.0 and r["20日出貨均價"] == 32.0       # 舊鍵沿用舊口徑
    assert r["20日買超成本"] == 30.0 and r["20日現價vs買超成本%"] == round((31 / 30 - 1) * 100, 2)


def test_net_buy_marks_buy():
    f = _flows([300, -100] * 30, [30.0, 32.0] * 30)
    r = C.cost_table(f, 31.0).iloc[0]
    assert r["20日口徑"] == "buy" and r["20日成本"] == 30.0 == r["20日買超成本"] and r["20日出貨均價"] == 32.0


def test_gov8_watchlist_passes_basis():
    f = _flows([100, -300] * 30, [30.0, 32.0] * 30)
    G.histock.government_bank_stock = lambda sid: pd.DataFrame()
    wl = G.watchlist({"00981A": {"name": "x", "price": 31.0, "flows": f, "costs": C.cost_table(f, 31.0)}})
    r = wl["00981A"]
    assert r["cost20"] == 32.0 and r["basis20"] == "sell" and r["buy_cost20"] == 30.0 and r["sell_avg20"] == 32.0, r
    assert r["basis60"] == "sell" and r["buy_vs20"] == round((31 / 30 - 1) * 100, 2)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
