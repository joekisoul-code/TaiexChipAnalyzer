"""10-18 stocks-3：learn.summarize 的 stocks[sid] 對帳統計排除 ETF (00 開頭) 在 ETF 閘門 (2026-09-24) 之前的個股叫牌；
一般個股與閘門之後的 ETF 列不受影響。

    python tests/test_learn_etf_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import learn as L  # noqa: E402


def _row(sid, as_of, call, hit, h=5):
    return {"kind": "stk", "sid": sid, "as_of": as_of, "h": h, "call": call, "p_up": 0.4, "base_hit": 0.5, "call_hit": 0.6,
            "realized": {"hit": hit, "up": bool(hit) if hit is not None else True, "brier": 0.2, "rel": -1.0}}


def test_etf_pre_gate_excluded():
    rows = [_row("00988A", "2026-09-21", "偏空", False), _row("00988A", "2026-09-22", "偏空", False),
            _row("00988A", "2026-10-01", "中性", None),
            _row("2330", "2026-09-21", "偏多", True), _row("2330", "2026-09-22", "偏多", False)]
    st = L.summarize(rows)["stocks"]
    e = st["00988A"]["by_h"]["5"]
    assert e["n_calls"] == 0 and e["n_eval"] == 1, e          # 只剩閘門後的中性列
    assert st["00988A"]["recent"] == []
    assert st["2330"]["by_h"]["5"]["n_calls"] == 2             # 一般個股不受影響


def test_etf_only_pre_gate_drops_entry():
    st = L.summarize([_row("00685L", "2026-09-23", "偏多", True)])["stocks"]
    assert "00685L" not in st
    assert L._etf_pre_gate({"sid": "00981A", "as_of": "2026-09-23"}) and not L._etf_pre_gate({"sid": "00981A", "as_of": "2026-09-24"})
    assert not L._etf_pre_gate({"sid": "2330", "as_of": "2026-09-01"})


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
