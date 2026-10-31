"""2026-10-31 最新交易日開高低缺 (FinMind 晚到) 的處理：

- twse.taiex_ohlc 解析 MI_5MINS_HIST (民國日期、千分位)
- learn：高低價缺 → 觸及對帳延後 (path_pending)，不拿收盤代替；資料到了補上；舊的「路徑高低相同」與「跳空 = 全日漲跌」對帳在資料補齊後重算
- learn：跳空預判目標日開盤價缺 → 延後
- short_term：clv 在高低價缺時是 NaN (原本被當 0 = 收在最低)

    python tests/test_learn_ohlc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import learn as LN  # noqa: E402
from chip.sources import twse  # noqa: E402

D = ["2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]
FULL = pd.DataFrame({"date": D, "open": [46211.2, 46402.79, 47014.87, 46139.79], "high": [46585.45, 47060.46, 47015.45, 46139.79],
                     "low": [45676.67, 46402.79, 46055.07, 44726.26], "close": [46096.23, 47035.58, 46368.33, 45211.83]})


def partial(n_missing=1):
    f = FULL.copy()
    f.loc[f.index[-n_missing:], ["open", "high", "low"]] = np.nan
    return f


def mkt_row(as_of, h, buy, sell):
    return {"kind": "mkt", "sid": "TAIEX", "as_of": as_of, "target": None, "h": h, "p_up": 0.6, "call": "偏多", "base": float(FULL.set_index("date").close[as_of]),
            "buy_at": buy, "sell_at": sell, "stop": 44000.0, "target_px": 48000.0}


def main():
    # twse.taiex_ohlc
    orig = twse.get_json
    twse.get_json = lambda *a, **k: {"stat": "OK", "fields": ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"],
                                     "data": [["115/10/30", "46,139.79", "46,139.79", "44,726.26", "45,211.83"]]}
    orig_cached = twse.cached
    twse.cached = lambda key, ttl, fn, **kw: fn()
    try:
        o = twse.taiex_ohlc("202610")
    finally:
        twse.get_json, twse.cached = orig, orig_cached
    assert o.to_dict("records") == [{"date": "2026-10-30", "open": 46139.79, "high": 46139.79, "low": 44726.26, "close": 45211.83}], o

    # _price_map ok
    _, px = LN._price_map(partial())
    assert px["2026-10-29"]["ok"] and not px["2026-10-30"]["ok"] and px["2026-10-30"]["low"] == 45211.83

    # 觸及對帳：10-29 → h1 目標 10-30 高低缺 → 方向照對帳、觸及延後；補齊後補上
    r = mkt_row("2026-10-29", 1, 45500.0, 46600.0)
    assert LN.evaluate([r], {"TAIEX": partial()}) == 1
    rz = r["realized"]
    assert rz["path_pending"] is True and "buy_touch" not in rz and rz["hit"] is False, rz
    assert LN.evaluate([r], {"TAIEX": partial()}) == 0, "資料還沒到 → 不動"
    assert LN.evaluate([r], {"TAIEX": FULL}) == 1
    rz = r["realized"]
    assert "path_pending" not in rz and rz["path_low"] == 44726.26 and rz["path_high"] == 46139.79 and rz["buy_touch"] is True and rz["sell_touch"] is False and rz["repaired"] == "ohlc", rz
    assert LN.evaluate([r], {"TAIEX": FULL}) == 0, "修過不再動"

    # 舊對帳 (拿收盤代替：路徑高低相同) → 資料到了重算
    old = mkt_row("2026-10-27", 1, 46300.0, 47000.0)
    old["realized"] = {"date": "2026-10-28", "ret": 2.038, "hit": True, "path_low": 47035.58, "path_high": 47035.58, "buy_touch": False, "sell_touch": True, "stop_hit": False, "target_hit": False}
    assert LN.evaluate([old], {"TAIEX": FULL}) == 1
    assert old["realized"]["path_low"] == 46402.79 and old["realized"]["path_high"] == 47060.46 and old["realized"]["buy_touch"] is False and old["realized"]["sell_touch"] is True
    assert old["realized"]["ret"] == 2.038, "方向 / 報酬不動"

    # 跳空預判：目標日開盤缺 → 延後；舊的「跳空 = 全日漲跌」→ 重算
    g = {"kind": "gap", "sid": "TAIEX", "as_of": "2026-10-29", "target": "2026-10-30", "h": 1, "p_up": 0.4, "base": 46368.33, "est_gap": -0.3, "call": "中性"}
    assert LN.evaluate([g], {"TAIEX": partial()}) == 0 and "realized" not in g
    assert LN.evaluate([g], {"TAIEX": FULL}) == 1 and abs(g["realized"]["gap"] - (46139.79 / 46368.33 - 1) * 100) < 0.001
    g2 = {"kind": "gap", "sid": "TAIEX", "as_of": "2026-10-27", "target": "2026-10-28", "h": 1, "p_up": 0.7, "base": 46096.23, "est_gap": 0.01, "call": "中性",
          "realized": {"date": "2026-10-28", "gap": 2.038, "ret": 2.038, "err": 2.028, "dir_hit": None, "hold": None, "filled": False, "up": True, "hit": None}}
    assert LN.evaluate([g2], {"TAIEX": FULL}) == 1
    assert abs(g2["realized"]["gap"] - (46402.79 / 46096.23 - 1) * 100) < 0.001 and g2["realized"]["ret"] == 2.038 and g2["realized"]["repaired"] == "ohlc"

    # short_term clv：高低價缺 → NaN
    from chip.predict import short_term as ST
    sys.path.insert(0, str(ROOT / "tests"))
    from test_margin_lag1 import synth
    import warnings
    warnings.filterwarnings("ignore")
    sc = synth()
    sc.loc[sc.index[-1], ["open", "high", "low"]] = np.nan
    m = ST.build_matrix(sc, None)
    assert np.isnan(m["clv"].iloc[-1]) and np.isfinite(m["clv"].iloc[-2]), m["clv"].tail(2)
    print("test_learn_ohlc: 9 checks passed")


if __name__ == "__main__":
    main()
