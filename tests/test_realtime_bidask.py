"""10-18 RT-1 / RT-3 / RT-4：即時盤勢後端修正。

RT-1：MIS a/f = 五檔賣價/賣量、b/g = 五檔買價/買量 → bid_size 取 g、ask_size 取 f；bid_ask_ratio = 委買/委賣，
      盤勢分「委買賣比」只有委買 > 委賣才加分，ask_heavy 只有委賣遠大於委買才觸發。
RT-3：夜盤 futures_quotes(night=True) 不輸出 basis/basis_pct (現貨已收，不是期現價差)，改 vs_spot_close/vs_spot_close_pct；日盤照舊。
RT-4：session_phase 週一 00:00~05:00 為 closed (週日晚上沒有夜盤)。

    python tests/test_realtime_bidask.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip import config, realtime as RT  # noqa: E402

TSMC = {"c": "2330", "n": "台積電", "z": "2420", "y": "2440", "a": "2425_2430_2435_2440_2445_", "b": "2420_2415_2410_2405_2400_",
        "f": "698_394_491_604_525_", "g": "508_776_958_498_1996_", "v": "33257", "%": "14:30:00"}


def _cached(rows):
    return lambda key, ttl, load: rows


def test_large_caps_bid_is_g_ask_is_f():
    RT.cached = _cached([TSMC])
    r = RT.large_caps().iloc[0]
    assert r["bid_size"] == 4736 and r["ask_size"] == 2712, r.to_dict()


def test_ratio_score_and_alert_direction():
    RT.cached = _cached([TSMC])
    b = RT.breadth_of(RT.large_caps())
    assert abs(b["bid_ask_ratio"] - 4736 / 2712) < 1e-9 and b["bid_ask_ratio"] > 1
    part = next(p for p in RT.intraday_score({"breadth": b})["parts"] if p["name"] == "委買賣比")
    assert part["score"] > 0                          # 委買多 → 加分
    rule = dict(RT.ALERT_RULES)["ask_heavy"]
    assert rule({"breadth": b}, None) is None
    # 委賣遠大於委買 (f 大、g 小) → 比 < 0.6 → 觸發，分項為負
    heavy = dict(TSMC, f="2000_2000_2000_2000_2000_", g="100_100_100_100_100_")
    RT.cached = _cached([heavy])
    b2 = RT.breadth_of(RT.large_caps())
    assert b2["bid_ask_ratio"] == 500 / 10000
    assert rule({"breadth": b2}, None)
    part2 = next(p for p in RT.intraday_score({"breadth": b2})["parts"] if p["name"] == "委買賣比")
    assert part2["score"] < 0


def _tx_rows():
    return [{"SymbolID": "TXF-S", "CLastPrice": "47111.26", "CRefPrice": "47111.26", "DispCName": "現貨"},
            {"SymbolID": "TXFK6-F", "DispCName": "臺指期116", "CLastPrice": "46140", "CRefPrice": "47284", "CTotalVolume": "1",
             "OpenInterest": "1", "CTime": "045957", "CDate": "20261016", "SettlementPrice": "", "CBidPrice1": "", "CAskPrice1": "", "CBidSize1": "", "CAskSize1": ""}]


def test_night_quote_has_no_basis():
    RT.cached = _cached(_tx_rows())
    q = RT.futures_quotes(night=True)
    assert "basis" not in q and "basis_pct" not in q, q
    assert abs(q["vs_spot_close"] - (46140 - 47111.26)) < 1e-6 and q["vs_spot_close_pct"] < 0
    d = RT.futures_quotes(night=False)
    assert abs(d["basis"] - (46140 - 47111.26)) < 1e-6 and "vs_spot_close" not in d


def test_session_phase_monday_early_closed():
    T = lambda *a: dt.datetime(*a, tzinfo=config.TZ)
    assert RT.session_phase(T(2026, 10, 19, 2, 0)) == "closed"      # 週一凌晨：週日沒有夜盤
    assert RT.session_phase(T(2026, 10, 20, 2, 0)) == "night"       # 週二凌晨
    assert RT.session_phase(T(2026, 10, 17, 2, 0)) == "night"       # 週六凌晨 (週五夜盤)
    assert RT.session_phase(T(2026, 10, 17, 6, 0)) == "closed"
    assert RT.session_phase(T(2026, 10, 19, 8, 40)) == "pre"
    assert RT.session_phase(T(2026, 10, 19, 15, 10)) == "night"


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
