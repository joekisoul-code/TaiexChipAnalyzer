"""2026-11-01 盤中低點判斷 / 最低點時段改用 TWSE 09:05 口徑 (pullback._hour_low_table / _hour_of_low)：

- 開盤價 = 09:05、高低從 09:05 起 (marks 的 open / lo)；官方開盤失真 (09:00 以昨收計的第一筆) 不進表
- 最低點時段 = 第一個「累積低 = 全日低」的時間點所在小時 (10:00 → "09:00" … 13:30 → "13:00")
- 視窗 = 近 MARKS_WINDOW_DAYS 日曆天；intraday_marks 抓不到的日子 (None) 略過
- 表帶 open_at = "09:05" (App 依此切換即時口徑)
- cells_rb / marks_rb：再分現價離今日低點反彈幾 σ (r0/r1/r2)
- _hour_high_table：高點版 (鏡像；fin_q80)

    python tests/test_pullback_marks.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from chip.predict import pullback as P  # noqa: E402
from chip.sources import twse  # noqa: E402

MK = ("09:15", "09:30", "10:00", "11:00", "12:00", "13:00", "13:30")


def rec(prev, op, lows, px=None):
    """lows：各時間點累積低 (7 個)；px：各時間點價 (預設 = 累積低 + 50)。"""
    px = px or [lo + 50 for lo in lows]
    r = {"prev_close": prev, "open": op, "hi": {}, "lo": {}}
    for m, lo, p in zip(MK, lows, px):
        r[m], r["lo"][m], r["hi"][m] = p, lo, max(op, p) + 10
    return r


def main():
    n = 0
    # 1. 最低點時段：低點在 10:30 (第一個累積低 = 全日低的是 11:00 → "10:00")；開高 (09:05 價 > 昨收)
    a = rec(100.0, 101.0, [100.8, 100.6, 100.5, 99.0, 99.0, 99.0, 99.0])
    # 低點在 13:10 → "13:00"；開低
    b = rec(100.0, 99.5, [99.4, 99.4, 99.4, 99.4, 99.4, 99.4, 98.0])
    # 低點在 09:20 → "09:00"；開低
    c = rec(100.0, 99.0, [98.9, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5])
    marks = {"2026-10-27": a, "2026-10-28": b, "2026-10-29": c}
    h = P._hour_of_low(marks)
    assert h["n"] == 3 and h["open_at"] == "09:05", h
    assert h["all"] == {"09:00": 0.333, "10:00": 0.333, "13:00": 0.333}, h["all"]
    assert h["gap_up"] == {"10:00": 1.0} and h["gap_down"] == {"09:00": 0.5, "13:00": 0.5}, h
    n += 3
    # 不完整的日子 (少了 13:30 累積低) 不算
    bad = rec(100.0, 99.0, [98.9] * 7); del bad["lo"]["13:30"]
    assert P._hour_of_low({**marks, "2026-10-30": bad})["n"] == 3
    assert P._hour_of_low({}) == {}
    n += 2

    # 2. 查表：跳空與 deep 用 09:05 開盤；低點已出現 = 累積低 ≤ 全日低 (容差 0.03%)
    days = pd.bdate_range("2025-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    mk2 = {}
    for i, d in enumerate(days):
        if i % 2:   # 開高 1%，10:00 前已出現全日低
            mk2[d] = rec(100.0, 101.0, [100.9, 100.7, 100.6, 100.6, 100.6, 100.6, 100.6])
        else:       # 開低 1%、開盤後已回檔 (deep)，13:00 後才出現全日低
            mk2[d] = rec(100.0, 99.0, [98.5, 98.4, 98.3, 98.3, 98.3, 98.3, 97.0])
    m = pd.DataFrame({"date": days, "sigma_range": 1.0})
    t = P._hour_low_table(m, mk2)
    assert t["open_at"] == "09:05" and t["n_days"] == 60 and t["window_days"] == P.MARKS_WINDOW_DAYS, t
    assert t["cells"]["10:00|up|shallow"]["p_done"] == 1.0 and t["cells"]["10:00|dn|deep"]["p_done"] == 0.0, t["cells"]
    assert t["marks"]["13:00"]["p_done"] == 0.5 and t["marks"]["13:00"]["n"] == 60
    n += 3
    # 反彈維度：現價 = 累積低 + 50 (≈ 0.5σ) → r2 / r1 邊界；r0 = 還在低點附近
    assert P._rb_bin(100.1, 100.0, 1.0) == "r0" and P._rb_bin(100.3, 100.0, 1.0) == "r1" and P._rb_bin(100.6, 100.0, 1.0) == "r2"
    up_rb = P._rb_bin(100.6 + 50, 100.6, 1.0)
    assert t["cells_rb"][f"10:00|up|shallow|{up_rb}"]["p_done"] == 1.0 and t["rb_cuts"] == list(P.RB_CUTS), t["cells_rb"]
    assert sum(c["n"] for k, c in t["marks_rb"].items() if k.startswith("10:00|")) == 60
    n += 3

    # 2b. 高點版 = 低點版的鏡像：把每個價格對 100 鏡射 (200 − x)、lo ↔ hi → 結果與低點版相同 (開低 ↔ 開高)
    def mirror(r):
        f = lambda x: round(200 - x, 6)  # noqa: E731
        return {"prev_close": f(r["prev_close"]), "open": f(r["open"]), **{k: f(r[k]) for k in MK},
                "hi": {k: f(v) for k, v in r["lo"].items()}, "lo": {k: f(v) for k, v in r["hi"].items()}}
    th = P._hour_high_table(m, {d: mirror(r) for d, r in mk2.items()})
    sw = lambda k: k.replace("|dn|", "|@|").replace("|up|", "|dn|").replace("|@|", "|up|")  # noqa: E731
    for k, c in t["cells"].items():
        assert th["cells"][sw(k)]["p_done"] == c["p_done"] and th["cells"][sw(k)]["n"] == c["n"], (k, th["cells"].get(sw(k)), c)
    for k, c in t["cells_rb"].items():
        assert th["cells_rb"][sw(k)]["p_done"] == c["p_done"], k
    assert "fin_q80" in th["marks"]["10:00"] and th["marks"]["10:00"]["fin_q80"] > 0 and "fin_q20" not in th["marks"]["10:00"]
    n += 3

    # 3. 視窗：只取近 MARKS_WINDOW_DAYS 日曆天，None (抓不到) 略過
    orig = twse.intraday_marks
    try:
        twse.intraday_marks = lambda *a, **k: {"2020-01-02": a_ if (a_ := rec(1, 1, [1] * 7)) else None, "2026-10-29": c, "2026-10-30": None}
        w = P._marks_window(dt.date(2026, 11, 1))
        assert list(w) == ["2026-10-29"], w
    finally:
        twse.intraday_marks = orig
    n += 1
    print(f"test_pullback_marks: {n} checks passed")


if __name__ == "__main__":
    main()
