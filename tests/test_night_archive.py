"""2026-11-01 夜盤盤中快照 (chip.predict.night_archive) + pit_ledger.append 上限修正：

- 只有 phase == night 且夜盤有成交價才記一行；其他時段 / 缺值不記
- 帶回上一版 (假 Pages) → 追加 → 同 ts 覆蓋
- pit_ledger.append 內層 merge 也要帶 cap (原本預設 400 → cap 較大的歸檔每次追加都被截到 400 行)

    python tests/test_night_archive.py
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip import pit_ledger as PL  # noqa: E402
from chip.predict import night_archive as NA  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))


def snap(phase="night", last=23000.0, chg=-0.85):
    return {"phase": phase, "tx_night": {"last": last, "ref": 23197.0, "change_pct": chg, "time": "211500", "date": "20261102", "volume": 51234},
            "global": [{"key": "sp500", "chg_pct": -0.41234}, {"key": "sox", "chg_pct": -1.2}, {"key": "btc", "chg_pct": 2.0}, {"key": "nasdaq", "chg_pct": None}]}


def main():
    n = 0
    now = dt.datetime(2026, 11, 2, 21, 15, tzinfo=TZ)
    ln = NA.line_from(snap(), now)
    assert ln["ts"] == "2026-11-02 21:15" and ln["tx"]["chg"] == -0.85 and ln["tx"]["last"] == 23000.0, ln
    assert ln["g"] == {"sp500": -0.412, "sox": -1.2}, ln["g"]          # 只留 G_KEYS、去掉 None
    n += 2
    for s in (snap("open"), snap("closed"), snap(last=None), snap(chg=None), None, {"phase": "night"}):
        assert NA.line_from(s, now) is None, s
    n += 1

    with tempfile.TemporaryDirectory() as td:
        p, cache = Path(td) / NA.NAME, Path(td) / "cache.jsonl"
        prev = [{"ts": "2026-11-02 20:45", "mode": "night", "tx": {"chg": -0.5}}]
        n0 = PL.carry_over(p, "x", fetch=lambda u: prev, cache=cache, cap=NA.CAP)
        n1 = PL.append(p, NA.line_from(snap(), now), cap=NA.CAP, cache=cache)
        n2 = PL.append(p, NA.line_from(snap(chg=-0.9), now), cap=NA.CAP, cache=cache)   # 同 ts 覆蓋
        rows = PL.load(p)
        assert (n0, n1, n2) == (1, 2, 2) and rows[-1]["tx"]["chg"] == -0.9 and len(PL.load(cache)) == 2, (n0, n1, n2, rows)
        n += 1

        # 上限：既有 450 行 + 追加 1 行，cap 1000 → 451 (修正前內層 merge 截到 400)
        big = [{"ts": f"2026-01-01 00:{i:04d}", "mode": "night"} for i in range(450)]
        PL.write(p, big); PL.write(cache, big)
        assert PL.append(p, {"ts": "2026-12-31 23:59", "mode": "night"}, cap=1000, cache=cache) == 451
        assert PL.append(p, {"ts": "2027-01-01 00:00", "mode": "night"}, cap=300, cache=cache) == 300
        n += 2
    print(f"test_night_archive: {n} checks passed")


if __name__ == "__main__":
    main()
