"""2026-10-13：pit_ledger.carry_over 的上限由呼叫端決定 (exdiv_archive CAP 1200 原本被 merge 預設 400 截掉)：python tests/test_pit_carry_cap.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chip import pit_ledger as PL  # noqa: E402
from chip.predict import exdiv_archive as XA  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print("  ✓", name)


rows = [{"ts": f"2020-01-01T00:00:{i:04d}", "mode": "twt48u", "n": 0, "rows": []} for i in range(700)]
rows = [{**r, "ts": f"{2020 + i // 300}-{(i // 25) % 12 + 1:02d}-{i % 25 + 1:02d}"} for i, r in enumerate(rows)]
with tempfile.TemporaryDirectory() as d:
    p, c = Path(d) / "a.jsonl", Path(d) / "c.jsonl"
    n = PL.carry_over(p, "x", fetch=lambda u: rows, cache=c)
    check("預設上限 400 (pit 帳本行為不變)", n == 400 and len(PL.load(p)) == 400)
    n2 = PL.carry_over(Path(d) / "b.jsonl", "x", fetch=lambda u: rows, cache=Path(d) / "c2.jsonl", cap=1200)
    check("cap=1200 → 700 行全保留", n2 == 700)
    calls = {}
    orig = PL.carry_over
    PL.carry_over = lambda *a, **k: calls.update(k) or 0
    try:
        XA.carry(Path(d) / "x.jsonl", ["x"], cache=Path(d) / "c3.jsonl")
    finally:
        PL.carry_over = orig
    check("exdiv_archive.carry 把 CAP 1200 傳給 carry_over", calls.get("cap") == XA.CAP == 1200)
print(f"\npit carry cap: {ok} passed")
