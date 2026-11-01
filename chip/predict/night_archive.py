"""夜盤盤中快照 (2026-11-01)：隔日預測在夜盤收盤 (05:00) 前只能用「不含夜盤」版本 (OOS 方向命中約 55%，含夜盤約 71%)。
夜盤進行到一半 (例如 21:00 / 22:30，美股開盤後) 的漲跌能補回多少命中，要有歷史才能驗證 — 但 FinMind after_market 只有夜盤收盤值、
gh-pages 以 force_orphan 發布沒有歷史 → 從現在起每次夜盤時段的發布存一筆快照，累積幾個月後再研究 (研究前 App 不使用)。
格式 = JSON Lines，每次一行：
  {"ts": "YYYY-MM-DD HH:MM", "phase": "night", "tx": {last, ref, chg, time, date, volume}, "g": {sp500: chg%, nasdaq: …, sox: …, tsm_adr: …, vix: …}}
沿用 chip.pit_ledger 的 carry_over / append / 縮水保護 (Pages 整站覆蓋 → 每次先帶回上一版，data/cache 另存備份)，以 ts 去重。
"""
from __future__ import annotations

import datetime as dt

from .. import config
from .. import pit_ledger as PL

NAME = "night_partial.jsonl"
CACHE_PATH = config.CACHE_DIR / NAME
CAP = 6000            # 每晚約 12~30 筆 (17:00~22:30 每 30 分，Worker 加密時每 10 分) → 約 1 年
G_KEYS = ("sp500", "nasdaq", "sox", "tsm_adr", "vix")


def line_from(snap: dict | None, now: dt.datetime | None = None) -> dict | None:
    """夜盤進行中 (phase == night 且有成交價) 才回傳一行；其他時段 / 缺值 → None。"""
    if not isinstance(snap, dict) or snap.get("phase") != "night":
        return None
    tn = snap.get("tx_night") or {}
    if not isinstance(tn, dict) or not tn.get("last") or tn.get("change_pct") is None:
        return None
    now = now or dt.datetime.now(config.TZ)
    g = {}
    for q in snap.get("global") or []:
        if isinstance(q, dict) and q.get("key") in G_KEYS and q.get("chg_pct") is not None:
            g[q["key"]] = round(float(q["chg_pct"]), 3)
    return {"ts": now.strftime("%Y-%m-%d %H:%M"), "mode": "night", "phase": "night",
            "tx": {"last": tn.get("last"), "ref": tn.get("ref"), "chg": round(float(tn["change_pct"]), 3), "time": tn.get("time"), "date": tn.get("date"),
                   "volume": tn.get("volume")},
            "g": g}


def publish(path, snap: dict | None, urls) -> tuple[int, int]:
    """帶回上一版 (Pages ∪ cache ∪ 本機) → 夜盤中追加一行。回傳 (帶回筆數, 追加後筆數)。"""
    n0 = PL.carry_over(path, urls, cache=CACHE_PATH, cap=CAP)
    line = line_from(snap)
    n1 = PL.append(path, line, cap=CAP, cache=CACHE_PATH) if line else n0
    return n0, n1
