"""除權息預告表每日快照 (r7，2026-10-07)：證交所 TWT48U 只給「現在」已公告、未來約 3~4 週的除權息 (沒有歷史 API)。

為什麼要存：雷達研究 (scratchpad r7) 發現「股利相關特徵」(近 12 個月殖利率、剛除息、即將除息) 點估計 +3~4pt 命中，
但證據集中在 2025 年；要用「推薦當天已公告的除息日」做時點正確的特徵/驗證，必須從現在開始每天存快照。
格式 = JSON Lines (沿用 chip.pit_ledger 的 carry_over / append / 縮水保護)：每天一行
  {"ts": "YYYY-MM-DD", "mode": "twt48u", "n": 筆數, "rows": [[除權息日 ISO, 代號, 名稱, 權/息/權息, 無償配股率, 現增配股率, 現增認購價|None, 現金股利|None], ...]}
發布：data/exdiv_announce.jsonl (Pages 整站覆蓋 → 每次帶回上一版；data/cache 另存備份)。App 不讀，純研究用。"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

from .. import config
from .. import pit_ledger as PL
from ..http import session

log = logging.getLogger(__name__)
URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT48U"
CACHE_PATH = config.CACHE_DIR / "exdiv_announce.jsonl"
CAP = 1200            # 約 5 年交易日
UA = {"User-Agent": "Mozilla/5.0"}


def _num(s):
    try:
        v = float(str(s).replace(",", ""))
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _roc_date(s: str) -> str | None:
    """'115年10月12日' → '2026-10-12'。"""
    m = re.match(r"\s*(\d{2,3})年(\d{1,2})月(\d{1,2})日", str(s or ""))
    if not m:
        return None
    return f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def parse(j: dict) -> list[list]:
    """TWT48U JSON → rows；欄位缺/待公告 → None。stat 不是 OK 或沒有 data → 丟 LookupError (不要回空清單，免得寫進空快照)。"""
    if not isinstance(j, dict) or j.get("stat") != "OK" or not j.get("data"):
        raise LookupError(f"TWT48U: stat={j.get('stat') if isinstance(j, dict) else type(j)} rows=0")
    f = j.get("fields") or []
    ix = {k: i for i, k in enumerate(f)}
    need = ("除權除息日期", "股票代號", "名稱", "除權息", "無償配股率", "現金增資配股率", "現金增資認購價", "現金股利")
    if any(k not in ix for k in need):
        raise LookupError(f"TWT48U: fields changed {f[:9]}")
    out = []
    for r in j["data"]:
        d = _roc_date(r[ix["除權除息日期"]])
        code = str(r[ix["股票代號"]]).strip()
        if not d or not code:
            continue
        out.append([d, code, str(r[ix["名稱"]]).strip(), str(r[ix["除權息"]]).strip(), _num(r[ix["無償配股率"]]), _num(r[ix["現金增資配股率"]]),
                    _num(r[ix["現金增資認購價"]]), _num(r[ix["現金股利"]])])
    if not out:
        raise LookupError("TWT48U: no parsable rows")
    return out


def fetch() -> list[list]:
    r = session().get(URL, params={"response": "json"}, headers=UA, timeout=30)
    return parse(r.json())


def line_from(rows: list[list], asof: str | None = None) -> dict | None:
    if not rows:
        return None
    ts = asof or dt.datetime.now(config.TZ).strftime("%Y-%m-%d")
    return {"ts": ts, "mode": "twt48u", "n": len(rows), "rows": rows}


def carry(path: Path | str, urls, cache=None) -> int:
    """帶回上一版 (Pages ∪ raw gh-pages ∪ data/cache 備份 ∪ 本機)；回傳行數。"""
    return PL.carry_over(path, urls, cache=CACHE_PATH if cache is None else cache)


def snapshot(path: Path | str, cache=None, rows: list[list] | None = None, asof: str | None = None) -> int:
    """今天抓一次 TWT48U 追加一行 (同日覆蓋)；抓不到 → 不寫 (回傳既有行數)。"""
    try:
        rows = rows if rows is not None else fetch()
    except Exception as e:  # noqa: BLE001
        log.warning("exdiv_archive: TWT48U unavailable: %s", e)
        return len(PL.load(path))
    return PL.append(path, line_from(rows, asof), cap=CAP, cache=CACHE_PATH if cache is None else cache)
