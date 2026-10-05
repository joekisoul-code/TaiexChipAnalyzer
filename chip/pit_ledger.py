"""時點帳本 (pr2 P5，2026-10-05)：gh-pages 以 force_orphan 發布 → forecast.json 沒有歷史，「真的顯示過什麼」只能靠事後從 GitHub 活動 API 撈。

每次 **完整** 發布把當次 forecast / verdict / 挖寶 A/A+ 壓成一行 (約 3KB) 追加到 site/data/pit_ledger.jsonl；fast 發布只把上一版帶回
(整站覆蓋不帶就會消失)。發布前先 carry_over：Pages 上一版 ∪ data/cache 備份 ∪ 本機 site/data，以 (ts, mode) 去重、依 ts 排序、保留最後 PIT_MAX 行
(~400 次完整發布 ≈ 1 年)，寫回兩處。
保險機制 (同 treasure_live.LOCAL)：publish.yml 只快取 data/cache 與 chip.sqlite，site/data 每次 runner 都是空的 → 若只靠 Pages 抓取，一次 timeout/5xx/404
就會把整份歷史洗掉 (fast 發布沒有檔 → Pages 404 → 永遠抓不到)。所以 (1) 備份放 config.CACHE_DIR/pit_ledger.jsonl (在 Actions cache 內)；
(2) 抓取走 chip.http.session() (Retry 4 次) 並可給 raw gh-pages 備援網址；(3) 逐行容錯解析 (一行壞掉不丟整份)；(4) 縮水保護：不用 0 行覆蓋有內容的檔。
這是 learn.json 之外唯一可稽核的「顯示紀錄」(learn 帳本只記叫牌，不記 verdict 桶 / 水準 / 挖寶)。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from . import config
from .http import session
from .serialize import clean

log = logging.getLogger(__name__)
PIT_MAX = 400
CACHE_PATH = config.CACHE_DIR / "pit_ledger.jsonl"   # Actions cache 內的備份；site/data 是 gitignore 且每次重建
ND_KEYS = ("n", "date", "variant", "call", "call_strength", "call_model", "p_up", "conf_tier", "level", "buy_at", "sell_at", "stop", "target", "range_mode",
           "range_sigma_src", "recent_flag", "recent_hit", "recent_up_rate")
HZ_KEYS = ("call", "call_model", "call_strength", "p_up", "call_hit", "call_smart", "call_five", "variant")
VD_KEYS = ("verdict", "call", "call_action", "bucket", "bucket_role", "net", "agree", "disagree", "conf_tier")


def _get(x: dict | None, k: str):
    v = clean((x or {}).get(k))   # numpy 標量 → Python (np.int64 不是 int、np.bool_ 不是 bool，否則會被 str() 成字串寫進帳本)
    return v if v is None or isinstance(v, (int, float, str, bool)) else str(v)


def line_from(fc: dict, treasure: dict | None = None, snap: dict | None = None, mode: str = "full", now: str | None = None) -> dict | None:
    """把一次發布壓成一行 (純 Python 型別，已過 serialize.clean)；fc 無效 → None。"""
    if not isinstance(fc, dict) or fc.get("error") or not fc.get("next_days"):
        return None
    nd = [{**{k: _get(x, k) for k in ND_KEYS}, "call_model": _get(x, "call_model") or _get(x, "call")} for x in fc.get("next_days") or []]   # call_model 無則 = call (同 learn 帳本)
    hz = {}
    for h, r in (fc.get("horizons") or {}).items():
        if isinstance(r, dict):
            hz[str(h)] = {k: _get(r, k) for k in HZ_KEYS if r.get(k) is not None}
    V = fc.get("verdict") or {}
    fv = fc.get("five") or {}
    tl = treasure or {}
    sc = tl.get("scan") or {}
    aa = [{"code": r.get("code"), "tier": r.get("tier"), "p": r.get("p")} for r in (sc.get("treasure") or []) if r.get("tier") in ("A+", "A")]
    tre = {"date": sc.get("date"), "AA": aa, "n_B": sum(1 for r in (sc.get("treasure") or []) if r.get("tier") in ("B+", "B")),
           "surge": [{"code": r.get("code"), "ps": r.get("ps")} for r in (sc.get("surge") or [])[:3]], "model_ver": (tl.get("model") or {}).get("trained_at")} if sc.get("date") else None
    return clean({"ts": now or dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "mode": mode, "date": fc.get("date"), "close": fc.get("close"),
                  "phase": (snap or {}).get("phase"), "intraday": bool(fc.get("intraday")), "trained_at": fc.get("trained_at"),
                  "next_days": nd, "horizons": hz, "verdict": {k: _get(V, k) for k in VD_KEYS} if V else None,
                  "five": {"call": fv.get("call"), "net": fv.get("net")} if fv else None, "trend7": (fc.get("trend7") or {}).get("state"),
                  "learn_flags": (fc.get("learn") or {}).get("flags"), "treasure": tre})


def parse_lines(text: str) -> list[dict]:
    """JSON Lines → list[dict]，逐行容錯：壞行 (截斷 / 非 JSON / 非 dict) 跳過、其餘保留。load() 與 fetch_published() 共用，
    免得 Pages 上一行被 CDN 截斷就把整份歷史當成空的。"""
    out: list[dict] = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def load(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return parse_lines(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.debug("pit_ledger load %s: %s", p, e)
        return []


def merge(a: list[dict], b: list[dict], cap: int = PIT_MAX) -> list[dict]:
    """以 (ts, mode) 去重 (後者覆蓋)、依 ts 排序、保留最後 cap 行。"""
    m: dict[tuple, dict] = {}
    for r in list(a) + list(b):
        if isinstance(r, dict) and r.get("ts"):
            m[(str(r["ts"]), str(r.get("mode") or ""))] = r
    rows = sorted(m.values(), key=lambda r: str(r["ts"]))
    return rows[-cap:]


def write(path: Path | str, rows: list[dict]) -> bool:
    """寫檔 (每行過 serialize.clean)。縮水保護：rows 為空時一律不動既有檔 (有 N>0 行就絕不用 0 行覆蓋，也不建空檔)；回傳是否有寫。"""
    p = Path(path)
    if not rows:
        n = len(load(p))
        if n:
            log.warning("pit_ledger: refuse to overwrite %s (%d rows) with 0 rows", p, n)
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(clean(r), ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    return True


def fetch_published(url: str | list[str] | tuple[str, ...], timeout: int = 20) -> list[dict]:
    """抓上一版 (chip.http.session()：UA + Retry 4 次)。url 可給多個 (Pages → raw gh-pages 備援)，取第一個有內容的；
    404 / 連線失敗 / 全部壞行 → []。不用 r.text：text/plain 無 charset 會被猜成 latin-1，中文變亂碼後再寫回帳本。"""
    urls = [url] if isinstance(url, str) else [u for u in (url or ()) if u]
    for u in urls:
        try:
            r = session().get(u, timeout=timeout)
            if not r.ok:
                log.debug("published pit_ledger %s -> HTTP %s", u, r.status_code)
                continue
            rows = parse_lines(r.content.decode("utf-8", errors="replace"))
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001
            log.debug("published pit_ledger unavailable (%s): %s", u, e)
    return []


def _cache_path(cache) -> Path | None:
    """None → CACHE_PATH (data/cache 備份)；False → 不用備份 (測試 / 離線)；其餘視為路徑。"""
    if cache is False:
        return None
    return CACHE_PATH if cache is None else Path(cache)


def carry_over(path: Path | str, url: str | list[str] | tuple[str, ...], fetch=fetch_published, cache=None) -> int:
    """Pages 上一版 ∪ data/cache 備份 ∪ 本機 site/data → 寫回兩處；回傳合併後行數。三個來源都空時不建檔 (首次)。
    Pages 抓不到時以備份/本機為準 (不會把歷史洗成 0 行或 1 行)。"""
    cp = _cache_path(cache)
    prev = fetch(url)
    rows = merge(merge(prev, load(cp) if cp else []), load(path))
    if not rows:
        log.warning("pit_ledger: no rows from Pages / cache / local; leaving %s untouched", path)
        return 0
    if not prev:
        log.warning("pit_ledger: Pages fetch returned 0 rows; carrying %d rows from cache/local", len(rows))
    write(path, rows)
    if cp:
        write(cp, rows)
    return len(rows)


def append(path: Path | str, line: dict | None, cap: int = PIT_MAX, cache=None) -> int:
    """追加一行 (同 ts+mode 覆蓋)，保留最後 cap 行，同步寫到 data/cache 備份；回傳行數。"""
    cp = _cache_path(cache)
    if not line:
        return len(load(path))
    rows = merge(merge(load(cp) if cp else [], load(path)), [line], cap)
    write(path, rows)
    if cp:
        write(cp, rows)
    return len(rows)
