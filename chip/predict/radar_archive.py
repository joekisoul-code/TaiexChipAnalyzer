"""雷達候選池每日歸檔 (r14，2026-10-13)：每個掃描日把「整個評分過的候選池」(約 80 檔) 存成一行，給之後的前瞻 (forward) 檢定用。

為什麼要存：radar6 研究 (scratchpad radar6/research_results.json、u_prior.json promising_untested「Shadow/forward-test infrastructure」) 的結論是
剩下的點子 (A+ 當訊號級、嚴格 A、大盤成交集中度 conc20 當 A 閘門修飾、外資持股變化排飆股、全市場飆股排名) 都只能用「之後的資料」判斷；
但正式 treasure_live 的 scans 每天只留挖寶前 6 + 飆股前 3，整個評分過的候選池沒有存 → 之後無法重算同日池、等量對照、換門檻的結果。

來源：treasure_live.scan() 回傳的 'pool' (本模組 pool_block() 組；treasure_live.build() 會把它從 scan 拿掉 → treasure_live.json 內容不變，
另以私有鍵 '_pools' 交給 tools/export_static.py → publish())。backfill 補掃的日子也存，但標 src=backfill (補掃當時的模型，不一定時點正確)。
App 不讀，純研究用。

檔案 = JSON Lines：Pages data/radar_pool.jsonl (+ data/cache/radar_pool.jsonl 備份，在 Actions cache 內)。每個掃描日一行，依 ts 排序、保留最後 CAP 行。
發布流程同 exdiv_archive / pit_ledger：Pages 整站覆蓋 → 每次 (含 fast) 先帶回上一版 (Pages → raw gh-pages 備援 ∪ data/cache 備份 ∪ 本機)，
完整模式且掃描有日期才追加；同一天只留一行 (取捨順序：有列 > 0 列、live > backfill、模型在掃描日已存在 (model_ver 日期 ≤ ts) > 事後新模型重掃、
asof 較新 > 較舊 = latest wins)；
縮水保護：不用 0 行覆蓋、也不用比既有檔少的行數覆蓋 (CAP 截掉的除外)；任何錯誤只記 log，不讓發布失敗。

每行：
  {"ts": 掃描日 "YYYY-MM-DD", "mode": "scan", "v": 1, "src": "live" | "backfill", "asof": 寫入時間 "YYYY-MM-DD HH:MM:SS" (台北),
   "model_ver": 挖寶模型 trained_at, "th_A", "th_Aplus", "th_q92" (模型檔有才有；嚴格 A 影子門檻), "gate_lt" (A 級要大盤月線乖離 < 此值),
   "th_top10" (飆股模型門檻), "surge_k" (飆股取前幾名),
   "mkt_close" (加權收盤), "mkt_bias20" (大盤月線乖離 %), "mkt_pct" (大盤當日 %),
   "conc10", "conc10_stk", "mkt_value" (全市場成交金額，億元), "mkt_n", "mkt_n_stk" (見下),
   "pool_n" (候選池檔數), "scored_n" (評到分的檔數), "errors" (評分例外數), "n" (= len(rows)),
   "cols": COLS, "rows": [[...], ...]}
rows 每列 (順序 = cols；數值為整數時寫成整數)：
  code   代號 (str)
  srank  候選池名次 1.. = 模型宇宙內 (pool_rows) screen() power 由大到小、排除 00 開頭 ETF 後的名次 (沒評到分的名次不出現 → 會有空號)
  p      挖寶模型分 (4 位小數)
  ps     飆股模型分 (4 位；沒有飆股模型 → null)
  pct    當日漲跌 % (2 位；證交所漲跌價差 ÷ 前一日參考價)
  close  收盤 (證交所盤後原始價，未還原；2 位)
  value  成交金額，單位「億元」(2 位；元 = value × 1e8)
  tier   score_one 的等級 A+/A/B+/B (模型等級本身；不管有沒有進挖寶清單)
  pbr    淨值比 (BWIBBU；2 位；缺 → null)
  per    本益比 (MI_INDEX；2 位；證交所對虧損/不適用給 0 → 照存 0；缺 → null)
  power  screen() 候選分 (2 位；srank 依此排序)
  pick   位元旗標：1 = 在當日挖寶清單 (scan treasure：前 40 且 pct < 9.4)、2 = 飆股前 surge_k (scan surge)；0 = 都不是

conc10 (當日成交集中度水準；radar6 E1 的 conc20 = 這個序列的 5 日均 − 20 個交易日前的 5 日均，研究端自己算)：
  資料 = scan() 已載入的 TWSE 盤後全市場表 (treasure_live.market_snapshot：MI_INDEX type=ALLBUT0999 每日收盤行情，
         代號符合 \\d{4,6}[A-Z]? 且收盤 > 0 的全部列 — 含股票、ETF/ETN、TDR、特別股、受益證券，不含權證/牛熊證；不是只有模型宇宙)。
  去掉成交金額缺值或 ≤ 0 的列 (零成交) → conc10 = 成交金額前 10 大合計 ÷ 全部合計 (5 位小數)；mkt_value = 全部合計 (億，1 位)、mkt_n = 列數。
  conc10_stk = 同上但只算 4 碼普通股 (代號 ^[1-9]\\d{3}$：排除 00 開頭 ETF 與帶字母者) 且 |pct| ≤ 11% (同 E1 去掉 |日報酬| > 11% 的股日)，
         對應研究面板 twse_10y 的口徑 (研究驗證 MI 股票口徑與面板水準相關 .985、日變化 .966；含 ETF 口徑 .978 / .949)；mkt_n_stk = 列數。
  有效列 < 20 → null (不完整的表)。

大小 (2026-10-02 實際 MI_INDEX 1372 列、宇宙內前 80 候選、78 檔評到分的模擬)：每列約 64 bytes、一行約 5.5 KB (表頭約 0.5 KB)
  → CAP 1250 行 (約 5 年交易日) 約 6.8 MB；export_static 每次印本次一行、檔案大小與 CAP 估計。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from pathlib import Path

from .. import config
from .. import pit_ledger as PL
from ..serialize import clean

log = logging.getLogger(__name__)
NAME = "radar_pool.jsonl"
CACHE_PATH = config.CACHE_DIR / NAME                   # Actions cache 內的備份 (site/data 每次 runner 都是空的)
RAW_URL = "https://raw.githubusercontent.com/joekisoul-code/TaiexChipAnalyzer/gh-pages/data/" + NAME
CAP = 1250                                             # 約 5 年交易日
MODE, VER = "scan", 1
COLS = ["code", "srank", "p", "ps", "pct", "close", "value", "tier", "pbr", "per", "power", "pick"]
PICK_TREASURE, PICK_SURGE = 1, 2
CONC_K, CONC_MIN_N = 10, 20
_STK = re.compile(r"[1-9]\d{3}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _num(x, nd: int):
    """有限數 → round(nd)，整數值寫成 int (省位元組)；None / NaN / inf / 非數字 → None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    v = round(v, nd)
    return int(v) if v == int(v) else v


def conc(rows: dict | None) -> dict:
    """當日成交集中度 (定義見模組說明)。rows = market_snapshot() 的 {code: {value, pct, ...}}。"""
    allv: list[float] = []
    stk: list[float] = []
    for code, r in (rows or {}).items():
        try:
            v = float((r or {}).get("value"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(v) and v > 0):
            continue
        allv.append(v)
        try:
            pct = float((r or {}).get("pct"))
        except (TypeError, ValueError):
            pct = float("nan")
        if _STK.fullmatch(str(code).strip()) and math.isfinite(pct) and abs(pct) <= 11:
            stk.append(v)

    def share(vs: list[float]):
        if len(vs) < CONC_MIN_N:
            return None
        vs = sorted(vs, reverse=True)
        t = sum(vs)
        return round(sum(vs[:CONC_K]) / t, 5) if t > 0 else None

    return {"conc10": share(allv), "conc10_stk": share(stk), "mkt_value": _num(sum(allv) / 1e8, 1) if allv else None, "mkt_n": len(allv), "mkt_n_stk": len(stk)}


def pool_block(D: str, pool: list[dict], res: dict, treasure: list[dict] | None, surge: list[dict] | None, tm: dict | None,
               mkt_rows: dict | None, day: dict | None = None) -> dict:
    """scan() 的候選池 → {date, 日層級欄位..., cols, rows} (純函式、不連網)。
    pool = screen() 後的候選 (已排除 ETF、依 power 排序；名次 = 位置 + 1)；res = {code: score_one 結果} (沒有 = 沒評到分 → 不列)；
    treasure / surge = scan 的清單 (標 pick)；mkt_rows = market_snapshot 全部列 (算 conc10)；day = scan 的 mkt_close / mkt_bias20 / mkt_pct / pool_n / scored_n / errors。"""
    tm = tm or {}
    day = day or {}
    res = res or {}
    sg = tm.get("surge") or {}
    t_codes = {r.get("code") for r in treasure or []}
    s_codes = {r.get("code") for r in surge or []}
    rows = []
    for i, r in enumerate(pool or [], 1):
        code = r.get("code")
        s = res.get(code)
        if not s:
            continue
        try:
            val = float(r.get("value")) / 1e8
        except (TypeError, ValueError):
            val = None
        rows.append([str(code), i, _num(s.get("p"), 4), _num(s.get("ps"), 4), _num(r.get("pct"), 2), _num(r.get("close"), 2), _num(val, 2), s.get("tier"),
                     _num(r.get("pbr"), 2), _num(r.get("per"), 2), _num(r.get("power"), 2),
                     (PICK_TREASURE if code in t_codes else 0) | (PICK_SURGE if code in s_codes else 0)])
    try:
        cc = conc(mkt_rows)
    except Exception as e:  # noqa: BLE001
        log.warning("radar_archive conc: %s", e)
        cc = {"conc10": None, "conc10_stk": None, "mkt_value": None, "mkt_n": 0, "mkt_n_stk": 0}
    out = {"date": D, "model_ver": tm.get("trained_at"), "th_A": tm.get("th_A"), "th_Aplus": tm.get("th_Aplus"),
           **({"th_q92": tm["th_q92"]} if tm.get("th_q92") is not None else {}),
           "gate_lt": (tm.get("gate") or {}).get("m_bias20_lt"), "th_top10": sg.get("th_top10"), "surge_k": int(((sg.get("def") or {}).get("topk")) or 3),
           **{k: day.get(k) for k in ("mkt_close", "mkt_bias20", "mkt_pct", "pool_n", "scored_n", "errors")},
           **cc, "cols": list(COLS), "rows": rows}
    return clean(out)


def line_from(pool: dict | None, src: str | None = None, asof: str | None = None) -> dict | None:
    """pool_block() 的結果 → 一行 (純 Python 型別)；沒有掃描日 / 日期格式不對 / rows 不是 list → None。"""
    if not isinstance(pool, dict) or not isinstance(pool.get("rows"), list):
        return None
    ts = str(pool.get("date") or "")[:10]
    if not _DATE.fullmatch(ts):
        return None
    body = {k: v for k, v in pool.items() if k not in ("date", "cols", "rows", "src", "ts", "mode", "v", "asof", "n")}
    return clean({"ts": ts, "mode": MODE, "v": VER, "src": src or pool.get("src") or "live",
                  "asof": asof or dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), **body,
                  "n": len(pool["rows"]), "cols": list(pool.get("cols") or COLS), "rows": pool["rows"]})


def _rank(x: dict) -> tuple:
    """同一天兩行取哪個：有列 > 0 列、live > backfill、模型在掃描日當時已存在 (model_ver 日期 ≤ ts) > 事後的新模型、asof 較新 (latest wins)。
    第三項防的是：週日重訓後，週一盤後表還沒公布 → scan 退回上週五、用新模型重掃，不該蓋掉上週五當時真正的那一行。"""
    mv = str(x.get("model_ver") or "")[:10]
    return ((x.get("n") or len(x.get("rows") or [])) > 0, x.get("src") != "backfill", not mv or mv <= str(x.get("ts") or "")[:10], str(x.get("asof") or ""))


def merge(*groups, cap: int = CAP) -> list[dict]:
    """以掃描日 (ts 前 10 字) 去重 (依 _rank 取捨，同分取後面的)、依 ts 排序、保留最後 cap 行。"""
    m: dict[str, dict] = {}
    for g in groups:
        for r in g or []:
            if not (isinstance(r, dict) and _DATE.fullmatch(str(r.get("ts") or "")[:10])):
                continue
            k = str(r["ts"])[:10]
            if k not in m or _rank(r) >= _rank(m[k]):
                m[k] = r
    rows = sorted(m.values(), key=lambda r: str(r["ts"]))
    return rows[-cap:] if cap else rows


def _dumps(r: dict) -> str:
    return json.dumps(clean(r), ensure_ascii=False, separators=(",", ":"), default=str)


def _write(path: Path | str, rows: list[dict], cap: int = CAP) -> bool:
    """寫檔 (緊湊 JSON，每行一個掃描日)。縮水保護：0 行不寫 (不建空檔)；比既有檔少 (且不是 CAP 截掉的) 不寫。回傳是否有寫。"""
    p = Path(path)
    n_old = len(PL.load(p)) if p.exists() else 0
    if not rows:
        if n_old:
            log.warning("radar_archive: refuse to overwrite %s (%d lines) with 0 lines", p, n_old)
        return False
    if len(rows) < min(n_old, cap):
        log.warning("radar_archive: refuse to shrink %s from %d to %d lines", p, n_old, len(rows))
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(_dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)
    return True


def _cp(cache) -> Path | None:
    """None → CACHE_PATH (data/cache 備份)；False → 不用備份 (測試 / 離線)；其餘視為路徑。"""
    if cache is False:
        return None
    return CACHE_PATH if cache is None else Path(cache)


def carry(path: Path | str, urls, fetch=PL.fetch_published, cache=None, cap: int = CAP) -> int:
    """帶回上一版：Pages (urls 依序，Pages → raw gh-pages) ∪ data/cache 備份 ∪ 本機 → 寫回兩處；回傳行數。三個來源都空 → 不建檔 (首次)。"""
    cp = _cp(cache)
    prev = fetch(urls) if urls else []
    rows = merge(prev, PL.load(cp) if cp else [], PL.load(path), cap=cap)
    if not rows:
        log.info("radar_archive: no lines from Pages / cache / local (first run?)")
        return 0
    if not prev:
        log.warning("radar_archive: Pages fetch returned 0 lines; carrying %d lines from cache/local", len(rows))
    _write(path, rows, cap)
    if cp:
        _write(cp, rows, cap)
    return len(rows)


def append(path: Path | str, lines, cap: int = CAP, cache=None) -> int:
    """追加 (一行或多行；同一天依 _rank 取捨)，保留最後 cap 行，同步寫到 data/cache 備份；回傳行數。"""
    cp = _cp(cache)
    new = [ln for ln in (lines if isinstance(lines, list) else [lines]) if isinstance(ln, dict)]
    if not new:
        return len(PL.load(path))
    rows = merge(PL.load(cp) if cp else [], PL.load(path), new, cap=cap)
    _write(path, rows, cap)
    if cp:
        _write(cp, rows, cap)
    return len(rows)


def publish(path: Path | str, pools=None, fast: bool = False, urls=(), fetch=PL.fetch_published, cache=None, asof: str | None = None,
            cap: int = CAP) -> dict:
    """export_static 的掛鉤 (不丟例外)：先帶回上一版；完整模式 (fast=False) 且 pools 有掃描日才追加。
    pools = treasure_live.build() 的 '_pools' (list[pool_block 結果 + src])，也可給單一 dict / None。
    回傳 {carried, lines, added[{ts, src, n}], bytes, line_bytes (本次最後一行), est_cap_bytes (平均每行 × cap), error}。"""
    rep = {"carried": 0, "lines": 0, "added": [], "bytes": 0, "line_bytes": None, "est_cap_bytes": None, "cap": cap, "error": None}
    try:
        rep["carried"] = rep["lines"] = carry(path, urls, fetch=fetch, cache=cache, cap=cap)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"carry: {e}"
        log.warning("radar_archive carry failed: %s", e)
    if not fast:
        try:
            pl = [pools] if isinstance(pools, dict) else list(pools or [])
            lines = [ln for ln in (line_from(p, asof=asof) for p in pl) if ln]
            if lines:
                rep["lines"] = append(path, lines, cap=cap, cache=cache)
                rep["added"] = [{"ts": ln["ts"], "src": ln["src"], "n": ln["n"]} for ln in lines]
                rep["line_bytes"] = len(_dumps(lines[-1]).encode("utf-8")) + 1
        except Exception as e:  # noqa: BLE001
            rep["error"] = (rep["error"] + "; " if rep["error"] else "") + f"append: {e}"
            log.warning("radar_archive append failed: %s", e)
    try:
        p = Path(path)
        if p.exists():
            rep["bytes"] = p.stat().st_size
            n = rep["lines"] or len(PL.load(p))
            if n:
                rep["est_cap_bytes"] = round(rep["bytes"] / n * cap)
    except Exception as e:  # noqa: BLE001
        log.debug("radar_archive size: %s", e)
    return rep


def report_txt(rep: dict, fast: bool = False) -> str:
    """export_static 印的一行 (含大小估計)。"""
    add = "、".join(f"{a['ts']} {a['src']} {a['n']} 檔" for a in rep.get("added") or [])
    s = f"radar_pool: carried {rep.get('carried', 0)} → {rep.get('lines', 0)} lines"
    s += f" (+{len(rep.get('added') or [])}: {add})" if add else (" (fast：只帶回)" if fast else " (今天沒有新的掃描日)")
    if rep.get("line_bytes"):
        s += f"；本次一行 {rep['line_bytes'] / 1024:.1f} KB"
    if rep.get("bytes"):
        s += f"；檔案 {rep['bytes'] / 1024:.0f} KB"
    if rep.get("est_cap_bytes"):
        s += f"；CAP {rep.get('cap', CAP)} 行估 {rep['est_cap_bytes'] / 1e6:.1f} MB"
    if rep.get("error"):
        s += f"；錯誤 {rep['error']}"
    return s
