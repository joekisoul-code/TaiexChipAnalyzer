"""除息事件與除息處理 (操盤台用；研究 exdiv 2026-09-26，28/28 測試、獨立驗證：權證條款公式全市場 2,178/2,199 逐位吻合)。

資料源 (全部免費、GitHub Actions 可抓；實測 2026-09-26)：
  已公告未除息 (日期 + 金額)：
    1. TWSE 除權除息預告表  openapi /v1/exchangeReport/TWT48U_ALL (每日 ~05:28 批次) ；即時版 rwd /rwd/zh/exRight/TWT48U?response=json
       ETF 金額未定時 CashDividend 為空 (rwd 為「待公告實際收益分配金額」) → 只有日期。
    2. FinMind TaiwanStockDividend (逐檔查免費；不帶 data_id 需贊助)：CashExDividendTradingDate、CashEarningsDistribution+CashStatutorySurplus；
       'date' 欄 = 除息基準日 (在除息日之後)；ETF 先有 0 元佔位列、之後另一列補金額 → 同除息日要去重。
  已決議未公告除息日 (上市公司，ETF 無)：TWSE openapi /v1/opendata/t187ap45_L 股利分派情形 (2330 115Q2 7.00 元，董事會 2026-08-11)。
  已除息 / 除息前一晚即可得 (P=除息前收盤、參考價)：TWSE rwd /rwd/zh/exRight/TWT49U?startDate&endDate (除息日前一交易日收盤後即列出隔日的計算結果)；
    FinMind TaiwanStockDividendResult (before_price、stock_and_cache_dividend)。

處理規格：
  E5 區間：k 日窗口含除息日 (cal[0..k-1]) → low 分位扣全額股利 D、high 分位扣 D·(k−j+1)/k，標示近似。
  次日門檻 (MA60 出場/站回、交叉觸發、含息扣抵)：若下一交易日除息 → 乘 f = 1 − D/P (精確，不是近似)。
  均線顯示：訊號一律含息；看盤 MA 在除息後 n 日內偏高 ≈ D·m/n (m = 窗口內除息前天數)，標示何日消失。
  權證 (除息保護)：K' = K·(P−D)/P、ratio' = ratio·P/(P−D)，履約價四捨五入到 0.01、比例到 0.001；
    TWSE 條款檔未更新時用公式推算並標示；區間換算權證價時窗口跨除息日用推算條款 (等價於含息價 + 原條款)。
"""
from __future__ import annotations

import datetime as dt
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

from .. import config
from ..http import cached, session

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ------------------------------------------------------------------ 小工具
def roc_to_iso(s) -> str | None:
    """'1151008' / '115年10月08日' / '115/10/08' → '2026-10-08'。"""
    t = re.sub(r"[年月/]", "-", str(s or "").strip()).replace("日", "")
    m = re.fullmatch(r"(\d{2,3})-?(\d{2})-?(\d{2})", t)
    return f"{int(m.group(1)) + 1911}-{m.group(2)}-{m.group(3)}" if m else None


def num(s) -> float | None:
    try:
        v = float(str(s).replace(",", "").strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def rhu(x: float, dp: int) -> float:
    """四捨五入 (ROUND_HALF_UP；避免 Python round 的銀行家捨入)。"""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


# ------------------------------------------------------------------ 1. 抓取 (getter 可注入：正式用 chip.http.cached + session，測試用本機 JSON)
TWSE_OPEN = "https://openapi.twse.com.tw/v1"
TWSE_RWD = "https://www.twse.com.tw/rwd/zh"
FINMIND = "https://api.finmindtrade.com/api/v4/data"


def parse_twt48u_open(rows: list[dict]) -> dict[str, list[dict]]:
    """openapi TWT48U_ALL → {代號: [{ex_date, cash|None, kind}]}。CashDividend 空字串 = 金額待公告。"""
    out: dict[str, list[dict]] = {}
    for x in rows or []:
        ex, code = roc_to_iso(x.get("Date")), str(x.get("Code") or "").strip()
        if not ex or not code or "息" not in str(x.get("Exdividend") or ""):
            continue
        c = num(x.get("CashDividend"))
        out.setdefault(code, []).append({"ex_date": ex, "cash": c if c and c > 0 else None, "kind": str(x.get("Exdividend")), "src": "TWT48U"})
    return out


def parse_twt48u_rwd(j: dict) -> dict[str, list[dict]]:
    """rwd TWT48U (即時)：fields[0]=除權除息日期、[1]=代號、[3]=權/息、[7]=現金股利 (或 HTML『待公告…』)。"""
    out: dict[str, list[dict]] = {}
    if not isinstance(j, dict) or str(j.get("stat", "")).upper() != "OK":
        return out
    for r in j.get("data") or []:
        ex, code = roc_to_iso(r[0]), str(r[1]).strip()
        if not ex or "息" not in str(r[3]):
            continue
        c = num(r[7])
        out.setdefault(code, []).append({"ex_date": ex, "cash": c if c and c > 0 else None, "kind": str(r[3]), "src": "TWT48U"})
    return out


def parse_twt49u(j: dict) -> dict[str, list[dict]]:
    """rwd TWT49U 除權除息計算結果：[0]日期 [1]代號 [3]除權息前收盤 [4]參考價 [5]權值+息值 [6]權/息。
    除息日前一交易日收盤後就會列出隔日的結果 → 可在除息前一晚取得精確 P、D。"""
    out: dict[str, list[dict]] = {}
    if not isinstance(j, dict) or str(j.get("stat", "")).upper() != "OK":
        return out
    for r in j.get("data") or []:
        ex, code = roc_to_iso(r[0]), str(r[1]).strip()
        P, ref, D = num(r[3]), num(r[4]), num(r[5])
        if ex and P and D and "息" in str(r[6]) and str(r[6]).strip() == "息":       # 只處理純現金除息；除權 (配股) 另案
            out.setdefault(code, []).append({"ex_date": ex, "cash": D, "before": P, "ref": ref, "src": "TWT49U"})
    return out


def parse_finmind_announced(rows: list[dict]) -> list[dict]:
    """FinMind TaiwanStockDividend → 以除息日去重 (金額 > 0 優先、其次 date 較新)。"""
    ev: dict[str, dict] = {}
    for r in rows or []:
        ex = str(r.get("CashExDividendTradingDate") or "")[:10]
        if not ISO.match(ex):
            continue
        cash = (num(r.get("CashEarningsDistribution")) or 0.0) + (num(r.get("CashStatutorySurplus")) or 0.0)
        stock = (num(r.get("StockEarningsDistribution")) or 0.0) + (num(r.get("StockStatutorySurplus")) or 0.0)
        key = (cash > 0, str(r.get("date") or ""))
        if ex not in ev or key > ev[ex]["_key"]:
            ev[ex] = {"ex_date": ex, "cash": round(cash, 6) if cash > 0 else None, "stock": stock or None, "record_date": str(r.get("date") or "")[:10],
                      "pay_date": str(r.get("CashDividendPaymentDate") or "")[:10] or None, "announced": (str(r.get("AnnouncementDate") or "")[:10] or None),
                      "src": "FinMind", "_key": key}
    return [{k: v for k, v in e.items() if k != "_key"} for e in sorted(ev.values(), key=lambda e: e["ex_date"])]


def parse_finmind_result(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows or []:
        D, P = num(r.get("stock_and_cache_dividend")), num(r.get("before_price"))
        if D and P and str(r.get("stock_or_cache_dividend") or "").strip() == "息":
            out.append({"ex_date": str(r.get("date"))[:10], "cash": D, "before": P, "ref": num(r.get("reference_price")), "src": "FinMind"})
    return out


def parse_t187ap45(rows: list[dict], sid: str) -> list[dict]:
    """上市公司股利分派 (董事會決議/股東會確認)：金額已定、除息日未公告的來源。ETF 不在此表。"""
    out = []
    for x in rows or []:
        if str(x.get("公司代號") or "").strip() != sid:
            continue
        c = sum(num(x.get(k)) or 0.0 for k in ("股東配發-盈餘分配之現金股利(元/股)", "股東配發-法定盈餘公積發放之現金(元/股)", "股東配發-資本公積發放之現金(元/股)"))
        if c > 0:
            out.append({"period": f"{x.get('股利年度')}{x.get('股利所屬年(季)度')}", "cash": round(c, 6), "board_date": roc_to_iso(x.get("董事會（擬議）股利分派日")),
                        "status": str(x.get("決議（擬議）進度") or "").replace("<br>", ""), "src": "t187ap45_L"})
    return sorted(out, key=lambda e: e["board_date"] or "")


# ------------------------------------------------------------------ 2. 合併
def build_events(sid: str, asof: str, *, fm_ann=None, fm_res=None, t48: dict | None = None, t49: dict | None = None,
                 t45: list | None = None, errors: dict | None = None) -> dict:
    """asof = 最後收盤日。回傳 {realized[], upcoming[], board[], estimate, sources{}}。
    upcoming 的 status：confirmed (日期+金額)、date_only (金額待公告 → 用上次金額估計、標示「估」)。"""
    errors = errors or {}
    realized: dict[str, dict] = {}
    for e in (fm_res or []):
        realized[e["ex_date"]] = dict(e)
    for e in (t49 or {}).get(sid, []):                       # 官方優先
        realized[e["ex_date"]] = {**realized.get(e["ex_date"], {}), **e}
    upcoming: dict[str, dict] = {}
    for e in (fm_ann or []):
        if e["ex_date"] > asof:
            upcoming[e["ex_date"]] = {**e, "srcs": ["FinMind"]}
    for e in (t48 or {}).get(sid, []):
        if e["ex_date"] <= asof:
            continue
        u = upcoming.get(e["ex_date"])
        if u is None:
            upcoming[e["ex_date"]] = {**e, "srcs": ["TWT48U"]}
        else:
            u["srcs"].append("TWT48U")
            if e["cash"] and u.get("cash") and abs(e["cash"] - u["cash"]) > 5e-4:
                u["conflict"] = {"FinMind": u["cash"], "TWT48U": e["cash"]}
            if e["cash"]:
                u["cash"] = e["cash"]                          # TWSE 為準
    for ex, e in list(realized.items()):                       # TWT49U 在除息前一晚就有「隔日」的結果 → 也是 upcoming 的精確版
        if ex > asof:
            upcoming[ex] = {**upcoming.get(ex, {}), "ex_date": ex, "cash": e["cash"], "before": e.get("before"), "ref": e.get("ref"),
                            "srcs": sorted(set((upcoming.get(ex) or {}).get("srcs", []) + [e["src"]]))}
    past = sorted((e for ex, e in realized.items() if ex <= asof), key=lambda e: e["ex_date"])
    last_cash = past[-1]["cash"] if past else None
    ups = []
    for ex in sorted(upcoming):
        u = upcoming[ex]
        if u.get("cash"):
            u["status"] = "confirmed"
        else:
            u["status"], u["cash_est"] = "date_only", last_cash
        ups.append(u)
    est = None
    if len(past) >= 2 and not ups:                           # 只作顯示：依配息週期推估下一次 (不用來調整價位)
        ds = [dt.date.fromisoformat(e["ex_date"]) for e in past[-5:]]
        gaps = sorted((b - a).days for a, b in zip(ds, ds[1:]))
        gap = gaps[len(gaps) // 2]
        nxt = ds[-1] + dt.timedelta(days=gap)
        est = {"ex_date_est": nxt.isoformat(), "window": [(nxt - dt.timedelta(days=12)).isoformat(), (nxt + dt.timedelta(days=12)).isoformat()],
               "cash_last": last_cash, "cadence_days": gap, "note": "依過去除息間隔推估，未公告，不用於調整價位"}
    # 董事會已決議、尚未除息：每筆已除息事件只能「消耗」一筆決議 (依時間順序配對，金額差 < 0.01)
    board, used = [], set()
    for b in sorted(t45 or [], key=lambda x: x["board_date"] or ""):
        m = next((i for i, e in enumerate(past) if i not in used and e["ex_date"] > (b["board_date"] or "") and abs(e["cash"] - b["cash"]) < 0.01), None)
        if m is None:
            board.append(b)
        else:
            used.add(m)
    return {"sid": sid, "asof": asof, "realized": past[-8:], "upcoming": ups, "board_pending": board[-2:], "estimate": est,
            "sources": {k: ("ok" if k not in errors else f"error: {errors[k]}") for k in ("FinMind", "TWT48U", "TWT49U", "t187ap45_L")}}


# ------------------------------------------------------------------ 3. E5：區間價位 (range_block 的輸出) 的除息調整
def exdiv_in_window(upcoming: list[dict], cal: list[str], k: int) -> list[tuple[int, dict]]:
    """回傳 [(j, event)]：j = 除息日在 k 日窗口 (cal[0..k-1]) 的位置 (1-based)。"""
    win = cal[:k]
    return [(win.index(e["ex_date"]) + 1, e) for e in upcoming if e["ex_date"] in win]


def adjust_range(rg: dict, upcoming: list[dict], cal: list[str]) -> dict:
    """range_block 輸出 → 就地加上除息調整 (看盤價口徑)。原含息口徑價位保留在 px_tr。
    low 分位扣全額 D；high 分位扣 D·(k−j+1)/k。date_only 事件用上次金額 (cash_est) 並標「估」。
    實證 (0050+2330，2019~2026，47 次除息，事件群聚 bootstrap)：k≤3 low20 觸及率 不調整 29.8% [20,40] → 扣全額 17.4% [11,25]；
    high80 扣全額 28.7% [21,37]、依比例 21.3% [14,29]、不調整 12.8% [7,20]。"""
    close = rg.get("close")
    if not rg or not close or not upcoming:
        return rg
    hits_all = []
    for ks, row in (rg.get("levels") or {}).items():
        k = int(ks)
        hits = exdiv_in_window(upcoming, cal, k)
        if not hits:
            continue
        for q, v in row.items():
            if not v or v.get("px") is None:
                continue
            d_tot, est = 0.0, False
            for j, e in hits:
                D = e.get("cash") or e.get("cash_est")
                if not D or e.get("suspicious"):
                    continue
                est = est or not e.get("cash")
                d_tot += D if q.startswith("low") else D * (k - j + 1) / k
            if d_tot <= 0:
                v["exdiv_unknown"] = True
                continue
            v["px_tr"] = v["px"]
            v["px"] = round(v["px"] - d_tot, 2)
            v["pct"] = round((v["px"] / close - 1) * 100, 2)
            v["exdiv_adj"] = round(d_tot, 4)
            v["exdiv_est"] = est
        hits_all += [(k, j, e["ex_date"]) for j, e in hits]
    if hits_all:
        e0 = min((e for e in upcoming if any(e["ex_date"] == h[2] for h in hits_all)), key=lambda e: e["ex_date"])
        rg["exdiv"] = {"date": e0["ex_date"], "cash": e0.get("cash"), "cash_est": e0.get("cash_est"), "status": e0.get("status"),
                       "ks": sorted({h[0] for h in hits_all}),
                       "note": ("窗口含除息日：看盤價口徑 (low 扣全額股利、high 依除息後天數比例扣)，近似；px_tr 為含息口徑"
                                + ("；金額未公告，以上次配息估計" if e0.get("status") == "date_only" else ""))}
    return rg


# ------------------------------------------------------------------ 4. 次日門檻：下一交易日除息 → 乘 f (精確)
def next_day_factor(upcoming: list[dict], next_td: str | None, close: float) -> tuple[float, dict | None]:
    """所有「次日收盤 = X」型門檻 (MA60 出場/站回、交叉觸發、含息扣抵、close_eq) 都是過去含息價的一次齊次函數；
    次日除息時過去含息價全部乘 f = (P−D)/P (P = 今日收盤)，所以門檻 (看盤價口徑) = 原門檻 × f。"""
    for e in upcoming or []:
        if next_td and e["ex_date"] == next_td and not e.get("suspicious"):
            D = e.get("cash") or e.get("cash_est")
            P = e.get("before") or close
            if D and P:
                return (P - D) / P, e
    return 1.0, None


# ------------------------------------------------------------------ 5. 均線顯示 (看盤 vs 含息)
def ma_exdiv(dates: list[str], raw: list[float], adj: list[float], realized: list[dict], upcoming: list[dict], cal: list[str],
             mas=(5, 10, 30, 60)) -> dict:
    """看盤 MA (未還原) 在除息後 n 日內含除息前價格而偏高；回傳每條均線的偏差、何日消失，以及即將除息的預告。"""
    out: dict = {"per_ma": {}}
    last = [e for e in realized if e["ex_date"] <= dates[-1]]
    if last:
        e = last[-1]
        out["last_ex"] = {"date": e["ex_date"], "cash": e["cash"]}
        for n in mas:
            if len(dates) < n:
                continue
            win = dates[-n:]
            m = sum(1 for d in win if d < e["ex_date"])
            if m == 0:
                continue
            mr, ma = sum(raw[-n:]) / n, sum(adj[-n:]) / n
            # 偏差 = (D/P)·Σ(除息前的看盤價)/n (P = 除息前收盤；含息價 = 看盤價 × (1−D/P))；經驗法則 ≈ D·m/n
            pre_sum = sum(raw[-n:][:m])
            approx = e["cash"] / e["before"] * pre_sum / n if e.get("before") else e["cash"] * m / n
            out["per_ma"][str(n)] = {"pre_ex_days": m, "raw_minus_adj": round(mr - ma, 4), "gap_pct": round((mr / ma - 1) * 100, 3),
                                     "approx": round(approx, 4), "rule_of_thumb": round(e["cash"] * m / n, 4), "clears_on": cal[m - 1] if len(cal) >= m else None,
                                     "text": f"MA{n} 看盤價仍含 {m} 日除息前價格，比含息高 {mr - ma:.2f} ({(mr / ma - 1) * 100:.2f}%)，{cal[m - 1] if len(cal) >= m else '?'} 起一致"}
    nxt = [e for e in upcoming if cal and e["ex_date"] in cal[:max(mas)]]
    if nxt:
        e = nxt[0]
        D = e.get("cash") or e.get("cash_est")
        out["upcoming"] = {"date": e["ex_date"], "cash": e.get("cash"), "cash_est": e.get("cash_est"), "in_days": cal.index(e["ex_date"]) + 1,
                           "text": (f"{e['ex_date']} 除息 {D:.2f}{'' if e.get('cash') else ' (估)'}：之後看盤價均線會高於含息均線，最多約 {D:.2f}，"
                                    f"MA n 在除息後第 n 個交易日消失；訊號與交叉以含息價為準") if D else f"{e['ex_date']} 除息 (金額待公告)"}
    return out


# ------------------------------------------------------------------ 6. 權證除息保護
def adj_terms(K: float, ratio: float, P: float, D: float, *, rounded: bool = True) -> tuple[float, float]:
    """K' = K·(P−D)/P、ratio' = ratio·P/(P−D)。P = 除息前一日收盤 (= TWT49U 除權息前收盤價)，P−D = 除息參考價。"""
    f = (P - D) / P
    k2, r2 = K * f, ratio / f
    return (rhu(k2, 2), rhu(r2, 3)) if rounded else (k2, r2)


NOTE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*標的證券除息，調整後履約價格([\d.,]+)元，調整後行使比例([\d.,]+)")


def parse_note_schedule(note: str) -> list[tuple[str, float, float]]:
    return sorted((f"{a}-{b}-{c}", float(k.replace(",", "")), float(q.replace(",", ""))) for a, b, c, k, q in NOTE_RE.findall(note or ""))


def terms_schedule(K_orig: float, adjs: list[tuple[str, float, float]], results: dict[str, dict], listed: str = "0000-00-00") -> list[dict]:
    """[(生效日, K, ratio, 來源)]：原始條款 (K 取 TWSE 原始履約價；ratio 由第一次調整回推) + 每次除息調整。
    results = {除息日: {before, cash}} 用來回推原始比例與核對公告值。"""
    sched = []
    if adjs:
        e0, k1, r1 = adjs[0]
        r0 = None
        if e0 in results:
            f0 = (results[e0]["before"] - results[e0]["cash"]) / results[e0]["before"]
            r0 = rhu(r1 * f0, 3)
        sched.append({"from": listed, "K": K_orig, "ratio": r0, "src": "原始"})
        for e, k, r in adjs:
            chk = None
            prev = sched[-1]
            if e in results and prev["ratio"]:
                chk = adj_terms(prev["K"], prev["ratio"], results[e]["before"], results[e]["cash"])
            sched.append({"from": e, "K": k, "ratio": r, "src": "TWSE 備註", "formula": chk, "formula_ok": (chk == (k, r)) if chk else None})
    return sched


def terms_guard(terms: dict, sched: list[dict], realized: list[dict], d0: str, first_trade: str | None = None) -> dict:
    """d0 (權證/標的收盤日) 以前已除息、但 TWSE 條款檔 (每日 ~05:30 批次) 的「出表日期」早於除息日 → 條款尚未反映，用公式推算並標示。
    用出表日期判斷 (不靠備註解析)，備註截斷/解析失敗時也不會重複調整；重設型權證不推算。"""
    rep = terms.get("report_date") or "0000-00-00"
    K, r = terms["strike"], terms["ratio"]
    flags, steps = [], []
    if "重設" in str(terms.get("note") or "") or "重設" in str(terms.get("style") or ""):
        return {"strike": K, "ratio": r, "estimated": False, "flags": []}
    for e in sorted(realized, key=lambda x: x["ex_date"]):
        if rep < e["ex_date"] <= d0 and (first_trade is None or e["ex_date"] > first_trade) and e.get("before") and not e.get("suspicious"):
            K, r = adj_terms(K, r, e["before"], e["cash"])
            flags.append(f"{e['ex_date']} 標的除息 {e['cash']}：TWSE 條款尚未更新，改用公式推算 K={K}、比例={r}")
            steps.append((e["ex_date"], K, r))
    return {"strike": K, "ratio": r, "estimated": bool(flags), "flags": flags, "steps": steps}


def project_terms(K: float, ratio: float, upcoming: list[dict], close: float, last_trade: str) -> list[dict]:
    """存續期內的已公告除息 → 預告調整後條款 (P 用最新收盤估計；除息前一晚 TWT49U 出來後改用官方 P)。"""
    out = []
    for e in upcoming:
        if e["ex_date"] > last_trade:
            continue
        D = e.get("cash") or e.get("cash_est")
        if not D or e.get("suspicious"):
            continue
        P = e.get("before") or close
        k2, r2 = adj_terms(K, ratio, P, D)
        out.append({"ex_date": e["ex_date"], "cash": D, "cash_est": not e.get("cash"), "P": P, "P_src": "TWT49U" if e.get("before") else "最新收盤 (估)",
                    "K": k2, "ratio": r2, "est": not e.get("cash") or not e.get("before")})
        K, ratio = k2, r2
    return out


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig, r=0.017):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    sT = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / sT
    return S * _N(d1) - K * math.exp(-r * T) * _N(d1 - sT)


# ------------------------------------------------------------------ 7. 資料新鮮度 (desk 各區塊共用)
def freshness(name: str, content_date: str | None, expect_date: str | None, fetched_at: str, *, last_modified: str | None = None,
              carried: bool = False) -> dict:
    """content_date = 內容日期 (如 TXO Date、HiStock 最後一列、ezmoney TranDate)；expect_date = 該區塊應有的資料日 (通常 last_td)。
    level：ok / lag (落後 1 個交易日以上) / carried (本次抓失敗沿用上次) / missing。"""
    if not content_date:
        lv = "missing"
    elif carried:
        lv = "carried"
    elif expect_date and content_date < expect_date:
        lv = "lag"
    else:
        lv = "ok"
    return {"name": name, "date": content_date, "expect": expect_date, "level": lv, "fetched": fetched_at, "last_modified": last_modified}


# ------------------------------------------------------------------ 8. 抓取流程
STALE: list[str] = []      # 本次執行中「抓取失敗、改用舊快取」的 URL (load_all 據此把來源標成 stale，不冒充最新)


def _no_cache():
    raise RuntimeError("no cache")


def get_json(url: str, params: dict | None = None):
    """TWSE / FinMind JSON (快取 2 小時)。抓取失敗時才用舊快取 (不論多舊) 並記在 STALE。"""
    key = "exdiv:" + url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    try:
        return cached(key, 2 * 3600, lambda: session().get(url, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).json(),
                      allow_stale=False)
    except Exception:  # noqa: BLE001
        data = cached(key, 10 ** 10, _no_cache)          # 有舊快取就回舊資料，沒有就丟例外
        STALE.append(url)
        return data


def rebase(ev: dict, d_last: str | None) -> dict:
    """load_all 以全域最後交易日切分已除息/未除息；個別標的資料日較舊 (FinMind 未更新) 時，
    把 d_last 之後的已除息事件移回 upcoming (TWT49U 列已有 cash/before)，窗口調整與次日門檻才不會漏掉除息日。"""
    if not ev or not d_last:
        return ev or {}
    real = ev.get("realized") or []
    moved = [e for e in real if str(e.get("ex_date") or "") > d_last]
    if not moved:
        return ev
    ups = {u["ex_date"]: dict(u) for u in ev.get("upcoming") or []}
    for e in moved:
        u = ups.get(e["ex_date"], {})
        cash = e.get("cash") or u.get("cash")
        ups[e["ex_date"]] = {**u, "ex_date": e["ex_date"], "cash": cash, "before": e.get("before") or u.get("before"), "ref": e.get("ref"),
                             "srcs": sorted(set((u.get("srcs") or []) + [e.get("src") or "realized"])),
                             "status": "confirmed" if cash else u.get("status", "date_only")}
    return {**ev, "realized": [e for e in real if str(e.get("ex_date") or "") <= d_last],
            "upcoming": [ups[k] for k in sorted(ups)], "rebased_to": d_last}


def load_all(sids: list[str], asof: str, get_json: Callable[..., object], *, archive: dict | None = None, today: str | None = None) -> dict:
    """每個來源獨立 try；全部公告源都失敗時沿用 archive (上次成功) 並標 carried。回傳 {sid: events}。"""
    today = today or dt.date.today().isoformat()
    errors, t48, t49, t45, fm = {}, {}, {}, [], {}
    stale: set[str] = set()
    n0 = len(STALE)
    try:                                                     # 1) 預告表：rwd 即時版優先，openapi (05:30 批次) 備援 (rwd 丟例外或空都改抓)
        t48 = parse_twt48u_rwd(get_json(f"{TWSE_RWD}/exRight/TWT48U", params={"response": "json"}))
    except Exception as e:  # noqa: BLE001
        errors["TWT48U"] = repr(e)[:120]
    if not t48:
        try:
            t48 = parse_twt48u_open(get_json(f"{TWSE_OPEN}/exchangeReport/TWT48U_ALL"))
            errors.pop("TWT48U", None)
        except Exception as e:  # noqa: BLE001
            errors.setdefault("TWT48U", repr(e)[:120])
    if len(STALE) > n0:
        stale.add("TWT48U")
    n0 = len(STALE)
    try:                                                     # 2) 計算結果：近 120 日 + 未來 (除息前一晚即列出)
        s = (dt.date.fromisoformat(asof) - dt.timedelta(days=120)).strftime("%Y%m%d")
        e_ = (dt.date.fromisoformat(today) + dt.timedelta(days=10)).strftime("%Y%m%d")
        t49 = parse_twt49u(get_json(f"{TWSE_RWD}/exRight/TWT49U", params={"startDate": s, "endDate": e_, "response": "json"}))
    except Exception as e:  # noqa: BLE001
        errors["TWT49U"] = repr(e)[:120]
    if len(STALE) > n0:
        stale.add("TWT49U")
    n0 = len(STALE)
    try:                                                     # 3) 上市公司董事會已決議 (ETF 不在表內)
        t45 = get_json(f"{TWSE_OPEN}/opendata/t187ap45_L") or []
    except Exception as e:  # noqa: BLE001
        errors["t187ap45_L"] = repr(e)[:120]
    if len(STALE) > n0:
        stale.add("t187ap45_L")
    n0 = len(STALE)
    start = (dt.date.fromisoformat(asof) - dt.timedelta(days=800)).isoformat()      # 配息週期推估要 ≥ 2 次
    for sid in sids:                                         # 4) FinMind 逐檔 (免費層可；全市場查詢要贊助)
        try:
            j = get_json(FINMIND, params={"dataset": "TaiwanStockDividend", "data_id": sid, "start_date": start})
            if not isinstance(j, dict) or j.get("status") != 200:
                raise RuntimeError(str(j.get("msg") if isinstance(j, dict) else j)[:80])
            r = get_json(FINMIND, params={"dataset": "TaiwanStockDividendResult", "data_id": sid, "start_date": start})
            fm[sid] = (parse_finmind_announced(j.get("data")), parse_finmind_result((r or {}).get("data") if isinstance(r, dict) else []))
        except Exception as e:  # noqa: BLE001
            errors.setdefault("FinMind", repr(e)[:120])
    if len(STALE) > n0:
        stale.add("FinMind")
    out = {}
    for sid in sids:
        ann, res = fm.get(sid, ([], []))
        ev = build_events(sid, asof, fm_ann=ann, fm_res=res, t48=t48, t49=t49, t45=parse_t187ap45(t45, sid), errors=errors)
        for k in stale:                                      # 抓取失敗改用舊快取：不是最新，不能標 ok
            if ev["sources"].get(k) == "ok":
                ev["sources"][k] = "stale"
        all_failed = all(k in errors for k in ("FinMind", "TWT48U", "TWT49U"))
        if all_failed and archive and (archive.get(sid) or {}).get("upcoming"):
            ev["upcoming"] = [u for u in archive[sid]["upcoming"] if u["ex_date"] > asof]
            ev["carried_from"] = archive[sid].get("asof")
        for u in ev["upcoming"]:                              # 合理性：股利 > 前收 15% → 不採用
            ref = u.get("before") or u.get("close_ref")
            D = u.get("cash") or u.get("cash_est")
            if D and ref and D / ref > 0.15:
                u["suspicious"] = True
        out[sid] = ev
    return out
