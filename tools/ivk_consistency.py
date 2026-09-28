"""研究工具 (不進排程；r2m final_spec §4.4 spec_fix 6)：上線 ivk 與「產生 txo_ivk_seed.json 的同一演算法」的一致性檢查。

txo_ivk_seed.json (研究 desk/opt_bands/a1_seed.py) = 各序列 ATM IV → taifex_opt._cm 總變異數內插 (n≥2 序列、兩端平推)，
到期天數 n 用「事後」實際交易日曆；上線 (taifex_opt.features) 的 n 用當時的 TWSE 休市表 → 颱風假等臨時休市會讓上線 n 多算 1 天、IV 低估。
本工具讀操盤台歸檔 opt_hist 已存的 ss (各序列 n / atm) 與上線 ivk：
  (a) 以歸檔的 n 重算 (檢查內插本身一致)；
  (b) 以事後日曆重算 n (到期日 = 上線日曆第 n 個交易日；事後日曆 = FinMind 加權實際交易日，之後接 TWSE 休市表)，再重算 ivk；
|重算 − 上線| / 上線 > 1% (任一 k) 的日期列為 IV 無效。建議上線後前 20 個交易日每週跑一次。

    python tools/ivk_consistency.py                 # 最近 20 列，只印報表
    python tools/ivk_consistency.py --days 60 --out report.json
    python tools/ivk_consistency.py --apply         # 無效日寫入 data/models/ivk_invalid.json (range_levels.ivk_history 會剔除)

未來重建種子時，n 應改用「事前」TWSE 休市表 (與上線一致)，不要用事後實際交易日曆。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KS = (1, 2, 3, 5, 10, 20)
TOL = 0.01


def _ivk(ss: pd.DataFrame) -> dict:
    from chip.sources import taifex_opt
    out = {}
    for k in KS:
        v = taifex_opt._cm(ss, "atm", k)
        out[str(k)] = None if v is None or not np.isfinite(v) else round(float(v), 5)
    return out


def _rel(a: dict, b: dict) -> float | None:
    d = [abs(float(a[k]) - float(b[k])) / float(b[k]) for k in a if a.get(k) is not None and (b or {}).get(k) not in (None, 0)]
    return max(d) if d else None


def check(rows: list[dict], actual_days: list[str], live_cal=None) -> list[dict]:
    """rows = 歸檔 opt_hist (含 date / ivk / ss)；actual_days = 事後實際交易日 (升冪)；live_cal(date, n) = 當時的上線日曆 (預設 twse.next_trading_days)。"""
    if live_cal is None:
        from chip.sources import twse
        live_cal = twse.next_trading_days
    act = sorted(set(actual_days))
    last_act = act[-1] if act else ""
    out = []
    for r in rows:
        ss = pd.DataFrame(r.get("ss") or [])
        if ss.empty or not r.get("ivk") or "n" not in ss or "atm" not in ss:
            continue
        ss = ss.dropna(subset=["n", "atm"])
        ss["n"] = ss["n"].astype(int)
        d = str(r["date"])[:10]
        a = _ivk(ss)
        nmax = int(ss["n"].max())
        cal = live_cal(d, nmax)
        n2 = []
        for n in ss["n"]:
            exp = cal[n - 1]
            if exp <= last_act:                       # 到期日已過 → 用實際交易日計數
                n2.append(sum(1 for x in act if d < x <= exp))
            else:                                     # 到期日在未來 → 已實現段用實際、之後沿用上線日曆
                n2.append(sum(1 for x in act if d < x) + sum(1 for x in cal[:n] if x > last_act))
        ss2 = ss.assign(n=n2)
        b = _ivk(ss2[ss2["n"] >= 1])
        ra, rb = _rel(a, r["ivk"]), _rel(b, r["ivk"])
        out.append({"date": d, "live": r["ivk"], "recalc_same_n": a, "recalc_expost_n": b, "rel_same_n": ra, "rel_expost_n": rb,
                    "n_changed": [int(x) for x, y in zip(ss["n"], n2) if x != y], "invalid": bool((ra or 0) > TOL or (rb or 0) > TOL),
                    "calendar_fallback": bool(r.get("calendar_fallback"))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="desk_archive.json 路徑 (預設 desk.load_archive()：Pages ∪ 本機快取 ∪ repo 備份)")
    ap.add_argument("--days", type=int, default=20, help="檢查最近幾列上線 ivk")
    ap.add_argument("--out", help="報表 JSON 輸出路徑")
    ap.add_argument("--apply", action="store_true", help="無效日寫入 data/models/ivk_invalid.json")
    args = ap.parse_args()
    if args.archive:
        arc = json.loads(Path(args.archive).read_text(encoding="utf-8"))
    else:
        from chip.analysis import desk
        arc = desk.load_archive()
    rows = [r for r in (arc.get("opt_hist") or []) if r.get("ivk") and r.get("ss")][-args.days:]
    if not rows:
        print("歸檔沒有含 ss 的上線 ivk 列")
        return
    from chip.sources import finmind
    px = finmind.taiex_price((dt.date.fromisoformat(rows[0]["date"][:10]) - dt.timedelta(days=10)).isoformat())
    res = check(rows, px["date"].astype(str).str[:10].tolist())
    for x in res:
        print(f"{x['date']}  同 n 差 {x['rel_same_n'] if x['rel_same_n'] is None else round(x['rel_same_n'] * 100, 3)}%  "
              f"事後 n 差 {x['rel_expost_n'] if x['rel_expost_n'] is None else round(x['rel_expost_n'] * 100, 3)}%  n 變動 {x['n_changed']}  "
              + ("→ 無效" if x["invalid"] else "ok"))
    bad = [x["date"] for x in res if x["invalid"]]
    print(f"檢查 {len(res)} 列，無效 {len(bad)}：{bad}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.apply:
        from chip.predict import model as M
        cur = set((M.load_json("ivk_invalid") or {}).get("dates") or [])
        M.save_json("ivk_invalid", {"dates": sorted(cur | set(bad)), "updated": dt.date.today().isoformat(), "tol": TOL,
                                    "note": "tools/ivk_consistency.py：上線 ivk 與種子演算法 (事後交易日曆) 差 >1% 的日期；range_levels.ivk_history 擬合時剔除"})
        print("寫入 ivk_invalid.json")


if __name__ == "__main__":
    main()
