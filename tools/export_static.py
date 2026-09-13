"""把 App 需要的資料算好、輸出成靜態 JSON (site/data/*.json)，讓 PWA 不需要伺服器。

    python tools/export_static.py            # 完整：判讀/預測/訊號/追蹤/國際 (約 2~4 分鐘)
    python tools/export_static.py --fast     # 只更新即時快照 + 判讀 (盤中排程用，約 30~60 秒)
    python tools/export_static.py --train    # 先重訓 ML 模型再輸出 (每週一次)

輸出：site/ (index.html, app.js, style.css, manifest.json, sw.js, icons, data/*.json)
GitHub Actions 依排程執行本腳本並發布到 GitHub Pages；平板打開 Pages 網址即可。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from chip import config, notify, realtime  # noqa: E402
from chip.analysis import backtest, chips, cross_market, global_study, gov8, market, signals  # noqa: E402
from chip.predict import intraday, market_forecast  # noqa: E402
from chip.serialize import clean  # noqa: E402

SITE = ROOT / "site"
DATA = SITE / "data"


def dump(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(clean(obj), ensure_ascii=False, default=str), encoding="utf-8")
    print("  wrote", name, f"{(DATA / f'{name}.json').stat().st_size / 1024:.0f} KB")


def copy_shell() -> None:
    SITE.mkdir(exist_ok=True)
    for f in ("app.js", "style.css", "manifest.json", "sw.js", "icon-192.png", "icon-512.png"):
        shutil.copy(ROOT / "mobile" / f, SITE / f)
    html = (ROOT / "mobile" / "index.html").read_text(encoding="utf-8")
    html = html.replace('window.API_MODE = "api"', 'window.API_MODE = "static"')
    html = html.replace('href="/manifest.json"', 'href="manifest.json"').replace('href="/static/', 'href="').replace('src="/static/', 'src="')
    (SITE / "index.html").write_text(html, encoding="utf-8")
    man = json.loads((ROOT / "mobile" / "manifest.json").read_text(encoding="utf-8"))
    man["start_url"], man["scope"] = "./", "./"
    for ic in man["icons"]:
        ic["src"] = ic["src"].replace("/static/", "./")
    (SITE / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    sw = (ROOT / "mobile" / "sw.js").read_text(encoding="utf-8")
    sw = sw.replace('["/", "/static/app.js", "/static/style.css", "/manifest.json", "/static/icon-192.png"]', '["./", "./app.js", "./style.css", "./manifest.json", "./icon-192.png"]')
    sw = sw.replace('url.pathname.startsWith("/api/")', 'url.pathname.includes("/data/")')
    (SITE / "sw.js").write_text(sw, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--no-wantgoo", action="store_true")
    args = ap.parse_args()
    use_wg = not args.no_wantgoo and not args.fast
    t0 = time.time()
    copy_shell()
    if args.train:
        print("training models…")
        market_forecast.train()
        intraday.train()
        from chip.predict import range_levels, short_term, trend7
        short_term.train(verbose=False)
        try:   # 7 個交易日趨勢閘門 (走 walk-forward + 最終擬合，約 10 秒) 與 路徑型買賣點乘數 (數秒)
            print("  trend7:")
            trend7.train(verbose=True)
            print("  range_levels:")
            range_levels.fit_multipliers(backtest.load_long(), short_term._night_hist(), verbose=True)
        except Exception as e:  # noqa: BLE001
            print("  trend7/range_levels train failed:", e)
    scored, A, meta = market.run(use_wantgoo=use_wg)
    snap = realtime.snapshot(scored)
    snap["combined"] = realtime.combined_view(A["composite_smooth"], A["regime"], snap["score"]["score"], snap["score"]["label"], snap["phase"])
    snap["alerts_today"] = notify.read_log(snap["ts"][:10], 20)
    dump("realtime", snap)
    sg = {}
    try:
        r = signals.run(backtest.load_long("2010-01-01"), write=False)
        sg = {"current": r["current"], "points": r["points"].tail(120), "evaluation": r["evaluation"]}
    except Exception as e:  # noqa: BLE001
        sg = {"error": str(e)}
    dump("market", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "assessment": A, "signals": sg, "meta": {k: v for k, v in meta.items() if not k.startswith("_")},
                    "history": scored.tail(250)[["date", "open", "high", "low", "close", "foreign", "trust", "dealer", "margin_amt", "fut_foreign_net_oi", "gov8_net", "composite", "composite_smooth", "ma20", "ma60"]]})
    fc = {}
    try:
        fc = market_forecast.forecast(scored, snap)
    except Exception as e:  # noqa: BLE001
        fc = {"error": str(e)}
    hr = {}
    try:
        hr = intraday.forecast(scored, snap)
    except Exception as e:  # noqa: BLE001
        hr = {"error": str(e)}
    try:   # 近五日精修：夜盤跳空 β、隔天用小時模型、5 日規則覆蓋
        fc = market_forecast.refine_short_term(fc, hr if not hr.get("error") else None, snap, sg if "error" not in sg else None, scored)
    except Exception as e:  # noqa: BLE001
        print("  refine_short_term failed:", e)
    dump("forecast", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "forecast": fc, "hourly": hr})
    res = None
    if not args.fast:
        res = chips.assess_watchlist(chips.WATCHLIST, use_wantgoo=use_wg)
        slim = {}
        for sid, a in res.items():
            if "error" in a:
                slim[sid] = a
                continue
            slim[sid] = {k: a.get(k) for k in ("stock_id", "name", "price", "date", "notes", "score", "label", "holders", "broker_date", "quote")}
            slim[sid]["costs"] = a["costs"]
            p = a.get("profile60") or {}
            slim[sid]["profile"] = {k: p.get(k) for k in ("poc", "va_lo", "va_hi", "above_pct", "below_pct", "last")}
            slim[sid]["profile_bins"] = p.get("bins")
            slim[sid]["brokers"] = a["brokers"].head(10)
            fl = a.get("flows")
            if fl is not None and hasattr(fl, "tail"):   # 近 60 日各路資金 (主力/籌碼集中度 有玩股網時才有)
                slim[sid]["flows_tail"] = fl.tail(60)[[c for c in ("date", "close", "foreign", "trust", "dealer", "main", "skp5", "skp20", "gov8", "margin_chg") if c in fl]]
        dump("watchlist", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stocks": slim})
        dump("global", {"global": global_study.load_report(), "cross": cross_market.load_report()})
    else:
        # fast 模式不重算追蹤清單/國際研究；Pages 部署是整站覆蓋 (force_orphan)，若不把上次發布的檔案帶回來，
        # 盤中每 5 分鐘一次的 fast 會把 watchlist.json / global.json 洗掉 (SKYNET 個股籌碼成本會 404)
        for name in ("watchlist", "global"):
            if not (DATA / f"{name}.json").exists():
                try:
                    r = requests.get(gov8.PAGES_URL.rstrip("/") + f"/data/{name}.json", timeout=20)
                    if r.ok and r.text.strip().startswith("{"):
                        (DATA / f"{name}.json").write_text(r.text, encoding="utf-8")
                        print(f"  carried over {name}.json from Pages")
                except Exception as e:  # noqa: BLE001
                    print(f"  carry {name} failed:", e)
    # 八大行庫監測：全市場序列 (累積) + 排行 (連續上榜) + 追蹤清單各行庫張數；fast 模式追蹤清單沿用上次發布
    try:
        dump("gov8", gov8.build(scored, res))
    except Exception as e:  # noqa: BLE001
        print("  gov8 failed:", e)
    dump("status", {"ready": True, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "fast" if args.fast else "full", "seconds": round(time.time() - t0)})
    print(f"done in {time.time() - t0:.0f}s → {SITE}")


if __name__ == "__main__":
    main()
