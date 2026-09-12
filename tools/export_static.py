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

from chip import config, notify, realtime  # noqa: E402
from chip.analysis import backtest, chips, cross_market, global_study, market, signals  # noqa: E402
from chip.predict import intraday, market_forecast  # noqa: E402
from server import clean  # noqa: E402

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
    dump("forecast", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "forecast": fc, "hourly": hr})
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
        dump("watchlist", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stocks": slim})
        dump("global", {"global": global_study.load_report(), "cross": cross_market.load_report()})
    dump("status", {"ready": True, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "fast" if args.fast else "full", "seconds": round(time.time() - t0)})
    print(f"done in {time.time() - t0:.0f}s → {SITE}")


if __name__ == "__main__":
    main()
