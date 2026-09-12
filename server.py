"""平板 / 手機 App 版後端：FastAPI + PWA 靜態頁 (mobile/)。

    python server.py            # http://<本機IP>:8600  (平板同一 Wi-Fi 開啟後「加入主畫面」)

背景執行緒每 5 分鐘重算籌碼判讀/預測/訊號，每 15 分鐘重算追蹤清單；即時快照每次請求 (內部 5 秒快取)。
"""
from __future__ import annotations

import dataclasses
import logging
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chip import config, notify, realtime
from chip.analysis import backtest, chips, cross_market, global_study, market, signals
from chip.predict import intraday, market_forecast

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("server")
ROOT = Path(__file__).resolve().parent
MOBILE = ROOT / "mobile"
PORT = int(os.getenv("CHIP_PORT", "8600"))
USE_WANTGOO = os.getenv("CHIP_WANTGOO", "1") == "1"

app = FastAPI(title="台股籌碼分析 App API")
STATE: dict = {"ready": False, "error": None, "updated": None}
LOCK = threading.Lock()


# ------------------------------------------------------------------ 序列化
from chip.serialize import clean  # noqa: E402


# ------------------------------------------------------------------ 背景重算
def refresh_slow():
    scored, A, meta = market.run(use_wantgoo=USE_WANTGOO)
    snap = realtime.snapshot(scored)
    out = {"assessment": A, "meta": {k: v for k, v in meta.items() if not k.startswith("_")},
           "scored_tail": scored.tail(250)[["date", "open", "high", "low", "close", "foreign", "trust", "dealer", "margin_amt", "fut_foreign_net_oi",
                                            "gov8_net", "composite", "composite_smooth", "ma20", "ma60"]]}
    try:
        out["forecast"] = market_forecast.forecast(scored, snap)
    except Exception as e:  # noqa: BLE001
        out["forecast"] = {"error": str(e)}
    try:
        out["hourly"] = intraday.forecast(scored, snap)
    except Exception as e:  # noqa: BLE001
        out["hourly"] = {"error": str(e)}
    try:
        sg = signals.run(backtest.load_long("2010-01-01"), write=False)
        out["signals"] = {"current": sg["current"], "points": sg["points"].tail(120), "evaluation": sg["evaluation"]}
    except Exception as e:  # noqa: BLE001
        out["signals"] = {"error": str(e)}
    out["global_report"] = global_study.load_report()
    out["cross_report"] = cross_market.load_report()
    with LOCK:
        STATE.update(out)
        STATE["scored"] = scored
        STATE["ready"] = True
        STATE["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")


def refresh_watchlist():
    ids = [s.strip() for s in os.getenv("CHIP_WATCHLIST", ",".join(chips.WATCHLIST)).split(",") if s.strip()]
    res = chips.assess_watchlist(ids, use_wantgoo=USE_WANTGOO)
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
        slim[sid]["flows_tail"] = a["flows"].tail(60)[[c for c in ("date", "close", "foreign", "trust", "main", "gov8", "margin_lots") if c in a["flows"]]]
        slim[sid]["concentration"] = a.get("concentration")
        slim[sid]["brokers"] = a["brokers"].head(10)
    with LOCK:
        STATE["watchlist"] = slim
        STATE["watchlist_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")


def loop():
    last_slow = last_watch = 0.0
    fired: set[str] = set()
    fired_day = None
    prev = None
    while True:
        try:
            if time.time() - last_slow > 300:
                refresh_slow()
                last_slow = time.time()
            if time.time() - last_watch > 900:
                refresh_watchlist()
                last_watch = time.time()
            # 警示
            sc = STATE.get("scored")
            if sc is not None:
                s = realtime.snapshot(sc)
                if fired_day != s["ts"][:10]:
                    fired, fired_day = set(), s["ts"][:10]
                for ev in realtime.check_alerts(s, prev, fired):
                    notify.notify(ev["message"])
                realtime.persist(s)
                prev = s
        except Exception as e:  # noqa: BLE001
            log.warning("background loop: %s", e)
            STATE["error"] = str(e)
        time.sleep(20)


# ------------------------------------------------------------------ API
@app.get("/api/status")
def api_status():
    return {"ready": STATE["ready"], "updated": STATE.get("updated"), "watchlist_updated": STATE.get("watchlist_updated"), "error": STATE.get("error")}


@app.get("/api/realtime")
def api_realtime():
    sc = STATE.get("scored")
    s = realtime.snapshot(sc)
    A = STATE.get("assessment") or {}
    s["combined"] = realtime.combined_view(A.get("composite_smooth", 0), A.get("regime", ""), (s.get("score") or {}).get("score", 0),
                                          (s.get("score") or {}).get("label", ""), s["phase"]) if A else ""
    s["alerts_today"] = notify.read_log(s["ts"][:10], 20)
    s.pop("large_caps_full", None)
    return JSONResponse(clean(s))


@app.get("/api/market")
def api_market():
    if not STATE["ready"]:
        return JSONResponse({"ready": False, "message": "資料計算中，約 1~2 分鐘"}, status_code=202)
    with LOCK:
        A = STATE["assessment"]
        return JSONResponse(clean({"updated": STATE["updated"], "assessment": A, "signals": STATE.get("signals"), "meta": STATE.get("meta"),
                                   "history": STATE.get("scored_tail")}))


@app.get("/api/forecast")
def api_forecast():
    if not STATE["ready"]:
        return JSONResponse({"ready": False}, status_code=202)
    with LOCK:
        return JSONResponse(clean({"updated": STATE["updated"], "forecast": STATE.get("forecast"), "hourly": STATE.get("hourly")}))


@app.get("/api/watchlist")
def api_watchlist():
    if "watchlist" not in STATE:
        return JSONResponse({"ready": False, "message": "追蹤清單計算中"}, status_code=202)
    with LOCK:
        return JSONResponse(clean({"updated": STATE.get("watchlist_updated"), "stocks": STATE["watchlist"]}))


@app.get("/api/global")
def api_global():
    with LOCK:
        return JSONResponse(clean({"global": STATE.get("global_report"), "cross": STATE.get("cross_report")}))


@app.get("/manifest.json")
def manifest():
    return FileResponse(MOBILE / "manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def sw():
    return FileResponse(MOBILE / "sw.js", media_type="application/javascript")


@app.get("/")
def index():
    return FileResponse(MOBILE / "index.html")


app.mount("/static", StaticFiles(directory=str(MOBILE)), name="static")


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    url = f"http://{lan_ip()}:{PORT}/"
    print(f"\n平板/手機請用同一個 Wi-Fi 開啟：{url}\n(Safari：分享 → 加入主畫面；Chrome：選單 → 安裝應用程式)\n")
    try:
        import qrcode
        qrcode.QRCode(border=1).add_data(url)
        q = qrcode.QRCode(border=1)
        q.add_data(url)
        q.make()
        q.print_ascii(invert=True)
    except Exception:  # noqa: BLE001
        pass
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
