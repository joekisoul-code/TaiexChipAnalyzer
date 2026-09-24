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
            from chip.predict import confidence, patterns as _pt
            _pat = _pt.build(short_term.build_matrix(backtest.load_long("2010-01-01"), None), write=True)
            confidence.train(short_term.build_matrix(backtest.load_long("2010-01-01"), short_term._night_hist()), _pat, write=True, verbose=True)
            from chip.predict import verdict as _vd   # 判斷總結：共識票數 → OOS 命中率 (2026-09-23)
            _vd.train(short_term.build_matrix(backtest.load_long("2010-01-01"), short_term._night_hist()), _pat, write=True, verbose=True)
            from chip.predict import logic as _lg   # 漲跌邏輯：決策規則走動式驗證 (2026-09-23)
            _lg.train(short_term.build_matrix(backtest.load_long("2010-01-01"), short_term._night_hist()), write=True, verbose=True)
            from chip.predict import pullback as _pb   # 回落進場點：條件式分位模型 + 支撐止跌率 + 最低點時段 (2026-09-23)
            _pb.train(backtest.load_long("2010-01-01"), write=True, verbose=True)
            from chip.predict import stock_pullback as _sp   # 個股回落模型 (pooled) + 止跌機率 + 前端查表 (2026-09-23)
            _sp.train(write=True, verbose=True)
            from chip.predict import precheck as _pc   # 預判邏輯：開盤跳空條件統計 + 日曆效應驗證 (2026-09-23)
            _pc.train(backtest.load_long("2010-01-01"), write=True, verbose=True)
            from chip.predict import infomap as _im   # 預測邏輯總表：所有資訊 × 視野 vs 歷史漲跌 (2026-09-24)
            _im.train(backtest.load_long("2010-01-01"), short_term._night_hist(), write=True, verbose=True)
            from chip.predict import five_day as _fd   # 後五日方向模組：驗證訊號投票 + 五日模型 (2026-09-24)
            _fd.train(backtest.load_long("2010-01-01"), short_term._night_hist(), write=True, verbose=True)
            try:   # 挖寶雷達模型 (2026-09-24)：~230 檔個股 FinMind 日 K，走動式驗證後輸出前端樹模型
                from chip.predict import treasure as _tr
                _tr.train(None, write=True, verbose=True)
            except Exception as e:  # noqa: BLE001
                print("  treasure train failed:", e)
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
    # 歷史漲跌規律 (2026-09-22)：66 條規律逐年驗證 + 相似走勢；今日符合的規律加註到 forecast
    try:
        from chip.predict import patterns, short_term as _st
        pm = _st.build_matrix(backtest.load_long("2010-01-01"), None)
        pat = patterns.build(pm, write=True)
        fc["patterns"] = patterns.for_forecast(pat)
        from chip.predict import crossmkt
        fc["deep"] = crossmkt.build(pm)   # 跨市場關係 × 主力籌碼：今日讀數 + 有效條件 + 領先落後表
        print(f"  patterns: {pat['n_valid']}/{pat['n_rules']} valid, today {len([a for a in pat['today'] if a['valid']])} valid active")
    except Exception as e:  # noqa: BLE001
        print("  patterns failed:", e)
    # 線上自學 (2026-09-22)：先拉回已發布的預測帳本，用今天的資料對帳 → 近期命中率/校準/買賣點乘數，再套到本次預測
    from chip.predict import learn, stock_forecast
    from chip.sources import finmind
    learn_prev = learn.published()
    learn_summary = {"market": learn_prev.get("market") or {}, "stocks": learn_prev.get("stocks") or {}}
    try:   # 近五日精修：夜盤跳空 β、隔天用小時模型、5 日規則覆蓋 (+ 自學的買賣點水準乘數)
        fc = market_forecast.refine_short_term(fc, hr if not hr.get("error") else None, snap, sg if "error" not in sg else None, scored, learn_summary)
    except Exception as e:  # noqa: BLE001
        print("  refine_short_term failed:", e)
    try:   # 回落進場點 (2026-09-23)：模型 (走動式驗證優於 sigma 乘數的視野) 取代 buy_at/stop，原值保留為 buy_at_sigma/stop_sigma；支撐清單與最低點時段給前端
        from chip.predict import pullback
        _bp = (fc.get("intraday") or {}).get("price") or fc.get("close")
        try:
            from chip.predict import range_levels as _RL
            _nv, _ = _RL._night_final(snap, fc.get("next_days") or [], scored)
        except Exception:  # noqa: BLE001
            _nv = None
        pb = pullback.build(scored, _bp, night_ret=_nv)
        if pb:
            for x in fc.get("next_days") or []:
                r = (pb.get("k") or {}).get(str(x.get("n")))
                _mode = (x.get("range_mode") or "")
                _ok = (r or {}).get("use_model") and x.get("buy_at") and ((_mode.startswith("base") and r.get("variant") == "base") or (_mode.startswith("night") and r.get("variant") == "night"))   # 變體要與 range_levels 模式一致
                if _ok:
                    x["buy_at_sigma"], x["stop_sigma"] = x["buy_at"], x.get("stop")
                    x["buy_at"], x["stop"], x["buy_src"] = r["buy_model"], min(r["stop_model"], r["buy_model"]), "model"
                    if x.get("level_lo") == x["buy_at_sigma"]:
                        x["level_lo"] = x["buy_at"]
            fc["pullback"] = pb
    except Exception as e:  # noqa: BLE001
        print("  pullback failed:", e)
    try:   # 預判邏輯 (2026-09-23)：明日開盤跳空預判 (實際開盤 / 夜盤 β) + 目標日日曆效應
        from chip.predict import precheck
        fc["precheck"] = precheck.build(scored, snap, fc)
    except Exception as e:  # noqa: BLE001
        print("  precheck failed:", e)
    # 追蹤清單個股 ML 預測 (5/10/20 日相對大盤)：full 與 fast 都算 (每檔 <1 秒，走快取)，供 watchlist.json / 自學帳本
    stock_fc, stock_frames = {}, {}
    for sid in chips.WATCHLIST:
        try:
            sf = stock_forecast.forecast(sid, market=(fc.get("horizons") if isinstance(fc, dict) else None))
            if sf and not sf.get("error"):
                stock_fc[sid] = sf
                pf = finmind.stock_price(sid, "2024-01-01"); stock_frames[sid] = pf[[c for c in ("date", "close", "high", "low") if c in pf]]
        except Exception as e:  # noqa: BLE001
            print(f"  stock_forecast {sid} failed:", e)
    try:
        learn_out = learn.run(fc, snap, scored, stock_fc, stock_frames, prev=learn_prev, hourly=hr if not hr.get("error") else None)
        fc = learn.adjust_forecast(fc, learn_out)
        for sid, sf in stock_fc.items():   # 個股預測也帶近期命中
            st = (learn_out.get("stocks") or {}).get(sid) or {}
            for h, r in (sf.get("horizons") or {}).items():
                g = (st.get("by_h") or {}).get(str(h)) or {}
                r["recent_hit"], r["recent_n"] = g.get("hit_ewm"), g.get("n_calls")
                if (g.get("adjust") or {}).get("degrade"):
                    r["degraded"] = True
        dump("learn", learn_out)
        print(f"  learn: ledger {learn_out['n_ledger']} (+{learn_out['n_new']} new, {learn_out['n_evaluated_now']} evaluated now)")
    except Exception as e:  # noqa: BLE001
        print("  learn failed:", e)
    try:   # 漲跌邏輯 (2026-09-23)：歷史學出的決策規則 → 今日適用規則 + 最近 8 次實例 + 該變體走動式 OOS
        from chip.predict import logic, short_term as _st2
        _nd0 = (fc.get("next_days") or [{}])[0]
        fc["logic"] = logic.build(_st2.build_matrix(backtest.load_long("2010-01-01"), _st2._night_hist()), night_final=(_nd0.get("variant") == "night"))
    except Exception as e:  # noqa: BLE001
        print("  logic failed:", e)
    try:   # 後五日 (交易日) 方向 (2026-09-24)：只有多方可預測 → 訊號淨票 ≥2 給偏多 (走動式 65%)，空方停用；寫進 horizons[5]
        from chip.predict import five_day, short_term as _st4
        fv = five_day.build(backtest.load_long("2010-01-01"), _st4._night_hist())
        if fv:
            fc["five"] = fv
            r5 = (fc.get("horizons") or {}).get(5) or (fc.get("horizons") or {}).get("5")
            if r5 is not None:
                r5["call_five"], r5["five_net"], r5["five_note"] = fv["call"], fv["net"], fv["text"]
                if fv.get("by_net"):
                    r5["five_up"] = fv["by_net"].get("up")
                if fv["call"] == "偏多" and (fv.get("rule") or {}).get("hit"):
                    r5["five_hit"] = fv["rule"]["hit"]
    except Exception as e:  # noqa: BLE001
        print("  five_day failed:", e)
    try:   # 預測邏輯總表 (2026-09-24)：今日各有效資訊讀數所在檔位 → 各視野多空票
        from chip.predict import infomap, short_term as _st3
        fc["infomap"] = infomap.build(backtest.load_long("2010-01-01"), _st3._night_hist())
        fc["asia_dates"] = _st3.asia_last_dates()   # 亞股同日資料新鮮度 (2026-09-24：嚴格同日，缺就 NaN)
    except Exception as e:  # noqa: BLE001
        print("  infomap failed:", e)
    try:   # 判斷總結 (2026-09-23)：模型叫牌 + 6 票獨立訊號的共識 → 一句可執行判斷 (含同狀況 OOS 命中率)
        from chip.predict import verdict
        fc["verdict"] = verdict.build(fc, hr if not hr.get("error") else None, snap, scored, learn_summary)
    except Exception as e:  # noqa: BLE001
        print("  verdict failed:", e)
    dump("forecast", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "forecast": fc, "hourly": hr, "stocks": stock_fc})
    try:   # 挖寶雷達樹模型 (前端算分)；模型檔在 repo (每週 --train 重訓)
        from chip.predict import model as _M
        _tm = _M.load_json("treasure_model")
        if _tm:
            dump("treasure_model", _tm)
    except Exception as e:  # noqa: BLE001
        print("  treasure_model dump failed:", e)
    res = None
    if not args.fast:
        res = chips.assess_watchlist(chips.WATCHLIST, use_wantgoo=use_wg)
        slim = {}
        for sid, a in res.items():
            if "error" in a:
                slim[sid] = a
                continue
            slim[sid] = {k: a.get(k) for k in ("stock_id", "name", "price", "date", "notes", "score", "label", "holders", "broker_date", "quote")}
            slim[sid]["forecast"] = stock_fc.get(sid)   # 個股 ML 預測 (5/10/20 日相對大盤) + 近期命中
            slim[sid]["costs"] = a["costs"]
            p = a.get("profile60") or {}
            slim[sid]["profile"] = {k: p.get(k) for k in ("poc", "va_lo", "va_hi", "above_pct", "below_pct", "last")}
            slim[sid]["profile_bins"] = p.get("bins")
            slim[sid]["brokers"] = a["brokers"].head(10)
            fl = a.get("flows")
            if fl is not None and hasattr(fl, "tail"):   # 近 60 日各路資金 (主力/籌碼集中度 有玩股網時才有)
                slim[sid]["flows_tail"] = fl.tail(60)[[c for c in ("date", "close", "foreign", "trust", "dealer", "main", "skp5", "skp20", "gov8", "margin_chg") if c in fl]]
        # 個股回落模型 (2026-09-23)：追蹤清單每檔的模型買點/停損/止跌機率/支撐；離線查表給前端套用到任何股票
        _pb_tbl = None
        try:
            from chip.predict import stock_pullback as _sp2
            for sid in list(slim.keys()):
                try:
                    r = _sp2.build(sid)
                    if r:
                        slim[sid]["pullback"] = r
                except Exception as e:  # noqa: BLE001
                    print(f"  stock_pullback {sid} failed:", e)
            _pb_tbl = _sp2.client_table()
        except Exception as e:  # noqa: BLE001
            print("  stock_pullback failed:", e)
        dump("watchlist", {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stocks": slim, "pullback_table": _pb_tbl})
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
