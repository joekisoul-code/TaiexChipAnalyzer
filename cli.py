"""命令列介面。

  python cli.py market            大盤籌碼判讀
  python cli.py market --no-wantgoo
  python cli.py stock 2330        個股籌碼判讀 (含大盤環境)
  python cli.py backtest          綜合分 vs 未來報酬驗證
  python cli.py watch [秒數]      盤中監看：即時指數/台指期/量能/廣度/盤勢分 + 警示推播 (預設 60 秒)
  python cli.py rt                印一次即時快照
  python cli.py optimize          長歷史 (2010~) 回測 + 產生調整後權重 (data/tuned_weights.json)
  python cli.py global            國際市場 (美日韓股、VIX、原油、黃金、比特幣、匯率) × 台股歷史研究
  python cli.py chips [代碼...]    追蹤清單籌碼分布 (成本、分價量、大戶、券商均價)
  python cli.py signals           大盤買點/賣點規則驗證與目前狀態
  python cli.py train [--stock]   訓練 LightGBM 走勢預測模型 (大盤；--stock 加個股) 並輸出樣本外指標
  python cli.py forecast [2330]   即時預測：未來 5/10/20 日上漲機率、期望報酬區間、驅動因子 (+個股)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from chip.analysis import market, stock  # noqa: E402
from chip.report import market_report, stock_report  # noqa: E402
from chip.sources import twse  # noqa: E402


def cmd_market(args):
    scored, a, meta = market.run(use_wantgoo=not args.no_wantgoo)
    print(market_report(a))
    try:
        q = twse.index_realtime()
        hint = market.intraday_hint(scored, q[0] if q else None)
        if hint:
            print(f"\n── 盤中提示 ({hint['time']}) 現價 {hint['last']:,.2f} ({hint['ret']:+.2f}%) MA5 {hint['ma5']:,.0f} MA20 {hint['ma20']:,.0f} ──")
            for m in hint["messages"]:
                print("  " + m)
    except Exception as e:  # noqa: BLE001
        print("盤中提示失敗:", e)
    print("\n── 資料來源 ──")
    for k, v in meta.items():
        if not k.startswith("_"):
            print(f"  {k:<24} {v}")


def cmd_stock(args):
    _, ma, _ = market.run(use_wantgoo=not args.no_wantgoo)
    a = stock.assess(args.stock_id, ma)
    print(stock_report(a))
    print(f"\n(大盤：{ma['regime']} {ma['composite']:+.1f}，{ma['action']})")


def cmd_backtest(args):
    scored, a, _ = market.run(use_wantgoo=not args.no_wantgoo)
    ev = market.evaluate(scored)
    print("Spearman 相關 (平滑綜合分 vs 未來報酬)：", ev.get("corr"))
    print("全體基準：", ev.get("baseline"))
    for h, tbl in ev.get("buckets", {}).items():
        print(f"\n未來 {h} 日報酬 by 分區：")
        print(tbl.to_string())
    sw = market.suggest_weights(scored)
    if sw:
        print(f"\n── 資料驅動權重參考 (未來 {sw['horizon']} 日，訓練 {sw['n_train']} / 測試 {sw['n_test']} 日) ──")
        print(f"樣本外排序相關：預設權重 {sw['oos_corr_default']:+.3f}｜擬合權重 {sw['oos_corr_fitted']:+.3f}")
        print(sw["coef"].to_string(index=False))


def cmd_watch(args):
    """盤中監看：每 interval 秒抓即時快照、計算盤勢分、檢查警示並推播、寫入 SQLite。籌碼資料每 30 分鐘重算。"""
    from chip import notify, realtime
    scored, a, _ = market.run(use_wantgoo=not args.no_wantgoo)
    last_chip = time.time()
    fired: set[str] = set()
    fired_day = None
    prev = None
    print(f"開始監看 (每 {args.interval} 秒)。籌碼分 {a['composite_smooth']:+.1f}【{a['regime']}｜{a['state']}】{a['action']}")
    while True:
        try:
            if time.time() - last_chip > 1800:
                scored, a, _ = market.run(use_wantgoo=not args.no_wantgoo)
                last_chip = time.time()
            s = realtime.snapshot(scored)
            if fired_day != s["ts"][:10]:
                fired, fired_day = set(), s["ts"][:10]
            idx, tx, b, sc = s.get("taiex") or {}, s.get("tx") or {}, s.get("breadth") or {}, s["score"]
            line = (f"{s['ts'][11:19]} [{s['phase']}] 加權 {idx.get('last', 0):,.2f} ({idx.get('chg_pct', 0) or 0:+.2f}%)"
                    f" | 台指期 {tx.get('last', 0) or 0:,.0f} 價差 {tx.get('basis', 0) or 0:+.0f}"
                    f" | 量能 {s.get('vol_pace', 0) or 0:.2f}x | 廣度 {b.get('up', 0)}↑{b.get('down', 0)}↓ 委比 {b.get('bid_ask_ratio', 0) or 0:.2f}"
                    f" | 盤勢分 {sc['score']:+.1f}【{sc['label']}】 籌碼分 {a['composite_smooth']:+.1f}【{a['regime']}】")
            print(line)
            txn = s.get("tx_night")
            if txn:
                print(f"   🌙 夜盤台指期 {txn['last']:,.0f} ({txn['change_pct']:+.2f}%)")
            for ev in realtime.check_alerts(s, prev, fired):
                notify.notify(ev["message"])
            realtime.persist(s)
            prev = s
        except Exception as e:  # noqa: BLE001
            print("error:", e)
        if args.once:
            break
        time.sleep(args.interval)


def cmd_optimize(args):
    """長歷史 (2010~) 回測：因子 IC、事件研究、逐年樣本外、寫入 data/tuned_weights.json 與 data/backtest_report.json"""
    import pandas as pd
    from chip.analysis import backtest
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    r = backtest.run(write=not args.dry_run)
    rep = r["report"]
    print(f"═══ 長歷史回測 {rep['start']} ~ {rep['end']} ({rep['rows']} 日，未來 {rep['horizon']} 日報酬) ═══")
    print("\n── 因子 IC (排序相關) ──")
    print(r["ic"]["overall"].to_string())
    print("\n── 依市場狀態 ──")
    print(r["ic"]["by_state"].to_string())
    print("\n── 事件研究 (10 日超額報酬 vs 全體基準，|t|≥2 較可信) ──")
    ev = r["events"].copy()
    print(ev[["訊號", "樣本數", "10日均報酬%", "10日勝率%", "20日均報酬%", "20日勝率%", "10日超額%", "t值(10日)"]].to_string(index=False))
    print("\n── 逐年樣本外 (IC) ──")
    print(r["walk_forward"].to_string(index=False))
    print("\n── 權重倍數建議 ──")
    for k, v in r["weights"].items():
        print(f"  {market.NAMES[k]:<22} ×{v['multiplier']:<5} {v['reason']}")
    print("\n── 融資背離長期分位數 ──", r["thresholds"]["margin_div20_quantiles"])
    if not args.dry_run:
        print(f"\n已寫入 {backtest.TUNED_PATH} 與 {backtest.REPORT_PATH}；模型會自動套用倍數 (CHIP_USE_TUNED=0 可停用)。")


def _print_metrics(h, m):
    if not m:
        print(f"  {h} 日：無指標")
        return
    print(f"  {h} 日：n={m['n']}  rank-IC {m['rank_ic']:+.3f} (逐年平均 {m['ic_year_mean']:+.3f}，正 IC 年份 {m['ic_positive_years']})  "
          f"基準上漲率 {m['base_hit']:.1%} 平均 {m['base_mean']:+.2f}%")
    print(f"        預測五分位 (弱→強) 實際上漲率 {[round(x * 100, 1) if x is not None else None for x in m['bin_hit']]}  平均報酬 {m['bin_mean']}  頂底差 {m['spread_top_bottom']:+.2f}%")
    print("        逐年 IC：" + " ".join(f"{r['year']}:{r['ic']:+.2f}" for r in m["by_year"]))


def cmd_train(args):
    from chip.predict import intraday, market_forecast, stock_forecast
    only = args.stock_only or args.intraday_only
    if not only:
        print("═══ 訓練大盤模型 (2010~，逐年 walk-forward 自 2014，淺層 LightGBM ×5 種子；視野 1/2/3/5/10/20 日) ═══")
        for h, m in market_forecast.train().items():
            _print_metrics(h, m)
    if args.intraday or args.intraday_only or not only:
        print("\n═══ 訓練小時模型 (Yahoo ^TWII 小時 K 近 2 年 + 前日籌碼 + 夜盤；逐月 walk-forward) ═══")
        for tm, m in intraday.train().items():
            if m:
                print(f"  到 {tm}：n={m['n']}  rank-IC {m['rank_ic']:+.3f} (逐月平均 {m['ic_year_mean']:+.3f}，正 IC 月份 {m['ic_positive_years']})  "
                      f"基準上漲率 {m['base_hit']:.0%}  五分位上漲率 {[round(x * 100) if x is not None else None for x in m['bin_hit']]}  平均 {m['bin_mean']}")
                print(f"        各起點方向命中率%：{m.get('hit_by_mark')}  基準：{m.get('base_by_mark')}  IC：{m.get('ic_by_mark')}")
            else:
                print(f"  到 {tm}：資料不足")
    if args.stock or args.stock_only:
        print("\n═══ 訓練個股模型 (權值股 40 檔 2018~，目標=相對大盤超額報酬，walk-forward 自 2021) ═══")
        for h, m in stock_forecast.train().items():
            _print_metrics(h, m)


def cmd_forecast(args):
    from chip import realtime
    from chip.predict import market_forecast, stock_forecast
    scored, a, _ = market.run(use_wantgoo=not args.no_wantgoo)
    snap = realtime.snapshot(scored)
    fc = market_forecast.forecast(scored, snap)
    if "error" in fc:
        print(fc["error"])
        return
    from chip.predict import intraday
    print(f"═══ 大盤走勢預測  資料日 {fc['date']}  收盤 {fc['close']:,.2f}  (模型訓練至 {fc['train_end']}) ═══")
    print(fc["summary"])
    print("\n── 隔天 / 後天 / 第三天 (跳過休市) ──")
    for nd in fc.get("next_days", []):
        print(f"  {nd['label']} {nd['date']}：上漲率 {nd['p_up']:.0%} (基準 {nd['base_hit']:.0%})｜預估收盤 {nd['level']:,.0f} (20~80% {nd['level_lo']:,.0f} ~ {nd['level_hi']:,.0f})"
              f"｜模型期望 {nd['pred']:+.2f}% 第 {nd['bin'] + 1}/5 分位")
    ih = intraday.forecast(scored, snap)
    print("\n── 當日每小時 ──")
    if "error" in ih:
        print("  " + ih["error"])
    else:
        print(f"  預測日 {ih['day']}｜{'盤中' if ih['live'] else '開盤前 (以前收為基準)'}｜目前時間點 {ih['mark']}｜基準價 {ih['price']:,.2f}"
              + (f"｜前晚夜盤 {ih['night_chg_pct']:+.2f}%" if ih.get("night_chg_pct") is not None else ""))
        for tm, r in ih["targets"].items():
            m = r.get("metrics", {})
            print(f"  到 {tm}：上漲率 {r.get('p_up', 0) or 0:.0%} (基準 {r.get('base_hit', 0) or 0:.0%})｜預估 {r['level']:,.0f} ({r['level_lo']:,.0f}~{r['level_hi']:,.0f})"
                  f"｜模型 {r['pred']:+.2f}%｜樣本外 IC {m.get('rank_ic', 'n/a')}")
            print("      推升：" + "；".join(f"{d['name']}→{d['contrib']:+.2f}" for d in r["drivers"]["positive"][:3]) +
                  "｜壓抑：" + "；".join(f"{d['name']}→{d['contrib']:+.2f}" for d in r["drivers"]["negative"][:3]))

    def _rows(hz):
        for h, r in hz.items():
            print(f"\n  未來 {h} 日：模型期望 {r['pred']:+.2f}% → 歷史第 {r['bin'] + 1}/5 分位｜同分位實際上漲率 {r['p_up']:.0%} (基準 {r['base_hit']:.0%})"
                  f"｜平均 {r['hist_mean']:+.2f}%｜區間 {r['q20']:+.2f}% ~ {r['q80']:+.2f}% (n={r['n']})")
            print("    推升：" + "；".join(f"{d['name']}({d['value']:+.2f})→{d['contrib']:+.2f}" for d in r["drivers"]["positive"][:4] if d["value"] is not None))
            print("    壓抑：" + "；".join(f"{d['name']}({d['value']:+.2f})→{d['contrib']:+.2f}" for d in r["drivers"]["negative"][:4] if d["value"] is not None))
    print("\n── 5 / 10 / 20 日 ──")
    _rows({h: r for h, r in fc["horizons"].items() if h in (5, 10, 20)})
    if fc.get("intraday"):
        i = fc["intraday"]
        print(f"\n  ── 盤中以現價 {i['price']:,.0f} ({(i['chg_pct'] or 0):+.2f}%) 推估 ──")
        _rows(i["horizons"])
    print("\n  模型樣本外指標：")
    for h, m in fc["metrics"].items():
        _print_metrics(h, m)
    if args.stock_id:
        sf = stock_forecast.forecast(args.stock_id)
        print(f"\n═══ 個股 {args.stock_id} 相對大盤預測 ═══")
        if "error" in sf:
            print(sf["error"])
        else:
            print(sf["summary"])
            _rows(sf["horizons"])


def cmd_global(args):
    """國際市場 × 台股歷史研究 (2007~)：同日效應、預測力、事件研究、滾動相關；寫入 data/global_report.json"""
    import pandas as pd
    from chip.analysis import global_study
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    r = global_study.run()
    print(f"═══ 國際市場 × 台股 {r['report']['start']} ~ {r['report']['end']} ({r['report']['rows']} 日) ═══")
    print("\n── 同日效應：前晚/前日報酬 vs 台股當日 (rank corr) ──")
    print(r["same_day"].to_string(index=False))
    print("\n── 預測力 (rank-IC vs 未來報酬，依 10 日 |IC| 排序，前 15) ──")
    print(r["predictive"].head(15).to_string(index=False))
    print("\n── 事件研究 (5 日超額 vs 基準，|t|≥2 較可信) ──")
    ev = r["events"]
    print(ev[["事件", "樣本數", "跳空%", "當日%", "5日均報酬%", "5日勝率%", "20日均報酬%", "20日勝率%", "5日超額%", "t值(5日)"]].to_string(index=False))
    print("\n── 近 60 日滾動相關 ──")
    print(r["rolling"].to_string(index=False))


def cmd_cross(args):
    """外匯/利率/波動率/原物料 × 台股、美股、韓股、日股 (2007~)。"""
    import pandas as pd
    from chip.analysis import cross_market
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    r = cross_market.run()
    for line in r["summary"]:
        print(line)
    for target, res in r["results"].items():
        print(f"\n═══ {target}  {res['start']} ~ {res['end']} ═══")
        for g, tbl in res["groups"].items():
            print(f"\n── {g} (依 10 日 |IC| 排序，前 8) ──")
            print(tbl.head(8).to_string(index=False))
        print("\n── 事件研究 (|t|≥2 較可信) ──")
        ev = res["events"]
        print(ev[ev["t值"].abs() >= 1.5].to_string(index=False) if not ev.empty else "n/a")


def cmd_chips(args):
    """追蹤清單籌碼分布：各路資金成本、分價量、大戶持股、券商均價。"""
    import pandas as pd
    from chip.analysis import chips
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ids = args.ids or chips.WATCHLIST
    res = chips.assess_watchlist(ids, use_wantgoo=not args.no_wantgoo)
    for sid, a in res.items():
        print(f"\n═══ {sid} {a.get('name', '')} ═══")
        if "error" in a:
            print("  ", a["error"])
            continue
        print(f"現價 {a['price']:,.2f}｜資料日 {a['date']}｜{a['label']} (分 {a['score']:+.2f})")
        for n in a["notes"]:
            print("  -", n)
        if not a["costs"].empty:
            print(a["costs"].to_string(index=False))
        vp = a.get("profile60")
        if vp:
            print(f"  分價量 60 日：密集區 {vp['poc']}，價值區 {vp['va_lo']}~{vp['va_hi']}，現價上方 {vp['above_pct']}% / 下方 {vp['below_pct']}%")
        if not a["brokers"].empty:
            print("  券商均價 (前 8)：\n" + a["brokers"].head(8).to_string(index=False))


def cmd_signals(args):
    """大盤買點/賣點：規則長歷史驗證 + 目前狀態。"""
    import pandas as pd
    from chip.analysis import backtest, signals
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    r = signals.run(backtest.load_long("2010-01-01"))
    print(r["evaluation"].to_string(index=False))
    c = r["current"]
    print(f"\n目前 ({c['date']})：{c['label']}｜買點強度 {c['buy_strength']}｜賣點強度 {c['sell_strength']}")
    for s in c["buy_signals"]:
        print(f"  🟥 買點 {s['name']} {s['days']} 10日超額 {s['excess10']:+.2f}% 驗證 {s['valid']}")
    for s in c["sell_signals"]:
        print(f"  🟩 賣點 {s['name']} {s['days']} 10日超額 {s['excess10']:+.2f}% 驗證 {s['valid']}")


def cmd_rt(args):
    from chip import realtime
    scored, a, _ = market.run(use_wantgoo=not args.no_wantgoo)
    s = realtime.snapshot(scored)
    idx, tx, txn, b = s.get("taiex") or {}, s.get("tx") or {}, s.get("tx_night") or {}, s.get("breadth") or {}
    print(f"═══ 即時快照 {s['ts']} [{s['phase']}] ═══")
    print(f"加權 {idx.get('last', 0):,.2f} ({idx.get('chg_pct', 0) or 0:+.2f}%)  高 {idx.get('high', 0):,.0f} 低 {idx.get('low', 0):,.0f}  MA5 {s.get('ma5', 0) or 0:,.0f} MA20 {s.get('ma20', 0) or 0:,.0f} 乖離 {s.get('bias20', 0) or 0:+.2f}%")
    print(f"台指期 {tx.get('symbol', '')} {tx.get('last', 0) or 0:,.0f} ({tx.get('change_pct', 0) or 0:+.2f}%) 期現價差 {tx.get('basis', 0) or 0:+.0f} 點 ({tx.get('basis_pct', 0) or 0:+.2f}%) OI {tx.get('oi', 0) or 0:,.0f}")
    if txn:
        print(f"夜盤 {txn['last']:,.0f} ({txn['change_pct']:+.2f}%) 較日盤結算 {txn['change']:+.0f}")
    print(f"成交 {s.get('amount_so_far', 0) or 0:,.0f} 億，推估全日 {s.get('amount_projected', 0) or 0:,.0f} 億 = 20 日均 {s.get('vol_pace', 0) or 0:.2f}x")
    if b:
        print(f"權值股 {b['n']} 檔：漲 {b['up']} 跌 {b['down']} 均 {b['avg_chg']:+.2f}%  委買/委賣 {b['bid_ask_ratio']:.2f}  台積電 {s.get('tsmc_chg', 0) or 0:+.2f}%")
    if s.get("global"):
        print("國際盤：" + "  ".join(f"{q['name']} {q['chg_pct']:+.2f}%" for q in s["global"] if q.get("chg_pct") is not None))
    sc = s["score"]
    print(f"盤勢即時分 {sc['score']:+.1f}【{sc['label']}】")
    for p in sc["parts"]:
        print(f"  {p['name']:<8} {p['score']:+.2f}  {p['text']}")
    print("籌碼×盤勢：", realtime.combined_view(a["composite_smooth"], a["regime"], sc["score"], sc["label"], s["phase"]))


def main():
    p = argparse.ArgumentParser(description="台股大盤籌碼分析")
    p.add_argument("--no-wantgoo", action="store_true", help="不使用玩股網 (Playwright)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("market").set_defaults(fn=cmd_market)
    s = sub.add_parser("stock")
    s.add_argument("stock_id")
    s.set_defaults(fn=cmd_stock)
    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    w = sub.add_parser("watch")
    w.add_argument("interval", type=int, nargs="?", default=60)
    w.add_argument("--once", action="store_true", help="只跑一輪 (測試用)")
    w.set_defaults(fn=cmd_watch)
    sub.add_parser("rt").set_defaults(fn=cmd_rt)
    t = sub.add_parser("train")
    t.add_argument("--stock", action="store_true", help="同時訓練個股模型 (約 120 次 FinMind 請求)")
    t.add_argument("--stock-only", action="store_true")
    t.add_argument("--intraday", action="store_true", help="同時訓練小時模型")
    t.add_argument("--intraday-only", action="store_true")
    t.add_argument("--days", type=int, default=200, help="小時模型使用的交易日數")
    t.set_defaults(fn=cmd_train)
    f = sub.add_parser("forecast")
    f.add_argument("stock_id", nargs="?")
    f.set_defaults(fn=cmd_forecast)
    sub.add_parser("global").set_defaults(fn=cmd_global)
    sub.add_parser("cross").set_defaults(fn=cmd_cross)
    c = sub.add_parser("chips")
    c.add_argument("ids", nargs="*", help="股票代碼，預設追蹤清單 2330 00631L 00685L 00981A 00988A")
    c.set_defaults(fn=cmd_chips)
    sub.add_parser("signals").set_defaults(fn=cmd_signals)
    o = sub.add_parser("optimize")
    o.add_argument("--dry-run", action="store_true", help="只顯示結果，不寫入 tuned_weights.json")
    o.set_defaults(fn=cmd_optimize)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
