"""margin_dyn 掛鉤測試 (SPEC §3.7 (k)；比照 tests/test_pr2_static.py 讀原始碼)：export_static 的 dump("margin" 在 try 內、fast 帶回名單含 margin、
publish.yml 含 MARGIN_BACKFILL_DAYS 與 margin_backfill 且排程 cron 未動；另測 write_seed (Tier-1 種子) 與 backfill 對「當日尚未公布」不寫 margin_nodata。
不連網、不需 LightGBM；SQLite 用暫存 CHIP_DATA_DIR。

    python tests/test_margin_hook.py
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="margin_hook_test_")
os.environ["CHIP_DATA_DIR"] = _TMP                     # 不碰真實 data/chip.sqlite

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip.analysis import margin_dyn as M  # noqa: E402

M.THROTTLE = 0
EXPORT = (ROOT / "tools" / "export_static.py").read_text(encoding="utf-8")
YML = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _block(src: str, start_pat: str, end_pat: str) -> str:
    i = src.index(start_pat)
    j = src.index(end_pat, i)
    return src[i:j]


def test_export_static_mount_in_try_after_pools_before_pit_ledger():
    assert 'dump("margin"' in EXPORT
    i_dump = EXPORT.index('dump("margin"')
    i_pools = EXPORT.index("_pools = _tlo.pop(")
    i_radar = EXPORT.index("radar_archive as _ra")
    i_pit = EXPORT.index("pit_ledger as _pit")
    assert i_pools < i_radar < i_dump < i_pit, "margin 區塊要在 treasure_live / radar_pool 之後、pit_ledger 之前 (需要 _pools)"
    blk = _block(EXPORT, "from chip.analysis import margin_dyn", "pit_ledger as _pit")
    # 整段在 try 內：區塊開頭前一行是 try:，且區塊內有對應的 except 印 margin failed (失敗不中斷發布)
    head = EXPORT[:EXPORT.index("from chip.analysis import margin_dyn")].rstrip().splitlines()[-1]
    assert head.strip() == "try:", head
    assert 'print("  margin failed:", e)' in blk
    assert "margin_dyn.published()" in blk and "margin_dyn.build(scored, _mg_res, pools=_pools, prev=_mg_prev, fast=args.fast)" in blk
    # watch_res 帶追蹤清單除息事件 (exdiv.load_all 的 {sid: events}) 給 FIFO 扣股利；fast 模式沒有 → 回落到 res
    assert '"exdiv": ev' in blk and "_mg_exd" in blk and "_mg_exd = None" in EXPORT and "_mg_exd = _exd" in EXPORT
    assert 'os.getenv("MARGIN_BACKFILL_DAYS", "40")' in blk and "if not args.fast:" in blk, "完整模式才 backfill，天數由 MARGIN_BACKFILL_DAYS 控制"
    assert "import os" in EXPORT.splitlines()[:30] or "\nimport os\n" in EXPORT
    # backfill 用獨立 try/except：回補失敗 (sqlite 鎖 / 非預期格式) 不影響 build 與帶回
    i_bf = blk.index("margin_dyn.backfill(")
    pre = blk[:i_bf].rstrip().splitlines()
    assert pre[-1].strip().startswith("print(") and pre[-2].strip().startswith("try:"), pre[-3:]
    assert 'print("  margin backfill failed:", e)' in blk and blk.index("margin backfill failed") < blk.index("margin_dyn.build(")
    # 建置失敗 → 帶回上次發布 (比照 desk _guard_degraded)，且 train 模式寫種子
    assert "carrying over last published" in blk and '"carried": True' in blk
    assert "margin_dyn.write_seed(_mg)" in blk and "if args.train:" in blk
    # fast 且 published() 兩來源皆失敗：先讀帶回名單剛復原的本機 DATA/margin.json 當 prev；仍為空 → 不 dump (不把帶回檔蓋成空骨架)
    i_loc = blk.index('if not _mg_prev and (DATA / "margin.json").exists():')
    assert blk.index('_mg_prev = json.loads((DATA / "margin.json").read_text', i_loc) < i_bf, "本機 prev 讀取在 backfill / build 之前"
    assert "if args.fast and not _mg_prev:" in blk and blk.index("if args.fast and not _mg_prev:") < blk.index('dump("margin"')
    i_guard = blk.index("if args.fast and not _mg_prev:")
    i_else = blk.index("else:", i_guard)
    assert 'dump("margin"' not in blk[i_guard:i_else] and 'dump("margin"' in blk[i_else:], "dump 只在 else 分支 (有 prev 或非 fast)"


def test_export_static_fast_carry_list_has_margin():
    m = re.search(r'for name in \(([^)]*)\):\s*\n\s*if not \(DATA / f"\{name\}\.json"\)\.exists\(\):', EXPORT)
    assert m, "fast 帶回名單迴圈不見了"
    names = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]
    assert {"watchlist", "global", "desk", "desk_archive", "margin"} <= set(names), names


def test_publish_yml_env_mode_and_cron_unchanged():
    assert re.search(r'^\s+MARGIN_BACKFILL_DAYS:\s*"40"', YML, re.M), "env MARGIN_BACKFILL_DAYS"
    assert re.search(r'^\s+MARGIN_PANEL_FROM:\s*\$\{\{ vars\.MARGIN_PANEL_FROM \}\}', YML, re.M), "env MARGIN_PANEL_FROM (repo var，空 = 預設)"
    src = (ROOT / "chip" / "analysis" / "margin_dyn.py").read_text(encoding="utf-8")
    assert re.search(r'^PANEL_FROM = os\.getenv\("MARGIN_PANEL_FROM"\) or "2024-10-01"', src, re.M), "空字串 env 要回預設 (vars 未設時 Actions 給空字串)"
    assert M.PANEL_FROM == "2024-10-01" and M.TPEX_FROM == "2007-01-01"
    assert "margin_backfill" in YML
    # workflow_dispatch 說明與 export case 都有 margin_backfill，case 先回補 450 日再照常 full export
    case = re.search(r'^\s+margin_backfill\)\s+(.*?);;\s*$', YML, re.M)
    assert case, "export step 缺 margin_backfill case"
    # 回補以時間預算 (max_seconds) 守門：到時回 pending 由下次 dispatch 續補，避免 job 90 分超時被取消時 Actions cache 不存檔、整批回補丟失
    m_bf = re.search(r"margin_dyn\.backfill\(450,\s*max_seconds=([0-9*]+)\)", case.group(1))
    assert m_bf and eval(m_bf.group(1)) <= 45 * 60 and "tools/export_static.py" in case.group(1) and "--fast" not in case.group(1), case.group(1)  # noqa: S307
    assert "max_seconds" in inspect.signature(M.backfill).parameters
    desc = re.search(r'description:\s*"(fast \| full \| train[^"]*)"', YML)
    assert desc and "margin_backfill" in desc.group(1)
    # train 模式 git add 種子 (檔案不存在時不炸)
    assert re.search(r'if \[ -f data/margin_market\.json \]; then git add data/margin_market\.json; fi', YML)
    # 排程 cron 一字未動
    crons = re.findall(r'- cron: "([^"]+)"', YML)
    assert crons == ["*/10 1-5 * * 1-5", "40 7 * * 1-5", "*/30 9-14 * * 1-5", "45 13 * * 1-5", "0 23 * * 0-4", "40 21 * * 0-4", "0 22 * * 6"], crons
    # decide mode 的 case 未加 margin_backfill 分支 (dispatch 直接給 MODE)，且 margin_backfill 不會被排除在 cache / deploy 之外 (只有 research* 被排除)
    assert "startsWith(steps.mode.outputs.mode, 'research')" in YML and "startsWith(steps.mode.outputs.mode, 'margin')" not in YML


def test_write_seed_roundtrip_and_guards():
    p = Path(_TMP) / "seed_test.json"
    rows = [[f"2024-{m:02d}-01", 2500.0 + m, 1_000_000 + m, 40_000, 165.5, 1000.0] for m in range(1, 13)]
    out = {"meta": {"generated": "2026-10-14 21:50:00", "carried_parts": []},
           "history": {"cols": list(M.HISTORY_COLS), "rows": rows},
           "flows": {"cols": list(M.FLOW_COLS), "rows": [["2024-12-01"] + [1.0] * (len(M.FLOW_COLS) - 1)]}}
    assert M.write_seed(out, p) == p
    sj = json.loads(p.read_text(encoding="utf-8"))
    assert sj["history"]["cols"] == M.HISTORY_COLS and len(sj["history"]["rows"]) == 12 and sj["flows"]["cols"] == M.FLOW_COLS and sj["meta"]["history_days"] == 12
    # market_history 的種子讀法能吃這份 (離線：所有 fetch 失敗、prev 空 → 只剩種子)
    fails = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))  # noqa: E731
    with patched(M, "SEED_PATH", p), patched(M, "_fetch_total_flows", fails), patched(M, "_fetch_total_margin", fails), patched(M, "_fetch_twse_ms", fails), \
            patched(M, "_fetch_taiex", fails):
        h = M.market_history({})
    assert len(h) == 12 and list(h["date"])[:2] == ["2024-01-01", "2024-02-01"] and float(h["margin_amt"].iloc[-1]) == 2512.0
    # 守門：沿用的 history 不覆寫；縮水 >2% 不覆寫；空的不寫
    out2 = {**out, "meta": {"carried_parts": ["history"]}}
    assert M.write_seed(out2, p) is None
    out3 = {**out, "history": {"cols": list(M.HISTORY_COLS), "rows": rows[:5]}}
    assert M.write_seed(out3, p) is None and len(json.loads(p.read_text(encoding="utf-8"))["history"]["rows"]) == 12
    assert M.write_seed({"history": {"cols": list(M.HISTORY_COLS), "rows": []}}, Path(_TMP) / "none.json") is None
    assert M.write_seed({}, Path(_TMP) / "none2.json") is None and not (Path(_TMP) / "none2.json").exists()


def test_backfill_today_nodata_is_not_marked():
    """當日與前一日 (NODATA_GRACE_DAYS=1，同 tpex；21:45 cron 延遲跨午夜時 TWSE 可能尚未公布) 的 NoData 不寫 margin_nodata；更早的 NoData 才寫。"""
    from chip import store
    with store.conn() as c:
        for t in ("margin_stock_daily", "margin_nodata", "daily_metrics"):
            if c.execute("SELECT name FROM sqlite_master WHERE name=?", (t,)).fetchone():
                c.execute(f"DELETE FROM {t}")
    calls = []

    def fake_all(ymd):
        calls.append(ymd)
        raise M.NoData("沒有符合")
    assert M.NODATA_GRACE_DAYS == 1
    with patched(M, "_fetch_margin_all", fake_all), patched(M, "_fetch_closes", lambda y: {}), patched(M, "_fetch_tpex", lambda y: None), \
            patched(M, "_fetch_holidays", lambda y: set()):
        rep = M.backfill(5, today="2026-10-14", panel_from="2026-10-12")
    nod = M._nodata("twse_all")
    assert "2026-10-14" not in nod and "2026-10-13" not in nod and "2026-10-12" in nod, nod
    assert rep["notyet"] == 2 and rep["nodata"] == 1 and rep["pending"] == 2, rep
    assert [M._iso(x) for x in calls] == ["2026-10-14", "2026-10-13", "2026-10-12"], "冷啟動由新往舊"
    # 週六 00:xx 跑 (cron 延遲跨午夜)：週五 (today−1) 的 NoData 不寫
    calls.clear()
    with patched(M, "_fetch_margin_all", fake_all), patched(M, "_fetch_closes", lambda y: {}), patched(M, "_fetch_tpex", lambda y: None), \
            patched(M, "_fetch_holidays", lambda y: set()):
        rep = M.backfill(5, today="2026-10-03", panel_from="2026-10-01")
    nod = M._nodata("twse_all")
    assert "2026-10-02" not in nod and "2026-10-01" in nod and rep["notyet"] == 1 and rep["nodata"] == 1, (nod, rep)


def test_backfill_order_closes_before_back_and_tpex_catchup():
    """順序：最近缺日 → 種子日補收盤 → 更早缺日 → 上櫃合計補齊 (種子匯入的日子 TPEx 從未抓過)；NoData 寫 margin_nodata('tpex')、NotYet 不寫。"""
    from chip import store
    with store.conn() as c:
        for t in ("margin_stock_daily", "margin_nodata", "daily_metrics"):
            if c.execute("SELECT name FROM sqlite_master WHERE name=?", (t,)).fetchone():
                c.execute(f"DELETE FROM {t}")
    import pandas as pd
    cal = pd.bdate_range("2026-09-01", "2026-10-13").strftime("%Y-%m-%d").tolist()
    row = [{"code": "2330", "name": "x", "m_buy": 1, "m_sell": 1, "m_repay": 0, "m_prev": 100, "m_today": 100, "m_limit": 1000, "s_buy": 0, "s_sell": 0, "s_repay": 0,
            "s_prev": 0, "s_today": 0, "s_limit": 10, "offset": 0, "note": ""}]
    for d in cal[5:25]:                                  # 種子：20 日、無收盤、無 TPEx
        M._panel_upsert(d, row, None)
    seq = []

    def fake_all(ymd):
        seq.append(("all", M._iso(ymd)))
        return {"date": M._iso(ymd), "market": None, "stocks": [["2330", "x", "1", "1", "0", "100", "100", "1000", "0", "0", "0", "0", "0", "10", "0", ""]]}

    def fake_close(ymd):
        seq.append(("close", M._iso(ymd))); return {"2330": (1000.0, 5.0)}

    def fake_tpex(ymd):
        d = M._iso(ymd); seq.append(("tpex", d))
        if d == cal[24]:
            raise M.NotYet("0 列")
        if d == cal[23]:
            raise M.NoData("休市")
        return {"date": d, "status": "ok", "market": {"margin_lots": 10.0, "margin_amt": 1000.0, "short_lots": 1.0}, "stocks": []}
    with patched(M, "_fetch_margin_all", fake_all), patched(M, "_fetch_closes", fake_close), patched(M, "_fetch_tpex", fake_tpex), patched(M, "_fetch_holidays", lambda y: set()):
        rep = M.backfill(100, today=cal[-1], panel_from=cal[0])
    kinds = [k for k, _ in seq]
    i_fwd_all = [i for i, (k, d) in enumerate(seq) if k == "all" and d > cal[24]]
    i_back_all = [i for i, (k, d) in enumerate(seq) if k == "all" and d < cal[5]]
    i_seed_close = [i for i, (k, d) in enumerate(seq) if k == "close" and cal[5] <= d <= cal[24]]
    i_seed_tpex = [i for i, (k, d) in enumerate(seq) if k == "tpex" and cal[5] <= d <= cal[24]]
    assert i_fwd_all and i_seed_close and i_back_all and i_seed_tpex, (rep, kinds)
    assert max(i_fwd_all) < min(i_seed_close) < min(i_back_all) < min(i_seed_tpex), "fwd → seed closes → back → tpex catch-up"
    assert rep["pending"] == 0 and rep["pending_closes"] == 0 and rep["pending_tpex"] == 1 and "tpex" not in rep.get("error", ""), rep   # NotYet 那天仍 pending
    assert cal[23] in M._nodata("tpex") and cal[24] not in M._nodata("tpex")
    got = store.load_metrics([M.METRIC_KEYS["tpex_margin_lots"]])
    have_t = set(got.loc[got[M.METRIC_KEYS["tpex_margin_lots"]].notna(), "date"].astype(str).str[:10])
    assert set(cal[5:23]) <= have_t and cal[23] not in have_t and cal[24] not in have_t
    # 可重入：第二次不再抓已有 TPEx / nodata 的日子，只重試 NotYet 那天
    seq.clear()
    with patched(M, "_fetch_margin_all", fake_all), patched(M, "_fetch_closes", fake_close), patched(M, "_fetch_tpex", fake_tpex), patched(M, "_fetch_holidays", lambda y: set()):
        rep2 = M.backfill(100, today=cal[-1], panel_from=cal[0])
    assert [d for k, d in seq if k == "tpex"] == [cal[24]] and not [1 for k, _ in seq if k != "tpex"], seq
    assert rep2["pending_tpex"] == 1


def test_sleep_not_doubled_when_sources_self_throttle():
    """資料層 twse 有 MIN_GAP/_throttle 時 margin_dyn._sleep 不再重複等 (否則每日多 4.4 s)。"""
    import time as _t
    with patched(M, "THROTTLE", 2.2):
        t0 = _t.time(); M._sleep(); dt_ = _t.time() - t0
    assert dt_ < 0.5, dt_
    import chip.sources.twse as tw
    assert float(tw.MIN_GAP) >= 2.2 and callable(tw._throttle)


if __name__ == "__main__":
    fails = 0
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            n += 1
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'} ({n} tests)")
    sys.exit(1 if fails else 0)
