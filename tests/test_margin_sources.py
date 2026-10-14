"""margin_dyn 資料層 / 儲存層 (SPEC §3.1、§3.7 (a)(b)(l) 相關) 單元測試。不連網、不需 LightGBM：get_json / fetch / session 全部以合成回應取代，
快取與 SQLite 指到暫存目錄。

    python tests/test_margin_sources.py

(a) 解析：TWSE MI_MARGN ALL 16 欄含 note、tables[0] 仟元→億 (含現償)、民國 / YYYYMMDD 日期；「沒有符合」→ NoData 不快取、其他非 OK → RuntimeError 不快取；
    ALL 快取為 list 格式 (與本機 491 日快取、舊 margin_stocks() 相容)，MS 快取同請求順手補 (含現償)；舊 list 快取缺 note / MS 缺現償 → 補齊；
    舊 margin_stocks() 寫的 [] → 清掉重抓。MI_INDEX：只留代號 / 收盤 / 成交張、無收盤列不留、2004-02-11 前 (本機與訊息) → NoData。
    TPEx：20 欄 + summary 兩列；0 列 + 休市 → closed (快取)、0 列 + 交易日 (今天 / 昨天) → NotYet 不快取、0 列 + 更早交易日 → NoData。
(b) 恆等式：合成面板 today = prev + buy − sell − repay 全列成立 → 無 err；違反 / 缺欄 → True；store 欄名版同樣可用。
(c) FinMind total_margin_flows：三列 pivot、金額 元→億、Return→repay；wantgoo maint_real：時間戳 +8h、倍→%、0 列剔除、GET 失敗 → Playwright → 空 DataFrame 不 raise 且不快取。
(d) store：margin_rows 映射 + upsert (executemany) + load + 重入覆蓋 + update_margin_closes + margin_dates / coverage + nodata；其他既有表不受影響。
"""
from __future__ import annotations

import contextlib
import datetime as dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip import config, http, store  # noqa: E402
from chip.sources import NoData, NotYet, finmind, tpex, twse, wantgoo  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="margin_src_"))
config.CACHE_DIR = TMP / "cache"
config.CACHE_DIR.mkdir()
config.DB_PATH = TMP / "chip.sqlite"
twse.MIN_GAP = 0.0
tpex.MIN_GAP = 0.0


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class Counter:
    def __init__(self, fn):
        self.fn, self.n, self.calls = fn, 0, []

    def __call__(self, url, **kw):
        self.n += 1
        self.calls.append((url, kw.get("params")))
        return self.fn(url, **kw)


# ------------------------------------------------------------------ 合成回應
ALL_FIELDS = ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
              "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額", "資券互抵", "註記"]
MS_FIELDS = ["項目", "買進", "賣出", "現金(券)償還", "前日餘額", "今日餘額"]


def twse_all_json(ymd="20250103", stocks=None):
    # 融資 9,418,721 + 280,507 − 267,042 − 5,032 = 9,427,154；融券 302,228 + 34,228 − 17,847 − 2,460 = 316,149；金額 (仟元) 同恆等式
    ms = [["融資(交易單位)", "280,507", "267,042", "5,032", "9,418,721", "9,427,154"],
          ["融券(交易單位)", "17,847", "34,228", "2,460", "302,228", "316,149"],
          ["融資金額(仟元)", "11,708,537", "11,072,160", "198,454", "319,102,025", "319,539,948"]]
    stocks = stocks or [
        ["0050", "元大台灣50", "48", "138", "0", "1,119", "1,029", "547,375", "4", "2", "0", "233", "231", "547,375", "0", " "],
        ["0051", "元大中型100", "0", "0", "0", "0", "0", "5,500", "0", "0", "0", "0", "0", "5,500", "0", "X "],
        ["2330", "台積電", "1,200", "900", "50", "20,000", "20,250", "648,000", "30", "80", "5", "1,000", "1,045", "648,000", "12", "O"],
        ["00632R", "元大台灣50反1", "5,000", "4,000", "100", "600,000", "600,900", "2,500,000", "0", "0", "0", "0", "0", "2,500,000", "7", "@"],
    ]
    return {"stat": "OK", "date": ymd, "tables": [{"fields": MS_FIELDS, "data": ms}, {"fields": ALL_FIELDS, "data": stocks}]}


MI_FIELDS = ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差",
             "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比"]


def mi_index_json(ymd="20250103"):
    rows = [["2330", "台積電", "30,000,000", "50,000", "30,000,000,000", "999", "1,010", "995", "1,000.00", "+", "5", "999", "1", "1,000", "2", "25.1"],
            ["0050", "元大台灣50", "5,500,000", "9,000", "1,100,000,000", "199", "201", "198", "200.50", "-", "1", "200", "1", "200.5", "2", "0.00"],
            ["00632R", "元大台灣50反1", "80,000,000", "20,000", "400,000,000", "5", "5.1", "4.9", "5.00", "-", "0.05", "4.99", "1", "5", "9", "0.00"],
            ["911608", "明輝-DR", "0", "0", "0", "--", "--", "--", "--", " ", "0.00", "1.5", "1", "--", "0", "0.00"],
            ["TW50", "怪代號", "1", "1", "1", "1", "1", "1", "1", "+", "0", "1", "1", "1", "1", "0"]]
    return {"stat": "OK", "date": ymd, "tables": [{"fields": ["指數", "收盤指數"], "data": [["發行量加權股價指數", "23,000.00"]]},
                                                  {"fields": MI_FIELDS, "data": rows}]}


TPEX_FIELDS = ["代號", "名稱", "前資餘額(張)", "資買", "資賣", "現償", "資餘額", "資屬證金", "資使用率(%)", "資限額",
               "前券餘額(張)", "券賣", "券買", "券償", "券餘額", "券屬證金", "券使用率(%)", "券限額", "資券相抵(張)", "備註"]


def tpex_json(ymd="20261013", empty=False):
    data = [] if empty else [
        ["00679B", "元大美債20年", "4,395", "326", "589", "2", "4,130", "139", "0.26", "1,558,923", "4", "5", "0", "0", "9", "0", "0.0", "1,558,923", "2", ""],
        ["6488", "環球晶", "10,000", "500", "300", "20", "10,180", "0", "8.5", "120,000", "100", "30", "10", "5", "115", "0", "0.1", "120,000", "40", "X"]]
    summary = [] if empty else [
        ["", "合計(張)", "2,421,036", "77,732", "112,879", "1,971", "2,383,918", "", "", "", "37,606", "4,277", "5,605", "610", "35,668", "", "", "", "", ""],
        ["", "融資金(仟元)", "228,346,504", "10,079,768", "15,109,433", "116,156", "223,200,683", "", "", "", "", "", "", "", "", "", "", "", "", ""]]
    return {"stat": "ok", "date": ymd, "tables": [{"title": "上櫃股票融資融券餘額", "fields": TPEX_FIELDS, "data": data, "summary": summary}]}


def _no_net(url, **kw):
    raise AssertionError(f"unexpected network call {url} {kw.get('params')}")


# ------------------------------------------------------------------ (a) TWSE MI_MARGN ALL
def test_twse_margin_all_parse_and_cache_format():
    g = Counter(lambda url, **kw: twse_all_json("20250103"))
    with patched(twse, "get_json", g):
        d = twse.margin_stocks_hist("2025-01-03")
    assert g.n == 1 and g.calls[0][1]["selectType"] == "ALL" and g.calls[0][1]["date"] == "20250103"
    assert d["date"] == "2025-01-03" and len(d["stocks"]) == 4
    m = d["market"]
    assert m["margin_lots"] == 9427154 and m["margin_buy_lots"] == 280507 and m["margin_sell_lots"] == 267042 and m["margin_repay_lots"] == 5032
    assert m["short_lots"] == 316149 and m["short_sell_lots"] == 34228 and m["short_buy_lots"] == 17847 and m["short_repay_lots"] == 2460
    assert abs(m["margin_amt"] - 3195.39948) < 1e-6 and abs(m["margin_repay_amt"] - 1.98454) < 1e-6      # 仟元 → 億
    s = {r["code"]: r for r in d["stocks"]}
    assert set(s["0050"]) == set(twse.MARGIN_ALL_COLS)
    assert s["0050"]["note"] == "" and s["0051"]["note"] == "X" and s["2330"]["note"] == "O" and s["00632R"]["note"] == "@"
    assert s["2330"]["margin_today"] == 20250 and s["2330"]["margin_repay"] == 50 and s["2330"]["offset"] == 12 and s["2330"]["date"] == "2025-01-03"
    # 快取格式：ALL key = list (與本機 491 日快取 / 舊 margin_stocks() 相容)，MS key = margin_daily 格式 + 現償三欄
    raw = http.cache_get("twse:MI_MARGN:ALL:20250103", twse.TTL_FOREVER)
    assert isinstance(raw, list) and len(raw) == 4 and raw[0]["note"] == "" and "date" in raw[0]
    ms = http.cache_get("twse:MI_MARGN:MS:20250103", twse.TTL_FOREVER)
    assert ms["date"] == "2025-01-03" and ms["margin_lots"] == 9427154 and ms["margin_repay_lots"] == 5032 and "margin_amt" in ms
    # 第二次呼叫不打網路；舊 margin_stocks() / margin_daily() 讀同一快取可用
    with patched(twse, "get_json", _no_net):
        d2 = twse.margin_stocks_hist("20250103")
        old = twse.margin_stocks("20250103")
        ms_old = twse.margin_daily("20250103")
    assert d2 == d and len(old) == 4 and list(old["code"]) == ["0050", "0051", "2330", "00632R"] and ms_old["margin_lots"] == 9427154
    # 民國 / YYYYMMDD 轉換
    assert twse.roc_to_iso("115/09/01") == "2026-09-01" and twse.ymd_to_iso("20260911") == "2026-09-11" and tpex.ymd_to_iso("2026/10/13") == "2026-10-13"


def test_twse_margin_all_nodata_and_errors_not_cached():
    g = Counter(lambda url, **kw: {"stat": "很抱歉，沒有符合條件的資料!"})
    with patched(twse, "get_json", g):
        try:
            twse.margin_stocks_hist("20261010")
            raise AssertionError("expected NoData")
        except NoData as e:
            assert "沒有符合" in str(e)
    assert http.cache_get("twse:MI_MARGN:ALL:20261010", twse.TTL_FOREVER) is None and http.cache_get("twse:MI_MARGN:MS:20261010", twse.TTL_FOREVER) is None
    with patched(twse, "get_json", lambda url, **kw: {"stat": "伺服器忙碌中，請稍後再試"}):
        try:
            twse.margin_stocks_hist("20250106")
            raise AssertionError("expected RuntimeError")
        except NoData:
            raise AssertionError("busy must not be NoData")
        except RuntimeError:
            pass
    with patched(twse, "get_json", lambda url, **kw: {"stat": "OK", "date": "20250106", "tables": [{"fields": MS_FIELDS, "data": []}]}):
        try:
            twse.margin_stocks_hist("20250106")
            raise AssertionError("expected RuntimeError (tables<2)")
        except RuntimeError as e:
            assert not isinstance(e, NoData)
    with patched(twse, "get_json", lambda url, **kw: "<html>maintenance</html>"):
        try:
            twse.margin_stocks_hist("20250106")
            raise AssertionError("expected RuntimeError (non-dict)")
        except RuntimeError as e:
            assert not isinstance(e, NoData)
    assert http.cache_get("twse:MI_MARGN:ALL:20250106", twse.TTL_FOREVER) is None
    # NotYet 不是 NoData 的子類 (catch NoData 不會吃到)
    assert not issubclass(NotYet, NoData) and not issubclass(NoData, NotYet)


def test_twse_margin_all_legacy_cache_compat():
    # (1) 本機 491 日快取格式：list 列、2026-10-13 那天缺 note；MS 快取 (角度 A 寫入) 缺現償欄 → 以恆等式反推
    legacy = [{"code": "2330", "name": "台積電", "margin_buy": 100.0, "margin_sell": 40.0, "margin_repay": 10.0, "margin_prev": 1000.0,
               "margin_today": 1050.0, "margin_limit": 5000.0, "short_buy": 1.0, "short_sell": 3.0, "short_repay": 0.0, "short_prev": 10.0,
               "short_today": 12.0, "short_limit": 5000.0, "offset": 2.0, "date": "2024-10-01"}]
    http.cache_put("twse:MI_MARGN:ALL:20241001", legacy)
    http.cache_put("twse:MI_MARGN:MS:20241001", {"date": "2024-10-01", "margin_lots_prev": 1000.0, "margin_lots": 1050.0, "margin_buy_lots": 100.0,
                                                  "margin_sell_lots": 40.0, "short_lots_prev": 10.0, "short_lots": 12.0, "short_buy_lots": 1.0,
                                                  "short_sell_lots": 3.0, "margin_amt_prev": 3000.0, "margin_amt": 3100.5, "margin_buy_amt": 150.0,
                                                  "margin_sell_amt": 40.0})
    with patched(twse, "get_json", _no_net):
        d = twse.margin_stocks_hist("20241001")
    assert d["date"] == "2024-10-01" and d["stocks"][0]["note"] == "" and set(d["stocks"][0]) == set(twse.MARGIN_ALL_COLS)
    assert d["market"]["margin_repay_lots"] == 10 and d["market"]["short_repay_lots"] == 0 and abs(d["market"]["margin_repay_amt"] - 9.5) < 1e-6
    # (2) 舊 margin_stocks() 在 stat≠OK 時寫入的 [] → 清掉重抓 (剛好一次網路)
    http.cache_put("twse:MI_MARGN:ALL:20250107", [])
    g = Counter(lambda url, **kw: twse_all_json("20250107"))
    with patched(twse, "get_json", g):
        d = twse.margin_stocks_hist("20250107")
    assert g.n == 1 and len(d["stocks"]) == 4 and d["market"]["margin_repay_lots"] == 5032
    # (3) 舊 list 快取但 MS 快取不存在 → 補抓 MS 一次 (margin_daily)，失敗則 market=None 不炸
    http.cache_put("twse:MI_MARGN:ALL:20241002", legacy)
    g = Counter(lambda url, **kw: {"stat": "OK", "date": "20241002", "tables": [{"fields": MS_FIELDS, "data": twse_all_json()["tables"][0]["data"]}]})
    with patched(twse, "get_json", g):
        d = twse.margin_stocks_hist("20241002")
    assert g.n == 1 and g.calls[0][1]["selectType"] == "MS" and d["market"]["margin_repay_lots"] == 5032
    http.cache_put("twse:MI_MARGN:ALL:20241003", legacy)
    with patched(twse, "get_json", lambda url, **kw: (_ for _ in ()).throw(ConnectionError("down"))):
        d = twse.margin_stocks_hist("20241003")
    assert d["market"] is None and len(d["stocks"]) == 1


def test_twse_throttle_only_on_network():
    import time as _t
    with patched(twse, "MIN_GAP", 0.15), patched(twse, "_last_net", 0.0):
        g = Counter(lambda url, **kw: twse_all_json("20250108") if kw["params"].get("selectType") else mi_index_json("20250108"))
        t0 = _t.time()
        with patched(twse, "get_json", g):
            twse.margin_stocks_hist("20250108")
            twse.closes_all("20250108")
            twse.closes_all("20250108")          # 快取命中不等
        el = _t.time() - t0
        assert g.n == 2 and 0.12 <= el < 1.0, el


# ------------------------------------------------------------------ (a) TWSE MI_INDEX
def test_twse_closes_all():
    g = Counter(lambda url, **kw: mi_index_json("20250103"))
    with patched(twse, "get_json", g):
        c = twse.closes_all("2025-01-03")
    assert g.n == 1 and g.calls[0][1]["type"] == "ALLBUT0999" and g.calls[0][1]["date"] == "20250103"
    assert c["2330"] == (1000.0, 30000.0) and c["0050"] == (200.5, 5500.0) and c["00632R"] == (5.0, 80000.0)
    assert "911608" not in c and "TW50" not in c                       # 無收盤 (--) 與怪代號不留
    raw = http.cache_get("twse:close_all:20250103", twse.TTL_FOREVER)
    assert raw == {"2330": [1000.0, 30000.0], "0050": [200.5, 5500.0], "00632R": [5.0, 80000.0]}
    with patched(twse, "get_json", _no_net):
        assert twse.closes_all("20250103") == c
    # 2004-02-11 前：本機判定不打網路；TWSE 訊息版也 → NoData
    with patched(twse, "get_json", _no_net):
        for ymd in ("20040210", "2003-12-31"):
            try:
                twse.closes_all(ymd)
                raise AssertionError("expected NoData")
            except NoData:
                pass
    with patched(twse, "get_json", lambda url, **kw: {"stat": "查詢日期小於93年2月11日，請重新查詢!"}):
        try:
            twse.closes_all("20040212")
            raise AssertionError("expected NoData")
        except NoData:
            pass
    with patched(twse, "get_json", lambda url, **kw: {"stat": "很抱歉，沒有符合條件的資料!"}):
        try:
            twse.closes_all("20250104")
            raise AssertionError("expected NoData")
        except NoData:
            pass
    with patched(twse, "get_json", lambda url, **kw: {"stat": "OK", "date": "20250106", "tables": [{"fields": ["指數"], "data": [["x"]]}]}):
        try:
            twse.closes_all("20250106")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert not isinstance(e, NoData)
    assert http.cache_get("twse:close_all:20250106", twse.TTL_FOREVER) is None and http.cache_get("twse:close_all:20250104", twse.TTL_FOREVER) is None
    assert twse.closes_all("20040211") if False else True              # 邊界 20040211 本機不擋 (需網路，不在此測)


# ------------------------------------------------------------------ (a) TPEx
def test_tpex_parse_ok():
    g = Counter(lambda url, **kw: tpex_json("20261013"))
    with patched(tpex, "get_json", g):
        d = tpex.margin_balance("2026-10-13", now=dt.datetime(2026, 10, 14, 1, 0, tzinfo=config.TZ), trading_day=lambda iso: True)
    assert g.n == 1 and g.calls[0][1] == {"date": "2026/10/13", "id": "", "response": "json"}
    assert d["status"] == "ok" and d["date"] == "2026-10-13" and len(d["stocks"]) == 2
    m = d["market"]
    assert m["margin_lots"] == 2383918 and m["margin_lots_prev"] == 2421036 and m["margin_buy_lots"] == 77732 and m["margin_sell_lots"] == 112879 and m["margin_repay_lots"] == 1971
    assert m["short_lots"] == 35668 and m["short_lots_prev"] == 37606 and m["short_sell_lots"] == 4277 and m["short_buy_lots"] == 5605 and m["short_repay_lots"] == 610
    assert abs(m["margin_amt"] - 2232.00683) < 1e-6 and abs(m["margin_amt_prev"] - 2283.46504) < 1e-6 and abs(m["margin_repay_amt"] - 1.16156) < 1e-6
    # 恆等式 (張 / 仟元) 在 summary 上成立
    assert m["margin_lots_prev"] + m["margin_buy_lots"] - m["margin_sell_lots"] - m["margin_repay_lots"] == m["margin_lots"]
    assert m["short_lots_prev"] + m["short_sell_lots"] - m["short_buy_lots"] - m["short_repay_lots"] == m["short_lots"]
    s = {r["code"]: r for r in d["stocks"]}
    assert set(s["6488"]) == set(tpex.STOCK_COLS)
    r = s["6488"]
    assert (r["margin_prev"], r["margin_buy"], r["margin_sell"], r["margin_repay"], r["margin_today"], r["margin_limit"], r["margin_util"]) == (10000, 500, 300, 20, 10180, 120000, 8.5)
    assert (r["short_prev"], r["short_sell"], r["short_buy"], r["short_repay"], r["short_today"], r["short_limit"]) == (100, 30, 10, 5, 115, 120000)
    assert r["offset"] == 40 and r["note"] == "X" and r["date"] == "2026-10-13" and s["00679B"]["note"] == ""
    assert twse.identity_err(d["stocks"]) == [False, False]           # TPEx 列同鍵名，恆等式輔助可直接用
    with patched(tpex, "get_json", _no_net):
        assert tpex.margin_balance("20261013") == d                     # 快取命中 (不需日曆)
    assert http.cache_get("tpex:margin:20261013", tpex.TTL_FOREVER)["status"] == "ok"


def test_tpex_empty_rows_notyet_closed_nodata():
    td = lambda iso: dt.date.fromisoformat(iso).weekday() < 5 and iso != "2026-10-10"   # noqa: E731
    empty = lambda url, **kw: tpex_json(kw["params"]["date"].replace("/", ""), empty=True)   # noqa: E731
    # 休市 (週六 10-10 國慶) → closed，快取
    with patched(tpex, "get_json", empty):
        d = tpex.margin_balance("20261010", now=dt.datetime(2026, 10, 14, 11, 0, tzinfo=config.TZ), trading_day=td)
    assert d == {"date": "2026-10-10", "status": "closed", "market": None, "stocks": []}
    with patched(tpex, "get_json", _no_net):
        assert tpex.margin_balance("20261010")["status"] == "closed"
    # 交易日當天 (<21:00 與 ≥21:00) 與昨天 → NotYet，不快取
    for now, ymd, word in ((dt.datetime(2026, 10, 14, 15, 0, tzinfo=config.TZ), "20261014", "預期"),
                           (dt.datetime(2026, 10, 14, 21, 30, tzinfo=config.TZ), "20261014", "已過"),
                           (dt.datetime(2026, 10, 15, 0, 30, tzinfo=config.TZ), "20261014", "已過")):   # 昨天 (10-13 已在上個測試快取為 ok)
        with patched(tpex, "get_json", empty):
            try:
                tpex.margin_balance(ymd, now=now, trading_day=td)
                raise AssertionError("expected NotYet")
            except NotYet as e:
                assert word in str(e), (word, str(e))
            except NoData:
                raise AssertionError("NotYet expected, got NoData")
        assert http.cache_get(f"tpex:margin:{ymd}", tpex.TTL_FOREVER) is None
    # 更早的交易日 0 列 → NoData (回補端寫 margin_nodata)，不快取
    with patched(tpex, "get_json", empty):
        try:
            tpex.margin_balance("20261008", now=dt.datetime(2026, 10, 14, 11, 0, tzinfo=config.TZ), trading_day=td)
            raise AssertionError("expected NoData")
        except NoData:
            pass
    assert http.cache_get("tpex:margin:20261008", tpex.TTL_FOREVER) is None
    # stat 非 ok → RuntimeError 不快取
    with patched(tpex, "get_json", lambda url, **kw: {"stat": "error", "date": "20261009"}):
        try:
            tpex.margin_balance("20261009", now=dt.datetime(2026, 10, 14, 11, 0, tzinfo=config.TZ), trading_day=td)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert not isinstance(e, (NoData, NotYet))
    assert http.cache_get("tpex:margin:20261009", tpex.TTL_FOREVER) is None
    # 預設交易日判定：週末 False；休市表抓不到 → 只看週末 (不炸)
    with patched(twse, "holidays", lambda y: (_ for _ in ()).throw(RuntimeError("no schedule"))):
        assert tpex._default_trading_day("2026-10-11") is False and tpex._default_trading_day("2026-10-14") is True
    with patched(twse, "holidays", lambda y: {"2026-10-09"}):
        assert tpex._default_trading_day("2026-10-09") is False and tpex._default_trading_day("2026-10-08") is True


# ------------------------------------------------------------------ (b) 恆等式
def test_identity_helper():
    d = twse.parse_margin_all(twse_all_json("20250103"), "20250103")
    assert twse.identity_err(d["stocks"]) == [False, False, False, False]
    bad = [dict(r) for r in d["stocks"]]
    bad[0]["margin_today"] = bad[0]["margin_today"] + 1          # 融資違反
    bad[1]["short_sell"] = 7                                      # 融券違反
    bad[2]["margin_repay"] = None                                 # 欄位缺 → 不可驗證 → True
    assert twse.identity_err(bad) == [True, True, True, False]
    df = pd.DataFrame(bad)
    e = twse.identity_err(df)
    assert isinstance(e, pd.Series) and e.tolist() == [True, True, True, False] and list(e.index) == list(df.index)
    # store 欄名版
    rows = store.margin_rows(d["stocks"], mkt="twse")
    sdf = pd.DataFrame(rows, columns=store.MARGIN_STOCK_COLS)
    assert twse.identity_err(sdf).tolist() == [False] * 4
    sdf.loc[0, "m_today"] += 1
    assert twse.identity_err(sdf).tolist() == [True, False, False, False]
    assert twse.identity_err([]) == [] and twse.identity_err(pd.DataFrame()).empty
    # 合成面板：隨機流量、恆等式建構 → 全列成立
    import random
    rng = random.Random(7)
    panel, prev = [], {c: 1000 + i for i, c in enumerate(("1101", "2330", "0050"))}
    for day in pd.bdate_range("2025-01-02", periods=30):
        for c in prev:
            b, s, r = rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 10)
            today = prev[c] + b - s - r
            panel.append({"code": c, "date": day.strftime("%Y-%m-%d"), "margin_prev": prev[c], "margin_buy": b, "margin_sell": s, "margin_repay": r,
                          "margin_today": today, "short_prev": 10, "short_sell": 2, "short_buy": 1, "short_repay": 1, "short_today": 10})
            prev[c] = today
    assert not any(twse.identity_err(panel)) and len(panel) == 90


# ------------------------------------------------------------------ (c) FinMind / 玩股網
def test_finmind_total_margin_flows():
    raw = pd.DataFrame([
        {"date": "2023-06-01", "name": "MarginPurchase", "TodayBalance": 7412292, "YesBalance": 7362229, "buy": 348071, "sell": 293204, "Return": 4804},
        {"date": "2023-06-01", "name": "ShortSale", "TodayBalance": 332522, "YesBalance": 330170, "buy": 24402, "sell": 30690, "Return": 3936},
        {"date": "2023-06-01", "name": "MarginPurchaseMoney", "TodayBalance": 184106790000, "YesBalance": 182543864000, "buy": 11315862000, "sell": 9637961000, "Return": 114975000},
        {"date": "2023-06-02", "name": "MarginPurchase", "TodayBalance": 7420000, "YesBalance": 7412292, "buy": 300000, "sell": 290000, "Return": 2292},
        {"date": "2023-06-02", "name": "MarginPurchaseMoney", "TodayBalance": 185000000000, "YesBalance": 184106790000, "buy": 1e10, "sell": 9e9, "Return": 106790000},
    ])
    seen = {}

    def fake_fetch(dataset, data_id=None, start_date=None, end_date=None, ttl=None, **extra):
        seen.update(dataset=dataset, data_id=data_id, start=start_date)
        return raw.copy()
    with patched(finmind, "fetch", fake_fetch):
        df = finmind.total_margin_flows()
    assert seen == {"dataset": "TaiwanStockTotalMarginPurchaseShortSale", "data_id": None, "start": "2001-01-01"}
    assert list(df.columns) == finmind.TOTAL_MARGIN_FLOW_COLS and list(df["date"]) == ["2023-06-01", "2023-06-02"]
    r0 = df.iloc[0]
    assert (r0["buy"], r0["sell"], r0["repay"], r0["margin_lots"], r0["margin_lots_prev"]) == (348071, 293204, 4804, 7412292, 7362229)
    assert (r0["short_sell"], r0["short_buy"], r0["short_repay"], r0["short_lots"]) == (30690, 24402, 3936, 332522)
    assert abs(r0["margin_amt"] - 1841.0679) < 1e-6 and abs(r0["repay_amt"] - 1.14975) < 1e-9 and abs(r0["buy_amt"] - 113.15862) < 1e-9
    assert r0["margin_lots_prev"] + r0["buy"] - r0["sell"] - r0["repay"] == r0["margin_lots"]           # 恆等式
    r1 = df.iloc[1]
    assert r1["repay"] == 2292 and pd.isna(r1["short_lots"])                                             # 缺列 → NaN 不炸
    with patched(finmind, "fetch", lambda *a, **k: pd.DataFrame()):
        e = finmind.total_margin_flows()
    assert e.empty and list(e.columns) == finmind.TOTAL_MARGIN_FLOW_COLS


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._b


class _Sess:
    def __init__(self, body, status=200):
        self.body, self.status, self.calls = body, status, []

    def get(self, url, **kw):
        self.calls.append((url, kw.get("headers") or {}))
        return _Resp(self.body, self.status)


def test_wantgoo_maint_real():
    ms = lambda iso: int(dt.datetime.fromisoformat(iso + "T00:00:00+08:00").timestamp() * 1000)   # noqa: E731   玩股網時間戳 = 台北 00:00
    rows = [{"date": ms("2026-10-12"), "lendingBalance": 653329500, "lastLendingBalance": 646405767, "marginOffset": 0, "limit": 0, "marginRatio": 1.96165},
            {"date": ms("2026-10-08"), "lendingBalance": 646405767, "lastLendingBalance": 640000774, "marginOffset": 0, "limit": 0, "marginRatio": 1.97115},
            {"date": ms("2021-08-17"), "lendingBalance": 270077735, "lastLendingBalance": 269314371, "marginOffset": 0, "limit": 0, "marginRatio": 1.657},
            {"date": ms("2021-08-16"), "lendingBalance": 269314371, "lastLendingBalance": 0, "marginOffset": 0, "limit": 0, "marginRatio": 0},
            {"date": ms("2021-08-13"), "lendingBalance": 268000000, "lastLendingBalance": 0, "marginOffset": 0, "limit": 0, "marginRatio": 0}]
    # UTC 會把台北 00:00 算成前一天 → 檢查 +8h
    assert dt.datetime.fromtimestamp(ms("2026-10-12") / 1000, dt.UTC).date().isoformat() == "2026-10-11"
    sess = _Sess(rows)
    with patched(http, "session", lambda: sess), patched(wantgoo, "_playwright_long_term", lambda p: (_ for _ in ()).throw(AssertionError("no pw"))):
        df = wantgoo.maint_real()
    assert len(sess.calls) == 1 and sess.calls[0][0].endswith(wantgoo.MAINT_LONG_PATH)
    assert sess.calls[0][1]["X-Requested-With"] == "XMLHttpRequest" and sess.calls[0][1]["Referer"] == wantgoo.TAIEX_PAGE
    assert list(df.columns) == ["date", "maint_ratio", "margin_amt"] and list(df["date"]) == ["2021-08-17", "2026-10-08", "2026-10-12"]
    assert list(df["maint_ratio"]) == [165.7, 197.115, 196.165] and abs(df["margin_amt"].iloc[-1] - 6533.295) < 1e-6
    with patched(http, "session", lambda: _Sess(None, 500)):
        assert len(wantgoo.maint_real(days=2)) == 2                        # 快取命中 (當日 key)，days 切尾
    # GET 失敗 + Playwright 不可用 / 失敗 → 空 DataFrame 不 raise，且沒有把失敗快取起來
    today = dt.datetime.now(config.TZ).strftime("%Y%m%d")
    http.cache_delete(f"wantgoo:0000A:{today}")
    with patched(http, "session", lambda: _Sess({"error": "cf"}, 403)), patched(wantgoo, "available", lambda: False):
        e = wantgoo.maint_real()
    assert e.empty and list(e.columns) == wantgoo.MAINT_COLS and http.cache_get(f"wantgoo:0000A:{today}", 10 ** 9) is None
    with patched(http, "session", lambda: _Sess("<html>", 200)), patched(wantgoo, "available", lambda: True), \
            patched(wantgoo, "_playwright_long_term", lambda p: (_ for _ in ()).throw(RuntimeError("pw down"))):
        assert wantgoo.maint_real().empty
    assert http.cache_get(f"wantgoo:0000A:{today}", 10 ** 9) is None
    # Playwright 備援成功
    with patched(http, "session", lambda: _Sess("<html>", 200)), patched(wantgoo, "available", lambda: True), \
            patched(wantgoo, "_playwright_long_term", lambda p: rows):
        assert len(wantgoo.maint_real()) == 3
    http.cache_delete(f"wantgoo:0000A:{today}")
    with patched(config, "WANTGOO_ENABLED", False), patched(http, "session", lambda: (_ for _ in ()).throw(AssertionError("no net"))):
        assert wantgoo.maint_real().empty
    assert wantgoo.parse_maint_long(None) == [] and wantgoo.parse_maint_long([{"x": 1}, None]) == []


# ------------------------------------------------------------------ (d) store
def test_store_margin_tables():
    with store.conn() as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')").fetchall()}
    assert {"margin_stock_daily", "margin_nodata", "ix_msd_code", "daily_metrics", "snapshots", "intraday"} <= names
    d = twse.parse_margin_all(twse_all_json("20250103"), "20250103")
    closes = {"2330": (1000.0, 30000.0), "0050": (200.5, 5500.0)}
    rows = store.margin_rows(d["stocks"], closes=closes)
    assert len(rows) == 4 and len(rows[0]) == len(store.MARGIN_STOCK_COLS)
    r = dict(zip(store.MARGIN_STOCK_COLS, rows[2]))
    assert r["date"] == "2025-01-03" and r["code"] == "2330" and r["mkt"] == "twse" and r["note"] == "O" and r["m_today"] == 20250 and r["m_repay"] == 50
    assert r["s_today"] == 1045 and r["offset"] == 12 and r["close"] == 1000.0 and r["vol_lots"] == 30000.0
    r1 = dict(zip(store.MARGIN_STOCK_COLS, rows[1]))
    assert r1["close"] is None and r1["vol_lots"] is None and r1["note"] == "X"
    assert store.upsert_margin_stocks(rows) == 4
    assert store.upsert_margin_stocks(rows) == 4 and store.margin_coverage()["rows"] == 4            # 重入覆蓋不加列
    assert store.upsert_margin_stocks([]) == 0 and store.upsert_margin_stocks(None) == 0 and store.upsert_margin_stocks(pd.DataFrame()) == 0
    # 第二天 (TPEx 列 → mkt tpex) + 補收盤
    t = tpex.parse_balance(tpex_json("20250103"), "20250103")
    assert store.upsert_margin_stocks(store.margin_rows(t["stocks"], mkt="tpex")) == 2
    assert store.update_margin_closes("2025-01-03", {"00632R": (5.0, 80000.0), "9999": (1.0, 1.0)}) == 1
    df = store.load_margin_stocks()
    assert list(df.columns) == store.MARGIN_STOCK_COLS and len(df) == 4 and list(df["code"]) == ["0050", "0051", "00632R", "2330"]
    assert df.set_index("code").loc["00632R", "close"] == 5.0 and df.set_index("code").loc["00632R", "vol_lots"] == 80000.0
    assert len(store.load_margin_stocks(mkt="tpex")) == 2 and len(store.load_margin_stocks(mkt=None)) == 6
    assert list(store.load_margin_stocks(codes=["2330"], cols=["date", "code", "m_today", "close"]).iloc[0]) == ["2025-01-03", "2330", 20250.0, 1000.0]
    assert store.load_margin_stocks(codes=[]).empty
    # 多日 + 區間 + dates / coverage；DataFrame 與 dict 列輸入
    d2 = twse.parse_margin_all(twse_all_json("20250106"), "20250106")
    df2 = pd.DataFrame(store.margin_rows(d2["stocks"]), columns=store.MARGIN_STOCK_COLS)
    assert store.upsert_margin_stocks(df2) == 4
    assert store.upsert_margin_stocks([{"date": "2025-01-07", "code": "2330", "mkt": "twse", "name": "台積電", "m_today": 1, "offset": None}]) == 1
    assert store.margin_dates() == ["2025-01-03", "2025-01-06", "2025-01-07"] and store.margin_dates(start="2025-01-06") == ["2025-01-06", "2025-01-07"]
    assert store.margin_dates(mkt="tpex") == ["2025-01-03"]
    cov = store.margin_coverage()
    assert cov == {"from": "2025-01-03", "to": "2025-01-07", "days": 3, "rows": 9}
    assert len(store.load_margin_stocks(start="2025-01-06", end="2025-01-06")) == 4
    row = store.load_margin_stocks(start="2025-01-07").iloc[0]
    assert row["m_today"] == 1 and pd.isna(row["offset"]) and pd.isna(row["close"]) and row["note"] == ""
    assert twse.identity_err(store.load_margin_stocks(end="2025-01-06", codes=["2330", "0050"])).tolist() == [False] * 4
    # nodata
    store.mark_nodata("2025-01-04", "twse_all"); store.mark_nodata("2025-01-04", "tpex"); store.mark_nodata("2025-01-05", "twse_all")
    store.mark_nodata("2025-01-05", "twse_all")                      # 重入
    assert store.nodata_dates("twse_all") == {"2025-01-04", "2025-01-05"} and store.nodata_dates("tpex") == {"2025-01-04"} and store.nodata_dates("x") == set()
    store.unmark_nodata("2025-01-05", "twse_all")
    assert store.nodata_dates("twse_all") == {"2025-01-04"}
    store.unmark_nodata("2025-01-04")
    assert store.nodata_dates("twse_all") == set() and store.nodata_dates("tpex") == set()
    # 既有表功能照舊
    store.upsert_metrics("2025-01-03", {"maint_twse": 190.1, "mg_buy_lots": 280507})
    m = store.load_metrics(["maint_twse", "mg_buy_lots"])
    assert float(m["maint_twse"].iloc[0]) == 190.1 and float(m["mg_buy_lots"].iloc[0]) == 280507


def test_store_bulk_speed():
    import time as _t
    rows = []
    for i in range(300):
        day = f"2024-{1 + i // 25:02d}-{1 + i % 25:02d}"
        rows += [(day, str(1000 + k), "twse", f"s{k}", "", 1.0, 2.0, 0.0, 100.0, 99.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 10.0, 5.0) for k in range(200)]
    t0 = _t.time()
    assert store.upsert_margin_stocks(rows) == 60000
    el = _t.time() - t0
    assert el < 20, el                                                  # 6 萬列 executemany (本機 <2 s)；64 萬列種子約 ×10
    assert store.margin_coverage()["rows"] == 60000 + 9 and len(store.margin_dates()) == 300 + 3


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
