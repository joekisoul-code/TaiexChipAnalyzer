"""2026-10-12 小時模型改 TWSE 每 5 秒指數 (2016~) + 早盤模型：python tests/test_intraday_twse.py
本機 LightGBM DLL 可能被系統封鎖 → 以簡單線性學習器替換 M.fit_ensemble / predict_ensemble / explain (只測資料流程與選擇邏輯)。"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chip.predict import intraday as I, model as M  # noqa: E402
from chip.sources import twse  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print("  ✓", name)


# ---------------------------------------------------------------- day_marks
def ticks(prev, path):
    """path: {HH:MM:SS: value}；中間以前值補成每 5 秒。"""
    out, t, v = [["09:00:00", prev]], dt.datetime(2026, 1, 1, 9, 0, 5), None
    keys = dict(path)
    while t <= dt.datetime(2026, 1, 1, 13, 30, 0):
        s = t.strftime("%H:%M:%S")
        v = keys.get(s, v)
        out.append([s, v])
        t += dt.timedelta(seconds=5)
    return out


rows = ticks(100.0, {"09:00:05": 100.2, "09:01:00": 98.0, "09:05:00": 101.0, "09:14:00": 102.0, "09:20:00": 99.5, "10:30:00": 103.0, "13:30:00": 101.5})
r = twse.day_marks(rows)
check("前收 = 09:00:00、開盤 = 09:05 (避開 09:00:05 失真與 09:01 的 98)", r["prev_close"] == 100.0 and r["open"] == 101.0)
check("各時間點 = 該時點 (含) 前最後一筆", r["09:15"] == 102.0 and r["09:30"] == 99.5 and r["10:00"] == 99.5 and r["11:00"] == 103.0 and r["13:30"] == 101.5)
check("累積高低從 09:05 起算 (09:01 的 98 不算)", r["lo"]["09:15"] == 101.0 and r["lo"]["09:30"] == 99.5 and r["hi"]["11:00"] == 103.0 and r["hi"]["09:15"] == 102.0)
check("沒有 13:30 (盤中/不完整) → None", twse.day_marks(rows[:-10]) is None)

# ---------------------------------------------------------------- mark_rows (早盤)
rec = {"prev_close": 100.0, "open": 101.0, "09:15": 102.0, "09:30": 99.5, "10:00": 99.5, "11:00": 103.0, "12:00": 103, "13:00": 102, "13:30": 101.5,
       "hi": {"09:15": 102.0, "09:30": 102.0}, "lo": {"09:15": 101.0, "09:30": 99.5}}
er = I.mark_rows("2026-01-02", rec, {}, None, marks_list=I.EARLY_MARKS)
e30 = next(x for x in er if x["mark"] == "09:30")
check("早盤列：pre / 09:15 / 09:30，mark_idx 依早盤清單", [x["mark"] for x in er] == ["pre", "09:15", "09:30"] and e30["mark_idx"] == 2)
check("09:30 的 ret_1h 以 09:15 價計、目標 = 到 13:30 報酬", abs(e30["ret_1h"] - (99.5 / 102 - 1) * 100) < 1e-9 and abs(e30["to_13:30"] - (101.5 / 99.5 - 1) * 100) < 1e-9)
check("原主模型清單不受影響 (pre/10:00..13:00)", [x["mark"] for x in I.mark_rows("2026-01-02", rec, {}, None)] == I.MARKS)

# ---------------------------------------------------------------- train()：來源比較 + 早盤模型 (合成資料、替換學習器)
rng = np.random.default_rng(0)
dates = pd.bdate_range("2016-01-04", "2026-09-30").strftime("%Y-%m-%d").tolist()


def synth(ds, informative):
    out, pc = {}, 100.0
    for d in ds:
        g = rng.normal(0, 0.5)
        o = pc * (1 + g / 100)
        rec = {"prev_close": pc, "open": o, "hi": {}, "lo": {}}
        px, hi, lo = o, o, o
        drift = (o / pc - 1) * 100 * (0.8 if informative else 0.0)   # 有訊號：跳空 (%) 後續同向延續
        for m in twse.MARK_TIMES:
            px = px * (1 + drift / 100 / 7 + rng.normal(0, 0.15) / 100)
            hi, lo = max(hi, px), min(lo, px)
            rec[m], rec["hi"][m], rec["lo"][m] = px, hi, lo
        out[d] = rec
        pc = rec["13:30"]
    return out


TW = synth(dates, True)
YH = {d: {k: v for k, v in r.items() if k not in ("09:15", "09:30")} for d, r in synth(dates[-500:], False).items()}
for r in YH.values():
    r["hi"] = {k: v for k, v in r["hi"].items() if k not in ("09:15", "09:30")}
    r["lo"] = {k: v for k, v in r["lo"].items() if k not in ("09:15", "09:30")}


class _Lin:
    def fit(self, X, y):
        Z = np.nan_to_num(X.astype(float).values)
        self.w = np.linalg.lstsq(np.c_[Z, np.ones(len(Z))], y.values, rcond=None)[0]
        return self

    def predict(self, X):
        Z = np.nan_to_num(X.astype(float).values)
        return np.c_[Z, np.ones(len(Z))] @ self.w


M.fit_ensemble = lambda X, y, p=None: [_Lin().fit(X, y)]
M.predict_ensemble = lambda ms, X: np.mean([m.predict(X) for m in ms], axis=0)
M.explain = lambda ms, x, names, top=4: {}
daily = {d: {f: 0.0 for f in I.FEATURES if f not in ("mark_idx", "gap_pct", "ret_open", "ret_prev", "hi_pos", "range_pct", "ret_1h", "night_chg_pct")} for d in dates}
I.market_matrix = lambda x: pd.DataFrame({"date": dates})
I._daily_features = lambda mat: daily
I.backtest.load_long = lambda *a, **k: None
I.finmind.tx_night_history = lambda *a, **k: pd.DataFrame()
I.twse.intraday_marks = lambda *a, **k: TW
I.yahoo.daily_marks = lambda *a, **k: YH
saved = {}
I.M.save = lambda name, b: saved.__setitem__(name, b)
I.M.save_json = lambda name, o: saved.__setitem__(name + ".json", o)
I.twse.save_intraday_marks = lambda d: saved.__setitem__("marks", len(d))
res = I.train(write=True)
src = res["_source"]
check(f"兩來源在共同月份比較 ({src.get('months')} 月)：有訊號的 TWSE 勝出", src["chosen"] == "twse" and src["twse"] > src["yahoo"] and src["months"] >= 12)
b = saved["intraday"]
check("bundle 記錄來源與比較、主模型 5 個目標", b["source"] == "twse" and set(b["models"]) == set(I.TARGET_MARKS))
check("早盤模型：09:15 / 09:30，5 個目標皆有模型與校準", b["early"]["marks"] == ["09:15", "09:30"] and all(b["early"]["calibration"][t] for t in I.TARGET_MARKS))
check("walk-forward 自 2018 起、逐季重訓 (指標月份 ≥ 2018-01)", min(x["year"] for x in b["metrics"]["13:30"]["by_year"]) >= "2018-01")
check("補抓的歷史寫回 data/intraday_marks.json", saved.get("marks") == len(TW))
check("intraday_metrics.json 帶 _source / _early", "_source" in saved["intraday_metrics.json"] and saved["intraday_metrics.json"]["_early"])

# Yahoo 勝出時不建早盤模型
TW2 = synth(dates, False)
I.twse.intraday_marks = lambda *a, **k: {d: r for d, r in TW2.items()}
YH2 = {d: TW[d] for d in dates[-500:]}   # Yahoo 改用有訊號的資料
I.yahoo.daily_marks = lambda *a, **k: YH2
res2 = I.train(write=False)
check("TWSE 明顯較差 → 選 Yahoo，且不建早盤模型", res2["_source"]["chosen"] == "yahoo" and not any(k.startswith("early") for k in res2))

# ---------------------------------------------------------------- 盤中即時：09:20 用早盤模型
bars = [{"time": f"09:{m:02d}", "close": 100 + (m % 7) * 0.1} for m in range(0, 21)]
snap = {"intraday": {"bars": bars}, "taiex": {"prev": 99.0, "last": 100.4}}


class _FakeDT(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return dt.datetime(2026, 10, 12, 9, 20, tzinfo=tz)


I.dt = type("dtmod", (), {"datetime": _FakeDT, "date": dt.date, "timedelta": dt.timedelta})
rec2, mk = I._live_rec(snap, early=True)
check("09:20 + 早盤模型 → mark 09:15，高低從 09:05 起", mk == "09:15" and rec2["lo"]["09:15"] == min(b["close"] for b in bars if b["time"] >= "09:05"))
rec3, mk3 = I._live_rec(snap, early=False)
check("沒有早盤模型 → 維持舊行為 (10:00 近似)", mk3 == "10:00")
I.M.load = lambda name: b
I.market_matrix = lambda scored: pd.DataFrame({"date": ["2026-10-08"], "close": [99.0], "ret1": [0], "ret5": [0], "bias20": [0], "vola20": [1],
                                               "composite_smooth": [0], "fut_foreign_pct": [0.5], "foreign_z5": [0], "state_bull": [1], "state_bear": [0], "days_to_settle": [5]})
fc = I.forecast(None, {**snap, "phase": "open"})
check("forecast：早盤模型、mark 09:15、5 個目標、註記訊號弱", fc["model"] == "early" and fc["mark"] == "09:15" and len(fc["targets"]) == 5 and "早盤模型" in fc["note"] and fc["source"] == "twse")
print(f"\nintraday twse: {ok} passed")
