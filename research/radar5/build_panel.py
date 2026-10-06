"""潛力股雷達 研究 5 共用資料 (10-07)：全上市 (TWSE) 個股 Yahoo 10y 日 K + ^TWII → parquet，研究代理離線共用。
只讀：用 chip.http.cached 同鍵 (tl:bars:{sym}:10y) 先吃本機快取 (忽略 TTL)，沒有才經 Worker 抓。不寫模型/發布檔。
注意：清單 = 2026-10-02 仍上市的股票 → 2016~ 期間下市的股票不在內 (全市場也有存活偏差，但大型股很少下市)。"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

sys.path.insert(0, "D:/TaiexChipAnalyzer")
from chip.http import cached  # noqa: E402
from chip.predict import treasure_live as L, model as M  # noqa: E402

OUT = Path(__file__).parent
t0 = time.time()
_, snap = L.market_snapshot("2026-10-02")
codes = sorted(c for c, s in snap.items() if len(c) == 4 and c.isdigit() and not c.startswith("00") and s.get("value", 0) > 0)
print("codes", len(codes), flush=True)


def fetch(sym):
    j = L.session().get(L.WORKER + "/idxh", params={"sym": sym, "range": "10y", "interval": "1d"}, headers=L.UA, timeout=60).json()
    return [x for x in (j.get("data") or []) if x.get("close")]


def one(c):
    for suf in (".TW", ".TWO"):
        sym = c + suf
        for _ in range(3):
            try:
                rows = cached(f"tl:bars:{sym}:10y", 10 ** 9, lambda sym=sym: fetch(sym), allow_stale=True)
                if rows and len(rows) > 60:
                    g = pd.DataFrame(rows)
                    g["date"] = g["date"].astype(str).str[:10]
                    return g[["date", "open", "high", "low", "close", "volume"]].assign(code=c)
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
    return None


with ThreadPoolExecutor(8) as ex:
    frames = [x for x in ex.map(one, codes) if x is not None]
P = pd.concat(frames, ignore_index=True)
for k in ("open", "high", "low", "close", "volume"):
    P[k] = pd.to_numeric(P[k], errors="coerce")
P = P[P["close"] > 0].drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
P.to_parquet(OUT / "twse_10y.parquet", index=False)
tw = cached("tl:twii:10y:research", 10 ** 9, lambda: fetch("^TWII"), allow_stale=True)
mk = pd.DataFrame(tw)[["date", "close"]]; mk["date"] = mk["date"].astype(str).str[:10]
mk.drop_duplicates("date").sort_values("date").to_parquet(OUT / "twii_10y.parquet", index=False)
tm = M.load_json("treasure_model")
pd.Series(sorted(tm["universe"]), name="code").to_frame().to_parquet(OUT / "universe170.parquet", index=False)
print("stocks", P["code"].nunique(), "rows", len(P), "dates", P["date"].min(), P["date"].max(), "secs", round(time.time() - t0), flush=True)
