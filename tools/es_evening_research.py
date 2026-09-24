"""重算 chip/predict/es_evening 的叫牌表 / 跳空 / 個股表：
    python tools/es_evening_research.py --write --panel=<權值股面板 pkl (stock_forecast.build_panel() 存成 pickle)>
沒帶 --panel 時保留 data/models/es_evening_research.json 原有的個股表 (不會清空)。

門檻分級 (2026-09-25 審查修正)：'0.5' = |Δ|≥0.5%；'0.3' = 0.3%≤|Δ|<0.5% 的「區間」命中 (舊版是 ≥0.3% 累計，
混入強訊號而高估：23 點累計 70.7%，區間只有 62%)。App 在 0.3~0.5% 時用的就是這個區間值。
統計為 2024-05 起全期 (非走動式)；ES=F 連續合約的季度換月跳動 (約 9 天) 未排除，影響小。

時間對齊 (2026-09-25 修正)：Yahoo 60m 的時間戳是 K 棒「開始」時間，K 棒收盤要到開始後 60 分才知道。
- 基準：當天 13:00 開始那根 K 的收盤 (約 14:00，台股 13:30 收盤之後)，與 es_evening.build 相同。
- 整點 H 的已知價：K 棒「結束」≤ H，也就是開始時間 ≤ H−1 小時的最後一根。
  舊版用開始時間 ≤ H，等於偷看 1 小時，21~23 點命中被高估 2~5 個百分點。
輸出：整點 × 門檻 (0.3/0.5%) 的隔天收盤方向命中、樣本數、逐年最低；跳空 a+β×ES；權值股隔天同向表 (23:00，|ES|≥0.5%)。"""
import json, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from chip.analysis import backtest
from chip.predict import es_evening as EV

HOURS = list(range(21, 29))   # 21:00 ~ 隔日 04:00


def main(write: bool = False, stock_panel: str | None = None):
    es = EV._es_hourly("730d")
    tw = backtest.load_long("2024-01-01")[["date", "close", "open"]].copy(); tw["date"] = tw["date"].astype(str)
    tw["nr"] = (tw["close"].shift(-1) / tw["close"] - 1) * 100; tw["gap"] = (tw["open"].shift(-1) / tw["close"] - 1) * 100
    tw = tw.dropna(subset=["nr"])

    def base_px(d):   # 13:00 開始那根
        t0 = pd.Timestamp(d + " 13:00", tz="Asia/Taipei"); s = es[(es.index <= t0) & (es.index > t0 - pd.Timedelta(hours=3))]
        return s.iloc[-1] if len(s) else np.nan

    def known_at(t):  # K 棒結束 ≤ t
        s = es[(es.index <= t - pd.Timedelta(hours=1)) & (es.index > t - pd.Timedelta(hours=4))]
        return s.iloc[-1] if len(s) else np.nan

    rows = []
    for d, nr, gp in zip(tw.date, tw.nr, tw.gap):
        b = base_px(d)
        if not np.isfinite(b):
            continue
        r = {"date": d, "y": d[:4], "nr": nr, "gap": gp}
        for h in HOURS:
            v = known_at(pd.Timestamp(d, tz="Asia/Taipei") + pd.Timedelta(hours=h)); r[h] = (v / b - 1) * 100 if np.isfinite(v) else np.nan
        rows.append(r)
    x = pd.DataFrame(rows)
    print("天數", len(x), "隔天上漲", round((x.nr > 0).mean(), 3), x.date.min(), "~", x.date.max())
    table, gap = {}, {}
    for h in HOURS:
        key = f"{h % 24:02d}"; row = {}
        for th, lo_, hi_ in (("0.3", 0.3, 0.5), ("0.5", 0.5, 1e9)):
            a = x[h].abs(); sel = x[x[h].notna() & (a >= lo_) & (a < hi_)]; hit = (sel.nr > 0) == (sel[h] > 0); by = hit.groupby(sel.y).mean()
            row[th] = {"hit": round(float(hit.mean()), 3), "n": int(len(sel)), "yr_min": round(float(by.min()), 3), "band": [lo_, None if hi_ > 1e8 else hi_]}
        table[key] = row
        ok = x[h].notna(); xv, g = x.loc[ok, h].values, x.loc[ok, "gap"].values; yrs = x.loc[ok, "y"].values
        b = np.polyfit(xv, g, 1); p = np.polyval(b, xv); sel = np.abs(p) > 0.15
        hit = np.sign(p[sel]) == np.sign(g[sel]); by = pd.Series(hit).groupby(yrs[sel]).mean()
        gap[key] = {"beta": round(float(b[0]), 3), "a": round(float(b[1]), 3), "dir_hit": round(float(hit.mean()), 3), "n": int(sel.sum()), "yr_min": round(float(by.min()), 3),
                    "r2": round(float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum()), 3)}
        print(key, "≥0.5:", row["0.5"], "0.3~0.5:", row["0.3"], "gap:", gap[key])
    stocks = {}
    if stock_panel:
        P = pd.read_pickle(stock_panel)[["date", "stock_id", "close"]].sort_values(["stock_id", "date"])
        P["nr"] = P.groupby("stock_id")["close"].transform(lambda c: (c.shift(-1) / c - 1) * 100)
        S = P.merge(x[["date", "y", 23]].rename(columns={23: "es"}), on="date").dropna(subset=["nr", "es"]); S = S[S.es.abs() >= 0.5]
        S["same"] = np.sign(S.nr) == np.sign(S.es)
        for sid, g in S.groupby("stock_id"):
            by = g.groupby("y")["same"].mean()
            stocks[sid] = {"hit": round(float(g.same.mean()), 3), "n": int(len(g)), "yr_min": round(float(by.min()), 3)}
        print("個股 ≥60%:", sum(1 for v in stocks.values() if v["hit"] >= 0.6), "台積電", stocks.get("2330"))
    if write and not stocks:
        try:
            stocks = json.load(open("data/models/es_evening_research.json", encoding="utf-8")).get("stocks") or {}
            print("未帶 --panel：保留原有個股表", len(stocks), "檔")
        except Exception:  # noqa: BLE001
            pass
    out = {"table": table, "gap": gap, "stocks": stocks, "stocks_rule": "23:00 已知 ES 相對 13:00 |Δ|≥0.5% 的日子", "n_days": int(len(x)), "base_up": round(float((x.nr > 0).mean()), 3), "range": [x.date.min(), x.date.max()]}
    if write:
        json.dump(out, open("data/models/es_evening_research.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    sp = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--panel=")), None)
    main(write="--write" in sys.argv, stock_panel=sp)
