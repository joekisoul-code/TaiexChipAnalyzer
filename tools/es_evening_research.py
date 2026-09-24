"""重算 chip/predict/es_evening.TABLE：python tools/es_evening_research.py
Yahoo ES=F 60m (最長 730 天) × 台股隔一交易日收盤方向，逐整點 × 門檻 (0.3/0.5%) 命中、樣本數、逐年最低。"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from chip.analysis import backtest
from chip.predict import es_evening as EV
es = EV._es_hourly("730d")
tw = backtest.load_long("2024-01-01")[["date", "close"]].copy(); tw["date"] = tw["date"].astype(str)
tw["nr"] = (tw["close"].shift(-1) / tw["close"] - 1) * 100; tw = tw.dropna(subset=["nr"])
def at(t):
    s = es[(es.index <= t) & (es.index > t - pd.Timedelta(hours=3))]; return s.iloc[-1] if len(s) else np.nan
rows = []
for d, nr in zip(tw.date, tw.nr):
    b = at(pd.Timestamp(d + " 13:00", tz="Asia/Taipei"))
    if not np.isfinite(b): continue
    r = {"y": d[:4], "nr": nr}
    for h in range(16, 29): v = at(pd.Timestamp(d, tz="Asia/Taipei") + pd.Timedelta(hours=h)); r[h] = (v / b - 1) * 100 if np.isfinite(v) else np.nan
    rows.append(r)
x = pd.DataFrame(rows); print("天數", len(x), "隔天上漲", round((x.nr > 0).mean(), 3))
for h in range(16, 29):
    for th in (0.3, 0.5):
        sel = x[x[h].notna() & (x[h].abs() >= th)]; hit = (sel.nr > 0) == (sel[h] > 0); by = hit.groupby(sel.y).mean()
        print(f"{h % 24:02d}:00 ≥{th}: hit {hit.mean():.3f} n {len(sel)} yr_min {by.min():.2f}")
