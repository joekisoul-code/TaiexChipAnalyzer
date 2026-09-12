"""驗證：(1) 原油與股市負相關？ (2) 美元/台幣持續上漲 → 台股下跌、資金流出？"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
from chip.analysis import backtest  # noqa: E402
from chip.sources import global_markets as gm  # noqa: E402

mk = gm.all_markets()
tw = backtest.load_long("2007-01-01")[["date", "close", "foreign", "ret1", "ret20"]].copy()
tw["foreign_20d"] = tw["foreign"].rolling(20).sum()
tw["fwd5"] = (tw["close"].shift(-5) / tw["close"] - 1) * 100
tw["fwd20"] = (tw["close"].shift(-20) / tw["close"] - 1) * 100
tw["fwd60"] = (tw["close"].shift(-60) / tw["close"] - 1) * 100


def align(key, dates):
    d = mk[key].copy()
    d["dt"] = pd.to_datetime(d["date"])
    d = d.sort_values("dt").drop_duplicates("dt")
    for w in (1, 5, 20, 60):
        d[f"r{w}"] = d["close"].pct_change(w) * 100
    d["hi60"] = (d["close"] >= d["close"].rolling(60).max()).astype(int)
    d["up_streak"] = (d["close"].diff() > 0).astype(int).groupby((d["close"].diff() <= 0).cumsum()).cumsum()
    m = pd.merge_asof(pd.DataFrame({"dt": pd.to_datetime(dates)}).sort_values("dt"), d, on="dt", direction="backward", allow_exact_matches=False)
    return m


def corr_table(name, feat, target_cols):
    rows = []
    for tc in target_cols:
        m = feat.notna() & tw[tc].notna()
        rows.append({"目標": tc, "Pearson": round(feat[m].corr(tw[tc][m]), 3), "Spearman": round(feat[m].rank().corr(tw[tc][m].rank()), 3), "n": int(m.sum())})
    print(f"\n{name}")
    print(pd.DataFrame(rows).to_string(index=False))


def by_year(feat, target):
    tw["year"] = tw["date"].str[:4]
    out = {}
    for y, g in tw.groupby("year"):
        f = feat[g.index]
        m = f.notna() & g[target].notna()
        if m.sum() > 50:
            out[y] = round(f[m].corr(g[target][m]), 2)
    return out


targets = ["ret1", "ret20", "fwd5", "fwd20", "fwd60"]
# ---------------- 原油
for key, nm in (("oil", "WTI"), ("brent", "布蘭特")):
    a = align(key, tw["date"])
    print(f"\n======== {nm} ========")
    corr_table(f"{nm} 前一日報酬 r1 vs 台股", a["r1"].values * np.ones(len(tw)) if False else pd.Series(a["r1"].values, index=tw.index), ["ret1", "fwd5", "fwd20"])
    r20 = pd.Series(a["r20"].values, index=tw.index)
    r60 = pd.Series(a["r60"].values, index=tw.index)
    corr_table(f"{nm} 20 日漲跌 vs 台股 (同期 ret20 = 同步；fwd = 領先)", r20, ["ret20", "fwd5", "fwd20", "fwd60"])
    corr_table(f"{nm} 60 日漲跌 vs 台股", r60, ["ret20", "fwd20", "fwd60"])
    print("  逐年：油 20 日 vs 台股同期 20 日 (同步相關)：", by_year(r20, "ret20"))
    print("  逐年：油 20 日 vs 台股未來 20 日 (領先相關)：", by_year(r20, "fwd20"))
    # 情境：油價大漲/大跌 60 日
    for label, mask in (("油 60 日 > +25% (供給/通膨衝擊)", r60 > 25), ("油 60 日 < -25%", r60 < -25), ("油 20 日 > +15%", r20 > 15), ("油 20 日 < -15%", r20 < -15)):
        g = tw[mask.fillna(False)]
        print(f"  {label}: n={len(g)} 同期20日 {g['ret20'].mean():+.2f}% | 未來 5/20/60 日 {g['fwd5'].mean():+.2f}/{g['fwd20'].mean():+.2f}/{g['fwd60'].mean():+.2f}%  勝率20日 {(g['fwd20'] > 0).mean() * 100:.0f}% (基準 {(tw['fwd20'] > 0).mean() * 100:.0f}%)")

# ---------------- 美元/台幣
a = align("usdtwd", tw["date"])
print("\n\n======== 美元/台幣 (USD/TWD 上漲 = 台幣貶值) ========")
for w in (5, 20, 60):
    s = pd.Series(a[f"r{w}"].values, index=tw.index)
    corr_table(f"美元/台幣 {w} 日變化 vs 台股", s, ["ret20", "fwd5", "fwd20", "fwd60"])
    fm = tw["foreign_20d"].notna() & s.notna()
    print(f"   與外資 20 日累計淨買 相關 (同期)：Pearson {s[fm].corr(tw['foreign_20d'][fm]):+.3f}")
r20 = pd.Series(a["r20"].values, index=tw.index)
r60 = pd.Series(a["r60"].values, index=tw.index)
print("  逐年：美元/台幣 20 日 vs 台股同期 20 日：", by_year(r20, "ret20"))
print("  逐年：美元/台幣 20 日 vs 台股未來 20 日：", by_year(r20, "fwd20"))
hi60 = pd.Series(a["hi60"].values, index=tw.index)
streak = pd.Series(a["up_streak"].values, index=tw.index)
for label, mask in (("台幣 20 日貶 > 2% (持續貶)", r20 > 2), ("台幣 60 日貶 > 3%", r60 > 3), ("美元/台幣創 60 日新高", hi60 == 1),
                    ("美元/台幣連漲 ≥ 5 日", streak >= 5), ("台幣 20 日升 > 2%", r20 < -2), ("台幣 60 日升 > 3%", r60 < -3),
                    ("台幣 20 日貶 > 2% 且外資 20 日淨賣 > 500 億", (r20 > 2) & (tw["foreign_20d"] < -500)),
                    ("台幣 20 日貶 > 2% 但外資 20 日淨買", (r20 > 2) & (tw["foreign_20d"] > 0))):
    g = tw[mask.fillna(False)]
    if len(g) < 10:
        continue
    print(f"  {label}: n={len(g)} 同期20日 {g['ret20'].mean():+.2f}% | 外資20日 {g['foreign_20d'].mean():+,.0f} 億 | 未來 5/20/60 日 {g['fwd5'].mean():+.2f}/{g['fwd20'].mean():+.2f}/{g['fwd60'].mean():+.2f}%  勝率20日 {(g['fwd20'] > 0).mean() * 100:.0f}%")
print(f"  全體基準：未來 5/20/60 日 {tw['fwd5'].mean():+.2f}/{tw['fwd20'].mean():+.2f}/{tw['fwd60'].mean():+.2f}%  勝率20日 {(tw['fwd20'] > 0).mean() * 100:.0f}%")
# 貨幣 → 資金流向：外資淨買 對 匯率 的領先/落後
print("\n  匯率變化與外資淨買的領先落後 (相關係數，正 lag = 匯率領先外資)：")
f5 = tw["foreign"].rolling(5).sum()
u5 = pd.Series(a["r5"].values, index=tw.index)
for lag in (-10, -5, -1, 0, 1, 5, 10):
    m = u5.shift(lag).notna() & f5.notna()
    print(f"    lag {lag:+d}: {u5.shift(lag)[m].corr(f5[m]):+.3f}", end="")
print()
