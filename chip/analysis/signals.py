"""大盤買點 / 賣點訊號：規則式事件，逐條在 2010~ 長歷史上驗證 (5/10/20 日報酬、勝率、超額、t 值)，
再依驗證結果加權，輸出「目前買點強度 / 賣點強度」與歷史訊號點 (畫在 K 線上)。

規則只用已在 backtest / global_study 中證實有效或方向明確的條件，避免事後湊規則。

畫在 K 線上的買/賣點 (history_points) 2026-09-13 起加三道過濾，避免「高檔一堆買點」：
1. 只用驗證有效 (✓) 的規則；待定 (?) 只進文字不畫圖。
2. 規則分「低接/反轉 (rev)」與「動能 (mom)」：B7 費半動能、B10 台幣升值、S4/S7 跌破月線這類順勢規則只計入 mom_b/mom_s，不畫 ▲▼。
3. 價格位置過濾：買點只在 60 日區間位置 ≤ 75% 且月線乖離 ≤ +3% 才標；賣點只在位置 ≥ 25% 且乖離 ≥ -3% 才標；
   同一規則連續成立只標第一天 (邊緣觸發)，且 3 日內不重複標同方向。
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from .common import zscore

log = logging.getLogger(__name__)
REPORT_PATH = config.DATA_DIR / "signals_report.json"
HORIZONS = (5, 10, 20)


def _cross_up(s: pd.Series, level: float) -> pd.Series:
    return (s > level) & (s.shift(1) <= level)


def _cross_down(s: pd.Series, level: float) -> pd.Series:
    return (s < level) & (s.shift(1) >= level)


def rules(d: pd.DataFrame) -> dict[str, dict]:
    """回傳 {name: {'side': 'buy'|'sell', 'mask': Series, 'desc': str}}"""
    g = lambda c: d[c] if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    vix, sox20, kospi = g("g_vix_level"), g("g_sox_r20"), g("g_kospi_r1")
    fut_pct, fut_chg5 = g("fut_foreign_pct"), g("fut_foreign_chg5")
    smooth = g("composite_smooth")
    m20, r20 = g("margin_pct20"), g("ret20")
    fut_chg_z = zscore(fut_chg5, 60)
    return {
        "B1 月線負乖離 < -6% (超跌)": {"side": "buy", "mask": d["bias20"] < -6, "desc": "歷史 10 日 +3.5%、勝率 75%"},
        "B2 融資斷頭清洗後止穩": {"side": "buy", "mask": (m20 < -4) & (r20 < -3) & (d["ret5"] > 0), "desc": "融資 20 日 <-4% 且指數 <-3%，近 5 日翻正"},
        "B3 VIX > 30 恐慌且當日收紅": {"side": "buy", "mask": (vix > 30) & (d["ret1"] > 0), "desc": "恐慌後偏強"},
        "B4 外資期貨由極空快速回補": {"side": "buy", "mask": (fut_pct < 0.35) & (fut_chg_z > 1.5), "desc": "淨部位在低檔且 5 日變化 z>1.5"},
        "B5 KOSPI 前日崩跌 < -2.5%": {"side": "buy", "mask": kospi < -2.5, "desc": "韓股恐慌後台股 5 日反彈"},
        "B6 籌碼平滑分由空轉多 (上穿 -15)": {"side": "buy", "mask": _cross_up(smooth, -15), "desc": "綜合分轉折"},
        "B7 費半 20 日 > +15% 且站上月線": {"side": "buy", "kind": "mom", "mask": (sox20 > 15) & (d["close"] > d["ma20"]), "desc": "半導體動能外溢 (動能，不當低接)"},
        "B8 空頭狀態 + 平滑分 ≥ 25 + 站回月線": {"side": "buy", "mask": (d["state"] == "空頭") & (smooth >= 25) & (d["close"] > d["ma20"]), "desc": "空頭反轉確認"},
        "S1 VIX < 13 自滿 + 月線正乖離 > 4%": {"side": "sell", "mask": (vix < 13) & (d["bias20"] > 4), "desc": "自滿且過熱"},
        "S2 融資追價過熱 (背離 > 6%)": {"side": "sell", "mask": (m20 - r20 > 6) & (r20 > 0), "desc": "融資增速遠超指數 (長期 p90)"},
        "S3 外資連買 ≥5 日 + 乖離 > 3%": {"side": "sell", "mask": (d["foreign_streak"] >= 5) & (d["bias20"] > 3), "desc": "外資連買後歷史落後 (t=-5)"},
        "S4 外資現貨期貨空方一致 + 跌破月線": {"side": "sell", "kind": "mom", "mask": (g("foreign_consistency") == -1) & (d["close"] < d["ma20"]), "desc": "外資同步偏空 (順勢)"},
        "S5 籌碼平滑分由多轉空 (下穿 +15)": {"side": "sell", "mask": _cross_down(smooth, 15), "desc": "綜合分轉折"},
        "S6 爆量長黑": {"side": "sell", "mask": g("f_volume") == -2, "desc": "量能 1.8x 且跌逾 1%"},
        "S7 費半 20 日 < -10% 且跌破月線": {"side": "sell", "kind": "mom", "mask": (sox20 < -10) & (d["close"] < d["ma20"]), "desc": "半導體轉弱 (順勢)"},
        "S8 美元/台幣連漲 ≥5 日 (台幣持續貶)": {"side": "sell", "mask": g("g_usdtwd_streak") >= 5, "desc": "資金外流，歷史 20 日 -2.7%、勝率 40%"},
        "S9 台幣 20 日貶 >2% 且外資 20 日淨賣 >500 億": {"side": "sell", "mask": (g("g_usdtwd_r20") > 2) & (d["foreign_20d"] < -500), "desc": "匯率與外資同步流出"},
        "B9 油價 60 日跌 >25%": {"side": "buy", "mask": g("g_oil_r60") < -25, "desc": "油價崩跌後 60 日 +9%、勝率 67%"},
        "B10 台幣 60 日升 >3%": {"side": "buy", "kind": "mom", "mask": g("g_usdtwd_r60") < -3, "desc": "資金流入，20 日 +1.45%、勝率 68% (動能)"},
    }


def evaluate(d: pd.DataFrame, since: str = "2010-06-01") -> pd.DataFrame:
    d = d.copy()
    for h in HORIZONS:
        if f"fwd{h}" not in d:
            d[f"fwd{h}"] = (d["close"].shift(-h) / d["close"] - 1) * 100
    x = d[d["date"] >= since]
    base = {h: x[f"fwd{h}"].mean() for h in HORIZONS}
    rows = [{"訊號": "全體基準", "方向": "-", "樣本數": int(x["fwd10"].notna().sum()), **{f"{h}日均報酬%": round(base[h], 2) for h in HORIZONS},
             **{f"{h}日勝率%": round((x[f"fwd{h}"] > 0).mean() * 100, 1) for h in HORIZONS}, "10日超額%": 0.0, "t值": 0.0, "有效": "-"}]
    for name, r in rules(x).items():
        m = r["mask"].fillna(False)
        gsub = x[m]
        n = int(gsub["fwd10"].notna().sum())
        if n == 0:
            continue
        ex = gsub["fwd10"].mean() - base[10]
        sd = gsub["fwd10"].std()
        t = float(ex / (sd / np.sqrt(n))) if n > 1 and sd and sd == sd else np.nan
        sign = 1 if r["side"] == "buy" else -1
        valid = "✓" if (t == t and sign * t >= 1.5 and n >= 15) else ("?" if n < 15 else "✗")
        rows.append({"訊號": name, "方向": "買點" if r["side"] == "buy" else "賣點", "樣本數": n,
                     **{f"{h}日均報酬%": round(gsub[f"fwd{h}"].mean(), 2) for h in HORIZONS},
                     **{f"{h}日勝率%": round((gsub[f"fwd{h}"] > 0).mean() * 100, 1) for h in HORIZONS},
                     "10日超額%": round(ex, 2), "t值": round(t, 2) if t == t else None, "有效": valid, "說明": r["desc"], "類型": r.get("kind", "rev")})
    return pd.DataFrame(rows)


def current(d: pd.DataFrame, ev: pd.DataFrame, lookback: int = 3) -> dict:
    """近 lookback 日觸發的訊號，依驗證的 10 日超額加權成買/賣點強度。"""
    weights = {r["訊號"]: (r["10日超額%"] or 0, r["有效"]) for _, r in ev.iterrows() if r["訊號"] != "全體基準"}
    tail = d.tail(lookback)
    pos60, bias20 = price_position(d)
    p_last = float(pos60.iloc[-1]) if pd.notna(pos60.iloc[-1]) else 0.5
    b_last = float(bias20.iloc[-1]) if pd.notna(bias20.iloc[-1]) else 0.0
    high_zone = p_last > 0.75 or b_last > 3     # 價格在高檔：低接型買點打折
    low_zone = p_last < 0.25 or b_last < -3     # 價格在低檔：賣點打折
    fired_buy, fired_sell = [], []
    bs = ss = 0.0
    for name, r in rules(d).items():
        m = r["mask"].fillna(False).tail(lookback)
        if m.any():
            ex, valid = weights.get(name, (0, "?"))
            days = [str(x) for x in tail.loc[m.values, "date"]]
            kind = r.get("kind", "rev")
            rec = {"name": name, "excess10": ex, "valid": valid, "days": days, "desc": r["desc"], "kind": kind}
            w = max(0.0, ex if r["side"] == "buy" else -ex) * (1.0 if valid == "✓" else 0.5)
            if r["side"] == "buy" and high_zone:
                w *= 0.5; rec["note"] = "價格高檔，買點強度減半"
            if r["side"] == "sell" and low_zone:
                w *= 0.5; rec["note"] = "價格低檔，賣點強度減半"
            if r["side"] == "buy":
                fired_buy.append(rec)
                bs += w
            else:
                fired_sell.append(rec)
                ss += w
    net = bs - ss
    label = ("強力買點" if net >= 2.0 and bs > 0 else "買點" if net >= 0.8 else "強力賣點" if net <= -1.5 else "賣點" if net <= -0.6 else "中性")
    return {"buy_strength": round(bs, 2), "sell_strength": round(ss, 2), "net": round(net, 2), "label": label,
            "buy_signals": fired_buy, "sell_signals": fired_sell, "date": str(d["date"].iloc[-1]),
            "price_pos60": round(p_last * 100, 0), "bias20": round(b_last, 2),
            "zone": "高檔" if high_zone else "低檔" if low_zone else "中段"}


def price_position(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """60 日區間位置 (0=最低 1=最高) 與月線乖離 %。"""
    c = d["close"].astype(float)
    lo, hi = c.rolling(60, min_periods=20).min(), c.rolling(60, min_periods=20).max()
    pos = (c - lo) / (hi - lo).replace(0, np.nan)
    bias = d["bias20"] if "bias20" in d else (c / c.rolling(20).mean() - 1) * 100
    return pos, bias


def history_points(d: pd.DataFrame, ev: pd.DataFrame, days: int = 250, filtered: bool = True) -> pd.DataFrame:
    """近 days 日每天的買/賣訊號數與名稱 (畫圖用)。

    filtered=True (預設)：只用 ✓ 規則、低接/反轉型才算 ▲▼ (動能型記在 mom_b/mom_s)、價格位置過濾、邊緣觸發、3 日去重。
    filtered=False：舊行為 (✓/? 規則每天都標)，供比較。
    """
    valid = {r["訊號"]: r["有效"] for _, r in ev.iterrows()}
    rs = rules(d)
    pos60, bias20 = price_position(d)
    idx = d.index
    fired = {}
    for n, r in rs.items():
        m = r["mask"].reindex(idx).fillna(False).astype(bool)
        if filtered:
            m = m & ~m.shift(1, fill_value=False)   # 邊緣觸發：連續成立只標第一天
        fired[n] = m
    tail = d.tail(days).copy()
    buy_n, sell_n, mom_b, mom_s, names = [], [], [], [], []
    last_b = last_s = -10
    for k, i in enumerate(tail.index):
        ok_valid = ("✓",) if filtered else ("✓", "?")
        b_all = [n for n, r in rs.items() if r["side"] == "buy" and valid.get(n) in ok_valid and bool(fired[n].get(i, False))]
        s_all = [n for n, r in rs.items() if r["side"] == "sell" and valid.get(n) in ok_valid and bool(fired[n].get(i, False))]
        if not filtered:
            buy_n.append(len(b_all)); sell_n.append(len(s_all)); mom_b.append(0); mom_s.append(0)
            names.append("；".join(b_all + s_all))
            continue
        b_rev = [n for n in b_all if rs[n].get("kind", "rev") == "rev"]
        s_rev = [n for n in s_all if rs[n].get("kind", "rev") == "rev"]
        b_mom = [n for n in b_all if n not in b_rev]
        s_mom = [n for n in s_all if n not in s_rev]
        p = float(pos60.get(i, np.nan)); bz = float(bias20.get(i, np.nan))
        at_high = (p == p and p > 0.75) or (bz == bz and bz > 3)
        at_low = (p == p and p < 0.25) or (bz == bz and bz < -3)
        b_ok = b_rev if (b_rev and not at_high and k - last_b > 3) else []
        s_ok = s_rev if (s_rev and not at_low and k - last_s > 3) else []
        if b_ok:
            last_b = k
        if s_ok:
            last_s = k
        buy_n.append(len(b_ok)); sell_n.append(len(s_ok)); mom_b.append(len(b_mom)); mom_s.append(len(s_mom))
        tags = b_ok + s_ok + [f"{n} (動能)" for n in b_mom + s_mom]
        if b_rev and not b_ok and at_high:
            tags += [f"{n} (高檔不標)" for n in b_rev]
        if s_rev and not s_ok and at_low:
            tags += [f"{n} (低檔不標)" for n in s_rev]
        names.append("；".join(tags))
    tail["buy_n"], tail["sell_n"], tail["mom_b"], tail["mom_s"], tail["signal_names"] = buy_n, sell_n, mom_b, mom_s, names
    tail["pos60"] = (pos60.reindex(tail.index) * 100).round(0)
    return tail[["date", "close", "buy_n", "sell_n", "mom_b", "mom_s", "pos60", "signal_names"]]


def run(scored_long: pd.DataFrame, write: bool = True) -> dict:
    ev = evaluate(scored_long)
    cur = current(scored_long, ev)
    pts = history_points(scored_long, ev)
    rep = {"generated": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M"), "evaluation": ev.to_dict("records"), "current": cur}
    if write:
        REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"evaluation": ev, "current": cur, "points": pts}


def load_report() -> dict | None:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
