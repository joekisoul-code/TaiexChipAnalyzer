"""台股大盤籌碼即時分析儀表板 (Streamlit)。

    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import logging
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from chip import config, notify, realtime, store
from chip.analysis import backtest, chips, cross_market, global_study, market, signals, stock
from chip.predict import intraday, market_forecast, stock_forecast
from chip.report import market_report, stock_report
from chip.sources import twse

logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="台股籌碼分析", page_icon="📊", layout="wide")

UP, DOWN, NEUTRAL = "#d64545", "#2e9e5b", "#8a8f98"   # 台股慣例：紅漲綠跌


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📊 台股籌碼分析")
    refresh = st.slider("自動更新 (秒，0=關閉)", 0, 300, 30, 5)
    use_wg = st.checkbox("使用玩股網 (Playwright)", value=config.WANTGOO_ENABLED,
                         help="取得大盤融資維持率、八大行庫各行庫明細、借券賣出歷史。需 pip install playwright && playwright install chromium")
    token = st.text_input("FinMind Token (選填)", value=config.FINMIND_TOKEN, type="password",
                          help="免費註冊可提高流量上限；贊助等級可用券商分點/八大行庫個股 API")
    if token != config.FINMIND_TOKEN:
        config.FINMIND_TOKEN = token
        os.environ["FINMIND_TOKEN"] = token
    stock_id = st.text_input("個股代碼", value="2330")
    watch_ids = st.text_input("追蹤清單 (逗號分隔)", value=",".join(chips.WATCHLIST))
    alerts_on = st.checkbox("盤中警示推播 (console/log/webhook)", value=True,
                            help="觸發時寫入 data/alerts.log；設定 CHIP_WEBHOOK_URL 或 CHIP_TELEGRAM_TOKEN/CHAT 可推播")
    if st.button("🔄 強制重新抓取"):
        st.cache_data.clear()
    st.caption("資料來源：TWSE、TAIFEX、FinMind、HiStock、玩股網。盤後籌碼於 15:00–21:30 陸續公布。")

if refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh * 1000, key="auto")
    except ImportError:
        st.sidebar.warning("pip install streamlit-autorefresh 以啟用自動更新")


# ------------------------------------------------------------------ data
@st.cache_data(ttl=300, show_spinner="抓取籌碼資料中…")
def load_market(use_wantgoo: bool):
    scored, a, meta = market.run(use_wantgoo=use_wantgoo)
    wg = meta.pop("_wantgoo", None)
    rank = meta.pop("_gov8_rank", None)
    return scored, a, meta, wg, rank


@st.cache_data(ttl=300, show_spinner="分析個股中…")
def load_stock(sid: str, market_regime: str, market_state: str, market_ret20: float | None):
    return stock.assess(sid, {"regime": market_regime, "state": market_state, "ret20": market_ret20})


@st.cache_data(ttl=600, show_spinner="計算大盤買賣點訊號（長歷史）…")
def load_signals():
    long = backtest.load_long("2010-01-01")
    return signals.run(long)


@st.cache_data(ttl=600, show_spinner="抓取追蹤清單籌碼（玩股網/FinMind/HiStock）…")
def load_watchlist(ids: tuple, use_wantgoo: bool):
    return chips.assess_watchlist(list(ids), use_wantgoo)


def color(v):
    return UP if (v or 0) > 0 else DOWN if (v or 0) < 0 else NEUTRAL


def gauge(score: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": title},
        number={"suffix": "", "font": {"size": 36}},
        gauge={"axis": {"range": [-100, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [-100, -40], "color": "#2e9e5b"}, {"range": [-40, -15], "color": "#9fd3b4"},
                         {"range": [-15, 15], "color": "#e6e6e6"}, {"range": [15, 40], "color": "#f2b5b5"},
                         {"range": [40, 100], "color": "#d64545"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def factor_table(factors):
    rows = []
    for f in factors:
        rows.append({"因子": f.name, "權重": f.weight, "分數": round(f.score, 2) if f.available else None,
                     "數值": f.value, "解讀": f.comment, "標籤": "、".join(f.tags)})
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"分數": st.column_config.ProgressColumn("分數", min_value=-2, max_value=2, format="%.2f")})


# ------------------------------------------------------------------ header
rt = realtime_quotes = None
scored, A, meta, wg, rank = load_market(use_wg)
last = scored.iloc[-1]
SNAP = realtime.snapshot(scored)          # 每次 rerun 都抓 (內部 5 秒快取)
rt = [dict(name="發行量加權股價指數", **SNAP["taiex"])] if SNAP.get("taiex") else []
if "alert_fired" not in st.session_state:
    st.session_state.alert_fired = set()
    st.session_state.alert_day = SNAP["ts"][:10]
    st.session_state.prev_snap = None
if st.session_state.alert_day != SNAP["ts"][:10]:
    st.session_state.alert_fired = set()
    st.session_state.alert_day = SNAP["ts"][:10]
NEW_ALERTS = realtime.check_alerts(SNAP, st.session_state.prev_snap, st.session_state.alert_fired)
st.session_state.prev_snap = SNAP
if NEW_ALERTS and alerts_on:
    for ev in NEW_ALERTS:
        notify.notify(ev["message"])
try:
    realtime.persist(SNAP)
except Exception:  # noqa: BLE001
    pass
RT = SNAP.get("score") or {"score": 0, "label": "無資料", "parts": []}

tx, txn, idx, otc = SNAP.get("tx") or {}, SNAP.get("tx_night") or {}, SNAP.get("taiex") or {}, SNAP.get("otc") or {}
cols = st.columns([1.2, 1, 1.1, 1, 1, 1, 1])
cols[0].metric(f"加權指數 ({idx.get('time', '')})", f"{idx.get('last', 0):,.2f}", f"{idx.get('chg', 0) or 0:+,.2f} ({idx.get('chg_pct', 0) or 0:+.2f}%)", delta_color="inverse")
cols[1].metric("櫃買", f"{otc.get('last', 0):,.2f}", f"{otc.get('chg_pct', 0) or 0:+.2f}%", delta_color="inverse")
cols[2].metric(f"台指期 {tx.get('symbol', '')[-3:]} ({'夜盤' if txn else '日盤'})",
               f"{(txn or tx).get('last', 0) or 0:,.0f}", f"{(txn or tx).get('change_pct', 0) or 0:+.2f}%｜價差 {(txn or tx).get('basis', 0) or 0:+.0f}", delta_color="inverse")
cols[3].metric("盤勢即時分", f"{RT['score']:+.1f}", RT["label"], delta_color="off")
cols[4].metric("綜合籌碼分 (3日平滑)", f"{A['composite_smooth']:+.1f}", f"{A['regime']}｜{A['state']}", delta_color="off")
cols[5].metric("外資現貨 (億)", f"{last['foreign']:+,.0f}", f"5日 {last['foreign_5d']:+,.0f}", delta_color="inverse")
cols[6].metric("量能推估 (20日均倍數)", f"{SNAP.get('vol_pace', 0) or 0:.2f}x", f"{SNAP.get('amount_projected', 0) or 0:,.0f} 億", delta_color="off")
phase_txt = {"open": "盤中", "pre": "開盤前", "post": "收盤後 (盤後交易)", "night": "夜盤時段", "closed": "休市"}[SNAP["phase"]]
st.caption(f"{phase_txt}｜即時快照 {SNAP['ts']}｜籌碼資料日期 {A['date']}｜計算時間 {meta.get('_built_at')}｜即時報價 {config.TTL_REALTIME} 秒快取")

tabs = st.tabs(["⚡ 即時追蹤", "🎯 大盤判讀", "📈 籌碼圖表", "🏦 八大行庫", "🔍 個股分析", "🧪 訊號驗證", "🔌 資料來源", "🔮 走勢預測", "🌏 國際連動", "📋 追蹤清單"])

with tabs[9]:
    st.markdown("### 📋 追蹤清單籌碼分布（籌碼在幾塊錢）")
    ids = tuple(s.strip() for s in watch_ids.split(",") if s.strip())
    WL = load_watchlist(ids, use_wg)
    summary = []
    for sid, a in WL.items():
        if "error" in a:
            summary.append({"代碼": sid, "狀態": a["error"]})
            continue
        q = a.get("quote") or {}
        h = a.get("holders") or {}
        vp = a.get("profile60") or {}
        summary.append({"代碼": sid, "名稱": a.get("name", sid), "現價": a["price"], "漲跌%": round(q.get("chg_pct", q.get("change_pct", 0)) or 0, 2) if q else None,
                        "籌碼判讀": a["label"], "分數": a["score"], "大戶>400張%": h.get("big400"), "大戶4週變化": h.get("big400_chg4w"),
                        "散戶<20張%": h.get("retail20"), "60日密集區": vp.get("poc"), "現價上方套牢%": vp.get("above_pct")})
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    for sid, a in WL.items():
        if "error" in a:
            continue
        with st.expander(f"{sid} {a.get('name', '')}｜{a['label']}｜現價 {a['price']:,.2f}", expanded=(sid == ids[0] if ids else False)):
            for n in a["notes"]:
                st.markdown("• " + n)
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown("**各路資金累積部位與平均成本**（5 / 20 / 60 日）")
                if not a["costs"].empty:
                    st.dataframe(a["costs"], width="stretch", hide_index=True)
                vp = a.get("profile60")
                if vp:
                    b = vp["bins"]
                    fvp = go.Figure(go.Bar(y=b["mid"].round(2), x=b["pct"], orientation="h",
                                           marker_color=[UP if m > vp["last"] else DOWN for m in b["mid"]]))
                    fvp.add_hline(y=vp["last"], line_color="#333", line_dash="dash", annotation_text=f"現價 {vp['last']}")
                    fvp.add_hline(y=vp["poc"], line_color="#e0a800", line_dash="dot", annotation_text=f"密集區 {vp['poc']}")
                    fvp.update_layout(height=360, title=f"60 日分價量（紅=現價上方套牢 {vp['above_pct']}%，綠=下方獲利 {vp['below_pct']}%）",
                                      xaxis_title="成交量 %", yaxis_title="價格", margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fvp, width="stretch")
            with c2:
                fl = a["flows"].tail(120)
                fc = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.3, 0.3],
                                   subplot_titles=("收盤價", "每日買賣超 (張)：外資 / 投信 / 主力 / 官股", "融資餘額 (張)"))
                fc.add_trace(go.Scatter(x=fl["date"], y=fl["close"], name="收盤", line=dict(color="#333")), row=1, col=1)
                for col, nm, clr in (("foreign", "外資", "#1f77b4"), ("trust", "投信", "#2ca02c"), ("main", "主力", "#d62728"), ("gov8", "官股", "#9467bd")):
                    if col in fl:
                        fc.add_trace(go.Bar(x=fl["date"], y=fl[col], name=nm, marker_color=clr), row=2, col=1)
                if "margin_lots" in fl:
                    fc.add_trace(go.Scatter(x=fl["date"], y=fl["margin_lots"], name="融資餘額", line=dict(color="#ff7f0e")), row=3, col=1)
                fc.update_layout(height=520, barmode="relative", legend_orientation="h", margin=dict(l=40, r=20, t=40, b=20))
                fc.update_xaxes(type="category", showticklabels=False)
                fc.update_xaxes(showticklabels=True, row=3, col=1, nticks=8)
                st.plotly_chart(fc, width="stretch")
            conc = a.get("concentration")
            if conc is not None and not conc.empty:
                fh = make_subplots(specs=[[{"secondary_y": True}]])
                fh.add_trace(go.Scatter(x=conc["date"], y=conc["big400"], name="大戶 >400 張 %", line=dict(color="#d62728")))
                fh.add_trace(go.Scatter(x=conc["date"], y=conc["retail20"], name="散戶 <20 張 %", line=dict(color="#2ca02c")))
                fh.add_trace(go.Scatter(x=conc["date"], y=conc["close"], name="收盤", line=dict(color="#999", dash="dot")), secondary_y=True)
                fh.update_layout(height=300, title="集保週資料：大戶 / 散戶持股比例 vs 股價", legend_orientation="h", margin=dict(l=40, r=20, t=40, b=20))
                st.plotly_chart(fh, width="stretch")
            if not a["brokers"].empty:
                st.markdown(f"**券商買賣均價**（{a.get('broker_date', '')}，玩股網）")
                st.dataframe(a["brokers"].head(15), width="stretch", hide_index=True)

with tabs[8]:
    st.markdown("### 🌏 國際市場即時")
    gq = SNAP.get("global") or []
    if gq:
        cols = st.columns(min(7, len(gq)))
        for i, q in enumerate(gq):
            with cols[i % len(cols)]:
                if q.get("last") is not None:
                    st.metric(f"{q['name']} ({q['time_tw']})", f"{q['last']:,.2f}", f"{(q['chg_pct'] or 0):+.2f}%", delta_color="inverse")
    grep = global_study.load_report()
    st.markdown("### 國際市場 × 台股歷史研究 (2007 起)")
    if grep:
        st.caption(f"產生 {grep['generated']}｜{grep['start']} ~ {grep['end']}，{grep['rows']} 日。更新：`python cli.py global`")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**同日效應**：前晚/前日報酬 與 台股當日 跳空 / 開盤後 / 全日 的排序相關")
            st.dataframe(pd.DataFrame(grep["same_day"]), width="stretch", hide_index=True)
            st.markdown("**近 60 日滾動相關** (目前連動狀態)")
            st.dataframe(pd.DataFrame(grep["rolling"]), width="stretch", hide_index=True)
        with c2:
            st.markdown("**預測力**：對台股未來 1/5/10/20 日報酬的 rank-IC 與逐年一致性 (前 20)")
            st.dataframe(pd.DataFrame(grep["predictive"]).head(20), width="stretch", hide_index=True)
        st.markdown("**事件研究**：海外事件發生後台股表現 vs 全體基準 (|t| ≥ 2 較可信)")
        evd = pd.DataFrame(grep["events"])
        st.dataframe(evd[["事件", "樣本數", "跳空%", "當日%", "5日均報酬%", "5日勝率%", "10日均報酬%", "20日均報酬%", "20日勝率%", "5日超額%", "t值(5日)"]],
                     width="stretch", hide_index=True, height=520)
        st.markdown("""
**結論摘要**
- 前晚美股 (Nasdaq / S&P500 / 費半 / 台積電 ADR) 與台股**跳空**的排序相關 0.54~0.63，但與**開盤後走勢**只有 0.07~0.09：海外消息在開盤已反映，不能拿來追盤中。
- **VIX 水準**是最穩定的領先指標：對台股未來 5/10/20 日 IC 0.08/0.11/0.15，94% 年份為正。VIX < 13 (自滿) 之後 5 日顯著偏弱 (t=−4)；VIX > 30 之後 20 日平均 +2.5%。
- **費半 20 日漲逾 15%** 後台股 5 日 +0.97% (t=3.5)、20 日勝率 74%：半導體動能會外溢，而台股自身趨勢卻是反指標。
- **KOSPI 前日跌逾 2.5%** 後台股 5 日 +1.2% (t=2.5)：韓股恐慌後台股反彈。
- 原油、黃金、匯率、日經、美債殖利率對台股未來報酬幾乎沒有可用的預測力；比特幣 20 日動能有微弱正向 (風險偏好代理)。
- 這些已整合為因子模型的「國際盤」因子 (VIX 水準 + 費半 20 日動能 + KOSPI 崩跌反彈) 與 ML 模型特徵；前晚美股則用於開盤前跳空預估。
""")
    else:
        st.info("尚未執行研究，請執行 `python cli.py global`。")
    st.markdown("### 外匯／利率／波動率／原物料 × 台股・美股・韓股・日股 (2007 起)")
    crep = cross_market.load_report()
    if crep:
        st.caption(f"產生 {crep['generated']}。特徵一律取目標市場交易日開盤前已知的最後一筆。更新：`python cli.py cross`")
        for line in crep["summary"]:
            st.markdown("• " + line)
        tsel = st.selectbox("目標市場", list(crep["targets"].keys()))
        tr = crep["targets"][tsel]
        gsel = st.radio("特徵群組", list(tr["groups"].keys()), horizontal=True)
        st.dataframe(pd.DataFrame(tr["groups"][gsel]), width="stretch", hide_index=True)
        st.markdown(f"**{tsel} 事件研究**（|t| ≥ 1.5 標粗）")
        evd = pd.DataFrame(tr["events"])
        st.dataframe(evd.style.apply(lambda row: ["font-weight: bold" if (row.get("t值") is not None and abs(row.get("t值") or 0) >= 1.5) else "" for _ in row], axis=1),
                     width="stretch", hide_index=True, height=520)
        st.markdown("""
**跨市場結論**
- **外匯**：台幣急貶 (美元/台幣 5 日 >1%) 後台股 5 日勝率只有 48%（基準 57%），急升則 20 日 +1.24%、勝率 65%；但整體 IC 很小，匯率是「同步指標」多於領先指標。
  **日圓急升 (美元/日圓 5 日 < −2%，套利交易平倉)** 當日四大市場皆跌，但之後 5 日全部反彈：台股 +0.61% (t=2.6)、韓股 +0.62% (t=2.2)、日經 +0.88% (t=3.1)。
  日圓急貶則對美股不利 (5 日 −0.51%，t=−3.0)。美元/日圓 5～20 日上升對台韓美 10 日皆為負向 IC。
- **利率**：美債 5Y/10Y 20 日上升對四個市場都是負向 (S&P IC −0.09～−0.11、韓 −0.11、台 −0.05～−0.07)。
  殖利率曲線 (10Y−3M) 倒掛期間台股 20 日 +1.70%、勝率 67% (t=2.6)、美股 +1.41% (t=2.5)：倒掛本身不是賣點，通常對應寬鬆預期。
- **波動率**：VIX／VIX3M／VXN 水準對台美韓 10～20 日報酬都是穩定正向 (美股 100% 年份、韓 94%、台 82%)；VIX 期限倒掛後台股 20 日 +1.96%。SKEW 沒有可用的預測力。
- **原物料**：**銅金比 20 日 >+8%** 是最一致的景氣訊號：台 +0.49% (t=3.9)、美 +0.28% (t=3.0)、韓 +0.57% (t=4.0)、日 +0.37% (t=2.7)。
  銅 20 日 >+8% 後台股 20 日 +2.67%、勝率 72%。**天然氣 20 日漲幅** 對台美韓都是負向 (IC −0.13，台股逐年 88% 為負)。
  乾散貨運價 ETF 20 日 >+15% 後台 +0.32% (t=2.6)、美 +0.30% (t=3.4)。原油單獨的 20 日漲跌幾乎無預測力；布蘭特單日 >5% 對韓股 5 日 +1.13% (t=2.5)。
  黃金 5 日 >4% 對韓股 +0.52% (t=2.0)，對台美日無效。
- 以上已整合：國際盤因子加入銅金比、天然氣、日圓急升、曲線倒掛、乾散貨；ML 模型加入對應特徵。

**「原油負相關」「台幣貶→台股跌」驗證 (2007 起)**
- 原油與台股**同期為正相關** (20 日 +0.24～+0.30，20 年中 17 年為正)，只有 2017/2022/2023/2026 這種供給型油價年份為負。油價 20 日變化沒有領先力；
  油 60 日**跌逾 25% 後** 60 日 +9.0%、勝率 67%；60 日**漲逾 25% 後** 20 日勝率降到 57%。
- 美元/台幣 20 日變化與台股同期 20 日 **Spearman −0.39，20 年中 18 年為負**：台幣貶 2% 時同期台股 −2.9%、外資 20 日 −709 億；創 60 日新高時外資 −971 億。
  但貶值「已發生」後未來 20 日反而 +1.19%；真正的領先賣訊是 **美元/台幣連漲 ≥5 日** (之後 20 日 −2.67%、60 日 −5.57%、勝率 40%)。
  台幣 60 日升值 >3% 後 20 日 +1.45%、勝率 68%。匯率與外資淨買在日頻上沒有領先落後，是同一件事的兩面。
- 已加入「匯率資金流」因子與買賣點 S8/S9/B9/B10。
""")
    else:
        st.info("尚未執行跨市場研究，請執行 `python cli.py cross`。")


def forecast_block(hz: dict, title: str):
    st.markdown(f"**{title}**")
    cols = st.columns(3)
    for i, (h, r) in enumerate(hz.items()):
        with cols[i]:
            diff = (r["p_up"] or 0) - r["base_hit"]
            tone = "偏多" if diff >= 0.04 else "偏空" if diff <= -0.04 else "中性"
            st.metric(f"未來 {h} 日｜{tone}", f"{r['p_up']:.0%} 上漲率", f"基準 {r['base_hit']:.0%}｜模型期望 {r['pred']:+.2f}%", delta_color="off")
            st.caption(f"歷史第 {r['bin'] + 1}/5 分位 (n={r['n']})：平均 {r['hist_mean']:+.2f}%，20~80% 區間 {r['q20']:+.2f}% ~ {r['q80']:+.2f}%")
    r10 = hz[10]
    c1, c2 = st.columns(2)
    c1.markdown("推升 (10 日)：" + "；".join(f"**{d['name']}** {d['contrib']:+.2f}" for d in r10["drivers"]["positive"][:5]) or "無")
    c2.markdown("壓抑 (10 日)：" + "；".join(f"**{d['name']}** {d['contrib']:+.2f}" for d in r10["drivers"]["negative"][:5]) or "無")


def metrics_table(mets: dict) -> pd.DataFrame:
    rows = []
    for h, m in mets.items():
        if m:
            rows.append({"視野": f"{h} 日", "樣本外 rank-IC": m["rank_ic"], "逐年 IC 平均": m["ic_year_mean"], "正 IC 年份": m["ic_positive_years"],
                         "基準上漲率": m["base_hit"], "五分位上漲率 (弱→強)": m["bin_hit"], "五分位平均報酬": m["bin_mean"], "頂底差%": m["spread_top_bottom"]})
    return pd.DataFrame(rows)


with tabs[7]:
    st.markdown("### 🔮 機器學習走勢預測（淺層 LightGBM × 5 種子，2010 起逐年樣本外驗證，經驗校準）")
    st.caption("方向 (漲/跌) 在樣本外幾乎不可預測 (AUC≈0.5)，本模型只做「期望報酬排序」，再把預測值對應到歷史同分位的實際上漲率與報酬區間。"
               "10/20 日視野有微弱但跨年穩定的優勢 (rank-IC 約 0.07)，5 日幾乎沒有；請把它當作機率性參考，而非買賣指令。")
    with st.spinner("預測中…"):
        FC = market_forecast.forecast(scored, SNAP)
    if "error" in FC:
        st.warning(FC["error"] + "（約需 2~3 分鐘）")
    else:
        st.info(FC["summary"])
        # ---- 隔天 / 後天 / 第三天
        st.markdown("#### 📅 隔天 / 後天 / 第三個交易日（跳過休市）")
        cols = st.columns(3)
        for i, nd in enumerate(FC.get("next_days", [])):
            diff = (nd["p_up"] or 0) - nd["base_hit"]
            tone = "偏多" if diff >= 0.04 else "偏空" if diff <= -0.04 else "中性"
            with cols[i]:
                st.metric(f"{nd['label']} {nd['date']}｜{tone}", f"{nd['level']:,.0f}", f"上漲率 {nd['p_up']:.0%} (基準 {nd['base_hit']:.0%})", delta_color="off")
                st.caption(f"20~80% 區間 {nd['level_lo']:,.0f} ~ {nd['level_hi']:,.0f}｜模型期望 {nd['pred']:+.2f}% (第 {nd['bin'] + 1}/5 分位)")
        # ---- 當日每小時
        st.markdown("#### ⏱️ 當日每小時走勢預測")
        try:
            IH = intraday.forecast(scored, SNAP)
        except Exception as e:  # noqa: BLE001
            IH = {"error": f"小時模型與目前程式版本不一致，請重新執行 python cli.py train --intraday-only（{e}）"}
        if IH.get("note"):
            st.caption("⚠️ " + IH["note"])
        if "error" in IH:
            st.warning(IH["error"])
        else:
            st.caption(f"預測日 {IH['day']}｜{'盤中即時' if IH['live'] else '開盤前，以前收為基準'}｜目前時間點 {IH['mark']}｜基準價 {IH['price']:,.2f}"
                       + (f"｜前晚夜盤台指期 {IH['night_chg_pct']:+.2f}%" if IH.get("night_chg_pct") is not None else "")
                       + f"｜模型以近 {IH['days']} 個交易日分時訓練")
            trows = []
            for tm, r in IH["targets"].items():
                m = r.get("metrics", {})
                trows.append({"到": tm, "預估指數": r["level"], "區間低": r["level_lo"], "區間高": r["level_hi"],
                              "上漲率": f"{(r.get('p_up') or 0):.0%}", "基準": f"{(r.get('base_hit') or 0):.0%}", "模型期望%": r["pred"],
                              "樣本外IC": m.get("rank_ic"), "推升": "、".join(d["name"] for d in r["drivers"]["positive"][:2]),
                              "壓抑": "、".join(d["name"] for d in r["drivers"]["negative"][:2])})
            if trows:
                tdf = pd.DataFrame(trows)
                st.dataframe(tdf, width="stretch", hide_index=True)
                fh = go.Figure()
                xs = ["現在" if IH["live"] else "前收"] + list(tdf["到"])
                fh.add_trace(go.Scatter(x=xs, y=[IH["price"]] + list(tdf["預估指數"]), mode="lines+markers", name="預估", line=dict(color="#1f77b4", width=3)))
                fh.add_trace(go.Scatter(x=xs, y=[IH["price"]] + list(tdf["區間高"]), mode="lines", line=dict(width=0), showlegend=False))
                fh.add_trace(go.Scatter(x=xs, y=[IH["price"]] + list(tdf["區間低"]), mode="lines", line=dict(width=0), fill="tonexty",
                                        fillcolor="rgba(31,119,180,0.15)", name="20~80% 區間"))
                bars = pd.DataFrame((SNAP.get("intraday") or {}).get("bars") or [])
                if IH["live"] and not bars.empty:
                    fh.add_trace(go.Scatter(x=bars["time"], y=bars["close"], mode="lines", name="今日實際", line=dict(color="#d64545", width=1.5)))
                fh.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=30), legend_orientation="h", title="當日路徑預估 (以同分位歷史平均與 20~80% 區間)")
                st.plotly_chart(fh, width="stretch")
            else:
                st.caption("今日已收盤，無剩餘時間點；下一交易日的每小時預測會在收盤資料更新後顯示。")
        st.markdown("#### 5 / 10 / 20 日")
        forecast_block({h: r for h, r in FC["horizons"].items() if h in (5, 10, 20)}, f"盤後資料 {FC['date']}（收盤 {FC['close']:,.2f}）")
        if FC.get("intraday"):
            forecast_block({h: r for h, r in FC["intraday"]["horizons"].items() if h in (5, 10, 20)}, f"盤中以現價 {FC['intraday']['price']:,.0f} ({(FC['intraday']['chg_pct'] or 0):+.2f}%) 推估")
        st.markdown("**模型樣本外表現** (逐年 walk-forward，2014~)")
        st.dataframe(metrics_table(FC["metrics"]), width="stretch", hide_index=True)
        oos_path = market_forecast.M.MODEL_DIR / "market_h10_oos.csv"
        if oos_path.exists():
            oos = pd.read_csv(oos_path)
            oos["bin"] = pd.qcut(oos["pred"].rank(method="first"), 5, labels=["最弱", "弱", "中", "強", "最強"])
            g = oos.groupby("bin", observed=True)["actual"].agg(["mean", lambda s: (s > 0).mean() * 100, "count"]).reset_index()
            g.columns = ["預測分位", "實際 10 日平均報酬%", "上漲率%", "樣本"]
            fb = go.Figure(go.Bar(x=g["預測分位"], y=g["實際 10 日平均報酬%"], marker_color=[color(v) for v in g["實際 10 日平均報酬%"]],
                                  text=[f"{h:.0f}%" for h in g["上漲率%"]], textposition="outside"))
            fb.update_layout(height=300, title="樣本外：模型預測分位 vs 實際 10 日報酬 (柱上為上漲率)", margin=dict(l=40, r=20, t=50, b=30))
            st.plotly_chart(fb, width="stretch")
        st.caption(f"模型訓練時間 {FC['trained_at'][:16]}，訓練資料至 {FC['train_end']}。重新訓練：`python cli.py train`")
    st.markdown("---")
    st.markdown(f"### 個股 {stock_id} 相對大盤預測")
    try:
        SF = stock_forecast.forecast(stock_id.strip())
    except Exception as e:  # noqa: BLE001
        SF = {"error": str(e)}
    if "error" in SF:
        st.warning(SF["error"])
    else:
        st.info(SF["summary"])
        cols = st.columns(3)
        for i, (h, r) in enumerate(SF["horizons"].items()):
            cols[i].metric(f"未來 {h} 日 跑贏大盤率", f"{r['p_up']:.0%}", f"基準 {r['base_hit']:.0%}｜期望超額 {r['pred']:+.2f}%", delta_color="off")
            cols[i].caption(f"歷史第 {r['bin'] + 1}/5 分位：平均超額 {r['hist_mean']:+.2f}%，區間 {r['q20']:+.2f}% ~ {r['q80']:+.2f}%")
        r10 = SF["horizons"][10]
        st.markdown("推升：" + "；".join(f"**{d['name']}** {d['contrib']:+.2f}" for d in r10["drivers"]["positive"][:5]))
        st.markdown("壓抑：" + "；".join(f"**{d['name']}** {d['contrib']:+.2f}" for d in r10["drivers"]["negative"][:5]))
        st.dataframe(metrics_table(SF["metrics"]), width="stretch", hide_index=True)
        st.caption("個股模型以 40 檔權值股 2018~ 訓練，目標為相對大盤的超額報酬；若樣本外 IC 接近 0，代表籌碼特徵對權值股的相對強弱沒有可驗證的預測力，請只當參考。")

# ------------------------------------------------------------------ tab 0 即時追蹤
with tabs[0]:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(gauge(RT["score"], f"盤勢即時分【{RT['label']}】"), width="stretch")
        st.info(realtime.combined_view(A["composite_smooth"], A["regime"], RT["score"], RT["label"], SNAP["phase"]))
        st.dataframe(pd.DataFrame([{"項目": p["name"], "分數": round(p["score"], 2), "權重": p["w"], "說明": p["text"]} for p in RT["parts"]]),
                     width="stretch", hide_index=True,
                     column_config={"分數": st.column_config.ProgressColumn("分數", min_value=-2, max_value=2, format="%.2f")})
        if txn:
            st.markdown(f"🌙 **夜盤台指期** {txn['last']:,.0f}（{txn['change_pct']:+.2f}%，較日盤結算 {txn['change']:+.0f} 點）→ 隔日開盤參考")
        vt = SNAP.get("vixtwn")
        if vt:
            st.markdown(f"📉 **臺指選擇權波動率指數 VIXTWN** {vt['last']:.2f}（{vt['date']} {vt['time']}，今日區間 {vt['low']:.2f}~{vt['high']:.2f}）"
                        + ("　⚠️ 恐慌區 (>30)" if vt["last"] > 30 else "　偏高 (>25)" if vt["last"] > 25 else "　自滿區 (<15)" if vt["last"] < 15 else ""))
        gq = SNAP.get("global") or []
        if gq:
            st.markdown("🌏 " + "　".join(f"{q['name']} {q['chg_pct']:+.2f}%" for q in gq if q.get("chg_pct") is not None))
        st.markdown("**今日警示**")
        today_alerts = notify.read_log(SNAP["ts"][:10], 30)
        if today_alerts:
            for line in reversed(today_alerts):
                st.markdown(f"🔔 {line}")
        else:
            st.caption("尚無警示")
    with c2:
        intra = SNAP.get("intraday") or {}
        bars = pd.DataFrame(intra.get("bars") or [])
        if not bars.empty:
            fi = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
                               subplot_titles=(f"加權指數分時 {intra.get('date', '')}", "每分鐘成交量 (張)"))
            fi.add_trace(go.Scatter(x=bars["time"], y=bars["close"], name="指數", line=dict(color=UP if (idx.get("chg") or 0) >= 0 else DOWN, width=2)), row=1, col=1)
            if idx.get("prev"):
                fi.add_hline(y=idx["prev"], line_dash="dot", line_color="#888", annotation_text="昨收", row=1, col=1)
            if SNAP.get("ma20"):
                fi.add_hline(y=SNAP["ma20"], line_dash="dash", line_color="#e0a800", annotation_text="MA20", row=1, col=1)
            if SNAP.get("ma5"):
                fi.add_hline(y=SNAP["ma5"], line_dash="dash", line_color="#17becf", annotation_text="MA5", row=1, col=1)
            fi.add_trace(go.Bar(x=bars["time"], y=bars["vol"], name="成交量", marker_color="#999"), row=2, col=1)
            fi.update_layout(height=430, margin=dict(l=40, r=40, t=40, b=20), showlegend=False)
            fi.update_xaxes(type="category", nticks=10)
            st.plotly_chart(fi, width="stretch")
        b = SNAP.get("breadth")
        if b:
            st.markdown(f"**權值股廣度** ({b['n']} 檔)：🔺 漲 {b['up']}　🔻 跌 {b['down']}　平 {b['flat']}　均 {b['avg_chg']:+.2f}%　委買/委賣 {b['bid_ask_ratio']:.2f}"
                        + (f"　台積電 {SNAP['tsmc_chg']:+.2f}%" if SNAP.get("tsmc_chg") is not None else ""))
            lc = pd.DataFrame(SNAP["large_caps"])
            fb = go.Figure(go.Bar(x=lc["name"], y=lc["chg_pct"], marker_color=[color(v) for v in lc["chg_pct"]]))
            fb.update_layout(height=260, margin=dict(l=30, r=10, t=10, b=60), yaxis_title="%")
            st.plotly_chart(fb, width="stretch")
        series = store.load_intraday(SNAP["ts"][:10])
        if not series.empty and len(series) > 2:
            fs2 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]], subplot_titles=("今日追蹤：盤勢即時分 / 期現價差 (點)",))
            fs2.add_trace(go.Scatter(x=series["ts"].str[11:16], y=series.get("rt_score"), name="盤勢分", line=dict(color="#1f77b4")))
            if "basis" in series:
                fs2.add_trace(go.Scatter(x=series["ts"].str[11:16], y=series["basis"], name="期現價差", line=dict(color="#ff7f0e")), secondary_y=True)
            fs2.update_layout(height=260, margin=dict(l=30, r=30, t=40, b=20), legend_orientation="h")
            st.plotly_chart(fs2, width="stretch")
        else:
            st.caption("盤勢分／期現價差時間序列：讓頁面自動更新或執行 `python cli.py watch` 累積後顯示。")

# ------------------------------------------------------------------ tab 1
with tabs[1]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(gauge(A["composite_smooth"], f"綜合籌碼分 (3日平滑)【{A['regime']}】"), width="stretch")
        st.caption(f"當日原始分 {A['composite']:+.1f}｜5 日動能 {A['momentum']:+.1f}｜市場狀態 **{A['state']}** (權重依狀態調整)")
        st.subheader(A["action"])
        st.write(A["detail"])
        conf_icon = {"高": "🟢", "中": "🟡", "低": "🔴"}[A["confidence"]]
        st.info(f"建議持股水位：**{A['position']}**　　信心度：{conf_icon} **{A['confidence']}** (因子一致 {A['agree_ratio']:.0%}，資料完整 {A['coverage']:.0%})")
        if A.get("turning"):
            st.warning("🔀 轉折：" + A["turning"])
        if A["bottom_signals"]:
            st.success("底部/逆勢訊號：" + "；".join(A["bottom_signals"]))
        if A["top_risks"]:
            st.warning("高檔風險：" + "；".join(A["top_risks"]))
    with c2:
        hint = market.intraday_hint(scored, rt[0] if rt else None)
        if hint:
            st.markdown(f"**盤中提示** ({hint['time']}) 現價 {hint['last']:,.2f} ({hint['ret']:+.2f}%)｜MA5 {hint['ma5']:,.0f}｜MA20 {hint['ma20']:,.0f}｜乖離 {hint['bias20']:+.2f}%")
            st.caption(" ".join(hint["messages"]))
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**主要多方理由**")
            for f in A["reasons_pos"] or []:
                st.markdown(f"🔺 **{f.name}**：{f.comment}")
            if not A["reasons_pos"]:
                st.caption("無明顯多方因子")
        with r2:
            st.markdown("**主要空方理由**")
            for f in A["reasons_neg"] or []:
                st.markdown(f"🔻 **{f.name}**：{f.comment}")
            if not A["reasons_neg"]:
                st.caption("無明顯空方因子")
        st.markdown(f"**進場檢查表** ({A['passed']}/{A['total']} 通過)")
        for name, ok in A["checklist"]:
            st.markdown(("✅ " if ok else "❔ " if ok is None else "❌ ") + name)
    # ---- 大盤買點 / 賣點訊號 (長歷史驗證)
    SG = load_signals()
    cur = SG["current"]
    st.markdown(f"#### 🎯 大盤買點／賣點訊號：**{cur['label']}**（買點強度 {cur['buy_strength']}，賣點強度 {cur['sell_strength']}，近 3 日）")
    if cur["buy_signals"] or cur["sell_signals"]:
        for s in cur["buy_signals"]:
            st.markdown(f"🟥 買點 {s['name']}（{s['days'][-1]}，歷史 10 日超額 {s['excess10']:+.2f}%，驗證 {s['valid']}）")
        for s in cur["sell_signals"]:
            st.markdown(f"🟩 賣點 {s['name']}（{s['days'][-1]}，歷史 10 日超額 {s['excess10']:+.2f}%，驗證 {s['valid']}）")
    else:
        st.caption("近 3 日無買賣點規則觸發")
    pts = SG["points"]
    fsig = go.Figure()
    fsig.add_trace(go.Scatter(x=pts["date"], y=pts["close"], name="加權指數", line=dict(color="#333", width=1.5)))
    bp, sp = pts[pts["buy_n"] > 0], pts[pts["sell_n"] > 0]
    fsig.add_trace(go.Scatter(x=bp["date"], y=bp["close"] * 0.985, mode="markers", name="買點", marker=dict(symbol="triangle-up", color=UP, size=9 + 3 * bp["buy_n"]), text=bp["signal_names"]))
    fsig.add_trace(go.Scatter(x=sp["date"], y=sp["close"] * 1.015, mode="markers", name="賣點", marker=dict(symbol="triangle-down", color=DOWN, size=9 + 3 * sp["sell_n"]), text=sp["signal_names"]))
    fsig.update_layout(height=360, title="近一年買賣點訊號（只畫長歷史驗證有效或待定的規則；滑鼠移到三角形看規則名）", legend_orientation="h", margin=dict(l=40, r=20, t=50, b=20))
    fsig.update_xaxes(type="category", nticks=12)
    st.plotly_chart(fsig, width="stretch")
    with st.expander("買賣點規則的長歷史驗證（2010 起，10 日超額報酬與 t 值；✓ = |t|≥1.5 且方向正確）"):
        st.dataframe(SG["evaluation"], width="stretch", hide_index=True)
    st.markdown("**因子明細** (分數 -2 ~ +2，紅正綠負)")
    factor_table(A["factors"])
    with st.expander("文字報告 (可複製)"):
        st.code(market_report(A), language=None)

# ------------------------------------------------------------------ tab 2
with tabs[2]:
    n = st.slider("顯示天數", 30, len(scored), min(120, len(scored)), key="ndays")
    d = scored.tail(n)
    x = d["date"]

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.3, 0.18, 0.18, 0.17, 0.17],
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("加權指數 vs 外資現貨累計淨買 (億)", "融資餘額 (億) vs 指數", "外資台指期淨未平倉 (口)",
                                        "三大法人每日買賣超 (億)", "綜合籌碼分"))
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="加權指數",
                                 increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ma20"], name="MA20", line=dict(color="#e0a800", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ma60"], name="MA60", line=dict(color="#6f42c1", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["foreign"].fillna(0).cumsum(), name="外資累計淨買", line=dict(color="#1f77b4", width=2)), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=d["margin_amt"], name="融資餘額", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["close"], name="指數", line=dict(color="#999", dash="dot")), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=x, y=d["fut_foreign_net_oi"], name="外資期貨淨OI", marker_color=[color(v) for v in d["fut_foreign_net_oi"]]), row=3, col=1)
    if d["maint_ratio"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=d["maint_ratio"], name="融資維持率%", line=dict(color="#17becf")), row=3, col=1, secondary_y=True)
    for who, cl in (("foreign", "#1f77b4"), ("trust", "#2ca02c"), ("dealer", "#9467bd")):
        fig.add_trace(go.Bar(x=x, y=d[who], name={"foreign": "外資", "trust": "投信", "dealer": "自營商"}[who], marker_color=cl), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=d["composite"], name="綜合分", marker_color=[color(v) for v in d["composite"]]), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["composite_smooth"], name="3日平滑", line=dict(color="#333", width=2)), row=5, col=1)
    fig.update_layout(height=1100, barmode="relative", xaxis_rangeslider_visible=False, legend_orientation="h", margin=dict(l=40, r=40, t=60, b=20))
    fig.update_xaxes(type="category", showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=5, col=1, nticks=12)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("八大行庫每日買賣超 (億)", "融券餘額 (張) / 借券賣出增減 (張)"),
                           specs=[[{}], [{"secondary_y": True}]])
        f2.add_trace(go.Bar(x=x, y=d["gov8_net"], name="八大行庫", marker_color=[color(v) for v in d["gov8_net"]]), row=1, col=1)
        f2.add_trace(go.Scatter(x=x, y=d["short_lots"], name="融券餘額", line=dict(color="#2ca02c")), row=2, col=1)
        f2.add_trace(go.Bar(x=x, y=d["sbl_chg"], name="借券賣出增減", marker_color="#bbb", opacity=0.6), row=2, col=1, secondary_y=True)
        f2.update_layout(height=450, margin=dict(l=40, r=40, t=40, b=20), legend_orientation="h")
        f2.update_xaxes(type="category", nticks=8)
        st.plotly_chart(f2, width="stretch")
    with c2:
        f3 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("選擇權 P/C 比 (OI, %)", "投信買賣超累計 (億)"))
        f3.add_trace(go.Scatter(x=x, y=d["pcr_oi"], name="P/C OI%", line=dict(color="#d62728")), row=1, col=1)
        f3.add_hline(y=100, line_dash="dot", row=1, col=1)
        f3.add_trace(go.Scatter(x=x, y=d["trust"].fillna(0).cumsum(), name="投信累計", line=dict(color="#2ca02c")), row=2, col=1)
        f3.update_layout(height=450, margin=dict(l=40, r=40, t=40, b=20), legend_orientation="h")
        f3.update_xaxes(type="category", nticks=8)
        st.plotly_chart(f3, width="stretch")

# ------------------------------------------------------------------ tab 3
with tabs[3]:
    st.markdown(f"**八大公股行庫 (台銀、土銀、合庫、一銀、華南、彰銀、兆豐、台企銀) 買賣超**　5日 {last['gov8_5d']:+,.1f} 億｜20日 {last['gov8_20d']:+,.1f} 億")
    if wg is not None and not wg["banks"].empty:
        b = wg["banks"].tail(30).copy()
        money_cols = [c for c in b.columns if c.startswith("money_")]
        fb = go.Figure()
        for c in money_cols:
            fb.add_trace(go.Bar(x=b["date"], y=b[c] / 1e4, name=c.replace("money_", "")))
        fb.update_layout(barmode="relative", height=380, title="各行庫每日買賣超金額 (億，玩股網)", legend_orientation="h", margin=dict(l=40, r=20, t=50, b=20))
        st.plotly_chart(fb, width="stretch")
        show = b[["date"] + money_cols + ["gov8_net", "gov8_lots"]].copy()
        show.columns = ["日期"] + [c.replace("money_", "") for c in money_cols] + ["合計(萬)", "合計(張)"]
        st.dataframe(show.sort_values("日期", ascending=False).round(0), width="stretch", hide_index=True)
    else:
        st.caption("各行庫明細需啟用玩股網 (Playwright)。下方為 HiStock 全市場合計。")
        g = scored.tail(60)
        st.plotly_chart(go.Figure(go.Bar(x=g["date"], y=g["gov8_net"], marker_color=[color(v) for v in g["gov8_net"]])).update_layout(height=300, title="八大行庫每日買賣超 (億)"), width="stretch")
    if rank:
        st.markdown(f"**{rank['date']} 八大行庫買超 / 賣超排行 (萬元)**")
        c1, c2 = st.columns(2)
        c1.dataframe(rank["buy"].head(20), width="stretch", hide_index=True)
        c2.dataframe(rank["sell"].head(20), width="stretch", hide_index=True)

# ------------------------------------------------------------------ tab 4
with tabs[4]:
    sid = stock_id.strip()
    if sid:
        try:
            S = load_stock(sid, A["regime"], A["state"], float(last["ret20"]) if pd.notna(last["ret20"]) else None)
        except Exception as e:  # noqa: BLE001
            st.error(f"個股 {sid} 分析失敗：{e}")
            S = None
        if S:
            q = S["quote"]
            c = st.columns(5)
            c[0].metric(f"{S['name']} ({sid})", f"{q['last']:,.2f}" if q else f"{S['close']:,.2f}",
                        f"{q.get('change', 0):+,.2f} ({q.get('change_pct', 0):+.2f}%) {q['time']}" if q else "", delta_color="inverse")
            c[1].metric("個股籌碼分", f"{S['composite_raw']:+.1f}", "")
            c[2].metric("含大盤環境", f"{S['composite']:+.1f}", f"大盤調整 {S['market_adj']:+.0f}", delta_color="off")
            c[3].metric("判讀", S["regime"], "")
            c[4].markdown(f"### {S['action']}")
            if S["tags"]:
                st.write("標籤：", "　".join(f"`{t}`" for t in S["tags"]))
            factor_table(S["factors"])
            fr = S["frame"].tail(120)
            fs = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.45, 0.3, 0.25],
                               specs=[[{"secondary_y": True}], [{}], [{"secondary_y": True}]],
                               subplot_titles=("股價 vs 外資/投信累計買賣超 (張)", "三大法人每日買賣超 (張)", "融資餘額 (張) / 八大行庫 (萬元)"))
            fs.add_trace(go.Candlestick(x=fr["date"], open=fr["open"], high=fr["high"], low=fr["low"], close=fr["close"], name="股價",
                                        increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
            fs.add_trace(go.Scatter(x=fr["date"], y=fr["ma20"], name="MA20", line=dict(color="#e0a800", width=1)), row=1, col=1)
            fs.add_trace(go.Scatter(x=fr["date"], y=fr["foreign"].fillna(0).cumsum(), name="外資累計", line=dict(color="#1f77b4")), row=1, col=1, secondary_y=True)
            fs.add_trace(go.Scatter(x=fr["date"], y=fr["trust"].fillna(0).cumsum(), name="投信累計", line=dict(color="#2ca02c")), row=1, col=1, secondary_y=True)
            for who, cl in (("foreign", "#1f77b4"), ("trust", "#2ca02c"), ("dealer", "#9467bd")):
                fs.add_trace(go.Bar(x=fr["date"], y=fr[who], name={"foreign": "外資", "trust": "投信", "dealer": "自營商"}[who], marker_color=cl), row=2, col=1)
            fs.add_trace(go.Scatter(x=fr["date"], y=fr["margin_lots"], name="融資餘額", line=dict(color="#ff7f0e")), row=3, col=1)
            fs.add_trace(go.Bar(x=fr["date"], y=fr["gov8_net"], name="八大行庫(萬)", marker_color=[color(v) for v in fr["gov8_net"]], opacity=0.7), row=3, col=1, secondary_y=True)
            fs.update_layout(height=800, barmode="relative", xaxis_rangeslider_visible=False, legend_orientation="h", margin=dict(l=40, r=40, t=50, b=20))
            fs.update_xaxes(type="category", showticklabels=False)
            fs.update_xaxes(showticklabels=True, row=3, col=1, nticks=10)
            st.plotly_chart(fs, width="stretch")
            with st.expander("券商分點 (需 FinMind 贊助等級 Token)"):
                br = stock.broker_report(sid, S["date"])
                if br is None:
                    st.caption("未設定 Token 或無權限。免費替代：玩股網/嗨投資的主力券商頁面。")
                elif br.empty:
                    st.caption("無資料")
                else:
                    c1, c2 = st.columns(2)
                    c1.dataframe(br.head(15), hide_index=True, width="stretch")
                    c2.dataframe(br.tail(15).sort_values("net"), hide_index=True, width="stretch")
            with st.expander("文字報告"):
                st.code(stock_report(S), language=None)

# ------------------------------------------------------------------ tab 5
with tabs[5]:
    rep = backtest.load_report()
    st.markdown("### 長歷史驗證 (2010 起，FinMind)")
    if rep:
        st.caption(f"產生時間 {rep['generated']}｜{rep['start']} ~ {rep['end']}，{rep['rows']} 個交易日，未來 {rep['horizon']} 日報酬。執行 `python cli.py optimize` 可更新。")
        mult = market.tuned_multipliers()
        wrow = [{"因子": market.NAMES[k], "基本權重": market.WEIGHTS[k], "倍數": v["multiplier"], "生效權重(盤整)": round(market.WEIGHTS[k] * v["multiplier"], 2), "依據": v["reason"]}
                for k, v in rep["weights"].items()]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**因子 IC (與未來報酬的排序相關) 與跨年一致性**")
            st.dataframe(pd.DataFrame(rep["ic_overall"]), width="stretch", hide_index=True)
            st.markdown("**逐年樣本外 IC**（狀態自適應 = 目前模型）")
            st.dataframe(pd.DataFrame(rep["walk_forward"]), width="stretch", hide_index=True)
        with c2:
            st.markdown("**權重倍數** " + ("（已套用）" if mult else "（未套用）"))
            st.dataframe(pd.DataFrame(wrow), width="stretch", hide_index=True)
            st.markdown("**事件研究**（訊號發生後 vs 全體基準，|t| ≥ 2 較可信）")
            evd = pd.DataFrame(rep["events"])[["訊號", "樣本數", "10日均報酬%", "10日勝率%", "20日均報酬%", "10日超額%", "t值(10日)"]]
            st.dataframe(evd, width="stretch", hide_index=True, height=420)
    else:
        st.info("尚未執行長歷史回測。執行 `python cli.py optimize` 後這裡會顯示因子 IC、事件研究、逐年樣本外與權重倍數。")
    st.markdown("### 近一年 (含全部因子)")
    st.markdown("以近一年資料驗證：**綜合籌碼分** 與 **未來 5/10/20 日指數報酬** 的關係。")
    ev = market.evaluate(scored)
    if ev.get("corr"):
        st.write({f"未來{h}日 相關": v for h, v in ev["corr"].items()}, {f"未來{h}日 全體基準": v for h, v in ev["baseline"].items()})
        cs = st.columns(len(ev["buckets"]))
        for i, (h, tbl) in enumerate(ev["buckets"].items()):
            cs[i].markdown(f"**未來 {h} 日**")
            cs[i].dataframe(tbl, width="stretch")
        fr = ev["frame"].dropna(subset=["fwd5"])
        st.plotly_chart(go.Figure(go.Scatter(x=fr["composite"], y=fr["fwd5"], mode="markers", marker=dict(color=[color(v) for v in fr["fwd5"]], size=6),
                                             text=fr["date"])).update_layout(height=380, title="綜合分 vs 未來 5 日報酬 (%)", xaxis_title="綜合籌碼分", yaxis_title="未來5日報酬%"),
                        width="stretch")
    else:
        st.caption("歷史資料不足")
    sw = market.suggest_weights(scored)
    if sw:
        st.markdown(f"**資料驅動權重參考** (未來 {sw['horizon']} 日報酬，前 {sw['n_train']} 日訓練 / 後 {sw['n_test']} 日測試)")
        st.write(f"樣本外排序相關：預設權重 **{sw['oos_corr_default']:+.3f}**　擬合權重 **{sw['oos_corr_fitted']:+.3f}**　(樣本內 {sw['is_corr_default']:+.3f} / {sw['is_corr_fitted']:+.3f})")
        st.dataframe(sw["coef"], width="stretch", hide_index=True)
        st.caption("係數為正代表該因子加分在歷史上對應較高的未來報酬；為負代表歷史上是反指標 (例如外資大賣後常反彈)。樣本僅一年，請當作調整權重的參考，不要直接套用。")
    st.caption("注意：這是簡單的同期關聯統計，不含交易成本，且部分指標 (PCR/維持率/大額交易人) 歷史需由本程式每日累積後才會完整。")

# ------------------------------------------------------------------ tab 6
with tabs[6]:
    rows = [{"來源": k, **(v if isinstance(v, dict) else {"status": str(v)})} for k, v in meta.items() if not k.startswith("_")]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("""
| 資料 | 來源 | 公布時間 (交易日) |
|---|---|---|
| 即時指數 / 報價 | TWSE MIS | 盤中每 5 秒 |
| 三大法人買賣金額 | TWSE BFI82U / FinMind | 約 15:00 |
| 個股法人買賣超 | TWSE T86 / FinMind | 約 16:00 |
| 融資融券餘額 | TWSE MI_MARGN / FinMind | 約 21:00 |
| 借券賣出餘額 | TWSE TWT93U / 玩股網 | 約 21:00 |
| 期貨法人未平倉 / PCR / 大額交易人 | TAIFEX OpenAPI / FinMind | 約 15:00 (OpenAPI 隔日) |
| 八大行庫買賣超 | HiStock / 玩股網 | 約 17:00 |
| 大盤融資維持率 | 玩股網 (Playwright) | 約 21:00 |
| 券商分點 | FinMind (贊助) | 約 17:00 |
""")
    st.download_button("下載合併後日資料 (CSV)", scored.to_csv(index=False).encode("utf-8-sig"), f"taiex_chip_{A['date']}.csv")
