"""把操盤台研究 (desk t1~t6，2026-09-25，含對抗式驗證修正) 的結果整理成靜態證據檔 data/models/desk_evidence.json。

用法：python tools/desk_evidence.py <研究目錄 desk/>
研究目錄是一次性的 scratch；這個 JSON 才是 repo 內保存的版本，chip/analysis/desk.py 與 App 操盤台都讀它。
重跑研究 (每季) 後再執行一次即可更新。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models" / "desk_evidence.json"
KS = ("1", "2", "3", "5", "10", "20")
QS = ("low10", "low20", "high80", "high90")


def r(v, n=4):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main(src: str) -> None:
    D = Path(src)
    ev: dict = {"asof": "2026-09-24", "built": "2026-09-25",
                "source": "操盤台研究 t1 均線 / t2 高低點 / t3 選擇權區間 / t4 八大與價位帶 / t5 停損與資金防守 / t6 00981A·權證·2330；每條研究線均經獨立對抗式驗證，數字已套用驗證修正"}

    # ---------------- t3 選擇權區間乘數 (滾動 750 日；指數另附 2017~ 擴張版 = 保守)
    cr = load(D / "t3_options" / "current_readout.json")
    idx = {k: {q: {"m": r(cr["index"]["levels"][k][q]["m"]), "m_exp": r(cr["index"]["levels"][k][q].get("m_expanding_2017"))} for q in QS}
           for k in KS if k in cr["index"]["levels"]}
    assets = {}
    for sid, a in cr["assets"].items():
        assets[sid] = {"beta": r(a.get("beta")), "idio": r(a.get("idio_daily_pct")), "method": a.get("method"),
                       "m": {k: {q: r((a["levels"][k].get(q) or {}).get("m_asset")) for q in QS} for k in KS if k in a.get("levels", {})}}
    ev["range"] = {"ks": [int(k) for k in KS], "index": idx, "assets": assets,
                   "evidence": "iv 當 sigma 的 k 日路徑高低分位：pinball 比 App ATR+EWMA 低 6.7% [−9.9%,−3.9%]、比 rv20 低 12%；各標的校準後比自身 ATR+EWMA 低 4~7.5% (2330 k=20 不顯著)。乘數每日盤後以滾動 750 日重算 (只用已實現的 k 日窗口，不偷看)，並監控近 250 日觸及率；抓不到資料時退回 2026-09-24 凍結值並標示。指數乘數改以加權自身高低點校準 (台指期校準值保留作對照)；另附 2017~ 擴張版 (保守)。滾動窗口含 2023~26 多頭漂移 → 下檔可能偏淺，看監控警示。k 日窗口若遇除息，低點扣全額股利、高點依除息後天數比例扣 (看盤價口徑)，含息價另列；00981A 未校準，沿用台指期乘數映射。嘗試過的改良 (500/1000 日/擴張窗口、ACI 自適應、依波動區間條件化、σ 混合) 樣本外都沒有勝過滾動 750 日。",
                   "exec_note": "把 low 分位當限價買單等回檔，平均比隔日開盤直接買貴 (0050 k=5 low20 +0.45% [+0.19,+0.68])；high 分位當限價賣單有利 (+0.35%)，但來自多頭漂移。",
                   "walls": "最近到期/月選的 Put/Call OI 牆沒有支撐/壓力效果 (被穿越機率 vs 同距離對照 put −0.3pp [−2.7,+2.2]、call −1.0pp [−3.7,+1.7])；max pain 無可用磁吸 (方向命中 49.1%，當點預測比現價差)。僅供描述。",
                   "chips": "pcr、大額交易人、偏斜、期限結構對未來 5/10 日最大回檔沒有超出 IV 的增量預測力；外資選擇權方向只有約 3 年資料 (2023-10~)，IV 標準化後 IC 0.10~0.12 但 t 只有 1.0~1.2，無法判定。"}

    # ---------------- t2 波段 / 條件回撤 / 觸及基準率 / 分批承接
    zz = load(D / "t2_pivots" / "s01_zigzag.json")
    ev["zigzag"] = {}
    for sid, ths in zz.items():
        ev["zigzag"][sid] = {}
        for th, x in ths.items():
            db = x.get("down_bull") or {}
            ev["zigzag"][sid][th] = {"n": (db.get("amp") or {}).get("n"), "amp_p50": r((db.get("amp") or {}).get("p50")), "amp_p75": r((db.get("amp") or {}).get("p25")),
                                     "amp_p90": r((db.get("amp") or {}).get("p10")), "days_p50": r((db.get("days") or {}).get("p50"), 1), "days_p90": r((db.get("days") or {}).get("p90"), 1),
                                     "recover_p50": r((db.get("recover_days") or {}).get("p50"), 1), "exceed": x.get("bull_down_exceed"), "swings_per_year": r(x.get("n_swings_per_year"), 2)}
    cd = load(D / "t2_pivots" / "s03_cond_dd.json")
    ev["cond_dd"] = {}
    UNDERCOVER_FROM = {"TWII": 0.20, "0050": 0.20, "2330L": 0.15, "SYN2X": 0.30}   # verify_pivots：這些層級以上走動式覆蓋率不足 (80% 邊界 0.56~0.79)
    for sid, e in cd["episodes"].items():
        ev["cond_dd"][sid] = {"period": e.get("period"), "swing_R": e.get("swing_R"), "levels": {}}
        for d, x in (e.get("levels") or {}).items():
            cov = x.get("wf_coverage_swing") or {}
            ev["cond_dd"][sid]["levels"][d] = {"n": x.get("n_events"), "add_q50": r((x.get("add") or {}).get("q50")), "add_q80": r((x.get("add") or {}).get("q80")),
                                               "add_q90": r((x.get("add") or {}).get("q90")), "days_q50": r((x.get("days_to_trough") or {}).get("q50"), 1),
                                               "rec_days_q50": r(((x.get("to_peak") or {}).get("rec_days") or {}).get("q50"), 1),
                                               "cov80": cov.get("cov80"), "cov90": cov.get("cov90"),
                                               "undercovered": bool(cov.get("cov80") is None or cov["cov80"] < 0.72 or float(d) >= UNDERCOVER_FROM.get(sid, 9.0) - 1e-9)}
    ev["touch"] = {sid: {lv: {h: (x.get(h) or {}).get("p") for h in ("N20", "N60", "N120", "N250")} for lv, x in (t.get("levels") or {}).items()}
                   for sid, t in cd["touch_from_near_high"].items()}
    ld = load(D / "t2_pivots" / "s04_ladder.json")
    ev["ladder"] = {}
    for sid, x in ld["A"].items():
        h = x.get("H250") or {}
        ev["ladder"][sid] = {k: {"mean_rel": r((h.get(k) or {}).get("mean_rel")), "p_beat": r((h.get(k) or {}).get("p_beat_lump"), 3),
                                 "ci250": [r(v) for v in ((h.get(k) or {}).get("ci95_block250") or [])], "worst": (h.get(k) or {}).get("worst_year")}
                             for k in ("LADDER_60", "HYBRID_120", "ANCHOR_120", "DCA12", "ORACLE60") if k in h}
    ev["ladder_cluster_ci"] = {"0050": {"LADDER_60": [-0.0258, 0.0002], "HYBRID_120": [-0.0341, 0.0008]}, "2330": {"LADDER_60": [-0.0471, -0.0103]},
                               "00631L": {"LADDER_60": [-0.0558, 0.0121]}}
    ev["ladder_levels"] = {"1x": [0.05, 0.08, 0.12], "2330": [0.08, 0.12, 0.18], "2x": [0.10, 0.16, 0.24]}

    # ---------------- t1 均線狀態 (20 日；描述) 與風險
    st = load(D / "t1_ma" / "s1_states.json")
    ev["ma_states"] = {}
    for sid, a in st.items():
        key = "TWII" if sid.startswith("TWII") else sid
        h20 = ((a.get("horizons") or {}).get("20") or {})
        ev["ma_states"][key] = {"uncond_mean": r(h20.get("uncond_mean")), "uncond_win": r(h20.get("uncond_win"), 3), "states": {}}
        for name, s in (h20.get("states") or {}).items():
            wf = s.get("wf") or {}
            ev["ma_states"][key]["states"][name] = {"n": s.get("n"), "mean": r(s.get("mean")), "win": r(s.get("win"), 3), "excess": r(s.get("excess")),
                                                    "ci": [r(v) for v in (s.get("excess_ci") or [])], "oos": r(wf.get("pooled_signed_excess")),
                                                    "oos_ci": [r(v) for v in (wf.get("pooled_ci") or [])]}
    rk = load(D / "t1_ma" / "s1b_risk.json")
    ev["ma_risk"] = {("TWII" if sid.startswith("TWII") else sid): {"uncond": {k: r(v) for k, v in (a.get("uncond") or {}).items()},
                                                                   "states": {n: {"rv_ratio": r(s.get("rv_ratio"), 3), "p_tail": r(s.get("p_tail"), 3), "mdd20": r(s.get("mdd20"))}
                                                                              for n, s in (a.get("states") or {}).items()}}
                     for sid, a in rk.items()}
    ev["ma_notes"] = [
        "均線狀態 (站上/跌破、排列、斜率、交叉、乖離) 對未來 5/10/20 日平均報酬沒有穩健預測力：870 個檢定只有 9 個通過，與隨機平移的中位數 11 相當 (年內去均值評估)。",
        "破五日線全出等 MA5 系規則明顯傷績效：0050 年化 −17.5% [−24.5,−10.7]、2330 −40.5%、正2 −23~−25% (成本拖累 10~16%/年，扣成本前也輸)。",
        "MA60 加 2% 緩衝 (a2) 是回撤控制：0050 全樣本 15.5%/−22.1% vs 持有 15.4%/−52.3%；OOS 2012~ 16.8%/−22.1% vs 20.3%/−33.8%。不是報酬增強。",
        "跌破 MA60 時未來波動較高，但控制近期實現波動後只剩 ×1.06~1.11 (正2 不顯著) → 風險主要看 iv/rv。",
        "多頭排列中回檔觸及 MA10/30/60 不是可靠支撐 (90 個檢定與多重檢定一致)。",
        "均線訊號一律用含息價；00981A 09-16 除息造成看盤價 MA10<MA30 的假死亡交叉。",
    ]
    ev["overlay"] = {
        "0050": {"rule": "a2", "b": 0.02, "level": "weak", "text": "收盤 < MA60×0.98 → 次日開盤出；收盤 > MA60 → 次日開盤回補",
                 "hist": "全樣本 15.5%/−22.1% vs 持有 15.4%/−52.3% (差距主要來自 2008)；OOS 2012~ 16.8%/−22.1% vs 20.3%/−33.8%；走動式 2011~ 最大回撤改善 +11.7pp [−6.3,+24.8]、每年少約 3pp；參數實質由 2008 單一崩盤選出",
                 "grade": "弱：降回撤 (改善的信賴區間跨 0)、不增報酬"},
        "00631L": {"rule": "a", "b": 0.0, "level": "none", "text": "收盤 < MA60 → 出；站回 MA60 → 回補 (另列追蹤 15%)",
                   "hist": "6 次大回檔中 5 次 MA60 損失 ≤ MA30/60；2024-07→2025-04 反覆洗出仍 −50%；實際正2 樣本 (2015~) 套 MA60 年化代價約 −10.9%", "grade": "無：未通過驗證，只作回撤參考"},
        "00663L": {"rule": "a", "b": 0.0, "level": "none", "text": "收盤 < MA60 → 出；站回 MA60 → 回補 (另列追蹤 15%)",
                   "hist": "與 00631L 相同型態；實際樣本 (2016-10~) 套 MA60 年化代價約 −17.1% [−34.4,+1.4]", "grade": "無：未通過驗證，只作回撤參考"},
        "2330": {"rule": None, "level": "off", "text": "1 倍個股所有停損 overlay 走動式都讓 Sharpe 變差；跨族走動式在 2023~26 選『不停損』", "hist": "ATR×6 吊燈 24.5%/−34.6% vs 持有 28.9%/−44.8%", "grade": "不啟用：研究預設不停損"},
        "00981A": {"rule": None, "b": 0.02, "level": "none", "text": "只有 328 日，無回測；僅列 MA20×0.97、MA60×0.98 參考價", "grade": "無：只描述 (無回測)"},
    }

    # ---------------- t4 價位帶 / 八大
    ev["zones_evidence"] = {"HVN": "無 (HVN60 回落 ATT −0.03pp [−3.0,+3.1]；HVN120 反向效果加年份層後不顯著)", "前波": "無 (+0.8pp [−1.9,+3.3])",
                            "60日極值": "弱 (排除最近 5 日的 60 日高：碰到前停頓 +2.8pp [−1.2,+6.1]，碰到後不反轉)", "整數": "弱 (排除槓桿 ETF 後 CI 含 0)",
                            "MA": "無 (MA20 +2.1pp、MA60 +1.3pp，CI 含 0)", "rule": "『接近壓力就減碼/空手』走動式扣成本：0050 −10.2pp/年 [−15.1,−5.3]、2330 −19.6pp、00631L −16.4pp、00663L −22.4pp"}
    ev["gov8_evidence"] = ["八大行庫在各標的都是跌日買、漲日賣 (跌日買超率 0050 90%、2330 96%、00631L 80%、00981A 93%；00663L 53% 接近擲硬幣)。",
                           "只有 0050 與 2330 有淨累積 (單向度 0.31 / 0.27)；00631L 幾乎完全來回 (0.01)；00981A 淨賣 71% 來自華南永昌單一行庫 (疑似 ETF 參與券商/造市，非護盤)。",
                           "控制當日與 5 日報酬後，對次日開盤起的報酬沒有穩健預測力 (不重疊樣本 t −1.1~+0.8)。",
                           "成本是『自首筆歸檔日起的流量成本』，不是持倉成本；ETF 的 FIFO 常超賣 (窗外舊部位不可見)。"]

    # ---------------- t5 停損 / 資金防守 / 槓桿 ETF / 組合
    ev["risk"] = {
        "stops": ["1 倍標的 (0050、2330) 沒有任何停損 overlay 在走動式 OOS 勝過持有；0050 ATR×3 吊燈顯著變差 [−12.4,−0.8]。",
                  "停損出場 66~100% 是被洗出 (再進場價高於出場價)；價值全部來自少數崩盤 (2020-03、2022、2025-04)，代價在多頭年。",
                  "『固定 % 自進場價』停損長抱時失效 (漲上去後停損價遠在下方)，改用追蹤式 (自最高收盤)。",
                  "正2 的 MA20×0.95 是看過 OOS 才挑的族；跨族走動式 00631L 31.8%/−51.7% vs 持有 36.9%/−55.2% → 只作回撤參考。"],
        "defense": ["波動目標化 (目標 15%/20%，σ = iv21 × 波動比) 降回撤主要因為平均曝險變低；對同曝險固定比例 ΔSharpe ≈ 0 (rv60 版顯著較差)。",
                    "回撤預算 (−10% 減半、−15% 降到核心、站回 MA60 回滿) 只在 2008 型慢熊明顯有效；走動式 2011~ 每年少 3.2pp，選到的核心是 0 (完全出場)。槓桿 ETF 版 (−20%/−30%) 走動式每年少約 10pp，最大回撤只改善約 0~3.5pp。",
                    "部位大小 = 帳戶 × 單筆風險 R ÷ (進場價 − 停損價)；停損在隔日開盤執行，跳空會讓實際損失超過 R。"],
        "letf": ["00631L ≠ 2×0050：日 β 1.89，2024~26 實際曝險約 0.94×0050 + 1.05×加權 (係數跨期不穩)；0050 大勝加權的年份 00631L 會明顯落後 2×0050。",
                 "槓桿耗損 ≈ −σ²/年 (σ 為標的年化波動)；持有 h 日要贏 2×標的，標的報酬需大於約 σ√h。",
                 "00631L 與 00663L 報酬/風險幾乎相同 (2016-07~ CAGR 43.4% vs 43.4%、MDD −55.1% vs −55.2%、日相關 0.984)；00631L 成交值約 9 倍 → 流動性較佳。"],
        "portfolio": ["50% 00631L + 50% 現金 (約 0.94 倍) 沒有勝過 100% 0050：20.2%/−30.2% vs 21.2%/−33.8%，ΔSharpe −0.02。",
                      "0050 70% + 00631L 30% 只是放大同一個 Sharpe (26.4%/−39.2%，ΔSharpe 0.00)；2008 型壓力 MDD 約 −63%。",
                      "加入 2330 的組合 Sharpe 較高是後見之明，不能當配置證據。"],
    }
    # ---------------- t6
    ev["special"] = {
        "00981A": ["報酬面是高 beta 大盤代理：對 0050 beta 1.14 [1.01,1.26]、對加權 1.27；滾動 beta 樣本外 alpha +0.5%/年 (CI ±45%，1.28 年樣本無法辨識)。",
                   "0050 跌 >2% 的 18 天，00981A 平均跌 1.32 倍；漲 >2% 的 39 天為 1.05 倍 (描述性；換基準/週頻後不顯著)。",
                   "持股集中 AI 供應鏈：前十大 67.5%，台積電僅 9.8% (0050 為 56%) → 相當於押注台積電以外的科技股。",
                   "上市以來 3 次配息 (0.41、0.63、0.63)，最新一季年化約 8% (只有 3 季，不能當穩定殖利率)。"],
        "warrant": ["造市 IV 幾乎固定約 60% (上市至 2026-09-24)，86% 的日子高於之後 20 日實現波動 (不重疊有效樣本約 5)；買方做 delta 中性上市以來約 −46%。",
                    "零漂移下持有到期期望值為負 (以 20 日、60 日、EWMA 任一實現波動皆然；即時數字見上方「到期機率與期望值」)；只有假設強多頭漂移才轉正。",
                    "剩餘天數越少，時間價值衰減越快 (00981A 不動時的逐日耗損見上方敏感度區)。定位是短線槓桿工具，不適合持有。"],
        "2330": ["加權指數權重約 41% (2026-08-31)、0050 約 56% (2026-09-24)；00981A 權重見 00981A 分頁的即時持股。",
                 "ADR 溢價的日變化可預估 2330 次日開盤跳空 (樣本外 R² 約 0.30)，但從開盤起對 2330 相對 0050 無可交易預測力 (毛超額 < 成本 0.94%)。",
                 "月營收公布後只有隔夜跳空有超額 (2017 起已不顯著)；除息日開盤溢價扣成本與股利稅後約 0；相對加權強弱沒有可交易週期。"],
    }
    OUT.write_text(json.dumps(ev, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1])
