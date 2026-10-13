"""挖寶雷達 / 飆股雷達 雲端每日掃描 + 永久追蹤帳本 (2026-10-05)。

目的：App 的掃描只在裝置開著時跑、追蹤紀錄只存在各裝置 localStorage (加密同步，雲端讀不到)，
所以沒開 App 的日子沒有紀錄、換裝置看不到完整歷史。這裡在 GitHub Actions 盤後用「和 App 相同的模型與候選規則」
每天掃一次並永久歸檔，隔天起逐檔對帳，統計真正的上線成績 (依等級/年月)，失準時警示。

與 App (learning.js / app.js) 的對應：
- 候選池：TWSE 全市場盤後表 (MI_INDEX ALLBUT0999，同 App 13:35 後的 after-close 來源) + BWIBBU_ALL 本益比/淨值比 →
  scoring.js screen() 的 power 公式 (治學倍率 treasureMult 固定 1：那是各裝置自己的統計)，取前 80 並排除 ETF (代號 00 開頭) → 挖寶看前 40、飆股看前 80。
- 特徵：treasure.features (與 App tmFeatures 同款)；日 K 用 Yahoo 2 年 (Worker /idxh，失敗改直連)，D 日那根以盤後表取代 (含成交值)；大盤用 Yahoo ^TWII。
- 分級：p ≥ th_A 且大盤月線乖離 < 0 → A (≥ th_Aplus → A+)；高分但大盤在月線上 → B+；其餘 B。飆股 = 前 80 中 ps 前 3 名。
- 記錄：挖寶每天最多 6 檔 (池內依 p 排序、pct < 9.4)。A/A+ (r6 10-07 起，同回測規則)：同檔 30 個日曆日內沒記過 (任何等級) 就記，不受前 20 / 追蹤上限限制；
  B/B+ (舊規則)：池內前 20、同檔 60 日內未命中不重複、追蹤中不重複、追蹤中不超過 40。每筆標 rec_rule (A30 / B20；之前的列沒有)。飆股前 3 名 (同檔 28 日內不重複)。
- 結案 (改用回測定義，App 的 30 日曆天/不分先後 與回測不同，所以兩邊成績對不上)：
  挖寶 21 交易日：收盤峰值 ≥ +6% 或相對大盤任一日 ≥ +4pt (先盤中低點 ≤ −8% 且當時峰值 < 6% 為停損未命中)，結案收盤 > 0 才算命中；曾達標但結案 ≤ 0 為「回落」。
  飆股 20 交易日：最高價先到 +20% 且之前最低價未破 −10% 為「飆」；出場規則：漲 10% 後自最高點回落 8% 出場，否則第 20 日收盤。
輸出 data/treasure_live.json：{asof, scan{date, treasure[], surge[], pool_n}, ledger{treasure[], surge[]}, stats, alerts, model{trained_at, th_A, th_Aplus}}。
歸檔跨次累積：Pages 上一版 ∪ 本機 data/cache (同 gov8 的保險機制)；縮水保護。

pr2 (2026-10-05，LC-04 pass / T5-01~03 / T5-missed / LC-05~06 fail)：
- A/A+ 當訊號 (role=signal)、B/B+ 當描述 (role=descriptive；回測即無選股力：B 39% ≈ 同日池 40%，上線 12~22% 反映 7 月大盤 −10.7%)。
- A∪A+ 的預期命中改為 55~57% (回測 A 0.548 / A+ 0.638；OOS 分數器重掃回填 0.568)，不是帳本的 64%/75% (分數器看過答案的回填、門檻為樣本內)；
  大半優勢來自「大盤月線下的日子」(同日池 52%)，等級內 p 高低與命中無關 (Spearman 0.05) → 警報只寫「模型分」，不寫「命中機率」。
  10-07 (radar5 時點正確回測)：預期改 44~50% (舊值用今天的 170 檔名單回測，有後見之明)；分數分 5 組有小幅單調差異 (34%→66%，走動式 3/4 年較好) → 附「同分數組歷史」當參考 (RADAR_DAY0)。
  r6 (10-07，訓練窗與正式相同 2020 起；radar5 用了 2018 起)：預期改 49~54% (A+ 65%、A 42%)；A/A+ 帳本記錄規則改同回測 (30 日內未記過就記，不受前 20 / 追蹤上限)，警報 A+ 與 A 分開寫。
- 帳本誠實化：每筆記 model_ver / th_A / th_Aplus / mkt_bias20 / scan_src (live|backfill)；停損列也追到第 21 日補 fin21 (cur 仍是結案日報酬)；
  stats 分 by_source (真實發布 vs 回填)、signal (A∪A+ 主數字)；漂移只對訊號級且真實發布 ≥15 筆判定，B/B+ 不再產生「失準」。
- 門檻 th_A 0.5062 / th_Aplus 0.6362、飆股 th_top10 0.3977 與前 3 名維持 (LC-05/06)；模型檔不動、無重訓。

2026-10-13 帳本正確性 (審查 BE-1~3)：
- BE-1 日 K 對齊證交所交易日曆 (TradeCal = FMTQIK 逐月)：去掉 Yahoo 幽靈日、Yahoo 缺的交易日用當天 MI_INDEX 補 → 評分 (score_one) 與對帳 (evaluate) 的
  「第幾天 / 第 21 日 / 停損日 / >11% 跳動濾網」都以真實交易日計；已結案列一次性核對 (calv)，標記不一致的存 cal_orig 後重算 (cal_fix)。
- BE-2 遲到的執行 (is_late：模型晚於推薦日、或推薦日隔天 09:00 後才跑) 不把推薦記成真實發布：已發布的日子不再記、不蓋 scans[D]、不發新推薦提醒；
  沒發布過的記成 scan_src='late' + backfill；同一推薦日多次記錄共用每日上限 (挖寶 6 / 飆股 topk)；既有的遲到列由 tag_late() 改標。
- BE-3 缺當天 ^TWII 不再把大盤當 0% (相對大盤只在兩個收盤都有時算)；build() 把 scan 以盤後表補的加權收盤帶進對帳；追蹤中的列每次清掉上次的達標標記。
- (上面 pr2「模型檔不動、無重訓」只指 pr2 那次；之後每週日重訓，門檻 / 宇宙會變，每列記 model_ver / th_A / th_Aplus / in_uni。)

2026-10-13 雷達其餘審查項：
- BE-4 新列 (挖寶 / 飆股) 記 in_uni = 掃描當時是否在模型宇宙內 (舊列不遷移，App 退回目前宇宙)。
- N3 飆股列加 fin20 / fin20Date (第 20 個交易日收盤報酬)，飆 / 跌破 / trail 提早出場後仍追到第 20 日 (同 fin21)；stats.surge 加 n_fin20 / win20 / avg_fin20。
- BE-7 stats.surge 加 n_published / n_backfill / published{done, hits, rate}；飆股失準只看真實發布結案 ≥20 筆。
- N4 飆股提醒寫「飆股分 0.xx」(模型原始分數，未校準)，不再寫「機率 NN%」。
- N8 / BE-5 radar_bt.same 加 from / to (推薦日界線，含頭尾)；N9 radar_bt.day0.period、radar_bt.path_period (2021~ 樣本外，不是 radar_bt.period)。
- BE-6 build() 先掃當天再回填：第一次 full run 拖到午夜後 (GitHub 排程延遲) 時，當天仍記成真實發布 (隔天 09:00 前)，不被回填搶先標成 backfill。
- BE-9 帳本超過 LEDGER_MAX 只刪最舊的描述級 (A/A+ 永不刪) 並記 log；舊列的大盤乖離趁 scans 還在時寫進列本身 (mb_src='scan')。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..http import cached, session
from ..sources.twse import roc_to_iso
from . import model as M
from . import treasure as T

log = logging.getLogger(__name__)
PAGES = "https://joekisoul-code.github.io/TaiexChipAnalyzer/data/treasure_live.json"
WORKER = "https://skynet-proxy.joekisoul.workers.dev"
LOCAL = config.CACHE_DIR / "treasure_live.json"
UA = {"User-Agent": "Mozilla/5.0"}
POOL, TREASURE_POOL, PER_DAY, MAX_ACTIVE, REENTRY_MISS_DAYS = 80, 40, 6, 40, 60
A_COOL_DAYS = 30               # r6：A/A+ 記錄規則 = 回測 (harness.simulate cool=30)：同檔 30 個日曆日內記過 (任何等級) 才跳過；不受前 20 / MAX_ACTIVE 限制
REC_RULE_A, REC_RULE_B = "A30", "B20"   # 每筆記錄規則 (rec_rule)：A30 = 上面的回測規則 (10-07 起)；B20 = 舊規則 (前 20、追蹤 ≤ 40、未命中 60 日)。舊列沒有此欄
SURGE_DEDUP_DAYS = 28
LEDGER_MAX = 1500
EXPECT_AAPLUS = (0.49, 0.54)   # r6 10-07：時點正確前 170 大、訓練窗與正式相同 (2020 起)，10 組 A∪A+ 命中 0.490~0.541；10-07 早上的 0.44~0.50 是 2018 起訓練的窗，更早的 0.55~0.57 有後見之明
ROLE = {"A+": "signal", "A": "signal", "B+": "descriptive", "B": "descriptive"}
DRIFT_MIN_PUBLISHED = 15       # 訊號級漂移判定：真實發布結案 ≥15 筆且命中低於預期下限 15pt
NOTE_TIERS = "A/A+ 為訊號級；B/B+ 為描述級 (回測即無選股力 #40/#84)"
MAX_FIN21_WAIT_DAYS = 60       # 結案列補 fin21：日 K 已延伸 60 個日曆日仍湊不到 21 根 (下市/停牌) → 放棄
WHY_TOP = 3                    # 10-06：A/A+ 原因 = 模型分下降最多的前 3 項
# 挖寶 × 飆股 回測 (10-07 改為 radar5 時點正確版；10-06 radar_study.py 的 170 檔固定名單有後見之明，已淘汰)
# hit = 挖寶定義命中、surge = 飆股定義、win21 = 第 21 交易日報酬 > 0、fin21 = 21 日平均報酬、q10 = 最差一成。
# 結論：兩邊同日都選到 (多為 B 級) 並沒有比較好 (時點正確版 n≈61 勝率 56%，A 級 69%) → 不另設「雙訊號」等級，只並列顯示。
RADAR_BT = {
    'A': {'n': 298, 'hit': 0.52, 'win21': 0.688, 'fin21': 7.59, 'med21': 5.56, 'q10': -11.51, 'win_rng': [0.64, 0.712], 'hit_rng': [0.49, 0.541]},
    'A+': {'n': 128, 'hit': 0.648, 'win21': 0.813, 'fin21': 12.37, 'med21': 9.99, 'q10': -6.88, 'win_rng': [0.77, 0.847], 'hit_rng': [0.614, 0.696]},
    'S3': {'n': 681, 'surge': 0.321, 'hit': 0.37, 'win21': 0.519, 'fin21': 4.48, 'med21': 0.97, 'q10': -17.51, 'win_rng': [0.505, 0.536], 'surge_rng': [0.313, 0.328]},
    'dual': {'n': 61, 'hit': 0.415, 'win21': 0.565, 'fin21': 6.32, 'med21': 3.35, 'q10': -17.57, 'win_rng': [0.466, 0.625], 'hit_rng': [0.293, 0.482]},
    'B': {'n': 5958, 'hit': 0.375, 'win21': 0.49, 'fin21': 1.87, 'med21': -0.19, 'q10': -14.47, 'win_rng': [0.484, 0.494], 'hit_rng': [0.368, 0.379]},
    "pool": {'n': 45218, 'hit': 0.38, 'win21': 0.5, 'fin21': 2.73, 'surge': 0.195, 'win20': 0.5},   # 同日掃描前 40 (飆股前 80) 的平均 = 不靠模型的基準 (只算已結案列)
    "period": "2022-01~2026-09", "note": "時點正確的前 170 大成交值股票 (每月依過去 120 日平均成交值，不用今天的名單)、訓練窗與正式相同 (2020 起)、逐年走動式樣本外、模擬每日掃描；5 個種子 × 2 條選股路徑平均，_rng = 10 組的最低~最高；同日兩邊都選到沒有比較好",
    "sparse": {'A': {'per_month': 5.23, 'months_with': 25.2, 'months': 57}, 'A+': {'per_month': 2.25, 'months_with': 11.9, 'months': 57}},   # 每月平均推薦數 / 有推薦的月份 (A 級只在大盤月線下的日子出現 → 很多月份沒有)
    # 10-07 更正：之前 (10-04~10-06) 的回測用 2026-09 當下的 170 檔名單去回測 2022~ → 事後才知道哪些股票會變大 (後見之明)，A 勝率高估、飆股也偏高
    # r6 10-07：radar5 重建時訓練窗用了 2018 起 (研究工具預設)，正式模型 treasure.train 是 2020 起 → 改成訓練窗與正式相同 (2020 起) 重算；
    #   A∪A+ 命中 0.473→0.520、勝率 0.624→0.688；兩個窗的差在 3 個月區塊 bootstrap 下不顯著 (r6 radarshadow 驗證)，只是讓 App 顯示的回測和正式模型一致
    "method": {"pit": True, "since": "2026-10-07", "train_start": "2020-01-01", "old": {"A": {"win21": 0.717, "hit": 0.563}, "S3": {"win21": 0.554, "surge": 0.33}},
               "prev_2018": {'A': {'win21': 0.624, 'hit': 0.473}, 'A+': {'win21': 0.757, 'hit': 0.596}, 'S3': {'win21': 0.513, 'surge': 0.313}},   # 10-07 早上 (radar5，2018 起訓練) 的版本
               "note": "舊回測用今天的名單 (事後才知道哪些股票變大)；現在改成每個時點只用當時的前 170 大，訓練窗與正式相同 (2020 起)"},
    # 2026 年 (1~9 月，模型只用 2025-11 以前資料訓練) — 跟「實際」同一年的樣本外對照
    "y2026": {"A": {'n': 62, 'hit': 0.514, 'win21': 0.777, 'fin21': 13.61, 'med21': 11.43, 'q10': -3.75, 'win_rng': [0.735, 0.809], 'hit_rng': [0.474, 0.559]},
              "S3": {'n': 141, 'surge': 0.349, 'hit': 0.351, 'win21': 0.577, 'fin21': 8.39, 'med21': 2.99, 'q10': -16.68, 'win_rng': [0.547, 0.613], 'surge_rng': [0.331, 0.373]},
              "B": {'n': 864, 'hit': 0.391, 'win21': 0.553, 'fin21': 5.87, 'med21': 2.32, 'q10': -16.17, 'win_rng': [0.55, 0.556], 'hit_rng': [0.382, 0.397]}},
    # 與雲端帳本「同一段期間」(推薦日 2026-07-01~09-03) 的樣本外回測 → 實際 (回填) 比它高的部分 ≈ 回填偏樂觀
    "path": None, "fail": None, "day0": None,   # 下方 RADAR_PATH / RADAR_FAIL / RADAR_DAY0，模組載入後填入
    # r6 候選池研究 (訓練窗與正式相同 2020 起；限定 PIT 前 170 大 vs 全上市訓練+候選，5 種子 × 2 路徑；差值 = 限定 − 全上市，95% = 3 個月區塊 bootstrap)：
    #   A 勝率 +3.3pt [-4.0, +7.3]、A 命中 +4.0pt [-1.3, +7.5]、A+ 勝率 +7.7pt [-4.3, +16.8]、飆股勝率 +4.1pt [-0.4, +9.7]、飆股率 +1.6pt [-1.2, +5.3]、B 勝率 +2.9pt [+1.7, +4.1]
    "pool_rule": {"rule": "候選池只用成交值前 170 大的股票 (與模型訓練一致)", "since": "2026-10-06", "window": "2022-01~2026-09，訓練窗與正式相同 (2020 起)",
             "before": {'A': {'n': 460, 'hit': 0.48, 'win21': 0.655, 'fin21': 7.44, 'win_rng': [0.645, 0.667], 'hit_rng': [0.472, 0.489]}, 'A+': {'n': 233, 'hit': 0.567, 'win21': 0.737, 'win_rng': [0.724, 0.75]}, 'S3': {'n': 1076, 'surge': 0.305, 'win21': 0.478, 'fin21': 3.01, 'q10': -20.72, 'win_rng': [0.46, 0.488], 'surge_rng': [0.296, 0.319]}, 'B': {'n': 6326, 'win21': 0.462}},
             "after": {'A': {'n': 298, 'hit': 0.52, 'win21': 0.688, 'fin21': 7.59, 'win_rng': [0.64, 0.712], 'hit_rng': [0.49, 0.541]}, 'A+': {'n': 128, 'hit': 0.648, 'win21': 0.813, 'win_rng': [0.77, 0.847]}, 'S3': {'n': 681, 'surge': 0.321, 'win21': 0.519, 'fin21': 4.48, 'q10': -17.51, 'win_rng': [0.505, 0.536], 'surge_rng': [0.313, 0.328]}, 'B': {'n': 5958, 'win21': 0.49}},
             "diff": {'A_win': {'d': 0.033, 'ci95': [-0.04, 0.073], 'noise': True}, 'A_hit': {'d': 0.04, 'ci95': [-0.013, 0.075], 'noise': True}, 'A+_win': {'d': 0.077, 'ci95': [-0.043, 0.168], 'noise': True}, 'S3_win': {'d': 0.041, 'ci95': [-0.004, 0.097], 'noise': True}, 'S3_surge': {'d': 0.016, 'ci95': [-0.012, 0.053], 'noise': True}, 'B_win': {'d': 0.029, 'ci95': [0.017, 0.041], 'noise': False}}, "within_noise": True,   # within_noise = App 顯示的 A / A+ / 飆股差值都在 95% 範圍內 (B 級另見 diff)
             "old_claim": {"A": [0.617, 0.695], "S3": [0.469, 0.552]}, "honest": True,
             "note": '時點正確重算 (訓練窗與正式相同 2020 起)：限定前 170 大的 A / A+ / 飆股都略好但在誤差內；B 級 (描述級) 勝率較高 (超出誤差)；之前的 62→70%、47→55% 主要來自用今天的名單回測'},
    "same": {"period": "2026-07~09", "from": "2026-07-01", "to": "2026-09-03",   # N8：推薦日界線 (含頭尾)；App 只拿同一段的實際列來比
             "A": {'n': 49, 'hit': 0.514, 'win21': 0.804, 'fin21': 14.4, 'med21': 12.34, 'q10': -2.63, 'win_rng': [0.766, 0.86], 'hit_rng': [0.469, 0.571]},
             "S3": {'n': 36, 'surge': 0.289, 'hit': 0.444, 'win21': 0.599, 'fin21': 8.08, 'med21': 3.35, 'q10': -13.5, 'win_rng': [0.5, 0.657], 'surge_rng': [0.265, 0.316]},
             "B": {'n': 212, 'hit': 0.314, 'win21': 0.503, 'fin21': 1.08, 'med21': 0.18, 'q10': -18.07, 'win_rng': [0.478, 0.531], 'hit_rng': [0.285, 0.341]}},
}
# r6 10-07 (final_tables_2020.py：時點正確前 170 大、訓練窗與正式相同 (2020 起)、2021~ 樣本外、5 種子 × 2 路徑合併；n = 平均每組筆數)
# A：全部 A/A+ 推薦在第 k 天的收盤報酬 (含已跌破的) → 最後命中；走動式 Brier (表 vs 平均) 2024 0.2501/0.2535, 2025 0.1639/0.2504, 2026 0.1771/0.2511
# S3：只用「第 k~下一個 k 天仍在追蹤」(還沒碰到 +20% 或 −10%) 的推薦 — 舊表含已飆 +20% 結案的 → 後段高估 30~50pt；Brier 2023 0.1945/0.2366, 2024 0.1969/0.2394, 2025 0.1764/0.2202, 2026 0.1647/0.2008
RADAR_PATH = {"A": {"1": [{"bin": "≤−8%", "n": 13, "hit": 0.0, "pos": 0.458, "avg": 3.9}, {"bin": "−8~−5%", "n": 16, "hit": 0.248, "pos": 0.554, "avg": 4.47}, {"bin": "−5~−2%", "n": 47, "hit": 0.378, "pos": 0.632, "avg": 6.12}, {"bin": "−2~+2%", "n": 140, "hit": 0.517, "pos": 0.688, "avg": 5.38}, {"bin": "+2~+5%", "n": 50, "hit": 0.652, "pos": 0.716, "avg": 8.89}, {"bin": "≥+5%", "n": 44, "hit": 0.853, "pos": 0.871, "avg": 18.27}], "2": [{"bin": "≤−8%", "n": 22, "hit": 0.0, "pos": 0.431, "avg": 1.31}, {"bin": "−8~−5%", "n": 14, "hit": 0.175, "pos": 0.416, "avg": -0.85}, {"bin": "−5~−2%", "n": 46, "hit": 0.377, "pos": 0.68, "avg": 5.99}, {"bin": "−2~+2%", "n": 98, "hit": 0.448, "pos": 0.624, "avg": 3.28}, {"bin": "+2~+5%", "n": 58, "hit": 0.658, "pos": 0.731, "avg": 9.0}, {"bin": "≥+5%", "n": 72, "hit": 0.86, "pos": 0.896, "avg": 17.61}], "3": [{"bin": "≤−8%", "n": 22, "hit": 0.0, "pos": 0.385, "avg": -0.37}, {"bin": "−8~−5%", "n": 13, "hit": 0.079, "pos": 0.433, "avg": 1.66}, {"bin": "−5~−2%", "n": 42, "hit": 0.214, "pos": 0.536, "avg": 0.31}, {"bin": "−2~+2%", "n": 89, "hit": 0.451, "pos": 0.634, "avg": 5.05}, {"bin": "+2~+5%", "n": 59, "hit": 0.71, "pos": 0.783, "avg": 7.77}, {"bin": "≥+5%", "n": 86, "hit": 0.845, "pos": 0.888, "avg": 17.26}], "5": [{"bin": "≤−8%", "n": 35, "hit": 0.003, "pos": 0.33, "avg": -2.19}, {"bin": "−8~−5%", "n": 17, "hit": 0.094, "pos": 0.565, "avg": 1.81}, {"bin": "−5~−2%", "n": 35, "hit": 0.256, "pos": 0.493, "avg": 0.39}, {"bin": "−2~+2%", "n": 72, "hit": 0.473, "pos": 0.648, "avg": 3.25}, {"bin": "+2~+5%", "n": 44, "hit": 0.692, "pos": 0.77, "avg": 7.57}, {"bin": "≥+5%", "n": 108, "hit": 0.83, "pos": 0.896, "avg": 17.47}], "10": [{"bin": "≤−8%", "n": 37, "hit": 0.04, "pos": 0.099, "avg": -11.13}, {"bin": "−8~−5%", "n": 16, "hit": 0.123, "pos": 0.362, "avg": -3.95}, {"bin": "−5~−2%", "n": 22, "hit": 0.156, "pos": 0.455, "avg": -0.33}, {"bin": "−2~+2%", "n": 52, "hit": 0.408, "pos": 0.613, "avg": 2.03}, {"bin": "+2~+5%", "n": 44, "hit": 0.56, "pos": 0.809, "avg": 8.03}, {"bin": "≥+5%", "n": 137, "hit": 0.809, "pos": 0.926, "avg": 17.73}], "15": [{"bin": "≤−8%", "n": 45, "hit": 0.0, "pos": 0.013, "avg": -14.03}, {"bin": "−8~−5%", "n": 12, "hit": 0.017, "pos": 0.172, "avg": -5.48}, {"bin": "−5~−2%", "n": 22, "hit": 0.164, "pos": 0.342, "avg": -1.06}, {"bin": "−2~+2%", "n": 37, "hit": 0.315, "pos": 0.712, "avg": 1.95}, {"bin": "+2~+5%", "n": 33, "hit": 0.565, "pos": 0.879, "avg": 5.94}, {"bin": "≥+5%", "n": 161, "hit": 0.806, "pos": 0.926, "avg": 17.75}]}, "S3": {"1": [{"bin": "≤−8%", "n": 22, "hit": 0.009, "pos": 0.472, "avg": -3.45, "nd": 22}, {"bin": "−8~−5%", "n": 47, "hit": 0.144, "pos": 0.404, "avg": -1.01, "nd": 47}, {"bin": "−5~−2%", "n": 139, "hit": 0.22, "pos": 0.443, "avg": 0.71, "nd": 139}, {"bin": "−2~+2%", "n": 319, "hit": 0.279, "pos": 0.479, "avg": 2.16, "nd": 319}, {"bin": "+2~+5%", "n": 124, "hit": 0.419, "pos": 0.578, "avg": 8.57, "nd": 124}, {"bin": "≥+5%", "n": 107, "hit": 0.613, "pos": 0.729, "avg": 16.24, "nd": 107}], "2": [{"bin": "≤−8%", "n": 7, "hit": 0.043, "pos": 0.486, "avg": -0.52, "nd": 7}, {"bin": "−8~−5%", "n": 65, "hit": 0.095, "pos": 0.292, "avg": -5.32, "nd": 65}, {"bin": "−5~−2%", "n": 136, "hit": 0.181, "pos": 0.394, "avg": -0.7, "nd": 136}, {"bin": "−2~+2%", "n": 221, "hit": 0.252, "pos": 0.461, "avg": 1.69, "nd": 221}, {"bin": "+2~+5%", "n": 118, "hit": 0.414, "pos": 0.623, "avg": 8.19, "nd": 118}, {"bin": "≥+5%", "n": 153, "hit": 0.623, "pos": 0.733, "avg": 14.89, "nd": 153}], "3": [{"bin": "≤−8%", "n": 35, "hit": 0.027, "pos": 0.198, "avg": -9.27, "nd": 37}, {"bin": "−8~−5%", "n": 116, "hit": 0.108, "pos": 0.318, "avg": -4.44, "nd": 142}, {"bin": "−5~−2%", "n": 183, "hit": 0.188, "pos": 0.395, "avg": -0.75, "nd": 231}, {"bin": "−2~+2%", "n": 251, "hit": 0.238, "pos": 0.498, "avg": 1.6, "nd": 331}, {"bin": "+2~+5%", "n": 155, "hit": 0.393, "pos": 0.623, "avg": 8.11, "nd": 190}, {"bin": "≥+5%", "n": 205, "hit": 0.626, "pos": 0.724, "avg": 15.41, "nd": 307}], "5": [{"bin": "≤−8%", "n": 47, "hit": 0.073, "pos": 0.255, "avg": -5.74, "nd": 56}, {"bin": "−8~−5%", "n": 147, "hit": 0.064, "pos": 0.298, "avg": -5.1, "nd": 237}, {"bin": "−5~−2%", "n": 201, "hit": 0.124, "pos": 0.35, "avg": -2.55, "nd": 352}, {"bin": "−2~+2%", "n": 279, "hit": 0.217, "pos": 0.516, "avg": 2.44, "nd": 561}, {"bin": "+2~+5%", "n": 207, "hit": 0.326, "pos": 0.578, "avg": 6.0, "nd": 351}, {"bin": "≥+5%", "n": 262, "hit": 0.574, "pos": 0.762, "avg": 13.74, "nd": 665}], "10": [{"bin": "≤−8%", "n": 31, "hit": 0.0, "pos": 0.151, "avg": -8.04, "nd": 38}, {"bin": "−8~−5%", "n": 88, "hit": 0.031, "pos": 0.225, "avg": -5.76, "nd": 157}, {"bin": "−5~−2%", "n": 112, "hit": 0.067, "pos": 0.373, "avg": -1.88, "nd": 197}, {"bin": "−2~+2%", "n": 167, "hit": 0.111, "pos": 0.463, "avg": -0.3, "nd": 331}, {"bin": "+2~+5%", "n": 140, "hit": 0.212, "pos": 0.604, "avg": 5.07, "nd": 254}, {"bin": "≥+5%", "n": 187, "hit": 0.44, "pos": 0.813, "avg": 11.65, "nd": 506}], "15": [{"bin": "≤−8%", "n": 17, "hit": 0.0, "pos": 0.197, "avg": -6.12, "nd": 22}, {"bin": "−8~−5%", "n": 57, "hit": 0.004, "pos": 0.124, "avg": -5.96, "nd": 103}, {"bin": "−5~−2%", "n": 83, "hit": 0.003, "pos": 0.222, "avg": -3.23, "nd": 152}, {"bin": "−2~+2%", "n": 108, "hit": 0.034, "pos": 0.525, "avg": 1.13, "nd": 204}, {"bin": "+2~+5%", "n": 86, "hit": 0.055, "pos": 0.723, "avg": 3.91, "nd": 144}, {"bin": "≥+5%", "n": 130, "hit": 0.258, "pos": 0.911, "avg": 10.85, "nd": 373}]}}
RADAR_FAIL = {
    "n": {'A': 298, 'S3': 681},
    "pre_note": '推薦當天看得到的約 60 項訊號，輸家與贏家幾乎分不開 (多數 AUC 0.44~0.56)；模型已經用掉大部分資訊',
    "A_market": {'lose_mvol': 1.5, 'hit_mvol': 2.86, 'lose_mb60': -4.5, 'hit_mb60': -9.1, 'yrs': '3/4', 'note': '描述：A 級失敗多在大盤溫和下跌、波動低時；命中多在急跌恐慌後。走動式濾網年數不足，未採用'},
    "S_ps": {'lose': 0.333, 'hit': 0.329, 'yrs': '4/5', 'wf_dhit': -0.034, 'note': '同為前 3 名，失敗與飆的飆股分數幾乎一樣；用之前年份濾掉最低兩成，命中沒有變好'},
    "early": {"A": {'rule': '第 3 天收盤 ≤ −5%', 'share': 0.114, 'hit': 0.029, 'lose': 0.534, 'hold': 0.42, 'exit': -12.75, 'n': 34},
              "S3": {'rule': '第 3 天收盤 ≤ −5%', 'share': 0.215, 'hit': 0.045, 'lose': 0.695, 'hold': -6.04, 'exit': -8.89, 'n': 147}},
}
# r6 10-07 推薦當天 A/A+ 分數校準 (訓練窗與正式相同 2020 起)：z = logit(p) − logit(th_A) 分 5 組 (邊界 edges)，各組歷史命中 (單調化)；走動式 Brier 2024 0.2426/0.2535, 2025 0.2285/0.2504, 2026 0.2332/0.2511 (3/3 年較好，小幅)
RADAR_DAY0 = {"edges": [0.112, 0.288, 0.505, 0.937], "hit": [0.431, 0.454, 0.454, 0.613, 0.697], "n": [62, 62, 62, 62, 62], "base": 0.53, "period": "2021-01~2026-09", "note": "A/A+ 依推薦分數分 5 組的歷史命中 (2021~ 樣本外，訓練窗與正式相同 2020 起)；小幅優於只看等級平均，當參考"}

_PATH_EDGES = (-8, -5, -2, 2, 5)
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def day0_prob(p: float | None, th_a: float | None) -> float | None:
    """10-07：A/A+ 推薦當天依分數組查歷史命中 (RADAR_DAY0；z = logit(p) − logit(th_A))。缺值 → None。"""
    try:
        p, th_a = float(p), float(th_a)
        if not (0 < p < 1 and 0 < th_a < 1):
            return None
        z = math.log(p / (1 - p)) - math.log(th_a / (1 - th_a))
        i = sum(1 for e in RADAR_DAY0["edges"] if z > e)
        return RADAR_DAY0["hit"][i]
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def path_prob(kind: str, days: int, cur: float) -> dict | None:
    """追蹤中推薦：依第 k 天 (取 ≤ days 的最大 k) 收盤報酬查歷史 → 最後命中 / 期滿為正。kind = "A" | "S3"。"""
    tb = RADAR_PATH.get(kind) or {}
    ks = sorted(int(k) for k in tb if int(k) <= (days or 0))
    if not ks or cur is None or not np.isfinite(cur):
        return None
    k = ks[-1]; i = next((j for j, e in enumerate(_PATH_EDGES) if cur <= e), len(_PATH_EDGES))
    rows = tb[str(k)]; lab = ["≤−8%", "−8~−5%", "−5~−2%", "−2~+2%", "+2~+5%", "≥+5%"][i]
    r = next((x for x in rows if x["bin"] == lab), None)
    return {"k": k, **r} if r else None


# 特徵中文名與格式 (App learning.js TM_LABEL 同款；改這裡要一起改)
FEAT_LABEL = {
    "pct": ("當日漲幅", "s%"), "amp": ("當日振幅", "%"), "lval": ("成交金額", "amt"), "b5": ("距 5 日線", "s%"), "b10": ("距 10 日線", "s%"),
    "b20": ("距月線", "s%"), "b60": ("距季線", "s%"), "align": ("均線", "align"), "ret5": ("近 5 日漲幅", "s%"), "ret20": ("近 20 日漲幅", "s%"),
    "ret60": ("近 60 日漲幅", "s%"), "dd_hi20": ("距 20 日高點", "s%"), "dd_hi60": ("距 60 日高點", "s%"), "lo20_dist": ("距 20 日低點", "s%"),
    "clv": ("收盤在當日高低區間", "clv"), "uw": ("上影線", "%"), "lw": ("下影線", "%"), "vol_ratio": ("量比 (對 20 日均量)", "x"),
    "vola20": ("20 日波動", "%"), "streak": ("連續", "streak"), "lag": ("當日落後大盤", "pt"), "rs20": ("20 日相對大盤", "pt"),
    "m_ret1": ("大盤當日", "s%"), "m_bias20": ("大盤月線乖離", "s%"),
}


def fmt_feat(f: str, v: float) -> str:
    kind = FEAT_LABEL.get(f, (f, ""))[1]
    if v is None or not np.isfinite(v):
        return "—"
    if kind == "s%":
        return f"{v:+.1f}%"
    if kind == "%":
        return f"{v:.1f}%"
    if kind == "pt":
        return f"{v:+.1f}pt"
    if kind == "x":
        return f"{v:.1f} 倍"
    if kind == "amt":
        return f"{10 ** v / 1e8:.1f} 億"
    if kind == "clv":
        return f"{v * 100:.0f}% 位置"
    if kind == "align":
        return "多頭排列" if v > 0 else "空頭排列" if v < 0 else "糾結"
    if kind == "streak":
        return f"連漲 {int(v)} 天" if v > 0 else f"連跌 {int(-v)} 天" if v < 0 else "平"
    return f"{v:.2f}"


def explain(tm: dict, x: dict, med: dict, k: int = WHY_TOP) -> list[dict]:
    """A/A+ 原因 (10-06)：逐項把特徵換成「今日候選池中位數」，看模型分掉多少 → 掉最多的前 k 項就是這檔「比同儕突出、模型最看重」的地方。
    是模型怎麼看 (單項替換、不含交互作用的完整拆解)，不是因果。App learning.js tmExplain() 同演算法。"""
    feats = tm["features"]; arr = [x.get(f, float("nan")) for f in feats]
    p0 = 1 / (1 + math.exp(-T._eval(tm, arr)))
    out = []
    for i, f in enumerate(feats):
        m = med.get(f)
        if m is None or not np.isfinite(m) or not np.isfinite(arr[i]) or abs(arr[i] - m) < 1e-12:
            continue
        a2 = list(arr); a2[i] = m
        d = p0 - 1 / (1 + math.exp(-T._eval(tm, a2)))
        if d > 0.002:
            lab = FEAT_LABEL.get(f, (f, ""))[0]
            out.append({"f": f, "v": round(float(arr[i]), 3), "med": round(float(m), 3), "d": round(d, 4),
                        "txt": f"{lab} {fmt_feat(f, arr[i])} (今日候選中位 {fmt_feat(f, m)})"})
    out.sort(key=lambda z: -z["d"])
    return out[:k]


def gate_txt(tier: str, mb: float | None) -> str:
    return (f"大盤在月線下 (乖離 {mb:+.1f}%)：A 級的必要條件" if mb is not None else "大盤在月線下：A 級的必要條件") + ("；分數達 A+ 門檻 (前 3%)" if tier == "A+" else "")


# ------------------------------------------------------------------ 資料
def _num(x):
    try:
        s = str(x).replace(",", "").strip()
        return float(s) if s not in ("", "--", "-", "X") else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def market_snapshot(date: str | None = None) -> tuple[str | None, dict]:
    """TWSE 盤後全市場表 (rwd MI_INDEX)。回傳 (資料日, {code: {name, open, high, low, close, volume, value, change, pct, amp, per}})。
    date 省略 → 今天；當天沒有資料 (休市/未公布) 往前找最多 6 天。"""
    d0 = dt.date.fromisoformat(date) if date else dt.datetime.now(config.TZ).date()
    for back in range(0, 7):
        d = d0 - dt.timedelta(days=back)
        if d.weekday() >= 5:
            continue
        ds = d.strftime("%Y%m%d")
        def load(ds=ds):
            r = session().get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", params={"date": ds, "type": "ALLBUT0999", "response": "json", "_": int(dt.datetime.now().timestamp())},
                              headers=UA, timeout=60)
            r.raise_for_status()
            j = r.json()
            tbl = [t for t in (j.get("tables") or []) if len(t.get("data") or []) > 500]
            if not tbl:   # 10-06：當天還沒公布 → 丟例外 (不寫快取)；原本回 None 會被快取 30 天，清晨跑過一次那天就永遠掃不到 (10-05 即此)
                raise LookupError(f"MI_INDEX {ds} 尚未公布")
            f = tbl[0]["fields"]; ix = {k: f.index(k) for k in f}
            out = {}
            for row in tbl[0]["data"]:
                code = str(row[ix["證券代號"]]).strip()
                if not re.fullmatch(r"\d{4,6}[A-Z]?", code):
                    continue
                c, o, h, l = _num(row[ix["收盤價"]]), _num(row[ix["開盤價"]]), _num(row[ix["最高價"]]), _num(row[ix["最低價"]])
                if not (c > 0):
                    continue
                sign = -1 if "-" in str(row[ix["漲跌(+/-)"]]) else 1
                chg = sign * _num(row[ix["漲跌價差"]])
                if not np.isfinite(chg):
                    chg = 0.0
                out[code] = {"name": str(row[ix["證券名稱"]]).strip(), "open": o, "high": h, "low": l, "close": c, "volume": _num(row[ix["成交股數"]]),
                             "value": _num(row[ix["成交金額"]]), "change": chg, "pct": chg / ((c - chg) or 1e-9) * 100, "amp": (h - l) / (l or 1e-9) * 100 if h > 0 and l > 0 else 0.0,
                             "per": _num(row[ix["本益比"]])}
            if not out:
                raise LookupError(f"MI_INDEX {ds} 無資料列")
            return {"date": d.isoformat(), "rows": out}
        try:
            j = cached(f"twse:mi_index_all2:{ds}", 30 * 86400, load, allow_stale=False)   # 10-06 換鍵：舊鍵可能存了「未公布」的 null
        except Exception as e:  # noqa: BLE001
            log.warning("MI_INDEX %s: %s", ds, e)
            j = None
        if j and j.get("rows"):
            return j["date"], j["rows"]
    return None, {}


def _twii_close_from_twse(D: str) -> float:
    """MI_INDEX 第一張表 (大盤統計資訊) 的「發行量加權股價指數」收盤。"""
    r = session().get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", params={"date": D.replace("-", ""), "type": "ALLBUT0999", "response": "json"}, headers=UA, timeout=60)
    for t in r.json().get("tables") or []:
        for row in t.get("data") or []:
            if row and "發行量加權股價指數" in str(row[0]) and len(row) > 1:
                v = _num(row[1])
                if v > 0:
                    return v
    raise ValueError("no TWII row")


def valuation() -> dict:
    """BWIBBU_ALL (經 Worker；Actions 直連 openapi 不通)：{code: {per, pbr}}。失敗 → {} (只影響候選池排序的 nav/valu 項)。"""
    def load():
        j = session().get(WORKER + "/twse/v1/exchangeReport/BWIBBU_ALL", headers=UA, timeout=60).json()
        return {str(x.get("Code")).strip(): {"per": _num(x.get("PEratio")), "pbr": _num(x.get("PBratio"))} for x in j if x.get("Code")}
    try:
        return cached("twse:bwibbu_all", config.TTL_DAILY, load) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("BWIBBU_ALL: %s", e)
        return {}


def screen(rows: dict, val: dict, mkt_pct: float, limit: int = POOL) -> list[dict]:
    """scoring.js screen()：power = mom + liq + eng + nav + valu + lag + near (treasureMult 固定 1)；成交值 > 5 千萬、pct < 9.3，前 limit。"""
    out = []
    for code, s in rows.items():
        if not (s["value"] > 5e7) or not np.isfinite(s["close"]):
            continue
        pct, amp = s["pct"], s["amp"]
        v = val.get(code) or {}
        pbr, per = v.get("pbr", float("nan")), v.get("per", float("nan"))
        mom = min(pct, 7) * 1.6 if pct >= 0 else pct * 0.6
        liq = math.log10(s["value"]) * 2
        eng = min(amp, 8) * 0.8
        nav = (1.2 - pbr) * 10 if pbr > 0 and pbr < 1.2 else 0
        valu = (15 - per) * 0.5 if per > 0 and per < 15 else 0
        lag = min(mkt_pct - pct, 5) * 1.2 if pct < mkt_pct else 0
        near = (pct - 3) * 0.6 if 3 <= pct < 9 else 0
        if pct < 9.3:
            out.append({"code": code, **s, "pbr": pbr, "power": mom + liq + eng + nav + valu + lag + near})
    out.sort(key=lambda x: -x["power"])
    return out[:limit]


def bars(code: str, years: str = "2y") -> list[dict]:
    """Yahoo 日 K (date/open/high/low/close/volume)：Worker /idxh 優先 (與 App 同源)，失敗改直連 Yahoo。.TW 不夠長再試 .TWO。"""
    def via_worker(sym):
        j = session().get(WORKER + "/idxh", params={"sym": sym, "range": years, "interval": "1d"}, headers=UA, timeout=60).json()
        return [x for x in (j.get("data") or []) if x.get("close")]
    def via_yahoo(sym):
        j = session().get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"interval": "1d", "range": years}, headers=UA, timeout=60).json()
        res = j["chart"]["result"][0]; ts, q = res["timestamp"], res["indicators"]["quote"][0]; off = res["meta"].get("gmtoffset", 0)
        return [{"date": (dt.datetime.fromtimestamp(t, dt.UTC) + dt.timedelta(seconds=off)).strftime("%Y-%m-%d"), "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i], "volume": q["volume"][i]}
                for i, t in enumerate(ts) if q["close"][i]]
    for suf in (".TW", ".TWO"):
        sym = code + suf
        for fn in (via_worker, via_yahoo):
            try:
                rows = cached(f"tl:bars:{sym}:{years}", config.TTL_INTRADAY * 3, lambda fn=fn, sym=sym: fn(sym), allow_stale=False)
                if rows and len(rows) > 60:
                    return rows
            except Exception as e:  # noqa: BLE001
                log.debug("bars %s %s: %s", sym, fn.__name__, e)
    return []


def twii_hist() -> dict:
    """{date: close} (Yahoo ^TWII 10y，Worker 優先)。"""
    rows = []
    try:
        rows = cached("tl:twii:10y", config.TTL_INTRADAY * 3, lambda: session().get(WORKER + "/idxh", params={"sym": "^TWII", "range": "10y", "interval": "1d"}, headers=UA, timeout=60).json().get("data"),
                      allow_stale=False) or []
    except Exception as e:  # noqa: BLE001
        log.debug("twii via worker: %s", e)
    if not rows:
        try:
            from ..sources import global_markets
            df = global_markets.history("^TWII", "10y")
            rows = df.to_dict("records")
        except Exception as e:  # noqa: BLE001
            log.warning("twii fallback: %s", e)
    return {str(x["date"])[:10]: float(x["close"]) for x in rows if x.get("close")}


# ------------------------------------------------------------------ 交易日曆 (BE-1，2026-10-13)
# Yahoo 日 K 的日子與證交所不一致：有幽靈日 (2026-07-10 非交易日卻有一根)，也會缺真實交易日 (2026-10-12 個股收盤全是 null、2025-08-01 永久缺)。
# 帳本 (第幾天 / 第 21 日 / 停損日) 與特徵 (近 N 日、>11% 跳動濾網) 都把 Yahoo 的列當交易日 → 整段位移一天、整批股票被誤判「疑似減資」略過。
# 這裡以證交所 FMTQIK (每月每日市場成交資訊) 當日曆：去掉日曆外的列、日曆內缺的日子用當天 MI_INDEX 盤後表補 (必要時依比例換到 Yahoo 的基準)。
# 只用在雷達的評分 (score_one) 與帳本對帳 (evaluate)；App 端 (learning.js) 另行處理。
TWSE_RWD = "https://www.twse.com.tw/rwd/zh"
CAL_FINAL_TTL = 400 * 86400    # 已結束月份的交易日不會再變 → 長期快取 (另一個鍵，見 _fmtqik_days)
CAL_LOOKBACK_DAYS = 200        # score_one 對齊多少個日曆天 (特徵最長 120 個交易日 ≈ 175 天；>11% 濾網看 61 根)
FILL_MAX_RUN = 3               # 連續缺幾個交易日以內才用盤後表補 (Yahoo 的洞是零星單日；更長 = 停牌或 Yahoo 該檔整段沒更新 → 不補，也不逐日抓盤後表)
MI_FETCH_MAX = 60              # 每次執行 (每個 TradeCal) 最多查幾天的盤後表 (補洞 + 對照基準；每天約 250 KB、30 天磁碟快取)


def _fmtqik_days(ym: str, final: bool) -> list[str]:
    """證交所 FMTQIK 某月 (ym = 'YYYYMM') 的交易日 (ISO，排序)。
    抓不到 / stat 不是 OK / 沒有資料列 / 日期不在該月 → 丟例外，不回 None 或 [] (cached() 會把回傳值存起來，#87：存了「沒有交易日」就會整月誤刪)。
    final = 該月已結束：用 ':final' 鍵長期快取；本月用短 TTL 的另一個鍵 (月中抓的不完整清單不會在月底後被當成整月)。"""
    def load():
        r = session().get(TWSE_RWD + "/afterTrading/FMTQIK", params={"date": ym + "01", "response": "json"}, headers=UA, timeout=30)
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, dict) or j.get("stat") != "OK" or not j.get("data"):
            raise LookupError(f"FMTQIK {ym}: {j.get('stat') if isinstance(j, dict) else type(j).__name__}")
        want = f"{ym[:4]}-{ym[4:6]}"
        days = sorted({roc_to_iso(str(row[0])) for row in j["data"] if row})
        if not days or any(d[:7] != want for d in days):
            raise LookupError(f"FMTQIK {ym}: 日期不在 {want} ({days[:1]})")
        return days
    return cached(f"twse:fmtqik_days:{ym}" + (":final" if final else ""), CAL_FINAL_TTL if final else config.TTL_INTRADAY, load)


class TradeCal:
    """證交所交易日曆 (逐月懶載入：用到哪個月才抓，測試替換 score_one / 空帳本時不連網) + 補洞用的單日盤後表 (每次執行記憶)。
    每月狀態：'exact' = FMTQIK；'fallback' = FMTQIK 抓不到 → 掃描日 ∪ ^TWII 的日子 (extra)，只拿來補洞、不刪列 (^TWII 自己也會缺日)；
    None = 未知 (未來月份、都抓不到) → 不處理。已結束的月份整月已知；本月只到已知的最後一個交易日 (之後的列不判斷)。"""

    def __init__(self, extra=None, today: str | None = None, months: dict | None = None, mi: dict | None = None):
        self._extra, self._fb = extra, None       # extra：iterable 或回傳 iterable 的函式 (懶取，FMTQIK 抓不到才用)
        self.today = today or dt.datetime.now(config.TZ).strftime("%Y-%m-%d")
        self._m: dict[str, tuple] = {k: (v[0], set(v[1])) for k, v in (months or {}).items()}   # 'YYYY-MM' → (kind, {days})；測試可直接給
        self._mi: dict[str, dict] = dict(mi or {})
        self._n_mi = 0

    def _fallback(self) -> set:
        if self._fb is None:
            try:
                src = self._extra() if callable(self._extra) else self._extra
                self._fb = {str(d)[:10] for d in (src or ()) if d}
            except Exception as e:  # noqa: BLE001
                log.debug("trade cal extra: %s", e); self._fb = set()
        return self._fb

    def month(self, ym: str) -> tuple:
        if ym not in self._m:
            if ym > self.today[:7]:
                self._m[ym] = (None, set())
            else:
                try:
                    self._m[ym] = ("exact", set(_fmtqik_days(ym.replace("-", ""), final=ym < self.today[:7])))
                except Exception as e:  # noqa: BLE001
                    fb = {d for d in self._fallback() if d[:7] == ym}
                    log.warning("FMTQIK %s: %s → %s", ym, e, f"後備日曆 {len(fb)} 天 (只補洞)" if fb else "日曆未知")
                    self._m[ym] = ("fallback" if fb else None, fb)
        return self._m[ym]

    def known(self, d: str) -> str | None:
        """d 的日曆狀態 ('exact' / 'fallback' / None)；本月 (含之後) 已知的最後交易日之後 → None。"""
        kind, days = self.month(d[:7])
        if not kind or (d[:7] >= self.today[:7] and (not days or d > max(days))):
            return None
        return kind

    def is_day(self, d: str) -> bool:
        return d in self.month(d[:7])[1]

    def days(self, a: str, b: str) -> list[str]:
        """[a, b] 內日曆上的交易日 (exact 或 fallback 月份)。"""
        out, y, m = [], int(a[:4]), int(a[5:7])
        while f"{y:04d}-{m:02d}" <= b[:7]:
            out += [d for d in self.month(f"{y:04d}-{m:02d}")[1] if a <= d <= b]
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return sorted(out)

    def exact(self, a: str, b: str) -> bool:
        """[a, b] 每個月份都是 FMTQIK 日曆、且 b 在已知範圍內 (可用來驗證帳本標記)。"""
        y, m = int(a[:4]), int(a[5:7])
        while f"{y:04d}-{m:02d}" <= b[:7]:
            if self.month(f"{y:04d}-{m:02d}")[0] != "exact":
                return False
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return self.known(b) == "exact"

    def mi(self, d: str) -> dict:
        """d 當天的 MI_INDEX 盤後表 {code: row}；那天沒有資料 (market_snapshot 退回別天) / 抓不到 → {}。"""
        if d not in self._mi:
            if self._n_mi >= MI_FETCH_MAX:
                if self._n_mi == MI_FETCH_MAX:
                    log.warning("trade cal：盤後表已查 %d 天，其餘缺日本次不補 (下次再補)", MI_FETCH_MAX)
                    self._n_mi += 1
                return {}
            self._n_mi += 1
            try:
                got, rows = market_snapshot(d)
            except Exception as e:  # noqa: BLE001
                log.debug("trade cal MI %s: %s", d, e); got, rows = None, {}
            self._mi[d] = rows if got == d else {}
        return self._mi[d]


def _fin(v) -> float:
    """有限數 → float；None / NaN / 非數字 → 0。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _mi_scale(rows: list[dict], d: str, code: str, cal: TradeCal) -> float | None:
    """補洞列的價格基準：最近的前一根 (沒有再看後一根) Yahoo 與 MI 都有的日子，k = Yahoo 收盤 ÷ MI 收盤 (差 ≤ PX_TOL 視為 1)。
    兩邊都對不到 → None (不補，維持缺口)。同 score_one 對 D 日盤後表的縮放 (r7)。"""
    before = [r for r in rows if str(r["date"])[:10] < d]
    after = [r for r in rows if str(r["date"])[:10] > d]
    for r in ([before[-1]] if before else []) + ([after[0]] if after else []):
        m = cal.mi(str(r["date"])[:10]).get(code)
        yc, mc = _fin(r.get("close")), _fin((m or {}).get("close"))
        if yc > 0 and mc > 0:
            k = yc / mc
            return 1.0 if abs(k - 1) <= PX_TOL else k
    return None


def _gap_runs(cal: TradeCal, a: str, z: str, have: set) -> list[list[str]]:
    """[a, z] 內日曆已知、Yahoo 沒有的交易日，依交易日序連續分段 (每段 = 一個洞)。"""
    runs, cur = [], []
    for d in cal.days(a, z):
        if d not in have and cal.known(d):
            cur.append(d)
        elif cur:
            runs.append(cur); cur = []
    return runs + ([cur] if cur else [])


def align_bars(b: list[dict], cal: TradeCal | None, code: str, lo: str | None = None, hi: str | None = None) -> list[dict]:
    """BE-1：Yahoo 日 K 對齊證交所日曆 (只處理 lo 之後、日曆已知的日子；其餘原樣)。回傳新 list (不改 bars() 快取裡的原 list)。
    1. 刪掉 FMTQIK 日曆外的列 (幽靈日)。
    2. 第一根 (或 lo) 到最後一根 (或 hi，取較晚) 之間、日曆上有但 Yahoo 缺的交易日 (連續 ≤ FILL_MAX_RUN 天的洞) → 用當天 MI_INDEX 補 OHLCV (標 fill='mi')；
       該股當天不在盤後表 (停牌/沒成交) → 維持缺口；Yahoo 因配股/減資回溯縮放過 → 補的這根依 _mi_scale 同比例換算。"""
    if not b or cal is None:
        return b
    rows = []
    for r in b:
        d = str(r.get("date"))[:10]
        if (lo is None or d >= lo) and cal.known(d) == "exact" and not cal.is_day(d):
            log.debug("align %s: 去掉非交易日 %s", code, d)
            continue
        rows.append(r)
    if not rows:
        return rows
    rows.sort(key=lambda r: str(r["date"])[:10])
    have = {str(r["date"])[:10] for r in rows}
    a = str(rows[0]["date"])[:10]; z = str(rows[-1]["date"])[:10]
    a = max(a, lo) if lo else a
    z = max(z, hi) if hi else z
    add = []
    for d in [d for run in _gap_runs(cal, a, z, have) if len(run) <= FILL_MAX_RUN for d in run]:
        m = cal.mi(d).get(code)
        if not m or not _fin(m.get("close")) > 0:
            continue
        k = _mi_scale(rows, d, code, cal)
        if k is None:
            continue
        c = float(m["close"])

        def px(v, c=c, k=k):          # 開高低缺 (--) → 用收盤
            v = _fin(v)
            return (v if v > 0 else c) * k
        add.append({"date": d, "open": px(m.get("open")), "high": px(m.get("high")), "low": px(m.get("low")), "close": c * k,
                    "volume": max(_fin(m.get("volume")), 0.0) / k, "fill": "mi"})
    if add:
        log.info("align %s: 以盤後表補 %s", code, ",".join(x["date"] for x in add))
        rows = sorted(rows + add, key=lambda r: str(r["date"])[:10])
    return rows


def cal_bars(code: str, cal: TradeCal | None, lo: str | None = None, hi: str | None = None) -> list[dict]:
    """bars() 對齊證交所日曆 (cal = None → 原樣)。"""
    return align_bars(bars(code), cal, code, lo, hi)


# ------------------------------------------------------------------ 評分 (與 App tmScore 相同)
def _mk_frame(mk: dict, upto: str) -> pd.DataFrame:
    d = sorted(k for k in mk if k <= upto)
    s = pd.Series([mk[k] for k in d], index=d)
    return pd.DataFrame({"date": d, "m_close": s.values, "m_ret1": s.pct_change().values * 100, "m_ret20": s.pct_change(20).values * 100,
                         "m_bias20": (s / s.rolling(20).mean() - 1).values * 100})


def score_one(code: str, snap_row: dict, D: str, mkf: pd.DataFrame, tm: dict, cal: TradeCal | None = None) -> dict | None:
    """cal (BE-1)：日 K 先對齊證交所日曆 (D 前 CAL_LOOKBACK_DAYS 天起；補到 D 為止)，避免 Yahoo 缺日造成「兩天報酬」被當成 >11% 跳動、特徵視窗位移。"""
    b = bars(code)
    if cal is not None:
        b = align_bars([x for x in b if str(x["date"])[:10] <= D], cal, code, lo=(dt.date.fromisoformat(D) - dt.timedelta(days=CAL_LOOKBACK_DAYS)).isoformat(), hi=D)
    rows = [{"date": str(x["date"])[:10], "open": x.get("open"), "high": x.get("high"), "low": x.get("low"), "close": x["close"], "volume": x.get("volume") or 0, "amount": np.nan}
            for x in b if x.get("close") and str(x["date"])[:10] <= D]
    if not rows:
        return None
    r = snap_row
    bar = {"date": D, "open": r["open"] if r["open"] > 0 else r["close"], "high": r["high"] if r["high"] > 0 else r["close"], "low": r["low"] if r["low"] > 0 else r["close"],
           "close": r["close"], "volume": r["volume"] or (rows[-1]["volume"] if rows[-1]["date"] == D else 0), "amount": r["value"]}
    if rows[-1]["date"] == D:
        yc = float(rows[-1]["close"] or 0)
        if yc > 0 and r["close"] > 0 and abs(yc / r["close"] - 1) > PX_TOL:   # r7：回填日之後有配股/減資 → Yahoo 已縮放，MI 原始價換到同一基準 (否則 D 日假跳動)
            k = yc / r["close"]
            bar.update(open=bar["open"] * k, high=bar["high"] * k, low=bar["low"] * k, close=bar["close"] * k)
        rows[-1] = bar
    elif D > rows[-1]["date"]:
        rows.append(bar)
    if rows[-1]["date"] != D or len(rows) < 62:
        return None
    g = pd.DataFrame(rows)
    # 10-06：近 60 日有單日 >11% 的跳動 = 減資/分割未還原的壞資料 (台股漲跌幅 10%)；例 6949 09-14「近 20 日 −94.7%」被評 A → 不評分
    if (g["close"].astype(float).pct_change().abs().tail(61) > 0.11).any():
        log.info("score %s: 近 60 日價格跳動 >11%%，疑似減資/分割未還原，略過", code)
        return None
    g["amount"] = g["amount"].fillna(g["close"] * g["volume"])      # App：amount 只有 D 日有值，其餘以 close×volume (lval 只用 D 日)
    f = T.features(g, mkf, label=False).iloc[-1]
    x = {k: (float(f[k]) if k in f and pd.notna(f[k]) else float("nan")) for k in set(tm["features"]) | set((tm.get("surge") or {}).get("features") or [])}
    arr = [x[k] for k in tm["features"]]
    if any(not np.isfinite(v) for v in arr):
        return None
    p = 1 / (1 + math.exp(-T._eval(tm, arr)))
    gl = (tm.get("gate") or {}).get("m_bias20_lt")
    hi = p >= tm.get("th_A", 0.5); gate = gl is None or x["m_bias20"] < gl
    tier = "B" if not hi else ("B+" if not gate else ("A+" if tm.get("th_Aplus") is not None and p >= tm["th_Aplus"] else "A"))
    ps = None
    sg = tm.get("surge")
    if sg and sg.get("trees"):
        try:
            ps = 1 / (1 + math.exp(-T._eval(sg, [x[k] if np.isfinite(x[k]) else None for k in sg.get("features") or tm["features"]])))
        except Exception as e:  # noqa: BLE001
            log.debug("surge eval %s: %s", code, e)
    return {"p": round(p, 4), "tier": tier, "ps": round(ps, 4) if ps is not None else None, "m_bias20": round(x["m_bias20"], 3), "_x": {f: x[f] for f in tm["features"]}}


def pool_rows(rows: dict, tm: dict) -> dict:
    """10-06：候選池限定模型訓練宇宙 (tm["universe"])；模型對宇宙外的中小型股選股力差 (radar_study2)。舊模型檔沒有 universe → 全市場。"""
    uni = set(tm.get("universe") or [])
    return {c: v for c, v in rows.items() if c in uni} if uni else rows


def scan(tm: dict, date: str | None = None) -> dict:
    """當日掃描：回傳 {date, pool_n, treasure[(前 40 中 pct<9.4、依 p 排序)], surge[前 3], mkt_close, mkt_bias20, errors}。
    r14 (2026-10-13) 另回傳 pool = radar_archive.pool_block()：所有評到分的候選 (緊湊列) + 門檻 / 大盤 / 成交集中度 conc10；
    只給 radar_archive 歸檔 (build()/backfill() 會從 scan 拿掉，不進 treasure_live.json)。"""
    D, rows = market_snapshot(date)
    if not D:
        return {"date": None, "error": "no market snapshot"}
    mk = twii_hist()
    if D not in mk:                       # Yahoo ^TWII 尚未有 D 日 (常晚數小時) → 以盤後表內的加權指數收盤補上
        try:
            mk = dict(mk); mk[D] = _twii_close_from_twse(D)
        except Exception as e:  # noqa: BLE001
            log.warning("TWII %s missing (%s)：大盤特徵以前一日計", D, e)
    mkf = _mk_frame(mk, D)
    mkt_pct = float(mkf["m_ret1"].iloc[-1]) if len(mkf) and pd.notna(mkf["m_ret1"].iloc[-1]) else 0.0
    pool = [r for r in screen(pool_rows(rows, tm), valuation(), mkt_pct, POOL) if not r["code"].startswith("00")][:POOL]   # 10-06：限定訓練宇宙
    res, errors = {}, 0
    cal = TradeCal(extra=lambda: set(mk) | {D}, mi={D: rows})    # BE-1：證交所日曆 (懶載入)；D 日盤後表已在手上
    for r in pool:
        try:
            s = score_one(r["code"], r, D, mkf, tm, cal=cal)
            if s:
                res[r["code"]] = s
        except Exception as e:  # noqa: BLE001
            errors += 1; log.debug("score %s: %s", r["code"], e)
    med = {f: float(np.nanmedian([v["_x"][f] for v in res.values()])) for f in tm["features"]} if res else {}
    tre = [{"code": r["code"], "name": r["name"], "close": r["close"], "pct": round(r["pct"], 2), "value": r["value"], "pbr": r.get("pbr"),
            **{k: v for k, v in res[r["code"]].items() if k != "_x"}}
           for r in pool[:TREASURE_POOL] if r["code"] in res and r["pct"] < 9.4]
    for t in tre:   # 10-06：A/A+ 附原因 (模型分拆解 + 大盤閘門)
        if t["tier"] in ("A", "A+"):
            try:
                ex = explain(tm, res[t["code"]]["_x"], med)
                t["why"] = [w["txt"] for w in ex]
                t["why_d"] = [round(float(w["d"]), 3) for w in ex]     # 2026-10-12 r13：每項原因的模型分下降 (與 why 同順序)，App 畫小橫條

                t["why_gate"] = gate_txt(t["tier"], t.get("m_bias20"))
                t["d0"] = day0_prob(t.get("p"), tm.get("th_A"))      # 10-07 同分數組歷史命中 (參考)
            except Exception as e:  # noqa: BLE001
                log.debug("explain %s: %s", t["code"], e)
    tre.sort(key=lambda x: -x["p"])
    sg_k = int(((tm.get("surge") or {}).get("def") or {}).get("topk") or 3)
    sur = sorted([{"code": r["code"], "name": r["name"], "close": r["close"], "pct": round(r["pct"], 2), "ps": res[r["code"]]["ps"]} for r in pool if r["code"] in res and res[r["code"]]["ps"] is not None and r["pct"] < 9.4],
                 key=lambda x: -x["ps"])[:sg_k]
    out = {"date": D, "pool_n": len(pool), "scored_n": len(res), "errors": errors, "treasure": tre, "surge": sur, "pool_rule": "universe" if tm.get("universe") else "all", "uni_n": len(tm.get("universe") or []),
           "mkt_close": mk.get(D), "mkt_bias20": round(float(mkf["m_bias20"].iloc[-1]), 3) if len(mkf) and pd.notna(mkf["m_bias20"].iloc[-1]) else None,
           "mkt_pct": round(mkt_pct, 2)}
    try:   # 2026-10-13 r14：整個評分過的候選池 + 當日成交集中度 (radar_archive 每日歸檔；build()/backfill() 會拿掉，treasure_live.json 不變)
        from . import radar_archive as RA
        out["pool"] = RA.pool_block(D, pool, res, tre, sur, tm, rows, out)
    except Exception as e:  # noqa: BLE001
        log.warning("radar pool block: %s", e)
    return out


# ------------------------------------------------------------------ 帳本
def load_prev() -> dict:
    """上次發布 (Pages) ∪ 本機備份，帳本以 (kind, code, date) 去重、取較完整者。"""
    out = {"ledger": {"treasure": [], "surge": []}, "scans": []}
    srcs = []
    try:
        r = session().get(PAGES, headers=UA, timeout=30)
        if r.ok and r.text.strip().startswith("{"):
            srcs.append(r.json())
    except Exception as e:  # noqa: BLE001
        log.debug("published treasure_live: %s", e)
    try:
        if LOCAL.exists():
            srcs.append(json.loads(LOCAL.read_text(encoding="utf-8")))
    except Exception as e:  # noqa: BLE001
        log.debug("local treasure_live: %s", e)
    for s in srcs:
        for kind in ("treasure", "surge"):
            have = {(x["code"], x["date"]): i for i, x in enumerate(out["ledger"][kind])}
            for x in ((s.get("ledger") or {}).get(kind) or []):
                k = (x.get("code"), x.get("date"))
                if k in have:
                    old = out["ledger"][kind][have[k]]
                    if (x.get("status") != "追蹤" and old.get("status") == "追蹤") or (len(json.dumps(x)) > len(json.dumps(old)) and old.get("status") == "追蹤")                             or (old.get("status") != "追蹤" and x.get("status") == old.get("status") and ((x.get("fin21") is not None and old.get("fin21") is None) or (x.get("fin20") is not None and old.get("fin20") is None))):   # pr2：已結案但另一份已補 fin21 (N3：飆股 fin20)
                        out["ledger"][kind][have[k]] = x
                else:
                    have[k] = len(out["ledger"][kind]); out["ledger"][kind].append(x)
        seen = {x["date"] for x in out["scans"]}
        for x in (s.get("scans") or []):
            if x.get("date") and x["date"] not in seen:
                out["scans"].append(x); seen.add(x["date"])
    for kind in ("treasure", "surge"):
        out["ledger"][kind].sort(key=lambda x: (x.get("date") or "", x.get("code") or ""))
    out["scans"].sort(key=lambda x: x["date"])
    return out


LATE_CUTOFF = "09:00:00"       # BE-2：掃描日隔天這個時間 (台北) 之後才跑 = 遲到 (同 r6 shadow_spec / radar_archive)
LIVE_SINCE = "2026-10-05"      # treasure_live 第一次上線 (d04369e 2026-10-05 13:34)：更早推薦日的「live」列其實都是事後補掃


def is_late(D: str | None, asof=None, model_ver: str | None = None) -> bool:
    """BE-2：掃描日 D 的這次執行是不是「遲到」(不是 D 當天盤後準時發布)：
    模型在 D 之後才訓練 (model_ver 日期 > D：週日重訓後退回上週五重掃)，或 asof (台北) ≥ D 隔天 09:00
    (平日休市的 full run、MI_INDEX 還沒公布就跑 → market_snapshot 退回前一交易日)。
    radar_archive._on_time / _rank 共用這個判斷。D 壞掉 → False；沒給 asof / model_ver → 該項不算遲到。"""
    d = str(D or "")[:10]
    try:
        cut = (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat() + " " + LATE_CUTOFF
    except ValueError:
        return False
    mv = str(model_ver or "")[:10]
    if mv and mv > d:
        return True
    a = asof.strftime("%Y-%m-%d %H:%M:%S") if isinstance(asof, (dt.datetime, dt.date)) else str(asof or "")
    return bool(a) and a >= cut


def tag_late(prev: dict) -> int:
    """BE-2 帳本修正 (冪等，每次 build 都跑)：其實是遲到重掃、卻標成真實發布的列 → scan_src='late'、backfill=True
    (App 與 stats 都當回填)。判定：model_ver 日期晚於推薦日 (例 10-02 的 4989/6155/6213、10-08 飆股 2383/3189)，
    或推薦日早於 LIVE_SINCE (10-02 那批是 10-05 才補掃)。回傳改了幾列。"""
    n = 0
    for kind in ("treasure", "surge"):
        for x in (prev.get("ledger") or {}).get(kind) or []:
            D = str(x.get("date") or "")[:10]
            if x.get("backfill") or not _ISO.fullmatch(D):
                continue
            if is_late(D, None, x.get("model_ver")) or D < LIVE_SINCE:
                x["scan_src"], x["backfill"] = "late", True
                n += 1
    return n


def _recent_codes(rows: list[dict], D: str, days: int) -> set:
    """與 D 相距不到 days 個日曆日 (同 harness.simulate 的 (td − lp).days < cool) 記過的代號 (任何等級)。
    取絕對值 (同飆股 SURGE_DEDUP_DAYS)：回填較早的日子時，也不和之後已記的列擠在一起。日期壞掉的列略過。"""
    out = set()
    try:
        d0 = pd.Timestamp(D)
    except Exception:  # noqa: BLE001
        return out
    for x in rows:
        try:
            if x.get("code") and x.get("date") and abs((d0 - pd.Timestamp(x["date"])).days) < days:
                out.add(x["code"])
        except Exception:  # noqa: BLE001
            continue
    return out


def record(prev: dict, sc: dict, tm: dict | None = None, scan_src: str = "live") -> dict:
    """把本日掃描結果記入帳本 (同 App recordDiscoveries / recordSurge 的規則)。
    pr2：每筆加 model_ver (掃描時模型 trained_at)、th_A / th_Aplus (當時門檻)、mkt_bias20 (閘門狀態)、scan_src ('live' 真實發布 | 'backfill' 回填)，
    讓帳本不再路徑相依、可事後重算等級。
    BE-2 (2026-10-13)：每日上限以「該推薦日已有的真實發布列」起算、飆股每個推薦日最多 topk 檔；遲到的重掃由 build() 決定記不記 (scan_src='late')。"""
    D = sc.get("date")
    if not D:
        return prev
    tm = tm or {}
    meta = {"model_ver": tm.get("trained_at"), "th_A": tm.get("th_A"), "th_Aplus": tm.get("th_Aplus"), "mkt_bias20": sc.get("mkt_bias20"), "scan_src": scan_src,
            **({"th_q92": tm["th_q92"]} if tm.get("th_q92") is not None else {})}   # 2026-10-12：影子觀察嚴格 A 用
    # BE-4 (2026-10-13)：掃描當時是否在模型宇宙內 (in_uni)。宇宙每週重訓都會換 → App 不再用「目前」的宇宙回頭標舊列；
    # 模型檔沒有 universe (全市場候選) → 不寫此欄 (App 退回目前宇宙)。舊列維持無欄位，不做歷史遷移。
    uni = {str(c) for c in (tm.get("universe") or [])}
    def _uni(code):
        return {"in_uni": str(code) in uni} if uni else {}
    led = prev["ledger"]
    tre = led["treasure"]
    active = [x for x in tre if x.get("status") == "追蹤"]
    act_codes = {x["code"] for x in active}
    recent_miss = {x["code"] for x in tre if x.get("status") == "未命中" and x.get("evalAt") and (pd.Timestamp(D) - pd.Timestamp(x["evalAt"])).days < REENTRY_MISS_DAYS}
    recent_rec = _recent_codes(tre, D, A_COOL_DAYS)
    have = {(x["code"], x["date"]) for x in tre}
    n = sum(1 for x in tre if x.get("date") == D and not x.get("backfill"))   # BE-2：同一天第二次發布 (15:40 + 21:45) 也共用每日 6 檔上限
    s_codes = {r["code"] for r in sc.get("surge") or []}                          # 10-06：今日飆股前 3
    t_tier = {r["code"]: (r["tier"], r["p"]) for r in sc.get("treasure") or []}   # 今日挖寶候選 (含 B 級) 的等級與模型分
    # r6 (10-07)：A/A+ 改用回測的記錄規則 — 池內 (前 40、依 p 排序) 只要同檔 30 個日曆日內沒被記過 (任何等級) 就記，
    # 不受「前 20 名」與「追蹤中 ≤ 40」限制 (radarshadow F6：舊規則只留下回測 A 集合的約 1/4，帳本的 A 級成績跟回測不是同一群)；
    # 每日 6 檔上限仍共用 (回測 cap 6)。B/B+ 維持舊規則 (前 20、追蹤中 ≤ 40、未命中 60 日內不重複、追蹤中不重複)。
    for i, r in enumerate(sc["treasure"]):
        if n >= PER_DAY:
            break
        if (r["code"], D) in have:
            continue
        is_a = r.get("tier") in ("A", "A+")
        if is_a:
            if r["code"] in recent_rec:
                continue
        elif i >= 20 or len(active) + n >= MAX_ACTIVE or r["code"] in act_codes or r["code"] in recent_miss:
            continue
        tre.append({"code": r["code"], "name": r["name"], "date": D, "entry": r["close"], "mktEntry": sc.get("mkt_close"), "p": r["p"], "tier": r["tier"], "ps": r.get("ps"),
                    "pct": r["pct"], "status": "追蹤", "peak": 0.0, "trough": 0.0, "cur": 0.0, "rel": 0.0, "days": 0, **meta, **_uni(r["code"]), "rec_rule": REC_RULE_A if is_a else REC_RULE_B,
                    **({"why": r["why"]} if r.get("why") else {}), **({"sPick": True} if r["code"] in s_codes else {})})
        have.add((r["code"], D)); recent_rec.add(r["code"]); act_codes.add(r["code"])
        n += 1
    sur = led["surge"]
    sg_k = int((((tm.get("surge") or {}).get("def") or {}).get("topk")) or 3)
    for r in sc["surge"]:
        if sum(1 for x in sur if x.get("date") == D) >= sg_k:      # BE-2：每個推薦日最多 sg_k 檔 (重掃不再追加)
            break
        if any(x["code"] == r["code"] and abs((pd.Timestamp(D) - pd.Timestamp(x["date"])).days) < SURGE_DEDUP_DAYS for x in sur):
            continue
        tt = t_tier.get(r["code"])
        sur.append({"code": r["code"], "name": r["name"], "date": D, "entry": r["close"], "ps": r["ps"], "status": "追蹤", "days": 0,
                    **({"tier": tt[0], "p": tt[1]} if tt else {}),
                    "model_ver": meta["model_ver"], "th_top10": (tm.get("surge") or {}).get("th_top10"), "scan_src": scan_src, **_uni(r["code"])})
    scans = [x for x in prev.get("scans") or [] if x.get("date") != D]
    scans.append({"date": D, "n_treasure": len(sc["treasure"]), "n_A": sum(1 for r in sc["treasure"] if r["tier"] in ("A", "A+")), "mkt_bias20": sc.get("mkt_bias20"),
                  "top": [{"code": r["code"], "tier": r["tier"], "p": r["p"]} for r in sc["treasure"][:6]], "surge": [{"code": r["code"], "ps": r["ps"]} for r in sc["surge"]]})
    # BE-9：舊列 (回填 / 早期 live) 沒記大盤乖離 → 趁當日掃描紀錄還在時寫進列本身 (mb_src='scan')，scans 只留 400 天後嚴格 A 影子統計仍分得出來
    mbd = {s_["date"]: s_.get("mkt_bias20") for s_ in scans if s_.get("date") and s_.get("mkt_bias20") is not None}
    for x in tre:
        if x.get("mkt_bias20") is None and x.get("date") in mbd:
            x["mkt_bias20"], x["mb_src"] = mbd[x["date"]], "scan"
    prev["scans"] = scans[-400:]
    trimmed = trim_ledger(led)
    if trimmed:
        prev["trimmed"] = {k: (prev.get("trimmed") or {}).get(k, 0) + v for k, v in trimmed.items()}
    return prev


def trim_ledger(led: dict) -> dict:
    """BE-9：帳本每類最多 LEDGER_MAX 筆 (控制 JSON 大小)。挖寶的訊號級 (A/A+) 列永不刪 (約 90 筆/年)，只刪最舊的描述級 (B/B+)；
    飆股照舊刪最舊。刪了就記 log 並回傳 {kind: 筆數} (不再無聲截斷)。"""
    out = {}
    tre = led.get("treasure") or []
    if len(tre) > LEDGER_MAX:
        sig = [x for x in tre if x.get("tier") in ("A", "A+")]
        desc = sorted((x for x in tre if x.get("tier") not in ("A", "A+")), key=lambda x: (x.get("date") or "", x.get("code") or ""))
        keep = desc[len(desc) - max(0, LEDGER_MAX - len(sig)):] if LEDGER_MAX > len(sig) else []
        led["treasure"] = sorted(sig + keep, key=lambda x: (x.get("date") or "", x.get("code") or ""))
        out["treasure"] = len(tre) - len(led["treasure"])
    sur = led.get("surge") or []
    if len(sur) > LEDGER_MAX:
        led["surge"] = sur[-LEDGER_MAX:]
        out["surge"] = len(sur) - LEDGER_MAX
    out = {k: v for k, v in out.items() if v}
    if out:
        log.warning("treasure_live 帳本超過 %d 筆，刪去最舊的描述級 / 飆股列：%s (A/A+ 保留)", LEDGER_MAX, out)
    return out


def _fin21_of(x: dict):
    """第 21 個交易日收盤報酬 (回測口徑)：新列有 fin21；舊結案列若在第 21 日結案，cur 即 fin21。"""
    if x.get("fin21") is not None:
        return x["fin21"]
    if x.get("status") not in (None, "追蹤") and (x.get("days") or 0) >= T.H and x.get("cur") is not None:
        return x["cur"]
    return None


PX_TOL = 0.01   # r7：推薦價 (證交所盤後原始收盤) 與 Yahoo 同日收盤差 >1% = Yahoo 事後因配股/減資把整段歷史回溯縮放 (09 筆實測 1.5~33%，其餘 <0.05%)
PX_KEYS = ("status", "reason", "cur", "peak", "trough", "rel", "relMax", "days", "lastDate", "evalAt", "trail", "trailDate", "fin21", "fin21Date", "fin21_na",
           "hitDay", "hitDate", "hitVal", "hitBy", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate", "chk")
WALK_KEYS = ("hitDay", "hitDate", "hitVal", "hitBy", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate")   # _walk_* 產生的「第幾天」標記


def _bar_on(b: list[dict], d: str) -> dict | None:
    return next((r for r in b if str(r["date"])[:10] == d and r.get("close")), None)


def _basis(x: dict, b: list[dict]) -> float:
    """r7 價格基準：帳本 entry 是證交所原始收盤，日 K 是 Yahoo (配股/減資後整段回溯縮放 → 假停損 / 假飆股，研究 r7/ledger F1)。
    同一序列的推薦日收盤與 entry 差 >PX_TOL → 改用序列的推薦日收盤當報酬基準 (entry 本身不改，畫面仍顯示當天真實價)，記 px_adj = 比值。"""
    e = float(x["entry"])
    d0 = _bar_on(b, x["date"])
    c0 = float(d0["close"]) if d0 else 0.0
    if c0 > 0 and e > 0 and abs(c0 / e - 1) > PX_TOL:
        x["px_adj"] = round(c0 / e, 4)
        return c0
    x.pop("px_adj", None)
    return e


def _px_repair(x: dict, b: list[dict]) -> bool:
    """r7 一次性 (pxk=1)：已結案列若價格基準不一致，原結果存到 orig、重設為追蹤，交給 _eval_* 以正確基準重算 (停損日語意同每日對帳)。"""
    x["pxk"] = 1
    if x.get("status") in (None, "追蹤"):
        return False
    d0 = _bar_on(b, x["date"]); e = float(x["entry"]); c0 = float(d0["close"]) if d0 else 0.0
    if not (c0 > 0 and e > 0 and abs(c0 / e - 1) > PX_TOL):
        return False
    x["orig"] = {k: x[k] for k in CAL_KEYS if k in x}     # CAL_KEYS = PX_KEYS + 飆股 fin20 (N3)
    for k in CAL_KEYS:
        x.pop(k, None)
    x["status"] = "追蹤"; x["px_fix"] = 1
    return True


def _walk_treasure(since: list[dict], e: float, me, mk: dict) -> dict:
    """回測定義逐日走訪 (與原 _eval_treasure 迴圈同邏輯)；10-06 另記第一次達標 / 跌破 / 峰值的交易日序與日期，讓 App 能顯示「實際有沒有命中、哪天」。
    BE-3：相對大盤只在「推薦日大盤 (me) 與當天大盤收盤都有」時才算 (原本缺當天 ^TWII → 大盤當 0%，+4% 收盤就算贏大盤 4pt)；
    整段都沒有大盤 → rel_max = None。"""
    cm = -1e9; stop = False; reached = False; peak = -1e9; trough = 1e9; rel_max = None; w: dict = {}
    for k, r in enumerate(since, 1):
        d = str(r["date"])[:10]
        cr = (float(r["close"]) / e - 1) * 100; lr = (float(r.get("low") or r["close"]) / e - 1) * 100
        cm = max(cm, cr); trough = min(trough, lr)
        if cr > peak:
            peak = cr; w["peakDay"], w["peakDate"] = k, d
        m = mk.get(d)
        rel = (cr - (float(m) / float(me) - 1) * 100) if (me and m) else None
        if rel is not None and not np.isfinite(rel):
            rel = None
        if rel is not None:
            rel_max = rel if rel_max is None else max(rel_max, rel)
        if not stop and not reached and lr <= T.STOP and cm < T.TGT:
            stop = True; w["stopDay"], w["stopDate"], w["stopVal"] = k, d, round(lr, 2)
        if not stop and (cr >= T.TGT or (rel is not None and rel >= T.REL)):
            if not reached:
                w["hitDay"], w["hitDate"], w["hitVal"] = k, d, round(cr, 2); w["hitBy"] = "漲幅" if cr >= T.TGT else "相對大盤"
            reached = True
    w.update(stop=stop, reached=reached, peak=peak, trough=trough, rel_max=rel_max)
    return w


def _treasure_reason(st: str, w: dict, cur: float, H: int) -> str:
    """結案原因 (10-06：寫出第幾天、哪一天)。"""
    if st == "未命中" and w.get("stopDay"):
        return f"第 {w['stopDay']} 天 ({w['stopDate'][5:]}) 盤中跌破 {T.STOP:.0f}% (低點 {w['stopVal']:+.1f}%)"
    if st == "命中":
        return f"第 {w['hitDay']} 天 ({w['hitDate'][5:]}) {'收盤 ' + format(w['hitVal'], '+.1f') + '% 達標' if w.get('hitBy') == '漲幅' else '贏大盤 4pt 達標'}；結案 {cur:+.1f}% (峰值 {w['peak']:+.1f}%)"
    if st == "回落":
        return f"第 {w.get('hitDay', '?')} 天曾達標 (峰值 {w['peak']:+.1f}%)，但第 {H} 天結案 {cur:+.1f}%"
    return f"{H} 個交易日內未達 +6% / 贏大盤 4pt (峰值 {w['peak']:+.1f}%)"


def _annotate_closed_treasure(x: dict, since: list[dict], mk: dict, e: float | None = None) -> None:
    """已結案列補「第幾天」資訊 (一次性：chk=1)；狀態/數值不動。e = 價格基準 (r7；預設 entry)。"""
    n = int(x.get("days") or 0)
    if n <= 0 or len(since) < n:
        return
    me = x.get("mktEntry") or (mk.get(x["date"]) if x["date"] in mk else None)
    w = _walk_treasure(since[:n], float(x["entry"]) if e is None else e, me, mk)
    for k in ("hitDay", "hitDate", "hitVal", "hitBy", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate"):
        if k in w:
            x[k] = w[k]
    if x.get("cur") is not None and x.get("status") in ("命中", "回落", "未命中"):
        x["reason"] = _treasure_reason(x["status"], w, float(x["cur"]), T.H)
    x["chk"] = 1


def _eval_treasure(x: dict, b: list[dict], mk: dict) -> None:
    """回測定義 (treasure.features 的 label)：21 交易日；峰值取收盤、停損取盤中低點 (先後順序)、相對大盤任一日 ≥ 4pt。
    pr2：fin21 = 第 21 個交易日收盤報酬，不論是否已停損 (停損後仍追到 21 日)；已結案列只補 fin21 (cur/peak/days 等結案值不動)。"""
    e = _basis(x, b); since = [r for r in b if str(r["date"])[:10] > x["date"] and r.get("close")][:T.H]
    if not since:
        return
    if len(since) >= T.H and x.get("fin21") is None:
        x["fin21"] = round((float(since[T.H - 1]["close"]) / e - 1) * 100, 2); x["fin21Date"] = str(since[T.H - 1]["date"])[:10]
    elif len(since) < T.H and x.get("fin21") is None and (pd.Timestamp(str(b[-1]["date"])[:10]) - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
        x["fin21_na"] = True      # 日 K 不足 (下市/停牌/資料斷)：不再重試
    if x.get("status") not in (None, "追蹤"):
        if x.get("chk") is None:      # 10-06：舊結案列補「第幾天達標/跌破」
            _annotate_closed_treasure(x, since, mk, e)
        if x.get("trail") is None:   # 舊結案列 (理論上結案時已填)：補移動停利
            pk = e
            for r in since:
                pk = max(pk, float(r["close"]))
                if pk >= e * 1.1 and float(r["close"]) <= pk * 0.92:
                    x["trail"] = round((float(r["close"]) / e - 1) * 100, 2); x["trailDate"] = str(r["date"])[:10]; break
            if x.get("trail") is None and len(since) >= T.H:
                x["trail"] = x.get("cur")
        return
    me = x.get("mktEntry") or (mk.get(x["date"]) if x["date"] in mk else None)
    w = _walk_treasure(since, e, me, mk)
    if w["stop"] and (w.get("stopDay") or 0) < len(since):   # r7：每日對帳語意 = 停損當天結案 (晚對帳 / 重算時不用停損後的收盤；fin21 已另外用第 21 日)
        since = since[:w["stopDay"]]; w = _walk_treasure(since, e, me, mk)
    stop, reached, peak, trough, rel_max = w["stop"], w["reached"], w["peak"], w["trough"], w["rel_max"]
    for k in WALK_KEYS + ("trail", "trailDate"):   # BE-3：追蹤中的列每次從頭走訪 → 先清掉上次的標記 (上次缺 ^TWII 時的假「相對大盤」達標不可殘留)
        x.pop(k, None)
    for k in WALK_KEYS:
        if k in w:
            x[k] = w[k]
    x["chk"] = 1
    cur = (float(since[-1]["close"]) / e - 1) * 100
    d_last = str(since[-1]["date"])[:10]
    last_m = mk.get(d_last); rel = (cur - ((last_m / me - 1) * 100)) if (me and last_m) else None
    x.update(peak=round(peak, 2), trough=round(trough, 2), cur=round(cur, 2), rel=round(rel, 2) if rel is not None else None,
             relMax=round(rel_max, 2) if rel_max is not None else None, days=len(since), lastDate=d_last)
    # 移動停利 (A 級出場建議：漲 10% 後自最高收盤回落 8%)
    if x.get("trail") is None:
        pk = e
        for r in since:
            pk = max(pk, float(r["close"]))
            if pk >= e * 1.1 and float(r["close"]) <= pk * 0.92:
                x["trail"] = round((float(r["close"]) / e - 1) * 100, 2); x["trailDate"] = str(r["date"])[:10]; break
    # BE-3：第 21 日期滿結案要有當天大盤收盤 (相對大盤那一項才算得完整)；缺 → 留到下次 (已有更晚的 K = 大盤永久缺那天 → 照常結案，不卡住)
    mk_ok = d_last in mk or not me or str(b[-1]["date"])[:10] > d_last
    if x.get("status") == "追蹤" and (stop or (len(since) >= T.H and mk_ok)):
        st = "未命中" if stop else ("命中" if reached and cur > 0 else ("回落" if reached else "未命中"))
        x["status"], x["reason"] = st, _treasure_reason(st, w, cur, T.H)
        x["evalAt"] = str(since[-1]["date"])[:10]
        if x.get("trail") is None:
            x["trail"] = round(cur, 2)


def _walk_surge(since: list[dict], e: float) -> dict:
    """飆股定義逐日走訪 (同一根 K 先判跌破再判漲到，與原邏輯相同)；10-06 另記第幾天先到 +20% / 先破 −10% / 最高點。"""
    peak = e; res = None; trail = None; tdate = None; w: dict = {}
    for k, r in enumerate(since, 1):
        d = str(r["date"])[:10]
        h = float(r.get("high") or r["close"]); l = float(r.get("low") or r["close"]); c = float(r["close"])
        if res is None and l <= e * (1 + T.SURGE_DN / 100):
            res = "未飆"; w["stopDay"], w["stopDate"], w["stopVal"] = k, d, round((l / e - 1) * 100, 2)
        if res is None and h >= e * (1 + T.SURGE_UP / 100):
            res = "飆"; w["hitDay"], w["hitDate"], w["hitVal"] = k, d, round((h / e - 1) * 100, 2)
        if h > peak:
            peak = h; w["peakDay"], w["peakDate"] = k, d
        if trail is None and peak >= e * 1.1 and c <= peak * 0.92:
            trail, tdate = (c / e - 1) * 100, d
    w.update(res=res, peak=peak, trail=trail, tdate=tdate)
    return w


def _surge_reason(st: str, w: dict, e: float) -> str:
    if st == "飆":
        return f"第 {w['hitDay']} 天 ({w['hitDate'][5:]}) 盤中最高 {w['hitVal']:+.1f}%，先到 +{T.SURGE_UP:.0f}%"
    if w.get("stopDay"):
        return f"第 {w['stopDay']} 天 ({w['stopDate'][5:]}) 盤中跌破 {T.SURGE_DN:.0f}% (低點 {w['stopVal']:+.1f}%)"
    return f"{T.SURGE_N} 個交易日內最高 {(w['peak'] / e - 1) * 100:+.1f}%，未到 +{T.SURGE_UP:.0f}%"


def _eval_surge(x: dict, b: list[dict]) -> None:
    e = _basis(x, b); since = [r for r in b if str(r["date"])[:10] > x["date"] and r.get("close")][:T.SURGE_N]
    if not since:
        return
    # N3 (2026-10-13)：fin20 = 第 20 個交易日收盤報酬 (回測口徑)，飆 / 跌破 / trail 提早出場後仍追到第 20 日補上 (同挖寶 fin21)
    if len(since) >= T.SURGE_N and x.get("fin20") is None:
        x["fin20"] = round((float(since[T.SURGE_N - 1]["close"]) / e - 1) * 100, 2); x["fin20Date"] = str(since[T.SURGE_N - 1]["date"])[:10]
    elif len(since) < T.SURGE_N and x.get("fin20") is None and (pd.Timestamp(str(b[-1]["date"])[:10]) - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
        x["fin20_na"] = True      # 日 K 不足 (下市/停牌/資料斷)：不再重試
    w = _walk_surge(since, e)
    peak, res, trail, tdate = w["peak"], w["res"], w["trail"], w["tdate"]
    if x.get("status") == "追蹤":       # BE-3：追蹤中的列每次從頭走訪 → 先清掉上次的標記 (日 K 對齊後可能變)
        for k in WALK_KEYS + ("trail", "trailDate"):
            x.pop(k, None)
    for k in ("hitDay", "hitDate", "hitVal", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate"):
        if k in w:
            x[k] = w[k]
    x["chk"] = 1
    last = float(since[-1]["close"])
    x.update(cur=round((last / e - 1) * 100, 2), peak=round((peak / e - 1) * 100, 2), days=len(since), lastDate=str(since[-1]["date"])[:10])
    if trail is not None:
        x["trail"], x["trailDate"] = round(trail, 2), tdate
    if x.get("status") == "追蹤" and (res in ("飆", "未飆") or len(since) >= T.SURGE_N):
        x["status"] = "飆" if res == "飆" else "未飆"; x["evalAt"] = str(since[-1]["date"])[:10]
    if x.get("status") in ("飆", "未飆"):
        x["reason"] = _surge_reason(x["status"], w, e)
    if x.get("trail") is None and len(since) >= T.SURGE_N:
        x["trail"] = x["cur"]


CAL_MARKS = (("days", "lastDate"), ("hitDay", "hitDate"), ("stopDay", "stopDate"), ("peakDay", "peakDate"))   # (第幾個交易日, 日期) 成對的標記
CAL_KEYS = PX_KEYS + ("fin20", "fin20Date", "fin20_na")      # 日曆重算時清掉的結果欄 (fin20 = 飆股第 20 日，若有)


def _cal_check(x: dict, b: list[dict], cal: TradeCal) -> bool | None:
    """BE-1：已結案列的「第幾天」標記 (days/lastDate、hit/stop/peak、fin21Date) 與對齊證交所日曆後的日 K 是否一致。
    True = 一致 (標 calv=1，之後不再查)；False = 不一致 (當時算進幽靈日或漏掉真實交易日 → 要重算)；
    None = 無法驗證 (那段不是 FMTQIK 日曆、或缺的交易日盤後表抓不到 / 該股在表內卻沒補上) → 下次再查。"""
    D = str(x.get("date") or "")[:10]
    if not _ISO.fullmatch(D):
        return None
    try:
        marks = [(int(x[kd]), str(x[dd])[:10]) for kd, dd in CAL_MARKS if x.get(kd) and x.get(dd)]
    except (TypeError, ValueError):
        return None
    if x.get("fin21Date"):
        marks.append((T.H, str(x["fin21Date"])[:10]))
    if x.get("fin20Date"):
        marks.append((T.SURGE_N, str(x["fin20Date"])[:10]))
    if not marks:
        return True
    end = max(d for _, d in marks)
    if not cal.exact(D, end):
        return None
    since = [str(r["date"])[:10] for r in b if str(r["date"])[:10] > D and r.get("close")]
    nxt = (dt.date.fromisoformat(D) + dt.timedelta(days=1)).isoformat()
    for run in _gap_runs(cal, nxt, end, set(since)):
        if len(run) > FILL_MAX_RUN:             # 長段缺口 = 停牌 (align_bars 也不補) → 當成真的沒交易
            continue
        for d in run:
            m = cal.mi(d)
            if not m or x.get("code") in m:     # 盤後表抓不到、或該股當天有成交卻沒補上 → 不能斷定
                return None
    return all(k <= len(since) and since[k - 1] == d for k, d in marks)


def _closed(x: dict) -> bool:
    return x.get("status") not in (None, "追蹤")


def _cal_reset(x: dict) -> None:
    """BE-1 一次性：標記與證交所日曆不一致的結案列 → 原結果存 cal_orig、重設為追蹤，交給 _eval_* 以對齊後的日 K 重算 (同 r7 _px_repair)。"""
    x["cal_orig"] = {k: x[k] for k in CAL_KEYS if k in x}
    for k in CAL_KEYS:
        x.pop(k, None)
    x["status"] = "追蹤"; x["cal_fix"] = 1


def _todo(prev: dict, cal_on: bool = False) -> list[tuple]:
    """evaluate 要處理的列 [(kind, x, need)]。need = 要重算 (追蹤中 / 補 fin21 / trail / chk)；其餘只做一次性檢查
    (pxk 價格基準；cal_on 時已結案列的 calv 日曆標記)。need 的列排前面 (一次性檢查的大批舊列不擠掉每日對帳)。"""
    todo = []
    for x in prev["ledger"]["treasure"]:
        need = x.get("status") == "追蹤" or x.get("trail") is None or (x.get("fin21") is None and not x.get("fin21_na") and (x.get("days") or 0) < T.H) or x.get("chk") is None
        if need or x.get("pxk") is None or (cal_on and x.get("calv") is None and _closed(x)):
            todo.append(("t", x, need))     # pr2：停損提早結案的列繼續追到第 21 日補 fin21；10-06：舊結案列補一次「第幾天」(chk)
    for x in prev["ledger"]["surge"]:
        need = x.get("status") == "追蹤" or (x.get("trail") is None and (x.get("days") or 0) < T.SURGE_N) or x.get("chk") is None \
            or (x.get("fin20") is None and not x.get("fin20_na"))   # N3：trail 提早出場 / 舊列 (days 停在 20 以前或 =20 卻沒 fin20) 也追到補上 fin20
        if need or x.get("pxk") is None or (cal_on and x.get("calv") is None and _closed(x)):
            todo.append(("s", x, need))   # r7：pxk = 價格基準一次性檢查 (配股/減資回溯縮放)；只檢查不需修的列不重算 (結果維持原樣)
    return [t for t in todo if t[2]] + [t for t in todo if not t[2]]


def evaluate(prev: dict, mk: dict, max_fetch: int = 200, upto: str | None = None, cal: TradeCal | None = None) -> int:
    """逐檔對帳 (未結案或尚未出場的紀錄)；每次最多抓 max_fetch 檔 (控制 Yahoo 呼叫)。upto：只用該日 (含) 以前的 K 棒 (回填時逐日模擬)。
    cal (BE-1)：日 K 先對齊證交所日曆 (去幽靈日、盤後表補缺日)；已結案列一次性核對「第幾天」標記 (calv)，不一致 → 重算。cal = None → 原樣 (舊行為)。"""
    n = 0
    todo = _todo(prev, cal is not None)
    lo = min((str(x["date"])[:10] for _, x, _ in todo if _ISO.fullmatch(str(x.get("date") or "")[:10])), default=None) if cal is not None else None   # 對齊範圍 = 最早一筆要處理的推薦日起
    cache: dict[str, list] = {}
    for kind, x, need in todo[:max_fetch * 2]:
        if n >= max_fetch and x["code"] not in cache:
            break
        try:
            b = cache.get(x["code"])
            if b is None:
                b = bars(x["code"]); n += 1          # 與評分同一份 2 年日 K 快取
                if cal is not None:
                    b = align_bars(b, cal, x["code"], lo=lo)
                cache[x["code"]] = b
            if upto:
                b = [r for r in b if str(r["date"])[:10] <= upto]
            if not b:
                # 取不到任何日 K (下市/改代號)：已結案、尚缺 fin21 的列超過 MAX_FIN21_WAIT_DAYS 個日曆日 → 標 fin21_na，不再每次重抓
                if kind == "t" and x.get("status") not in (None, "追蹤") and x.get("fin21") is None and not x.get("fin21_na"):
                    ref = pd.Timestamp(upto) if upto else pd.Timestamp(dt.datetime.now(config.TZ).strftime("%Y-%m-%d"))
                    if (ref - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
                        x["fin21_na"] = True
                if kind == "s" and x.get("status") not in (None, "追蹤") and x.get("fin20") is None and not x.get("fin20_na"):   # N3：同上 (飆股 fin20)
                    ref = pd.Timestamp(upto) if upto else pd.Timestamp(dt.datetime.now(config.TZ).strftime("%Y-%m-%d"))
                    if (ref - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
                        x["fin20_na"] = True
                continue
            fix = _px_repair(x, b) if x.get("pxk") is None else False
            if cal is not None and x.get("calv") is None and _closed(x):
                ok = _cal_check(x, b, cal)
                if ok is False:
                    log.info("evaluate %s %s: 第幾天標記與證交所日曆不一致 → 重算", x["code"], x["date"])
                    _cal_reset(x); fix = True
                elif ok:
                    x["calv"] = 1
            if not (need or fix):
                continue
            (_eval_treasure(x, b, mk) if kind == "t" else _eval_surge(x, b))
            if cal is not None and x.get("calv") is None and _closed(x) and _cal_check(x, b, cal):
                x["calv"] = 1
        except Exception as e:  # noqa: BLE001
            log.debug("evaluate %s: %s", x["code"], e)
    return n


# 2026-10-12 影子觀察「嚴格 A」：正式 A/A+ 裡，大盤月線乖離 ≤ −1.5% 且模型分 ≥ 0.92 分位 (radar5 研究：回測方向一致、差距不顯著 → 前瞻追蹤)。
# 只統計、不改正式等級與推薦。舊列沒有 th_q92 → 以 0.90 (th_A) 與 0.97 (th_Aplus) 分位間線性內插近似。
SHADOW_STRICT = {"gate": -1.5, "q": 0.92}


def strict_a_th(x: dict, tm: dict | None = None) -> tuple[float | None, bool]:
    """(門檻, 是否為近似)。順序：列的 th_q92 → 列的 0.90/0.97 分位內插 → 目前模型的 th_q92 / 內插 (舊列沒記門檻，近似)。"""
    if x.get("th_q92") is not None:
        return float(x["th_q92"]), False
    for src in (x, tm or {}):
        if src is not tm and src.get("th_q92") is not None:
            return float(src["th_q92"]), False
        if src is tm and src.get("th_q92") is not None:
            return float(src["th_q92"]), True
        a, b = src.get("th_A"), src.get("th_Aplus")
        if a is not None and b is not None:
            return float(a) + (float(b) - float(a)) * (SHADOW_STRICT["q"] - 0.90) / (0.97 - 0.90), True
    return None, True


def is_strict_a(x: dict, tm: dict | None = None, mb_by_date: dict | None = None) -> bool | None:
    """True/False；資料不足 (沒有 p / 大盤乖離 / 門檻) → None。舊列沒記 mkt_bias20 → 用當日掃描紀錄 (scans) 的大盤乖離。"""
    if x.get("tier") not in ("A", "A+"):
        return False
    th, _ = strict_a_th(x, tm)
    mb = x.get("mkt_bias20") if x.get("mkt_bias20") is not None else (mb_by_date or {}).get(x.get("date"))
    if x.get("p") is None or mb is None or th is None:
        return None
    return float(mb) <= SHADOW_STRICT["gate"] and float(x["p"]) >= th


def _published(x: dict) -> bool:
    """真實發布列 = 不是回填、也不是遲到重掃 (BE-2：late 列本來就帶 backfill=True，這裡再擋一次)。"""
    return not x.get("backfill") and x.get("scan_src") != "late"


def stats(prev: dict, tm: dict) -> dict:
    for kind in ("treasure", "surge"):
        prev["ledger"][kind].sort(key=lambda x: (x.get("date") or "", x.get("code") or ""))
    tre = prev["ledger"]["treasure"]; sur = prev["ledger"]["surge"]
    done = [x for x in tre if x.get("status") != "追蹤"]
    def agg(rows):
        if not rows:
            return None
        hit = sum(1 for x in rows if x["status"] == "命中")
        cur = [x["cur"] for x in rows if x.get("cur") is not None]; rel = [x["rel"] for x in rows if x.get("rel") is not None]
        tr = [x["trail"] for x in rows if x.get("trail") is not None]
        f21 = [v for v in (_fin21_of(x) for x in rows) if v is not None]
        return {"n": len(rows), "hit": hit, "rate": round(hit / len(rows), 3), "fallback": sum(1 for x in rows if x["status"] == "回落"), "miss": sum(1 for x in rows if x["status"] == "未命中"),
                "avg_fin": round(float(np.mean(cur)), 2) if cur else None, "win": round(float(np.mean([c > 0 for c in cur])), 3) if cur else None,
                "avg_rel": round(float(np.mean(rel)), 2) if rel else None, "trail_avg": round(float(np.mean(tr)), 2) if tr else None, "trail_win": round(float(np.mean([t > 0 for t in tr])), 3) if tr else None,
                # pr2：21 日報酬 (同回測口徑，停損列也追到 21 日)、真實發布 / 回填筆數
                "avg_fin21": round(float(np.mean(f21)), 2) if f21 else None, "n_fin21": len(f21), "win21": round(float(np.mean([v > 0 for v in f21])), 3) if f21 else None,
                "n_published": sum(1 for x in rows if not x.get("backfill")), "n_backfill": sum(1 for x in rows if x.get("backfill"))}
    oos = ((tm.get("oos") or {}).get("tiers") or {})
    pub = [x for x in done if _published(x)]
    by_tier = {}
    for t in ("A+", "A", "B+", "B"):
        g = agg([x for x in done if x.get("tier") == t])
        if g:
            bt = RADAR_TIER.get(t) or oos.get("A-" if t == "A" else t) or {}      # 10-07：優先用時點正確回測 (模型檔 oos 用今天的名單，偏樂觀)
            g["bt_hit"] = bt.get("hit"); g["bt_fin"] = bt.get("fin")
            g["role"] = ROLE[t]
            gp = agg([x for x in pub if x.get("tier") == t])
            g["published"] = {k: gp[k] for k in ("n", "hit", "rate", "avg_fin", "avg_fin21")} if gp else None
            g["below_bt"] = bool(g["n"] >= 15 and bt.get("hit") is not None and g["rate"] < bt["hit"] - 0.15)   # 資訊旗標 (含回填)，不是失準判定
            if g["role"] == "signal":
                g["expect"] = list(EXPECT_AAPLUS)
                g["drift"] = bool(gp and gp["n"] >= DRIFT_MIN_PUBLISHED and gp["rate"] < EXPECT_AAPLUS[0] - 0.15)   # 只看真實發布結案
            else:
                g["drift"] = False
                g["note"] = "描述級：不作失準判定 (時點正確回測 B 38% ≈ 同日池 38%，無選股力；上線 12~22% 反映 7 月大盤 −10.7%)"
            by_tier[t] = g
    sig_rows = [x for x in done if x.get("tier") in ("A+", "A")]
    signal = agg(sig_rows)
    if signal:
        signal["role"] = "signal"; signal["expect"] = list(EXPECT_AAPLUS)
        gp = agg([x for x in sig_rows if _published(x)])
        signal["published"] = {k: gp[k] for k in ("n", "hit", "rate", "avg_fin", "avg_fin21")} if gp else None
        signal["drift"] = bool(gp and gp["n"] >= DRIFT_MIN_PUBLISHED and gp["rate"] < EXPECT_AAPLUS[0] - 0.15)
        # r6：依記錄規則分開 (A30 = 10-07 起同回測規則；legacy = 之前前 20 / 追蹤上限 40 的舊規則，只留下回測 A 集合的一小部分)
        def _brief(rows):
            g = agg(rows)
            return {k: g[k] for k in ("n", "hit", "rate", "win21", "n_fin21", "n_published")} if g else None
        signal["by_rule"] = {REC_RULE_A: _brief([x for x in sig_rows if x.get("rec_rule") == REC_RULE_A]),
                             "legacy": _brief([x for x in sig_rows if x.get("rec_rule") != REC_RULE_A])}
    sig_all = [x for x in tre if x.get("tier") in ("A+", "A")]
    mbd = {sc_["date"]: sc_.get("mkt_bias20") for sc_ in (prev.get("scans") or []) if sc_.get("date") and sc_.get("mkt_bias20") is not None}
    st_rows = [x for x in sig_all if is_strict_a(x, tm, mbd)]
    st_done = [x for x in st_rows if x.get("status") != "追蹤"]
    g_st = agg(st_done)
    shadow = {"strict_A": {**({k: g_st[k] for k in ("n", "hit", "rate", "win21", "avg_fin21", "n_fin21", "n_published")} if g_st else {"n": 0}),
                           "tracking": len(st_rows) - len(st_done), "n_unknown": sum(1 for x in sig_all if is_strict_a(x, tm, mbd) is None),
                           "n_approx_th": sum(1 for x in st_rows if strict_a_th(x, tm)[1]), "n_signal": len(sig_all), "since": st_rows[0]["date"] if st_rows else None,
                           "rule": f"A/A+ 裡，大盤月線乖離 ≤ {SHADOW_STRICT['gate']}% 且模型分在前 {round((1 - SHADOW_STRICT['q']) * 100)}% 的推薦",
                           "note": "前瞻觀察，不是正式訊號；回測方向一致但差距不顯著 (radar5)，累積真實發布結案後再判斷"}}
    by_source = {"published": agg(pub), "backfill": agg([x for x in done if x.get("backfill")])}
    by_month = {}
    for x in done:
        by_month.setdefault(x["date"][:7], []).append(x)
    by_month = {m: {"n": len(v), "rate": round(sum(1 for x in v if x["status"] == "命中") / len(v), 3), "avg_fin": round(float(np.mean([x["cur"] for x in v if x.get("cur") is not None] or [0])), 2),
                    "avg_fin21": (lambda f: round(float(np.mean(f)), 2) if f else None)([q for q in (_fin21_of(x) for x in v) if q is not None]),
                    "n_A": sum(1 for x in v if x.get("tier") in ("A+", "A"))} for m, v in sorted(by_month.items())}
    sd = [x for x in sur if x.get("status") != "追蹤"]; trl = [x for x in sur if x.get("trail") is not None]
    sg_oos = ((tm.get("surge") or {}).get("oos") or {})
    surge = {"n": len(sur), "done": len(sd), "hits": sum(1 for x in sd if x["status"] == "飆"), "rate": round(sum(1 for x in sd if x["status"] == "飆") / len(sd), 3) if sd else None,
             "tracking": len(sur) - len(sd), "trail_n": len(trl), "trail_win": round(float(np.mean([x["trail"] > 0 for x in trl])), 3) if trl else None,
             "trail_avg": round(float(np.mean([x["trail"] for x in trl])), 2) if trl else None, "bt_hit": RADAR_BT["S3"]["surge"] if RADAR_BT.get("method") else sg_oos.get("app_hit", sg_oos.get("hit")),
             "bt_fin": RADAR_BT["S3"]["fin21"] if RADAR_BT.get("method") else sg_oos.get("app_fin", sg_oos.get("fin")),   # 10-07 時點正確回測
             "since": sur[0]["date"] if sur else None}
    # BE-7 (2026-10-13)：真實發布 / 回填分開；失準只看真實發布結案 (≥20 筆，同挖寶只看 published)，回填 (樣本內) 不再稀釋或觸發
    sp = [x for x in sd if _published(x)]; sph = sum(1 for x in sp if x["status"] == "飆")
    surge.update(n_published=len(sp), n_backfill=len(sd) - len(sp),
                 published=({"done": len(sp), "hits": sph, "rate": round(sph / len(sp), 3)} if sp else None))
    # N3：第 20 個交易日收盤報酬 (fin20；同回測口徑，提早出場的也追到第 20 日)
    f20 = [float(x["fin20"]) for x in sd if x.get("fin20") is not None]
    surge.update(n_fin20=len(f20), win20=round(float(np.mean([v > 0 for v in f20])), 3) if f20 else None, avg_fin20=round(float(np.mean(f20)), 2) if f20 else None)
    surge["drift"] = bool(sp and len(sp) >= 20 and surge["bt_hit"] is not None and sph / len(sp) < surge["bt_hit"] - 0.12)
    return {"treasure": {"all": agg(done), "signal": signal, "shadow": shadow, "by_source": by_source, "tracking": sum(1 for x in tre if x.get("status") == "追蹤"), "by_tier": by_tier, "by_month": by_month,
                         "since": tre[0]["date"] if tre else None, "roles": dict(ROLE), "expect_AAplus": list(EXPECT_AAPLUS), "note_tiers": NOTE_TIERS,
                         "def": "結案採回測定義：21 交易日、峰值取收盤、停損 −8% 取盤中低點 (先後順序)、相對大盤任一日 ≥ +4pt；與 App 本機帳本 (30 日曆天) 不同。"
                                "avg_fin = 結案日報酬 (含停損提早結案)；avg_fin21 = 第 21 個交易日報酬 (同回測口徑)",
                         "note_src": "published = 真實發布 (掃描當天記下)；backfill = 事後回填 (以 10-04 模型重算、等級門檻為樣本內)，真實發布結案 <15 筆前成績以回填為主"},
            "surge": surge}


# r6 10-07 各等級時點正確回測，訓練窗與正式相同 (2020 起) (A = 不含 A+；B = 不含 B+)：A+ 命中 65%、A 42%、B+ 43%、B 37% — 訊號主要在 A+ 與「大盤月線下的日子」
RADAR_TIER = {'A+': {'n': 128, 'hit': 0.648, 'fin': 12.37, 'win21': 0.813}, 'A': {'n': 170, 'hit': 0.424, 'fin': 3.99, 'win21': 0.594}, 'B+': {'n': 483, 'hit': 0.432, 'fin': 4.42, 'win21': 0.56}, 'B': {'n': 5475, 'hit': 0.37, 'fin': 1.64, 'win21': 0.484}}
RADAR_BT["path"], RADAR_BT["fail"], RADAR_BT["day0"], RADAR_BT["tiers"] = RADAR_PATH, RADAR_FAIL, RADAR_DAY0, RADAR_TIER
# N9：路徑表 / 分數組表 (final_tables_2020.py) 是 2021~ 樣本外 (連續路徑 2021~、冷啟動路徑 2022~)，和 radar_bt.period (2022-01~) 不同窗
RADAR_BT["path_period"] = "2021-01~2026-09 (冷啟動路徑 2022-01~)"
# BE-5：這些回測常數是 10-07 手動研究 (radar5/r6，訓練窗 2020 起) 的結果，不隨每週重訓更新；App 可據此標「回測研究日」
RADAR_BT["basis"] = {"research_date": "2026-10-07", "train_start": "2020-01-01"}
# r7 (2026-10-07) 除息研究 (scratchpad r7/tr + 反方重跑)：回測/帳本用未還原收盤價；含息重算現行推薦 A∪A+ 命中 .520→.532、勝率 .688→.701
# (10% 的 A 推薦 21 日內遇除息，6 月 A 命中 .12→.16、7 月 .49→.54)；改用含息報酬重新訓練 25 種子只 +0.7~1pt (與隨機擾動同級) → 不採用；
# 股利特徵 (近 12 月殖利率等) 點估計 +3~4pt 但集中 2025 → 需更多資料 (已開始每日存除權息預告表 data/exdiv_announce.jsonl)
RADAR_BT["div"] = {"A_hit": [0.520, 0.532], "A_win": [0.688, 0.701], "jun_hit": [0.119, 0.159], "jul_hit": [0.489, 0.539], "share": 0.101,
                   "retrain": "改用含息報酬重新訓練：25 個種子平均只多 0.7~1 個百分點，和隨機擾動同級 → 不採用",
                   "note": "回測與帳本用未還原收盤價 (除息日的下跌算虧損)；含息計算差異小，主要在 6~7 月"}
CLOSE_ALERT_DAYS = 4   # 10-06：最近幾個日曆日內結案的訊號級推薦發「結案」提醒 (A/A+ 與飆股；B 級描述級不發)


def weak_alerts(prev: dict | None) -> list[dict]:
    """10-06：追蹤中的 A/A+ 與飆股在第 3 天收盤 ≤ −5% → 「轉弱」提醒 (只在第 3 天發一次)，附歷史同樣情況的命中機率。"""
    if not prev:
        return []
    led = prev.get("ledger") or {}; out = []
    for kind, x in [("A", x) for x in led.get("treasure") or [] if x.get("tier") in ("A", "A+")] + [("S3", x) for x in led.get("surge") or []]:
        if x.get("status") != "追蹤" or x.get("days") != 3 or x.get("cur") is None or x["cur"] > -5:
            continue
        pp = path_prob(kind, 3, float(x["cur"]))
        if not pp:
            continue
        lab = f"💎 挖寶 {x['tier']}" if kind == "A" else "🚀 飆股"
        out.append({"level": "mid", "kind": "weak", "code": x["code"], "name": x.get("name", ""), "date": x.get("lastDate") or x["date"], "radar": kind,
                    "msg": f"⚠ {lab} 轉弱：{x['code']} {x.get('name', '')} ({x['date'][5:]} 推薦) 第 3 天 {x['cur']:+.1f}% — 過去同樣情況最後{'命中' if kind == 'A' else '飆'} {round(pp['hit'] * 100)}%、"
                               f"期滿仍賺 {round(pp['pos'] * 100)}% ({pp['n']} 筆)；這類推薦之後常回彈一部分 (期滿平均好於當下)，屬降低期待的提醒。歷史統計，非買賣建議。"})
    return out[:6]


def close_alerts(prev: dict | None, ref: str | None = None) -> list[dict]:
    """10-06：最近結案 (evalAt 在最新結案日往前 CLOSE_ALERT_DAYS 天內) 的 A/A+ 挖寶與飆股 → 「實際有沒有命中」主動告知。"""
    if not prev:
        return []
    led = prev.get("ledger") or {}
    rows = [("t", x) for x in led.get("treasure") or [] if x.get("tier") in ("A", "A+") and x.get("status") not in (None, "追蹤") and x.get("evalAt")] + \
           [("s", x) for x in led.get("surge") or [] if x.get("status") not in (None, "追蹤") and x.get("evalAt")]
    if not rows:
        return []
    last = ref or max(x["evalAt"] for _, x in rows)
    lo = (pd.Timestamp(last) - pd.Timedelta(days=CLOSE_ALERT_DAYS)).strftime("%Y-%m-%d")
    out = []
    for kind, x in sorted(rows, key=lambda z: (z[1]["evalAt"], z[1]["code"]), reverse=True):
        if x["evalAt"] < lo or x["evalAt"] > last:
            continue
        ok = x["status"] in ("命中", "飆")
        head = (f"💎 挖寶 {x['tier']}" if kind == "t" else "🚀 飆股") + f" 結案 {'✓' if ok else '↩' if x['status'] == '回落' else '✗'} {x['status']}"
        out.append({"level": "mid", "kind": "close", "code": x["code"], "name": x.get("name", ""), "date": x["evalAt"], "rec_date": x["date"], "radar": kind, "status": x["status"],
                    "msg": f"{head}：{x['code']} {x.get('name', '')} ({x['date'][5:]} 推薦)" + (f" — {x['reason']}" if x.get("reason") else "") + "。歷史紀錄，非買賣建議。"})
    return out[:6]


def tier_expect_txt() -> str:
    """r6：A+ 與 A 的歷史命中分開寫 (RADAR_TIER：時點正確回測、訓練窗與正式相同)，後面附 A/A+ 合計區間；缺值的等級略過。"""
    lo, hi = round(EXPECT_AAPLUS[0] * 100), round(EXPECT_AAPLUS[1] * 100)
    parts = []
    for t in ("A+", "A"):
        h = ((globals().get("RADAR_TIER") or {}).get(t) or {}).get("hit")
        if h is None and t == "A+":
            h = (RADAR_BT.get("A+") or {}).get("hit")
        if isinstance(h, (int, float)) and np.isfinite(h):
            parts.append(f"{t} 約 {round(h * 100)}%")
    # r6 review：A+ 樣本多集中在 2025-03~04 那一波 (3 個月區塊自助法區間碰到 0)；合計區間是 10 組模型種子的範圍，不是信賴區間
    return (f"命中 {'、'.join(parts)} (A/A+ 合計 {lo}~{hi}%；A+ 樣本多集中在 2025-04 一波，誤差大)" if parts else f"A/A+ 命中約 {lo}~{hi}%")


def alerts(sc: dict, st: dict, prev: dict | None = None) -> list[dict]:
    out = []
    D = sc.get("date") or ""
    exp_txt = tier_expect_txt()
    for r in [x for x in sc.get("treasure") or [] if x["tier"] in ("A+", "A")][:4]:
        # pr2：寫「模型分」不寫「命中機率」；r6：A+ / A 分開寫 (RADAR_TIER，訓練窗與正式相同)，另附同分數組歷史 (d0，參考)
        out.append({"level": "high", "kind": "treasure", "code": r["code"], "name": r["name"], "date": D, "tier": r["tier"], "p": r["p"], "d0": r.get("d0"),
                    "why": r.get("why") or [], "why_gate": r.get("why_gate") or "",
                    "msg": f"💎 挖寶 {r['tier']} ({D[5:]} 收盤)：{r['code']} {r['name']} 模型分 {r['p']:.2f}" + (f" — 原因：{'；'.join(r['why'])}" if r.get("why") else "")
                           + f" — 歷史統計 (時點正確回測)：訊號日收盤起算 21 個交易日，{exp_txt}"
                           + (f"、同分數組歷史約 {round(r['d0'] * 100)}%" if r.get("d0") is not None else "") + "；大盤月線下的日子整體較佳。非買賣建議。"})
    for r in (sc.get("surge") or [])[:3]:
        if r.get("ps") is not None and r["ps"] >= ((((st or {}).get("surge") or {}).get("th_top10")) or 0.3977):
            # N4 (2026-10-13)：ps 是模型原始分數 (未校準，平均高估 6~10pt) → 寫「飆股分」，機率只引用前 3 名回測實際飆股率
            out.append({"level": "mid", "kind": "surge", "code": r["code"], "date": D, "msg": f"🚀 飆股雷達 ({D[5:]})：{r['code']} {r['name']} 飆股分 {r['ps']:.2f} (前 3 名回測 20 日內先漲 20% 約 {round(RADAR_BT['S3']['surge'] * 100)}%；高風險)"})
    for t, g in ((st.get("treasure") or {}).get("by_tier") or {}).items():
        if g.get("role") == "signal" and g.get("drift") and g.get("published"):   # pr2：只對訊號級、以真實發布結案判定；B/B+ 不再警示
            gp = g["published"]
            out.append({"level": "mid", "kind": "drift", "code": "", "date": D, "msg": f"⚠ 挖寶 {t} 級真實發布命中 {round(gp['rate'] * 100)}% ({gp['n']} 筆結案)，低於預期 {round(EXPECT_AAPLUS[0] * 100)}~{round(EXPECT_AAPLUS[1] * 100)}% 達 15pt 以上：請檢視模型"})
    if (st.get("surge") or {}).get("drift"):
        s = st["surge"]; sp = s.get("published") or {"rate": s["rate"], "done": s["done"]}     # BE-7：只報真實發布結案
        out.append({"level": "mid", "kind": "drift", "code": "", "date": D, "msg": f"⚠ 飆股雷達真實發布飆股率 {round(sp['rate'] * 100)}% (回測 {round(s['bt_hit'] * 100)}%，{sp['done']} 筆結案)：實盤失準"})
    try:
        out += close_alerts(prev) + weak_alerts(prev)
    except Exception as e:  # noqa: BLE001
        log.debug("close/weak alerts: %s", e)
    return out


def backfill(prev: dict, tm: dict, dates: list[str], mk: dict | None = None, cal: TradeCal | None = None) -> dict:
    """回填：對過去的交易日逐日掃描並記入帳本 (紀錄標 backfill=True；用 D 日以前的日 K，與當日掃描口徑相同)。
    已在 scans 內的日期跳過。回填的推薦不是「當時真的發布過」，只用來快速累積上線口徑的追蹤樣本。"""
    have = {x["date"] for x in prev.get("scans") or []}
    mk = mk if mk is not None else twii_hist()
    for D in sorted(dates):
        if D in have:
            continue
        try:
            sc = scan(tm, D)
        except Exception as e:  # noqa: BLE001
            log.warning("backfill %s: %s", D, e); continue
        pl = sc.pop("pool", None)          # r14：候選池交給 build() 的 _pools (radar_archive，標 backfill)
        if sc.get("date") != D:
            continue
        if pl:
            prev.setdefault("_pools", []).append({**pl, "src": "backfill"})
        n0 = len(prev["ledger"]["treasure"]); n1 = len(prev["ledger"]["surge"])
        prev = record(prev, sc, tm, scan_src="backfill")
        for x in prev["ledger"]["treasure"][n0:] + prev["ledger"]["surge"][n1:]:
            x["backfill"] = True
        evaluate(prev, mk, max_fetch=10 ** 6, upto=D, cal=cal)       # 逐日結案 → 追蹤上限/60 日不重複 與即時一致 (K 棒已在快取)
        log.info("backfill %s: %d treasure / %d surge", D, len(prev["ledger"]["treasure"]) - n0, len(prev["ledger"]["surge"]) - n1)
    return prev


def build(date: str | None = None, do_scan: bool = True, backfill_days: int = 0) -> dict:
    tm = M.load_json("treasure_model") or {}
    prev = load_prev()
    n_tag = tag_late(prev)                                      # BE-2：既有的遲到重掃列改標 late / 回填 (冪等)
    if n_tag:
        log.info("treasure_live: %d 列遲到重掃改標 late (回填)", n_tag)
    mk = twii_hist()
    # BE-1：證交所交易日曆 (懶載入：帳本沒有要對帳的列就不抓)；FMTQIK 抓不到的月份以掃描日 ∪ ^TWII 的日子補洞
    cal = TradeCal(extra=lambda: set(mk) | {s_.get("date") for s_ in prev.get("scans") or [] if s_.get("date")})
    # BE-6：先掃當天、再回填其他日子 — 第一次 full run 被 GitHub 排程拖到午夜後時，「昨天以前」的回填會先把當天
    # (= 掃描日 D) 記成 backfill，之後的 live 掃描因同檔同日已在帳本而什麼都不記；所以 D 不進回填，留給下面依 is_late 判定。
    sc = scan(tm, date) if (do_scan and tm.get("trees")) else {"date": None}
    if backfill_days and tm.get("trees"):
        end = dt.datetime.now(config.TZ).date() - dt.timedelta(days=1)
        days = [d for d in sorted(mk) if (end - dt.timedelta(days=backfill_days)).isoformat() <= d <= end.isoformat() and d != sc.get("date")]
        prev = backfill(prev, tm, days, mk, cal=cal)
    pools = prev.pop("_pools", None) or []                      # r14：backfill 掃描的候選池
    pool = sc.pop("pool", None)                                 # r14：候選池不進 treasure_live.json / LOCAL (App 不讀)
    late = False
    if sc.get("date"):
        D = sc["date"]
        # BE-2：遲到的執行 (週日重訓後重掃上週五、平日休市 full run、MI_INDEX 未出就跑) 不把 D 日推薦記成真實發布：
        # D 已發布過 → 不記 (scans[D] 也不蓋)；還沒發布 → 記成 scan_src='late' + backfill=True。候選池照交 radar_archive (它有自己的 _rank)。
        late = is_late(D, dt.datetime.now(config.TZ), tm.get("trained_at"))
        if not late:
            prev = record(prev, sc, tm, scan_src="live")
        elif D in {s_.get("date") for s_ in prev.get("scans") or []}:
            log.info("treasure_live: %s 已發布過，這次是遲到的重掃 → 不記入帳本", D)
        else:
            prev = record(prev, sc, tm, scan_src="late")
            for x in prev["ledger"]["treasure"] + prev["ledger"]["surge"]:
                if x.get("date") == D and x.get("scan_src") == "late":
                    x["backfill"] = True
            log.info("treasure_live: %s 遲到的掃描 → 記成回填 (late)", D)
        if late:
            sc["late"] = True                                   # App 可據此不把重掃當成當天的推薦
        if pool:
            pools.append({**pool, "src": "live"})
        if sc.get("mkt_close") and D not in mk:                 # BE-3：Yahoo ^TWII 還沒有 D 日 → 用 scan 以盤後表補上的加權收盤 (同推薦時的 mktEntry)
            mk = {**mk, D: float(sc["mkt_close"])}
    n_fetch = evaluate(prev, mk, cal=cal)
    st = stats(prev, tm)
    sc_al = {**sc, "treasure": [], "surge": []} if late else sc   # BE-2：遲到的重掃不再發「新推薦」提醒 (結案 / 轉弱 / 失準照發)
    out = {"asof": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "scan": sc, "ledger": prev["ledger"], "scans": prev["scans"], "stats": st,
           "alerts": alerts(sc_al, st | {"surge": {**(st.get("surge") or {}), "th_top10": ((tm.get("surge") or {}).get("th_top10"))}}, prev),
           "model": {"radar_bt": RADAR_BT, "trained_at": tm.get("trained_at"), "th_A": tm.get("th_A"), "th_Aplus": tm.get("th_Aplus"), "oos": (tm.get("oos") or {}).get("tiers"), "surge_oos": (tm.get("surge") or {}).get("oos"),
                     "exit": tm.get("exit"), "surge_exit": (tm.get("surge") or {}).get("exit"), "entry": tm.get("entry"),
                     "expect_AAplus": list(EXPECT_AAPLUS), "note_tiers": NOTE_TIERS, "roles": dict(ROLE), "spec": "pr2 2026-10-05"},
           "n_fetch": n_fetch,
           "disclaimer": "雲端每日盤後掃描 (與 App 相同模型)；成績為歷史追蹤統計，不是個人化投資建議。"}
    try:
        LOCAL.parent.mkdir(parents=True, exist_ok=True)
        LOCAL.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.debug("local save: %s", e)
    out["_pools"] = pools    # r14：只給 tools/export_static.py → radar_archive.publish() (pop 掉；serialize.clean 也會丟掉 _ 開頭的鍵)
    return out
