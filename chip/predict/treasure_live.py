"""挖寶雷達 / 飆股雷達 雲端每日掃描 + 永久追蹤帳本 (2026-10-05)。

目的：App 的掃描只在裝置開著時跑、追蹤紀錄只存在各裝置 localStorage (加密同步，雲端讀不到)，
所以沒開 App 的日子沒有紀錄、換裝置看不到完整歷史。這裡在 GitHub Actions 盤後用「和 App 相同的模型與候選規則」
每天掃一次並永久歸檔，隔天起逐檔對帳，統計真正的上線成績 (依等級/年月)，失準時警示。

與 App (learning.js / app.js) 的對應：
- 候選池：TWSE 全市場盤後表 (MI_INDEX ALLBUT0999，同 App 13:35 後的 after-close 來源) + BWIBBU_ALL 本益比/淨值比 →
  scoring.js screen() 的 power 公式 (治學倍率 treasureMult 固定 1：那是各裝置自己的統計)，取前 80 並排除 ETF (代號 00 開頭) → 挖寶看前 40、飆股看前 80。
- 特徵：treasure.features (與 App tmFeatures 同款)；日 K 用 Yahoo 2 年 (Worker /idxh，失敗改直連)，D 日那根以盤後表取代 (含成交值)；大盤用 Yahoo ^TWII。
- 分級：p ≥ th_A 且大盤月線乖離 < 0 → A (≥ th_Aplus → A+)；高分但大盤在月線上 → B+；其餘 B。飆股 = 前 80 中 ps 前 3 名。
- 記錄：挖寶每天最多 6 檔 (池內前 20 依 p 排序、pct < 9.4、同檔 60 日內未命中不重複、追蹤中不超過 40)；飆股前 3 名 (同檔 28 日內不重複)。
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
- 帳本誠實化：每筆記 model_ver / th_A / th_Aplus / mkt_bias20 / scan_src (live|backfill)；停損列也追到第 21 日補 fin21 (cur 仍是結案日報酬)；
  stats 分 by_source (真實發布 vs 回填)、signal (A∪A+ 主數字)；漂移只對訊號級且真實發布 ≥15 筆判定，B/B+ 不再產生「失準」。
- 門檻 th_A 0.5062 / th_Aplus 0.6362、飆股 th_top10 0.3977 與前 3 名維持 (LC-05/06)；模型檔不動、無重訓。
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
from . import model as M
from . import treasure as T

log = logging.getLogger(__name__)
PAGES = "https://joekisoul-code.github.io/TaiexChipAnalyzer/data/treasure_live.json"
WORKER = "https://skynet-proxy.joekisoul.workers.dev"
LOCAL = config.CACHE_DIR / "treasure_live.json"
UA = {"User-Agent": "Mozilla/5.0"}
POOL, TREASURE_POOL, PER_DAY, MAX_ACTIVE, REENTRY_MISS_DAYS = 80, 40, 6, 40, 60
SURGE_DEDUP_DAYS = 28
LEDGER_MAX = 1500
EXPECT_AAPLUS = (0.44, 0.50)   # 10-07 誠實回測 (時點正確前 170 大，10 組 A∪A+ 命中 0.444~0.496)；舊 0.55~0.57 用今天的名單回測，有後見之明
ROLE = {"A+": "signal", "A": "signal", "B+": "descriptive", "B": "descriptive"}
DRIFT_MIN_PUBLISHED = 15       # 訊號級漂移判定：真實發布結案 ≥15 筆且命中低於預期下限 15pt
NOTE_TIERS = "A/A+ 為訊號級；B/B+ 為描述級 (回測即無選股力 #40/#84)"
MAX_FIN21_WAIT_DAYS = 60       # 結案列補 fin21：日 K 已延伸 60 個日曆日仍湊不到 21 根 (下市/停牌) → 放棄
WHY_TOP = 3                    # 10-06：A/A+ 原因 = 模型分下降最多的前 3 項
# 挖寶 × 飆股 回測 (10-07 改為 radar5 時點正確版；10-06 radar_study.py 的 170 檔固定名單有後見之明，已淘汰)
# hit = 挖寶定義命中、surge = 飆股定義、win21 = 第 21 交易日報酬 > 0、fin21 = 21 日平均報酬、q10 = 最差一成。
# 結論：兩邊同日都選到 (多為 B 級) 並沒有比較好 (時點正確版 n≈67 勝率 49%，A 級 62%) → 不另設「雙訊號」等級，只並列顯示。
RADAR_BT = {
    'A': {'n': 376, 'hit': 0.473, 'win21': 0.624, 'fin21': 5.41, 'med21': 3.85, 'q10': -14.97, 'win_rng': [0.592, 0.646], 'hit_rng': [0.444, 0.496]},
    'A+': {'n': 144, 'hit': 0.596, 'win21': 0.757, 'fin21': 10.05, 'med21': 6.93, 'q10': -7.84, 'win_rng': [0.735, 0.774], 'hit_rng': [0.571, 0.62]},
    'S3': {'n': 669, 'surge': 0.313, 'hit': 0.37, 'win21': 0.513, 'fin21': 4.55, 'med21': 0.67, 'q10': -17.73, 'win_rng': [0.493, 0.531], 'surge_rng': [0.301, 0.32]},
    'dual': {'n': 67, 'hit': 0.344, 'win21': 0.488, 'fin21': 2.5, 'med21': -0.35, 'q10': -19.31, 'win_rng': [0.417, 0.545], 'hit_rng': [0.266, 0.398]},
    'B': {'n': 5875, 'hit': 0.376, 'win21': 0.49, 'fin21': 1.95, 'med21': -0.2, 'q10': -14.23, 'win_rng': [0.488, 0.493], 'hit_rng': [0.373, 0.381]},
    "pool": {'n': 45218, 'hit': 0.38, 'win21': 0.5, 'fin21': 2.73, 'surge': 0.195, 'win20': 0.5},   # 同日掃描前 40 (飆股前 80) 的平均 = 不靠模型的基準 (只算已結案列)
    "period": "2022-01~2026-09", "note": "時點正確的前 170 大成交值股票 (每月依過去 120 日平均成交值，不用今天的名單)、逐年走動式樣本外、模擬每日掃描；5 個種子 × 2 條選股路徑平均，_rng = 10 組的最低~最高；同日兩邊都選到沒有比較好",
    # 10-07 更正：之前 (10-04~10-06) 的回測用 2026-09 當下的 170 檔名單去回測 2022~ → 事後才知道哪些股票會變大 (後見之明)，A 勝率高估約 5~9pt、飆股約 4~5pt
    "method": {"pit": True, "since": "2026-10-07", "old": {"A": {"win21": 0.717, "hit": 0.563}, "S3": {"win21": 0.554, "surge": 0.33}},
               "note": "舊回測用今天的名單 (事後才知道哪些股票變大)；現在改成每個時點只用當時的前 170 大"},
    # 2026 年 (1~9 月，模型只用 2025-12 以前資料訓練) — 跟「實際」同一年的樣本外對照
    "y2026": {"A": {'n': 59, 'hit': 0.53, 'win21': 0.753, 'fin21': 14.28, 'med21': 8.58, 'q10': -5.55, 'win_rng': [0.717, 0.817], 'hit_rng': [0.482, 0.567]},
              "S3": {'n': 147, 'surge': 0.364, 'hit': 0.382, 'win21': 0.578, 'fin21': 9.31, 'med21': 4.72, 'q10': -18.36, 'win_rng': [0.557, 0.599], 'surge_rng': [0.348, 0.384]},
              "B": {'n': 866, 'hit': 0.398, 'win21': 0.56, 'fin21': 6.05, 'med21': 2.48, 'q10': -15.94, 'win_rng': [0.551, 0.573], 'hit_rng': [0.391, 0.412]}},
    # 與雲端帳本「同一段期間」(推薦日 2026-07-01~09-03) 的樣本外回測 → 實際 (回填) 比它高的部分 ≈ 回填偏樂觀
    "path": None, "fail": None, "day0": None,   # 下方 RADAR_PATH / RADAR_FAIL / RADAR_DAY0，模組載入後填入
    # 10-07 候選池研究 (radar5/universe：全上市 1,032 檔、時點正確宇宙、5 種子)：限定前 170 大 vs 全市場 → A 勝率差不多 (−1pt)、A+ +4pt、飆股勝率 +1pt (都在噪音內)；
    # 之前說的「A 62→70%、飆股 47→55%」大部分是後見之明。保留限定 (與訓練一致；170 檔訓練卻選全市場的飆股顯著較差)。
    "pool_rule": {"rule": "候選池只用成交值前 170 大的股票 (與模型訓練一致)", "since": "2026-10-06",
             "before": {'A': {'n': 522, 'hit': 0.468, 'win21': 0.629, 'fin21': 6.59}, 'A+': {'hit': 0.565, 'win21': 0.715}, 'S3': {'n': 1057, 'surge': 0.323, 'win21': 0.486, 'fin21': 3.82, 'q10': -20.12}, 'B': {'win21': 0.458}},
             "after": {'A': {'n': 319, 'hit': 0.48, 'win21': 0.619, 'fin21': 6.32}, 'A+': {'hit': 0.62, 'win21': 0.752}, 'S3': {'n': 681, 'surge': 0.287, 'win21': 0.497, 'fin21': 3.72, 'q10': -16.91}, 'B': {'win21': 0.481}},
             "old_claim": {"A": [0.617, 0.695], "S3": [0.469, 0.552]}, "honest": True,
             "note": "時點正確重算：限定大型股對勝率幾乎沒影響 (都在噪音內)；之前的 62→70%、47→55% 主要來自用今天的名單回測"},
    "same": {"period": "2026-07~09",
             "A": {'n': 44, 'hit': 0.55, 'win21': 0.764, 'fin21': 14.81, 'med21': 9.53, 'q10': -4.36, 'win_rng': [0.721, 0.841], 'hit_rng': [0.488, 0.595]},
             "S3": {'n': 39, 'surge': 0.292, 'hit': 0.385, 'win21': 0.596, 'fin21': 5.25, 'med21': 3.6, 'q10': -22.98, 'win_rng': [0.474, 0.667], 'surge_rng': [0.211, 0.333]},
             "B": {'n': 216, 'hit': 0.317, 'win21': 0.52, 'fin21': 1.17, 'med21': 0.5, 'q10': -17.6, 'win_rng': [0.491, 0.537], 'hit_rng': [0.299, 0.339]}},
}
# 10-07 誠實重建 (radar5/final_tables.py：時點正確前 170 大、2020~ 樣本外、5 種子 × 2 路徑合併；n = 平均每組筆數)
# A：全部 A/A+ 推薦在第 k 天的收盤報酬 (含已跌破的) → 最後命中；走動式 Brier (表 vs 平均) 2023 0.1949/0.2478, 2024 0.211/0.2485, 2025 0.1553/0.25, 2026 0.1582/0.2519
# S3：只用「第 k~下一個 k 天仍在追蹤」(還沒碰到 +20% 或 −10%) 的推薦 — 舊表含已飆 +20% 結案的 → 後段高估 30~50pt；Brier 2023 0.1867/0.2061, 2024 0.1622/0.1954, 2025 0.1619/0.2037, 2026 0.192/0.2203
RADAR_PATH = {"A": {"1": [{"bin": "≤−8%", "n": 20, "hit": 0.0, "pos": 0.46, "avg": 2.47}, {"bin": "−8~−5%", "n": 27, "hit": 0.168, "pos": 0.469, "avg": 0.48}, {"bin": "−5~−2%", "n": 82, "hit": 0.359, "pos": 0.587, "avg": 4.92}, {"bin": "−2~+2%", "n": 215, "hit": 0.457, "pos": 0.603, "avg": 3.6}, {"bin": "+2~+5%", "n": 68, "hit": 0.608, "pos": 0.656, "avg": 8.11}, {"bin": "≥+5%", "n": 55, "hit": 0.903, "pos": 0.923, "avg": 19.26}], "2": [{"bin": "≤−8%", "n": 32, "hit": 0.0, "pos": 0.415, "avg": -0.2}, {"bin": "−8~−5%", "n": 24, "hit": 0.141, "pos": 0.34, "avg": -3.21}, {"bin": "−5~−2%", "n": 72, "hit": 0.338, "pos": 0.628, "avg": 4.19}, {"bin": "−2~+2%", "n": 169, "hit": 0.415, "pos": 0.56, "avg": 2.08}, {"bin": "+2~+5%", "n": 86, "hit": 0.591, "pos": 0.671, "avg": 7.94}, {"bin": "≥+5%", "n": 83, "hit": 0.892, "pos": 0.906, "avg": 19.13}], "3": [{"bin": "≤−8%", "n": 37, "hit": 0.0, "pos": 0.383, "avg": -1.2}, {"bin": "−8~−5%", "n": 28, "hit": 0.046, "pos": 0.384, "avg": -1.94}, {"bin": "−5~−2%", "n": 69, "hit": 0.165, "pos": 0.468, "avg": -0.74}, {"bin": "−2~+2%", "n": 135, "hit": 0.435, "pos": 0.562, "avg": 2.76}, {"bin": "+2~+5%", "n": 86, "hit": 0.645, "pos": 0.713, "avg": 7.71}, {"bin": "≥+5%", "n": 112, "hit": 0.862, "pos": 0.898, "avg": 17.56}], "5": [{"bin": "≤−8%", "n": 54, "hit": 0.015, "pos": 0.292, "avg": -5.47}, {"bin": "−8~−5%", "n": 26, "hit": 0.051, "pos": 0.473, "avg": -1.57}, {"bin": "−5~−2%", "n": 65, "hit": 0.262, "pos": 0.52, "avg": 0.89}, {"bin": "−2~+2%", "n": 107, "hit": 0.385, "pos": 0.537, "avg": 2.17}, {"bin": "+2~+5%", "n": 71, "hit": 0.617, "pos": 0.684, "avg": 5.66}, {"bin": "≥+5%", "n": 143, "hit": 0.829, "pos": 0.884, "avg": 17.39}], "10": [{"bin": "≤−8%", "n": 68, "hit": 0.056, "pos": 0.177, "avg": -10.55}, {"bin": "−8~−5%", "n": 30, "hit": 0.098, "pos": 0.26, "avg": -4.07}, {"bin": "−5~−2%", "n": 30, "hit": 0.151, "pos": 0.295, "avg": -4.31}, {"bin": "−2~+2%", "n": 79, "hit": 0.313, "pos": 0.529, "avg": 0.31}, {"bin": "+2~+5%", "n": 65, "hit": 0.521, "pos": 0.739, "avg": 6.93}, {"bin": "≥+5%", "n": 194, "hit": 0.789, "pos": 0.907, "avg": 17.22}], "15": [{"bin": "≤−8%", "n": 71, "hit": 0.001, "pos": 0.032, "avg": -15.4}, {"bin": "−8~−5%", "n": 32, "hit": 0.035, "pos": 0.146, "avg": -7.89}, {"bin": "−5~−2%", "n": 37, "hit": 0.141, "pos": 0.259, "avg": -3.34}, {"bin": "−2~+2%", "n": 60, "hit": 0.287, "pos": 0.603, "avg": 1.28}, {"bin": "+2~+5%", "n": 47, "hit": 0.474, "pos": 0.748, "avg": 4.13}, {"bin": "≥+5%", "n": 219, "hit": 0.806, "pos": 0.94, "avg": 18.4}]}, "S3": {"1": [{"bin": "≤−8%", "n": 31, "hit": 0.032, "pos": 0.391, "avg": -5.71, "nd": 31}, {"bin": "−8~−5%", "n": 48, "hit": 0.171, "pos": 0.46, "avg": 3.29, "nd": 48}, {"bin": "−5~−2%", "n": 142, "hit": 0.194, "pos": 0.42, "avg": 0.5, "nd": 142}, {"bin": "−2~+2%", "n": 348, "hit": 0.274, "pos": 0.487, "avg": 2.98, "nd": 348}, {"bin": "+2~+5%", "n": 135, "hit": 0.397, "pos": 0.56, "avg": 7.22, "nd": 135}, {"bin": "≥+5%", "n": 126, "hit": 0.602, "pos": 0.715, "avg": 15.69, "nd": 126}], "2": [{"bin": "≤−8%", "n": 14, "hit": 0.043, "pos": 0.362, "avg": -3.86, "nd": 14}, {"bin": "−8~−5%", "n": 71, "hit": 0.122, "pos": 0.298, "avg": -3.46, "nd": 71}, {"bin": "−5~−2%", "n": 152, "hit": 0.182, "pos": 0.44, "avg": 0.47, "nd": 152}, {"bin": "−2~+2%", "n": 234, "hit": 0.264, "pos": 0.48, "avg": 3.47, "nd": 234}, {"bin": "+2~+5%", "n": 132, "hit": 0.365, "pos": 0.556, "avg": 6.04, "nd": 132}, {"bin": "≥+5%", "n": 157, "hit": 0.612, "pos": 0.742, "avg": 14.87, "nd": 157}], "3": [{"bin": "≤−8%", "n": 25, "hit": 0.06, "pos": 0.278, "avg": -6.79, "nd": 25}, {"bin": "−8~−5%", "n": 122, "hit": 0.101, "pos": 0.327, "avg": -3.74, "nd": 146}, {"bin": "−5~−2%", "n": 185, "hit": 0.181, "pos": 0.409, "avg": 0.32, "nd": 228}, {"bin": "−2~+2%", "n": 273, "hit": 0.262, "pos": 0.492, "avg": 2.41, "nd": 351}, {"bin": "+2~+5%", "n": 182, "hit": 0.322, "pos": 0.578, "avg": 5.89, "nd": 212}, {"bin": "≥+5%", "n": 239, "hit": 0.583, "pos": 0.713, "avg": 14.82, "nd": 351}], "5": [{"bin": "≤−8%", "n": 49, "hit": 0.064, "pos": 0.205, "avg": -7.67, "nd": 59}, {"bin": "−8~−5%", "n": 153, "hit": 0.052, "pos": 0.256, "avg": -5.88, "nd": 261}, {"bin": "−5~−2%", "n": 221, "hit": 0.12, "pos": 0.362, "avg": -2.04, "nd": 381}, {"bin": "−2~+2%", "n": 285, "hit": 0.196, "pos": 0.473, "avg": 1.7, "nd": 596}, {"bin": "+2~+5%", "n": 212, "hit": 0.299, "pos": 0.576, "avg": 5.97, "nd": 366}, {"bin": "≥+5%", "n": 277, "hit": 0.534, "pos": 0.753, "avg": 13.26, "nd": 741}], "10": [{"bin": "≤−8%", "n": 26, "hit": 0.014, "pos": 0.17, "avg": -7.61, "nd": 29}, {"bin": "−8~−5%", "n": 89, "hit": 0.029, "pos": 0.237, "avg": -5.28, "nd": 147}, {"bin": "−5~−2%", "n": 133, "hit": 0.048, "pos": 0.313, "avg": -2.83, "nd": 233}, {"bin": "−2~+2%", "n": 189, "hit": 0.115, "pos": 0.445, "avg": 0.12, "nd": 389}, {"bin": "+2~+5%", "n": 157, "hit": 0.178, "pos": 0.593, "avg": 3.76, "nd": 273}, {"bin": "≥+5%", "n": 196, "hit": 0.403, "pos": 0.786, "avg": 10.9, "nd": 543}], "15": [{"bin": "≤−8%", "n": 18, "hit": 0.0, "pos": 0.066, "avg": -7.88, "nd": 23}, {"bin": "−8~−5%", "n": 60, "hit": 0.002, "pos": 0.14, "avg": -6.01, "nd": 106}, {"bin": "−5~−2%", "n": 95, "hit": 0.009, "pos": 0.198, "avg": -3.16, "nd": 175}, {"bin": "−2~+2%", "n": 127, "hit": 0.025, "pos": 0.399, "avg": -0.29, "nd": 265}, {"bin": "+2~+5%", "n": 99, "hit": 0.042, "pos": 0.73, "avg": 3.27, "nd": 166}, {"bin": "≥+5%", "n": 136, "hit": 0.214, "pos": 0.888, "avg": 9.58, "nd": 398}]}}
RADAR_FAIL = {
    "n": {'A': 376, 'S3': 669},
    "pre_note": '推薦當天看得到的約 60 項訊號，輸家與贏家幾乎分不開 (多數 AUC 0.44~0.56)；模型已經用掉大部分資訊',
    "A_market": {'lose_mvol': 1.3, 'hit_mvol': 2.43, 'lose_mb60': -3.3, 'hit_mb60': -9.0, 'yrs': '4/5', 'note': '描述：A 級失敗多在大盤溫和下跌、波動低時；命中多在急跌恐慌後。走動式濾網年數不足，未採用'},
    "S_ps": {'lose': 0.32, 'hit': 0.329, 'yrs': '4/5', 'wf_dhit': -0.009, 'note': '同為前 3 名，失敗與飆的飆股分數幾乎一樣；用之前年份濾掉最低兩成，命中沒有變好'},
    "early": {"A": {'rule': '第 3 天收盤 ≤ −5%', 'share': 0.129, 'hit': 0.019, 'lose': 0.627, 'hold': -2.62, 'exit': -11.7, 'n': 48},
              "S3": {'rule': '第 3 天收盤 ≤ −5%', 'share': 0.22, 'hit': 0.054, 'lose': 0.676, 'hold': -5.4, 'exit': -9.07, 'n': 147}},
}
# 10-07 推薦當天 A/A+ 分數校準：z = logit(p) − logit(th_A) 分 5 組 (邊界 edges)，各組歷史命中 (單調化)；走動式 Brier 2023 0.2373/0.2478, 2024 0.2568/0.2485, 2025 0.2245/0.25, 2026 0.2408/0.2519 (3/4 年較好，小幅)
RADAR_DAY0 = {"edges": [0.08, 0.184, 0.363, 0.679], "hit": [0.342, 0.386, 0.485, 0.517, 0.661], "n": [93, 93, 93, 93, 93], "base": 0.478, "note": "A/A+ 依推薦分數分 5 組的歷史命中 (2020~ 樣本外)；小幅優於只看等級平均，當參考"}

_PATH_EDGES = (-8, -5, -2, 2, 5)


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


# ------------------------------------------------------------------ 評分 (與 App tmScore 相同)
def _mk_frame(mk: dict, upto: str) -> pd.DataFrame:
    d = sorted(k for k in mk if k <= upto)
    s = pd.Series([mk[k] for k in d], index=d)
    return pd.DataFrame({"date": d, "m_close": s.values, "m_ret1": s.pct_change().values * 100, "m_ret20": s.pct_change(20).values * 100,
                         "m_bias20": (s / s.rolling(20).mean() - 1).values * 100})


def score_one(code: str, snap_row: dict, D: str, mkf: pd.DataFrame, tm: dict) -> dict | None:
    b = bars(code)
    rows = [{"date": str(x["date"])[:10], "open": x.get("open"), "high": x.get("high"), "low": x.get("low"), "close": x["close"], "volume": x.get("volume") or 0, "amount": np.nan}
            for x in b if x.get("close") and str(x["date"])[:10] <= D]
    if not rows:
        return None
    r = snap_row
    bar = {"date": D, "open": r["open"] if r["open"] > 0 else r["close"], "high": r["high"] if r["high"] > 0 else r["close"], "low": r["low"] if r["low"] > 0 else r["close"],
           "close": r["close"], "volume": r["volume"] or (rows[-1]["volume"] if rows[-1]["date"] == D else 0), "amount": r["value"]}
    if rows[-1]["date"] == D:
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
    """當日掃描：回傳 {date, pool_n, treasure[(前 40 中 pct<9.4、依 p 排序)], surge[前 3], mkt_close, mkt_bias20, errors}。"""
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
    for r in pool:
        try:
            s = score_one(r["code"], r, D, mkf, tm)
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
                t["why"] = [w["txt"] for w in explain(tm, res[t["code"]]["_x"], med)]
                t["why_gate"] = gate_txt(t["tier"], t.get("m_bias20"))
                t["d0"] = day0_prob(t.get("p"), tm.get("th_A"))      # 10-07 同分數組歷史命中 (參考)
            except Exception as e:  # noqa: BLE001
                log.debug("explain %s: %s", t["code"], e)
    tre.sort(key=lambda x: -x["p"])
    sg_k = int(((tm.get("surge") or {}).get("def") or {}).get("topk") or 3)
    sur = sorted([{"code": r["code"], "name": r["name"], "close": r["close"], "pct": round(r["pct"], 2), "ps": res[r["code"]]["ps"]} for r in pool if r["code"] in res and res[r["code"]]["ps"] is not None and r["pct"] < 9.4],
                 key=lambda x: -x["ps"])[:sg_k]
    return {"date": D, "pool_n": len(pool), "scored_n": len(res), "errors": errors, "treasure": tre, "surge": sur, "pool_rule": "universe" if tm.get("universe") else "all", "uni_n": len(tm.get("universe") or []),
            "mkt_close": mk.get(D), "mkt_bias20": round(float(mkf["m_bias20"].iloc[-1]), 3) if len(mkf) and pd.notna(mkf["m_bias20"].iloc[-1]) else None,
            "mkt_pct": round(mkt_pct, 2)}


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
                    if (x.get("status") != "追蹤" and old.get("status") == "追蹤") or (len(json.dumps(x)) > len(json.dumps(old)) and old.get("status") == "追蹤")                             or (old.get("status") != "追蹤" and x.get("status") == old.get("status") and x.get("fin21") is not None and old.get("fin21") is None):   # pr2：已結案但另一份已補 fin21
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


def record(prev: dict, sc: dict, tm: dict | None = None, scan_src: str = "live") -> dict:
    """把本日掃描結果記入帳本 (同 App recordDiscoveries / recordSurge 的規則)。
    pr2：每筆加 model_ver (掃描時模型 trained_at)、th_A / th_Aplus (當時門檻)、mkt_bias20 (閘門狀態)、scan_src ('live' 真實發布 | 'backfill' 回填)，
    讓帳本不再路徑相依、可事後重算等級。"""
    D = sc.get("date")
    if not D:
        return prev
    tm = tm or {}
    meta = {"model_ver": tm.get("trained_at"), "th_A": tm.get("th_A"), "th_Aplus": tm.get("th_Aplus"), "mkt_bias20": sc.get("mkt_bias20"), "scan_src": scan_src}
    led = prev["ledger"]
    tre = led["treasure"]
    active = [x for x in tre if x.get("status") == "追蹤"]
    act_codes = {x["code"] for x in active}
    recent_miss = {x["code"] for x in tre if x.get("status") == "未命中" and x.get("evalAt") and (pd.Timestamp(D) - pd.Timestamp(x["evalAt"])).days < REENTRY_MISS_DAYS}
    have = {(x["code"], x["date"]) for x in tre}
    n = 0
    s_codes = {r["code"] for r in sc.get("surge") or []}                          # 10-06：今日飆股前 3
    t_tier = {r["code"]: (r["tier"], r["p"]) for r in sc.get("treasure") or []}   # 今日挖寶候選 (含 B 級) 的等級與模型分
    for r in sc["treasure"][:20]:
        if n >= PER_DAY or len(active) + n >= MAX_ACTIVE:
            break
        if r["code"] in act_codes or r["code"] in recent_miss or (r["code"], D) in have:
            continue
        tre.append({"code": r["code"], "name": r["name"], "date": D, "entry": r["close"], "mktEntry": sc.get("mkt_close"), "p": r["p"], "tier": r["tier"], "ps": r.get("ps"),
                    "pct": r["pct"], "status": "追蹤", "peak": 0.0, "trough": 0.0, "cur": 0.0, "rel": 0.0, "days": 0, **meta,
                    **({"why": r["why"]} if r.get("why") else {}), **({"sPick": True} if r["code"] in s_codes else {})})
        n += 1
    sur = led["surge"]
    for r in sc["surge"]:
        if any(x["code"] == r["code"] and abs((pd.Timestamp(D) - pd.Timestamp(x["date"])).days) < SURGE_DEDUP_DAYS for x in sur):
            continue
        tt = t_tier.get(r["code"])
        sur.append({"code": r["code"], "name": r["name"], "date": D, "entry": r["close"], "ps": r["ps"], "status": "追蹤", "days": 0,
                    **({"tier": tt[0], "p": tt[1]} if tt else {}),
                    "model_ver": meta["model_ver"], "th_top10": (tm.get("surge") or {}).get("th_top10"), "scan_src": scan_src})
    scans = [x for x in prev.get("scans") or [] if x.get("date") != D]
    scans.append({"date": D, "n_treasure": len(sc["treasure"]), "n_A": sum(1 for r in sc["treasure"] if r["tier"] in ("A", "A+")), "mkt_bias20": sc.get("mkt_bias20"),
                  "top": [{"code": r["code"], "tier": r["tier"], "p": r["p"]} for r in sc["treasure"][:6]], "surge": [{"code": r["code"], "ps": r["ps"]} for r in sc["surge"]]})
    prev["scans"] = scans[-400:]
    for kind in ("treasure", "surge"):
        led[kind] = led[kind][-LEDGER_MAX:]
    return prev


def _fin21_of(x: dict):
    """第 21 個交易日收盤報酬 (回測口徑)：新列有 fin21；舊結案列若在第 21 日結案，cur 即 fin21。"""
    if x.get("fin21") is not None:
        return x["fin21"]
    if x.get("status") not in (None, "追蹤") and (x.get("days") or 0) >= T.H and x.get("cur") is not None:
        return x["cur"]
    return None


def _walk_treasure(since: list[dict], e: float, me, mk: dict) -> dict:
    """回測定義逐日走訪 (與原 _eval_treasure 迴圈同邏輯)；10-06 另記第一次達標 / 跌破 / 峰值的交易日序與日期，讓 App 能顯示「實際有沒有命中、哪天」。"""
    cm = -1e9; stop = False; reached = False; peak = -1e9; trough = 1e9; rel_max = -1e9; w: dict = {}
    for k, r in enumerate(since, 1):
        d = str(r["date"])[:10]
        cr = (float(r["close"]) / e - 1) * 100; lr = (float(r.get("low") or r["close"]) / e - 1) * 100
        cm = max(cm, cr); trough = min(trough, lr)
        if cr > peak:
            peak = cr; w["peakDay"], w["peakDate"] = k, d
        mr = ((mk.get(d) or float("nan")) / me - 1) * 100 if me else 0.0
        if not np.isfinite(mr):
            mr = 0.0
        rel_max = max(rel_max, cr - mr)
        if not stop and not reached and lr <= T.STOP and cm < T.TGT:
            stop = True; w["stopDay"], w["stopDate"], w["stopVal"] = k, d, round(lr, 2)
        if not stop and (cr >= T.TGT or (cr - mr) >= T.REL):
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


def _annotate_closed_treasure(x: dict, since: list[dict], mk: dict) -> None:
    """已結案列補「第幾天」資訊 (一次性：chk=1)；狀態/數值不動。"""
    n = int(x.get("days") or 0)
    if n <= 0 or len(since) < n:
        return
    me = x.get("mktEntry") or (mk.get(x["date"]) if x["date"] in mk else None)
    w = _walk_treasure(since[:n], float(x["entry"]), me, mk)
    for k in ("hitDay", "hitDate", "hitVal", "hitBy", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate"):
        if k in w:
            x[k] = w[k]
    if x.get("cur") is not None and x.get("status") in ("命中", "回落", "未命中"):
        x["reason"] = _treasure_reason(x["status"], w, float(x["cur"]), T.H)
    x["chk"] = 1


def _eval_treasure(x: dict, b: list[dict], mk: dict) -> None:
    """回測定義 (treasure.features 的 label)：21 交易日；峰值取收盤、停損取盤中低點 (先後順序)、相對大盤任一日 ≥ 4pt。
    pr2：fin21 = 第 21 個交易日收盤報酬，不論是否已停損 (停損後仍追到 21 日)；已結案列只補 fin21 (cur/peak/days 等結案值不動)。"""
    e = float(x["entry"]); since = [r for r in b if str(r["date"])[:10] > x["date"] and r.get("close")][:T.H]
    if not since:
        return
    if len(since) >= T.H and x.get("fin21") is None:
        x["fin21"] = round((float(since[T.H - 1]["close"]) / e - 1) * 100, 2); x["fin21Date"] = str(since[T.H - 1]["date"])[:10]
    elif len(since) < T.H and x.get("fin21") is None and (pd.Timestamp(str(b[-1]["date"])[:10]) - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
        x["fin21_na"] = True      # 日 K 不足 (下市/停牌/資料斷)：不再重試
    if x.get("status") not in (None, "追蹤"):
        if x.get("chk") is None:      # 10-06：舊結案列補「第幾天達標/跌破」
            _annotate_closed_treasure(x, since, mk)
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
    stop, reached, peak, trough, rel_max = w["stop"], w["reached"], w["peak"], w["trough"], w["rel_max"]
    for k in ("hitDay", "hitDate", "hitVal", "hitBy", "stopDay", "stopDate", "stopVal", "peakDay", "peakDate"):
        if k in w:
            x[k] = w[k]
    x["chk"] = 1
    cur = (float(since[-1]["close"]) / e - 1) * 100
    last_m = mk.get(str(since[-1]["date"])[:10]); rel = (cur - ((last_m / me - 1) * 100)) if (me and last_m) else None
    x.update(peak=round(peak, 2), trough=round(trough, 2), cur=round(cur, 2), rel=round(rel, 2) if rel is not None else None, relMax=round(rel_max, 2), days=len(since), lastDate=str(since[-1]["date"])[:10])
    # 移動停利 (A 級出場建議：漲 10% 後自最高收盤回落 8%)
    if x.get("trail") is None:
        pk = e
        for r in since:
            pk = max(pk, float(r["close"]))
            if pk >= e * 1.1 and float(r["close"]) <= pk * 0.92:
                x["trail"] = round((float(r["close"]) / e - 1) * 100, 2); x["trailDate"] = str(r["date"])[:10]; break
    if x.get("status") == "追蹤" and (stop or len(since) >= T.H):
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
    e = float(x["entry"]); since = [r for r in b if str(r["date"])[:10] > x["date"] and r.get("close")][:T.SURGE_N]
    if not since:
        return
    w = _walk_surge(since, e)
    peak, res, trail, tdate = w["peak"], w["res"], w["trail"], w["tdate"]
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


def evaluate(prev: dict, mk: dict, max_fetch: int = 200, upto: str | None = None) -> int:
    """逐檔對帳 (未結案或尚未出場的紀錄)；每次最多抓 max_fetch 檔 (控制 Yahoo 呼叫)。upto：只用該日 (含) 以前的 K 棒 (回填時逐日模擬)。"""
    n = 0
    todo = []
    for x in prev["ledger"]["treasure"]:
        if x.get("status") == "追蹤" or x.get("trail") is None or (x.get("fin21") is None and not x.get("fin21_na") and (x.get("days") or 0) < T.H) or x.get("chk") is None:
            todo.append(("t", x))     # pr2：停損提早結案的列繼續追到第 21 日補 fin21；10-06：舊結案列補一次「第幾天」(chk)
    for x in prev["ledger"]["surge"]:
        if x.get("status") == "追蹤" or (x.get("trail") is None and (x.get("days") or 0) < T.SURGE_N) or x.get("chk") is None:
            todo.append(("s", x))
    cache: dict[str, list] = {}
    for kind, x in todo[:max_fetch * 2]:
        if n >= max_fetch and x["code"] not in cache:
            break
        try:
            b = cache.get(x["code"])
            if b is None:
                b = bars(x["code"]); cache[x["code"]] = b; n += 1          # 與評分同一份 2 年日 K 快取
            if upto:
                b = [r for r in b if str(r["date"])[:10] <= upto]
            if not b:
                # 取不到任何日 K (下市/改代號)：已結案、尚缺 fin21 的列超過 MAX_FIN21_WAIT_DAYS 個日曆日 → 標 fin21_na，不再每次重抓
                if kind == "t" and x.get("status") not in (None, "追蹤") and x.get("fin21") is None and not x.get("fin21_na"):
                    ref = pd.Timestamp(upto) if upto else pd.Timestamp(dt.datetime.now(config.TZ).strftime("%Y-%m-%d"))
                    if (ref - pd.Timestamp(x["date"])).days > MAX_FIN21_WAIT_DAYS:
                        x["fin21_na"] = True
                continue
            (_eval_treasure(x, b, mk) if kind == "t" else _eval_surge(x, b))
        except Exception as e:  # noqa: BLE001
            log.debug("evaluate %s: %s", x["code"], e)
    return n


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
    pub = [x for x in done if not x.get("backfill")]
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
                g["note"] = "描述級：不作失準判定 (時點正確回測 B 37% ≈ 同日池 38%，無選股力；上線 12~22% 反映 7 月大盤 −10.7%)"
            by_tier[t] = g
    sig_rows = [x for x in done if x.get("tier") in ("A+", "A")]
    signal = agg(sig_rows)
    if signal:
        signal["role"] = "signal"; signal["expect"] = list(EXPECT_AAPLUS)
        gp = agg([x for x in sig_rows if not x.get("backfill")])
        signal["published"] = {k: gp[k] for k in ("n", "hit", "rate", "avg_fin", "avg_fin21")} if gp else None
        signal["drift"] = bool(gp and gp["n"] >= DRIFT_MIN_PUBLISHED and gp["rate"] < EXPECT_AAPLUS[0] - 0.15)
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
    surge["drift"] = bool(surge["done"] >= 20 and surge["bt_hit"] is not None and surge["rate"] is not None and surge["rate"] < surge["bt_hit"] - 0.12)
    return {"treasure": {"all": agg(done), "signal": signal, "by_source": by_source, "tracking": sum(1 for x in tre if x.get("status") == "追蹤"), "by_tier": by_tier, "by_month": by_month,
                         "since": tre[0]["date"] if tre else None, "roles": dict(ROLE), "expect_AAplus": list(EXPECT_AAPLUS), "note_tiers": NOTE_TIERS,
                         "def": "結案採回測定義：21 交易日、峰值取收盤、停損 −8% 取盤中低點 (先後順序)、相對大盤任一日 ≥ +4pt；與 App 本機帳本 (30 日曆天) 不同。"
                                "avg_fin = 結案日報酬 (含停損提早結案)；avg_fin21 = 第 21 個交易日報酬 (同回測口徑)",
                         "note_src": "published = 真實發布 (掃描當天記下)；backfill = 事後回填 (以 10-04 模型重算、等級門檻為樣本內)，真實發布結案 <15 筆前成績以回填為主"},
            "surge": surge}


# 10-07 各等級時點正確回測 (A = 不含 A+；B = 不含 B+)：A 級 (90~97 分位) 本身只比 B+ 差不多，訊號主要在 A+ 與「大盤月線下的日子」
RADAR_TIER = {'A+': {'n': 144, 'hit': 0.596, 'fin': 10.05, 'win21': 0.757}, 'A': {'n': 233, 'hit': 0.398, 'fin': 2.54, 'win21': 0.542}, 'B+': {'n': 443, 'hit': 0.462, 'fin': 5.68, 'win21': 0.576}, 'B': {'n': 5432, 'hit': 0.369, 'fin': 1.64, 'win21': 0.483}}
RADAR_BT["path"], RADAR_BT["fail"], RADAR_BT["day0"], RADAR_BT["tiers"] = RADAR_PATH, RADAR_FAIL, RADAR_DAY0, RADAR_TIER
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
                               f"期滿仍賺 {round(pp['pos'] * 100)}% ({pp['n']} 筆)；歷史上這時候離場平均比放到期滿差，降低期待即可。歷史統計，非買賣建議。"})
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


def alerts(sc: dict, st: dict, prev: dict | None = None) -> list[dict]:
    out = []
    D = sc.get("date") or ""
    for r in [x for x in sc.get("treasure") or [] if x["tier"] in ("A+", "A")][:4]:
        # pr2：寫「模型分」不寫「命中機率」；10-07 預期值改用時點正確回測 (A∪A+ 44~50%、A+ 約 6 成)，另附同分數組歷史 (d0，參考)
        out.append({"level": "high", "kind": "treasure", "code": r["code"], "name": r["name"], "date": D, "tier": r["tier"], "p": r["p"], "d0": r.get("d0"),
                    "why": r.get("why") or [], "why_gate": r.get("why_gate") or "",
                    "msg": f"💎 挖寶 {r['tier']} ({D[5:]} 收盤)：{r['code']} {r['name']} 模型分 {r['p']:.2f}" + (f" — 原因：{'；'.join(r['why'])}" if r.get("why") else "")
                           + f" — 歷史統計：訊號日收盤起算 21 個交易日，A/A+ 命中約 {round(EXPECT_AAPLUS[0] * 100)}~{round(EXPECT_AAPLUS[1] * 100)}% (A+ 約 {round(RADAR_BT['A+']['hit'] * 100)}%)"
                           + (f"、同分數組歷史約 {round(r['d0'] * 100)}%" if r.get("d0") is not None else "") + "；大盤月線下的日子整體較佳。非買賣建議。"})
    for r in (sc.get("surge") or [])[:3]:
        if r.get("ps") is not None and r["ps"] >= ((((st or {}).get("surge") or {}).get("th_top10")) or 0.3977):
            out.append({"level": "mid", "kind": "surge", "code": r["code"], "date": D, "msg": f"🚀 飆股雷達 ({D[5:]})：{r['code']} {r['name']} 20 日內先漲 20% 機率 {round(r['ps'] * 100)}% (回測前 3 名約 {round(RADAR_BT['S3']['surge'] * 100)}%；高風險)"})
    for t, g in ((st.get("treasure") or {}).get("by_tier") or {}).items():
        if g.get("role") == "signal" and g.get("drift") and g.get("published"):   # pr2：只對訊號級、以真實發布結案判定；B/B+ 不再警示
            gp = g["published"]
            out.append({"level": "mid", "kind": "drift", "code": "", "date": D, "msg": f"⚠ 挖寶 {t} 級真實發布命中 {round(gp['rate'] * 100)}% ({gp['n']} 筆結案)，低於預期 {round(EXPECT_AAPLUS[0] * 100)}~{round(EXPECT_AAPLUS[1] * 100)}% 達 15pt 以上：請檢視模型"})
    if (st.get("surge") or {}).get("drift"):
        s = st["surge"]; out.append({"level": "mid", "kind": "drift", "code": "", "date": D, "msg": f"⚠ 飆股雷達上線飆股率 {round(s['rate'] * 100)}% (回測 {round(s['bt_hit'] * 100)}%，{s['done']} 筆)：實盤失準"})
    try:
        out += close_alerts(prev) + weak_alerts(prev)
    except Exception as e:  # noqa: BLE001
        log.debug("close/weak alerts: %s", e)
    return out


def backfill(prev: dict, tm: dict, dates: list[str], mk: dict | None = None) -> dict:
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
        if sc.get("date") != D:
            continue
        n0 = len(prev["ledger"]["treasure"]); n1 = len(prev["ledger"]["surge"])
        prev = record(prev, sc, tm, scan_src="backfill")
        for x in prev["ledger"]["treasure"][n0:] + prev["ledger"]["surge"][n1:]:
            x["backfill"] = True
        evaluate(prev, mk, max_fetch=10 ** 6, upto=D)       # 逐日結案 → 追蹤上限/60 日不重複 與即時一致 (K 棒已在快取)
        log.info("backfill %s: %d treasure / %d surge", D, len(prev["ledger"]["treasure"]) - n0, len(prev["ledger"]["surge"]) - n1)
    return prev


def build(date: str | None = None, do_scan: bool = True, backfill_days: int = 0) -> dict:
    tm = M.load_json("treasure_model") or {}
    prev = load_prev()
    mk = twii_hist()
    if backfill_days and tm.get("trees"):
        end = dt.datetime.now(config.TZ).date() - dt.timedelta(days=1)
        days = [d for d in sorted(mk) if (end - dt.timedelta(days=backfill_days)).isoformat() <= d <= end.isoformat()]
        prev = backfill(prev, tm, days, mk)
    sc = scan(tm, date) if (do_scan and tm.get("trees")) else {"date": None}
    if sc.get("date"):
        prev = record(prev, sc, tm, scan_src="live")
    n_fetch = evaluate(prev, mk)
    st = stats(prev, tm)
    out = {"asof": dt.datetime.now(config.TZ).strftime("%Y-%m-%d %H:%M:%S"), "scan": sc, "ledger": prev["ledger"], "scans": prev["scans"], "stats": st,
           "alerts": alerts(sc, st | {"surge": {**(st.get("surge") or {}), "th_top10": ((tm.get("surge") or {}).get("th_top10"))}}, prev),
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
    return out
