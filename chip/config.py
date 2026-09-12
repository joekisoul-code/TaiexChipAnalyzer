"""設定：路徑、Token、快取時間。全部可用環境變數覆寫。"""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CHIP_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "chip.sqlite"
TZ = ZoneInfo("Asia/Taipei")

# FinMind：免費不需 token（300 次/小時）；註冊後填 token 可提高到 600 次/小時，
# 贊助等級才有「八大行庫個股」「券商分點」資料集。
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# 玩股網需要真實瀏覽器 (Playwright)。設 CHIP_WANTGOO=0 可停用。
WANTGOO_ENABLED = os.getenv("CHIP_WANTGOO", "1") == "1"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# 快取秒數
TTL_REALTIME = 5          # 即時指數
TTL_INTRADAY = 60 * 10    # 盤中會變動的盤後表 (當日尚未公布時重試)
TTL_DAILY = 60 * 60 * 6   # 日資料
TTL_HISTORY = 60 * 60 * 12

# 分析參數
HISTORY_DAYS = 400        # 抓多少天歷史做統計

for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
