"""資料來源 adapters：twse / tpex / taifex / finmind / histock / wantgoo。

共用例外 (margin_dyn 資料層，2026-10-14)：
  NoData  來源明確回「沒有資料」：休市、週間假日、尚未公布、查詢日期超出範圍 (TWSE 三者同一回應 C1，交易日判定由呼叫端用
          twse.holidays 做)。不可快取；回補端在交易日收到時寫 margin_nodata 後人工覆核。
  NotYet  已能判定「交易日但尚未公布」(TPEx 0 列 + 交易日曆 + 時刻)。不可快取；下次再試。
兩者皆為 SourceUnavailable 子類、彼此不互為子類 (catch NoData 不會吃到 NotYet)。
"""
from __future__ import annotations


class SourceUnavailable(RuntimeError):
    """來源暫時或永久沒有這一天的資料 (不是程式錯誤)。"""


class NoData(SourceUnavailable):
    """來源回「沒有符合條件的資料」/ 查詢日期超出範圍：休市、週間假日、尚未公布 (不可分辨，交由交易日曆)。"""


class NotYet(SourceUnavailable):
    """交易日但尚未公布 (只有能判定時刻的來源會丟)。"""
