"""10-20 投資口徑 (treasure_live._inv_fill / RADAR_BT["inv"])：隔日開盤買、第 H 日收盤賣、扣來回 0.585%、照規則、同期加權。

    python tests/test_treasure_inv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from chip.predict import treasure_live as TL  # noqa: E402

BANNED = ["買進", "賣出", "加碼", "減碼", "放空", "做多", "做空", "建議買", "建議賣", "抄底", "逃頂", "進場", "持有"]


def bars(closes, opens=None, start="2026-09-01"):
    ds = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    opens = opens or closes
    return [{"date": d, "open": o, "high": max(o, c), "low": min(o, c), "close": c} for d, o, c in zip(ds, opens, closes)]


def main():
    H = 21
    # 推薦日收盤 100、隔日開盤 102、之後一路到第 21 日收盤 110
    cl = [100] + [102 + i * 8 / 20 for i in range(21)]
    op = [100, 102] + cl[2:]
    b = bars(cl, op)
    mk = {b[0]["date"]: 20000.0, b[21]["date"]: 20400.0}
    x = {"date": b[0]["date"], "code": "1234", "entry": 100.0}
    TL._inv_fill(x, b, mk, H)
    exp = round((110 * (1 - TL.INV_SELL) / (102 * (1 + TL.INV_BUY)) - 1) * 100, 2)
    assert x["inv_ret"] == exp, (x["inv_ret"], exp)
    assert abs(x["inv_ret"] - 7.24) < 0.05, x["inv_ret"]                 # 110/102 −1 = 7.84%，扣 0.585% 左右
    assert x["inv_trail"] == exp and x["inv_trailDate"] == b[21]["date"]   # 沒觸發照規則 → 同期滿
    assert x["inv_date"] == b[21]["date"] and x["inv_mkt"] == 2.0 and x["inv_gap"] == 2.0
    # 只填一次
    x["inv_ret"] = 99.0; TL._inv_fill(x, b, mk, H); assert x["inv_ret"] == 99.0

    # 照規則：開盤 100 → 漲到 115 (≥ +10%) → 回落到 105 (≤ 115×0.92=105.8) 出場；期滿 120
    cl2 = [100, 100, 108, 115, 112, 105] + [110 + i for i in range(15)] + [120]
    b2 = bars(cl2[:22])
    y = {"date": b2[0]["date"], "code": "1", "entry": 100.0}
    TL._inv_fill(y, b2, {}, H)
    assert y["inv_trailDate"] == b2[5]["date"], y
    assert y["inv_trail"] == round((105 * (1 - TL.INV_SELL) / (100 * (1 + TL.INV_BUY)) - 1) * 100, 2)
    assert y["inv_ret"] > y["inv_trail"] and y["inv_mkt"] is None

    # 日 K 不足：未超過 60 天 → 不填、不標 na；超過 → inv_na
    z = {"date": b[0]["date"], "code": "2", "entry": 100.0}
    TL._inv_fill(z, b[:10], mk, H); assert "inv_ret" not in z and not z.get("inv_na")
    old = bars([100] * 10, start="2026-01-01") + [{"date": "2026-06-01", "open": 100, "high": 100, "low": 100, "close": 100}]
    z2 = {"date": "2026-01-01", "code": "3", "entry": 100.0}
    TL._inv_fill(z2, old, {}, H); assert z2.get("inv_na") is True
    # 開盤價缺 → inv_na
    b3 = bars(cl, op); b3[1]["open"] = None
    w = {"date": b3[0]["date"], "code": "4", "entry": 100.0}; TL._inv_fill(w, b3, mk, H); assert w.get("inv_na") is True

    # _todo：缺 inv_ret 的舊結案列要處理 (need)；已有 inv_ret 或 inv_na 的不因此列入
    done = {"code": "5", "date": "2026-08-01", "status": "命中", "trail": 1.0, "fin21": 3.0, "days": 21, "chk": 1, "pxk": 1}
    prev = {"ledger": {"treasure": [dict(done), dict(done, inv_ret=1.0), dict(done, inv_na=True)],
                       "surge": [dict(done, status="飆", fin20=2.0), dict(done, status="飆", fin20=2.0, inv_ret=0.5)]}}
    td = TL._todo(prev)
    assert [(k, bool(n)) for k, _, n in td] == [("t", True), ("s", True)], td
    # 價格基準修復時一起清掉
    assert {"inv_ret", "inv_trail", "inv_mkt", "inv_na"} <= set(TL.PX_KEYS)

    # 回測口徑常數：五個等級、每組數字合理；收盤口徑與 RADAR_TIER 一致；字串無禁用字
    IV = TL.RADAR_BT["inv"]
    assert set(IV["tiers"]) == {"A+", "A", "B+", "B", "S3"} and IV["cost"] == 0.585 and IV["n_sets"] == 10
    for k in ("A+", "A", "B+", "B"):
        assert abs(IV["tiers"][k]["win_close"] - TL.RADAR_TIER[k]["win21"]) < 0.002, k
        assert IV["tiers"][k]["win"] < IV["tiers"][k]["win_close"], k     # 隔日開盤 + 成本 → 勝率較低
    assert abs(IV["tiers"]["S3"]["win_close"] - TL.RADAR_BT["S3"]["win21"]) < 0.002
    assert IV["tiers"]["A+"]["win"] > IV["tiers"]["A"]["win"] > IV["tiers"]["B"]["win"]
    for w_ in BANNED:
        assert w_ not in IV["rule"], w_
    print("test_treasure_inv: 8 checks passed")


if __name__ == "__main__":
    main()
