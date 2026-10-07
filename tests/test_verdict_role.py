"""判斷總結 (pr2 P3) 單元測試：合成 fc、verdict/confidence 表以 patch 取代、不連網。

    python tests/test_verdict_role.py
    python -m pytest tests/test_verdict_role.py

variant night → bucket_role == 'info'、verdict 無 ‧高共識/‧分歧 後綴、head 以信心分層領頭、action 不含「縮小部位」；
variant base → 'filter'、head 字串與舊版相同 (迴歸)；action 自 r6 (2026-10-07) 改中性描述 (下緣/上緣參考、轉弱，無承接/減碼/停損/縮小部位)。
"""
from __future__ import annotations

import contextlib
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chip.predict import model as M  # noqa: E402
from chip.predict import verdict as VD  # noqa: E402

VT = {"tiers": {
    "1_base": {"hit_all": 0.562, "bucket": {"高共識": {"n": 573, "hit": 0.628, "cov": 0.338, "yr_min": 0.485}, "一般": {"n": 1025, "hit": 0.535, "cov": 0.604, "yr_min": 0.431}, "分歧": {"n": 98, "hit": 0.459, "cov": 0.058, "yr_min": 0.231}},
               "nocall": {"n": 917, "base_up": 0.549, "by_net": {"4": {"n": 61, "up": 0.705, "yr_min": 0.45}, "3": {"n": 150, "up": 0.594, "yr_min": 0.5}}}},
    "1_night": {"hit_all": 0.858, "bucket": {"高共識": {"n": 243, "hit": 0.86, "cov": 0.362, "yr_min": 0.738}, "一般": {"n": 399, "hit": 0.855, "cov": 0.595, "yr_min": 0.802}, "分歧": {"n": 29, "hit": 0.897, "cov": 0.043, "yr_min": 0.8}}}}}
CT = {"tiers": {"1_night": {"高": {"n": 350, "cov": 0.304, "hit": 0.903, "yr_min": 0.881}, "中": {"n": 319, "cov": 0.277, "hit": 0.812, "yr_min": 0.702}},
                "1_base": {"高": {"n": 578, "cov": 0.221, "hit": 0.607, "yr_min": 0.508}}}}


@contextlib.contextmanager
def tables():
    old = M.load_json
    M.load_json = lambda name: copy.deepcopy({"verdict": VT, "confidence": CT}.get(name))
    try:
        yield
    finally:
        M.load_json = old


def _fc(variant, conf_tier="高", conf_hit=None, call="偏多", strength="強"):
    nd = {"n": 1, "date": "2026-10-06", "variant": variant, "call": call, "call_strength": strength, "p_up": 0.85, "base_hit": 0.55,
          "level": 49775, "buy_at": 49189, "sell_at": 50298, "stop": 48962, "target": 50466}
    if conf_tier:
        nd.update({"conf_tier": conf_tier, "conf_hit": conf_hit if conf_hit is not None else CT["tiers"][f"1_{variant}"][conf_tier]["hit"], "conf_cov": CT["tiers"][f"1_{variant}"][conf_tier]["cov"]})
    return {"date": "2026-10-02", "close": 49500.0, "next_days": [nd], "horizons": {},
            "patterns": {"direction1": "偏空", "score1": -0.5}, "deep": {"today": {"hsi_r0": -1.2, "kospi_r0": -0.8, "smart2": -1.5}}, "trend7": {"available": False}}


def test_night_mode_info_role():
    snap = {"phase": "closed", "tx_night": {"change_pct": 1.2, "final": True}}
    with tables():
        V = VD.build(_fc("night"), None, snap, None, None)
    assert V["variant"] == "night" and V["bucket_role"] == "info"
    assert V["agree"] == 1 and V["disagree"] == 4 and V["net"] == -3 and V["bucket"] == "分歧"          # 票數仍算、仍輸出 (資訊)
    assert V["verdict"] == "偏多" and "‧" not in V["verdict"]
    assert V["head"] == "模型偏多強；信心分層 高 (歷史 90%，覆蓋 30%)；其他 6 票 淨 -3 (夜盤模式下票數不加分，僅供參考)", V["head"]
    assert V["action"] == "下緣參考 49,189，跌破 48,962 轉弱，上緣 (一成機率) 50,466", V["action"]   # r6 中性用語
    assert "縮小部位" not in V["action"] and "縮小部位" not in V["head"] and "縮小部位" not in V["text"]
    assert V["conf_tier"] == "高" and V["conf_oos"] == CT["tiers"]["1_night"]["高"] and V["conf_hit"] == 0.903
    # 桶的 OOS 仍附上 (資訊)；r6：n=29 < 30 → hit/yr_min 為 null，原值留 hit_raw
    assert V["oos"]["n"] == 29 and V["oos"]["hit"] is None and V["oos"]["yr_min"] is None and V["oos"]["hit_raw"] == 0.897 and V["oos"]["small_n"] is True
    assert V["call"] == "偏多" and V["call_action"] == "偏多"
    # 偏空 + 分歧 (夜盤) 也不加「不追空」
    with tables():
        fc = _fc("night", call="偏空", strength="")
        fc["patterns"] = {"direction1": "偏多", "score1": 0.5}; fc["deep"]["today"] = {"hsi_r0": 1.2, "kospi_r0": 0.8, "smart2": 1.5}
        V2 = VD.build(fc, None, {"phase": "closed", "tx_night": {"change_pct": -1.2, "final": True}}, None, None)
    assert V2["bucket"] == "分歧" and V2["verdict"] == "偏空" and "不追空" not in V2["action"] and V2["bucket_role"] == "info"
    # 夜盤無信心分層 (理論上不會發生) → head 顯示 — 不崩
    with tables():
        V3 = VD.build(_fc("night", conf_tier=None), None, snap, None, None)
    assert V3["head"].startswith("模型偏多強；信心分層 —；其他 6 票") and V3["conf_oos"] is None


def test_base_mode_unchanged():
    snap = {"phase": "closed"}
    with tables():
        V = VD.build(_fc("base"), None, snap, None, None)
    assert V["variant"] == "base" and V["bucket_role"] == "filter"
    assert len(V["votes"]) == 5 and V["agree"] == 0 and V["disagree"] == 4 and V["net"] == -4 and V["bucket"] == "分歧"
    assert V["verdict"] == "偏多‧分歧"
    assert V["head"] == "模型偏多強，其他 5 票同向 0、反向 4 (淨 -4) → 分歧；同狀況歷史 OOS 命中 46% (覆蓋 6%，逐年最低 23%)；信心分層 高 (61%)", V["head"]
    assert V["action"] == "下緣參考 49,189，跌破 48,962 轉弱，上緣 (一成機率) 50,466；訊號分歧：叫牌可信度較低", V["action"]   # r6 中性用語
    assert V["conf_tier"] == "高" and V["conf_oos"] == CT["tiers"]["1_base"]["高"]
    assert V["text"] == f"判斷總結：{V['verdict']}。{V['head']}。{V['action']}。"
    # 高共識 (全同向) 的後綴與 head 格式不變
    with tables():
        fc = _fc("base")
        fc["patterns"] = {"direction1": "偏多", "score1": 0.5}; fc["deep"]["today"] = {"hsi_r0": 1.2, "kospi_r0": 0.8, "smart2": 1.5}
        V2 = VD.build(fc, None, snap, None, None)
    assert V2["verdict"] == "偏多‧高共識" and V2["net"] == 4 and "→ 高共識；同狀況歷史 OOS 命中 63% (覆蓋 34%，逐年最低 48%)" in V2["head"] and V2["bucket_role"] == "filter"


def test_no_call_unchanged():
    with tables():
        fc = _fc("base", conf_tier=None, call="中性", strength="")
        fc["patterns"] = {"direction1": "偏多", "score1": 0.5}; fc["deep"]["today"] = {"hsi_r0": 1.2, "kospi_r0": 0.8, "smart2": 1.5}
        V = VD.build(fc, None, {"phase": "closed"}, None, None)
    # 既有行為：訊號共識把 cs 轉 1 → bucket 以 net=0 落在「一般」、oos 換成未叫牌表 (不改)
    assert V["verdict"] == "偏多‧訊號共識" and V["call_action"] == "偏多" and V["bucket"] == "一般" and V["bucket_role"] == "filter" and V["conf_tier"] is None and V["conf_oos"] is None
    assert V["oos"] == {"n": 61, "hit": 0.705, "cov": None, "yr_min": 0.45}
    assert "模型無叫牌但 5 票中淨多 4" in V["head"] and V["action"] == "下緣參考 49,189，跌破 48,962 轉弱，上緣 (一成機率) 50,466"


def test_learn_extra_uses_recent_flag():
    with tables():
        fc = _fc("base")
        fc["next_days"][0].update({"recent_hit": 0.56, "recent_n": 88, "recent_up_rate": 0.558, "recent_flag": "below"})
        V = VD.build(fc, None, {"phase": "closed"}, None, None)
    ln = [e for e in V["extra"] if e["key"] == "learn"][0]["note"]
    assert "近期實際命中 56%/88 次 (同期上漲率 56%)" in ln and "不改叫牌" in ln and "校準" not in ln and "降級" not in ln


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
