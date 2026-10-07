"""r6 (10-07)：把 final/final_constants_2020.json (final_tables_2020.py，訓練窗與正式相同 2020 起) 寫進 chip/predict/treasure_live.py
(RADAR_BT / RADAR_PATH / RADAR_FAIL / RADAR_DAY0 / RADAR_TIER / EXPECT_AAPLUS 與相關註解、B 級說明的數字)。
用法：python patch_py_2020.py [treasure_live.py 路徑]   (預設 D:/TaiexChipAnalyzer/chip/predict/treasure_live.py)
舊值 (10-07 早上 radar5 2018 起訓練版) 從 research/radar5/final_constants.json 讀，記進 RADAR_BT["method"]["prev_2018"]。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_cj = HERE / "final" / "final_constants_2020.json"                      # final_tables_2020.py 的輸出位置
C = json.loads((_cj if _cj.exists() else HERE / "final_constants_2020.json").read_text(encoding="utf-8"))   # research/radar5 內放在同一層
OLD = json.loads(Path("D:/TaiexChipAnalyzer/research/radar5/final_constants.json").read_text(encoding="utf-8"))
p = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/TaiexChipAnalyzer/chip/predict/treasure_live.py")
raw = p.read_bytes(); crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
assert C["train_start"] == "2020-01-01" and len(C["sets"]) == 10, (C["train_start"], C["sets"])


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:90], s.count(a)); s = s.replace(a, b)


def block(start, end_marker, new, end_inclusive_line=False):
    """把 s 中從 start 開始、到 end_marker (含) 的區段換成 new。end_inclusive_line：end_marker 所在行整行都換掉。"""
    global s
    if isinstance(start, tuple):   # 第一次 (10-07 早上版的註解) / 重跑 (本腳本寫的註解) 都找得到
        start = next(x for x in start if x in s)
    i0 = s.index(start); i1 = s.index(end_marker, i0)
    i1 = s.index("\n", i1) + 1 if end_inclusive_line else i1 + len(end_marker)
    s = s[:i0] + new + s[i1:]


pc = lambda v: f"{v * 100:.0f}%"
bt, y26, same, pool, PR, SP = C["bt"], C["y2026"], C["same"], C["pool"], C.get("pool_rule"), C["sparse"]
ob = OLD["bt"]
EXP = (round(bt["A"]["hit_rng"][0], 2), round(bt["A"]["hit_rng"][1], 2))

# ---------------- RADAR_BT ----------------
L = ['RADAR_BT = {']
for k in ("A", "A+", "S3", "dual", "B"):
    L.append(f'    {k!r}: {bt[k]!r},')
L.append(f'    "pool": {pool!r},   # 同日掃描前 40 (飆股前 80) 的平均 = 不靠模型的基準 (只算已結案列)')
L.append('    "period": "2022-01~2026-09", "note": "時點正確的前 170 大成交值股票 (每月依過去 120 日平均成交值，不用今天的名單)、訓練窗與正式相同 (2020 起)、逐年走動式樣本外、模擬每日掃描；5 個種子 × 2 條選股路徑平均，_rng = 10 組的最低~最高；同日兩邊都選到沒有比較好",')
L.append(f'    "sparse": {SP!r},   # 每月平均推薦數 / 有推薦的月份 (A 級只在大盤月線下的日子出現 → 很多月份沒有)')
L.append('    # 10-07 更正：之前 (10-04~10-06) 的回測用 2026-09 當下的 170 檔名單去回測 2022~ → 事後才知道哪些股票會變大 (後見之明)，A 勝率高估、飆股也偏高')
L.append('    # r6 10-07：radar5 重建時訓練窗用了 2018 起 (研究工具預設)，正式模型 treasure.train 是 2020 起 → 改成訓練窗與正式相同 (2020 起) 重算；')
L.append(f'    #   A∪A+ 命中 {ob["A"]["hit"]:.3f}→{bt["A"]["hit"]:.3f}、勝率 {ob["A"]["win21"]:.3f}→{bt["A"]["win21"]:.3f}；兩個窗的差在 3 個月區塊 bootstrap 下不顯著 (r6 radarshadow 驗證)，只是讓 App 顯示的回測和正式模型一致')
L.append('    "method": {"pit": True, "since": "2026-10-07", "train_start": "2020-01-01", "old": {"A": {"win21": 0.717, "hit": 0.563}, "S3": {"win21": 0.554, "surge": 0.33}},')
prev18 = {"A": {"win21": ob["A"]["win21"], "hit": ob["A"]["hit"]}, "A+": {"win21": ob["A+"]["win21"], "hit": ob["A+"]["hit"]}, "S3": {"win21": ob["S3"]["win21"], "surge": ob["S3"]["surge"]}}
L.append(f'               "prev_2018": {prev18!r},   # 10-07 早上 (radar5，2018 起訓練) 的版本')
L.append('               "note": "舊回測用今天的名單 (事後才知道哪些股票變大)；現在改成每個時點只用當時的前 170 大，訓練窗與正式相同 (2020 起)"},')
L.append('    # 2026 年 (1~9 月，模型只用 2025-11 以前資料訓練) — 跟「實際」同一年的樣本外對照')
L.append(f'    "y2026": {{"A": {y26["A"]!r},')
L.append(f'              "S3": {y26["S3"]!r},')
L.append(f'              "B": {y26["B"]!r}}},')
L.append('    # 與雲端帳本「同一段期間」(推薦日 2026-07-01~09-03) 的樣本外回測 → 實際 (回填) 比它高的部分 ≈ 回填偏樂觀')
L.append('    "path": None, "fail": None, "day0": None,   # 下方 RADAR_PATH / RADAR_FAIL / RADAR_DAY0，模組載入後填入')
if PR:
    d = PR["diff"]
    dd = lambda k: f'{d[k]["d"] * 100:+.1f}pt [{d[k]["ci95"][0] * 100:+.1f}, {d[k]["ci95"][1] * 100:+.1f}]'
    L.append(f'    # r6 候選池研究 (訓練窗與正式相同 2020 起；限定 PIT 前 170 大 vs 全上市訓練+候選，{len(PR["seeds"])} 種子 × 2 路徑；差值 = 限定 − 全上市，95% = 3 個月區塊 bootstrap)：')
    L.append(f'    #   A 勝率 {dd("A_win")}、A 命中 {dd("A_hit")}、A+ 勝率 {dd("A+_win")}、飆股勝率 {dd("S3_win")}、飆股率 {dd("S3_surge")}、B 勝率 {dd("B_win")}')
    NM = {"A_win": "A 勝率", "A_hit": "A 命中", "A+_win": "A+ 勝率", "S3_win": "飆股勝率", "S3_surge": "飆股率", "B_win": "B 級 (描述級) 勝率"}
    sig = [k for k, v in d.items() if not v["noise"]]
    noise_sig = all(d[k]["noise"] for k in ("A_win", "A_hit", "A+_win", "S3_win", "S3_surge"))   # App 顯示的 A / A+ / 飆股
    note = ("時點正確重算 (訓練窗與正式相同 2020 起)：限定前 170 大的 A / A+ / 飆股" + ("都略好但在誤差內" if noise_sig and all(d[k]["d"] >= 0 for k in ("A_win", "A+_win", "S3_win")) else "差別在誤差內" if noise_sig else "有超出誤差的差別")
            + ("" if not sig else "；" + "、".join(NM[k] + ("較高" if d[k]["d"] > 0 else "較低") + " (超出誤差)" for k in sig))
            + "；之前的 62→70%、47→55% 主要來自用今天的名單回測")
    L.append('    "pool_rule": {"rule": "候選池只用成交值前 170 大的股票 (與模型訓練一致)", "since": "2026-10-06", "window": "2022-01~2026-09，訓練窗與正式相同 (2020 起)",')
    L.append(f'             "before": {PR["before"]!r},')
    L.append(f'             "after": {PR["after"]!r},')
    L.append(f'             "diff": {PR["diff"]!r}, "within_noise": {noise_sig!r},   # within_noise = App 顯示的 A / A+ / 飆股差值都在 95% 範圍內 (B 級另見 diff)')
    L.append('             "old_claim": {"A": [0.617, 0.695], "S3": [0.469, 0.552]}, "honest": True,')
    L.append(f'             "note": {note!r}}},')
else:
    L.append('    # r6：2020 窗的全上市對照沒重算 (見 radar5/universe：2018 窗下限定 vs 全上市都在噪音內)')
    L.append('    "pool_rule": {"rule": "候選池只用成交值前 170 大的股票 (與模型訓練一致)", "since": "2026-10-06", "before": None, "after": None,')
    L.append('             "old_claim": {"A": [0.617, 0.695], "S3": [0.469, 0.552]}, "honest": True,')
    L.append('             "note": "限定大型股 vs 全上市的差別在誤差內 (radar5 時點正確研究)；之前的 62→70%、47→55% 主要來自用今天的名單回測"},')
L.append('    "same": {"period": "2026-07~09",')
L.append(f'             "A": {same["A"]!r},')
L.append(f'             "S3": {same["S3"]!r},')
L.append(f'             "B": {same["B"]!r}}},')
L.append('}')
block("RADAR_BT = {", "\n}\n", "\n".join(L) + "\n")

# ---------------- RADAR_PATH ----------------
cm = ("# r6 10-07 (final_tables_2020.py：時點正確前 170 大、訓練窗與正式相同 (2020 起)、2021~ 樣本外、5 種子 × 2 路徑合併；n = 平均每組筆數)\n"
      "# A：全部 A/A+ 推薦在第 k 天的收盤報酬 (含已跌破的) → 最後命中；走動式 Brier (表 vs 平均) " + ", ".join(f"{w['year']} {w['table']}/{w['base']}" for w in C["wf_path"]["A"]) + "\n"
      "# S3：只用「第 k~下一個 k 天仍在追蹤」(還沒碰到 +20% 或 −10%) 的推薦 — 舊表含已飆 +20% 結案的 → 後段高估 30~50pt；Brier " + ", ".join(f"{w['year']} {w['table']}/{w['base']}" for w in C["wf_path"]["S3"]) + "\n")
block(("# 10-07 誠實重建", "# r6 10-07 (final_tables_2020.py"), "RADAR_PATH = ", cm + "RADAR_PATH = " + json.dumps(C["path"], ensure_ascii=False) + "\n", end_inclusive_line=True)

# ---------------- RADAR_FAIL + RADAR_DAY0 ----------------
F = C["fail"]
F["S_ps"]["note"] = "同為前 3 名，失敗與飆的飆股分數幾乎一樣；用之前年份濾掉最低兩成，命中沒有變好" if F["S_ps"]["wf_dhit"] <= 0.005 else F["S_ps"]["note"]
FL = ['RADAR_FAIL = {', f'    "n": {F["n"]!r},', f'    "pre_note": {F["pre_note"]!r},', f'    "A_market": {F["A_market"]!r},', f'    "S_ps": {F["S_ps"]!r},',
      f'    "early": {{"A": {F["early"]["A"]!r},', f'              "S3": {F["early"]["S3"]!r}}},', '}']
D0 = C["day0"]
better = sum(1 for w in D0["wf"] if w["cal"] < w["const"])
FL.append("# r6 10-07 推薦當天 A/A+ 分數校準 (訓練窗與正式相同 2020 起)：z = logit(p) − logit(th_A) 分 5 組 (邊界 edges)，各組歷史命中 (單調化)；走動式 Brier "
          + ", ".join(f"{w['year']} {w['cal']}/{w['const']}" for w in D0["wf"]) + f" ({better}/{len(D0['wf'])} 年較好，小幅)")
FL.append(f'RADAR_DAY0 = {{"edges": {D0["edges"]!r}, "hit": {D0["hit"]!r}, "n": {D0["n"]!r}, "base": {D0["base"]!r}, "note": "A/A+ 依推薦分數分 5 組的歷史命中 (2021~ 樣本外，訓練窗與正式相同 2020 起)；小幅優於只看等級平均，當參考"}}')
block("RADAR_FAIL = {", "RADAR_DAY0 = ", "\n".join(FL) + "\n", end_inclusive_line=True)

# ---------------- RADAR_TIER ----------------
TI = C["tiers"]
tier = {"A+": TI["A+"], "A": TI["A-"], "B+": TI["B+"], "B": TI["Bo"]}
block(("# 10-07 各等級時點正確回測", "# r6 10-07 各等級時點正確回測"), "RADAR_TIER = ",
      f"# r6 10-07 各等級時點正確回測，訓練窗與正式相同 (2020 起) (A = 不含 A+；B = 不含 B+)：A+ 命中 {pc(tier['A+']['hit'])}、A {pc(tier['A']['hit'])}、B+ {pc(tier['B+']['hit'])}、B {pc(tier['B']['hit'])}"
      " — 訊號主要在 A+ 與「大盤月線下的日子」\n"
      f"RADAR_TIER = {tier!r}\n", end_inclusive_line=True)

# ---------------- EXPECT_AAPLUS 與其他註解 ----------------
i0 = s.index("EXPECT_AAPLUS = ("); i1 = s.index("\n", i0)
s = s[:i0] + f"EXPECT_AAPLUS = {EXP!r}   # r6 10-07：時點正確前 170 大、訓練窗與正式相同 (2020 起)，10 組 A∪A+ 命中 {bt['A']['hit_rng'][0]:.3f}~{bt['A']['hit_rng'][1]:.3f}；10-07 早上的 0.44~0.50 是 2018 起訓練的窗，更早的 0.55~0.57 有後見之明" + s[i1:]
i0 = s.index("# 結論：兩邊同日都選到"); i1 = s.index("\n", i0)
s = s[:i0] + f"# 結論：兩邊同日都選到 (多為 B 級) 並沒有比較好 (時點正確版 n≈{bt['dual']['n']} 勝率 {pc(bt['dual']['win21'])}，A 級 {pc(bt['A']['win21'])}) → 不另設「雙訊號」等級，只並列顯示。" + s[i1:]
i0 = s.index('g["note"] = "描述級：不作失準判定 (時點正確回測 B '); i1 = s.index("\n", i0)
s = s[:i0] + f'g["note"] = "描述級：不作失準判定 (時點正確回測 B {pc(bt["B"]["hit"])} ≈ 同日池 {pc(pool["hit"])}，無選股力；上線 12~22% 反映 7 月大盤 −10.7%)"' + s[i1:]
if "  r6 (10-07，訓練窗與正式相同" not in s:
    rep("- 帳本誠實化：每筆記 model_ver",
        f"  r6 (10-07，訓練窗與正式相同 2020 起；radar5 用了 2018 起)：預期改 {EXP[0] * 100:.0f}~{EXP[1] * 100:.0f}% (A+ {pc(tier['A+']['hit'])}、A {pc(tier['A']['hit'])})；"
        "A/A+ 帳本記錄規則改同回測 (30 日內未記過就記，不受前 20 / 追蹤上限)，警報 A+ 與 A 分開寫。\n- 帳本誠實化：每筆記 model_ver")
out = s.replace("\n", "\r\n") if crlf else s
p.write_bytes(out.encode("utf-8"))
print("ok", p, len(s), "EXPECT", EXP, "crlf", crlf)
