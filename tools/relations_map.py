"""資料關係圖 (pr2 relations_map，2026-10-05)：把研究產出的 relations_map_app.json 驗證後複製進 data/models/relations_map.json 並蓋 asof。

    python tools/relations_map.py <src.json>            # 驗證 + 安裝到 data/models/relations_map.json
    python tools/relations_map.py <src.json> --check    # 只驗證、不寫
    python tools/relations_map.py --check               # 驗證已安裝的檔

靜態檔、不需每日重算 (81 條有效邊 / 28 節點 / 13 類無效關係；沒有一條新邊進模型)。tools/export_static.py 每次 (full 與 fast) 都
dump 成 site/data/relations_map.json 給 App 畫八層時間軸 + 等級徽章 + 無效面板。所有邊 tradable=false (#74)，不畫買/賣字樣。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "models" / "relations_map.json"
SCHEMA = "pr2.relations_map_app/1.0"
TOP_REQUIRED = ("schema", "generated", "disclaimer", "grade_legend", "layers", "nodes", "edges", "rejected", "summary")
LAYER_REQUIRED = ("zh", "order", "kind", "desc")
NODE_REQUIRED = ("id", "label", "kind", "group")
EDGE_REQUIRED = ("id", "source", "target", "layer", "kind", "usable_at", "horizon", "sign", "ic", "rho", "pos_years", "wf_hit",
                 "grade", "grade_researcher", "production", "label", "note", "caveat", "tradable")
KINDS = {"predictive", "descriptive", "calibration"}
GRADES = {"A", "B", "C"}
ADVICE = re.compile(r"買進|賣出|做多|做空|加碼|減碼|建議買|建議賣")   # label/note 不得出現買賣建議字樣 (caveat 可引用 #73 等研究結論)


def _num_or_none(v) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def validate(doc: dict) -> list[str]:
    """回傳問題清單 (空 = 通過)。只檢查 App 會讀的結構與「不是買賣建議」的硬規則，不重算任何統計。"""
    pr: list[str] = []
    if not isinstance(doc, dict):
        return ["not a dict"]
    for k in TOP_REQUIRED:
        if k not in doc:
            pr.append(f"missing top key {k}")
    if pr:
        return pr
    if doc["schema"] != SCHEMA:
        pr.append(f"schema {doc['schema']!r} != {SCHEMA!r}")
    layers = doc["layers"]
    if not isinstance(layers, dict) or not layers:
        pr.append("layers empty")
        layers = {}
    for lid, L in layers.items():
        for k in LAYER_REQUIRED:
            if k not in (L or {}):
                pr.append(f"layer {lid} missing {k}")
        if (L or {}).get("kind") not in KINDS:
            pr.append(f"layer {lid} kind {(L or {}).get('kind')!r}")
    orders = [L.get("order") for L in layers.values() if isinstance(L, dict)]
    if len(set(orders)) != len(orders):
        pr.append("layer orders not unique")
    nodes = doc["nodes"]
    ids = [n.get("id") for n in nodes if isinstance(n, dict)]
    if len(set(ids)) != len(ids) or not ids:
        pr.append("node ids empty or not unique")
    for n in nodes:
        for k in NODE_REQUIRED:
            if k not in n:
                pr.append(f"node {n.get('id')} missing {k}")
    nid = set(ids)
    eids: set[str] = set()
    grade_cnt = {"A": 0, "B": 0, "C": 0}
    n_prod = 0
    for e in doc["edges"]:
        eid = e.get("id")
        for k in EDGE_REQUIRED:
            if k not in e:
                pr.append(f"edge {eid} missing {k}")
        if eid in eids:
            pr.append(f"edge id {eid} duplicated")
        eids.add(eid)
        if e.get("layer") not in layers:
            pr.append(f"edge {eid} layer {e.get('layer')!r} unknown")
        if e.get("kind") not in KINDS:
            pr.append(f"edge {eid} kind {e.get('kind')!r}")
        if e.get("source") not in nid or e.get("target") not in nid:
            pr.append(f"edge {eid} source/target not in nodes")
        if e.get("grade") not in GRADES:
            pr.append(f"edge {eid} grade {e.get('grade')!r} (F 不畫邊，應列 rejected)")
        else:
            grade_cnt[e["grade"]] += 1
        if e.get("tradable") is not False:
            pr.append(f"edge {eid} tradable must be false")
        if not _num_or_none(e.get("ic")) or not _num_or_none(e.get("rho")):
            pr.append(f"edge {eid} ic/rho not numeric")
        if e.get("ic") is None and e.get("rho") is None and e.get("kind") != "calibration":
            pr.append(f"edge {eid} has neither ic nor rho")
        wf = e.get("wf_hit")
        if wf is not None and not (isinstance(wf, list) and len(wf) == 3 and all(_num_or_none(v) for v in wf)):
            pr.append(f"edge {eid} wf_hit must be [hi, lo, base] or null")
        if not str(e.get("label") or "").strip():
            pr.append(f"edge {eid} empty label")
        for k in ("label", "note"):
            if ADVICE.search(str(e.get(k) or "")):
                pr.append(f"edge {eid} {k} contains advice wording")
        if e.get("production") not in (True, False):
            pr.append(f"edge {eid} production must be bool")
        n_prod += int(bool(e.get("production")))
    for r in doc["rejected"]:
        if not str(r.get("relation") or "").strip() or not str(r.get("why") or "").strip():
            pr.append(f"rejected entry missing relation/why: {r}")
    sm = doc["summary"] or {}
    if sm.get("n_edges") != len(doc["edges"]):
        pr.append(f"summary.n_edges {sm.get('n_edges')} != {len(doc['edges'])}")
    bg = sm.get("by_grade") or {}
    for g, c in grade_cnt.items():
        if bg.get(g, 0) != c:
            pr.append(f"summary.by_grade[{g}] {bg.get(g)} != {c}")
    nvp = sm.get("new_vs_production") or {}
    if nvp and nvp.get("production") != n_prod:
        pr.append(f"summary.new_vs_production.production {nvp.get('production')} != {n_prod}")
    if sm.get("model_changes_adopted") not in (0, None):
        pr.append("summary.model_changes_adopted should be 0 (沒有邊進模型)")
    return pr


def load(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def install(src: Path | str, dst: Path | str = DEST, asof: str | None = None) -> dict:
    """驗證 → 複製 → 蓋 asof / installed_at / source_file；有問題就 raise ValueError (不寫檔)。"""
    doc = load(src)
    pr = validate(doc)
    if pr:
        raise ValueError("relations_map invalid:\n  " + "\n  ".join(pr))
    doc["asof"] = asof or dt.date.today().isoformat()
    doc["installed_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc["source_file"] = Path(src).name
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="relations_map_app.json (省略 = 已安裝的 data/models/relations_map.json)")
    ap.add_argument("--check", action="store_true", help="只驗證")
    ap.add_argument("--dst", default=str(DEST))
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()
    src = Path(a.src) if a.src else Path(a.dst)
    if a.check or not a.src:
        pr = validate(load(src))
        print(f"{src}: {'OK' if not pr else str(len(pr)) + ' problem(s)'}")
        for x in pr:
            print("  -", x)
        return 1 if pr else 0
    doc = install(src, a.dst, a.asof)
    sm = doc["summary"]
    print(f"installed → {a.dst}: {sm['n_edges']} edges, grades {sm['by_grade']}, nodes {len(doc['nodes'])}, rejected {len(doc['rejected'])}, asof {doc['asof']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
