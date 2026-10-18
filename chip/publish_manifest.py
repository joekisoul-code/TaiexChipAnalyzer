"""發布檔內容雜湊清單 (status.json 的 files，2026-10-18)。

GitHub Pages force_orphan 整站重佈 → 所有檔 ETag 每次都變，App 的 304 失效；fast 模式帶回的 watchlist / treasure_live / margin
內容其實沒變。status.json 列出每個 data/*.json / *.jsonl 的「內容雜湊」，App 比對雜湊沒變就不重新下載。

契約 (前後端一致)：
- .json：sha1(canonical)[:12]，canonical = json.dumps(去掉易變鍵後的物件, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
- 易變鍵 (每次發布都會變、不代表內容變)：頂層 VOLATILE_TOP；meta.VOLATILE_META
- .jsonl 或無法解析的 .json：sha1(位元組)[:12]
- 鍵：.json 用不含副檔名的檔名 (= App cloudUrl(name) 的 name，例 "margin")；.jsonl 用完整檔名 (例 "pit_ledger.jsonl")
- status.json 自己不列入
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FILES_V = 1
VOLATILE_TOP = ("generated", "updated", "ts", "asof_run")
VOLATILE_META = ("generated", "carried", "carried_parts", "mode", "publish_log")
EXCLUDE = ("status.json",)


def strip_volatile(obj):
    if not isinstance(obj, dict):
        return obj
    out = {k: v for k, v in obj.items() if k not in VOLATILE_TOP}
    if isinstance(out.get("meta"), dict):
        out["meta"] = {k: v for k, v in out["meta"].items() if k not in VOLATILE_META}
    return out


def canonical(obj) -> str:
    return json.dumps(strip_volatile(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_obj(obj) -> str:
    return hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()[:12]


def hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]


def hash_file(path: Path | str) -> str:
    p = Path(path)
    b = p.read_bytes()
    if p.suffix == ".json":
        try:
            return hash_obj(json.loads(b.decode("utf-8-sig")))
        except Exception:  # noqa: BLE001  (損壞 / 非 JSON → 位元組雜湊，內容變了一樣會變)
            pass
    return hash_bytes(b)


def file_key(p: Path) -> str:
    return p.stem if p.suffix == ".json" else p.name


def manifest(data_dir: Path | str) -> dict[str, str]:
    """{鍵: hash12}，依檔名排序 (鍵見模組說明)；必須在所有檔寫完之後、寫 status.json 之前呼叫。"""
    d = Path(data_dir)
    if not d.is_dir():
        return {}
    out = {}
    for p in sorted(list(d.glob("*.json")) + list(d.glob("*.jsonl")), key=lambda x: x.name):
        if p.name in EXCLUDE or not p.is_file():
            continue
        try:
            out[file_key(p)] = hash_file(p)
        except OSError:
            continue
    return out
