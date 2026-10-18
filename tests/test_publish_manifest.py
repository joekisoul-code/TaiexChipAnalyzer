"""status.json files manifest (chip/publish_manifest.py) + margin.json / margin_hist.json 拆檔 (margin_dyn.split / merge_hist / published / carry_hist)
+ export_static 掛鉤 (2026-10-18)。不連網 (假 session 從暫存目錄回檔)；SQLite 用暫存 CHIP_DATA_DIR；不跑 export 全流程。

    python tests/test_publish_manifest.py
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="publish_manifest_test_")
os.environ["CHIP_DATA_DIR"] = _TMP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from chip import publish_manifest as P  # noqa: E402
from chip.analysis import margin_dyn as M  # noqa: E402
from chip.serialize import clean  # noqa: E402

M.THROTTLE = 0
EXPORT = (ROOT / "tools" / "export_static.py").read_text(encoding="utf-8")


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def dump(d: Path, name: str, obj) -> None:
    """與 tools/export_static.dump 相同的序列化。"""
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(clean(obj), ensure_ascii=False, default=str), encoding="utf-8")


class _Resp:
    def __init__(self, b: bytes | None):
        self.ok = b is not None
        self.content = b or b""
        self.text = self.content.decode("utf-8") if b else ""


class _FakeSession:
    """把 Pages / raw URL 的 /data/<file> 對到 site 目錄；fail 內的檔名回 404。"""
    def __init__(self, site: Path, fail=()):
        self.site, self.fail, self.calls = site, set(fail), []

    def get(self, url, timeout=None):
        name = url.rsplit("/", 1)[-1].split("?")[0]
        self.calls.append(name)
        p = self.site / name
        return _Resp(p.read_bytes() if p.exists() and name not in self.fail else None)


def _full_out(n_hist=400, gen="2026-10-17 15:40:00") -> dict:
    rows = [[f"{2001 + i // 250}-{1 + (i // 21) % 12:02d}-{1 + i % 21:02d}", 1000.0 + i, 9000000 + i, 200000, None, None] for i in range(n_hist)]
    frows = [[r[0]] + [1.0] * (len(M.FLOW_COLS) - 1) for r in rows[-150:]]
    drows = [[r[0]] + [0.5] * (len(M.DERIVED_COLS) - 2) + ["中性"] for r in rows[-400:]]
    return {"meta": {"generated": gen, "latest_trading": "2026-10-17", "mode": "full", "version": M.VERSION, "carried": False, "carried_parts": [],
                     "sources": {"twse_all": {"date": "2026-10-17", "ok": True}}, "coverage": {"history_days": n_hist}, "labels": M.LABELS, "validated": [],
                     "tests_total": M.TESTS_TOTAL, "publish_log": [{"date": "2026-10-17", "first_seen": gen}], "size_guard": {"steps": [], "bytes": 1234}},
            "market": {"date": "2026-10-17", "amt": 3456.7}, "history": {"cols": list(M.HISTORY_COLS), "rows": rows},
            "flows": {"cols": list(M.FLOW_COLS), "rows": frows}, "derived": {"cols": list(M.DERIVED_COLS), "rows": drows},
            "ranking": {"date": "2026-10-17", "n_days": 60, "inc5": [{"code": "2330"}]}, "stocks": {"2330": {"name": "台積電", "daily": [[1] * 9]}},
            "watch_ids": ["2330"], "pool_ids": ["3443"], "alerts": ["a"]}


# ------------------------------------------------------------------ 契約 1：雜湊
def test_hash_ignores_volatile_keys():
    base = {"x": [1, 2, {"b": 1, "a": "中"}], "meta": {"k": 1, "generated": "g1", "carried": False, "carried_parts": [], "mode": "full", "publish_log": [1]},
            "generated": "2026-10-18 10:00:00", "updated": "u1", "ts": "t1", "asof_run": "r1"}
    vol = {"x": [1, 2, {"a": "中", "b": 1}], "meta": {"generated": "g2", "carried": True, "carried_parts": ["market"], "mode": "fast", "publish_log": [1, 2], "k": 1},
           "generated": "2026-10-18 10:10:00", "updated": "u2", "ts": "t2", "asof_run": "r2"}
    assert P.hash_obj(base) == P.hash_obj(vol), "易變鍵 / 鍵順序不影響雜湊"
    assert P.hash_obj(base) == P.hash_obj({"x": [1, 2, {"a": "中", "b": 1}], "meta": {"k": 1}}), "易變鍵整個不存在也一樣"
    h = P.hash_obj(base)
    assert re.fullmatch(r"[0-9a-f]{12}", h)
    # 契約公式本身
    exp = hashlib.sha1(json.dumps({"x": [1, 2, {"a": "中", "b": 1}], "meta": {"k": 1}}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    assert h == exp
    assert P.VOLATILE_TOP == ("generated", "updated", "ts", "asof_run") and P.VOLATILE_META == ("generated", "carried", "carried_parts", "mode", "publish_log")


def test_hash_sensitive_to_content():
    a = {"x": 1, "meta": {"k": 1}, "rows": [[1, 2.5]]}
    assert P.hash_obj(a) != P.hash_obj({**a, "x": 2})
    assert P.hash_obj(a) != P.hash_obj({**a, "meta": {"k": 2}}), "meta 非易變鍵變了要變"
    assert P.hash_obj(a) != P.hash_obj({**a, "rows": [[1, 2.6]]})
    assert P.hash_obj(a) != P.hash_obj({**a, "latest_trading": "2026-10-18"}), "latest_trading 是內容 (App 判斷待公布)"
    assert P.hash_obj({"a": {"generated": 1}}) != P.hash_obj({"a": {"generated": 2}}), "只剝頂層與 meta 下的易變鍵"


def test_manifest_files_jsonl_and_status_excluded():
    d = Path(tempfile.mkdtemp(prefix="pm_", dir=_TMP))
    dump(d, "margin", {"meta": {"generated": "g"}, "v": 1})
    dump(d, "realtime", {"ts": "t", "v": 2})
    dump(d, "status", {"ready": True})
    (d / "pit_ledger.jsonl").write_bytes(b'{"a":1}\n{"a":2}\n')
    (d / "broken.json").write_bytes(b"{not json")
    (d / "note.txt").write_text("x")
    m = P.manifest(d)
    assert set(m) == {"margin", "realtime", "pit_ledger.jsonl", "broken"}, m
    assert m["pit_ledger.jsonl"] == hashlib.sha1(b'{"a":1}\n{"a":2}\n').hexdigest()[:12], ".jsonl = sha1(位元組)"
    assert m["broken"] == hashlib.sha1(b"{not json").hexdigest()[:12]
    assert m["margin"] == P.hash_obj({"meta": {}, "v": 1}) and m["realtime"] == P.hash_obj({"v": 2})
    assert list(m) == sorted(m, key=lambda k: k if k.endswith(".jsonl") else k + ".json")
    # 內容變 → 變；只有易變鍵變 / 重新序列化 (縮排不同) → 不變
    before = dict(m)
    (d / "margin.json").write_text(json.dumps({"v": 1, "meta": {"generated": "g2", "mode": "fast"}}, indent=2), encoding="utf-8")
    (d / "pit_ledger.jsonl").write_bytes(b'{"a":1}\n{"a":2}\n{"a":3}\n')
    m2 = P.manifest(d)
    assert m2["margin"] == before["margin"] and m2["pit_ledger.jsonl"] != before["pit_ledger.jsonl"]
    assert P.manifest(Path(_TMP) / "nope") == {}


# ------------------------------------------------------------------ 契約 2：拆檔
def test_split_core_and_hist():
    out = _full_out()
    core, hist = M.split(out)
    assert "history" not in core and "flows" not in core, "margin.json 不含 history / flows"
    assert core["meta"]["hist_file"] == "margin_hist.json" and core["meta"]["generated"] == out["meta"]["generated"]
    assert core["history_tail"]["cols"] == M.HISTORY_COLS and core["history_tail"]["rows"] == out["history"]["rows"][-M.HIST_TAIL:] and M.HIST_TAIL == 120
    assert len(core["derived"]["rows"]) == 250 == M.DERIVED_KEEP and core["derived"]["rows"] == out["derived"]["rows"][-250:]
    for k in ("market", "ranking", "stocks", "watch_ids", "pool_ids", "alerts"):
        assert core[k] == out[k], k
    assert hist["history"] == out["history"] and hist["flows"] == out["flows"]
    hm = hist["meta"]
    assert hm["generated"] == out["meta"]["generated"] and hm["history_from"] == out["history"]["rows"][0][0] and hm["history_to"] == out["history"]["rows"][-1][0]
    assert hm["history_days"] == 400 and hm["flows_days"] == 150 and hm["flows_from"] == out["flows"]["rows"][0][0] and hm["version"] == M.VERSION
    assert "history" in out and len(out["derived"]["rows"]) == 400, "split 不改輸入 (write_seed 仍拿完整 out)"
    # history 無列 → hist None (不寫，保留帶回檔)，核心仍有空的 history_tail
    c0, h0 = M.split({**_full_out(), "history": {"cols": M.HISTORY_COLS, "rows": []}})
    assert h0 is None and c0["history_tail"]["rows"] == [] and "history" not in c0
    # 往返：merge_hist(核心, hist) 還原 build 形狀
    back = M.merge_hist(core, hist)
    assert back["history"] == out["history"] and back["flows"] == out["flows"]
    old = _full_out()
    assert M.merge_hist(old, {"history": {"rows": [["x"]]}})["history"] == old["history"], "舊格式 (內嵌 history) 不被覆蓋"
    assert M.merge_hist(core, None) == core and M.merge_hist({}, hist) == {}


def test_published_merges_hist_and_old_format():
    site = Path(tempfile.mkdtemp(prefix="pages_", dir=_TMP)) / "data"
    core, hist = M.split(_full_out())
    dump(site, "margin", core)
    dump(site, "margin_hist", hist)
    fs = _FakeSession(site)
    with patched(M, "session", lambda: fs):
        j = M.published()
    assert fs.calls == ["margin.json", "margin_hist.json"] and j["history"]["rows"] == hist["history"]["rows"] and j["flows"] == hist["flows"]
    fs2 = _FakeSession(site, fail={"margin_hist.json"})
    with patched(M, "session", lambda: fs2):
        j2 = M.published()
    assert "history" not in j2 and j2["market"] == core["market"] and fs2.calls.count("margin_hist.json") == 2, "Pages 與 raw 都試過"
    dump(site, "margin", _full_out())      # 舊格式：內嵌 history → 不抓 margin_hist
    fs3 = _FakeSession(site)
    with patched(M, "session", lambda: fs3):
        j3 = M.published()
    assert fs3.calls == ["margin.json"] and len(j3["history"]["rows"]) == 400


def test_full_then_fast_hashes_unchanged():
    """full 發布 → 下一次 fast (帶回名單從 Pages 抓 → published() 併回 → build(fast) → split → dump)：margin / margin_hist / 帶回檔的 canonical 雜湊不變。"""
    pages = Path(tempfile.mkdtemp(prefix="full_", dir=_TMP)) / "data"
    out = _full_out()
    c, h = M.split(out)
    dump(pages, "margin", c)
    dump(pages, "margin_hist", h)
    dump(pages, "watchlist", {"updated": "2026-10-17 15:40:00", "stocks": {"2330": {"x": 1}}})
    m_full = P.manifest(pages)
    fast = Path(tempfile.mkdtemp(prefix="fast_", dir=_TMP)) / "data"
    fast.mkdir(parents=True)
    fs = _FakeSession(pages)
    with patched(M, "session", lambda: fs):
        for name in ("watchlist", "margin", "margin_hist"):       # export 的 fast 帶回名單 (位元組原樣)
            r = fs.get(f"https://x/data/{name}.json")
            (fast / f"{name}.json").write_text(r.text, encoding="utf-8")
        prev = M.published()
        out2 = M.build(None, None, None, prev=prev, fast=True)
    assert out2["meta"]["mode"] == "fast" and out2["meta"]["carried"] is True and out2["meta"]["generated"] != out["meta"]["generated"]
    assert out2["meta"]["size_guard"] == out["meta"]["size_guard"] and out2["meta"]["hist_file"] == "margin_hist.json", "fast 保留上次 meta 其餘鍵"
    c2, h2 = M.split(out2)
    dump(fast, "margin", c2)
    dump(fast, "margin_hist", h2)
    m_fast = P.manifest(fast)
    assert m_fast == m_full, (m_full, m_fast)
    assert json.loads((fast / "margin.json").read_text(encoding="utf-8"))["meta"]["generated"] != c["meta"]["generated"], "檔案位元組確實不同，只有易變鍵"
    # 再一次 fast (fast → fast) 仍不變
    with patched(M, "session", lambda: _FakeSession(fast)):
        c3, h3 = M.split(M.build(None, None, None, prev=M.published(), fast=True))
    assert P.hash_obj(json.loads(json.dumps(clean(c3), ensure_ascii=False))) == m_full["margin"] and P.hash_obj(json.loads(json.dumps(clean(h3), ensure_ascii=False))) == m_full["margin_hist"]
    # 舊格式 Pages (內嵌 history) → 第一次拆檔：margin_hist 內容 = 舊 history / flows
    old = Path(tempfile.mkdtemp(prefix="old_", dir=_TMP)) / "data"
    dump(old, "margin", out)
    with patched(M, "session", lambda: _FakeSession(old)):
        c4, h4 = M.split(M.build(None, None, None, prev=M.published(), fast=True))
    assert P.hash_obj(clean(h4)) == m_full["margin_hist"] and P.hash_obj(clean(c4)) == m_full["margin"]


def test_carry_hist_when_nothing_to_write():
    pages = Path(tempfile.mkdtemp(prefix="pg_", dir=_TMP)) / "data"
    _, h = M.split(_full_out())
    dump(pages, "margin_hist", h)
    dst = Path(tempfile.mkdtemp(prefix="dst_", dir=_TMP)) / "data" / "margin_hist.json"
    with patched(M, "session", lambda: _FakeSession(pages)):
        assert M.carry_hist(dst) is True and dst.read_bytes() == (pages / "margin_hist.json").read_bytes()
        assert M.carry_hist(dst) is False, "已存在不覆蓋"
    with patched(M, "session", lambda: _FakeSession(pages, fail={"margin_hist.json"})):
        assert M.carry_hist(dst.parent / "x" / "margin_hist.json") is False


# ------------------------------------------------------------------ export_static 掛鉤 (讀原始碼)
def test_export_static_hooks():
    m = re.search(r'for name in \(([^)]*)\):\s*\n\s*if not \(DATA / f"\{name\}\.json"\)\.exists\(\):', EXPORT)
    names = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]
    assert {"margin", "margin_hist"} <= set(names), "fast 帶回名單含 margin_hist"
    blk = EXPORT[EXPORT.index("from chip.analysis import margin_dyn"):EXPORT.index("pit_ledger as _pit")]
    i_split = blk.index("_mg_core, _mg_hist = margin_dyn.split(_mg)")
    assert i_split < blk.index('dump("margin", _mg_core)') < blk.index('dump("margin_hist", _mg_hist)')
    assert 'dump("margin", _mg)' not in blk and "margin_dyn.write_seed(_mg)" in blk, "種子仍用完整 out"
    assert 'margin_dyn.carry_hist(DATA / "margin_hist.json")' in blk
    assert 'margin_dyn.merge_hist(_mg_prev, json.loads((DATA / "margin_hist.json")' in blk and blk.index("merge_hist") < blk.index("margin_dyn.build(")
    # manifest 在最後一個 data 檔寫完之後、dump status 之前；status 帶 files / files_v
    i_man = EXPORT.index("_pm.manifest(DATA)")
    i_status = EXPORT.index('dump("status", _status)')
    last_other = max(mm.start() for mm in re.finditer(r'dump\("(?!status)', EXPORT))
    assert last_other < i_man < i_status
    tail = EXPORT[EXPORT.index("def main()"):]
    assert max(mm.start() for mm in re.finditer(r"\.write_text\(|shutil\.copy\(", tail)) < tail.index("_pm.manifest(DATA)"), "所有寫檔都在 manifest 之前"
    assert '_status["files"], _status["files_v"] = _files, _pm.FILES_V' in EXPORT and P.FILES_V == 1


if __name__ == "__main__":
    fails = 0
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            n += 1
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'} ({n} tests)")
    sys.exit(1 if fails else 0)
