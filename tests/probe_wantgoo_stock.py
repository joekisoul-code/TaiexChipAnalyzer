"""抓一檔個股的玩股網籌碼 JSON，印出各端點欄位。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from chip.sources import wantgoo  # noqa: E402

sid = sys.argv[1] if len(sys.argv) > 1 else "2330"
d = wantgoo.fetch_stocks([sid]).get(sid, {})
for k, v in d.items():
    if isinstance(v, list):
        print(f"== {k}: list[{len(v)}]  first={json.dumps(v[0], ensure_ascii=False)[:400] if v else None}")
    elif isinstance(v, dict):
        print(f"== {k}: dict keys={list(v.keys())[:8]}  sample={json.dumps(v, ensure_ascii=False)[:400]}")
    else:
        print("==", k, v)
