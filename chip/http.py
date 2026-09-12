"""HTTP 工具：重試、UA、磁碟快取 (JSON / 文字)。"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

log = logging.getLogger(__name__)

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": config.USER_AGENT,
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        })
        retry = Retry(total=4, backoff_factor=2.0, status_forcelist=(428, 429, 500, 502, 503, 504))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        _session = s
    return _session


def _cache_path(key: str):
    return config.CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def is_cached(key: str, ttl: int) -> bool:
    p = _cache_path(key)
    if not p.exists():
        return False
    try:
        return time.time() - json.loads(p.read_text(encoding="utf-8"))["ts"] < ttl
    except Exception:  # noqa: BLE001
        return False


def cached(key: str, ttl: int, loader, *, allow_stale: bool = True) -> Any:
    """以 key 快取 loader() 的結果 (需可 JSON 序列化)。

    loader 失敗時若有舊快取則回傳舊資料 (allow_stale)。
    """
    p = _cache_path(key)
    now = time.time()
    if p.exists():
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
            if now - blob["ts"] < ttl:
                return blob["data"]
        except Exception:  # noqa: BLE001
            blob = None
    else:
        blob = None
    try:
        data = loader()
    except Exception as e:  # noqa: BLE001
        if blob is not None and allow_stale:
            log.warning("%s 讀取失敗，使用舊快取: %s", key, e)
            return blob["data"]
        raise
    try:
        p.write_text(json.dumps({"ts": now, "key": key, "data": data}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.debug("cache write failed: %s", e)
    return data


def get_json(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: int = 30, verify: bool = True) -> Any:
    r = session().get(url, params=params, headers=headers, timeout=timeout, verify=verify)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig", errors="replace")
    return json.loads(text)


def get_text(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: int = 30, verify: bool = True) -> str:
    r = session().get(url, params=params, headers=headers, timeout=timeout, verify=verify)
    r.raise_for_status()
    return r.content.decode("utf-8-sig", errors="replace")


def num(s: Any) -> float | None:
    """'1,234.5' / '-' / '' / None -> float | None"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().replace(",", "").replace("%", "")
    if t in ("", "-", "--", "N/A", "無"):
        return None
    try:
        return float(t)
    except ValueError:
        return None
