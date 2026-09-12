"""警示推播：console + data/alerts.log + (選用) Webhook / Telegram。

環境變數：
  CHIP_WEBHOOK_URL      Discord / Slack incoming webhook (同時送 content 與 text 欄位)
  CHIP_TELEGRAM_TOKEN   Telegram bot token
  CHIP_TELEGRAM_CHAT    Telegram chat id
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

from . import config
from .http import session

log = logging.getLogger(__name__)
LOG_PATH = config.DATA_DIR / "alerts.log"


def notify(text: str, *, title: str = "台股即時警示") -> None:
    line = f"{dt.datetime.now(config.TZ).strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print("🔔", line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001
        log.debug("alert log write failed: %s", e)
    url = os.getenv("CHIP_WEBHOOK_URL")
    if url:
        try:
            session().post(url, json={"content": f"**{title}** {text}", "text": f"*{title}* {text}"}, timeout=10)
        except Exception as e:  # noqa: BLE001
            log.warning("webhook failed: %s", e)
    token, chat = os.getenv("CHIP_TELEGRAM_TOKEN"), os.getenv("CHIP_TELEGRAM_CHAT")
    if token and chat:
        try:
            session().post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": f"{title}\n{text}"}, timeout=10)
        except Exception as e:  # noqa: BLE001
            log.warning("telegram failed: %s", e)


def read_log(date: str | None = None, limit: int = 50) -> list[str]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    if date:
        lines = [x for x in lines if x.startswith(date)]
    return lines[-limit:]
