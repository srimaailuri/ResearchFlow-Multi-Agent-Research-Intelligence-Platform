"""Global spacing between Gemini chat calls (helps free-tier RPM limits)."""

from __future__ import annotations

import threading
import time

from researchflow.config.settings import get_settings

_lock = threading.Lock()
_last_chat_call: float = 0.0


def wait_before_chat_call() -> None:
    settings = get_settings()
    interval = float(settings.llm_min_interval_seconds or 0.0)
    if interval <= 0:
        return
    global _last_chat_call
    with _lock:
        now = time.monotonic()
        delay = interval - (now - _last_chat_call)
        if delay > 0:
            time.sleep(delay)
        _last_chat_call = time.monotonic()
