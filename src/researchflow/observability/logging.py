from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from researchflow.observability.context import get_request_id, get_run_id

_CONFIGURED = False


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        record.run_id = get_run_id() or "-"
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "run_id": getattr(record, "run_id", "-"),
        }
        log_fields = getattr(record, "log_fields", None)
        if isinstance(log_fields, dict):
            payload.update(log_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(*, level: str = "INFO", json_logs: bool = True) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(ContextFilter())
    if json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [req=%(request_id)s run=%(run_id)s] "
                "%(name)s: %(message)s"
            )
        )
    root.addHandler(handler)
    _CONFIGURED = True


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"log_fields": {"event": event, **fields}})
