from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Trace:
    trace_id: str

    @staticmethod
    def new() -> Trace:
        return Trace(trace_id=str(uuid.uuid4()))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Common structured fields
        for key in ("trace_id", "event", "doc_id", "path", "provider", "model", "attempt"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def setup_logging(*, level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
