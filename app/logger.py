"""Operational events deliberately exclude questions, evidence and model output."""

import json
import logging
import time

logger = logging.getLogger("rag")


def log_event(event: str, **fields):
    allowed = {"request_id", "status", "duration_ms", "code", "chunks", "stage"}
    safe = {key: value for key, value in fields.items() if key in allowed}
    logger.info(json.dumps({"timestamp": time.time(), "event": event, **safe}))
