import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_audit_log(event: dict[str, Any]) -> None:
    log_path = Path(AUDIT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
