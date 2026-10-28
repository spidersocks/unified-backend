"""Visitor counter for seanfontaine.dev /home (Sean's Plaza)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/plaza", tags=["Plaza"])

START_COUNT = int(os.environ.get("PLAZA_VISITOR_START", "900"))
COUNTER_PATH = Path(
    os.environ.get(
        "PLAZA_COUNTER_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "plaza_visitors.json"),
    )
)

_lock = threading.Lock()
_count: int | None = None


def _read_file() -> int | None:
    try:
        if not COUNTER_PATH.is_file():
            return None
        data = json.loads(COUNTER_PATH.read_text(encoding="utf-8"))
        value = int(data.get("count"))
        return value if value >= 0 else None
    except Exception as e:
        print(f"[PLAZA] Failed to read counter file: {e}", flush=True)
        return None


def _write_file(value: int) -> None:
    try:
        COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = COUNTER_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"count": value}), encoding="utf-8")
        tmp.replace(COUNTER_PATH)
    except Exception as e:
        print(f"[PLAZA] Failed to write counter file: {e}", flush=True)


def _ensure_loaded() -> int:
    global _count
    if _count is not None:
        return _count
    stored = _read_file()
    _count = START_COUNT if stored is None else stored
    if stored is None:
        _write_file(_count)
    return _count


@router.get("/visitors")
def get_visitors():
    with _lock:
        return {"count": _ensure_loaded()}


@router.post("/visit")
def record_visit():
    """Increment the counter and return the new total."""
    global _count
    with _lock:
        current = _ensure_loaded()
        _count = current + 1
        _write_file(_count)
        return {"count": _count}
