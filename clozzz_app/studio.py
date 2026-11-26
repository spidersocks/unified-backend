"""Active OBS channel (which routine the open display follows).

Persisted to a small JSON file when possible. Fly machines without a volume
lose the file on redeploy — same caveat as the routines catalog.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import store

_LOCK = threading.RLock()

STUDIO_PATH = Path(
    os.environ.get(
        "CLOZZZ_STUDIO_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "clozzz_studio.json"),
    )
)

# In-memory mirror so reads stay cheap and survive brief write failures.
_STATE: dict[str, Any] = {"activeRoutineId": None, "updatedAt": None}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    rid = data.get("activeRoutineId")
    if rid is not None:
        rid = str(rid).strip() or None
    return {
        "activeRoutineId": rid,
        "updatedAt": str(data.get("updatedAt") or "") or None,
    }


def _read_disk() -> dict[str, Any]:
    if not STUDIO_PATH.is_file():
        return {"activeRoutineId": None, "updatedAt": None}
    try:
        raw = json.loads(STUDIO_PATH.read_text(encoding="utf-8"))
        return _normalize(raw)
    except Exception as e:
        print(f"[CLOZZZ] Failed to read studio state: {e}", flush=True)
        return {"activeRoutineId": None, "updatedAt": None}


def _write_disk(state: dict[str, Any]) -> None:
    try:
        STUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STUDIO_PATH.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(STUDIO_PATH)
    except Exception as e:
        # Survive ephemeral FS / missing volume — keep in-memory state.
        print(f"[CLOZZZ] Failed to persist studio state: {e}", flush=True)


def _ensure_loaded() -> None:
    if _STATE.get("updatedAt") is not None or _STATE.get("activeRoutineId") is not None:
        return
    disk = _read_disk()
    _STATE["activeRoutineId"] = disk.get("activeRoutineId")
    _STATE["updatedAt"] = disk.get("updatedAt")


def get_channel() -> dict[str, Any]:
    with _LOCK:
        _ensure_loaded()
        return {
            "activeRoutineId": _STATE.get("activeRoutineId"),
            "updatedAt": _STATE.get("updatedAt"),
        }


def set_channel(routine_key: str | None) -> dict[str, Any]:
    """
    Set the active OBS channel.
    Pass None / "" to clear (open display shows waiting idle).
    """
    with _LOCK:
        _ensure_loaded()
        if routine_key is None or str(routine_key).strip() == "":
            _STATE["activeRoutineId"] = None
            _STATE["updatedAt"] = _utc_now()
            out = {
                "activeRoutineId": None,
                "updatedAt": _STATE["updatedAt"],
            }
            _write_disk(out)
            return dict(out)

        catalog = store.get_catalog()
        routine = store.find_routine(catalog, str(routine_key).strip())
        if not routine:
            raise KeyError(routine_key)

        _STATE["activeRoutineId"] = routine["id"]
        _STATE["updatedAt"] = _utc_now()
        out = {
            "activeRoutineId": routine["id"],
            "updatedAt": _STATE["updatedAt"],
        }
        _write_disk(out)
        return dict(out)


def resolve_active_routine() -> dict[str, Any] | None:
    """Catalog routine for the current channel, or None if unset/missing."""
    channel = get_channel()
    rid = channel.get("activeRoutineId")
    if not rid:
        return None
    catalog = store.get_catalog()
    return store.find_routine(catalog, rid)
