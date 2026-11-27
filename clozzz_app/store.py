"""File-backed Clozzz routine catalog (plaza-style atomic JSON writes)."""

from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

SEED_PATH = Path(__file__).resolve().parent / "seed_routines.json"
STORE_PATH = Path(
    os.environ.get(
        "CLOZZZ_ROUTINES_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "clozzz_routines.json"),
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_seed() -> dict[str, Any]:
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    return _normalize_catalog(raw, stamp=True)


def _strip_cue(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_strip_cue(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_cue(v) for k, v in obj.items() if k != "cue"}
    return obj


SPACE_OPTIONS = ("Floor mat", "Bed", "Others")


def _normalize_space(raw: Any) -> list[str]:
    """Bounded multi-select; migrate legacy free-text strings."""
    allowed = set(SPACE_OPTIONS)
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            s = str(item or "").strip()
            if s in allowed and s not in out:
                out.append(s)
        return out
    text = str(raw or "").strip()
    if not text:
        return []
    if text in allowed:
        return [text]
    lower = text.lower()
    if "floor" in lower or "mat" in lower:
        return ["Floor mat"]
    if "bed" in lower:
        return ["Bed"]
    return ["Others"]


def _normalize_catalog(data: dict[str, Any], *, stamp: bool = False) -> dict[str, Any]:
    cleaned = _strip_cue(data)
    routines = cleaned.get("routines")
    if not isinstance(routines, list):
        raise ValueError("catalog.routines must be a list")

    ids: set[str] = set()
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for i, routine in enumerate(routines):
        if not isinstance(routine, dict):
            raise ValueError(f"routines[{i}] must be an object")
        rid = str(routine.get("id") or "").strip()
        name = str(routine.get("name") or "").strip()
        title = str(routine.get("title") or "").strip()
        if not rid or not name or not title:
            raise ValueError(f"routines[{i}] requires id, name, and title")
        key_id = rid.lower()
        key_name = name.lower()
        if key_id in ids:
            raise ValueError(f"duplicate routine id: {rid}")
        if key_name in names:
            raise ValueError(f"duplicate routine name: {name}")
        ids.add(key_id)
        names.add(key_name)

        segments_in = routine.get("segments")
        if not isinstance(segments_in, list) or not segments_in:
            raise ValueError(f"routine {rid} needs at least one segment")

        segments: list[dict[str, Any]] = []
        seg_ids: set[str] = set()
        for j, seg in enumerate(segments_in):
            if not isinstance(seg, dict):
                raise ValueError(f"routine {rid} segments[{j}] must be an object")
            sid = str(seg.get("id") or "").strip()
            label = str(seg.get("label") or "").strip()
            try:
                duration = int(seg.get("durationSec"))
            except (TypeError, ValueError):
                duration = 0
            if not sid or not label or duration <= 0:
                raise ValueError(
                    f"routine {rid} segments[{j}] needs id, label, durationSec > 0"
                )
            if sid.lower() in seg_ids:
                raise ValueError(f"routine {rid} duplicate segment id: {sid}")
            seg_ids.add(sid.lower())
            segments.append({"id": sid, "label": label, "durationSec": duration})

        existing_updated = routine.get("updatedAt")
        # Prefer `description`; accept legacy `intent` on read.
        description = str(
            routine.get("description") or routine.get("intent") or ""
        ).strip()

        # Existing routines keep stored flags; missing → off (new drafts set ON in the editor).
        intro_enabled = bool(routine.get("introCountdownEnabled", False))
        try:
            intro_sec = int(routine.get("introCountdownSec", 1800))
        except (TypeError, ValueError):
            intro_sec = 1800
        # Pre-stream countdown floor ≥1 minute
        intro_sec = max(60, min(60 * 60, intro_sec))
        intro_message = str(routine.get("introCountdownMessage") or "").strip()

        ended_enabled = bool(routine.get("streamEndedEnabled", False))
        # Duration is not required from the editor; soft-cap ~1h for live auto-finish.
        ended_sec = 3600
        ended_message = str(routine.get("streamEndedMessage") or "").strip()

        breaks_enabled = bool(routine.get("breaksEnabled", False))
        try:
            break_sec = int(routine.get("breakSec", 10))
        except (TypeError, ValueError):
            break_sec = 10
        break_sec = max(1, min(5 * 60, break_sec))

        space = _normalize_space(routine.get("space"))

        item = {
            "id": rid,
            "name": name,
            "title": title,
            "routineId": str(routine.get("routineId") or rid).strip() or rid,
            "space": space,
            "description": description,
            "introCountdownEnabled": intro_enabled,
            "introCountdownSec": intro_sec,
            "introCountdownMessage": intro_message,
            "streamEndedEnabled": ended_enabled,
            "streamEndedSec": ended_sec,
            "streamEndedMessage": ended_message,
            "breaksEnabled": breaks_enabled,
            "breakSec": break_sec,
            "segments": segments,
            "updatedAt": str(existing_updated) if existing_updated else _utc_now(),
        }
        normalized.append(item)

    out = {
        "schemaVersion": int(cleaned.get("schemaVersion") or 1),
        "updatedAt": cleaned.get("updatedAt") or _utc_now(),
        "routines": normalized,
    }
    if stamp:
        out["updatedAt"] = _utc_now()
    return out


def _write_catalog(catalog: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STORE_PATH)


def _read_or_seed() -> dict[str, Any]:
    if STORE_PATH.is_file():
        try:
            raw = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            return _normalize_catalog(raw)
        except Exception as e:
            print(f"[CLOZZZ] Failed to read store, re-seeding: {e}", flush=True)
    catalog = _load_seed()
    _write_catalog(catalog)
    return catalog


def get_catalog() -> dict[str, Any]:
    with _LOCK:
        return copy.deepcopy(_read_or_seed())


def save_catalog(data: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        catalog = _normalize_catalog(data, stamp=True)
        _write_catalog(catalog)
        return copy.deepcopy(catalog)


def reset_to_seed() -> dict[str, Any]:
    with _LOCK:
        catalog = _load_seed()
        _write_catalog(catalog)
        return copy.deepcopy(catalog)


def find_routine(catalog: dict[str, Any], key: str) -> dict[str, Any] | None:
    needle = key.strip().lower()
    if not needle:
        return None
    for routine in catalog["routines"]:
        if (
            routine["id"].lower() == needle
            or routine["name"].lower() == needle
            or routine["routineId"].lower() == needle
            or routine["title"].lower() == needle
        ):
            return routine
    return None


def upsert_routine(routine: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        catalog = _read_or_seed()
        incoming = _normalize_catalog({"schemaVersion": 1, "routines": [routine]})["routines"][0]
        incoming["updatedAt"] = _utc_now()
        routines = catalog["routines"]
        idx = next((i for i, r in enumerate(routines) if r["id"].lower() == incoming["id"].lower()), None)

        # Uniqueness vs other routines
        for i, existing in enumerate(routines):
            if idx is not None and i == idx:
                continue
            if existing["id"].lower() == incoming["id"].lower():
                raise ValueError(f"duplicate routine id: {incoming['id']}")
            if existing["name"].lower() == incoming["name"].lower():
                raise ValueError(f"duplicate routine name: {incoming['name']}")

        if idx is None:
            routines.append(incoming)
        else:
            routines[idx] = incoming

        catalog["routines"] = routines
        catalog["updatedAt"] = _utc_now()
        _write_catalog(catalog)
        return copy.deepcopy(catalog)


def delete_routine(key: str) -> dict[str, Any]:
    with _LOCK:
        catalog = _read_or_seed()
        before = len(catalog["routines"])
        catalog["routines"] = [
            r
            for r in catalog["routines"]
            if r["id"].lower() != key.lower() and r["name"].lower() != key.lower()
        ]
        if len(catalog["routines"]) == before:
            raise KeyError(key)
        catalog["updatedAt"] = _utc_now()
        _write_catalog(catalog)
        return copy.deepcopy(catalog)


def reorder_routines(ordered_ids: list[str]) -> dict[str, Any]:
    with _LOCK:
        catalog = _read_or_seed()
        by_id = {r["id"].lower(): r for r in catalog["routines"]}
        if len(ordered_ids) != len(by_id):
            raise ValueError("order must include every routine id exactly once")
        seen: set[str] = set()
        next_list: list[dict[str, Any]] = []
        for rid in ordered_ids:
            key = rid.strip().lower()
            if key in seen or key not in by_id:
                raise ValueError(f"invalid order id: {rid}")
            seen.add(key)
            next_list.append(by_id[key])
        catalog["routines"] = next_list
        catalog["updatedAt"] = _utc_now()
        _write_catalog(catalog)
        return copy.deepcopy(catalog)
