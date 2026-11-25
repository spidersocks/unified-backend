"""In-memory Clozzz live-run state (one run per routine id)."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from copy import deepcopy
from typing import Any

from . import store

_LOCK = threading.RLock()
# routineId(lower) -> LiveRun dict
_RUNS: dict[str, dict[str, Any]] = {}

DEFAULT_COUNTDOWN_SEC = 180
DEFAULT_STREAM_ENDED_SEC = 45

COMMANDS = frozenset(
    {
        "start",
        "pause",
        "resume",
        "skip",
        "back",
        "restart",
        "start_countdown",
        "skip_countdown",
        "skip_stream_ended",
        "hide_chrome",
        "show_chrome",
    }
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def expected_live_token() -> str | None:
    """
    Token required on mutating live routes.
    Prefer CLOZZZ_LIVE_TOKEN; else SHA-256 of gate material (same as overlay cookie).
    None = auth disabled (local/dev).
    """
    explicit = (os.environ.get("CLOZZZ_LIVE_TOKEN") or "").strip()
    if explicit:
        return explicit
    password = (os.environ.get("CLOZZZ_OVERLAY_PASSWORD") or "").strip()
    if not password:
        return None
    material = f"clozzz-overlay-gate:v1:{password}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _segments_snapshot(routine: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in routine.get("segments") or []:
        out.append(
            {
                "id": str(seg.get("id") or ""),
                "label": str(seg.get("label") or ""),
                "durationSec": int(seg.get("durationSec") or 0),
            }
        )
    return out


def _routine_flags(routine: dict[str, Any]) -> dict[str, Any]:
    try:
        intro_sec = int(routine.get("introCountdownSec") or DEFAULT_COUNTDOWN_SEC)
    except (TypeError, ValueError):
        intro_sec = DEFAULT_COUNTDOWN_SEC
    intro_sec = max(5, min(60 * 60, intro_sec))
    try:
        ended_sec = int(routine.get("streamEndedSec") or DEFAULT_STREAM_ENDED_SEC)
    except (TypeError, ValueError):
        ended_sec = DEFAULT_STREAM_ENDED_SEC
    ended_sec = max(5, min(60 * 60, ended_sec))
    return {
        "introCountdownEnabled": bool(routine.get("introCountdownEnabled", False)),
        "introCountdownSec": intro_sec,
        "streamEndedEnabled": bool(routine.get("streamEndedEnabled", False)),
        "streamEndedSec": ended_sec,
        "streamEndedMessage": str(routine.get("streamEndedMessage") or "").strip(),
    }


def _new_idle(routine: dict[str, Any]) -> dict[str, Any]:
    segments = _segments_snapshot(routine)
    if not segments:
        raise ValueError("routine has no segments")
    first = segments[0]
    now = _now_ms()
    flags = _routine_flags(routine)
    return {
        "routineId": routine["id"],
        "title": routine.get("title") or routine.get("name") or routine["id"],
        "status": "idle",
        "segmentIndex": 0,
        "segmentDurationSec": first["durationSec"],
        "segmentStartedAtMs": None,
        "pausedRemainingMs": first["durationSec"] * 1000,
        "revision": 1,
        "updatedAt": now,
        "serverNowMs": now,
        "segments": segments,
        "currentLabel": first["label"],
        "segmentCount": len(segments),
        "chromeVisible": True,
        "countdownDurationSec": flags["introCountdownSec"],
        "countdownStartedAtMs": None,
        "countdownPausedRemainingMs": None,
        "introConsumed": False,
        "streamEndedStartedAtMs": None,
        "streamEndedDurationSec": flags["streamEndedSec"],
        **flags,
    }


def _public(run: dict[str, Any]) -> dict[str, Any]:
    """Copy with fresh serverNowMs for client clock skew handling."""
    out = deepcopy(run)
    out["serverNowMs"] = _now_ms()
    segs = out.get("segments") or []
    idx = int(out.get("segmentIndex") or 0)
    if 0 <= idx < len(segs):
        out["currentLabel"] = segs[idx].get("label") or ""
    else:
        out["currentLabel"] = ""
    out["segmentCount"] = len(segs)
    out.setdefault("chromeVisible", True)
    out.setdefault("countdownDurationSec", DEFAULT_COUNTDOWN_SEC)
    out.setdefault("introCountdownEnabled", False)
    out.setdefault("introCountdownSec", DEFAULT_COUNTDOWN_SEC)
    out.setdefault("introConsumed", False)
    out.setdefault("streamEndedEnabled", False)
    out.setdefault("streamEndedSec", DEFAULT_STREAM_ENDED_SEC)
    out.setdefault("streamEndedDurationSec", DEFAULT_STREAM_ENDED_SEC)
    out.setdefault("streamEndedMessage", "")
    return out


def _enter_countdown(run: dict[str, Any], dur: int, *, now: int | None = None) -> None:
    if now is None:
        now = _now_ms()
    run["status"] = "countdown"
    run["countdownDurationSec"] = dur
    run["countdownStartedAtMs"] = now
    run["countdownPausedRemainingMs"] = None
    run["segmentStartedAtMs"] = None
    run["chromeVisible"] = False
    run["streamEndedStartedAtMs"] = None


def _finish_countdown(run: dict[str, Any]) -> None:
    """Countdown ended or skipped → armed idle overlay (chrome on)."""
    segments: list[dict[str, Any]] = run["segments"]
    first = segments[0]
    run["status"] = "idle"
    run["segmentIndex"] = 0
    run["segmentDurationSec"] = first["durationSec"]
    run["segmentStartedAtMs"] = None
    run["pausedRemainingMs"] = first["durationSec"] * 1000
    run["countdownStartedAtMs"] = None
    run["countdownPausedRemainingMs"] = None
    run["introConsumed"] = True
    run["chromeVisible"] = True


def _enter_stream_ended(run: dict[str, Any], *, now: int | None = None) -> None:
    if now is None:
        now = _now_ms()
    dur = int(run.get("streamEndedSec") or DEFAULT_STREAM_ENDED_SEC)
    run["status"] = "stream_ended"
    run["streamEndedDurationSec"] = dur
    run["streamEndedStartedAtMs"] = now
    run["segmentStartedAtMs"] = None
    run["pausedRemainingMs"] = 0
    run["chromeVisible"] = False


def _finish_stream_ended(run: dict[str, Any]) -> None:
    run["status"] = "done"
    run["streamEndedStartedAtMs"] = None
    run["segmentStartedAtMs"] = None
    run["pausedRemainingMs"] = 0
    run["chromeVisible"] = True


def _mark_done(run: dict[str, Any], *, now: int | None = None) -> None:
    if run.get("streamEndedEnabled"):
        _enter_stream_ended(run, now=now)
    else:
        run["status"] = "done"
        run["segmentStartedAtMs"] = None
        run["pausedRemainingMs"] = 0
        run["streamEndedStartedAtMs"] = None


def _advance_countdown_if_needed(run: dict[str, Any]) -> None:
    if run["status"] != "countdown":
        return
    started = run.get("countdownStartedAtMs")
    if started is None:
        return
    now = _now_ms()
    duration_ms = int(run.get("countdownDurationSec") or DEFAULT_COUNTDOWN_SEC) * 1000
    if now >= int(started) + duration_ms:
        _finish_countdown(run)
        run["updatedAt"] = _now_ms()


def _advance_stream_ended_if_needed(run: dict[str, Any]) -> None:
    if run["status"] != "stream_ended":
        return
    started = run.get("streamEndedStartedAtMs")
    if started is None:
        return
    now = _now_ms()
    duration_ms = int(run.get("streamEndedDurationSec") or DEFAULT_STREAM_ENDED_SEC) * 1000
    if now >= int(started) + duration_ms:
        _finish_stream_ended(run)
        run["updatedAt"] = _now_ms()


def _advance_if_needed(run: dict[str, Any]) -> None:
    """Lazy server-side segment / countdown / stream-ended advance."""
    _advance_countdown_if_needed(run)
    _advance_stream_ended_if_needed(run)
    if run["status"] != "running":
        return
    segments: list[dict[str, Any]] = run["segments"]
    if not segments:
        _mark_done(run)
        return

    now = _now_ms()
    while run["status"] == "running":
        idx = int(run["segmentIndex"])
        if idx < 0 or idx >= len(segments):
            _mark_done(run, now=now)
            break
        started = run.get("segmentStartedAtMs")
        if started is None:
            break
        duration_ms = int(segments[idx]["durationSec"]) * 1000
        end_at = int(started) + duration_ms
        if now < end_at:
            run["segmentDurationSec"] = segments[idx]["durationSec"]
            break
        overflow = now - end_at
        if idx + 1 >= len(segments):
            run["segmentIndex"] = idx
            run["segmentDurationSec"] = segments[idx]["durationSec"]
            _mark_done(run, now=now)
            break
        idx += 1
        run["segmentIndex"] = idx
        run["segmentDurationSec"] = segments[idx]["durationSec"]
        next_dur = int(segments[idx]["durationSec"]) * 1000
        run["segmentStartedAtMs"] = now - min(overflow, next_dur - 1)
        if overflow >= next_dur:
            continue
        break
    run["updatedAt"] = _now_ms()


def _bump(run: dict[str, Any]) -> None:
    run["revision"] = int(run.get("revision") or 0) + 1
    run["updatedAt"] = _now_ms()


def get_or_arm(key: str) -> dict[str, Any]:
    catalog = store.get_catalog()
    routine = store.find_routine(catalog, key)
    if not routine:
        raise KeyError(key)
    rid = routine["id"]
    with _LOCK:
        existing = _RUNS.get(rid.lower())
        if existing is None:
            run = _new_idle(routine)
            _RUNS[rid.lower()] = run
            return _public(run)
        _advance_if_needed(existing)
        return _public(existing)


def get_run(key: str) -> dict[str, Any] | None:
    catalog = store.get_catalog()
    routine = store.find_routine(catalog, key)
    if not routine:
        raise KeyError(key)
    with _LOCK:
        existing = _RUNS.get(routine["id"].lower())
        if existing is None:
            return None
        _advance_if_needed(existing)
        return _public(existing)


def apply_command(
    key: str,
    command: str,
    duration_sec: int | None = None,
) -> dict[str, Any]:
    command = (command or "").strip().lower()
    if command not in COMMANDS:
        raise ValueError(f"Unknown command “{command}”")

    catalog = store.get_catalog()
    routine = store.find_routine(catalog, key)
    if not routine:
        raise KeyError(key)

    rid = routine["id"]
    with _LOCK:
        run = _RUNS.get(rid.lower())
        if run is None:
            run = _new_idle(routine)
            _RUNS[rid.lower()] = run
        else:
            _advance_if_needed(run)

        segments: list[dict[str, Any]] = run["segments"]
        if not segments:
            raise ValueError("routine has no segments")

        now = _now_ms()
        status = run["status"]

        if command == "hide_chrome":
            run["chromeVisible"] = False
            _bump(run)
            return _public(run)

        if command == "show_chrome":
            run["chromeVisible"] = True
            _bump(run)
            return _public(run)

        if command == "start_countdown":
            flags = _routine_flags(routine)
            default_dur = flags["introCountdownSec"]
            dur = int(duration_sec) if duration_sec is not None else default_dur
            if dur < 5:
                dur = 5
            if dur > 60 * 60:
                dur = 60 * 60
            _enter_countdown(run, dur, now=now)
            _bump(run)
            return _public(run)

        if command == "skip_countdown":
            if status != "countdown":
                return _public(run)
            _finish_countdown(run)
            _bump(run)
            return _public(run)

        if command == "skip_stream_ended":
            if status != "stream_ended":
                return _public(run)
            _finish_stream_ended(run)
            _bump(run)
            return _public(run)

        if command == "restart":
            run = _new_idle(routine)
            _RUNS[rid.lower()] = run
            return _public(run)

        # Routine controls ignore countdown — ask to skip first via skip_countdown
        if status == "countdown" and command in {
            "start",
            "pause",
            "resume",
            "skip",
            "back",
        }:
            if command == "skip":
                _finish_countdown(run)
                _bump(run)
                return _public(run)
            return _public(run)

        if status == "stream_ended" and command in {
            "start",
            "pause",
            "resume",
            "skip",
            "back",
        }:
            if command in ("skip", "start"):
                _finish_stream_ended(run)
                if command == "start":
                    # Fall through to start from done → restart path below
                    status = "done"
                    run["status"] = "done"
                else:
                    _bump(run)
                    return _public(run)
            else:
                return _public(run)

        if command == "start":
            if status == "done":
                run = _new_idle(routine)
                _RUNS[rid.lower()] = run
                status = "idle"
            if status == "paused":
                remaining = int(run.get("pausedRemainingMs") or 0)
                run["status"] = "running"
                run["segmentStartedAtMs"] = now - (
                    int(run["segmentDurationSec"]) * 1000 - remaining
                )
                run["pausedRemainingMs"] = None
                run["chromeVisible"] = True
                _bump(run)
                return _public(run)
            if status == "idle":
                # Routine-level intro: first Start runs pre-stream countdown.
                if run.get("introCountdownEnabled") and not run.get("introConsumed"):
                    dur = int(run.get("introCountdownSec") or DEFAULT_COUNTDOWN_SEC)
                    _enter_countdown(run, dur, now=now)
                    _bump(run)
                    return _public(run)
                idx = int(run["segmentIndex"])
                if idx < 0 or idx >= len(segments):
                    idx = 0
                    run["segmentIndex"] = idx
                run["segmentDurationSec"] = segments[idx]["durationSec"]
                run["status"] = "running"
                run["segmentStartedAtMs"] = now
                run["pausedRemainingMs"] = None
                run["chromeVisible"] = True
                _bump(run)
                return _public(run)
            return _public(run)

        if command == "resume":
            if status != "paused":
                return _public(run)
            remaining = int(run.get("pausedRemainingMs") or 0)
            dur = int(run["segmentDurationSec"]) * 1000
            run["status"] = "running"
            run["segmentStartedAtMs"] = now - (dur - remaining)
            run["pausedRemainingMs"] = None
            _bump(run)
            return _public(run)

        if command == "pause":
            if status != "running":
                return _public(run)
            started = int(run.get("segmentStartedAtMs") or now)
            dur = int(run["segmentDurationSec"]) * 1000
            remaining = max(0, dur - (now - started))
            run["status"] = "paused"
            run["pausedRemainingMs"] = remaining
            run["segmentStartedAtMs"] = None
            _bump(run)
            return _public(run)

        if command == "skip":
            if status == "done":
                return _public(run)
            idx = int(run["segmentIndex"])
            if idx + 1 >= len(segments):
                run["segmentIndex"] = idx
                _mark_done(run, now=now)
                _bump(run)
                return _public(run)
            idx += 1
            run["segmentIndex"] = idx
            run["segmentDurationSec"] = segments[idx]["durationSec"]
            if status == "running":
                run["segmentStartedAtMs"] = now
                run["pausedRemainingMs"] = None
            elif status in ("idle", "paused"):
                run["status"] = "paused" if status == "paused" else "idle"
                run["segmentStartedAtMs"] = None
                run["pausedRemainingMs"] = segments[idx]["durationSec"] * 1000
                if status == "idle":
                    run["status"] = "idle"
            _bump(run)
            return _public(run)

        if command == "back":
            if status == "done":
                idx = len(segments) - 1
                run["segmentIndex"] = idx
                run["segmentDurationSec"] = segments[idx]["durationSec"]
                run["status"] = "paused"
                run["segmentStartedAtMs"] = None
                run["pausedRemainingMs"] = segments[idx]["durationSec"] * 1000
                _bump(run)
                return _public(run)
            idx = int(run["segmentIndex"])
            if idx <= 0:
                run["segmentIndex"] = 0
                run["segmentDurationSec"] = segments[0]["durationSec"]
                if status == "running":
                    run["segmentStartedAtMs"] = now
                    run["pausedRemainingMs"] = None
                else:
                    run["segmentStartedAtMs"] = None
                    run["pausedRemainingMs"] = segments[0]["durationSec"] * 1000
                _bump(run)
                return _public(run)
            idx -= 1
            run["segmentIndex"] = idx
            run["segmentDurationSec"] = segments[idx]["durationSec"]
            if status == "running":
                run["segmentStartedAtMs"] = now
                run["pausedRemainingMs"] = None
            else:
                run["segmentStartedAtMs"] = None
                run["pausedRemainingMs"] = segments[idx]["durationSec"] * 1000
            _bump(run)
            return _public(run)

        return _public(run)
