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

DEFAULT_COUNTDOWN_SEC = 1800  # 30 min — matches new-routine editor defaults
# Soft cap for stream-ended count-up (~1h); dismissible anytime via remote.
STREAM_ENDED_SOFT_CAP_SEC = 3600
DEFAULT_STREAM_ENDED_SEC = STREAM_ENDED_SOFT_CAP_SEC
DEFAULT_BREAK_SEC = 10  # customary rest between exercises

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
        "set_cue_audio",
    }
)

DEFAULT_CUE_VOLUME = 0.75


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
    # Pre-stream countdown floor ≥1 minute
    intro_sec = max(60, min(60 * 60, intro_sec))
    # Stream ended runs indefinitely from the editor's POV; soft-cap ~1h internally.
    ended_sec = STREAM_ENDED_SOFT_CAP_SEC
    try:
        break_sec = int(routine.get("breakSec") or DEFAULT_BREAK_SEC)
    except (TypeError, ValueError):
        break_sec = DEFAULT_BREAK_SEC
    break_sec = max(1, min(5 * 60, break_sec))
    return {
        "introCountdownEnabled": bool(routine.get("introCountdownEnabled", False)),
        "introCountdownSec": intro_sec,
        "introCountdownMessage": str(routine.get("introCountdownMessage") or "").strip(),
        "streamEndedEnabled": bool(routine.get("streamEndedEnabled", False)),
        "streamEndedSec": ended_sec,
        "streamEndedMessage": str(routine.get("streamEndedMessage") or "").strip(),
        "breaksEnabled": bool(routine.get("breaksEnabled", False)),
        "breakSec": break_sec,
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
        "breakDurationSec": flags["breakSec"],
        "breakStartedAtMs": None,
        "breakPausedRemainingMs": None,
        "cueVolume": DEFAULT_CUE_VOLUME,
        "cueMuted": False,
        **flags,
    }


def _replace_idle(routine: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, Any]:
    """Fresh idle run that keeps revision monotonic + cue audio prefs."""
    old_rev = int((prev or {}).get("revision") or 0)
    run = _new_idle(routine)
    run["revision"] = old_rev + 1
    if prev is not None:
        try:
            vol = float(prev.get("cueVolume", DEFAULT_CUE_VOLUME))
        except (TypeError, ValueError):
            vol = DEFAULT_CUE_VOLUME
        run["cueVolume"] = max(0.0, min(1.0, vol))
        run["cueMuted"] = bool(prev.get("cueMuted", False))
    return run


def _sync_flags_from_routine(run: dict[str, Any], routine: dict[str, Any]) -> None:
    """
    Keep intro/end flags aligned with the routine catalog.

    In-memory runs outlive editor toggles; without this, OBS never sees a
    newly enabled pre-stream countdown until Restart (or process restart).
    """
    flags = _routine_flags(routine)
    run["introCountdownEnabled"] = flags["introCountdownEnabled"]
    run["introCountdownSec"] = flags["introCountdownSec"]
    run["introCountdownMessage"] = flags["introCountdownMessage"]
    run["streamEndedEnabled"] = flags["streamEndedEnabled"]
    run["streamEndedSec"] = flags["streamEndedSec"]
    run["streamEndedMessage"] = flags["streamEndedMessage"]
    run["streamEndedDurationSec"] = flags["streamEndedSec"]
    run["breaksEnabled"] = flags["breaksEnabled"]
    run["breakSec"] = flags["breakSec"]
    # Held idle: keep the frozen clock duration matching the editor value.
    if (
        run.get("status") == "idle"
        and flags["introCountdownEnabled"]
        and not run.get("introConsumed")
        and run.get("countdownStartedAtMs") is None
    ):
        run["countdownDurationSec"] = flags["introCountdownSec"]
    # Not mid-break: keep catalog duration for the next break insert.
    if run.get("status") != "break":
        run["breakDurationSec"] = flags["breakSec"]


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
    out.setdefault("introCountdownMessage", "")
    out.setdefault("introConsumed", False)
    out.setdefault("streamEndedEnabled", False)
    out.setdefault("streamEndedSec", STREAM_ENDED_SOFT_CAP_SEC)
    out.setdefault("streamEndedDurationSec", STREAM_ENDED_SOFT_CAP_SEC)
    out.setdefault("streamEndedMessage", "")
    out.setdefault("breaksEnabled", False)
    out.setdefault("breakSec", DEFAULT_BREAK_SEC)
    out.setdefault("breakDurationSec", DEFAULT_BREAK_SEC)
    out.setdefault("breakStartedAtMs", None)
    out.setdefault("breakPausedRemainingMs", None)
    out.setdefault("cueVolume", DEFAULT_CUE_VOLUME)
    out.setdefault("cueMuted", False)
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
    """Countdown ended or skipped → logo-only idle until Start."""
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
    # Hide exercise+timer until Start — logo-only after countdown/skip.
    run["chromeVisible"] = False


def _enter_stream_ended(run: dict[str, Any], *, now: int | None = None) -> None:
    if now is None:
        now = _now_ms()
    dur = STREAM_ENDED_SOFT_CAP_SEC
    run["status"] = "stream_ended"
    run["streamEndedSec"] = dur
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
    run["breakStartedAtMs"] = None
    run["breakPausedRemainingMs"] = None


def _enter_break(
    run: dict[str, Any],
    next_idx: int,
    *,
    now: int | None = None,
    overflow_ms: int = 0,
) -> None:
    """Rest between exercises. segmentIndex points at the upcoming exercise."""
    if now is None:
        now = _now_ms()
    segments: list[dict[str, Any]] = run["segments"]
    nxt = segments[next_idx]
    try:
        dur = int(run.get("breakSec") or DEFAULT_BREAK_SEC)
    except (TypeError, ValueError):
        dur = DEFAULT_BREAK_SEC
    dur = max(1, min(5 * 60, dur))
    duration_ms = dur * 1000
    run["status"] = "break"
    run["segmentIndex"] = next_idx
    run["segmentDurationSec"] = nxt["durationSec"]
    run["segmentStartedAtMs"] = None
    run["pausedRemainingMs"] = None
    run["breakDurationSec"] = dur
    run["breakPausedRemainingMs"] = None
    run["chromeVisible"] = True
    if overflow_ms >= duration_ms:
        # Break already elapsed — start the next exercise immediately.
        run["breakStartedAtMs"] = None
        run["status"] = "running"
        run["segmentStartedAtMs"] = now - min(
            overflow_ms - duration_ms, int(nxt["durationSec"]) * 1000 - 1
        )
    else:
        run["breakStartedAtMs"] = now - max(0, overflow_ms)


def _finish_break(run: dict[str, Any], *, now: int | None = None) -> None:
    if now is None:
        now = _now_ms()
    segments: list[dict[str, Any]] = run["segments"]
    idx = int(run.get("segmentIndex") or 0)
    if idx < 0 or idx >= len(segments):
        _mark_done(run, now=now)
        return
    run["status"] = "running"
    run["segmentDurationSec"] = segments[idx]["durationSec"]
    run["segmentStartedAtMs"] = now
    run["pausedRemainingMs"] = None
    run["breakStartedAtMs"] = None
    run["breakPausedRemainingMs"] = None
    run["chromeVisible"] = True


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
    duration_ms = int(
        run.get("streamEndedDurationSec") or STREAM_ENDED_SOFT_CAP_SEC
    ) * 1000
    if now >= int(started) + duration_ms:
        _finish_stream_ended(run)
        run["updatedAt"] = _now_ms()


def _advance_break_if_needed(run: dict[str, Any]) -> None:
    if run["status"] != "break":
        return
    # Held (paused) break — wait for resume.
    if run.get("breakStartedAtMs") is None:
        return
    now = _now_ms()
    duration_ms = int(run.get("breakDurationSec") or DEFAULT_BREAK_SEC) * 1000
    end_at = int(run["breakStartedAtMs"]) + duration_ms
    if now < end_at:
        return
    overflow = now - end_at
    _finish_break(run, now=now)
    # Carry leftover overflow into the newly started exercise.
    if overflow > 0 and run["status"] == "running":
        started = int(run["segmentStartedAtMs"] or now)
        run["segmentStartedAtMs"] = started - overflow
        # Re-enter segment advance in case the exercise also expired.
    run["updatedAt"] = _now_ms()


def _advance_if_needed(run: dict[str, Any]) -> None:
    """Lazy server-side segment / break / countdown / stream-ended advance."""
    _advance_countdown_if_needed(run)
    _advance_stream_ended_if_needed(run)
    # Break may finish into running; then segment loop may continue.
    _advance_break_if_needed(run)
    if run["status"] == "break":
        return
    if run["status"] != "running":
        return
    segments: list[dict[str, Any]] = run["segments"]
    if not segments:
        _mark_done(run)
        return

    now = _now_ms()
    breaks_on = bool(run.get("breaksEnabled"))
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
        next_idx = idx + 1
        if breaks_on:
            _enter_break(run, next_idx, now=now, overflow_ms=overflow)
            # If overflow ate the whole break and started the next exercise,
            # keep looping when that exercise also expired.
            if run["status"] == "running":
                continue
            break
        run["segmentIndex"] = next_idx
        run["segmentDurationSec"] = segments[next_idx]["durationSec"]
        next_dur = int(segments[next_idx]["durationSec"]) * 1000
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
        _sync_flags_from_routine(existing, routine)
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
        _sync_flags_from_routine(existing, routine)
        return _public(existing)


def apply_command(
    key: str,
    command: str,
    duration_sec: int | None = None,
    cue_volume: float | None = None,
    cue_muted: bool | None = None,
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
            _sync_flags_from_routine(run, routine)

        segments: list[dict[str, Any]] = run["segments"]
        if not segments:
            raise ValueError("routine has no segments")

        now = _now_ms()
        status = run["status"]

        if command == "set_cue_audio":
            if cue_volume is not None:
                try:
                    vol = float(cue_volume)
                except (TypeError, ValueError) as e:
                    raise ValueError("cueVolume must be a number") from e
                run["cueVolume"] = max(0.0, min(1.0, vol))
            if cue_muted is not None:
                run["cueMuted"] = bool(cue_muted)
            _bump(run)
            return _public(run)

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
            if dur < 1:
                dur = 1
            if dur > 60 * 60:
                dur = 60 * 60
            _enter_countdown(run, dur, now=now)
            _bump(run)
            return _public(run)

        if command == "skip_countdown":
            # Ticking countdown, or held idle before Start (intro armed).
            # Prefer catalog flags so an editor toggle is honored immediately.
            flags = _routine_flags(routine)
            intro_held = (
                status == "idle"
                and flags["introCountdownEnabled"]
                and not run.get("introConsumed")
            )
            if status != "countdown" and not intro_held:
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
            run = _replace_idle(routine, run)
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
                run = _replace_idle(routine, run)
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
            if status == "break":
                # Start during a held/ticking break → begin the upcoming exercise.
                _finish_break(run, now=now)
                _bump(run)
                return _public(run)
            if status == "idle":
                # Routine-level intro: first Start runs pre-stream countdown.
                # Read flags from catalog (not a possibly-stale in-memory run).
                flags = _routine_flags(routine)
                if flags["introCountdownEnabled"] and not run.get("introConsumed"):
                    dur = int(flags["introCountdownSec"] or DEFAULT_COUNTDOWN_SEC)
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
            if status == "break" and run.get("breakStartedAtMs") is None:
                remaining = int(
                    run.get("breakPausedRemainingMs")
                    or int(run.get("breakDurationSec") or DEFAULT_BREAK_SEC) * 1000
                )
                dur = int(run.get("breakDurationSec") or DEFAULT_BREAK_SEC) * 1000
                run["breakStartedAtMs"] = now - (dur - remaining)
                run["breakPausedRemainingMs"] = None
                _bump(run)
                return _public(run)
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
            if status == "break" and run.get("breakStartedAtMs") is not None:
                started = int(run["breakStartedAtMs"])
                dur = int(run.get("breakDurationSec") or DEFAULT_BREAK_SEC) * 1000
                remaining = max(0, dur - (now - started))
                run["breakStartedAtMs"] = None
                run["breakPausedRemainingMs"] = remaining
                _bump(run)
                return _public(run)
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
            # Skip during break → start the upcoming exercise now.
            if status == "break":
                _finish_break(run, now=now)
                _bump(run)
                return _public(run)
            idx = int(run["segmentIndex"])
            if idx + 1 >= len(segments):
                run["segmentIndex"] = idx
                _mark_done(run, now=now)
                _bump(run)
                return _public(run)
            next_idx = idx + 1
            flags = _routine_flags(routine)
            # From a live exercise: customary break before the next one.
            if status == "running" and flags["breaksEnabled"]:
                _enter_break(run, next_idx, now=now, overflow_ms=0)
                _bump(run)
                return _public(run)
            run["segmentIndex"] = next_idx
            run["segmentDurationSec"] = segments[next_idx]["durationSec"]
            run["breakStartedAtMs"] = None
            run["breakPausedRemainingMs"] = None
            if status == "running":
                run["segmentStartedAtMs"] = now
                run["pausedRemainingMs"] = None
            elif status in ("idle", "paused"):
                run["status"] = "paused" if status == "paused" else "idle"
                run["segmentStartedAtMs"] = None
                run["pausedRemainingMs"] = segments[next_idx]["durationSec"] * 1000
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
                run["breakStartedAtMs"] = None
                run["breakPausedRemainingMs"] = None
                _bump(run)
                return _public(run)
            # Break sits before upcoming exercise at segmentIndex — back restarts
            # the previous exercise (or restarts exercise 0).
            idx = int(run["segmentIndex"])
            if status == "break":
                prev = max(0, idx - 1)
                run["segmentIndex"] = prev
                run["segmentDurationSec"] = segments[prev]["durationSec"]
                run["status"] = "running"
                run["segmentStartedAtMs"] = now
                run["pausedRemainingMs"] = None
                run["breakStartedAtMs"] = None
                run["breakPausedRemainingMs"] = None
                _bump(run)
                return _public(run)
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
