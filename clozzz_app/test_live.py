"""Smoke checks for live-run state machine (run: python -m clozzz_app.test_live)."""

from __future__ import annotations

from copy import deepcopy

from clozzz_app import live, store


def _set_routine_flags(key: str, **flags: object) -> dict:
    catalog = store.get_catalog()
    routine = store.find_routine(catalog, key)
    assert routine is not None
    updated = deepcopy(routine)
    updated.update(flags)
    store.upsert_routine(updated)
    return updated


def main() -> None:
    live._RUNS.clear()
    catalog = store.get_catalog()
    key = catalog["routines"][0]["id"]
    original = deepcopy(store.find_routine(catalog, key))

    try:
        _set_routine_flags(
            key,
            introCountdownEnabled=False,
            streamEndedEnabled=False,
            breaksEnabled=False,
        )

        armed = live.get_or_arm(key)
        assert armed["status"] == "idle", armed
        assert armed["segmentCount"] >= 1
        assert "introCountdownEnabled" in armed
        assert "streamEndedEnabled" in armed
        assert "breaksEnabled" in armed
        assert "breakSec" in armed

        started = live.apply_command(key, "start")
        assert started["status"] == "running"

        paused = live.apply_command(key, "pause")
        assert paused["status"] == "paused"
        assert isinstance(paused["pausedRemainingMs"], int)

        resumed = live.apply_command(key, "resume")
        assert resumed["status"] == "running"

        skipped = live.apply_command(key, "skip")
        assert skipped["segmentIndex"] == 1

        back = live.apply_command(key, "back")
        assert back["segmentIndex"] == 0

        # Lazy advance: force segment expiry (breaks off → next exercise)
        run = live._RUNS[key.lower()]
        run["status"] = "running"
        run["segmentIndex"] = 0
        run["segmentDurationSec"] = 1
        run["segments"][0]["durationSec"] = 1
        run["segmentStartedAtMs"] = live._now_ms() - 2000
        live._advance_if_needed(run)
        assert run["status"] == "running"
        assert run["segmentIndex"] >= 1

        before_rev = live._RUNS[key.lower()]["revision"]
        restarted = live.apply_command(key, "restart")
        assert restarted["status"] == "idle"
        assert restarted["revision"] > before_rev, (
            f"restart must keep revision monotonic ({before_rev} → {restarted['revision']})"
        )
        assert restarted["introConsumed"] is False

        cd = live.apply_command(key, "start_countdown", duration_sec=60)
        assert cd["status"] == "countdown"
        assert cd["countdownDurationSec"] == 60
        assert cd["chromeVisible"] is False

        skipped = live.apply_command(key, "skip_countdown")
        assert skipped["status"] == "idle"
        assert skipped["chromeVisible"] is False
        assert skipped["introConsumed"] is True

        live.apply_command(key, "hide_chrome")
        assert live._RUNS[key.lower()]["chromeVisible"] is False
        live.apply_command(key, "show_chrome")
        assert live._RUNS[key.lower()]["chromeVisible"] is True

        # Intro on Start when routine flag enabled
        live.apply_command(key, "restart")
        _set_routine_flags(
            key,
            introCountdownEnabled=True,
            introCountdownSec=90,
            streamEndedEnabled=False,
        )
        intro = live.apply_command(key, "start")
        assert intro["status"] == "countdown"
        assert intro["countdownDurationSec"] == 90
        live.apply_command(key, "skip_countdown")
        after = live.apply_command(key, "start")
        assert after["status"] == "running"

        # Skip from held idle (frozen intro) before Start ticks countdown
        live.apply_command(key, "restart")
        _set_routine_flags(
            key,
            introCountdownEnabled=True,
            introCountdownSec=120,
            streamEndedEnabled=False,
        )
        held_skip = live.apply_command(key, "skip_countdown")
        assert held_skip["status"] == "idle"
        assert held_skip["introConsumed"] is True
        assert held_skip["chromeVisible"] is False
        started_after_hold = live.apply_command(key, "start")
        assert started_after_hold["status"] == "running"
        assert started_after_hold["chromeVisible"] is True

        # Stream-ended graphic after finish (indefinite / soft-cap; remote dismiss)
        live.apply_command(key, "restart")
        _set_routine_flags(
            key,
            introCountdownEnabled=False,
            streamEndedEnabled=True,
            streamEndedMessage="Thanks for joining",
        )
        live.apply_command(key, "start")
        run = live._RUNS[key.lower()]
        run["status"] = "running"
        run["segmentIndex"] = len(run["segments"]) - 1
        run["segmentDurationSec"] = 1
        run["segments"][-1]["durationSec"] = 1
        run["segmentStartedAtMs"] = live._now_ms() - 2000
        live._advance_if_needed(run)
        assert run["status"] == "stream_ended"
        assert run["chromeVisible"] is False
        assert run["streamEndedDurationSec"] == live.STREAM_ENDED_SOFT_CAP_SEC
        assert run["streamEndedMessage"] == "Thanks for joining"
        dismissed = live.apply_command(key, "skip_stream_ended")
        assert dismissed["status"] == "done"
        assert dismissed["chromeVisible"] is True

        # Schema round-trip for new fields
        sample = store.get_catalog()["routines"][0]
        normalized = store._normalize_catalog({"schemaVersion": 1, "routines": [sample]})
        r0 = normalized["routines"][0]
        assert r0["introCountdownSec"] >= 60
        assert r0["streamEndedSec"] == 3600
        assert isinstance(r0["space"], list)
        assert "introCountdownMessage" in r0
        assert isinstance(r0["introCountdownEnabled"], bool)
        assert isinstance(r0["streamEndedEnabled"], bool)

        # Space migrate + intro message persist
        _set_routine_flags(
            key,
            space=["Floor mat", "Bed"],
            introCountdownMessage="Starting soon",
            introCountdownEnabled=True,
            introCountdownSec=120,
        )
        flagged = store.find_routine(store.get_catalog(), key)
        assert flagged["space"] == ["Floor mat", "Bed"]
        assert flagged["introCountdownMessage"] == "Starting soon"
        live.apply_command(key, "restart")
        armed_msg = live.get_or_arm(key)
        assert armed_msg["introCountdownMessage"] == "Starting soon"

        # Stale in-memory flags must re-sync from the catalog on get_or_arm
        live.apply_command(key, "restart")
        run = live._RUNS[key.lower()]
        run["introCountdownEnabled"] = False
        run["introCountdownSec"] = 30
        run["countdownDurationSec"] = 30
        run["introConsumed"] = False
        _set_routine_flags(
            key,
            introCountdownEnabled=True,
            introCountdownSec=600,
            streamEndedEnabled=False,
        )
        synced = live.get_or_arm(key)
        assert synced["introCountdownEnabled"] is True, synced
        assert synced["introCountdownSec"] == 600, synced
        assert synced["countdownDurationSec"] == 600, synced
        assert synced["status"] == "idle"

        # Cue audio prefs via remote command; survive restart
        cued = live.apply_command(
            key, "set_cue_audio", cue_volume=0.4, cue_muted=True
        )
        assert cued["cueVolume"] == 0.4
        assert cued["cueMuted"] is True
        after_audio = live.apply_command(key, "restart")
        assert after_audio["cueVolume"] == 0.4
        assert after_audio["cueMuted"] is True

        # Breaks between exercises: expire → break → finish / skip / pause
        live.apply_command(key, "restart")
        _set_routine_flags(
            key,
            introCountdownEnabled=False,
            streamEndedEnabled=False,
            breaksEnabled=True,
            breakSec=10,
        )
        live.apply_command(key, "start")
        run = live._RUNS[key.lower()]
        run["status"] = "running"
        run["segmentIndex"] = 0
        run["segmentDurationSec"] = 1
        run["segments"][0]["durationSec"] = 1
        run["segmentStartedAtMs"] = live._now_ms() - 2000
        live._advance_if_needed(run)
        assert run["status"] == "break", run
        assert run["segmentIndex"] == 1
        assert run["breakDurationSec"] == 10
        assert run["breakStartedAtMs"] is not None

        paused_break = live.apply_command(key, "pause")
        assert paused_break["status"] == "break"
        assert paused_break["breakStartedAtMs"] is None
        assert isinstance(paused_break["breakPausedRemainingMs"], int)

        resumed_break = live.apply_command(key, "resume")
        assert resumed_break["status"] == "break"
        assert resumed_break["breakStartedAtMs"] is not None

        skipped_break = live.apply_command(key, "skip")
        assert skipped_break["status"] == "running"
        assert skipped_break["segmentIndex"] == 1

        # Skip from running with breaks → enter break before next
        into_break = live.apply_command(key, "skip")
        assert into_break["status"] == "break"
        assert into_break["segmentIndex"] == 2

        # Schema round-trip for break fields
        sample2 = store.get_catalog()["routines"][0]
        normalized2 = store._normalize_catalog(
            {"schemaVersion": 1, "routines": [sample2]}
        )
        assert normalized2["routines"][0]["breaksEnabled"] is True
        assert normalized2["routines"][0]["breakSec"] == 10
    finally:
        # Restore original routine JSON so local store stays clean
        store.upsert_routine(original)
        live._RUNS.clear()

    print("ok", key)


if __name__ == "__main__":
    main()
