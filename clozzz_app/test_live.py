"""Smoke checks for live-run state machine (run: python -m clozzz_app.test_live)."""

from __future__ import annotations

from clozzz_app import live, store


def main() -> None:
    live._RUNS.clear()
    catalog = store.get_catalog()
    key = catalog["routines"][0]["id"]

    armed = live.get_or_arm(key)
    assert armed["status"] == "idle", armed
    assert armed["segmentCount"] >= 1

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

    # Lazy advance: force segment expiry
    run = live._RUNS[key.lower()]
    run["status"] = "running"
    run["segmentIndex"] = 0
    run["segmentDurationSec"] = 1
    run["segments"][0]["durationSec"] = 1
    run["segmentStartedAtMs"] = live._now_ms() - 2000
    live._advance_if_needed(run)
    assert run["segmentIndex"] >= 1 or run["status"] == "done"

    live.apply_command(key, "restart")
    assert live._RUNS[key.lower()]["status"] == "idle"

    cd = live.apply_command(key, "start_countdown", duration_sec=60)
    assert cd["status"] == "countdown"
    assert cd["countdownDurationSec"] == 60
    assert cd["chromeVisible"] is False

    skipped = live.apply_command(key, "skip_countdown")
    assert skipped["status"] == "idle"
    assert skipped["chromeVisible"] is True

    live.apply_command(key, "hide_chrome")
    assert live._RUNS[key.lower()]["chromeVisible"] is False
    live.apply_command(key, "show_chrome")
    assert live._RUNS[key.lower()]["chromeVisible"] is True

    print("ok", key)


if __name__ == "__main__":
    main()
