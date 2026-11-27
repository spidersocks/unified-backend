"""Smoke checks for active OBS channel (run: python -m clozzz_app.test_studio)."""

from __future__ import annotations

from clozzz_app import studio, store


def main() -> None:
    catalog = store.get_catalog()
    assert catalog["routines"], "catalog empty"
    key = catalog["routines"][0]["id"]
    other = catalog["routines"][1]["id"] if len(catalog["routines"]) > 1 else key

    cleared = studio.set_channel(None)
    assert cleared["activeRoutineId"] is None
    assert cleared["updatedAt"]

    ch = studio.set_channel(key)
    assert ch["activeRoutineId"] == key
    assert studio.get_channel()["activeRoutineId"] == key

    ch2 = studio.set_channel(other)
    assert ch2["activeRoutineId"] == other

    try:
        studio.set_channel("__no_such_routine__")
        raise AssertionError("expected KeyError for unknown routine")
    except KeyError:
        pass

    # Disk round-trip (best-effort on ephemeral FS)
    studio._STATE["activeRoutineId"] = None
    studio._STATE["updatedAt"] = None
    reloaded = studio.get_channel()
    assert reloaded["activeRoutineId"] == other, reloaded

    # Ghost channel (deleted routine id) must prune on read
    studio._STATE["activeRoutineId"] = "__deleted_ghost__"
    studio._STATE["updatedAt"] = "2026-01-01T00:00:00Z"
    pruned = studio.get_channel()
    assert pruned["activeRoutineId"] is None, pruned

    studio.set_channel(key)
    cleared = studio.clear_if_routine(key)
    assert cleared["activeRoutineId"] is None, cleared

    studio.set_channel(None)
    print("ok: studio channel")


if __name__ == "__main__":
    main()
