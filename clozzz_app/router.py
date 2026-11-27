"""HTTP API for Clozzz routine catalog + live remote control."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from . import live, store, studio

router = APIRouter(prefix="/clozzz", tags=["Clozzz"])


class OrderBody(BaseModel):
    ids: list[str] = Field(min_length=1)


class LiveCommandBody(BaseModel):
    command: str = Field(min_length=1)
    durationSec: int | None = Field(default=None, ge=1, le=3600)


class ChannelBody(BaseModel):
    """Set or clear the OBS active channel (routine id / name / title)."""

    routineId: str | None = None


def _require_live_token(
    x_clozzz_live_token: str | None,
    authorization: str | None,
) -> None:
    expected = live.expected_live_token()
    if expected is None:
        return
    provided = (x_clozzz_live_token or "").strip()
    if not provided and authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        else:
            provided = auth
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing live token")


@router.get("/routines")
def list_routines() -> dict[str, Any]:
    return store.get_catalog()


@router.get("/routines/{key}")
def get_routine(key: str) -> dict[str, Any]:
    catalog = store.get_catalog()
    routine = store.find_routine(catalog, key)
    if not routine:
        raise HTTPException(status_code=404, detail=f"Unknown routine “{key}”")
    return routine


@router.put("/routines")
def put_catalog(body: dict[str, Any]) -> dict[str, Any]:
    try:
        catalog = store.save_catalog(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    studio.prune_missing_channel()
    return catalog


@router.put("/routines/order")
def put_order(body: OrderBody) -> dict[str, Any]:
    try:
        return store.reorder_routines(body.ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.put("/routines/{key}")
def put_routine(key: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    if "id" not in payload:
        payload["id"] = key
    elif str(payload.get("id")).strip().lower() != key.strip().lower():
        # Allow update keyed by current id/name; body.id wins as identity.
        pass
    try:
        return store.upsert_routine(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/routines/{key}")
def remove_routine(key: str) -> dict[str, Any]:
    try:
        catalog = store.delete_routine(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown routine “{key}”") from e
    # Drop studio channel if it pointed at the deleted routine.
    try:
        studio.clear_if_routine(key)
    except Exception:
        studio.prune_missing_channel()
    return catalog


@router.post("/routines/seed")
def seed_routines() -> dict[str, Any]:
    return store.reset_to_seed()


@router.get("/studio")
def get_studio() -> dict[str, Any]:
    """Active OBS channel (which routine the open display follows)."""
    return studio.get_channel()


@router.put("/studio/channel")
def put_studio_channel(
    body: ChannelBody,
    x_clozzz_live_token: str | None = Header(default=None, alias="X-Clozzz-Live-Token"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Remote selects the TV channel; open display polls this."""
    _require_live_token(x_clozzz_live_token, authorization)
    try:
        return studio.set_channel(body.routineId)
    except KeyError as e:
        raise HTTPException(
            status_code=404, detail=f"Unknown routine “{body.routineId}”"
        ) from e


@router.get("/studio/display")
def get_studio_display() -> dict[str, Any]:
    """
    One-poll payload for the open OBS display:
    active channel + live-run snapshot (armed idle when a channel is set).
    Never returns a deleted/missing routine id — prunes studio instead.
    """
    channel = studio.prune_missing_channel()
    rid = channel.get("activeRoutineId")
    if not rid:
        return {
            "activeRoutineId": None,
            "updatedAt": channel.get("updatedAt"),
            "live": None,
            "routine": None,
        }
    catalog = store.get_catalog()
    routine = store.find_routine(catalog, rid)
    if not routine:
        channel = studio.prune_missing_channel()
        return {
            "activeRoutineId": None,
            "updatedAt": channel.get("updatedAt"),
            "live": None,
            "routine": None,
        }
    try:
        snap = live.get_or_arm(routine["id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "activeRoutineId": routine["id"],
        "updatedAt": channel.get("updatedAt"),
        "live": snap,
        "routine": {
            "id": routine["id"],
            "title": routine.get("title") or routine.get("name") or routine["id"],
            "name": routine.get("name"),
            "introCountdownEnabled": bool(routine.get("introCountdownEnabled")),
            "introCountdownSec": routine.get("introCountdownSec"),
            "segments": routine.get("segments") or [],
        },
    }


@router.get("/live/{key}")
def get_live(key: str) -> dict[str, Any]:
    """Arm idle run if missing; return server-authoritative snapshot."""
    try:
        return live.get_or_arm(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown routine “{key}”") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/live/{key}/command")
def post_live_command(
    key: str,
    body: LiveCommandBody,
    x_clozzz_live_token: str | None = Header(default=None, alias="X-Clozzz-Live-Token"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_live_token(x_clozzz_live_token, authorization)
    try:
        snap = live.apply_command(key, body.command, duration_sec=body.durationSec)
        # Keep open display glued to whatever the controller is driving.
        try:
            studio.set_channel(key)
        except KeyError:
            pass
        return snap
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown routine “{key}”") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
