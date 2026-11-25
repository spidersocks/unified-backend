"""HTTP API for Clozzz routine catalog + live remote control."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from . import live, store

router = APIRouter(prefix="/clozzz", tags=["Clozzz"])


class OrderBody(BaseModel):
    ids: list[str] = Field(min_length=1)


class LiveCommandBody(BaseModel):
    command: str = Field(min_length=1)
    durationSec: int | None = Field(default=None, ge=5, le=3600)


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
        return store.save_catalog(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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
        return store.delete_routine(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown routine “{key}”") from e


@router.post("/routines/seed")
def seed_routines() -> dict[str, Any]:
    return store.reset_to_seed()


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
        return live.apply_command(key, body.command, duration_sec=body.durationSec)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown routine “{key}”") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
