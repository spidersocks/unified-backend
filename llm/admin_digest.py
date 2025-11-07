import os
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import asyncio

import httpx
import boto3
from boto3.dynamodb.conditions import Key, Attr
import pytz

from llm.config import SETTINGS
from llm.intent import classify_scheduling_context
from llm.opening_hours import is_hk_public_holiday

_TZ = pytz.timezone(SETTINGS.admin_digest_tz)

# DynamoDB setup (optional; will fallback to in-memory if not available)
_USE_DDB = os.environ.get("USE_ADMIN_DIGEST_DDB", "true").lower() in ("1","true","yes")
region = os.environ.get("AWS_REGION", SETTINGS.aws_region)
_dynamodb = None
_table = None

def _log(msg: str):
    print(f"[ADMIN-DIGEST] {msg}", flush=True)

def _get_table():
    global _dynamodb, _table
    if _table is None and _USE_DDB:
        try:
            _dynamodb = boto3.resource("dynamodb", region_name=region)
            _table = _dynamodb.Table(SETTINGS.admin_digest_table)
        except Exception as e:
            _log(f"DDB init failed: {e}")
            return None
    return _table

# In-memory fallback
_MEM: Dict[Tuple[str, int], Dict[str, Any]] = {}  # key=(session_id, ts)

def _today_str_hk() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%d")

def _topic_from_flags(flags: Dict[str, Any]) -> Optional[str]:
    if not flags:
        return None
    if flags.get("availability_request"):
        return "Availability/Timetable"
    if flags.get("has_sched_verbs"):
        return "Leave/Reschedule/Cancel"
    if flags.get("staff_contact_request"):
        return "Pass-on/Contact staff"
    if flags.get("individual_homework_request"):
        return "Individual homework/Pronunciation"
    if flags.get("placement_question"):
        return "Placement/Level"
    if flags.get("has_policy_intent"):
        return "Policy question"
    return None

def add_pending(session_id: str, message: str, lang: str, flags: Dict[str, Any], ts: Optional[float] = None):
    """
    Record a pending unanswered parent message for daily digest.
    """
    ts_int = int(ts or time.time())
    item = {
        "date": _today_str_hk(),
        "session_id": session_id,
        "ts": ts_int,
        "message": message or "",
        "lang": lang or "",
        "flags": flags or {},
        "resolved": False,
    }
    tbl = _get_table()
    if tbl is None:
        _MEM[(session_id, ts_int)] = item
        return
    try:
        tbl.put_item(Item=item)
    except Exception as e:
        _log(f"DDB put_item failed, using memory fallback: {e}")
        _MEM[(session_id, ts_int)] = item

def resolve_session(session_id: str):
    """
    Mark all pending items for this session_id (for today) as resolved.
    """
    today = _today_str_hk()
    tbl = _get_table()
    if tbl is None:
        keys = [k for k in _MEM.keys() if k[0] == session_id]
        for k in keys:
            _MEM[k]["resolved"] = True
        return
    try:
        # Scan just today's items for the session_id
        resp = tbl.scan(
            FilterExpression=Attr("date").eq(today) & Attr("session_id").eq(session_id) & Attr("resolved").eq(False)
        )
        items = resp.get("Items", [])
        with tbl.batch_writer() as batch:
            for it in items:
                it["resolved"] = True
                batch.put_item(Item=it)
    except Exception as e:
        _log(f"DDB resolve_session scan failed: {type(e).__name__}: {e}")
        # Fallback: mark mem
        keys = [k for k in _MEM.keys() if k[0] == session_id]
        for k in keys:
            _MEM[k]["resolved"] = True

def list_unresolved_today(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return latest unresolved per session for today (deduplicated by session_id, keeping the latest ts).
    """
    today = _today_str_hk()
    tbl = _get_table()
    items: List[Dict[str, Any]] = []
    if tbl is None:
        items = [v for v in _MEM.values() if v.get("date") == today and not v.get("resolved")]
    else:
        try:
            resp = tbl.scan(
                FilterExpression=Attr("date").eq(today) & Attr("resolved").eq(False)
            )
            items = resp.get("Items", [])
        except Exception as e:
            _log(f"DDB scan failed, fallback to memory: {e}")
            items = [v for v in _MEM.values() if v.get("date") == today and not v.get("resolved")]

    # Deduplicate by session_id -> pick latest ts
    latest_by_session: Dict[str, Dict[str, Any]] = {}
    for it in items:
        sid = it.get("session_id", "")
        cur = latest_by_session.get(sid)
        if (cur is None) or (int(it.get("ts", 0)) > int(cur.get("ts", 0))):
            latest_by_session[sid] = it

    out = list(sorted(latest_by_session.values(), key=lambda x: int(x.get("ts", 0)), reverse=True))
    return out[:max(1, min(limit, SETTINGS.admin_digest_max_items))]

async def _send_whatsapp_text(to: str, body: str) -> bool:
    if not SETTINGS.whatsapp_access_token or not SETTINGS.whatsapp_phone_number_id:
        _log("WhatsApp credentials not configured.")
        return False
    url = f"https://graph.facebook.com/{SETTINGS.whatsapp_graph_version}/{SETTINGS.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {SETTINGS.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        return True
    except Exception as e:
        _log(f"WhatsApp send failed: {e}")
        return False

def _format_time_hk(ts: int) -> str:
    dt = datetime.fromtimestamp(ts, _TZ)
    return dt.strftime("%H:%M")

def _format_digest_body(items: List[Dict[str, Any]]) -> str:
    date_str = datetime.now(_TZ).strftime("%Y-%m-%d (%a)")
    lines = [f"[Daily Digest] Unanswered parent messages — {date_str}", ""]
    for it in items:
        sid = it.get("session_id", "")
        ts = int(it.get("ts", 0))
        msg = (it.get("message") or "").strip().replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:217] + "…"
        flags = it.get("flags") or {}
        topic = _topic_from_flags(flags) or "—"
        lines.append(f"- Chat: {sid} | { _format_time_hk(ts) } | Topic: {topic}")
        lines.append(f"  “{msg}”")
    return "\n".join(lines)

def _next_run_delay_seconds(now: datetime) -> float:
    """
    Compute seconds until the next 17:00 local time on an eligible day (Mon–Sat, not HK public holiday).
    """
    # Start from 'now' in local tz
    cur = now.astimezone(_TZ)
    for i in range(0, 8):  # search up to a week ahead
        candidate = cur + timedelta(days=i)
        weekday = candidate.weekday()  # Mon=0 ... Sun=6
        is_eligible_day = weekday <= 5  # Mon-Sat
        # Build 17:00 on that day
        target = candidate.replace(hour=SETTINGS.admin_digest_hour_local, minute=SETTINGS.admin_digest_minute_local, second=0, microsecond=0)
        if i == 0 and target <= cur:
            # Today's 17:00 already passed, try next day
            continue
        if not is_eligible_day:
            continue
        if is_hk_public_holiday(target):
            continue
        # Found next run time
        return (target - cur).total_seconds()
    # Fallback: 24 hours
    return 24 * 3600

# Simple single-process sent-flag (resets daily)
_SENT_FOR_DAY: Optional[str] = None

async def digest_scheduler_loop():
    global _SENT_FOR_DAY
    if not SETTINGS.admin_digest_enabled:
        _log("Admin digest disabled by config.")
        return
    _log("Admin digest scheduler started.")
    while True:
        try:
            now = datetime.now(_TZ)
            delay = _next_run_delay_seconds(now)
            _log(f"Next digest in {int(delay)}s")
            await asyncio.sleep(delay)

            # Double-check day eligibility and prevent duplicates within the same process
            day_key = _today_str_hk()
            if _SENT_FOR_DAY == day_key:
                _log("Digest already sent for today (process flag). Skipping.")
                # Sleep a day minus a small offset
                await asyncio.sleep(23 * 3600)
                continue

            items = list_unresolved_today(limit=SETTINGS.admin_digest_max_items)
            if not items:
                _log("No unresolved items for today. No digest sent.")
                _SENT_FOR_DAY = day_key
                continue

            body = _format_digest_body(items)
            ok = await _send_whatsapp_text(SETTINGS.admin_digest_director_number, body)
            if ok:
                _log(f"Digest sent to {SETTINGS.admin_digest_director_number} with {len(items)} items.")
                _SENT_FOR_DAY = day_key
            else:
                _log("Digest send failed; will retry at next cycle.")
                # do not set _SENT_FOR_DAY so it can try next cycle/day

            # After sending, we DO NOT auto-resolve items; admin will follow up.
        except asyncio.CancelledError:
            _log("Scheduler loop cancelled.")
            break
        except Exception as e:
            _log(f"Scheduler error: {e}")
            # Backoff a bit then recompute next run
            await asyncio.sleep(60)

def start_scheduler_background():
    """
    Create the background task for the digest loop.
    Call from app startup.
    """
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(digest_scheduler_loop())
        _log("Background task created.")
    except Exception as e:
        _log(f"Failed to start scheduler background: {e}")