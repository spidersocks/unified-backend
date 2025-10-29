import os
import time
from typing import List, Dict, Optional
import boto3
from boto3.dynamodb.conditions import Key

# Optional in-memory fallback if DynamoDB is unavailable (local dev or credentials issue)
_USE_DDB = os.environ.get("USE_DYNAMODB_HISTORY", "true").lower() in ("1", "true", "yes")
CHAT_HISTORY_TABLE = os.environ.get("CHAT_HISTORY_TABLE", "ChatHistory")
region = os.environ.get("AWS_REGION", "us-east-1")

# Lazy init to avoid import-time failures
_dynamodb = None
_table = None

def _get_table():
    global _dynamodb, _table
    if _table is None and _USE_DDB:
        _dynamodb = boto3.resource("dynamodb", region_name=region)
        _table = _dynamodb.Table(CHAT_HISTORY_TABLE)
    return _table

# Simple in-memory fallback store
_MEM_HISTORY: Dict[str, List[Dict]] = {}

def _mem_save(item: Dict):
    sid = item["session_id"]
    _MEM_HISTORY.setdefault(sid, []).append(item)
    # Keep only latest 50 in memory
    _MEM_HISTORY[sid] = sorted(_MEM_HISTORY[sid], key=lambda x: x["ts"])[-50:]

def _mem_get_recent(session_id: str, limit: int, oldest_first: bool) -> List[Dict]:
    items = _MEM_HISTORY.get(session_id, [])
    items = sorted(items, key=lambda x: x["ts"], reverse=not oldest_first)[:limit]
    return list(sorted(items, key=lambda x: x["ts"]))

def _mem_prune(session_id: str, keep: int):
    items = _MEM_HISTORY.get(session_id, [])
    if len(items) > keep:
        _MEM_HISTORY[session_id] = sorted(items, key=lambda x: x["ts"])[-keep:]

def save_message(session_id: str, role: str, message: str, lang: Optional[str] = None, timestamp: Optional[float] = None):
    ts = int(timestamp or time.time())
    item = {
        "session_id": session_id,
        "ts": ts,
        "role": role,
        "message": message,
    }
    if lang:
        item["lang"] = lang
    try:
        tbl = _get_table()
        if tbl is None:
            _mem_save(item)
            return
        tbl.put_item(Item=item)
    except Exception:
        _mem_save(item)

def get_recent_history(session_id: str, limit: int = 6, oldest_first: bool = False) -> List[Dict]:
    try:
        tbl = _get_table()
        if tbl is None:
            return _mem_get_recent(session_id, limit, oldest_first)
        resp = tbl.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=oldest_first,  # True=oldest->newest, False=newest->oldest
            Limit=limit
        )
        items = resp.get("Items", [])
        # Ensure return oldest->newest
        return list(sorted(items, key=lambda x: x["ts"]))
    except Exception:
        return _mem_get_recent(session_id, limit, oldest_first)

def prune_history(session_id: str, keep: int = 6):
    """
    Keeps only the latest `keep` messages for the session.
    """
    try:
        tbl = _get_table()
        if tbl is None:
            _mem_prune(session_id, keep)
            return
        # Get all items (paginate if needed)
        items: List[Dict] = []
        last_evaluated_key = None
        while True:
            kwargs = {
                "KeyConditionExpression": Key("session_id").eq(session_id),
                "ScanIndexForward": True
            }
            if last_evaluated_key:
                kwargs["ExclusiveStartKey"] = last_evaluated_key
            resp = tbl.query(**kwargs)
            items.extend(resp.get("Items", []))
            last_evaluated_key = resp.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break
        items = sorted(items, key=lambda x: x["ts"])
        excess = len(items) - keep
        if excess > 0:
            to_delete = items[:excess]
            with tbl.batch_writer() as batch:
                for item in to_delete:
                    batch.delete_item(Key={"session_id": session_id, "ts": item["ts"]})
    except Exception:
        _mem_prune(session_id, keep)

def build_context_string(
    history: List[Dict],
    new_message: Optional[str] = None,
    user_role: str = "user",
    bot_role: str = "bot",
    include_new: bool = True,
) -> str:
    """
    Build a context string from history for LLM prompt.
    :param history: List of message dicts from get_recent_history.
    :param new_message: The latest user message (optional).
    :param user_role: Label for user turns.
    :param bot_role: Label for bot turns.
    :param include_new: If True, append new_message at the end.
    :return: Multiline context string.
    """
    lines = []
    for msg in history:
        prefix = "Parent:" if msg["role"] == user_role else "Bot:"
        lines.append(f"{prefix} {msg['message']}")
    if include_new and new_message:
        lines.append(f"Parent: {new_message}")
    return "\n".join(lines)

def clear_history(session_id: str) -> None:
    """
    Delete all messages for a session_id (useful for privacy or reset).
    """
    try:
        tbl = _get_table()
        if tbl is None:
            _MEM_HISTORY.pop(session_id, None)
            return
        # Fetch all and delete
        last_evaluated_key = None
        while True:
            kwargs = {
                "KeyConditionExpression": Key("session_id").eq(session_id),
                "ScanIndexForward": True
            }
            if last_evaluated_key:
                kwargs["ExclusiveStartKey"] = last_evaluated_key
            resp = tbl.query(**kwargs)
            items = resp.get("Items", [])
            if not items:
                break
            with tbl.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={"session_id": session_id, "ts": item["ts"]})
            last_evaluated_key = resp.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break
    except Exception:
        _MEM_HISTORY.pop(session_id, None)

# Optional: TTL save helper (uses DynamoDB TTL if configured on 'expire_at')
def save_message_with_ttl(
    session_id: str,
    role: str,
    message: str,
    lang: Optional[str] = None,
    ttl_seconds: int = 7 * 24 * 60 * 60,  # default: 7 days
):
    ts = int(time.time())
    item = {
        "session_id": session_id,
        "ts": ts,
        "role": role,
        "message": message,
        "expire_at": ts + ttl_seconds
    }
    if lang:
        item["lang"] = lang
    try:
        tbl = _get_table()
        if tbl is None:
            _mem_save(item)
            return
        tbl.put_item(Item=item)
    except Exception:
        _mem_save(item)