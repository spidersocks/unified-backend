import os
import time
from typing import List, Dict, Optional
import boto3
from boto3.dynamodb.conditions import Key

# Optional in-memory fallback if DynamoDB is unavailable (local dev or credentials issue)
_USE_DDB = os.environ.get("USE_DYNAMODB_HISTORY", "true").lower() in ("1", "true", "yes")
CHAT_HISTORY_TABLE = os.environ.get("CHAT_HISTORY_TABLE", "ChatHistory")
region = os.environ.get("AWS_REGION", "us-east-1")

# Keep-all window (seconds) before pruning-by-count kicks in (default 24h)
HISTORY_RECENT_WINDOW_SECS = int(os.environ.get("HISTORY_RECENT_WINDOW_SECS", "86400"))

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
    # Keep only latest 50 in memory (coarse safety cap for dev)
    _MEM_HISTORY[sid] = sorted(_MEM_HISTORY[sid], key=lambda x: x["ts"])[-50:]

def _mem_get_recent(session_id: str, limit: int, oldest_first: bool) -> List[Dict]:
    items = _MEM_HISTORY.get(session_id, [])
    items = sorted(items, key=lambda x: x["ts"], reverse=not oldest_first)[:limit]
    return list(sorted(items, key=lambda x: x["ts"]))

def _mem_prune(session_id: str, keep: int, keep_recent_window_secs: Optional[int]):
    items = _MEM_HISTORY.get(session_id, [])
    if not items:
        return
    items = sorted(items, key=lambda x: x["ts"])  # oldest -> newest
    if keep_recent_window_secs and keep_recent_window_secs > 0:
        cutoff = int(time.time()) - int(keep_recent_window_secs)
        older = [it for it in items if int(it.get("ts", 0)) < cutoff]
        newer = [it for it in items if int(it.get("ts", 0)) >= cutoff]
        # Keep last `keep` of older, keep ALL newer
        older_tail = older[-keep:] if keep > 0 else []
        _MEM_HISTORY[session_id] = older_tail + newer
    else:
        # Legacy behavior: keep only last `keep`
        _MEM_HISTORY[session_id] = items[-keep:] if keep > 0 else []

def save_message(session_id: str, role: str, message: str, lang: Optional[str] = None, timestamp: Optional[float] = None):
    """
    Legacy save without TTL. Kept for compatibility. Prefer save_message_with_ttl for production.
    """
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

def prune_history(session_id: str, keep: int = 6, keep_recent_window_secs: Optional[int] = HISTORY_RECENT_WINDOW_SECS):
    """
    Prune policy:
      - Keep ALL messages from the last `keep_recent_window_secs` (default: 24 hours).
      - For older messages, keep only the most recent `keep` (default: 6).
      - If `keep_recent_window_secs` <= 0 or None, behaves like legacy: keep only last `keep`.
    """
    try:
        tbl = _get_table()
        now_ts = int(time.time())
        if tbl is None:
            _mem_prune(session_id, keep, keep_recent_window_secs)
            return

        # Fetch all items for this session (paginated)
        items: List[Dict] = []
        last_evaluated_key = None
        while True:
            kwargs = {
                "KeyConditionExpression": Key("session_id").eq(session_id),
                "ScanIndexForward": True  # oldest -> newest
            }
            if last_evaluated_key:
                kwargs["ExclusiveStartKey"] = last_evaluated_key
            resp = tbl.query(**kwargs)
            items.extend(resp.get("Items", []))
            last_evaluated_key = resp.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

        if not items:
            return

        items = sorted(items, key=lambda x: int(x.get("ts", 0)))  # oldest -> newest

        if keep_recent_window_secs and keep_recent_window_secs > 0:
            cutoff = now_ts - int(keep_recent_window_secs)
            older = [it for it in items if int(it.get("ts", 0)) < cutoff]
            newer = [it for it in items if int(it.get("ts", 0)) >= cutoff]
            # Keep last `keep` of older, keep ALL newer
            to_keep = (older[-keep:] if keep > 0 else []) + newer
            # Delete anything that's not in to_keep
            keep_keys = {(it["session_id"], int(it["ts"])) for it in to_keep}
            to_delete = [it for it in items if (it["session_id"], int(it["ts"])) not in keep_keys]
        else:
            # Legacy behavior: keep only last `keep`
            to_delete = items[:-keep] if keep > 0 else items

        if not to_delete:
            return

        with tbl.batch_writer() as batch:
            for item in to_delete:
                batch.delete_item(Key={"session_id": session_id, "ts": int(item["ts"])})
    except Exception:
        # Fallback to in-memory pruning logic
        _mem_prune(session_id, keep, keep_recent_window_secs)

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
                    batch.delete_item(Key={"session_id": session_id, "ts": int(item["ts"])})
            last_evaluated_key = resp.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break
    except Exception:
        _MEM_HISTORY.pop(session_id, None)

# TTL save helper (uses DynamoDB TTL on 'expire_at')
HALF_YEAR_SECONDS = 180 * 24 * 60 * 60  # ~6 months

def save_message_with_ttl(
    session_id: str,
    role: str,
    message: str,
    lang: Optional[str] = None,
    timestamp: Optional[float] = None,
    ttl_seconds: int = HALF_YEAR_SECONDS,
):
    """
    Save a message with DynamoDB TTL set via 'expire_at'.
    To activate auto-deletion, enable TTL on the ChatHistory table with 'expire_at' as the TTL attribute.
    """
    ts = int(timestamp or time.time())
    item = {
        "session_id": session_id,
        "ts": ts,
        "role": role,
        "message": message,
        "expire_at": ts + int(ttl_seconds),
    }
    if lang:
        item["lang"] = lang
    try:
        tbl = _get_table()
        if tbl is None:
            # In-memory dev fallback: ignore TTL
            _mem_save(item)
            return
        tbl.put_item(Item=item)
    except Exception:
        _mem_save(item)