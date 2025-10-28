import os
import time
from typing import List, Dict, Optional
import boto3

CHAT_HISTORY_TABLE = os.environ.get("CHAT_HISTORY_TABLE", "ChatHistory")
region = os.environ.get("AWS_REGION", "us-east-1")
dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table(CHAT_HISTORY_TABLE)

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
    table.put_item(Item=item)

def get_recent_history(session_id: str, limit: int = 6) -> List[Dict]:
    resp = table.query(
        KeyConditionExpression="session_id = :sid",
        ExpressionAttributeValues={":sid": session_id},
        ScanIndexForward=False,  # newest to oldest
        Limit=limit
    )
    # Return oldest to newest
    return list(reversed(resp.get("Items", [])))

def prune_history(session_id: str, keep: int = 6):
    """
    Keeps only the latest `keep` messages for the session.
    """
    resp = table.query(
        KeyConditionExpression="session_id = :sid",
        ExpressionAttributeValues={":sid": session_id},
        ScanIndexForward=True
    )
    items = resp.get("Items", [])
    excess = len(items) - keep
    if excess > 0:
        to_delete = items[:excess]
        with table.batch_writer() as batch:
            for item in to_delete:
                batch.delete_item(Key={"session_id": session_id, "ts": item["ts"]})

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
    # DynamoDB doesn't support mass delete; fetch and delete each item
    history = get_recent_history(session_id, limit=50, oldest_first=True)
    with table.batch_writer() as batch:
        for item in history:
            batch.delete_item(
                Key={"session_id": session_id, "ts": item["ts"]}
            )

# Optional: You may add TTL (Time To Live) for auto-expiry in your table setup,
# e.g., add "expire_at" attribute and set it to int(time.time()) + desired_lifetime_seconds.
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
    table.put_item(Item=item)