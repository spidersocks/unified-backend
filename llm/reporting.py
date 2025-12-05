from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import csv
import io
import html

from boto3.dynamodb.conditions import Attr  # type: ignore

from llm.opening_hours import HK_TZ
from llm import chat_history as ch


def _day_bounds_hkt(date_str: Optional[str]) -> Tuple[int, int, datetime]:
    """
    Returns (start_ts, end_ts, date_obj) for the given YYYY-MM-DD in Asia/Hong_Kong.
    If date_str is None, uses today (HKT).
    """
    now_hk = datetime.now(HK_TZ)
    if date_str:
        y, m, d = map(int, date_str.split("-"))
        day = HK_TZ.localize(datetime(y, m, d, 0, 0, 0))
    else:
        day = HK_TZ.localize(datetime(now_hk.year, now_hk.month, now_hk.day, 0, 0, 0))
    start = int(day.timestamp())
    end = int((day + timedelta(days=1) - timedelta(seconds=1)).timestamp())
    return start, end, day


def _scan_ddb_between(start_ts: int, end_ts: int) -> List[Dict]:
    """
    Scan ChatHistory table for items between ts range.
    Note: This is a Scan+Filter (table-wide); suitable for modest volumes.
    """
    tbl = ch._get_table()
    if not tbl:
        # Local dev / in-memory fallback
        out: List[Dict] = []
        for sid, records in ch._MEM_HISTORY.items():  # type: ignore[attr-defined]
            for item in records:
                ts = int(item.get("ts", 0))
                if start_ts <= ts <= end_ts:
                    out.append(dict(item))
        out.sort(key=lambda x: int(x.get("ts", 0)))
        return out

    items: List[Dict] = []
    start_key = None
    while True:
        kwargs = {"FilterExpression": Attr("ts").between(start_ts, end_ts)}
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        resp = tbl.scan(**kwargs)
        items.extend(resp.get("Items", []))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break
    items.sort(key=lambda x: int(x.get("ts", 0)))
    return items


def fetch_messages_for_day(date_str: Optional[str]) -> Tuple[List[Dict], datetime]:
    start_ts, end_ts, day = _day_bounds_hkt(date_str)
    return _scan_ddb_between(start_ts, end_ts), day


def group_by_session(messages: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for m in messages:
        sid = str(m.get("session_id") or "unknown")
        grouped.setdefault(sid, []).append(m)
    return grouped


def _fmt_hkt(ts: int) -> str:
    dt = datetime.fromtimestamp(int(ts), tz=HK_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def render_html(messages: List[Dict], day: datetime) -> str:
    grouped = group_by_session(messages)
    title = f"Daily Chat Transcript — {day.strftime('%Y-%m-%d')} (HKT)"
    total = len(messages)
    sessions = len(grouped)

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:16px}h2{margin-top:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px}th{background:#f6f6f6;text-align:left}code{background:#f2f2f2;padding:1px 4px;border-radius:3px}</style>",
             "</head><body>"]
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f"<p>Total messages: <b>{total}</b> &nbsp; Unique sessions: <b>{sessions}</b></p>")
    parts.append("<p>Formats: <a href='?format=json'>JSON</a> | <a href='?format=csv'>CSV</a></p>")

    for sid, msgs in grouped.items():
        inbound = sum(1 for m in msgs if m.get("role") == "user")
        outbound = sum(1 for m in msgs if m.get("role") == "bot" and (m.get("message") or "").strip())
        parts.append(f"<h2>Session: <code>{html.escape(sid)}</code></h2>")
        parts.append(f"<p>Inbound: {inbound} &nbsp; Outbound: {outbound} &nbsp; Total: {len(msgs)}</p>")
        parts.append("<table><thead><tr><th>Time (HKT)</th><th>Role</th><th>Lang</th><th>Message</th></tr></thead><tbody>")
        for m in msgs:
            ts = _fmt_hkt(int(m.get("ts", 0)))
            role = html.escape(str(m.get("role", "")))
            lang = html.escape(str(m.get("lang", "")))
            msg = html.escape(str(m.get("message", "")))
            parts.append(f"<tr><td>{ts}</td><td>{role}</td><td>{lang}</td><td>{msg}</td></tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "".join(parts)


def render_csv(messages: List[Dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["session_id", "ts_iso_hkt", "role", "lang", "message"])
    for m in messages:
        w.writerow([
            m.get("session_id", ""),
            _fmt_hkt(int(m.get("ts", 0))),
            m.get("role", ""),
            m.get("lang", ""),
            (m.get("message") or "").replace("\r", " ").replace("\n", " ").strip(),
        ])
    return buf.getvalue().encode("utf-8")


def render_json(messages: List[Dict], day: datetime) -> Dict:
    grouped = group_by_session(messages)
    return {
        "date_hkt": day.strftime("%Y-%m-%d"),
        "total_messages": len(messages),
        "sessions": len(grouped),
        "by_session": grouped,
    }