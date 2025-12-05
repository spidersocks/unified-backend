from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import boto3
from llm.opening_hours import HK_TZ, is_hk_public_holiday
from llm.config import SETTINGS
from llm.reporting import fetch_messages_for_day, build_text_summary


def _next_run_time(now_hk: datetime) -> datetime:
    """
    Compute next run at daily_summary_hour_local:daily_summary_minute_local HKT,
    skipping Sundays and HK public holidays.
    """
    target_hour = SETTINGS.daily_summary_hour_local
    target_min = SETTINGS.daily_summary_minute_local
    candidate = now_hk.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
    if candidate <= now_hk:
        candidate = candidate + timedelta(days=1)
    while candidate.weekday() == 6 or is_hk_public_holiday(candidate):
        candidate = candidate + timedelta(days=1)
        candidate = candidate.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
    return candidate


def _send_email_ses(subject: str, body: str, recipients: List[str]) -> bool:
    """
    Send email via AWS SES. Returns True on success, False otherwise.
    """
    if not SETTINGS.daily_summary_email_from or not recipients:
        return False
    try:
        ses = boto3.client("ses", region_name=SETTINGS.aws_region)
        ses.send_email(
            Source=SETTINGS.daily_summary_email_from,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        return True
    except Exception as e:
        print(f"[DAILY_SUMMARY] SES send failed: {e}", flush=True)
        return False


async def _runner():
    print("[DAILY_SUMMARY] Scheduler started.", flush=True)
    while True:
        now_hk = datetime.now(HK_TZ)
        nxt = _next_run_time(now_hk)
        sleep_secs = (nxt - now_hk).total_seconds()
        print(f"[DAILY_SUMMARY] Next run at {nxt.isoformat()} (in {int(sleep_secs)}s)", flush=True)
        await asyncio.sleep(max(1, int(sleep_secs)))

        # Build transcript for the last 24 hours ending at scheduled run time (yesterday’s date for full day)
        # For simplicity, we email "yesterday" full-day transcript.
        day_yesterday = (nxt - timedelta(days=1)).strftime("%Y-%m-%d")
        messages, day = fetch_messages_for_day(day_yesterday)

        subject = f"[LS] Chat Transcript for {day.strftime('%Y-%m-%d')} (HKT)"
        body = build_text_summary(messages, day)

        if SETTINGS.base_public_url:
            link = f"\n\nLive report: {SETTINGS.base_public_url}/reports/daily?date={day.strftime('%Y-%m-%d')}"
            body = body + link

        recipients = [x.strip() for x in (SETTINGS.daily_summary_email_to or []) if x.strip()]
        if not recipients and SETTINGS.daily_summary_default_recipient:
            recipients = [SETTINGS.daily_summary_default_recipient.strip()]

        sent = _send_email_ses(subject, body, recipients)
        if sent:
            print(f"[DAILY_SUMMARY] Email sent to {', '.join(recipients)}", flush=True)
        else:
            print(f"[DAILY_SUMMARY] Email not sent (SES not configured or recipients missing).", flush=True)


def start_daily_summary_scheduler_background():
    if not SETTINGS.daily_summary_enabled:
        print("[DAILY_SUMMARY] Disabled via settings.", flush=True)
        return
    loop = asyncio.get_event_loop()
    loop.create_task(_runner())