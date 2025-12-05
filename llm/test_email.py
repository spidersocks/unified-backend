from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
import boto3

from llm.config import SETTINGS
from llm.reporting import fetch_messages_for_day, build_text_summary, render_csv
from llm.router import verify_credentials  # reuse Basic Auth
from llm.email_utils import send_raw_email_with_csv  # NEW

router = APIRouter(tags=["Admin/Test"])

def _build_and_send_today_summary() -> dict:
    messages, day = fetch_messages_for_day(None)
    subject = f"[LS] Chat Transcript (TEST) for {day.strftime('%Y-%m-%d')} (HKT)"
    body = build_text_summary(messages, day)
    if SETTINGS.base_public_url:
        body += f"\n\nLive report: {SETTINGS.base_public_url}/reports/daily?date={day.strftime('%Y-%m-%d')}"
    if SETTINGS.bridge_username and SETTINGS.bridge_password:
        body += f"\n\nAdmin access:\n- Username: {SETTINGS.bridge_username}\n- Password: {SETTINGS.bridge_password}"

    csv_bytes = render_csv(messages)
    csv_name = f"chat_transcript_{day.strftime('%Y-%m-%d')}.csv"

    recipients = [x.strip() for x in (SETTINGS.daily_summary_email_to or []) if x.strip()]
    if not recipients and SETTINGS.daily_summary_default_recipient:
        recipients = [SETTINGS.daily_summary_default_recipient.strip()]
    if not recipients:
        recipients = [SETTINGS.daily_summary_email_from.strip()]

    ok = send_raw_email_with_csv(subject, body, csv_bytes, csv_name, recipients)
    if not ok:
        raise RuntimeError("SES SendRawEmail failed")
    return {"ok": True, "sent_to": recipients, "count": len(messages)}

@router.post("/admin/test-email")
def send_test_email(_auth: bool = Depends(verify_credentials)):
    """
    Sends a test email with today's transcript (HKT) to DAILY_SUMMARY_EMAIL_TO,
    attaching the CSV to avoid body truncation.
    """
    try:
        return _build_and_send_today_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {e}")

# Optional GET convenience for browsers
@router.get("/admin/test-email")
def send_test_email_get(_auth: bool = Depends(verify_credentials)):
    try:
        return _build_and_send_today_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {e}")