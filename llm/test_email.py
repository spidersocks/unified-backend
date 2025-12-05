from __future__ import annotations

from typing import List
from fastapi import APIRouter, Depends, HTTPException
import boto3

from llm.config import SETTINGS
from llm.reporting import fetch_messages_for_day, build_text_summary
from llm.router import verify_credentials  # reuse Basic Auth

router = APIRouter(tags=["Admin/Test"])

def _send_email_ses(subject: str, body: str, recipients: List[str]) -> None:
    if not SETTINGS.aws_region:
        raise RuntimeError("AWS_REGION not configured")
    if not SETTINGS.daily_summary_email_from:
        raise RuntimeError("DAILY_SUMMARY_EMAIL_FROM not configured")
    if not recipients:
        raise RuntimeError("No recipients provided")

    ses = boto3.client("ses", region_name=SETTINGS.aws_region)
    ses.send_email(
        Source=SETTINGS.daily_summary_email_from,
        Destination={"ToAddresses": recipients},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
        },
    )

@router.post("/admin/test-email")
def send_test_email(_auth: bool = Depends(verify_credentials)):
    """
    Sends a test email with today's transcript (HKT) to DAILY_SUMMARY_EMAIL_TO.
    Uses SES in SETTINGS.aws_region. Auth via Basic Auth (same as /chat).
    """
    try:
        # Today (HKT)
        messages, day = fetch_messages_for_day(None)
        subject = f"[LS] Chat Transcript (TEST) for {day.strftime('%Y-%m-%d')} (HKT)"
        body = build_text_summary(messages, day)
        # Optional: link to live report if configured
        if SETTINGS.base_public_url:
            body += f"\n\nLive report: {SETTINGS.base_public_url}/reports/daily?date={day.strftime('%Y-%m-%d')}"

        recipients = [x.strip() for x in (SETTINGS.daily_summary_email_to or []) if x.strip()]
        if not recipients and SETTINGS.daily_summary_default_recipient:
            recipients = [SETTINGS.daily_summary_default_recipient.strip()]
        if not recipients:
            # If nothing set, default to sending to FROM address (self-email)
            recipients = [SETTINGS.daily_summary_email_from.strip()]

        _send_email_ses(subject, body, recipients)
        return {"ok": True, "sent_to": recipients, "count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {e}")