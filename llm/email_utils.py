from __future__ import annotations

import base64
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional
import boto3

from llm.config import SETTINGS

def build_raw_email_with_csv(
    subject: str,
    body_text: str,
    csv_bytes: bytes,
    csv_filename: str,
    from_email: str,
    to_emails: List[str],
) -> bytes:
    """
    Build a MIME message with a text body and a CSV attachment.
    Returns the raw bytes suitable for SES SendRawEmail.
    """
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)

    # Body
    text_part = MIMEText(body_text, "plain", "utf-8")
    msg.attach(text_part)

    # CSV attachment
    attach = MIMEApplication(csv_bytes, _subtype="csv")
    attach.add_header("Content-Disposition", f'attachment; filename="{csv_filename}"')
    msg.attach(attach)

    return msg.as_bytes()

def send_raw_email_with_csv(
    subject: str,
    body_text: str,
    csv_bytes: bytes,
    csv_filename: str,
    recipients: List[str],
    from_email: Optional[str] = None,
) -> bool:
    """
    Send a raw email with CSV via SES using SendRawEmail.
    """
    sender = from_email or SETTINGS.daily_summary_email_from
    if not sender or not recipients:
        return False
    try:
        raw = build_raw_email_with_csv(subject, body_text, csv_bytes, csv_filename, sender, recipients)
        ses = boto3.client("ses", region_name=SETTINGS.aws_region)
        ses.send_raw_email(
            Source=sender,
            Destinations=recipients,
            RawMessage={"Data": raw},
        )
        return True
    except Exception as e:
        print(f"[EMAIL] SendRawEmail failed: {e}", flush=True)
        return False