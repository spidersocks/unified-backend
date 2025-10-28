from fastapi import APIRouter, HTTPException, Request, Query, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language
from llm import tags_index
import httpx
import json
import re

router = APIRouter(tags=["LLM Chat (Bedrock KB)"])

ENROLLMENT_FORM_URL = "https://s24.q4cdn.com/216390268/files/doc_downloads/test.pdf"  # <-- Set your form's public URL here
ENROLLMENT_FORM_MARKER = "[SEND_ENROLLMENT_FORM]"

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None  # "en" | "zh-hk" | "zh-cn"
    session_id: Optional[str] = None
    debug: Optional[bool] = False   # return debug object in response

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    debug: Optional[Dict[str, Any]] = None

def extract_and_strip_marker(answer: str) -> (str, bool):
    # Detect the marker (case-insensitive, possibly with whitespace)
    pattern = re.compile(r"\s*\[SEND_ENROLLMENT_FORM\]\s*$", re.IGNORECASE)
    if answer and pattern.search(answer):
        answer = pattern.sub("", answer)
        return answer.rstrip(), True
    return answer, False

# ========== WhatsApp Helper: Send Message ==========
async def _send_whatsapp_message(to: str, message_body: str):
    if not SETTINGS.whatsapp_access_token or not SETTINGS.whatsapp_phone_number_id:
        print("ERROR: WhatsApp API credentials (access token or phone number ID) not configured. Cannot send message.")
        return

    url = f"https://graph.facebook.com/v18.0/{SETTINGS.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {SETTINGS.whatsapp_access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": message_body
        }
    }

    print(f"[WA] Sending WhatsApp message to: {to} | Body: {message_body}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            print(f"[WA] SUCCESS: WhatsApp message sent to {to}. Response: {response.json()}")
        except httpx.HTTPStatusError as e:
            print(f"[WA] ERROR: Failed to send WhatsApp message to {to}. Status: {e.response.status_code}, Detail: {e.response.text}")
        except httpx.RequestError as e:
            print(f"[WA] ERROR: An error occurred while requesting to send WhatsApp message to {to}: {e}")
        except Exception as e:
            print(f"[WA] ERROR: Unexpected error in _send_whatsapp_message: {e}")

async def _send_whatsapp_document(to: str, doc_url: str, filename: str = "enrollment_form.pdf"):
    if not SETTINGS.whatsapp_access_token or not SETTINGS.whatsapp_phone_number_id:
        print("ERROR: WhatsApp API credentials (access token or phone number ID) not configured. Cannot send document.")
        return

    url = f"https://graph.facebook.com/v18.0/{SETTINGS.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {SETTINGS.whatsapp_access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": {
            "link": doc_url,
            "filename": filename
        }
    }
    print(f"[WA] Sending WhatsApp document to: {to} | Document URL: {doc_url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            print(f"[WA] SUCCESS: WhatsApp document sent to {to}. Response: {response.json()}")
        except httpx.HTTPStatusError as e:
            print(f"[WA] ERROR: Failed to send WhatsApp document to {to}. Status: {e.response.status_code}, Detail: {e.response.text}")
        except httpx.RequestError as e:
            print(f"[WA] ERROR: An error occurred while requesting to send WhatsApp document to {to}: {e}")
        except Exception as e:
            print(f"[WA] ERROR: Unexpected error in _send_whatsapp_document: {e}")

# ===== /chat endpoint =====
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    print(f"[CHAT] /chat called with: message={req.message!r}, language={req.language!r}, session_id={req.session_id!r}, debug={req.debug!r}")
    print(f"[CHAT] Headers: {dict(request.headers)}")
    lang = req.language
    if not lang and req.session_id:
        lang = get_session_language(req.session_id)
        print(f"[CHAT] Used session_id={req.session_id} to get session language: {lang!r}")
    if not lang:
        lang = detect_language(req.message, accept_language=request.headers.get("accept-language"))
        print(f"[CHAT] Detected language: {lang!r}")
    if req.session_id and lang:
        remember_session_language(req.session_id, lang)
        print(f"[CHAT] Remembered session language: session_id={req.session_id}, lang={lang}")

    # Call LLM
    answer, citations, debug_info = chat_with_kb(
        req.message,
        lang,
        req.session_id,
        debug=bool(req.debug)
    )
    print(f"[CHAT] LLM raw answer: {answer!r}")
    print(f"[CHAT] LLM citations: {json.dumps(citations, indent=2)}")
    if debug_info:
        print(f"[CHAT] LLM debug_info: {json.dumps(debug_info, indent=2)}")

    # Postprocessing for enrollment form marker
    answer, send_form = extract_and_strip_marker(answer)

    if send_form:
        answer = (answer or "") + f"\n\nYou can download our enrollment form [here]({ENROLLMENT_FORM_URL})."
        print("[CHAT] Enrollment form marker detected: added PDF link to response.")

    silent_reason = debug_info.get("silence_reason") if isinstance(debug_info, dict) else None
    if not answer and silent_reason:
        print(f"[CHAT] LLM silenced answer. Reason: {silent_reason}")

    answer = answer or ""
    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

# ========== WhatsApp Webhook Endpoints ==========

@router.get("/whatsapp_webhook")
async def whatsapp_webhook_verification(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    token_preview = token[:5] + "..." if token else "None"
    print(f"[WA] Webhook Verification Request: Mode={mode}, Token={token_preview}, Challenge={challenge}")
    if mode == "subscribe" and token == SETTINGS.whatsapp_verify_token:
        print("[WA] Webhook verification successful!")
        return Response(content=challenge, media_type="text/plain", status_code=200)
    else:
        print(f"[WA] Webhook verification FAILED. Token mismatch or invalid mode. Expected: {SETTINGS.whatsapp_verify_token}")
        raise HTTPException(status_code=403, detail="Verification token mismatch or invalid mode")

@router.post("/whatsapp_webhook")
async def whatsapp_webhook_handler(request: Request):
    payload = await request.json()
    print(f"[WA] Received WhatsApp webhook payload:\n{json.dumps(payload, indent=2)}")

    try:
        if "object" in payload and "entry" in payload:
            for entry in payload["entry"]:
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        messages = value.get("messages", [])
                        contacts = value.get("contacts", [])

                        if messages and contacts:
                            message = messages[0]
                            contact = contacts[0]
                            from_number = message.get("from")
                            message_type = message.get("type")
                            print(f"[WA] Message details: from={from_number}, type={message_type}, contact={contact}")

                            if message_type == "text":
                                message_body = message["text"].get("body")
                                print(f"[WA] Text message from {from_number} (Name: {contact.get('profile',{}).get('name')}): '{message_body}'")

                                # Whitelist check
                                if from_number not in SETTINGS.whatsapp_test_numbers:
                                    print(f"[WA] WARNING: Message from non-whitelisted number {from_number} ignored during testing.")
                                    return {"status": "ignored", "reason": "not in test numbers"}

                                # Detect language
                                lang = detect_language(message_body)
                                print(f"[WA] Detected language: {lang}")
                                remember_session_language(from_number, lang)
                                print(f"[WA] Remembered session language for {from_number}: {lang}")

                                # Call LLM
                                answer, citations, debug_info = chat_with_kb(
                                    message_body,
                                    lang,
                                    session_id=None,
                                    debug=True
                                )
                                print(f"[WA] LLM raw answer: {answer!r}")
                                print(f"[WA] LLM citations: {json.dumps(citations, indent=2)}")
                                if debug_info:
                                    print(f"[WA] LLM debug_info: {json.dumps(debug_info, indent=2)}")
                                silent_reason = debug_info.get("silence_reason") if isinstance(debug_info, dict) else None
                                if not answer and silent_reason:
                                    print(f"[WA] LLM silenced answer. Reason: {silent_reason}")

                                # Postprocessing for enrollment form marker
                                answer, send_form = extract_and_strip_marker(answer)
                                if send_form:
                                    print("[WA] Enrollment form marker detected: sending PDF document.")
                                    await _send_whatsapp_document(from_number, ENROLLMENT_FORM_URL)
                                    # Optionally send text if present (e.g., extra instructions)
                                    if answer:
                                        await _send_whatsapp_message(from_number, answer)
                                elif answer:
                                    print(f"[WA] LLM Answer: '{answer}'")
                                    await _send_whatsapp_message(from_number, answer)
                                else:
                                    print(f"[WA] LLM provided no answer. No WhatsApp reply sent.")
                                return {"status": "ok", "message": "Message processed"}
                            else:
                                print(f"[WA] INFO: Received non-text message of type '{message_type}' from {from_number}. Ignoring.")
                                return {"status": "ignored", "reason": f"non-text message type: {message_type}"}
                        else:
                            print("[WA] INFO: No messages or contacts found in webhook payload.")
                            return {"status": "ignored", "reason": "no messages or contacts"}
        print("[WA] INFO: Webhook payload not a recognized message event.")
        return {"status": "ignored", "reason": "unrecognized payload structure"}

    except Exception as e:
        print(f"[WA] ERROR: Failed to process WhatsApp webhook payload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {e}")