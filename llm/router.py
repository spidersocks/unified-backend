from fastapi import APIRouter, HTTPException, Request, Query, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language
from llm import tags_index

import httpx
import json

# ========== WhatsApp Helper: Send Message ==========
async def _send_whatsapp_message(to: str, message_body: str):
    """Sends a text message back to a WhatsApp user via the Cloud API."""
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

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            print(f"SUCCESS: WhatsApp message sent to {to}. Response: {response.json()}")
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Failed to send WhatsApp message to {to}. Status: {e.response.status_code}, Detail: {e.response.text}")
        except httpx.RequestError as e:
            print(f"ERROR: An error occurred while requesting to send WhatsApp message to {to}: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected error in _send_whatsapp_message: {e}")

# ========== LLM Chat API ==========
router = APIRouter(tags=["LLM Chat (Bedrock KB)"])

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None  # "en" | "zh-hk" | "zh-cn"
    session_id: Optional[str] = None
    debug: Optional[bool] = False   # return debug object in response

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    debug: Optional[Dict[str, Any]] = None

# ---- Opening hours helpers omitted for brevity, keep as in your original router.py ----
# (If you want to keep the special opening hours logic, include those helper functions too)

# ===== /chat endpoint (unchanged) =====
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    lang = req.language
    if not lang and req.session_id:
        lang = get_session_language(req.session_id)
    if not lang:
        lang = detect_language(req.message, accept_language=request.headers.get("accept-language"))
    if req.session_id and lang:
        remember_session_language(req.session_id, lang)

    # ...Opening hours intent routing logic (as in your current router.py)...

    # For simplicity, we'll call chat_with_kb directly here (or your intent-boosted logic)
    answer, citations, debug_info = chat_with_kb(
        req.message,
        lang,
        req.session_id,
        debug=bool(req.debug)
    )
    answer = answer or ""
    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

# ========== WhatsApp Webhook Endpoints ==========

@router.get("/whatsapp_webhook")
async def whatsapp_webhook_verification(request: Request):
    """Webhook verification for Meta."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    token_preview = token[:5] + "..." if token else "None"
    print(f"INFO: Webhook Verification Request received. Mode: {mode}, Token: {token_preview}, Challenge: {challenge}")
    if mode == "subscribe" and token == SETTINGS.whatsapp_verify_token:
        print("INFO: Webhook verification successful!")
        return Response(content=challenge, media_type="text/plain", status_code=200)
    else:
        print(f"ERROR: Webhook verification failed. Token mismatch or invalid mode. Expected token: {SETTINGS.whatsapp_verify_token}")
        raise HTTPException(status_code=403, detail="Verification token mismatch or invalid mode")

@router.post("/whatsapp_webhook")
async def whatsapp_webhook_handler(request: Request):
    """Handles incoming WhatsApp messages, processes with LLM, and replies."""
    payload = await request.json()
    print(f"INFO: Received WhatsApp webhook payload:\n{json.dumps(payload, indent=2)}")

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
                            from_number = message.get("from")  # Sender's phone number (WA_ID)
                            message_type = message.get("type")

                            if message_type == "text":
                                message_body = message["text"].get("body")
                                print(f"INFO: Message from {from_number} (Name: {contact.get('profile',{}).get('name')}): '{message_body}'")

                                # Whitelist check
                                if from_number not in SETTINGS.whatsapp_test_numbers:
                                    print(f"WARNING: Message from non-whitelisted number {from_number} ignored during testing.")
                                    return {"status": "ignored", "reason": "not in test numbers"}

                                # Detect language (same as /chat)
                                lang = detect_language(message_body)
                                remember_session_language(from_number, lang)

                                # Call same core logic as /chat
                                answer, citations, debug_info = chat_with_kb(
                                    message_body,
                                    lang,
                                    session_id=from_number,
                                    debug=False
                                )

                                if answer:
                                    print(f"INFO: LLM Answer: '{answer}'")
                                    await _send_whatsapp_message(from_number, answer)
                                else:
                                    fallback_message = "Sorry, I couldn't find an answer to that. Please try rephrasing your question or contact our staff."
                                    print(f"WARNING: LLM provided no answer. Sending fallback: '{fallback_message}'")
                                    await _send_whatsapp_message(from_number, fallback_message)
                                return {"status": "ok", "message": "Message processed"}
                            else:
                                print(f"INFO: Received non-text message of type '{message_type}' from {from_number}. Ignoring for now.")
                                return {"status": "ignored", "reason": f"non-text message type: {message_type}"}
                        else:
                            print("INFO: No messages or contacts found in webhook payload.")
                            return {"status": "ignored", "reason": "no messages or contacts"}
        print("INFO: Webhook payload not a recognized message event.")
        return {"status": "ignored", "reason": "unrecognized payload structure"}

    except Exception as e:
        print(f"ERROR: Failed to process WhatsApp webhook payload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {e}")