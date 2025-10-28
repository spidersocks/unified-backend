from fastapi import APIRouter, Request, HTTPException, Response
from typing import List, Dict, Any, Optional
import httpx # For making HTTP requests to WhatsApp API
import os
import json # For pretty printing payload

# Import settings and your LLM chat function
from llm.config import SETTINGS
from llm.bedrock_kb_client import chat_with_kb # Assuming this is the core function you want to call

router = APIRouter(prefix="/whatsapp_webhook", tags=["WhatsApp Webhook"])

# --- Helper function to send WhatsApp messages ---
async def _send_whatsapp_message(to: str, message_body: str):
    """Sends a text message back to a WhatsApp user via the Cloud API."""
    if not SETTINGS.whatsapp_access_token or not SETTINGS.whatsapp_phone_number_id:
        print("ERROR: WhatsApp API credentials (access token or phone number ID) not configured. Cannot send message.")
        return

    # WhatsApp Cloud API endpoint
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
            response = await client.post(url, headers=headers, json=payload, timeout=30) # Added timeout
            response.raise_for_status() # Raise an exception for bad status codes
            print(f"SUCCESS: WhatsApp message sent to {to}. Response: {response.json()}")
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Failed to send WhatsApp message to {to}. Status: {e.response.status_code}, Detail: {e.response.text}")
        except httpx.RequestError as e:
            print(f"ERROR: An error occurred while requesting to send WhatsApp message to {to}: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected error in _send_whatsapp_message: {e}")


# --- WhatsApp Webhook Verification Endpoint (GET request) ---
@router.get("/")
async def whatsapp_webhook_verification(request: Request):
    """
    Handles WhatsApp webhook verification.
    Meta sends a GET request with specific query parameters.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    print(f"INFO: Webhook Verification Request received. Mode: {mode}, Token: {token[:5]}..., Challenge: {challenge}")

    if mode == "subscribe" and token == SETTINGS.whatsapp_verify_token:
        print("INFO: Webhook verification successful!")
        # Meta expects the challenge to be returned as an integer
        return Response(content=challenge, media_type="text/plain", status_code=200)
    else:
        print(f"ERROR: Webhook verification failed. Token mismatch or invalid mode. Expected token: {SETTINGS.whatsapp_verify_token}")
        raise HTTPException(status_code=403, detail="Verification token mismatch or invalid mode")

# --- WhatsApp Webhook Message Handling Endpoint (POST request) ---
@router.post("/")
async def whatsapp_webhook_handler(request: Request):
    """
    Handles incoming WhatsApp messages.
    """
    payload = await request.json()
    print(f"INFO: Received WhatsApp webhook payload:\n{json.dumps(payload, indent=2)}")

    # --- Parse the incoming message ---
    # This structure can be complex, so we'll try to safely extract the relevant parts.
    try:
        # Check if it's a message event
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
                            from_number = message.get("from") # Sender's phone number (WA_ID)
                            message_type = message.get("type")

                            if message_type == "text":
                                message_body = message["text"].get("body")
                                print(f"INFO: Message from {from_number} (Name: {contact.get('profile',{}).get('name')}): '{message_body}'")

                                # --- Whitelisting/Blacklisting for Testing ---
                                if from_number not in SETTINGS.whatsapp_test_numbers:
                                    print(f"WARNING: Message from non-whitelisted number {from_number} ignored during testing.")
                                    return {"status": "ignored", "reason": "not in test numbers"}

                                # --- Process message with your LLM ---
                                print(f"INFO: Calling LLM for message: '{message_body}'")
                                # You might want to pass session_id, language, debug based on your needs
                                # For simplicity, we'll use a placeholder for session_id and detect language
                                # The chat_with_kb function already handles language detection if not provided.
                                llm_answer, citations, debug_info = chat_with_kb(
                                    message_body,
                                    session_id=from_number, # Use phone number as session ID
                                    debug=False # Set to True if you want debug info in logs
                                )

                                if llm_answer:
                                    print(f"INFO: LLM Answer: '{llm_answer}'")
                                    await _send_whatsapp_message(from_number, llm_answer)
                                else:
                                    # Handle cases where LLM doesn't provide an answer (e.g., no context)
                                    fallback_message = "Sorry, I couldn't find an answer to that. Please try rephrasing your question or contact our staff."
                                    print(f"WARNING: LLM provided no answer. Sending fallback: '{fallback_message}'")
                                    await _send_whatsapp_message(from_number, fallback_message)
                                return {"status": "ok", "message": "Message processed"}
                            else:
                                print(f"INFO: Received non-text message of type '{message_type}' from {from_number}. Ignoring for now.")
                                # You can extend this to handle other message types (images, audio, etc.)
                                return {"status": "ignored", "reason": f"non-text message type: {message_type}"}
                        else:
                            print("INFO: No messages or contacts found in webhook payload.")
                            return {"status": "ignored", "reason": "no messages or contacts"}
        print("INFO: Webhook payload not a recognized message event.")
        return {"status": "ignored", "reason": "unrecognized payload structure"}

    except Exception as e:
        print(f"ERROR: Failed to process WhatsApp webhook payload: {e}")
        # Log the full payload for debugging if needed
        # print(f"Payload was: {json.dumps(payload, indent=2)}")
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {e}")