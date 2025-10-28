from fastapi import APIRouter, HTTPException, Request, Query, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb
from llm.config import SETTINGS
from llm.lang import detect_language
from llm import tags_index
from llm.chat_history import save_message, get_recent_history, prune_history, build_context_string

from llm.intent import detect_opening_hours_intent
from llm.opening_hours import compute_opening_answer

import httpx
import json
import re
import time
import sys
import traceback

# --- Llama client helper (should be moved to llm/llama_client.py) ---
def call_llama(prompt: str, max_tokens: int = 60, temperature: float = 0.0, stop: list = None) -> str:
    import boto3
    import json
    from llm.config import SETTINGS
    bedrock = boto3.client("bedrock-runtime", region_name=SETTINGS.aws_region)
    model_arn = SETTINGS.kb_model_arn
    body = {
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
    }
    if stop:
        body["stop_sequences"] = stop
    resp = bedrock.invoke_model(
        modelId=model_arn,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = resp["body"].read().decode("utf-8")
    try:
        out = json.loads(result)
        return out.get("generation", "").strip() or out.get("output", "").strip() or out.get("text", "").strip()
    except Exception:
        return result.strip()

def call_llm_rephrase(history_context: str, lang: str) -> str:
    prompts = {
        "en": (
            "Given the following conversation, rewrite the user's latest message as a self-contained, explicit question for the bot. "
            "If the last message is already explicit, return it unchanged.\n"
            "IMPORTANT: If you are unsure, if the question is ambiguous, or if you cannot confidently reformulate, DO NOT GUESS. Reply with [NO_CONTEXT] and do not send an answer.\n"
            "Conversation so far:\n"
            f"{history_context}\n"
            "Rewritten latest user message:"
        ),
        "zh-HK": (
            "請根據下列對話，把家長的最後一句重寫成完整、明確的自足問題（如「請問英語語文課學費是多少？」）。如果已經是完整問題，則原文返回。\n"
            "重要：如果你不能明確判斷問題內容、覺得資訊不足或不確定，請直接回覆[NO_CONTEXT]，不要嘗試猜測。\n"
            "對話如下：\n"
            f"{history_context}\n"
            "重寫後的家長問題："
        ),
        "zh-CN": (
            "请根据下列对话，把家长的最后一句重写成完整、明确的自足问题（如“请问英语语言艺术课学费是多少？”）。如果已经是完整问题，则原文返回。\n"
            "重要：如果你不能明确判断问题内容、觉得信息不足或不确定，请直接回复[NO_CONTEXT]，不要尝试猜测。\n"
            "对话如下：\n"
            f"{history_context}\n"
            "重写后的家长问题："
        ),
    }
    prompt = prompts.get(lang, prompts["en"])
    return call_llama(prompt, max_tokens=60, temperature=0.0).strip()

def _log(msg):
    print(f"[LLM ROUTER] {msg}", file=sys.stderr, flush=True)

# --- WhatsApp helpers (should be moved to llm/whatsapp_utils.py) ---
async def _send_whatsapp_message(to: str, message_body: str):
    if not SETTINGS.whatsapp_access_token or not SETTINGS.whatsapp_phone_number_id:
        _log("ERROR: WhatsApp API credentials (access token or phone number ID) not configured. Cannot send message.")
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
        "text": {"body": message_body}
    }
    _log(f"[WA] Sending WhatsApp message to: {to} | Body: {message_body}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            _log(f"[WA] SUCCESS: WhatsApp message sent to {to}. Response: {response.json()}")
        except Exception as e:
            _log(f"[WA] ERROR: Failed to send WhatsApp message: {e}")

async def _send_whatsapp_document(to: str, doc_url: str, filename: str = "document.pdf"):
    if not SETTINGS.whatsapp_access_token or not SETTINGS.whatsapp_phone_number_id:
        _log("ERROR: WhatsApp API credentials (access token or phone number ID) not configured. Cannot send document.")
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
        "document": {"link": doc_url, "filename": filename}
    }
    _log(f"[WA] Sending WhatsApp document to: {to} | Document URL: {doc_url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            _log(f"[WA] SUCCESS: WhatsApp document sent to {to}. Response: {response.json()}")
        except Exception as e:
            _log(f"[WA] ERROR: Failed to send WhatsApp document: {e}")

# --- Answer marker and guardrail helpers (should be moved to llm/answer_utils.py) ---
def extract_and_strip_marker(answer: str, marker: str) -> (str, bool):
    pattern = re.compile(rf"\s*{re.escape(marker)}\s*$", re.IGNORECASE)
    if answer and pattern.search(answer):
        answer = pattern.sub("", answer)
        return answer.rstrip(), True
    return answer, False

def _any_doc_cited(citations, doc_paths: list) -> bool:
    if not citations:
        return False
    for c in citations:
        uri = c.get("uri", "") or ""
        for doc in doc_paths:
            if uri and uri.endswith(doc):
                return True
    return False

def _answer_has_strong_phrase(answer: str, lang: str, phrases_by_lang: dict) -> bool:
    answer_lc = (answer or "").lower()
    phrases = phrases_by_lang.get(lang, []) + phrases_by_lang.get("en", [])
    return any(phrase.lower() in answer_lc for phrase in phrases)

def _answer_is_short(answer: str, max_words: int = 40) -> bool:
    words = (answer or "").split()
    return len(words) <= max_words

def is_followup_message(msg: str) -> bool:
    if not msg:
        return False
    msg_lc = msg.strip().lower()
    # 1. Classic follow-up/elliptical patterns (start of string)
    FOLLOWUP_PATTERNS = [
        r"^\s*(what about|how about|and|which ones|tell me more|like what|go on|what else|for example|can you elaborate|can you explain|例如|舉個例|可以再說說|举个例|还有呢|继续|再多一些|再讲讲|再說說)\b",
        r"^[\s\?]*$",  # just "?" or empty
    ]
    if len(msg_lc.split()) <= 7:
        for pat in FOLLOWUP_PATTERNS:
            if re.match(pat, msg_lc):
                return True
    # 2. Short, vague, ends with question mark and not explicit
    if len(msg_lc.split()) <= 3 and msg_lc.endswith("?"):
        if not re.match(r"^\s*(what|how|when|where|who|which|why)\b", msg_lc):
            return True
    # 3. Referential pronouns + query word
    PRONOUNS = r"\b(that|it|this|those|these)\b"
    QUERY_KEYWORDS = r"\b(tuition|fee|cost|price|schedule|age|time|when|how much|class|course|program|subject|writing|math|english|chinese|mandarin|lesson|session)\b"
    if re.search(PRONOUNS, msg_lc) and re.search(QUERY_KEYWORDS, msg_lc):
        return True
    # 4. Very short, referential commands
    if msg_lc in {"that", "this", "it", "those", "these"}:
        return True
    return False

NOINFO_PHRASES = [
    "not explicitly stated", "not specified", "not mentioned", "no information", "not provided", "no details",
    "not found", "please contact", "refer to tuition listing", "contact our staff", "contact us",
    "up-to-date fees", "available time slots", "no answer available", "unable to find", "unable to provide",
    "no details available", "please refer to", "no specific information", "the search results do not specify", "no start date mentioned"
]
def contains_apology_or_noinfo(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in NOINFO_PHRASES)

def likely_contains_fee(answer: str) -> bool:
    return bool(re.search(r"(hk\$|\$)\s*\d+", answer, re.I))

# --- Enrollment/Blooket markers ---
ENROLLMENT_FORM_URL = "https://drive.google.com/uc?export=download&id=1YTsUsTdf-k8ky-nJIFSZ7LtzzQ7BuzyA"
ENROLLMENT_FORM_MARKER = "[SEND_ENROLLMENT_FORM]"
ENROLLMENT_FORM_DOCS = [
    "/en/policies/enrollment_form.md",
    "/zh-HK/policies/enrollment_form.md",
    "/zh-CN/policies/enrollment_form.md",
]
BLOOKET_PDF_URL = "https://drive.google.com/uc?export=download&id=18Ti5H8EoR7rmzzk4KGMGdQZFuqQ4uY4M"
BLOOKET_MARKER = "[SEND_BLOOKET_PDF]"
BLOOKET_DOCS = [
    "/en/policies/blooket_instructions.md",
    "/zh-HK/policies/blooket_instructions.md",
    "/zh-CN/policies/blooket_instructions.md",
]
ENROLLMENT_STRONG_PHRASES = {
    "en": ["enrollment form", "registration form", "application form", "please fill out our enrollment form", "download the form", "attached form", "fill in the form", "submit the enrollment form", "send me the form"],
    "zh-HK": ["入學表格", "報名表格", "申請表", "下載表格", "填好表格", "發送表格", "填寫入學表格"],
    "zh-CN": ["入学表格", "报名表格", "申请表", "下载表格", "填好表格", "发送表格", "填写入学表格"]
}
BLOOKET_STRONG_PHRASES = {
    "en": ["blooket", "blooket instructions", "online game", "how to play blooket", "the blooket pdf", "blooket guide"],
    "zh-HK": ["blooket", "網上遊戲", "布魯克特", "blooket 指引", "blooket 教學", "blooket pdf"],
    "zh-CN": ["blooket", "在线游戏", "布鲁克特", "blooket 指南", "blooket 教程", "blooket pdf"]
}

# --- PATCH: Structured context injection for opening hours (RAG) ---
def extract_opening_context(message: str, lang: Optional[str] = None) -> str:
    """Create system-injected context for opening hours/date/weather queries."""
    from llm.opening_hours import _normalize_lang, _parse_datetime, _dow_window, _is_public_holiday, HK_TZ, _fmt_time, _fmt_date_human
    from llm.hko import get_weather_hint_for_opening
    import datetime as dt
    L = _normalize_lang(lang)
    now = dt.datetime.now(HK_TZ)
    dtval = _parse_datetime(message or "", now, L) or now
    open_t, close_t = _dow_window(dtval.weekday())
    is_holiday, holiday_name = _is_public_holiday(dtval)
    is_sunday = open_t is None or close_t is None
    weather_hint = get_weather_hint_for_opening(L)
    parts = []
    parts.append(f"Resolved date: {dtval.strftime('%Y-%m-%d')} ({_fmt_date_human(dtval, L)})")
    if is_holiday:
        parts.append(f"Public holiday: Yes ({holiday_name})")
    if is_sunday:
        parts.append("Day: Sunday (center closed)")
    if open_t and close_t and not is_holiday and not is_sunday:
        parts.append(f"Open hours: {_fmt_time(open_t)}–{_fmt_time(close_t)}")
    if weather_hint:
        parts.append(f"Weather: {weather_hint}")
    return "\n".join(parts)

# --- FastAPI schemas ---
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None
    session_id: Optional[str] = None
    debug: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    debug: Optional[Dict[str, Any]] = None

router = APIRouter(tags=["LLM Chat (Bedrock KB)"])

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _log(f"/chat called: message={req.message!r}, language={req.language!r}, session_id={req.session_id!r}, debug={req.debug!r}")
    _log(f"Headers: {dict(request.headers)}")
    session_id = req.session_id or ("web:" + str(hash(request.client.host)))
    lang = req.language or detect_language(req.message, accept_language=request.headers.get("accept-language"))
    _log(f"Detected language: {lang!r}")

    # --- PATCH: Opening hours intent routing ---
    is_hours_intent, debug_intent = detect_opening_hours_intent(req.message, lang)
    opening_context = None
    if is_hours_intent:
        opening_context = extract_opening_context(req.message, lang)
        _log(f"Opening hours intent detected. Structured context for LLM:\n{opening_context}")

    use_history = (req.session_id is not None and not req.session_id.startswith("web:"))
    if use_history:
        try:
            history = get_recent_history(session_id, limit=6)
            _log(f"Fetched {len(history)} prior messages for session_id={session_id}")
        except Exception as e:
            _log(f"ERROR retrieving DynamoDB history: {e}\n{traceback.format_exc()}")
            history = []
    else:
        history = []
        _log("Skipping DynamoDB chat history for this session/request.")

    history_context = build_context_string(history, new_message=req.message, user_role="user", bot_role="bot", include_new=True)

    rag_query = req.message
    if is_followup_message(req.message):
        try:
            rag_query = call_llm_rephrase(history_context, lang)
            _log(f"Reformulated query: {rag_query!r}")
        except Exception as e:
            _log(f"Failed to reformulate query, falling back to user message. Error: {e}")
            rag_query = req.message

    _log(f"Calling chat_with_kb with rag_query length={len(rag_query)}")
    try:
        answer, citations, debug_info = chat_with_kb(
            rag_query,
            lang,
            session_id,
            debug=bool(req.debug),
            extra_context=opening_context if is_hours_intent else None,
            hint_canonical="opening_hours" if is_hours_intent else None,
        )
    except Exception as e:
        _log(f"ERROR during chat_with_kb: {e}\n{traceback.format_exc()}")
        # Only fallback to deterministic for opening-hours if LLM/RAG fails:
        if is_hours_intent:
            answer = compute_opening_answer(req.message, lang)
            citations = []
            debug_info = {"source": "deterministic_opening_hours_fallback", "intent_debug": debug_intent}
            return ChatResponse(answer=answer, citations=citations, debug=debug_info)
        raise HTTPException(status_code=500, detail=f"LLM backend error: {e}")

    _log(f"LLM raw answer: {answer!r}")
    _log(f"LLM citations: {json.dumps(citations, indent=2)}")
    if debug_info:
        _log(f"LLM debug_info: {json.dumps(debug_info, indent=2)}")

    if not citations or contains_apology_or_noinfo(answer):
        _log("No citations found, or answer is a hedged/noinfo/apology. Silencing output.")
        # Fallback to deterministic answer for opening-hours queries ONLY:
        if is_hours_intent:
            answer = compute_opening_answer(req.message, lang)
            citations = []
            debug_info = {"source": "deterministic_opening_hours_fallback", "intent_debug": debug_intent}
        else:
            answer = ""
    fee_words = ["tuition", "fee", "price", "cost"]
    payment_words = ["how to pay", "payment", "pay", "bank transfer", "fps", "account", "method"]
    if (
        any(word in rag_query.lower() for word in fee_words)
        and not any(word in rag_query.lower() for word in payment_words)
        and not likely_contains_fee(answer)
    ):
        _log("Answer does not contain a fee amount for a tuition/fee query, silencing.")
        answer = ""

    if use_history:
        try:
            now = time.time()
            save_message(session_id, "user", req.message, lang, now)
            save_message(session_id, "bot", answer or "", lang, now + 0.01)
            prune_history(session_id, keep=6)
            _log(f"Saved and pruned DynamoDB history for session_id={session_id}")
        except Exception as e:
            _log(f"ERROR saving/pruning DynamoDB history: {e}\n{traceback.format_exc()}")

    answer, marker = extract_and_strip_marker(answer, ENROLLMENT_FORM_MARKER)
    send_form = marker or (
        _any_doc_cited(citations, ENROLLMENT_FORM_DOCS) and
        _answer_has_strong_phrase(answer, lang, ENROLLMENT_STRONG_PHRASES) and
        _answer_is_short(answer)
    )
    answer, marker_blooket = extract_and_strip_marker(answer, BLOOKET_MARKER)
    send_blooket = marker_blooket or (
        _any_doc_cited(citations, BLOOKET_DOCS) and
        _answer_has_strong_phrase(answer, lang, BLOOKET_STRONG_PHRASES) and
        _answer_is_short(answer)
    )

    silent_reason = debug_info.get("silence_reason") if isinstance(debug_info, dict) else None
    is_followup = is_followup_message(req.message)
    if not answer and silent_reason == "no_citations" and is_followup and history:
        _log("Allowing context-only answer for followup message due to chat history, but LLM did not produce anything.")
        answer = ""  # Do not send any placeholder/apology.

    if send_form:
        answer = (answer or "") + f"\n\nYou can download our enrollment form [here]({ENROLLMENT_FORM_URL})."
        _log("Enrollment form marker/trigger detected: added PDF link to response.")
    if send_blooket:
        answer = (answer or "") + f"\n\nYou can download the Blooket instructions [here]({BLOOKET_PDF_URL})."
        _log("Blooket marker/trigger detected: added Blooket PDF link to response.")

    if not answer and silent_reason:
        _log(f"LLM silenced answer. Reason: {silent_reason}")

    answer = answer or ""
    _log(f"Final answer length={len(answer)}. Returning response.")
    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

# WhatsApp handler uses the same guardrails as above.
@router.post("/whatsapp_webhook")
async def whatsapp_webhook_handler(request: Request):
    try:
        payload = await request.json()
        _log(f"Received WhatsApp webhook payload:\n{json.dumps(payload, indent=2)}")

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
                            _log(f"Message details: from={from_number}, type={message_type}, contact={contact}")

                            if message_type == "text":
                                message_body = message["text"].get("body")
                                _log(f"Text message from {from_number} (Name: {contact.get('profile',{}).get('name')}): '{message_body}'")

                                if from_number not in SETTINGS.whatsapp_test_numbers:
                                    _log(f"WARNING: Message from non-whitelisted number {from_number} ignored during testing.")
                                    return {"status": "ignored", "reason": "not in test numbers"}

                                lang = detect_language(message_body)
                                _log(f"Detected language: {lang}")

                                # --- PATCH: Opening hours intent routing ---
                                is_hours_intent, debug_intent = detect_opening_hours_intent(message_body, lang)
                                opening_context = None
                                if is_hours_intent:
                                    opening_context = extract_opening_context(message_body, lang)
                                    _log(f"Opening hours intent detected. Structured context for LLM:\n{opening_context}")

                                try:
                                    history = get_recent_history(from_number, limit=6)
                                    _log(f"Fetched {len(history)} prior messages for session_id={from_number}")
                                except Exception as e:
                                    _log(f"ERROR retrieving DynamoDB history: {e}\n{traceback.format_exc()}")
                                    history = []

                                history_context = build_context_string(history, new_message=message_body, user_role="user", bot_role="bot", include_new=True)

                                rag_query = message_body
                                if is_followup_message(message_body):
                                    try:
                                        rag_query = call_llm_rephrase(history_context, lang)
                                        _log(f"Reformulated query: {rag_query!r}")
                                    except Exception as e:
                                        _log(f"Failed to reformulate query, falling back to user message. Error: {e}")
                                        rag_query = message_body

                                _log(f"Calling chat_with_kb with rag_query length={len(rag_query)}")
                                try:
                                    answer, citations, debug_info = chat_with_kb(
                                        rag_query,
                                        lang,
                                        session_id=from_number,
                                        debug=True,
                                        extra_context=opening_context if is_hours_intent else None,
                                        hint_canonical="opening_hours" if is_hours_intent else None,
                                    )
                                except Exception as e:
                                    _log(f"ERROR during chat_with_kb: {e}\n{traceback.format_exc()}")
                                    # Fallback to deterministic answer for opening-hours only:
                                    if is_hours_intent:
                                        answer = compute_opening_answer(message_body, lang)
                                        citations = []
                                        debug_info = {"source": "deterministic_opening_hours_fallback", "intent_debug": debug_intent}
                                        await _send_whatsapp_message(from_number, answer)
                                        return {"status": "ok", "message": "Sent deterministic opening hours answer"}
                                    raise HTTPException(status_code=500, detail=f"LLM backend error: {e}")

                                if not citations or contains_apology_or_noinfo(answer):
                                    _log("No citations found, or answer is a hedged/noinfo/apology. Silencing output.")
                                    # Fallback to deterministic answer for opening-hours only:
                                    if is_hours_intent:
                                        answer = compute_opening_answer(message_body, lang)
                                        citations = []
                                        debug_info = {"source": "deterministic_opening_hours_fallback", "intent_debug": debug_intent}
                                    else:
                                        answer = ""
                                fee_words = ["tuition", "fee", "price", "cost"]
                                payment_words = ["how to pay", "payment", "pay", "bank transfer", "fps", "account", "method"]
                                if (
                                    any(word in rag_query.lower() for word in fee_words)
                                    and not any(word in rag_query.lower() for word in payment_words)
                                    and not likely_contains_fee(answer)
                                ):
                                    _log("Answer does not contain a fee amount for a tuition/fee query, silencing.")
                                    answer = ""

                                try:
                                    now = time.time()
                                    save_message(from_number, "user", message_body, lang, now)
                                    save_message(from_number, "bot", answer or "", lang, now + 0.01)
                                    prune_history(from_number, keep=6)
                                    _log(f"Saved and pruned DynamoDB history for session_id={from_number}")
                                except Exception as e:
                                    _log(f"ERROR saving/pruning DynamoDB history: {e}\n{traceback.format_exc()}")

                                _log(f"LLM raw answer: {answer!r}")
                                _log(f"LLM citations: {json.dumps(citations, indent=2)}")
                                if debug_info:
                                    _log(f"LLM debug_info: {json.dumps(debug_info, indent=2)}")
                                silent_reason = debug_info.get("silence_reason") if isinstance(debug_info, dict) else None
                                is_followup = is_followup_message(message_body)
                                if not answer and silent_reason == "no_citations" and is_followup and history:
                                    _log("Allowing context-only answer for followup message due to chat history, but LLM did not produce anything.")
                                    answer = ""  # Do not send any placeholder/apology.
                                if not answer and silent_reason:
                                    _log(f"LLM silenced answer. Reason: {silent_reason}")

                                answer, marker = extract_and_strip_marker(answer, ENROLLMENT_FORM_MARKER)
                                send_form = marker or (
                                    _any_doc_cited(citations, ENROLLMENT_FORM_DOCS)
                                    and _answer_has_strong_phrase(answer, lang, ENROLLMENT_STRONG_PHRASES)
                                    and _answer_is_short(answer)
                                )

                                answer, marker_blooket = extract_and_strip_marker(answer, BLOOKET_MARKER)
                                send_blooket = marker_blooket or (
                                    _any_doc_cited(citations, BLOOKET_DOCS)
                                    and _answer_has_strong_phrase(answer, lang, BLOOKET_STRONG_PHRASES)
                                    and _answer_is_short(answer)
                                )

                                sent = False
                                if send_form:
                                    _log("Enrollment form marker/trigger detected: sending PDF document.")
                                    await _send_whatsapp_document(from_number, ENROLLMENT_FORM_URL, "enrollment_form.pdf")
                                    sent = True
                                if send_blooket:
                                    _log("Blooket marker/trigger detected: sending Blooket instruction PDF.")
                                    await _send_whatsapp_document(from_number, BLOOKET_PDF_URL, "blooket_instructions.pdf")
                                    sent = True

                                if sent and answer:
                                    _log("Sending answer after document.")
                                    await _send_whatsapp_message(from_number, answer)
                                elif answer:
                                    _log(f"LLM Answer: '{answer}'")
                                    await _send_whatsapp_message(from_number, answer)
                                else:
                                    _log("LLM provided no answer. No WhatsApp reply sent.")
                                return {"status": "ok", "message": "Message processed"}
                            else:
                                _log(f"INFO: Received non-text message of type '{message_type}' from {from_number}. Ignoring.")
                                return {"status": "ignored", "reason": f"non-text message type: {message_type}"}
                        else:
                            _log("INFO: No messages or contacts found in webhook payload.")
                            return {"status": "ignored", "reason": "no messages or contacts"}
        _log("INFO: Webhook payload not a recognized message event.")
        return {"status": "ignored", "reason": "unrecognized payload structure"}
    except Exception as e:
        _log(f"ERROR: Failed to process WhatsApp webhook payload: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {e}")