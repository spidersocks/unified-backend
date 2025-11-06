from fastapi import APIRouter, HTTPException, Request, Query, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb
from llm.config import SETTINGS
from llm.lang import get_language_code
from llm import tags_index
from llm.chat_history import save_message, get_recent_history, prune_history, build_context_string
from llm.intent import detect_opening_hours_intent, is_general_hours_query, classify_scheduling_context
from llm.opening_hours import compute_opening_answer, extract_opening_context, center_is_open_now, summarize_user_date_intent

import httpx
import json
import re
import time
import sys
import traceback
import asyncio
from collections import deque

def _cites_admin_routing(citations: List[Dict[str, Any]]) -> bool:
    if not citations:
        return False
    for c in citations:
        uri = (c.get("uri") or "")
        if "/faq/admin_scheduling_routing.md" in uri:
            return True
    return False

_LEAVE_EN = re.compile(
    r"\b(can'?t|cannot|won'?t)\s+(attend|come|make\s+(?:it|the\s+class|the\s+lesson))\b"
    r"|won'?t\s+be\s+able\s+to\s+attend\b"
    r"|will\s+be\s+away\b",
    re.IGNORECASE
)

def _looks_like_leave_notification(text: str) -> bool:
    return bool(_LEAVE_EN.search(text or ""))

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
# Track message IDs our bot sent via Cloud API to distinguish from admin-sent
_BOT_MSG_IDS: Dict[str, float] = {}  # msg_id -> ts
# Track last detected admin activity per recipient (phone/session)
_LAST_ADMIN_ACTIVITY: Dict[str, float] = {}  # recipient_id -> ts
# Keep structure small by pruning old bot IDs
def _record_bot_msg_id(msg_id: Optional[str]):
    if not msg_id:
        return
    _BOT_MSG_IDS[msg_id] = time.time()
    # Simple prune: drop entries older than 24h or if over 500 entries
    if len(_BOT_MSG_IDS) > 500:
        cutoff = time.time() - 24 * 3600
        to_del = [mid for mid, ts in _BOT_MSG_IDS.items() if ts < cutoff]
        for mid in to_del:
            _BOT_MSG_IDS.pop(mid, None)

def _mark_admin_activity(recipient_id: Optional[str]):
    if not recipient_id:
        return
    _LAST_ADMIN_ACTIVITY[recipient_id] = time.time()
    # Cancel any pending ack for this chat
    _cancel_pending_ack(recipient_id)
    _log(f"[COOL] Marked admin activity for {recipient_id}. Cooling for {SETTINGS.admin_cooldown_secs}s")

def _in_admin_cooldown(session_id: str) -> bool:
    ts = _LAST_ADMIN_ACTIVITY.get(session_id)
    if not ts:
        return False
    return (time.time() - ts) < SETTINGS.admin_cooldown_secs

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
            data = response.json()
            _log(f"[WA] SUCCESS: WhatsApp message sent to {to}. Response: {data}")
            try:
                # Record bot message id so status webhooks for this id are recognized as bot
                msg_id = (data.get("messages") or [{}])[0].get("id")
                _record_bot_msg_id(msg_id)
            except Exception:
                pass
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
            data = response.json()
            _log(f"[WA] SUCCESS: WhatsApp document sent to {to}. Response: {data}")
            try:
                msg_id = (data.get("messages") or [{}])[0].get("id")
                _record_bot_msg_id(msg_id)
            except Exception:
                pass
        except Exception as e:
            _log(f"[WA] ERROR: Failed to send WhatsApp document: {e}")

_ACK_TASKS: Dict[str, asyncio.Task] = {}  # session_id/phone -> task

def _ack_text(lang: str, during_hours: bool) -> str:
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return "多謝你的訊息。我們的同事會盡快聯絡你。" if during_hours else "多謝你的訊息。我們的同事會喺下一個辦公時間盡快聯絡你。"
    if L.startswith("zh-cn") or L == "zh":
        return "感谢您的留言。我们的同事会尽快联系您。" if during_hours else "感谢您的留言。我们的同事会在下一个办公时间尽快联系您。"
    return "Thank you for your message. Our staff will contact you ASAP." if during_hours else "Thank you for your message. Our staff will contact you ASAP during next working hours."

def _cancel_pending_ack(session_id: str):
    task = _ACK_TASKS.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        _log(f"[ACK] Cancelled pending ack for session_id={session_id}")

async def _ack_worker(session_id: str, lang: str, base_ts: float, delay_secs: int):
    try:
        if delay_secs > 0:
            await asyncio.sleep(delay_secs)
        # Suppress ack if admin cooldown engaged meanwhile
        if _in_admin_cooldown(session_id):
            _log(f"[ACK] Suppress ack due to admin cooldown for {session_id}")
            return
        # Before sending, ensure no newer user messages have arrived and the bot hasn't sent a non-empty reply
        history = get_recent_history(session_id, limit=10, oldest_first=True)
        newer_user = any(item["role"] == "user" and float(item["ts"]) > base_ts for item in history)
        newer_bot_nonempty = any(item["role"] == "bot" and float(item["ts"]) > base_ts and (item.get("message") or "").strip() for item in history)
        if newer_user or newer_bot_nonempty:
            _log(f"[ACK] Skip ack (newer_user={newer_user}, newer_bot_nonempty={newer_bot_nonempty}) for {session_id}")
            return
        during_hours = center_is_open_now(lang)
        msg = _ack_text(lang, during_hours=during_hours)
        await _send_whatsapp_message(session_id, msg)
        _log(f"[ACK] Sent auto-ack to {session_id}")
    except asyncio.CancelledError:
        _log(f"[ACK] Ack task cancelled for session_id={session_id}")
    except Exception as e:
        _log(f"[ACK] Error sending ack: {e}")
    finally:
        if _ACK_TASKS.get(session_id):
            _ACK_TASKS.pop(session_id, None)

def _maybe_schedule_auto_ack_whatsapp(session_id: str, lang: str, base_ts: float):
    """
    Schedule a WhatsApp auto-ack if the bot did not answer (answer was silenced/empty).
    - During working hours (open): wait 30 minutes (1800s).
    - Outside working hours: send immediately (0 delay).
    """
    # Never schedule an ack if admin cooling is active
    if _in_admin_cooldown(session_id):
        _log(f"[ACK] Not scheduling ack due to admin cooldown for {session_id}")
        return

    during_hours = center_is_open_now(lang)
    delay_secs = SETTINGS.whatsapp_ack_delay_secs if during_hours else 0

    _cancel_pending_ack(session_id)
    task = asyncio.create_task(_ack_worker(session_id, lang, base_ts, delay_secs))
    _ACK_TASKS[session_id] = task
    _log(f"[ACK] Scheduled auto-ack for {session_id} in {delay_secs}s (during_hours={during_hours})")


# --- Answer marker and guardrail helpers (should be moved to llm/answer_utils.py) ---
def extract_and_strip_marker(answer: str, marker: str) -> (str, bool):
    pattern = re.compile(rf"\s*{re.escape(marker)}\s*", re.IGNORECASE)
    if answer and pattern.search(answer):
        cleaned_answer = pattern.sub("", answer).strip()
        return cleaned_answer, True
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
    FOLLOWUP_PATTERNS = [
        r"^\s*(what about|how about|and|which ones|tell me more|like what|go on|what else|for example|can you elaborate|can you explain|例如|舉個例|可以再說說|举个例|还有呢|继续|再多一些|再讲讲|再說說)\b",
        r"^[\s\?]*$",
    ]
    if len(msg_lc.split()) <= 7:
        for pat in FOLLOWUP_PATTERNS:
            if re.match(pat, msg_lc):
                return True
    if len(msg_lc.split()) <= 3 and msg_lc.endswith("?"):
        if not re.match(r"^\s*(what|how|when|where|who|which|why)\b", msg_lc):
            return True
    PRONOUNS = r"\b(that|it|this|those|these)\b"
    QUERY_KEYWORDS = r"\b(tuition|fee|cost|price|schedule|age|time|when|how much|class|course|program|subject|writing|math|english|chinese|mandarin|lesson|session)\b"
    if re.search(PRONOUNS, msg_lc) and re.search(QUERY_KEYWORDS, msg_lc):
        return True
    if msg_lc in {"that", "this", "it", "those", "these"}:
        return True
    return False

# NEW: Expanded with Chinese variants (zh-HK and zh-CN)
NOINFO_PHRASES = [
    # English
    "not explicitly stated", "not specified", "not mentioned", "no information", "not provided", "no details",
    "not found", "refer to tuition listing",
    "up-to-date fees", "available time slots", "no answer available", "unable to find", "unable to provide",
    "no details available", "please refer to", "no specific information", "the search results do not specify",
    "no start date mentioned", "we don't have specific information", "we do not have specific information",
    "we don't have details", "we do not have details", "no info on", "no details on", "information not available",
    "details not available", "no availability information",
    # Traditional Chinese (zh-HK)
    "未有具體資料", "未有資料", "未提供資料", "未能提供", "無法提供", "未有相關資料", "未有相關資訊",
    "暫無資料", "暫時沒有資料", "未有說明", "資料未有說明", "文件未有說明", "未有註明", "未有列出",
    "文件未有提及", "未有提及", "沒有提及", "找不到", "查不到", "未找到", "沒有記錄", "資料不詳",
    "資訊不足", "資料不足", "請參閱", "沒有具體資訊", "暫無相關資料", "暫無相關資訊",
    # Simplified Chinese (zh-CN)
    "没有具体信息", "没有资料", "未提供信息", "未能提供", "无法提供", "没有相关信息", "无相关信息",
    "暂无信息", "暂时没有信息", "未说明", "资料未说明", "文档未说明", "未注明", "未列出",
    "文档未提及", "未提及", "没有提及", "找不到", "查不到", "未找到", "没有记录", "信息不详",
    "信息不足", "资料不足", "请参阅", "没有具体资讯", "暂无相关信息", "暂无相关资料",
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
    """Unified chat endpoint for both web and WhatsApp routing (RAG + rules)."""
    _log(f"/chat called: message={req.message!r}, language={req.language!r}, session_id={req.session_id!r}, debug={req.debug!r}")
    _log(f"Headers: {dict(request.headers)}")

    # --- Session and language setup ---
    session_id = req.session_id or ("web:" + str(hash(request.client.host)))
    lang = req.language or get_language_code(req.message, accept_language_header=request.headers.get("accept-language"))
    _log(f"Detected language: {lang!r}")

    # --- Scheduling / Opening-hours detection ---
    sched_cls = classify_scheduling_context(req.message, lang)
    is_scheduling_action = bool(
        (sched_cls.get("has_sched_verbs")
         or sched_cls.get("availability_request")
         or sched_cls.get("admin_action_request")
         or sched_cls.get("staff_contact_request")
         or sched_cls.get("individual_homework_request"))
        and not sched_cls.get("has_policy_intent")
    )

    is_hours_intent = False
    opening_context = None
    hint_canonical = None

    if is_scheduling_action:
        opening_context = summarize_user_date_intent(req.message, lang)
        _log(f"Scheduling/action detected. Providing date hints only:\n{opening_context}")
    else:
        is_hours_intent, debug_intent = detect_opening_hours_intent(req.message, lang)
        if is_hours_intent:
            has_holiday_marker = bool((debug_intent or {}).get("holiday_hits"))
            if has_holiday_marker or not is_general_hours_query(req.message, lang):
                opening_context = extract_opening_context(req.message, lang)
                _log(f"Opening hours intent detected as SPECIFIC. Context:\n{opening_context}")
            hint_canonical = "opening_hours"
        else:
            _log("No specific opening hours intent detected.")

    # --- Chat history handling ---
    use_history = req.session_id is not None and not req.session_id.startswith("web:")
    history = []
    if use_history:
        try:
            history = get_recent_history(session_id, limit=6)
            _log(f"Fetched {len(history)} prior messages for session_id={session_id}")
        except Exception as e:
            _log(f"ERROR retrieving chat history: {e}\n{traceback.format_exc()}")
    else:
        _log("Skipping chat history for this request.")

    history_context = build_context_string(
        history, new_message=req.message, user_role="user", bot_role="bot", include_new=True
    )

    # --- Reformulate query if needed ---
    rag_query = req.message
    if sched_cls.get("has_policy_intent"):
        extra_keywords = ["policy", "absence", "make-up", "makeup", "quota", "notice", "doctor’s certificate"]
    else:
        extra_keywords = None

    if is_followup_message(req.message):
        try:
            new_query = call_llm_rephrase(history_context, lang)
            rag_query = new_query if new_query != "[NO_CONTEXT]" else req.message
            _log(f"Reformulated query: {rag_query!r}")
        except Exception as e:
            _log(f"Reformulation failed: {e}")

    # --- Call main LLM through Bedrock ---
    try:
        answer, citations, debug_info = chat_with_kb(
            rag_query,
            lang,
            session_id=session_id,
            debug=SETTINGS.debug_kb,
            extra_context=opening_context,
            hint_canonical=hint_canonical,
        )
    except Exception as e:
        _log(f"ERROR during chat_with_kb: {e}\n{traceback.format_exc()}")
        if is_hours_intent:
            return ChatResponse(
                answer=compute_opening_answer(req.message, lang),
                citations=[],
                debug={"source": "deterministic_opening_hours_fallback"},
            )
        raise HTTPException(status_code=500, detail=f"LLM backend error: {e}")

    _log(f"LLM raw answer: {answer!r}")
    _log(f"LLM citations: {json.dumps(citations, ensure_ascii=False, indent=2)}")
    if debug_info:
        _log(f"LLM debug_info: {json.dumps(debug_info, ensure_ascii=False, indent=2)}")

    # --- Guardrails / Answer suppression ---
    if not citations or contains_apology_or_noinfo(answer):
        _log("No citations found or noinfo phrase. Silencing output.")
        block_hours_fallback = any([
            sched_cls.get("has_sched_verbs"),
            sched_cls.get("availability_request"),
            sched_cls.get("admin_action_request"),
            sched_cls.get("staff_contact_request"),
            sched_cls.get("individual_homework_request"),
            sched_cls.get("placement_question"),
            _cites_admin_routing(citations),
            _looks_like_leave_notification(rag_query),
        ])

        if is_hours_intent and not block_hours_fallback:
            answer = compute_opening_answer(req.message, lang)
            citations = []
            debug_info = {"source": "deterministic_opening_hours_fallback"}
        else:
            answer = ""

    # Fee-related silence test
    fee_words = ["tuition", "fee", "price", "cost"]
    payment_words = ["how to pay", "payment", "pay", "bank transfer", "fps", "account", "method"]
    if (
        any(w in rag_query.lower() for w in fee_words)
        and not any(w in rag_query.lower() for w in payment_words)
        and not likely_contains_fee(answer)
    ):
        _log("Answer does not contain a fee amount for a tuition/fee query. Silencing.")
        answer = ""

    # --- Save updated chat history ---
    if use_history:
        try:
            now = time.time()
            save_message(session_id, "user", req.message, lang, now)
            save_message(session_id, "bot", answer or "", lang, now + 0.01)
            prune_history(session_id, keep=6)
            _log(f"Saved/pruned DynamoDB history for session_id={session_id}")
        except Exception as e:
            _log(f"ERROR saving chat history: {e}\n{traceback.format_exc()}")

    # --- Enrollment / Blooket document detection ---
    answer, marker_enroll = extract_and_strip_marker(answer, ENROLLMENT_FORM_MARKER)
    answer, marker_blooket = extract_and_strip_marker(answer, BLOOKET_MARKER)
    send_enrollment = marker_enroll or (
        _any_doc_cited(citations, ENROLLMENT_FORM_DOCS)
        and _answer_has_strong_phrase(answer, lang, {
            "en": ["enrollment form", "registration form", "application form"],
            "zh-HK": ["入學表格", "報名表格"],
            "zh-CN": ["入学表格", "报名表格"],
        })
        and _answer_is_short(answer)
    )
    send_blooket = marker_blooket or (
        _any_doc_cited(citations, BLOOKET_DOCS)
        and _answer_has_strong_phrase(answer, lang, {
            "en": ["blooket", "blooket instructions"],
            "zh-HK": ["blooket", "布魯克特"],
            "zh-CN": ["blooket", "布鲁克特"],
        })
        and _answer_is_short(answer)
    )

    if send_enrollment:
        answer += f"\n\nYou can download our enrollment form [here]({ENROLLMENT_FORM_URL})."
        _log("Enrollment form marker triggered.")
    if send_blooket:
        answer += f"\n\nYou can download the Blooket instructions [here]({BLOOKET_PDF_URL})."
        _log("Blooket marker triggered.")

    # --- Final response ---
    answer = answer or ""
    _log(f"Returning ChatResponse (len={len(answer)}).")
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
                        value = change.get("value", {})  # Cloud API bundles both messages and statuses here

                        # --- NEW: Handle business message status updates (detect admin activity) ---
                        for st in value.get("statuses", []) or []:
                            try:
                                msg_id = st.get("id") or st.get("message_id")
                                recipient_id = st.get("recipient_id")
                                if msg_id and msg_id not in _BOT_MSG_IDS:
                                    _mark_admin_activity(recipient_id)
                                else:
                                    _log(f"[COOL] Status for bot message id={msg_id} (ignored)")
                            except Exception as e:
                                _log(f"[COOL] Error processing status webhook: {e}")

                        messages = value.get("messages", [])
                        contacts = value.get("contacts", [])

                        if messages and contacts:
                            message = messages[0]
                            contact = contacts[0]
                            from_number = message.get("from")
                            message_type = message.get("type")
                            _log(f"Message details: from={from_number}, type={message_type}, contact={contact}")

                            if from_number:
                                _cancel_pending_ack(from_number)

                            if from_number and _in_admin_cooldown(from_number):
                                _log(f"[COOL] Admin cooldown active for {from_number}. Bot remains silent.")
                                try:
                                    now_ts = time.time()
                                    body_preview = message.get("text", {}).get("body") if message_type == "text" else f"<{message_type}>"
                                    save_message(from_number, "user", body_preview or "", get_language_code(body_preview or ""), now_ts)
                                    prune_history(from_number, keep=6)
                                except Exception as e:
                                    _log(f"[COOL] Error saving history during cooldown: {e}")
                                return {"status": "cooldown_active", "message": "Bot silenced due to recent admin activity"}

                            if message_type == "text":
                                message_body = message["text"].get("body")
                                _log(f"Text message from {from_number} (Name: {contact.get('profile',{}).get('name')}): '{message_body}'")

                                if from_number not in SETTINGS.whatsapp_test_numbers:
                                    _log(f"WARNING: Message from non-whitelisted number {from_number} ignored during testing.")
                                    return {"status": "ignored", "reason": "not in test numbers"}
                                
                                lang = get_language_code(message_body)
                                _log(f"Detected language: {lang}")

                                # NEW: Scheduling precedence for WhatsApp (treat ANY availability/pass-on as admin-handled)
                                sched_cls = classify_scheduling_context(message_body, lang)
                                is_scheduling_action = bool(
                                    (sched_cls.get("has_sched_verbs") or sched_cls.get("availability_request") or sched_cls.get("admin_action_request")
                                     or sched_cls.get("staff_contact_request") or sched_cls.get("individual_homework_request"))
                                    and not sched_cls.get("has_policy_intent")
                                )

                                opening_context = None
                                hint_canonical = None
                                is_hours_intent = False

                                if is_scheduling_action:
                                    opening_context = summarize_user_date_intent(message_body, lang)
                                    hint_canonical = None
                                    _log(f"Scheduling/action detected (WhatsApp). Providing date hints only:\n{opening_context}")
                                else:
                                    is_hours_intent, debug_intent = detect_opening_hours_intent(message_body, lang)
                                    if is_hours_intent:
                                        intent_debug_local = debug_intent or {}
                                        has_holiday_marker = bool(intent_debug_local.get("holiday_hits"))
                                        if is_general_hours_query(message_body, lang):
                                            opening_context = None
                                            hint_canonical = "opening_hours"
                                            _log("Opening hours intent detected as GENERAL. No system context injected; LLM will answer from policy docs.")
                                        else:
                                            opening_context = extract_opening_context(message_body, lang)
                                            hint_canonical = "opening_hours"
                                            _log(f"Opening hours intent detected as SPECIFIC. Structured context for LLM:\n{opening_context}")

                                # Build history and reformulation
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
                                        debug=SETTINGS.debug_kb,
                                        extra_context=opening_context,
                                        hint_canonical=hint_canonical,
                                    )
                                except Exception as e:
                                    _log(f"ERROR during chat_with_kb: {e}\n{traceback.format_exc()}")
                                    if is_hours_intent:
                                        answer = compute_opening_answer(message_body, lang)
                                        citations = []
                                        debug_info = {"source": "deterministic_opening_hours_fallback"}
                                        await _send_whatsapp_message(from_number, answer)
                                        return {"status": "ok", "message": "Sent deterministic opening hours answer"}
                                    raise HTTPException(status_code=500, detail=f"LLM backend error: {e}")

                                # ... inside chat(), after getting `answer, citations, debug_info` ...
                                # Avoid opening-hours fallback for any scheduling/availability/admin/teacher-contact requests
                                if not citations or contains_apology_or_noinfo(answer):
                                    _log("No citations found, or answer is a hedged/noinfo/apology. Silencing output.")
                                    block_hours_fallback = (
                                        sched_cls.get("has_sched_verbs")
                                        or sched_cls.get("availability_request")
                                        or sched_cls.get("admin_action_request")
                                        or sched_cls.get("staff_contact_request")
                                        or sched_cls.get("individual_homework_request")
                                        or sched_cls.get("placement_question")  # NEW: placement/judgement should not fall back to hours
                                        or _cites_admin_routing(citations)
                                        or _looks_like_leave_notification(rag_query)
                                    )
                                    if is_hours_intent and not block_hours_fallback:
                                        answer = compute_opening_answer(req.message, lang)
                                        citations = []
                                        debug_info = {"source": "deterministic_opening_hours_fallback"}
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

                                # Save history
                                try:
                                    now_ts = time.time()
                                    save_message(from_number, "user", message_body, lang, now_ts)
                                    save_message(from_number, "bot", answer or "", lang, now_ts + 0.01)
                                    prune_history(from_number, keep=6)
                                    _log(f"Saved and pruned DynamoDB history for session_id={from_number}")
                                except Exception as e:
                                    _log(f"ERROR saving/pruning DynamoDB history: {e}\n{traceback.format_exc()}")

                                _log(f"LLM raw answer: {answer!r}")
                                _log(f"LLM citations: {json.dumps(citations, ensure_ascii=False, indent=2)}")
                                if debug_info:
                                    _log(f"LLM debug_info: {json.dumps(debug_info, ensure_ascii=False, indent=2)}")
                                silent_reason = debug_info.get("silence_reason") if isinstance(debug_info, dict) else None
                                is_followup = is_followup_message(message_body)
                                if not answer and silent_reason == "no_citations" and is_followup and history:
                                    _log("Allowing context-only answer for followup message due to chat history, but LLM did not produce anything.")
                                    answer = ""  # Still silent

                                if not answer and silent_reason:
                                    _log(f"LLM silenced answer. Reason: {silent_reason}")

                                # Handle markers (enrollment/blooket)
                                answer, marker = extract_and_strip_marker(answer, "[SEND_ENROLLMENT_FORM]")
                                ENROLLMENT_FORM_URL = "https://drive.google.com/uc?export=download&id=1YTsUsTdf-k8ky-nJIFSZ7LtzzQ7BuzyA"
                                send_form = marker
                                answer, marker_blooket = extract_and_strip_marker(answer, "[SEND_BLOOKET_PDF]")
                                BLOOKET_PDF_URL = "https://drive.google.com/uc?export=download&id=18Ti5H8EoR7rmzzk4KGMGdQZFuqQ4uY4M"
                                send_blooket = marker_blooket

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
                                    _maybe_schedule_auto_ack_whatsapp(from_number, lang, base_ts=now_ts)

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