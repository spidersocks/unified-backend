"""
Thin client for Bedrock Knowledge Base.

This version uses a manual two-step orchestration (Retrieve then Generate)
to solve the issue of retrieval queries being polluted by generation instructions.
This approach carefully preserves the custom logic, guardrails, and retry mechanisms
from the previous version.

- STEP 1: `retrieve` is called using a clean query (user message + keywords) for high-quality results.
- STEP 2: `invoke_model` is called with a detailed, STRICT, and FULLY LOCALIZED prompt that includes the
           original file's custom instructions, guardrails, and the context chunks from Step 1.
- All existing helper functions, constants, retry logic, and caching are preserved.
"""
import os
import time
import re
import hashlib
import boto3
import json
from botocore.config import Config
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS
import pprint
import traceback
from llm.intent import classify_scheduling_context, is_politeness_only

# --- MODIFICATION: Add a client for the Bedrock Runtime (for InvokeModel) ---
_boto_cfg = Config(
    connect_timeout=SETTINGS.kb_rag_connect_timeout_secs,
    read_timeout=SETTINGS.kb_rag_read_timeout_secs,
    retries={"max_attempts": SETTINGS.kb_rag_max_attempts, "mode": "standard"},
)
# Client for Knowledge Base APIs (Retrieve)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region, config=_boto_cfg)
# Client for Foundation Model APIs (InvokeModel)
bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=SETTINGS.aws_region, config=_boto_cfg)

# --- MODIFIED: Persona, Strict Silence Instructions, and Guardrails ---

# The instructions now demand a specific token for silence, which is more reliable.
INSTRUCTIONS = {
    "en": (
        "Use short, helpful bullets. If the context is irrelevant or insufficient to answer confidently, you MUST reply with ONLY the exact text `[NO_ANSWER]` and nothing else."
    ),
    "zh-HK": (
        "用精簡要點友善作答。若內容不足或無關，你*必須*只回答 `[NO_ANSWER]`，唔可以加任何其他文字。"
    ),
    "zh-CN": (
        "用精简要点友善作答。若内容不足或无关，你*必须*仅回答 `[NO_ANSWER]`，不要加任何其他文字。"
    ),
}

# The role now establishes a clear persona as a "Little Scholars" admin agent.
PROMPT_SCAFFOLD = {
    "en": {
        "role": "You are a helpful admin agent for 'Little Scholars', a Hong Kong youth education center. Your task is to answer parents' questions based *only* on the provided information. Answer in the first person, using 'we' and 'our'. Never mention that you are an AI or that the information comes from a document. Your tone should be helpful and direct.",
        "use_results": "Here is the internal information to use:",
        "ask": "Based *only* on the information above, answer the following parent's question:",
        "answer_label": "Answer:",
    },
    "zh-HK": {
        "role": "你係「Little Scholars」（一間香港青少年教育中心）嘅行政助理。你嘅任務係*只*根據提供嘅資料回答家長問題。請用第一人稱（「我哋」）回答。絕對唔好提及你係AI或者資料來源。語氣要友善直接。",
        "use_results": "請使用以下內部資料：",
        "ask": "僅根據上述資料，回答家長嘅以下問題：",
        "answer_label": "答案：",
    },
    "zh-CN": {
        "role": "你是‘Little Scholars’（一间香港青少年教育中心）的行政助理。你的任务是*仅*根据提供的信息回答家长的问题。请用第一人称（‘我们’）回答。绝不提及你是AI或信息来源。语气应友善直接。",
        "use_results": "请使用以下内部信息：",
        "ask": "仅根据上述信息，回答家长的以下问题：",
        "answer_label": "答案：",
    },
}

# --- MODIFIED: CRITICAL GUARDRAIL FOR SCHEDULING & LEAVE ---
# Added a clarification to allow answering general opening hours questions.
CRITICAL_SCHEDULING_GUARDRAIL = {
    "en": (
        "ABSOLUTE RULES (Scheduling & Politeness):\n"
        "1) You are an admin assistant, NOT an admin. Do NOT arrange, approve, confirm or modify bookings.\n"
        "2) If the user is asking to book/reschedule/cancel/leave for a specific date/time (e.g., next Friday, 10/11, 3pm) and is NOT explicitly asking about policy, reply only with [NO_ANSWER].\n"
        "3) If the user asks about availability, time slots, timetable/schedule, teacher availability, or start date for a specific child (including messages mentioning a child’s name or 'after/completed assessment'), reply only with [NO_ANSWER]. These are admin‑handled.\n"
        "4) If the user mentions a date/time or a specific student but explicitly asks about our policy (e.g., 'what is the policy on rescheduling?'), answer the policy question from context, and clearly avoid making any arrangements.\n"
        "5) Politeness-only replies like 'You're welcome' are ONLY for messages that contain nothing but a thank-you.\n"
        "6) If the user asks you to pass/forward/relay/notify/tell/ask/remind a teacher or staff (e.g., 'please tell the teacher…', 'help ask teachers to…'), provide only [NO_ANSWER]. Do NOT relay messages."
    ),
    "zh-HK": (
        "絕對規則（行程安排與禮貌）：\n"
        "1）你係行政助理，唔係管理員。唔可以安排／批准／確認／更改任何預約。\n"
        "2）如家長就特定日期／時間提出預約／改期／取消／請假，而並非詢問政策，*只*回覆 [NO_ANSWER]。\n"
        "3）如家長查問『有冇時段／時間表／檔期／老師檔期／幾時開始上課』，或訊息提及具體學生名字、或『完成／之後評估』等，均屬行政安排，*只*回覆 [NO_ANSWER]。\n"
        "4）如訊息包含日期／學生，但明確問政策（例如「改期政策係點？」），請根據內容回答政策，並清楚表明不作任何安排。\n"
        "5）「不客氣」等純禮貌回覆只適用於訊息本身只有致謝。\n"
        "6）如家長要求『轉告／通知／幫手問／同老師講／提醒』老師或職員，*只*回覆 [NO_ANSWER]；唔好代人轉達訊息。"
    ),
    "zh-CN": (
        "绝对规则（行程与礼貌）：\n"
        "1）你是行政助理，不是管理员。不可安排／批准／确认／更改任何预约。\n"
        "2）如家长就特定日期／时间提出预约／改期／取消／请假，而不是询问政策，*仅*回复 [NO_ANSWER]。\n"
        "3）如家长询问『是否有可用时段／时间表／档期／老师档期／开课时间』，或消息提及具体学生姓名、或『完成／之后评估』等，一律*仅*回复 [NO_ANSWER]。\n"
        "4）如消息包含日期／学生，但明确询问政策（如“改期政策是什么？”），请根据内容回答政策，并明确说明不进行任何安排。\n"
        "5）“不客气”等纯礼貌回复仅用于消息只有致谢时。\n"
        "6）如家长要求『转告／通知／帮我问／跟老师说／提醒』老师或工作人员，*仅*回复 [NO_ANSWER]，不要代为转达信息。"
    ),
}


OPENING_HOURS_WEATHER_GUARDRAIL = {
    "en": "Important: Do NOT reference weather unless the user asked, or there is an active Black Rainstorm Signal or Typhoon Signal No. 8 (or above).",
    "zh-HK": "重要：除非用戶主動詢問天氣，或正生效黑雨或八號（或以上）風球，否則不要提及任何天氣資訊或天氣政策文件。",
    "zh-CN": "重要：除非用户主动询问天气，或正生效黑雨或八号（及以上）台风信号，否则不要引用任何天气信息或天气政策文档。",
}
OPENING_HOURS_HOLIDAY_GUARDRAIL = {
    "en": "Also: Do NOT mention public holidays unless the user asked, or the resolved date is a Hong Kong public holiday.",
    "zh-HK": "同時：除非用戶主動詢問或所涉日期是香港公眾假期，否則不要提及公眾假期。",
    "zh-CN": "同时：除非用户主动询问或所涉日期为香港公众假期，否则不要提及公众假期。",
}

CONTACT_MINIMAL_GUARDRAIL = {
    "en": "If the user asks for contact details, reply with ONLY phone and email on separate lines. Do not include address/map/social unless explicitly requested.",
    "zh-HK": "如用戶詢問聯絡方式，只回覆電話及電郵，各佔一行。除非用戶明確要求，請不要加入地址、地圖或社交連結。",
    "zh-CN": "如用户询问联系方式，只回复电话和电邮，各占一行。除非用户明确要求，请不要加入地址、地图或社交链接。",
}

STAFF = {
    "en": "If needed, contact our staff: +852 2537 9519 (Call), +852 5118 2819 (WhatsApp), info@decoders-ls.com",
    "zh-HK": "如需協助，請聯絡職員：+852 2537 9519（致電）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
    "zh-CN": "如需协助，请联系职员：+852 2537 9519（致电）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
}

# --- RE-INTRODUCED: Apology markers as a fallback silencing mechanism ---
APOLOGY_MARKERS = [
    "sorry","i am unable","i'm unable","i cannot","i can't", "not specified", "not mentioned",
    "抱歉","很抱歉","對不起","对不起",
    "無提供相關信息","沒有相關信息","沒有資料","沒有相关资料","暂无相关信息","暂无资料",
]

_CACHE: Dict[Tuple[str, str, str, str], Tuple[float, str, List[Dict], Dict[str, Any]]] = {}
_CACHE_TTL_SECS = int(os.environ.get("KB_RESPONSE_CACHE_TTL_SECS", "120"))

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"): return "zh-HK"
    if l.startswith("zh-cn") or l == "zh": return "zh-CN"
    return "en"

def _prompt_prefix(lang: str) -> str:
    return INSTRUCTIONS.get(lang, INSTRUCTIONS["en"])

def _is_contact_query(message: str, lang: Optional[str]) -> bool:
    m = (message or "").lower()
    if not m:
        return False
    if lang and str(lang).lower().startswith("zh-hk"):
        return bool(re.search(r"聯絡|聯絡資料|電話|致電|電郵|whatsapp|联系|联系方式", m, flags=re.IGNORECASE))
    if lang and (str(lang).lower().startswith("zh-cn") or str(lang).lower() == "zh"):
        return bool(re.search(r"联系|联系方式|电话|致电|电邮|邮箱|whatsapp", m, flags=re.IGNORECASE))
    return bool(re.search(r"\b(contact|phone|call|email|e-?mail|whatsapp)\b", m, flags=re.IGNORECASE))

def _norm_uri(loc: Dict) -> Optional[str]:
    s3 = loc.get("s3Location") or {}
    if s3.get("uri"):
        return s3["uri"]
    return None

# --- MODIFIED: Hybrid silencing logic using both token and apology markers ---
def _silence_reason(answer: str, citation_count: int) -> Optional[str]:
    """
    Determines if a response should be silenced using a hybrid approach.
    Silence is triggered by:
    1. The model explicitly outputting the '[NO_ANSWER]' token (Primary).
    2. The response containing a known apology phrase (Secondary Fallback).
    3. The response being empty.
    4. The response having no citations, if citations are required.
    """
    stripped = (answer or "").strip()
    
    # 1. Primary check: The model has explicitly stated it cannot answer.
    if stripped == "[NO_ANSWER]":
        return "no_answer_token"
    
    # 2. Secondary check: Heuristic fallback for natural language apologies.
    lower = stripped.lower()
    if SETTINGS.kb_silence_apology and any(m in lower for m in APOLOGY_MARKERS):
        return "apology_marker"
        
    # 3. Other checks
    if not stripped:
        return "empty"
    if SETTINGS.kb_require_citation and citation_count == 0:
        return "no_citations"
    
    return None

def _cache_key(lang: str, message: str, extra_context: Optional[str], hint_canonical: Optional[str]) -> Tuple[str, str, str, str]:
    ec = extra_context or ""
    ec_hash = hashlib.sha256(ec.encode("utf-8")).hexdigest()[:12] if ec else ""
    hc = (hint_canonical or "").strip().lower()
    return (lang, (message or "").strip(), ec_hash, hc)

def _cache_get(lang: str, message: str, extra_context: Optional[str], hint_canonical: Optional[str]):
    key = _cache_key(lang, message, extra_context, hint_canonical)
    now = time.time()
    entry = _CACHE.get(key)
    if not entry: return None
    ts, ans, cits, dbg = entry
    if now - ts > _CACHE_TTL_SECS:
        _CACHE.pop(key, None)
        return None
    return ans, cits, dbg

def _cache_set(lang: str, message: str, extra_context: Optional[str], hint_canonical: Optional[str], ans: str, cits: List[Dict], dbg: Dict[str, Any]):
    key = _cache_key(lang, message, extra_context, hint_canonical)
    _CACHE[key] = (time.time(), ans, cits, dbg)

def build_llm_prompt(lang: str, instruction_parts: List[str], query: str, context_chunks: List[str]) -> str:
    scaffold = PROMPT_SCAFFOLD.get(lang, PROMPT_SCAFFOLD["en"])

    # Always include core scheduling & politeness guardrail
    final_instructions = [CRITICAL_SCHEDULING_GUARDRAIL.get(lang, CRITICAL_SCHEDULING_GUARDRAIL["en"])]

    # Soft steering based on message classification (no hard gating)
    try:
        cls = classify_scheduling_context(query or "", lang)
    except Exception:
        cls = {"has_sched_verbs": False, "has_date_time": False, "has_policy_intent": False, "politeness_only": False, "availability_request": False, "admin_action_request": False}

    if cls.get("has_sched_verbs") and (cls.get("has_date_time") or cls.get("availability_request")) and not cls.get("has_policy_intent"):
        final_instructions.append("This looks like a scheduling action or availability/time-slot request. Provide only [NO_ANSWER]. Do not describe policy or processes.")
    if cls.get("admin_action_request") and not cls.get("has_policy_intent"):
        final_instructions.append("User asks to pass/relay a message to teacher/staff. Provide only [NO_ANSWER]. Do NOT relay messages.")
    if cls.get("has_policy_intent"):
        final_instructions.append("User is asking about policy. Answer from context. Do NOT make or confirm any arrangements.")
    if not cls.get("politeness_only"):
        final_instructions.append("Do NOT use a politeness-only reply.")

    final_instructions.extend(instruction_parts)
    instructions = "\n\n".join(final_instructions)

    formatted_context = ""
    for i, chunk in enumerate(context_chunks):
        formatted_context += f"<search_result index=\"{i+1}\">\n{chunk}\n</search_result>\n\n"

    prompt = (
        f"{scaffold['role']}\n\n"
        f"<instructions>\n{instructions}\n</instructions>\n\n"
        f"{scaffold['use_results']}\n"
        f"<search_results>\n{formatted_context.strip()}\n</search_results>\n\n"
        f"{scaffold['ask']}\n"
        f"<question>\n{query}\n</question>\n\n"
        f"{scaffold['answer_label']}"
    )
    return prompt

def chat_with_kb(
    message: str,
    language: Optional[str] = None,
    session_id: Optional[str] = None,
    debug: bool = False,
    extra_context: Optional[str] = None,
    extra_keywords: Optional[List[str]] = None,
    hint_canonical: Optional[str] = None,
) -> Tuple[str, List[Dict], Dict[str, Any]]:
    L = _lang_label(language)
    cached = _cache_get(L, message or "", extra_context, hint_canonical)
    if cached:
        ans, cits, dbg = cached
        return ans, cits, (dbg if debug else {})

    debug_info: Dict[str, Any] = {
        "orchestration_mode": "manual_retrieve_then_generate",
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id,
        "llm_model_id": SETTINGS.llm_model_id,
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "message_chars": len(message or ""),
        "error": None,
        "silenced": False,
        "silence_reason": None,
        "latency_ms": None,
    }
    if not SETTINGS.kb_id or not SETTINGS.llm_model_id:
        debug_info["error"] = "KB_ID or LLM_MODEL_ID not configured"
        return "", [], debug_info

    t0 = time.time()

    instruction_parts = [_prompt_prefix(L)]
    if extra_context:
        instruction_parts.append(f"\nSYSTEM CONTEXT:\n{extra_context.strip()}\n")

    if hint_canonical and hint_canonical.lower() == "opening_hours":
        instruction_parts.append(OPENING_HOURS_WEATHER_GUARDRAIL.get(L, OPENING_HOURS_WEATHER_GUARDRAIL['en']))
        instruction_parts.append(OPENING_HOURS_HOLIDAY_GUARDRAIL.get(L, OPENING_HOURS_HOLIDAY_GUARDRAIL['en']))
        if debug: debug_info["opening_hours_guardrail"] = True
    if _is_contact_query(message or "", L):
        instruction_parts.append(CONTACT_MINIMAL_GUARDRAIL.get(L, CONTACT_MINIMAL_GUARDRAIL['en']))
        if debug: debug_info["contact_guardrail"] = True

    retrieval_query = (message or "").strip()
    if extra_keywords:
        retrieval_query = f"{retrieval_query}\nKeywords: {', '.join(extra_keywords)}"
    
    debug_info["retrieval_query"] = repr(retrieval_query)
    
    try:
        def perform_rag_flow(retry_mode: bool = False) -> Tuple[str, List[Dict], Dict[str, Any]]:
            flow_debug_info = {}
            
            # --- STEP 1: RETRIEVE ---
            vec_cfg: Dict[str, Any] = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
            if retry_mode:
                vec_cfg["numberOfResults"] = max(vec_cfg.get("numberOfResults", 6), 12)
                flow_debug_info["retrieval_mode"] = "retry_no_filter"
            elif not SETTINGS.kb_disable_lang_filter:
                vec_cfg["filter"] = {"equals": {"key": "language", "value": L}}
                flow_debug_info["retrieval_mode"] = "initial_with_filter"

            retrieval_config = {"vectorSearchConfiguration": vec_cfg}
            flow_debug_info["retrieval_config"] = retrieval_config

            retrieve_response = bedrock_agent_client.retrieve(
                knowledgeBaseId=SETTINGS.kb_id,
                retrievalQuery={'text': retrieval_query},
                retrievalConfiguration=retrieval_config
            )
            flow_debug_info["retrieval_response"] = retrieve_response

            retrieval_results = retrieve_response.get('retrievalResults', [])
            if not retrieval_results:
                # If retrieval finds nothing, we can short-circuit and signal no answer.
                return "[NO_ANSWER]", [], flow_debug_info

            retrieved_chunks_text: List[str] = []
            parsed_citations: List[Dict] = []
            for result in retrieval_results:
                retrieved_chunks_text.append(result['content']['text'])
                parsed_citations.append({
                    "uri": _norm_uri(result.get('location', {})),
                    "score": result.get('score'),
                    "metadata": result.get('metadata', {})
                })
            
            flow_debug_info["retrieved_chunk_count"] = len(retrieved_chunks_text)
            flow_debug_info["parsed_citations"] = parsed_citations

            # --- STEP 2: GENERATE (using the localized prompt) ---
            llm_prompt = build_llm_prompt(L, instruction_parts, message, retrieved_chunks_text)
            flow_debug_info["llm_prompt"] = llm_prompt

            body = json.dumps({
                "prompt": llm_prompt,
                "max_gen_len": SETTINGS.gen_max_tokens,
                "temperature": SETTINGS.gen_temperature,
                "top_p": SETTINGS.gen_top_p,
            })

            invoke_response = bedrock_runtime_client.invoke_model(
                body=body, modelId=SETTINGS.llm_model_id,
                accept='application/json', contentType='application/json'
            )
            response_body = json.loads(invoke_response.get('body').read())
            answer = response_body.get('generation', '').strip()
            flow_debug_info["llm_raw_response"] = response_body
            
            return answer, parsed_citations, flow_debug_info

        answer, parsed, attempt_debug_info = perform_rag_flow(retry_mode=False)
        if debug: debug_info["initial_attempt"] = attempt_debug_info

        reason = _silence_reason(answer, len(parsed))
        debug_info["raw_answer"] = answer
        debug_info["silence_reason"] = reason

        need_retry_for_zero_citations = (len(parsed) == 0)
        # We retry if the answer is silenced OR if there are no citations (as before)
        if (reason or need_retry_for_zero_citations) and SETTINGS.kb_retry_nofilter:
            debug_info["retry_reason"] = (
                f"{'no citations' if need_retry_for_zero_citations else reason}. Retrying without filter."
            )
            
            answer2, parsed2, retry_debug_info = perform_rag_flow(retry_mode=True)
            if debug: debug_info["retry_attempt"] = retry_debug_info
            reason2 = _silence_reason(answer2, len(parsed2))
            
            # We accept the retry result if it's NOT silenced and has citations
            if not reason2 and len(parsed2) > 0:
                answer, parsed, reason = answer2, parsed2, None
                debug_info["retry_succeeded"] = True
                debug_info["raw_answer"] = answer
                debug_info["silence_reason"] = reason

        if reason:
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason
            # --- MODIFICATION: Return the explicit [NO_ANSWER] token on silence ---
            return "[NO_ANSWER]", [], (debug_info if debug else {})
        
        if answer and SETTINGS.kb_append_staff_footer:
            answer = f"{answer}\n\n{STAFF.get(L, STAFF['en'])}"

        debug_info["latency_ms"] = int((time.time() - t0) * 1000)
        
        _cache_set(L, message or "", extra_context, hint_canonical, answer, parsed, debug_info)
        return answer, parsed, (debug_info if debug else {})

    except Exception as e:
        err_trace = traceback.format_exc()
        debug_info["error"] = f"{type(e).__name__}: {e}\n{err_trace}"
        print(f"[BEDROCK ERROR] {debug_info['error']}", flush=True)
        # --- MODIFICATION: Return the explicit [NO_ANSWER] token on error ---
        return "[NO_ANSWER]", [], (debug_info if debug else {})