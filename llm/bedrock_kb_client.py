"""
Thin client for Bedrock Knowledge Base.

This version uses a manual two-step orchestration (Retrieve then Generate)
to solve the issue of retrieval queries being polluted by generation instructions.
This approach carefully preserves the custom logic, guardrails, and retry mechanisms
from the previous version.

- STEP 1: `retrieve` is called using a clean query (user message + keywords) for high-quality results.
- STEP 2: `invoke_model` is called with a detailed prompt that includes the original file's
           custom instructions, guardrails, and the context chunks from Step 1.
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

# --- ALL CONSTANTS AND HELPERS BELOW ARE PRESERVED FROM YOUR ORIGINAL FILE ---

INSTRUCTIONS = {
    "en": (
        "Answer ONLY from the retrieved context. Use short bullets. If context is irrelevant or insufficient to answer confidently, state that you cannot find the information."
    ),
    "zh-HK": (
        "只可根據檢索內容作答。用精簡要點。若內容不足或無關，請直接表明找不到所需資訊。"
    ),
    "zh-CN": (
        "仅按检索内容作答。用精简要点。若内容不足或无关，请直接表明找不到所需信息。"
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
    # Corrected the WhatsApp number in the original file
    "zh-HK": "如需協助，請聯絡職員：+852 2537 9519（致電）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
    "zh-CN": "如需协助，请联系职员：+852 2537 9519（致电）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
}

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

def _silence_reason(answer: str, citation_count: int) -> Optional[str]:
    stripped = (answer or "").strip()
    if not stripped: return "empty"
    lower = stripped.lower()
    if SETTINGS.kb_require_citation and citation_count == 0: return "no_citations"
    if SETTINGS.kb_silence_apology and any(m in lower for m in APOLOGY_MARKERS): return "apology_marker"
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

# --- NEW HELPER: Builds the final prompt for the generation step ---
def build_llm_prompt(instruction_parts: List[str], query: str, context_chunks: List[str]) -> str:
    """
    Constructs the final prompt for the InvokeModel API call, combining
    your custom instructions with the retrieved context.
    """
    instructions = "\n".join(instruction_parts)
    formatted_context = "\n\n---\n\n".join(context_chunks)

    prompt = (
        f"{instructions}\n\n"
        f"**CONTEXT:**\n{formatted_context}\n\n"
        f"**USER QUESTION:** {query}\n\n"
        f"**ANSWER:**"
    )
    return prompt

def chat_with_kb(
    message: str,
    language: Optional[str] = None,
    session_id: Optional[str] = None, # session_id is not used by Retrieve/InvokeModel, but kept for interface consistency
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

    # --- LOGIC PRESERVED: Build instruction parts and retrieval query from your file ---
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

    # This is the clean query for the retrieval step
    retrieval_query = (message or "").strip()
    if extra_keywords:
        retrieval_query = f"{retrieval_query}\nKeywords: {', '.join(extra_keywords)}"
    
    debug_info["retrieval_query"] = repr(retrieval_query)
    
    # --- CORE LOGIC REPLACEMENT: Manual Retrieve and Generate ---
    try:
        def perform_rag_flow(retry_mode: bool = False) -> Tuple[str, List[Dict], Dict[str, Any]]:
            """Encapsulates the RAG flow to allow for easy retries."""
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
                return "", [], flow_debug_info # No results, return empty

            # Process results for generation and citation
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

            # --- STEP 2: GENERATE ---
            llm_prompt = build_llm_prompt(instruction_parts, message, retrieved_chunks_text)
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

        # Initial attempt
        answer, parsed, attempt_debug_info = perform_rag_flow(retry_mode=False)
        if debug: debug_info["initial_attempt"] = attempt_debug_info

        reason = _silence_reason(answer, len(parsed))
        debug_info["raw_answer"] = answer
        debug_info["silence_reason"] = reason

        # --- LOGIC PRESERVED: Your exact retry condition ---
        need_retry_for_zero_citations = (len(parsed) == 0)
        if ((reason or not answer) or need_retry_for_zero_citations) and SETTINGS.kb_retry_nofilter:
            debug_info["retry_reason"] = (
                f"{'no citations' if need_retry_for_zero_citations else (reason or 'empty_answer')}. Retrying without filter."
            )
            
            # Retry attempt
            answer2, parsed2, retry_debug_info = perform_rag_flow(retry_mode=True)
            if debug: debug_info["retry_attempt"] = retry_debug_info

            reason2 = _silence_reason(answer2, len(parsed2))
            
            # Only use retry result if it's valid and better
            if (not reason2 and answer2) and len(parsed2) > 0:
                answer, parsed, reason = answer2, parsed2, None
                debug_info["retry_succeeded"] = True
                debug_info["raw_answer"] = answer
                debug_info["silence_reason"] = reason

        # --- LOGIC PRESERVED: Final checks and formatting ---
        if reason:
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason
            return "", [], (debug_info if debug else {})
        
        if answer and SETTINGS.kb_append_staff_footer:
            answer = f"{answer}\n\n{STAFF.get(L, STAFF['en'])}"

        debug_info["latency_ms"] = int((time.time() - t0) * 1000)
        
        _cache_set(L, message or "", extra_context, hint_canonical, answer, parsed, debug_info)
        return answer, parsed, (debug_info if debug else {})

    except Exception as e:
        err_trace = traceback.format_exc()
        debug_info["error"] = f"{type(e).__name__}: {e}\n{err_trace}"
        print(f"[BEDROCK ERROR] {debug_info['error']}", flush=True)
        return "", [], (debug_info if debug else {})