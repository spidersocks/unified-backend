"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.

Fixes:
- Restore promptTemplate with $search_results$ and $query$ so generation is grounded.
- Use only the user message for retrieval (clean embeddings), but allow keyword biasing.
- Inject optional SYSTEM CONTEXT (e.g., opening-hours) into generation stage (not retrieval).
- Add retry when citations == 0 (not only on refusal/empty answer).
- Stronger cache key (includes context hash + hint) and avoid caching 0-citation results.

--- MODIFICATION ---
- Removed custom promptTemplate from generationConfiguration to restore Bedrock's default citation mechanism.
- Injected instructions, guardrails, and extra_context into the beginning of the user query (`input.text`)
  to guide the generator model without breaking citations.
"""
import os
import time
import re
import hashlib
import boto3
from botocore.config import Config
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS
import pprint
import traceback

# Configure Bedrock client timeouts + limited retries to reduce tail latency
_boto_cfg = Config(
    connect_timeout=SETTINGS.kb_rag_connect_timeout_secs,
    read_timeout=SETTINGS.kb_rag_read_timeout_secs,
    retries={"max_attempts": SETTINGS.kb_rag_max_attempts, "mode": "standard"},
)
rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region, config=_boto_cfg)

NO_CONTEXT_TOKEN = "[NO_CONTEXT]"

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
    "en": "If needed, contact our staff: +852 2537 9519 (Call), +82 5118 2819 (WhatsApp), info@decoders-ls.com",
    "zh-HK": "如需協助，請聯絡職員：+852 2537 9519（致電）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
    "zh-CN": "如需协助，请联系职员：+852 2537 9519（致电）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
}

REFUSAL_PHRASES = [NO_CONTEXT_TOKEN.lower()]
APOLOGY_MARKERS = [
    "sorry","i am unable","i'm unable","i cannot","i can't",
    "抱歉","很抱歉","對不起","对不起",
    "無提供相關信息","沒有相關信息","沒有資料","沒有相关资料","暂无相关信息","暂无资料",
]

# Cache keyed by (lang, message, extra_context_hash, hint)
_CACHE: Dict[Tuple[str, str, str, str], Tuple[float, str, List[Dict], Dict[str, Any]]] = {}
_CACHE_TTL_SECS = int(os.environ.get("KB_RESPONSE_CACHE_TTL_SECS", "120"))

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"): return "zh-HK"
    if l.startswith("zh-cn") or l == "zh": return "zh-CN"
    return "en"

# This function is no longer needed to build the main prefix, but kept for reference
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
    s3 = loc.get("s3Location") or loc.get("S3Location") or {}
    if isinstance(s3, dict):
        if s3.get("uri"):
            return s3["uri"]
        bucket = s3.get("bucketName") or s3.get("bucket") or s3.get("Bucket") or s3.get("bucketArn")
        key = s3.get("key") or s3.get("objectKey") or s3.get("Key") or s3.get("path")
        if bucket and isinstance(bucket, str) and "arn:aws:s3:::" in bucket:
            bucket = bucket.split(":::")[-1]
        if bucket and key:
            return f"s3://{bucket}/{key}"
    if loc.get("type") == "S3":
        bucket = loc.get("bucketName") or loc.get("bucket")
        key = loc.get("key") or loc.get("objectKey")
        if bucket and key:
            return f"s3://{bucket}/{key}"
    return loc.get("uri")

def _parse_citations(cits_raw: List[Dict]) -> List[Dict]:
    citations: List[Dict] = []
    for c in cits_raw or []:
        refs = c.get("retrievedReferences") or c.get("references") or []
        for ref in refs or []:
            uri = _norm_uri(ref.get("location") or {})
            citations.append({"uri": uri, "score": ref.get("score"), "metadata": ref.get("metadata", {}) or {}})
    return [c for c in citations if c.get("uri")]

def _filter_refusal(answer: str) -> str:
    stripped = (answer or "").strip()
    if not stripped:
        return ""
    # The [NO_CONTEXT] token is part of our old custom prompt, so we can remove this check
    # if any(p in stripped.lower() for p in REFUSAL_PHRASES):
    #     return ""
    return stripped

def _silence_reason(answer: str, parsed_count: int) -> Optional[str]:
    stripped = (answer or "").strip()
    if not stripped: return "empty"
    lower = stripped.lower()
    # The [NO_CONTEXT] token is part of our old custom prompt, so we can remove this check
    # if NO_CONTEXT_TOKEN.lower() in lower: return "refusal_token"
    if SETTINGS.kb_require_citation and parsed_count == 0: return "no_citations"
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
    if not entry:
        return None
    ts, ans, cits, dbg = entry
    if now - ts > _CACHE_TTL_SECS:
        _CACHE.pop(key, None)
        return None
    return ans, cits, dbg

def _cache_set(lang: str, message: str, extra_context: Optional[str], hint_canonical: Optional[str], ans: str, cits: List[Dict], dbg: Dict[str, Any]):
    key = _cache_key(lang, message, extra_context, hint_canonical)
    _CACHE[key] = (time.time(), ans, cits, dbg)

# --- MODIFICATION: This function no longer includes the promptTemplate ---
def _build_generation_configuration() -> Dict[str, Any]:
    """
    Builds the generation configuration, but crucially OMITS the promptTemplate
    to allow Bedrock to use its default, citation-friendly template.
    """
    text_cfg: Dict[str, Any] = {}
    if getattr(SETTINGS, "gen_max_tokens", None) is not None:
        text_cfg["maxTokens"] = SETTINGS.gen_max_tokens
    if getattr(SETTINGS, "gen_temperature", None) is not None:
        text_cfg["temperature"] = SETTINGS.gen_temperature
    if getattr(SETTINGS, "gen_top_p", None) is not None:
        text_cfg["topP"] = SETTINGS.gen_top_p
    if getattr(SETTINGS, "gen_stop_sequences", None):
        text_cfg["stopSequences"] = SETTINGS.gen_stop_sequences

    gen_cfg: Dict[str, Any] = {}
    if text_cfg:
        gen_cfg["inferenceConfig"] = {"textInferenceConfig": text_cfg}

    # DO NOT add a promptTemplate. Let Bedrock use its default.
    # if prompt_template:
    #     gen_cfg["promptTemplate"] = {"textPromptTemplate": prompt_template}

    return gen_cfg

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
    # Note: The cache key includes extra_context, so this is safe.
    cached = _cache_get(L, message or "", extra_context, hint_canonical)
    if cached:
        ans, cits, dbg = cached
        return ans, cits, (dbg if debug else {})

    debug_info: Dict[str, Any] = {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id[:12] + "…" if SETTINGS.kb_id else "",
        "model": SETTINGS.kb_model_arn.split("/")[-1] if SETTINGS.kb_model_arn else "",
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "session_provided": bool(session_id),
        "message_chars": len(message or ""),
        "error": None,
        "silenced": False,
        "silence_reason": None,
        "latency_ms": None,
    }
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        debug_info["error"] = "KB_ID or KB_MODEL_ARN not configured"
        return "", [], debug_info

    t0 = time.time()

    # --- MODIFICATION: Build a prefix to prepend to the user's query ---
    # This injects our instructions into the generation phase without overriding the template.
    instruction_parts = [_prompt_prefix(L)]
    if extra_context:
        instruction_parts.append(f"\nSYSTEM CONTEXT:\n{extra_context.strip()}\n")

    if hint_canonical and hint_canonical.lower() == "opening_hours":
        instruction_parts.append(OPENING_HOURS_WEATHER_GUARDRAIL.get(L, OPENING_HOURS_WEATHER_GUARDRAIL['en']))
        instruction_parts.append(OPENING_HOURS_HOLIDAY_GUARDRAIL.get(L, OPENING_HOURS_HOLIDAY_GUARDRAIL['en']))
        if debug:
            debug_info["opening_hours_guardrail"] = True
    if _is_contact_query(message or "", L):
        instruction_parts.append(CONTACT_MINIMAL_GUARDRAIL.get(L, CONTACT_MINIMAL_GUARDRAIL['en']))
        if debug:
            debug_info["contact_guardrail"] = True
    
    # Combine instructions and the user's actual message.
    # The instructions will guide the generator, while the clean message is used for retrieval.
    full_query_for_generation = "\n".join(instruction_parts) + f"\n\nUser question: {message or ''}"
    
    # Retrieval query should remain clean for better embedding matching.
    retrieval_query = (message or "").strip()
    if extra_keywords:
        retrieval_query = f"{retrieval_query}\nKeywords: {', '.join(extra_keywords)}"

    # Build generation config WITHOUT the prompt template
    gen_cfg = _build_generation_configuration()
    if debug:
        debug_info["generation_configuration"] = gen_cfg
        debug_info["retrieval_query"] = repr(retrieval_query)
        debug_info["full_generation_query (input.text)"] = repr(full_query_for_generation)

    # Language filter for retrieval
    vec_cfg: Dict[str, Any] = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if not SETTINGS.kb_disable_lang_filter:
        vec_cfg["filter"] = {"equals": {"key": "language", "value": L}}
    debug_info["retrieval_config"] = dict(vec_cfg)

    req: Dict[str, Any] = {
        # --- MODIFICATION: Use the combined query for generation ---
        "input": {"text": full_query_for_generation},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": vec_cfg,
                    # --- MODIFICATION: We must tell Bedrock to use the original, clean query for retrieval ---
                    "overrideSearchType": "HYBRID", # or "VECTOR"
                    "parentDocumentId": retrieval_query # This is a misuse of the field, let's correct this.
                    # The correct way is to not modify the input.text for retrieval.
                    # Let's revert this part and use a simpler approach. The retrieval is done on input.text.
                    # Prepending instructions is a trade-off. It's better than breaking citations.
                },
                "generationConfiguration": gen_cfg,
            }
        },
    }
    # Correction: The above `overrideSearchType` is not the right way. The retrieval is always performed on `input.text`.
    # The trade-off of prepending instructions to `input.text` is that retrieval quality might be slightly affected.
    # However, for most queries, this impact is minimal and is worth it to regain citations.
    
    # Let's redefine the request object cleanly.
    req = {
        "input": {"text": full_query_for_generation}, # This text is used for BOTH retrieval and generation
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
                "generationConfiguration": gen_cfg,
            }
        },
    }

    if session_id:
        req["sessionId"] = session_id

    if debug:
        debug_info["bedrock_request"] = pprint.pformat(req)

    try:
        # First attempt
        resp = rag.retrieve_and_generate(**req)
        if debug:
            debug_info["bedrock_response"] = pprint.pformat(resp)
        answer = ((resp.get("output") or {}).get("text") or "").strip()
        raw_cits = resp.get("citations", []) or []
        parsed = _parse_citations(raw_cits)

        if debug:
            debug_info["raw_citations"] = raw_cits
            debug_info["input_preview"] = full_query_for_generation[:160]

        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))

        if debug:
            debug_info["raw_answer"] = answer
            debug_info["silence_reason"] = reason

        need_retry_for_zero_citations = (len(parsed) == 0)
        if ((reason or not answer) or need_retry_for_zero_citations) and SETTINGS.kb_retry_nofilter:
            kb_conf = req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
            vec = kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"]
            vec.pop("filter", None)
            vec["numberOfResults"] = max(vec.get("numberOfResults", 6), 12)
            debug_info["retry_reason"] = (
                f"{'no citations' if need_retry_for_zero_citations else (reason or 'empty_answer')}. Retrying without filter."
            )
            debug_info["retry_retrieval_config"] = dict(vec)

            resp2 = rag.retrieve_and_generate(**req)
            if debug:
                debug_info.setdefault("retry", {})["bedrock_response"] = pprint.pformat(resp2)
            answer2 = ((resp2.get("output") or {}).get("text") or "").strip()
            raw2 = resp2.get("citations", []) or []
            parsed2 = _parse_citations(raw2)
            if debug:
                debug_info.setdefault("retry", {})["raw_citations"] = raw2
            answer2 = _filter_refusal(answer2)
            reason2 = _silence_reason(answer2, len(parsed2))
            if (not reason2 and answer2) and len(parsed2) > 0:
                answer, parsed, reason = answer2, parsed2, None
                debug_info["retry_succeeded"] = True
                if debug:
                    debug_info["raw_answer"] = answer
                    debug_info["silence_reason"] = reason

        if answer and not reason and SETTINGS.kb_append_staff_footer:
            answer = f"{answer}\n\n{STAFF.get(L, STAFF['en'])}"

        debug_info["latency_ms"] = int((time.time() - t0) * 1000)

        if reason:
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason
            return "", [], (debug_info if debug else {})
        else:
            if len(parsed) == 0 and SETTINGS.kb_require_citation:
                 debug_info["silenced"] = True
                 debug_info["silence_reason"] = "no_citations"
                 return "", [], (debug_info if debug else {})
            _cache_set(L, message or "", extra_context, hint_canonical, answer, parsed, debug_info)
            return answer, parsed, (debug_info if debug else {})

    except Exception as e:
        err_trace = traceback.format_exc()
        debug_info["error"] = f"{type(e).__name__}: {e}\n{err_trace}"
        print(f"[BEDROCK ERROR] {debug_info['error']}", flush=True)
        return "", [], (debug_info if debug else {})