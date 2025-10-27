"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Optimized for latency: fewer retrieved chunks, smaller max tokens, no unconditional retry.
"""
import os
import time
import re
import boto3
from botocore.config import Config
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS
from llm import tags_index  # runtime tag index for query expansion

# Configure Bedrock client timeouts + limited retries to reduce tail latency
_boto_cfg = Config(
    connect_timeout=SETTINGS.kb_rag_connect_timeout_secs,
    read_timeout=SETTINGS.kb_rag_read_timeout_secs,
    retries={"max_attempts": SETTINGS.kb_rag_max_attempts, "mode": "standard"},
)
rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region, config=_boto_cfg)

NO_CONTEXT_TOKEN = "[NO_CONTEXT]"

# Shorter instructions (trim token budget)
INSTRUCTIONS = {
    "en": f"Answer ONLY from the retrieved context. Use short bullets. If context is insufficient or irrelevant, output {NO_CONTEXT_TOKEN} exactly.",
    "zh-HK": f"只可根據檢索內容作答。用精簡要點。若內容不足或無關，請輸出 {NO_CONTEXT_TOKEN}。",
    "zh-CN": f"仅按检索内容作答。用精简要点。若内容不足或无关，请输出 {NO_CONTEXT_TOKEN}。"
}

# Opening-hours specific guardrails
OPENING_HOURS_WEATHER_GUARDRAIL = {
    "en": "Important: Do NOT reference any weather information or weather policy unless the user asked about weather, or there is an active Black Rainstorm Signal or Typhoon Signal No. 8 (or above).",
    "zh-HK": "重要：除非用戶主動詢問天氣，或當前正生效黑雨或八號（或以上）風球，否則不要提及任何天氣資訊或天氣政策文件。",
    "zh-CN": "重要：除非用户主动询问天气，或当前正生效黑雨或八号（及以上）台风信号，否则不要引用任何天气信息或天气政策文档。",
}
OPENING_HOURS_HOLIDAY_GUARDRAIL = {
    "en": "Also: Do NOT mention public holidays unless the user asked about holidays, or the resolved date is a Hong Kong public holiday.",
    "zh-HK": "同時：除非用戶主動詢問或所涉日期是香港公眾假期，否則不要提及公眾假期。",
    "zh-CN": "同时：除非用户主动询问或所涉日期为香港公众假期，否则不要提及公众假期。",
}

# Contact-answer guardrail: return only phone and email by default
CONTACT_MINIMAL_GUARDRAIL = {
    "en": "If the user is asking for contact details, reply with ONLY phone and email on separate lines. Do not include address, map, or social links unless explicitly requested.",
    "zh-HK": "如用戶詢問聯絡方式，只回覆電話及電郵，各佔一行。除非用戶明確要求，請不要加入地址、地圖或社交連結。",
    "zh-CN": "如用户询问联系方式，只回复电话和电邮，各占一行。除非用户明确要求，请不要加入地址、地图或社交链接。",
}

# Optional staff footer (disabled by default; see SETTINGS.kb_append_staff_footer)
STAFF = {
    "en": "If needed, contact our staff: +852 2537 9519 (Call), +852 5118 2819 (WhatsApp), info@decoders-ls.com",
    "zh-HK": "如需協助，請聯絡職員：+852 2537 9519（致電）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
    "zh-CN": "如需协助，请联系职员：+852 2537 9519（致电）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
}

REFUSAL_PHRASES = [NO_CONTEXT_TOKEN.lower()]
APOLOGY_MARKERS = [
    "sorry","i am unable","i'm unable","i cannot","i can't",
    "抱歉","很抱歉","對不起","对不起",
    # generic “no info” markers to force silence
    "無提供相關信息","沒有相關信息","沒有資料","沒有相关资料","暂无相关信息","暂无资料",
]

# Tiny in-memory response cache to avoid double work on repeated/duplicate messages
_CACHE: Dict[Tuple[str,str], Tuple[float, str, List[Dict], Dict[str,Any]]] = {}
_CACHE_TTL_SECS = int(os.environ.get("KB_RESPONSE_CACHE_TTL_SECS", "120"))

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"): return "zh-HK"
    if l.startswith("zh-cn") or l == "zh": return "zh-CN"
    return "en"

def _prompt_prefix(lang: str) -> str:
    return f"{INSTRUCTIONS.get(lang, INSTRUCTIONS['en'])}\n\n"

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
    if not stripped: return ""
    if any(p in stripped.lower() for p in REFUSAL_PHRASES):
        return ""
    return stripped

def _silence_reason(answer: str, parsed_count: int) -> Optional[str]:
    stripped = (answer or "").strip()
    if not stripped: return "empty"
    lower = stripped.lower()
    if NO_CONTEXT_TOKEN.lower() in lower: return "refusal_token"
    if SETTINGS.kb_require_citation and parsed_count == 0: return "no_citations"
    if SETTINGS.kb_silence_apology and any(m in lower for m in APOLOGY_MARKERS): return "apology_marker"
    return None

def _cache_get(lang: str, message: str):
    key = (lang, message.strip())
    now = time.time()
    entry = _CACHE.get(key)
    if not entry: return None
    ts, ans, cits, dbg = entry
    if now - ts > _CACHE_TTL_SECS:
        _CACHE.pop(key, None)
        return None
    return ans, cits, dbg

def _cache_set(lang: str, message: str, ans: str, cits: List[Dict], dbg: Dict[str,Any]):
    _CACHE[(lang, message.strip())] = (time.time(), ans, cits, dbg)

def chat_with_kb(
    message: str,
    language: Optional[str] = None,
    session_id: Optional[str] = None,
    debug: bool = False,
    extra_context: Optional[str] = None,
    extra_keywords: Optional[List[str]] = None,
    hint_canonical: Optional[str] = None,
) -> Tuple[str, List[Dict], Dict[str, Any]]:
    # ...debug_info prep unchanged...

    L = _lang_label(language)
    t0 = time.time()

    # IMPORTANT: Use raw user message only to maximize retrieval recall
    input_text = (message or "")

    # Build language-only vector search config
    def make_lang_filter() -> Optional[Dict]:
        if SETTINGS.kb_disable_lang_filter:
            return None
        return {"equals": {"key": "language", "value": L}}

    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    lang_filter = make_lang_filter()
    if lang_filter:
        vec_cfg["filter"] = lang_filter

    debug_info["first_attempt_retrieval_config"] = dict(vec_cfg)
    debug_info["retrieval_config"] = dict(vec_cfg)

    base_req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
                "generationConfiguration": {
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": SETTINGS.gen_max_tokens,
                            "temperature": SETTINGS.gen_temperature,
                            "topP": SETTINGS.gen_top_p,
                        }
                    }
                },
            }
        },
    }
    if session_id:
        base_req["sessionId"] = session_id

    try:
        resp = rag.retrieve_and_generate(**base_req)
        answer = ((resp.get("output") or {}).get("text") or "").strip()
        raw_cits = resp.get("citations", []) or []
        parsed = _parse_citations(raw_cits)

        if debug:
            debug_info["raw_citations"] = raw_cits

        # Keep your existing post-filters
        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))

        # Optional retry without language filter if enabled
        if (reason or not answer) and SETTINGS.kb_retry_nofilter:
            kb_conf = base_req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
            vec = kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"]
            vec.pop("filter", None)
            debug_info["retry_reason"] = f"Initial attempt failed ({reason or 'empty_answer'}). Retrying without filter."
            debug_info["retry_retrieval_config"] = dict(vec)

            resp2 = rag.retrieve_and_generate(**base_req)
            answer2 = ((resp2.get("output") or {}).get("text") or "").strip()
            raw2 = resp2.get("citations", []) or []
            parsed2 = _parse_citations(raw2)
            if debug:
                debug_info.setdefault("retry", {})["raw_citations"] = raw2
            answer2 = _filter_refusal(answer2)
            reason2 = _silence_reason(answer2, len(parsed2))
            if not reason2 and answer2:
                answer, parsed, reason = answer2, parsed2, None
                debug_info["retry_succeeded"] = True

        debug_info["latency_ms"] = int((time.time() - t0) * 1000)
        if reason:
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason
            _cache_set(L, message or "", "", [], debug_info)
            return "", [], (debug_info if debug else {})
        _cache_set(L, message or "", answer, parsed, debug_info)
        return answer, parsed, (debug_info if debug else {})
    except Exception as e:
        debug_info["error"] = f"{type(e).__name__}: {e}"
        return "", [], (debug_info if debug else {})

    def make_vec_cfg(base_filter: Optional[Dict] = None) -> Dict:
        cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
        if base_filter:
            cfg["filter"] = base_filter
        return cfg

    def make_lang_filter() -> Optional[Dict]:
        if SETTINGS.kb_disable_lang_filter:
            return None
        return {"equals": {"key": "language", "value": L}}

    # Language-only metadata filter (no type/canonical bias)
    lang_filter = make_lang_filter()
    vec_cfg = make_vec_cfg(lang_filter)
    debug_info["first_attempt_retrieval_config"] = dict(vec_cfg)
    debug_info["retrieval_config"] = vec_cfg  # may be mutated for retry

    base_req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
                "generationConfiguration": {
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": SETTINGS.gen_max_tokens,
                            "temperature": SETTINGS.gen_temperature,
                            "topP": SETTINGS.gen_top_p,
                        }
                    }
                },
            }
        },
    }
    if session_id:
        base_req["sessionId"] = session_id

    try:
        resp = rag.retrieve_and_generate(**base_req)
        # Parse answer
        answer = ((resp.get("output") or {}).get("text") or "").strip()
        raw_cits = resp.get("citations", []) or []
        parsed = _parse_citations(raw_cits)

        if debug:
            debug_info["raw_citations"] = raw_cits

        # Post-filters
        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))

        # Optional retry: if enabled, truly drop ALL filters to diagnose retrieval blockage.
        if (reason or not answer) and SETTINGS.kb_retry_nofilter:
            kb_conf = base_req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
            vec = kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"]
            vec.pop("filter", None)  # drop filter entirely
            debug_info["retry_reason"] = f"Initial attempt failed ({reason or 'empty_answer'}). Retrying without filter."
            debug_info["retry_retrieval_config"] = dict(vec)

            resp2 = rag.retrieve_and_generate(**base_req)
            answer2 = ((resp2.get("output") or {}).get("text") or "").strip()
            raw2 = resp2.get("citations", []) or []
            parsed2 = _parse_citations(raw2)
            if debug:
                debug_info.setdefault("retry", {})["raw_citations"] = raw2
            answer2 = _filter_refusal(answer2)
            reason2 = _silence_reason(answer2, len(parsed2))
            if not reason2 and answer2:
                answer, parsed, reason = answer2, parsed2, None
                debug_info["retry_succeeded"] = True

        # Append staff footer only if explicitly enabled
        if answer and not reason and SETTINGS.kb_append_staff_footer:
            footer = STAFF.get(L, STAFF["en"])
            answer = f"{answer}\n\n{footer}"

        debug_info["latency_ms"] = int((time.time() - t0) * 1000)
        if reason:
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason
            _cache_set(L, message or "", "", [], debug_info)
            return "", [], (debug_info if debug else {})
        _cache_set(L, message or "", answer, parsed, debug_info)
        return answer, parsed, (debug_info if debug else {})

    except Exception as e:
        debug_info["error"] = f"{type(e).__name__}: {e}"
        return "", [], (debug_info if debug else {})