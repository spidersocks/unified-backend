"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Optimized for latency: fewer retrieved chunks, smaller max tokens, no unconditional retry.
"""
import os
import time
import boto3
from botocore.config import Config
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS

# Configure Bedrock client timeouts + limited retries to reduce tail latency
_boto_cfg = Config(
    connect_timeout=SETTINGS.kb_rag_connect_timeout_secs,
    read_timeout=SETTINGS.kb_rag_read_timeout_secs,
    retries={"max_attempts": SETTINGS.kb_rag_max_attempts, "mode": "standard"},
)
rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region, config=_boto_cfg)

NO_CONTEXT_TOKEN = "[NO_CONTEXT]"

# Known canonical course keys (duplicates llm.content_store.CANONICAL_COURSES without triggering a store load)
COURSE_KEYS = {
    "Playgroups","Phonics","LanguageArts","Clevercal","Alludio",
    "ToddlerCharRecognition","MandarinPinyin","ChineseLanguageArts","PrivateClass",
}

def _detect_canonical(message: str) -> Optional[str]:
    m = (message or "").lower()
    for k in COURSE_KEYS:
        if k.lower() in m:
            return k
    return None

# Short instructions
INSTRUCTIONS = {
    "en": f"Answer ONLY from the retrieved context. Use short bullets. If context is insufficient or irrelevant, output {NO_CONTEXT_TOKEN} exactly.",
    "zh-HK": f"只可根據檢索內容作答。用精簡要點。若內容不足或無關，請輸出 {NO_CONTEXT_TOKEN}。",
    "zh-CN": f"仅按检索内容作答。用精简要点。若内容不足或无关，请输出 {NO_CONTEXT_TOKEN}。"
}

# Optional footer (disabled by default; see SETTINGS.kb_append_staff_footer)
STAFF_HINT = {
    "en": "If you have other questions, a staff member will follow up here.",
    "zh-HK": "如有其他問題，稍後會有職員在此回覆您。",
    "zh-CN": "如有其他问题，稍后会有职员在此回复您。",
}

REFUSAL_PHRASES = [NO_CONTEXT_TOKEN.lower()]
APOLOGY_MARKERS = ["sorry","i am unable","i'm unable","i cannot","i can't","抱歉","很抱歉","對不起","对不起"]

# Tiny in-memory response cache
_CACHE: Dict[Tuple[str,str], Tuple[float, str, List[Dict], Dict[str,Any]]] = {}
_CACHE_TTL_SECS = int(os.environ.get("KB_RESPONSE_CACHE_TTL_SECS", "120"))

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"): return "zh-HK"
    if l.startswith("zh-cn") or l == "zh": return "zh-CN"
    return "en"

def _prompt_prefix(lang: str) -> str:
    return f"{INSTRUCTIONS.get(lang, INSTRUCTIONS['en'])}\n\n"

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
    key = (lang, (message or "").strip())
    now = time.time()
    entry = _CACHE.get(key)
    if not entry: return None
    ts, ans, cits, dbg = entry
    if now - ts > _CACHE_TTL_SECS:
        _CACHE.pop(key, None)
        return None
    return ans, cits, dbg

def _cache_set(lang: str, message: str, ans: str, cits: List[Dict], dbg: Dict[str,Any]):
    _CACHE[(lang, (message or "").strip())] = (time.time(), ans, cits, dbg)

def chat_with_kb(
    message: str,
    language: Optional[str] = None,
    session_id: Optional[str] = None,
    debug: bool = False,
    extra_context: Optional[str] = None,  # optional tool/context injection
) -> Tuple[str, List[Dict], Dict[str, Any]]:
    L = _lang_label(language)

    cached = _cache_get(L, message)
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
        "attempts": [],
        "silenced": False,
        "silence_reason": None,
        "latency_ms": None,
    }
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        debug_info["error"] = "KB_ID or KB_MODEL_ARN not configured"
        return "", [], debug_info

    t0 = time.time()
    prefix = _prompt_prefix(L)
    injected = ""
    if extra_context and extra_context.strip():
        injected = f"Additional verified context (non-retrieved):\n{extra_context.strip()}\n\n"
    input_text = prefix + injected + "User: " + (message or "")

    # Build vector search config with language (and canonical, if detected)
    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if not SETTINGS.kb_disable_lang_filter:
        lang_filter = {"equals": {"key": "language", "value": L}}
    else:
        lang_filter = None

    canonical = _detect_canonical(message or "")
    debug_info["applied_canonical_filter"] = canonical or None

    def build_filter(include_canonical: bool = True):
        if canonical and lang_filter and include_canonical:
            return {"andAll": [lang_filter, {"equals": {"key": "canonical", "value": canonical}}]}
        if lang_filter:
            return lang_filter
        if canonical and include_canonical:
            return {"equals": {"key": "canonical", "value": canonical}}
        return None

    kb_filter = build_filter(include_canonical=True)
    if kb_filter:
        vec_cfg["filter"] = kb_filter

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
        parsed = _parse_citations(resp.get("citations", []) or [])

        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))

        # Optional retry without the canonical filter (and optionally without lang filter)
        if (reason or not answer) and (canonical or (not SETTINGS.kb_disable_lang_filter and SETTINGS.kb_retry_nofilter)):
            kb_conf = base_req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
            # Drop canonical constraint first
            f1 = build_filter(include_canonical=False)
            if f1:
                kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = f1
            else:
                kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"].pop("filter", None)
            resp2 = rag.retrieve_and_generate(**base_req)
            answer2 = ((resp2.get("output") or {}).get("text") or "").strip()
            parsed2 = _parse_citations(resp2.get("citations", []) or [])
            answer2 = _filter_refusal(answer2)
            reason2 = _silence_reason(answer2, len(parsed2))
            if not reason2 and answer2:
                answer, parsed, reason = answer2, parsed2, None

        # Optional footer (disabled by default)
        if answer and not reason and SETTINGS.kb_append_staff_footer:
            footer = STAFF_HINT.get(L, STAFF_HINT["en"])
            answer = f"{answer}\n\n{footer}"

        debug_info["latency_ms"] = int((time.time() - t0) * 1000)
        if reason:
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason
            _cache_set(L, message, "", [], debug_info)
            return "", [], (debug_info if debug else {})
        _cache_set(L, message, answer, parsed, debug_info)
        return answer, parsed, (debug_info if debug else {})

    except Exception as e:
        debug_info["error"] = f"{type(e).__name__}: {e}"
        return "", [], (debug_info if debug else {})

def debug_retrieve_only(message: str, language: Optional[str] = None, canonical: Optional[str] = None, doc_type: Optional[str] = None, nofilter: bool = False) -> Dict[str, Any]:
    L = _lang_label(language)
    info: Dict[str, Any] = {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id[:12] + "…" if SETTINGS.kb_id else "",
        "model": SETTINGS.kb_model_arn.split("/")[-1] if SETTINGS.kb_model_arn else "",
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "message_chars": len(message or ""),
        "latency_ms": None,
        "error": None,
        "citations": [],
        "retrieval_config": None,
    }
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        info["error"] = "KB_ID or KB_MODEL_ARN not configured"
        return info

    t0 = time.time()
    input_text = _prompt_prefix(L) + "User: " + (message or "")

    # Build optional filter
    f = None
    if not nofilter and not SETTINGS.kb_disable_lang_filter:
        f = {"equals": {"key": "language", "value": L}}
    if doc_type:
        tfil = {"equals": {"key": "type", "value": doc_type}}
        f = {"andAll": [f, tfil]} if f else tfil
    if canonical:
        cfil = {"equals": {"key": "canonical", "value": canonical}}
        f = {"andAll": [f, cfil]} if f else cfil

    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if f:
        vec_cfg["filter"] = f
    info["retrieval_config"] = vec_cfg

    req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
                "generationConfiguration": {
                    "inferenceConfig": {
                        "textInferenceConfig": {"maxTokens": 1, "temperature": 0.0, "topP": 0.9}
                    }
                },
            }
        },
    }

    try:
        resp = rag.retrieve_and_generate(**req)
        info["citations"] = _parse_citations(resp.get("citations", []) or [])
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info