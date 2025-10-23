"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Optimized for latency: fewer retrieved chunks, smaller max tokens, no unconditional retry.
"""
import time
import boto3
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS

rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region)

NO_CONTEXT_TOKEN = "[NO_CONTEXT]"

# Shorter instructions (trim token budget)
INSTRUCTIONS = {
    "en": f"Answer ONLY from the retrieved context. Use short bullets. If context is insufficient or irrelevant, output {NO_CONTEXT_TOKEN} exactly.",
    "zh-HK": f"只可根據檢索內容作答。用精簡要點。若內容不足或無關，請輸出 {NO_CONTEXT_TOKEN}。",
    "zh-CN": f"仅按检索内容作答。用精简要点。若内容不足或无关，请输出 {NO_CONTEXT_TOKEN}。"
}

# Keep staff as a post-processing footer (not in the prompt)
STAFF = {
    "en": "If needed, contact our staff: +852 2537 9519 (Call), +852 5118 2819 (WhatsApp), info@decoders-ls.com",
    "zh-HK": "如需協助，請聯絡職員：+852 2537 9519（致電）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
    "zh-CN": "如需协助，请联系职员：+852 2537 9519（致电）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
}

REFUSAL_PHRASES = [NO_CONTEXT_TOKEN.lower()]
APOLOGY_MARKERS = ["sorry","i am unable","i'm unable","i cannot","i can't","抱歉","很抱歉","對不起","对不起"]

# Tiny in-memory response cache to avoid double work on repeated/duplicate messages
_CACHE: Dict[Tuple[str,str], Tuple[float, str, List[Dict], Dict[str,Any]]] = {}
_CACHE_TTL_SECS = int( os.environ.get("KB_RESPONSE_CACHE_TTL_SECS", "120") )

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"): return "zh-HK"
    if l.startswith("zh-cn") or l == "zh": return "zh-CN"
    return "en"

def _prompt_prefix(lang: str) -> str:
    return f"{INSTRUCTIONS.get(lang, INSTRUCTIONS['en'])}\n\n"

def _norm_uri(loc: Dict) -> Optional[str]:
    # (unchanged)
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
    if any(m in lower for m in APOLOGY_MARKERS): return "apology_marker"
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

def chat_with_kb(message: str, language: Optional[str] = None, session_id: Optional[str] = None, debug: bool = False) -> Tuple[str, List[Dict], Dict[str, Any]]:
    # Serve from cache if available
    L = _lang_label(language)
    cached = _cache_get(L, message or "")
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
    input_text = prefix + "User: " + (message or "")

    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if not SETTINGS.kb_disable_lang_filter:
        vec_cfg["filter"] = {"equals": {"key": "language", "value": L}}

    base_req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
                "generationConfiguration": {
                    "maxTokens": SETTINGS.gen_max_tokens,
                    "temperature": SETTINGS.gen_temperature,
                    "topP": SETTINGS.gen_top_p,
                },
            }
        },
        "responseConfiguration": {"type": "TEXT"},
    }
    if session_id:
        base_req["sessionId"] = session_id

    try:
        resp = rag.retrieve_and_generate(**base_req)
        answer = ((resp.get("output") or {}).get("text") or "").strip()
        raw_cits = resp.get("citations", []) or []
        parsed = _parse_citations(raw_cits)

        # Post-filters
        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))

        # Optional retry without filter (disabled by default for latency)
        if (reason or not answer) and (not SETTINGS.kb_disable_lang_filter) and SETTINGS.kb_retry_nofilter:
            kb_conf = base_req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
            kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"].pop("filter", None)
            resp2 = rag.retrieve_and_generate(**base_req)
            answer2 = ((resp2.get("output") or {}).get("text") or "").strip()
            raw2 = resp2.get("citations", []) or []
            parsed2 = _parse_citations(raw2)
            answer2 = _filter_refusal(answer2)
            reason2 = _silence_reason(answer2, len(parsed2))
            if not reason2 and answer2:
                answer, parsed, reason = answer2, parsed2, None

        # Append staff footer to valid answers (outside the prompt)
        if answer and not reason:
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