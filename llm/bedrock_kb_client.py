"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Notes:
- We filter by language via metadata when enabled.
- Robust citation parsing handles slight schema differences across regions/versions.
- Optional debug: set DEBUG_KB=true to emit server logs; pass debug=true to /chat to receive a debug object.
"""
import boto3
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS

rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region)

# --- NEW STRUCTURED REFUSAL TOKEN ---
NO_CONTEXT_TOKEN = "[NO_CONTEXT]"

INSTRUCTIONS = {
    # Instruction changed to output the specific token instead of an empty string
    "en": f"Strictly answer ONLY using the retrieved context. Be concise (use lists/bullets). If the context is insufficient, or if the question is irrelevant, you MUST output the single token: {NO_CONTEXT_TOKEN}. DO NOT apologize, explain lack of information, or ramble.",
    "zh-HK": f"請嚴格只根據檢索到的內容回答。請精簡（使用列表/要點）。如果內容不足以回答問題，您必須輸出單一標記: {NO_CONTEXT_TOKEN}。請勿道歉、解釋信息不足或冗長回答。",
    "zh-CN": f"请严格只根据检索到的内容回答。请简洁（使用列表/要点）。如果内容不足以回答问题，您必须输出单一标记: {NO_CONTEXT_TOKEN}。请勿道歉、解释信息不足或冗长回答。"
}

STAFF = {
    "en": "If needed, contact our staff: Phone +852 2537 9519; WhatsApp +852 5118 2819; Email info@decoders-ls.com",
    "zh-HK": "如有需要，請聯絡職員：電話 +852 2537 9519；WhatsApp +852 5118 2819；電郵 info@decoders-ls.com",
    "zh-CN": "如有需要，请联系职员：电话 +852 2537 9519；WhatsApp +852 5118 2819；电邮 info@decoders-ls.com",
}

# The list now only contains the structured token, making the filter deterministic.
REFUSAL_PHRASES = [
    NO_CONTEXT_TOKEN.lower(),
]

# NEW: common apology markers to blackhole even if the model ignores instructions
APOLOGY_MARKERS = [
    "sorry",
    "i am unable to assist",
    "i'm unable to assist",
    "i cannot assist",
    "i can't assist",
    "抱歉",
    "很抱歉",
    "對不起",
    "对不起",
]

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"):
        return "zh-HK"
    if l.startswith("zh-cn") or l == "zh":
        return "zh-CN"
    return "en"

def _prompt_prefix(lang: str) -> str:
    return f"{INSTRUCTIONS.get(lang, INSTRUCTIONS['en'])}\n\n{STAFF.get(lang, STAFF['en'])}\n"

def _norm_uri(loc: Dict) -> Optional[str]:
    if not loc:
        return None
    if isinstance(loc, str):
        return loc
    s3 = loc.get("s3Location") or loc.get("S3Location") or {}
    if isinstance(s3, dict):
        if s3.get("uri"):
            return s3["uri"]
        bucket = (
            s3.get("bucketName")
            or s3.get("bucket")
            or s3.get("Bucket")
            or s3.get("bucketArn")
        )
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
            citations.append({
                "uri": uri,
                "score": ref.get("score"),
                "metadata": ref.get("metadata", {}) or {}
            })
    return [c for c in citations if c.get("uri")]

def _apply_no_filter(req: Dict) -> Dict:
    nf = dict(req)
    kb_conf = dict(nf["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"])
    retr_conf = dict(kb_conf.get("retrievalConfiguration", {}))
    vec_conf = dict(retr_conf.get("vectorSearchConfiguration", {}))
    vec_conf.pop("filter", None)
    retr_conf["vectorSearchConfiguration"] = vec_conf
    kb_conf["retrievalConfiguration"] = retr_conf
    nf["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"] = kb_conf
    return nf

def _maybe_log(label: str, payload: Any):
    if SETTINGS.debug_kb:
        try:
            print(f"[KB DEBUG] {label}: {payload}", flush=True)
        except Exception:
            pass

def _filter_refusal(answer: str) -> str:
    """Checks if the answer contains the structured refusal token anywhere and returns an empty string if so."""
    stripped_answer = answer.strip()
    if not stripped_answer:
        return ""
    answer_lower = stripped_answer.lower()
    # Silences if the NO_CONTEXT token appears anywhere in the output
    if any(phrase in answer_lower for phrase in REFUSAL_PHRASES):
        _maybe_log("filter_applied", f"Refusal detected (structured token found): '{stripped_answer}' -> Silence")
        return ""
    return stripped_answer

def _silence_reason(answer: str, parsed_count: int) -> Optional[str]:
    """
    Returns a human-readable reason to silence the answer, or None if it's acceptable.
    Enforces:
      - [NO_CONTEXT] token (anywhere in output)
      - Require at least one parsed citation (configurable)
      - Blackhole apology markers in any language
    """
    stripped = (answer or "").strip()
    if not stripped:
        return "empty"
    lower = stripped.lower()
    # Silences if the NO_CONTEXT token appears anywhere in the output
    if NO_CONTEXT_TOKEN.lower() in lower:
        return "refusal_token"
    if SETTINGS.kb_require_citation and parsed_count == 0:
        return "no_citations"
    if any(marker in lower for marker in APOLOGY_MARKERS):
        return "apology_marker"
    return None

def chat_with_kb(message: str, language: Optional[str] = None, session_id: Optional[str] = None, debug: bool = False) -> Tuple[str, List[Dict], Dict[str, Any]]:
    """
    Returns (answer, citations, debug_info).
    - debug_info is empty unless debug=True or DEBUG_KB=true.
    """
    debug_info: Dict[str, Any] = {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id[:12] + "…" if SETTINGS.kb_id else "",
        "model": SETTINGS.kb_model_arn.split("/")[-1] if SETTINGS.kb_model_arn else "",
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "session_provided": bool(session_id),
        "message_chars": len(message or ""),
        "error": None,
        "attempts": [],
        # NEW: overall silence flag
        "silenced": False,
        "silence_reason": None,
    }

    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        err = "KB_ID or KB_MODEL_ARN not configured"
        debug_info["error"] = err
        return "", [], debug_info

    lang = _lang_label(language)
    prefix = _prompt_prefix(lang)
    input_text = prefix + "\nUser: " + (message or "")
    if SETTINGS.debug_kb_log_prompt:
        _maybe_log("prompt", input_text)
    else:
        _maybe_log("prompt_preview", input_text[:200] + ("…" if len(input_text) > 200 else ""))

    vec_cfg: Dict = {"numberOfResults": 12}
    if not SETTINGS.kb_disable_lang_filter:
        vec_cfg["filter"] = {"equals": {"key": "language", "value": lang}}

    base_req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": vec_cfg
                }
            }
        }
    }
    if session_id:
        base_req["sessionId"] = session_id

    try:
        _maybe_log("request.vectorSearchConfiguration", vec_cfg)
        resp = rag.retrieve_and_generate(**base_req)
        answer = (resp.get("output", {}) or {}).get("text", "") or ""
        raw_cits = resp.get("citations", []) or []
        parsed = _parse_citations(raw_cits)

        # Apply filters
        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))
        attempt = {
            "used_filter": "filter" in vec_cfg,
            "raw_citation_blocks": len(raw_cits),
            "parsed_citations": len(parsed),
            "first_uris": [c.get("uri") for c in parsed[:3]],
            "silenced": bool(reason),
            "silence_reason": reason,
        }
        debug_info["attempts"].append(attempt)
        _maybe_log("response.first_attempt", attempt)

        # If first attempt produced no acceptable answer and we used a filter, retry without filter
        if (reason or not answer) and not SETTINGS.kb_disable_lang_filter:
            resp2 = rag.retrieve_and_generate(**_apply_no_filter(base_req))
            answer2 = (resp2.get("output", {}) or {}).get("text", "") or ""
            raw2 = resp2.get("citations", []) or []
            parsed2 = _parse_citations(raw2)

            answer2 = _filter_refusal(answer2)
            reason2 = _silence_reason(answer2, len(parsed2))
            attempt2 = {
                "used_filter": False,
                "raw_citation_blocks": len(raw2),
                "parsed_citations": len(parsed2),
                "first_uris": [c.get("uri") for c in parsed2[:3]],
                "silenced": bool(reason2),
                "silence_reason": reason2,
            }
            debug_info["attempts"].append(attempt2)
            _maybe_log("response.second_attempt", attempt2)

            if not reason2 and answer2:
                return answer2, parsed2, (debug_info if (debug or SETTINGS.debug_kb) else {})

            # If still silenced, empty out any residual text
            debug_info["silenced"] = True
            debug_info["silence_reason"] = reason2 or "no_answer"
            return "", [], (debug_info if (debug or SETTINGS.debug_kb) else {})

        # First attempt acceptable
        if not reason and answer:
            return answer, parsed, (debug_info if (debug or SETTINGS.debug_kb) else {})

        # First attempt silenced; no retry
        debug_info["silenced"] = True
        debug_info["silence_reason"] = reason or "no_answer"
        return "", [], (debug_info if (debug or SETTINGS.debug_kb) else {})

    except Exception as e:
        debug_info["error"] = f"{type(e).__name__}: {e}"
        _maybe_log("exception", debug_info["error"])
        return "", [], (debug_info if (debug or SETTINGS.debug_kb) else {})

def debug_retrieve_only(message: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Call Bedrock 'retrieve' directly to inspect raw hits from the KB without generation.
    Returns a dict with raw results and normalized URIs.
    """
    lang = _lang_label(language)
    vec_cfg: Dict = {"numberOfResults": 8}
    if not SETTINGS.kb_disable_lang_filter:
        vec_cfg["filter"] = {"equals": {"key": "language", "value": lang}}

    req = {
        "knowledgeBaseId": SETTINGS.kb_id,
        "retrievalQuery": {"text": message},
        "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
    }
    out: Dict[str, Any] = {"used_filter": "filter" in vec_cfg, "raw_results": [], "uris": []}
    try:
        resp = rag.retrieve(**req)
        results = resp.get("retrievalResults", []) or []
        out["raw_results"] = results
        uris: List[str] = []
        for r in results:
            uri = _norm_uri(r.get("location") or {}) or _norm_uri((r.get("content") or {}).get("location") or {})
            if uri:
                uris.append(uri)
        out["uris"] = uris
        _maybe_log("retrieve.debug", {"used_filter": out["used_filter"], "count": len(results), "first_uris": uris[:3]})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        _maybe_log("retrieve.debug.error", out["error"])
    return out