"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Notes:
- We filter by language via metadata when enabled. If your KB doesn't map S3 tags to retrievable
  attributes yet, disable the filter (KB_DISABLE_LANG_FILTER=true) or map tags in the console.
- Robust citation parsing handles slight schema differences across regions/versions.
- Optional debug: set DEBUG_KB=true to emit server logs; pass debug=true to /chat to receive a debug object.
"""
import boto3
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS

rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region)

INSTRUCTIONS = {
    "en": "Answer ONLY using the retrieved context. If unsure, ask a short clarifying question or share staff contact. Be concise, use bullet points for lists.",
    "zh-HK": "請只根據檢索到的內容回答。如不確定，請先提出簡短問題澄清，或提供職員聯絡。請精簡、列表呈現。",
    "zh-CN": "请只根据检索到的内容回答。如不确定，请先提出简短问题澄清，或提供职员联系。请简洁、用要点列出。"
}

STAFF = {
    "en": "If needed, contact our staff: Phone +852 2537 9519; WhatsApp +852 5118 2819; Email info@decoders-ls.com",
    "zh-HK": "如有需要，請聯絡職員：電話 +852 2537 9519；WhatsApp +852 5118 2819；電郵 info@decoders-ls.com",
    "zh-CN": "如有需要，请联系职员：电话 +852 2537 9519；WhatsApp +852 5118 2819；电邮 info@decoders-ls.com",
}

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

    # Common shapes:
    # 1) {"s3Location":{"uri":"s3://bucket/key"}}
    # 2) {"type":"S3","s3Location":{"bucketName":"bucket","key":"key"}}
    # 3) {"s3Location":{"bucketArn":"arn:aws:s3:::bucket","key":"key"}}
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
            # Never let logging break the flow
            pass

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
        "attempts": []
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

    vec_cfg: Dict = {"numberOfResults": 8}
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
        attempt = {
            "used_filter": "filter" in vec_cfg,
            "raw_citation_blocks": len(raw_cits),
            "parsed_citations": len(parsed),
            "first_uris": [c.get("uri") for c in parsed[:3]]
        }
        debug_info["attempts"].append(attempt)
        _maybe_log("response.first_attempt", attempt)

        # Fallback without filter if no citations and filter was used
        if not parsed and not SETTINGS.kb_disable_lang_filter:
            resp2 = rag.retrieve_and_generate(**_apply_no_filter(base_req))
            answer2 = (resp2.get("output", {}) or {}).get("text", "") or ""
            raw2 = resp2.get("citations", []) or []
            parsed2 = _parse_citations(raw2)
            attempt2 = {
                "used_filter": False,
                "raw_citation_blocks": len(raw2),
                "parsed_citations": len(parsed2),
                "first_uris": [c.get("uri") for c in parsed2[:3]]
            }
            debug_info["attempts"].append(attempt2)
            _maybe_log("response.second_attempt", attempt2)
            if answer2:
                return answer2.strip(), parsed2, (debug_info if (debug or SETTINGS.debug_kb) else {})

        return answer.strip(), parsed, (debug_info if (debug or SETTINGS.debug_kb) else {})

    except Exception as e:
        debug_info["error"] = f"{type(e).__name__}: {e}"
        _maybe_log("exception", debug_info["error"])
        # Return empty answer so the caller can decide how to format
        return "", [], (debug_info if (debug or SETTINGS.debug_kb) else {})