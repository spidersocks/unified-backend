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
        f"Answer ONLY from the retrieved context. Use short bullets. If context is irrelevant, or insufficient to answer the user's question in full and with high confidence, output {NO_CONTEXT_TOKEN} exactly.\n\n"
        "IMPORTANT: Your answer MUST be in English. Do NOT answer in any other language, even if the user input is mixed.\n"
        "If the retrieved context describes sending an enrollment form, ALWAYS append the marker [SEND_ENROLLMENT_FORM] on its own line at the end of your answer.\n"
        "If the context describes sending Blooket instructions or the online game, ALWAYS append the marker [SEND_BLOOKET_PDF] on its own line at the end of your answer."
    ),
    "zh-HK": (
        f"只可根據檢索內容作答。用精簡要點。若內容不足或無關，無法完整且有把握地回答用戶問題，請輸出 {NO_CONTEXT_TOKEN}。\n\n"
        "你必須用繁體中文（香港）作答。嚴禁用英文或簡體中文作答，即使用戶輸入為混合語言亦然。\n"
        "重要：如檢索內容涉及發送入學表格，請務必在答案最後另起一行加上此標記：[SEND_ENROLLMENT_FORM]\n"
        "如涉及發送 Blooket 指引或網上遊戲說明，請務必在答案最後另起一行加上此標記：[SEND_BLOOKET_PDF]"
    ),
    "zh-CN": (
        f"仅按检索内容作答。用精简要点。若内容不足或无关，无法完整且有把握地回答用户问题，请输出 {NO_CONTEXT_TOKEN}。\n\n"
        "你必须用简体中文作答。严禁用英文或繁体中文作答，即使用户输入为混合语言亦然。\n"
        "重要：如检索内容涉及发送入学表格，请务必在答案末尾另起一行加上此标记：[SEND_ENROLLMENT_FORM]\n"
        "如涉及发送 Blooket 指南或在线游戏说明，请务必在答案末尾另起一行加上此标记：[SEND_BLOOKET_PDF]"
    )
}

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

CONTACT_MINIMAL_GUARDRAIL = {
    "en": "If the user is asking for contact details, reply with ONLY phone and email on separate lines. Do not include address, map, or social links unless explicitly requested.",
    "zh-HK": "如用戶詢問聯絡方式，只回覆電話及電郵，各佔一行。除非用戶明確要求，請不要加入地址、地圖或社交連結。",
    "zh-CN": "如用户询问联系方式，只回复电话和电邮，各占一行。除非用户明确要求，请不要加入地址、地图或社交链接。",
}

STAFF = {
    "en": "If needed, contact our staff: +852 2537 9519 (Call), +852 5118 2819 (WhatsApp), info@decoders-ls.com",
    "zh-HK": "如需協助，請聯絡職員：+852 2537 9519（致電）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
    "zh-CN": "如需协助，请联系职员：+852 2537 9519（致电）、+852 5118 2819（WhatsApp）、info@decoders-ls.com",
}

REFUSAL_PHRASES = [NO_CONTEXT_TOKEN.lower()]
APOLOGY_MARKERS = [
    "sorry","i am unable","i'm unable","i cannot","i can't",
    "抱歉","很抱歉","對不起","对不起",
    "無提供相關信息","沒有相關信息","沒有資料","沒有相关资料","暂无相关信息","暂无资料",
]

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

    # Use raw user question for retrieval embedding
    input_text = (message or "")

    # Optional guardrails (kept for generation stage only)
    prefix = _prompt_prefix(L)
    if hint_canonical and hint_canonical.lower() == "opening_hours":
        prefix = f"{prefix}{OPENING_HOURS_WEATHER_GUARDRAIL.get(L, OPENING_HOURS_WEATHER_GUARDRAIL['en'])}\n{OPENING_HOURS_HOLIDAY_GUARDRAIL.get(L, OPENING_HOURS_HOLIDAY_GUARDRAIL['en'])}\n\n"
        if debug:
            debug_info["opening_hours_guardrail"] = True
    if _is_contact_query(message or "", L):
        prefix = f"{prefix}{CONTACT_MINIMAL_GUARDRAIL.get(L, CONTACT_MINIMAL_GUARDRAIL['en'])}\n\n"
        if debug:
            debug_info["contact_guardrail"] = True

    # Language-only filter
    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if not SETTINGS.kb_disable_lang_filter:
        vec_cfg["filter"] = {"equals": {"key": "language", "value": L}}
    debug_info["first_attempt_retrieval_config"] = dict(vec_cfg)
    debug_info["retrieval_config"] = dict(vec_cfg)

    req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
                # Minimal body is enough; omit generationConfiguration unless you need custom decoding
            }
        },
    }
    # if session_id:
    #     req["sessionId"] = session_id

    try:
        resp = rag.retrieve_and_generate(**req)
        answer = ((resp.get("output") or {}).get("text") or "").strip()
        raw_cits = resp.get("citations", []) or []
        parsed = _parse_citations(raw_cits)

        if debug:
            debug_info["raw_citations"] = raw_cits
            debug_info["input_preview"] = input_text[:120]

        answer = _filter_refusal(answer)
        reason = _silence_reason(answer, len(parsed))

        # Optional retry without any filter
        if (reason or not answer) and SETTINGS.kb_retry_nofilter:
            kb_conf = req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
            vec = kb_conf["retrievalConfiguration"]["vectorSearchConfiguration"]
            vec.pop("filter", None)
            debug_info["retry_reason"] = f"Initial attempt failed ({reason or 'empty_answer'}). Retrying without filter."
            debug_info["retry_retrieval_config"] = dict(vec)

            resp2 = rag.retrieve_and_generate(**req)
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

        if answer and not reason and SETTINGS.kb_append_staff_footer:
            answer = f"{answer}\n\n{STAFF.get(L, STAFF['en'])}"

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