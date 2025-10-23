from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb  # only import the required symbol
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language

# Optional raw-retrieval helper (prefer ingest_bedrock_kb implementation)
try:
    from llm.ingest_bedrock_kb import debug_retrieve_only as _kb_debug_retrieve_only  # type: ignore
except Exception:
    _kb_debug_retrieve_only = None

# Opening-hours intent and tool
try:
    from llm.intent import detect_opening_hours_intent
    from llm.opening_hours import compute_opening_answer
except Exception:
    detect_opening_hours_intent = None
    compute_opening_answer = None

router = APIRouter(prefix="/chat", tags=["LLM Chat (Bedrock KB)"])

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None  # "en" | "zh-hk" | "zh-cn"
    session_id: Optional[str] = None
    debug: Optional[bool] = False   # return debug object in response

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    debug: Optional[Dict[str, Any]] = None

def _maybe_answer_opening_hours(message: str, lang: str) -> Optional[str]:
    if not SETTINGS.opening_hours_enabled:
        return None
    if not compute_opening_answer or not detect_opening_hours_intent:
        return None
    is_intent, _ = detect_opening_hours_intent(message, lang, SETTINGS.opening_hours_use_llm_intent)
    if not is_intent:
        return None
    # Compute deterministic answer. Weather hint (if any) is appended inside.
    return compute_opening_answer(message, lang)

def _opening_hours_keywords(lang: str) -> List[str]:
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return ["營業時間","開放時間","有冇開","星期日","公眾假期","上堂","上課","安排","颱風","黑雨","八號風球","明天","聽日"]
    if L.startswith("zh-cn") or L == "zh":
        return ["营业时间","开放时间","开门","周日","公众假期","上课","安排","台风","黑雨","八号风球","明天"]
    return ["opening hours","hours","open","closed","Sunday","public holiday","tomorrow","typhoon","rainstorm"]

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    # 1) explicit override
    lang = req.language
    # 2) session stickiness
    if not lang and req.session_id:
        lang = get_session_language(req.session_id)
    # 3) robust detection using Accept-Language header as hint
    if not lang:
        lang = detect_language(req.message, accept_language=request.headers.get("accept-language"))
    # 4) remember choice for this session
    if req.session_id and lang:
        remember_session_language(req.session_id, lang)

    # Try opening-hours tool; do NOT block KB
    tool_answer = _maybe_answer_opening_hours(req.message, lang)

    # If the intent is opening-hours, inject strong keywords and a canonical hint to boost recall
    extra_keywords: Optional[List[str]] = None
    hint_canonical: Optional[str] = None
    if tool_answer:
        extra_keywords = _opening_hours_keywords(lang or "")
        hint_canonical = "opening_hours"

    # Send tool context + retrieval hints to the LLM
    answer, citations, debug_info = chat_with_kb(
        req.message,
        lang,
        req.session_id,
        debug=bool(req.debug),
        extra_context=tool_answer,
        extra_keywords=extra_keywords,
        hint_canonical=hint_canonical,
    )
    answer = answer or ""

    # If LLM had no answer but tool did, fall back to tool’s deterministic answer
    appended_tool = False
    if not answer and tool_answer:
        answer = tool_answer
        appended_tool = True
    elif answer and tool_answer:
        # Append a short localized note so users see both
        if lang.lower().startswith("zh-hk"):
            note_hdr = "（營業時間／上課安排）"
        elif lang.lower().startswith("zh-cn") or lang.lower() == "zh":
            note_hdr = "（营业时间／上课安排）"
        else:
            note_hdr = "(Opening hours / class arrangement)"
        answer = f"{answer}\n\n{note_hdr}\n{tool_answer}"
        appended_tool = True

    if debug_info is not None:
        debug_info = dict(debug_info)
        debug_info["detected_language"] = lang
        if tool_answer:
            debug_info["opening_hours_tool"] = True
            debug_info["tool_appended_or_fallback"] = appended_tool
            debug_info["opening_hours_keywords_injected"] = True

    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

@router.get("/debug-retrieve")
def debug_retrieve(message: str, language: Optional[str] = None):
    """
    Admin probe: return raw citations from Bedrock RAG for a message, without generation.
    """
    if not language:
        # Minimal language detection just like /chat
        language = detect_language(message)
    if not _kb_debug_retrieve_only:
        raise HTTPException(status_code=501, detail="debug_retrieve_only not available in this build")
    info = _kb_debug_retrieve_only(message, language)
    info["detected_language"] = language
    return info