from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb  # only import the required symbol
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language
from llm import tags_index

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

def _opening_hours_keywords(lang: str) -> List[str]:
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return ["營業時間","開放時間","有冇開","星期日","公眾假期","上堂","上課","安排","颱風","黑雨","八號風球","明天","聽日"]
    if L.startswith("zh-cn") or L == "zh":
        return ["营业时间","开放时间","开门","几点","周日","公众假期","上课","安排","台风","黑雨","八号风球","明天","下午","今天"]
    return ["opening hours","hours","open","closed","Sunday","public holiday","tomorrow","afternoon","typhoon","rainstorm"]

def _has_opening_markers(message: str, lang: str) -> bool:
    m = (message or "")
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return any(x in m for x in ["營業","開放","開門","幾點","上課","聽日","星期","周日","公眾假期"])
    if L.startswith("zh-cn") or L == "zh":
        return any(x in m for x in ["营业","开放","开门","几点","上课","明天","星期","周日","公众假期"])
    return any(w in m.lower() for w in ["open","opening","hours","closed","sunday","public holiday","tomorrow"])

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

    # Opening-hours: robust trigger + debug
    tool_answer = None
    intent_debug: Optional[Dict[str, Any]] = None
    try:
        if SETTINGS.opening_hours_enabled and compute_opening_answer:
            is_intent = False
            if detect_opening_hours_intent:
                is_intent, intent_debug = detect_opening_hours_intent(req.message, lang, SETTINGS.opening_hours_use_llm_intent)
            # Safety net: if strong markers are present, treat as intent
            if not is_intent and _has_opening_markers(req.message, lang or ""):
                is_intent = True
                if intent_debug is None:
                    intent_debug = {"forced_by_markers": True}
            if is_intent:
                tool_answer = compute_opening_answer(req.message, lang)
    except Exception:
        tool_answer = None

    # If the intent is opening-hours, inject strong keywords and a canonical hint to boost recall
    extra_keywords: Optional[List[str]] = None
    hint_canonical: Optional[str] = None
    if tool_answer or (req.message and _has_opening_markers(req.message, lang or "")):
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
        if intent_debug is not None:
            debug_info["opening_intent_debug"] = intent_debug
        if tool_answer:
            debug_info["opening_hours_tool"] = True
            debug_info["tool_appended_or_fallback"] = appended_tool
            debug_info["opening_hours_keywords_injected"] = True

    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

@router.get("/debug-retrieve")
def debug_retrieve(
    message: str = Query(..., description="User query to probe retrieval"),
    language: str | None = Query(None, description="en | zh-HK | zh-CN"),
    canonical: str | None = Query(None, description="Optional canonical to bias filter"),
    doc_type: str | None = Query(None, description="Optional type (institution|course|policy|marketing|faq)"),
    nofilter: bool = Query(False, description="If true, do not send any metadata filter"),
):
    if _kb_debug_retrieve_only is None:
        raise HTTPException(status_code=501, detail="Debug retrieval not available in this build")
    try:
        return _kb_debug_retrieve_only(
            message=message,
            language=language,
            canonical=canonical,
            doc_type=doc_type,
            nofilter=nofilter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))