from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb, debug_retrieve_only
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language

# NEW: Opening-hours intent and tool
try:
    from llm.intent import detect_opening_hours_intent, mentions_weather
    from llm.opening_hours import compute_opening_answer
except Exception:
    detect_opening_hours_intent = None
    mentions_weather = None
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
    is_intent, dbg = detect_opening_hours_intent(message, lang, SETTINGS.opening_hours_use_llm_intent)
    if not is_intent:
        return None
    ans = compute_opening_answer(message, lang)
    if not ans:
        return None
    # Append a short severe-weather policy pointer if weather terms are mentioned
    try:
        if mentions_weather and mentions_weather(message or ""):
            if lang.lower().startswith("zh-hk"):
                ans += "\n注意：惡劣天氣安排視乎當時信號。黑雨或八號風球停課；其他情況照常。如有需要請聯絡職員。"
            elif lang.lower().startswith("zh-cn") or lang.lower() == "zh":
                ans += "\n注意：恶劣天气安排取决于当时信号。黑雨或八号风球停课；其他情况照常。如有需要请联系职员。"
            else:
                ans += "\nNote: Severe-weather arrangements depend on actual signals. Classes are suspended under Black Rain or Typhoon Signal No. 8; otherwise we operate. Contact us if needed."
    except Exception:
        pass
    return ans

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

    # Try opening-hours tool, but DO NOT block KB
    tool_answer = _maybe_answer_opening_hours(req.message, lang)

    # Send tool context to the LLM as additional prompt material
    answer, citations, debug_info = chat_with_kb(
        req.message, lang, req.session_id, debug=bool(req.debug), extra_context=tool_answer
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
            note_hdr = "（營業時間／上課安排提示）"
        elif lang.lower().startswith("zh-cn") or lang.lower() == "zh":
            note_hdr = "（营业时间／上课安排提示）"
        else:
            note_hdr = "(Opening-hours / class-arrangement note)"
        answer = f"{answer}\n\n{note_hdr}\n{tool_answer}"
        appended_tool = True

    # Attach detected language and tool flags to debug payload if present
    if debug_info is not None:
        debug_info = dict(debug_info)
        debug_info["detected_language"] = lang
        if tool_answer:
            debug_info["opening_hours_tool"] = True
            debug_info["tool_appended_or_fallback"] = appended_tool

    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

class RetrieveDebugRequest(BaseModel):
    message: str
    language: Optional[str] = None
    session_id: Optional[str] = None

@router.post("/_debug_retrieve")
def debug_retrieve(req: RetrieveDebugRequest, request: Request) -> Dict[str, Any]:
    if not SETTINGS.debug_kb:
        raise HTTPException(status_code=403, detail="DEBUG_KB is disabled.")
    # Same language selection path as chat()
    lang = req.language or (get_session_language(req.session_id) if req.session_id else None)
    if not lang:
        lang = detect_language(req.message, accept_language=request.headers.get("accept-language"))
    info = debug_retrieve_only(req.message, lang)
    return {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id,
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "query": req.message,
        "detected_language": lang,
        "retrieve": info,
    }