from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb, debug_retrieve_only
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language

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

    answer, citations, debug_info = chat_with_kb(req.message, lang, req.session_id, debug=bool(req.debug))
    # IMPORTANT: Do NOT replace empty answers. Empty means: emit nothing (admin will answer).
    answer = answer or ""

    # Attach detected language to debug payload if present
    if debug_info is not None:
        debug_info = dict(debug_info)
        debug_info["detected_language"] = lang

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