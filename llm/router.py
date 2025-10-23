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
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """
    Chat endpoint with robust language detection pipeline.
    
    Language selection order:
    1. Explicit req.language if provided
    2. Session memory (if req.session_id and previously detected)
    3. detect_language(req.message, Accept-Language header)
    """
    # Determine language
    if req.language:
        lang = req.language
    else:
        # Try session memory first
        lang = get_session_language(req.session_id)
        if not lang:
            # Detect from message and Accept-Language header
            accept_lang = request.headers.get("Accept-Language")
            lang = detect_language(req.message, accept_lang)
    
    # Remember language for this session
    remember_session_language(req.session_id, lang)
    
    # Call Bedrock KB
    answer, citations, debug_info = chat_with_kb(req.message, lang, req.session_id, debug=bool(req.debug))
    
    # Add detected language to debug info if debug mode
    if req.debug and debug_info:
        debug_info["detected_language"] = lang
    
    if not answer:
        answer = "Sorry, I am unable to assist you with this request."
    
    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

class RetrieveDebugRequest(BaseModel):
    message: str
    language: Optional[str] = None

@router.post("/_debug_retrieve")
async def debug_retrieve(req: RetrieveDebugRequest, request: Request) -> Dict[str, Any]:
    """
    Debug endpoint to inspect KB retrieval without generation.
    Uses the same language detection strategy as the main chat endpoint.
    """
    if not SETTINGS.debug_kb:
        raise HTTPException(status_code=403, detail="DEBUG_KB is disabled.")
    
    # Detect language using the same strategy
    if req.language:
        lang = req.language
    else:
        accept_lang = request.headers.get("Accept-Language")
        lang = detect_language(req.message, accept_lang)
    
    return {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id,
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "detected_language": lang,
        "query": req.message,
        "retrieve": debug_retrieve_only(req.message, lang),
    }