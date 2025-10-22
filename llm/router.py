from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb, debug_retrieve_only
from llm.config import SETTINGS

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

def _guess_lang(s: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return "zh-HK"
    return "en"

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    lang = req.language or _guess_lang(req.message)
    answer, citations, debug_info = chat_with_kb(req.message, lang, req.session_id, debug=bool(req.debug))
    if not answer:
        answer = "Sorry, I am unable to assist you with this request."
    return ChatResponse(answer=answer, citations=citations, debug=(debug_info or None))

class RetrieveDebugRequest(BaseModel):
    message: str
    language: Optional[str] = None

@router.post("/_debug_retrieve")
def debug_retrieve(req: RetrieveDebugRequest) -> Dict[str, Any]:
    if not SETTINGS.debug_kb:
        raise HTTPException(status_code=403, detail="DEBUG_KB is disabled.")
    lang = req.language or _guess_lang(req.message)
    return {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id,
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "query": req.message,
        "retrieve": debug_retrieve_only(req.message, lang),
    }