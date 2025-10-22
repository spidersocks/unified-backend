from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb

router = APIRouter(prefix="/chat", tags=["LLM Chat (Bedrock KB)"])

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None  # "en" | "zh-hk" | "zh-cn"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []

def _guess_lang(s: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return "zh-HK"
    return "en"

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    lang = req.language or _guess_lang(req.message)
    answer, citations = chat_with_kb(req.message, lang, req.session_id)
    return ChatResponse(answer=answer, citations=citations)