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

# Heuristic sets to distinguish Traditional vs Simplified Chinese
# We only need a small set of “discriminator” characters that differ across scripts.
TRAD_ONLY = set("學體車國廣馬門風愛聽話醫龍書氣媽齡費號聯網臺灣灣課師資簡介聯絡資料")
SIMP_ONLY = set("学体车国广马门风爱听话医龙书气妈龄费号联网台湾湾课师资简介联络资料")

def _contains_cjk(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

def _guess_lang(s: str) -> str:
    """
    - No CJK -> en
    - If CJK, prefer Traditional vs Simplified by counting script-specific chars.
    - Tie/unknown -> default zh-HK (safer for HK audience; adjust if your audience differs).
    """
    if not s or not _contains_cjk(s):
        return "en"

    trad = sum(1 for ch in s if ch in TRAD_ONLY)
    simp = sum(1 for ch in s if ch in SIMP_ONLY)

    if trad > simp:
        return "zh-HK"
    if simp > trad:
        return "zh-CN"
    # If the message is neutral (no discriminator chars), default to HK Traditional
    return "zh-HK"

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