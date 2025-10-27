from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from llm.bedrock_kb_client import chat_with_kb
from llm.config import SETTINGS
from llm.lang import detect_language, remember_session_language, get_session_language
from llm import tags_index

# Debug helpers
try:
    from llm.ingest_bedrock_kb import debug_retrieve_only, debug_retrieve_agent, aws_whoami
except ImportError:
    debug_retrieve_only = None
    debug_retrieve_agent = None
    aws_whoami = None

router = APIRouter(prefix="/chat", tags=["LLM Chat (Bedrock KB)"])

@router.get("/debug-retrieve")
def debug_retrieve(
    message: str = Query(..., description="User query to probe retrieval+generate"),
    language: str | None = Query(None, description="en | zh-HK | zh-CN"),
    canonical: str | None = Query(None),
    doc_type: str | None = Query(None),
    nofilter: bool = Query(False),
):
    if debug_retrieve_only is None:
        raise HTTPException(status_code=501, detail="Debug retrieval not available")
    try:
        return debug_retrieve_only(
            message=message,
            language=language,
            canonical=canonical,
            doc_type=doc_type,
            nofilter=nofilter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug-retrieve-agent")
def debug_retrieve_agent_endpoint(
    message: str = Query(...),
    language: str | None = Query(None),
    canonical: str | None = Query(None),
    doc_type: str | None = Query(None),
    nofilter: bool = Query(False),
):
    if debug_retrieve_agent is None:
        raise HTTPException(status_code=501, detail="Agent retrieve debug not available")
    try:
        return debug_retrieve_agent(
            message=message,
            language=language,
            canonical=canonical,
            doc_type=doc_type,
            nofilter=nofilter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug-aws")
def debug_aws():
    if aws_whoami is None:
        raise HTTPException(status_code=501, detail="AWS whoami not available")
    return aws_whoami()