"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Notes:
- We filter by language via metadata. This sample uses S3 object tags (language=...), which are
  available to KB filters in most regions. If your KB uses different metadata mapping, adjust the filter.
"""
import os
import uuid
import boto3
from typing import Optional, Tuple, List, Dict
from llm_agent.config import SETTINGS

# Bedrock Agent Runtime for RetrieveAndGenerate
rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region)

# Simple instruction to steer style; RnG combines this with retrieved context automatically.
INSTRUCTIONS = {
    "en": "Answer ONLY using the retrieved context. If unsure, ask a short clarifying question or share staff contact. Be concise, use bullet points for lists.",
    "zh-HK": "請只根據檢索到的內容回答。如不確定，請先提出簡短問題澄清，或提供職員聯絡。請精簡、列表呈現。",
    "zh-CN": "请只根据检索到的内容回答。如不确定，请先提出简短问题澄清，或提供职员联系。请简洁、用要点列出。"
}

STAFF = {
    "en": "If needed, contact our staff: Phone +852 2537 9519; WhatsApp +852 5118 2819; Email info@decoders-ls.com",
    "zh-HK": "如有需要，請聯絡職員：電話 +852 2537 9519；WhatsApp +852 5118 2819；電郵 info@decoders-ls.com",
    "zh-CN": "如有需要，请联系职员：电话 +852 2537 9519；WhatsApp +852 5118 2819；电邮 info@decoders-ls.com",
}

def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"):
        return "zh-HK"
    if l.startswith("zh-cn") or l == "zh":
        return "zh-CN"
    return "en"

def _prompt_prefix(lang: str) -> str:
    return f"{INSTRUCTIONS.get(lang, INSTRUCTIONS['en'])}\n\n{STAFF.get(lang, STAFF['en'])}\n"

def chat_with_kb(message: str, language: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
    """
    Calls RetrieveAndGenerate against the configured Knowledge Base.
    Attempts to filter to the user's language. Returns (answer_text, citations).
    """
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        raise RuntimeError("KB_ID or KB_MODEL_ARN not configured")

    lang = _lang_label(language)
    input_text = _prompt_prefix(lang) + "\nUser: " + message

    # Build optional language filter.
    # Depending on your KB setup, you may need to adapt this to your metadata mapping.
    # This example assumes your KB exposes S3 object tags as metadata and supports equals filter on 'language'.
    vector_search_cfg: Dict = {
        "numberOfResults": 8
    }
    # Try to filter by language; fall back to no filter if your KB doesn't support it.
    vector_search_cfg["filter"] = {"equals": {"key": "language", "value": lang}}

    req = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": vector_search_cfg
                }
            }
        }
    }
    if session_id:
        req["sessionId"] = session_id
    else:
        req["sessionId"] = str(uuid.uuid4())

    resp = rag.retrieve_and_generate(**req)

    # Extract answer text and citations (if available)
    answer = resp.get("output", {}).get("text", "") or ""
    cits_raw = resp.get("citations", []) or []
    citations: List[Dict] = []
    for c in cits_raw:
        for ref in c.get("retrievedReferences", []):
            citations.append({
                "uri": ref.get("location", {}).get("s3Location", {}).get("uri") or ref.get("location"),
                "score": ref.get("score"),
                "metadata": ref.get("metadata", {})
            })
    return answer.strip(), citations