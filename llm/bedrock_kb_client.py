"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Notes:
- We filter by language via metadata. This sample uses S3 object tags (language=...),
  which many KB setups expose as retrievable metadata. If your KB doesn't surface
  tags to filters, we automatically fall back to no-filter on empty citations.
- Works with any Bedrock model that supports RnG, including Qwen (qwen.qwen3-32b-v1:0).
"""
import uuid
import boto3
from typing import Optional, Tuple, List, Dict
from llm.config import SETTINGS

# Bedrock Agent Runtime for RetrieveAndGenerate
rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region)

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

def _parse_citations(cits_raw: List[Dict]) -> List[Dict]:
    citations: List[Dict] = []
    for c in cits_raw or []:
        for ref in c.get("retrievedReferences", []) or []:
            citations.append({
                "uri": ref.get("location", {}).get("s3Location", {}).get("uri") or ref.get("location"),
                "score": ref.get("score"),
                "metadata": ref.get("metadata", {})
            })
    return citations

def _apply_no_filter(req: Dict) -> Dict:
    nofilter_req = req.copy()
    kb_conf = dict(nofilter_req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"])
    retr_conf = dict(kb_conf.get("retrievalConfiguration", {}))
    vec_conf = dict(retr_conf.get("vectorSearchConfiguration", {}))
    vec_conf.pop("filter", None)
    retr_conf["vectorSearchConfiguration"] = vec_conf
    kb_conf["retrievalConfiguration"] = retr_conf
    nofilter_req["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"] = kb_conf
    return nofilter_req

def chat_with_kb(message: str, language: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
    """
    End-to-end RetrieveAndGenerate with Bedrock KB.
    Model is selected via SETTINGS.kb_model_arn (set this to Qwen's ARN to use Qwen).
    If the language filter yields no citations, automatically retries without the filter.
    """
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        raise RuntimeError("KB_ID or KB_MODEL_ARN not configured")

    lang = _lang_label(language)
    input_text = _prompt_prefix(lang) + "\nUser: " + message

    # Retrieval with language filter
    vector_search_cfg: Dict = {
        "numberOfResults": 8,
        "filter": {"equals": {"key": "language", "value": lang}}
    }

    base_req: Dict = {
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
        },
        "sessionId": session_id or str(uuid.uuid4())
    }

    resp = rag.retrieve_and_generate(**base_req)
    answer = (resp.get("output", {}) or {}).get("text", "") or ""
    citations = _parse_citations(resp.get("citations", []) or [])

    # Fallback: retry without language filter if nothing cited (metadata not mapped or empty slice)
    if not citations:
        resp2 = rag.retrieve_and_generate(**_apply_no_filter(base_req))
        answer2 = (resp2.get("output", {}) or {}).get("text", "") or ""
        cits2 = _parse_citations(resp2.get("citations", []) or [])
        if answer2:
            answer, citations = answer2, cits2

    return answer.strip(), citations