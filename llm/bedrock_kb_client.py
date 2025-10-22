"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Notes:
- We filter by language via metadata when enabled. If your KB doesn't map S3 tags to retrievable
  attributes yet, disable the filter (KB_DISABLE_LANG_FILTER=true) or map tags in the console.
- Robust citation parsing handles slight schema differences across regions/versions.
"""
import boto3
from typing import Optional, Tuple, List, Dict
from llm.config import SETTINGS

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

def _norm_uri(loc: Dict) -> Optional[str]:
    if not loc:
        return None
    if isinstance(loc, str):
        return loc

    # Common shapes:
    # 1) {"s3Location":{"uri":"s3://bucket/key"}}
    # 2) {"type":"S3","s3Location":{"bucketName":"bucket","key":"key"}}
    # 3) {"s3Location":{"bucketArn":"arn:aws:s3:::bucket","key":"key"}}
    s3 = loc.get("s3Location") or loc.get("S3Location") or {}
    if isinstance(s3, dict):
        # Direct uri present
        if s3.get("uri"):
            return s3["uri"]
        # Build uri from bucket + key (or bucketArn + key)
        bucket = (
            s3.get("bucketName")
            or s3.get("bucket")
            or s3.get("Bucket")
            or s3.get("bucketArn")
        )
        key = s3.get("key") or s3.get("objectKey") or s3.get("Key") or s3.get("path")
        if bucket:
            # If it’s an ARN, extract the name after ':::'
            if isinstance(bucket, str) and "arn:aws:s3:::" in bucket:
                bucket = bucket.split(":::")[-1]
        if bucket and key:
            return f"s3://{bucket}/{key}"

    # Sometimes bucket/key live at top level with type hint
    if loc.get("type") == "S3":
        bucket = loc.get("bucketName") or loc.get("bucket")
        key = loc.get("key") or loc.get("objectKey")
        if bucket and key:
            return f"s3://{bucket}/{key}"

    # Last resort
    return loc.get("uri")

def _parse_citations(cits_raw: List[Dict]) -> List[Dict]:
    citations: List[Dict] = []
    for c in cits_raw or []:
        refs = c.get("retrievedReferences") or c.get("references") or []
        for ref in refs or []:
            uri = _norm_uri(ref.get("location") or {})
            citations.append({
                "uri": uri,
                "score": ref.get("score"),
                "metadata": ref.get("metadata", {}) or {}
            })
    return [c for c in citations if c.get("uri")]

def _apply_no_filter(req: Dict) -> Dict:
    nf = dict(req)
    kb_conf = dict(nf["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"])
    retr_conf = dict(kb_conf.get("retrievalConfiguration", {}))
    vec_conf = dict(retr_conf.get("vectorSearchConfiguration", {}))
    vec_conf.pop("filter", None)
    retr_conf["vectorSearchConfiguration"] = vec_conf
    kb_conf["retrievalConfiguration"] = retr_conf
    nf["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"] = kb_conf
    return nf

def chat_with_kb(message: str, language: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        raise RuntimeError("KB_ID or KB_MODEL_ARN not configured")

    lang = _lang_label(language)
    input_text = _prompt_prefix(lang) + "\nUser: " + message

    vec_cfg: Dict = {"numberOfResults": 8}
    if not SETTINGS.kb_disable_lang_filter:
        vec_cfg["filter"] = {"equals": {"key": "language", "value": lang}}

    base_req: Dict = {
        "input": {"text": input_text},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": vec_cfg
                }
            }
        }
    }
    if session_id:
        base_req["sessionId"] = session_id

    resp = rag.retrieve_and_generate(**base_req)
    answer = (resp.get("output", {}) or {}).get("text", "") or ""
    citations = _parse_citations(resp.get("citations", []) or [])

    if not citations and not SETTINGS.kb_disable_lang_filter:
        resp2 = rag.retrieve_and_generate(**_apply_no_filter(base_req))
        answer2 = (resp2.get("output", {}) or {}).get("text", "") or ""
        cits2 = _parse_citations(resp2.get("citations", []) or [])
        if answer2:
            answer, citations = answer2, cits2

    return answer.strip(), citations