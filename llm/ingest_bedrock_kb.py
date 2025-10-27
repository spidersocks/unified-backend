"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
Optimized for latency: fewer retrieved chunks, smaller max tokens, no unconditional retry.
"""
import os
import time
import boto3
from botocore.config import Config
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS

# Configure Bedrock client timeouts + limited retries to reduce tail latency
_boto_cfg = Config(
    connect_timeout=SETTINGS.kb_rag_connect_timeout_secs,
    read_timeout=SETTINGS.kb_rag_read_timeout_secs,
    retries={"max_attempts": SETTINGS.kb_rag_max_attempts, "mode": "standard"},
)
bedrock_agent_client = boto3.client("bedrock-agent", region_name=SETTINGS.aws_region, config=_boto_cfg)


def _lang_label(lang: Optional[str]) -> str:
    l = (lang or "").lower()
    if l.startswith("zh-hk"): return "zh-HK"
    if l.startswith("zh-cn") or l == "zh": return "zh-CN"
    return "en"

def _norm_uri(loc: Dict) -> Optional[str]:
    s3 = loc.get("s3Location") or loc.get("S3Location") or {}
    if isinstance(s3, dict):
        if s3.get("uri"):
            return s3["uri"]
        bucket = s3.get("bucketName") or s3.get("bucket") or s3.get("Bucket") or s3.get("bucketArn")
        key = s3.get("key") or s3.get("objectKey") or s3.get("Key") or s3.get("path")
        if bucket and isinstance(bucket, str) and "arn:aws:s3:::" in bucket:
            bucket = bucket.split(":::")[-1]
        if bucket and key:
            return f"s3://{bucket}/{key}"
    if loc.get("type") == "S3":
        bucket = loc.get("bucketName") or loc.get("bucket")
        key = loc.get("key") or loc.get("objectKey")
        if bucket and key:
            return f"s3://{bucket}/{key}"
    return loc.get("uri")

def _parse_citations(cits_raw: List[Dict]) -> List[Dict]:
    citations: List[Dict] = []
    for c in cits_raw or []:
        refs = c.get("retrievedReferences") or c.get("references") or []
        for ref in refs or []:
            uri = _norm_uri(ref.get("location") or {})
            citations.append({"uri": uri, "score": ref.get("score"), "metadata": ref.get("metadata", {}) or {}})
    return [c for c in citations if c.get("uri")]


def debug_retrieve_only(message: str, language: Optional[str] = None, canonical: Optional[str] = None, doc_type: Optional[str] = None, nofilter: bool = False) -> Dict[str, Any]:
    L = _lang_label(language)
    info: Dict[str, Any] = {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id[:12] + "…" if SETTINGS.kb_id else "",
        "model": SETTINGS.kb_model_arn.split("/")[-1] if SETTINGS.kb_model_arn else "",
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "message_chars": len(message or ""),
        "latency_ms": None,
        "error": None,
        "citations": [],
        "retrieval_config": None,
        "nofilter": bool(nofilter),
    }
    if not SETTINGS.kb_id:
        info["error"] = "KB_ID not configured"
        return info

    t0 = time.time()
    
    # Build optional filter
    filters = []
    if not nofilter:
        if not SETTINGS.kb_disable_lang_filter:
            filters.append({"equals": {"key": "language", "value": L}})
        if doc_type:
            filters.append({"equals": {"key": "type", "value": doc_type}})
        if canonical:
            filters.append({"equals": {"key": "canonical", "value": canonical}})

    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if len(filters) > 1:
        vec_cfg["filter"] = {"andAll": filters}
    elif len(filters) == 1:
        vec_cfg["filter"] = filters[0]
    
    info["retrieval_config"] = vec_cfg

    req: Dict = {
        "knowledgeBaseId": SETTINGS.kb_id,
        "retrievalQuery": {"text": message},
        "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
    }

    try:
        # Use retrieve instead of retrieve_and_generate for a pure retrieval test
        resp = bedrock_agent_client.retrieve(**req)
        raw_cits = resp.get("retrievalResults", []) or []
        # Adapt the citation parsing for the 'retrieve' API shape
        citations: List[Dict] = []
        for res in raw_cits:
            uri = _norm_uri(res.get("location") or {})
            citations.append({
                "uri": uri, 
                "score": res.get("score"), 
                "text": res.get("content", {}).get("text"),
                "metadata": res.get("metadata", {}) or {}
            })
        
        info["citations"] = [c for c in citations if c.get("uri")]
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info