"""
Thin client for Bedrock Knowledge Base RetrieveAndGenerate.
"""
import os
import time
import boto3
from botocore.config import Config
from typing import Optional, Tuple, List, Dict, Any
from llm.config import SETTINGS

_boto_cfg = Config(
    connect_timeout=SETTINGS.kb_rag_connect_timeout_secs,
    read_timeout=SETTINGS.kb_rag_read_timeout_secs,
    retries={"max_attempts": SETTINGS.kb_rag_max_attempts, "mode": "standard"},
)

# Use the Runtime client for BOTH retrieve and retrieve_and_generate
rag = boto3.client("bedrock-agent-runtime", region_name=SETTINGS.aws_region, config=_boto_cfg)
sts = boto3.client("sts", region_name=SETTINGS.aws_region, config=_boto_cfg)

NO_CONTEXT_TOKEN = "[NO_CONTEXT]"

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

def _build_filter(language: Optional[str], doc_type: Optional[str], canonical: Optional[str], nofilter: bool):
    L = _lang_label(language)
    if nofilter:
        return None
    parts: List[Dict] = []
    if not SETTINGS.kb_disable_lang_filter:
        parts.append({"equals": {"key": "language", "value": L}})
    if doc_type:
        parts.append({"equals": {"key": "type", "value": doc_type}})
    if canonical:
        parts.append({"equals": {"key": "canonical", "value": canonical}})
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"andAll": parts}

# ========== DEBUG HELPERS ==========

def debug_retrieve_only(message: str, language: Optional[str] = None, canonical: Optional[str] = None, doc_type: Optional[str] = None, nofilter: bool = False) -> Dict[str, Any]:
    """
    retrieve_and_generate with minimal wrapping; returns parsed citations.
    """
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
        "input_preview": (message or "")[:120],
    }
    if not SETTINGS.kb_id or not SETTINGS.kb_model_arn:
        info["error"] = "KB_ID or KB_MODEL_ARN not configured"
        return info

    t0 = time.time()
    f = _build_filter(language, doc_type, canonical, nofilter)
    vec_cfg: Dict = {"numberOfResults": max(1, SETTINGS.kb_vector_results)}
    if f:
        vec_cfg["filter"] = f
    info["retrieval_config"] = dict(vec_cfg)

    # Keep body minimal (match AWS CLI style)
    req: Dict = {
        "input": {"text": (message or "")},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": SETTINGS.kb_id,
                "modelArn": SETTINGS.kb_model_arn,
                "retrievalConfiguration": {"vectorSearchConfiguration": vec_cfg},
            }
        },
    }

    try:
        resp = rag.retrieve_and_generate(**req)
        info["citations"] = _parse_citations(resp.get("citations", []) or [])
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info

def debug_retrieve_agent(message: str, language: Optional[str] = None, canonical: Optional[str] = None, doc_type: Optional[str] = None, nofilter: bool = False) -> Dict[str, Any]:
    """
    Pure retrieval (no generation) using bedrock-agent-runtime.retrieve, to isolate search.
    """
    L = _lang_label(language)
    info: Dict[str, Any] = {
        "region": SETTINGS.aws_region,
        "kb_id": SETTINGS.kb_id[:12] + "…" if SETTINGS.kb_id else "",
        "lang_filter_enabled": not SETTINGS.kb_disable_lang_filter,
        "message_chars": len(message or ""),
        "latency_ms": None,
        "error": None,
        "results": [],
        "retrieval_filter": None,
        "nofilter": bool(nofilter),
        "input_preview": (message or "")[:120],
    }
    if not SETTINGS.kb_id:
        info["error"] = "KB_ID not configured"
        return info

    f = _build_filter(language, doc_type, canonical, nofilter)
    if f:
        info["retrieval_filter"] = f

    t0 = time.time()
    try:
        req = {
            "knowledgeBaseId": SETTINGS.kb_id,
            "retrievalQuery": {"text": (message or "")},
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": max(1, SETTINGS.kb_vector_results)}},
        }
        if f:
            req["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = f
        resp = rag.retrieve(**req)  
        out = []
        for r in resp.get("retrievalResults", []) or []:
            out.append({
                "uri": _norm_uri(r.get("location") or {}),
                "score": r.get("score"),
                "text": (r.get("content", {}) or {}).get("text"),
                "metadata": r.get("metadata") or {},
            })
        info["results"] = out
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["latency_ms"] = int((time.time() - t0) * 1000)
        return info

def aws_whoami() -> Dict[str, Any]:
    out: Dict[str, Any] = {"region": SETTINGS.aws_region}
    try:
        ident = sts.get_caller_identity()
        out.update({
            "account": ident.get("Account"),
            "arn": ident.get("Arn"),
            "userId": ident.get("UserId"),
            "kb_id": SETTINGS.kb_id,
            "kb_model_arn": SETTINGS.kb_model_arn,
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out