import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Source (ContentStore)
    info_sheet_catalog_url: str = os.environ.get("INFO_SHEET_CATALOG_URL", "")

    # AWS / Bedrock
    aws_region: str = os.environ.get("AWS_REGION", "ap-northeast-1")
    kb_id: str = os.environ.get("KB_ID", "")
    kb_model_arn: str = os.environ.get("KB_MODEL_ARN", "")

    # S3 for KB data source
    kb_s3_bucket: str = os.environ.get("KB_S3_BUCKET", "")
    kb_s3_prefix: str = os.environ.get("KB_S3_PREFIX", "ls/kb/v1/").strip("/")

    # Inference config for generator
    gen_max_tokens: int = int(os.environ.get("KB_GEN_MAX_TOKENS", "300"))  # lower default
    gen_temperature: float = float(os.environ.get("KB_GEN_TEMPERATURE", "0.15"))
    gen_top_p: float = float(os.environ.get("KB_GEN_TOP_P", "0.9"))

    # Retrieval config
    kb_vector_results: int = int(os.environ.get("KB_VECTOR_RESULTS", "6"))  # lower default
    kb_retry_nofilter: bool = os.environ.get("KB_RAG_RETRY_NOFILTER", "false").lower() in ("1","true","yes")

    # Feature flags
    kb_disable_lang_filter: bool = os.environ.get("KB_DISABLE_LANG_FILTER", "false").lower() in ("1","true","yes")
    kb_require_citation: bool = os.environ.get("KB_REQUIRE_CITATION", "true").lower() in ("1","true","yes")

    # Opening-hours feature flags
    opening_hours_enabled: bool = os.environ.get("OPENING_HOURS_ENABLED", "true").lower() in ("1","true","yes")
    opening_hours_use_llm_intent: bool = os.environ.get("OPENING_HOURS_USE_LLM_INTENT", "true").lower() in ("1","true","yes")

    # Debugging
    debug_kb: bool = os.environ.get("DEBUG_KB", "false").lower() in ("1","true","yes")
    debug_kb_log_prompt: bool = os.environ.get("DEBUG_KB_LOG_PROMPT", "false").lower() in ("1","true","yes")

    # App behavior
    default_languages = ("en", "zh-HK", "zh-CN")

SETTINGS = Settings()