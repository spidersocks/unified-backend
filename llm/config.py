import os
from dataclasses import dataclass, field  # Import 'field'
from typing import List  # Import List

# Helper function to parse the WHATSAPP_TEST_NUMBERS environment variable
def _get_whatsapp_test_numbers_from_env() -> List[str]:
    """Parses the WHATSAPP_TEST_NUMBERS environment variable into a list of strings."""
    env_var = os.environ.get("WHATSAPP_TEST_NUMBERS", "")
    return [num.strip() for num in env_var.split(",") if num.strip()]

@dataclass
class Settings:
    # Source (ContentStore)
    info_sheet_catalog_url: str = os.environ.get("INFO_SHEET_CATALOG_URL", "")

    # AWS / Bedrock
    aws_region: str = os.environ.get("AWS_REGION", "ap-northeast-1")
    kb_id: str = os.environ.get("KB_ID", "")
    kb_model_arn: str = os.environ.get("KB_MODEL_ARN", "")

    # LLM for lightweight tasks (e.g., rephrasing)
    # Avoid using the KB model ARN here; this should be a modelId like "meta.llama3-70b-instruct-v1:0"
    llm_model_id: str = os.environ.get("LLM_MODEL_ID", "meta.llama3-70b-instruct-v1:0")

    # S3 for KB data source
    kb_s3_bucket: str = os.environ.get("KB_S3_BUCKET", "")
    kb_s3_prefix: str = os.environ.get("KB_S3_PREFIX", "ls/kb/v1/").strip("/")

    # Inference config for generator (RAG)
    gen_max_tokens: int = int(os.environ.get("KB_GEN_MAX_TOKENS", "300"))
    gen_temperature: float = float(os.environ.get("KB_GEN_TEMPERATURE", "0.15"))
    gen_top_p: float = float(os.environ.get("KB_GEN_TOP_P", "0.9"))

    # Retrieval config
    kb_vector_results: int = int(os.environ.get("KB_VECTOR_RESULTS", "6"))
    kb_retry_nofilter: bool = os.environ.get("KB_RAG_RETRY_NOFILTER", "false").lower() in ("1","true","yes")

    # Feature flags
    kb_disable_lang_filter: bool = os.environ.get("KB_DISABLE_LANG_FILTER", "false").lower() in ("1","true","yes")
    kb_require_citation: bool = os.environ.get("KB_REQUIRE_CITATION", "false").lower() in ("1","true","yes")
    kb_silence_apology: bool = os.environ.get("KB_SILENCE_APOLOGY", "false").lower() in ("1","true","yes")
    kb_append_staff_footer: bool = os.environ.get("KB_APPEND_STAFF_FOOTER", "false").lower() in ("1","true","yes")

    # Bedrock client timeouts (seconds)
    kb_rag_connect_timeout_secs: int = int(os.environ.get("KB_RAG_CONNECT_TIMEOUT", "5"))
    kb_rag_read_timeout_secs: int = int(os.environ.get("KB_RAG_READ_TIMEOUT", "25"))
    kb_rag_max_attempts: int = int(os.environ.get("KB_RAG_MAX_ATTEMPTS", "2"))

    # Opening-hours feature flags
    opening_hours_enabled: bool = os.environ.get("OPENING_HOURS_ENABLED", "true").lower() in ("1","true","yes")
    opening_hours_use_llm_intent: bool = os.environ.get("OPENING_HOURS_USE_LLM_INTENT", "true").lower() in ("1","true","yes")

    # Weather hints for opening hours
    # Only append hint when severe (Black Rain or Typhoon Signal No. 8+)
    opening_hours_weather_enabled: bool = os.environ.get("OPENING_HOURS_WEATHER_ENABLED", "true").lower() in ("1","true","yes")
    opening_hours_weather_only_severe: bool = os.environ.get("OPENING_HOURS_WEATHER_ONLY_SEVERE", "true").lower() in ("1","true","yes")

    # Debugging
    debug_kb: bool = os.environ.get("DEBUG_KB", "false").lower() in ("1","true","yes")
    debug_kb_log_prompt: bool = os.environ.get("DEBUG_KB_LOG_PROMPT", "false").lower() in ("1","true","yes")

    # App behavior
    default_languages: tuple[str, ...] = ("en", "zh-HK", "zh-CN")  # Added type hint for clarity

    # --- WhatsApp Integration Settings ---
    whatsapp_verify_token: str = os.environ.get("WHATSAPP_VERIFY_TOKEN", "spidersocks")
    whatsapp_access_token: str = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    whatsapp_phone_number_id: str = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
    whatsapp_graph_version: str = os.environ.get("WHATSAPP_GRAPH_VERSION", "v18.0")
    # WHATSAPP_TEST_NUMBERS should be a comma-separated string, e.g., "+1234567890,+1122334455"
    # Use default_factory for mutable default (list)
    whatsapp_test_numbers: List[str] = field(default_factory=_get_whatsapp_test_numbers_from_env)

    # --- Admin cooling configuration ---
    # Seconds to keep the bot silent after detecting an admin/human message
    admin_cooldown_secs: int = int(os.environ.get("ADMIN_COOLDOWN_SECS", "900"))  # default 15 minutes

    # Whatsapp auto delay
    whatsapp_ack_delay_secs : int = int(os.environ.get("WHATSAPP_ACK_DELAY_SECS", "1800"))

    # --- Daily Admin Digest (5pm roundup) ---
    admin_digest_enabled: bool = os.environ.get("ADMIN_DIGEST_ENABLED", "true").lower() in ("1","true","yes")
    admin_digest_director_number: str = os.environ.get("ADMIN_DIGEST_DIRECTOR_NUMBER", "+85295505456")
    admin_digest_hour_local: int = int(os.environ.get("ADMIN_DIGEST_HOUR_LOCAL", "17"))
    admin_digest_minute_local: int = int(os.environ.get("ADMIN_DIGEST_MINUTE_LOCAL", "0"))
    admin_digest_tz: str = os.environ.get("ADMIN_DIGEST_TZ", "Asia/Hong_Kong")
    admin_digest_table: str = os.environ.get("ADMIN_DIGEST_TABLE", "AdminDigestPending")
    admin_digest_max_items: int = int(os.environ.get("ADMIN_DIGEST_MAX_ITEMS", "50"))

SETTINGS = Settings()