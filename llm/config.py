import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Source (ContentStore)
    info_sheet_catalog_url: str = os.environ.get("INFO_SHEET_CATALOG_URL", "")

    # AWS / Bedrock
    aws_region: str = os.environ.get("AWS_REGION", "ap-northeast-1")
    kb_id: str = os.environ.get("KB_ID", "")
    kb_model_arn: str = os.environ.get("KB_MODEL_ARN", "")  # e.g. qwen.qwen3-32b-v1:0 ARN

    # S3 for KB data source
    kb_s3_bucket: str = os.environ.get("KB_S3_BUCKET", "")
    kb_s3_prefix: str = os.environ.get("KB_S3_PREFIX", "ls/kb/v1/").strip("/")

    # Inference config for generator (Qwen/others)
    gen_max_tokens: int = int(os.environ.get("KB_GEN_MAX_TOKENS", "800"))
    gen_temperature: float = float(os.environ.get("KB_GEN_TEMPERATURE", "0.2"))
    gen_top_p: float = float(os.environ.get("KB_GEN_TOP_P", "0.9"))

    # App behavior
    default_languages = ("en", "zh-HK", "zh-CN")

SETTINGS = Settings()