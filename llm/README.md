# LLM RAG (Bedrock Knowledge Base) – Using Llama

This module wires FastAPI `/chat` to Amazon Bedrock Knowledge Bases (RAG). It reuses llm.content_store to generate multilingual Markdown into S3, and calls RetrieveAndGenerate with a Bedrock model.

## Choose the Knowledge Base type

- Vector store (service-managed) – recommended
- Embeddings: Titan Embeddings G1 – Text v2
- Chunking: 1000 chars, overlap 150

## Model (generator)

You can use Qwen on Bedrock. Example ARNs:
- ap-northeast-1: `arn:aws:bedrock:ap-northeast-1::foundation-model/qwen.qwen3-32b-v1:0`
- us-east-1: `arn:aws:bedrock:us-east-1::foundation-model/qwen.qwen3-32b-v1:0`

Set via env: `KB_MODEL_ARN`.

Optional generation tuning via env:
- `KB_GEN_TEMPERATURE` (default 0.2)
- `KB_GEN_TOP_P` (default 0.9)
- `KB_GEN_MAX_TOKENS` (default 800)

## Environment

- INFO_SHEET_CATALOG_URL = <your catalog CSV>
- AWS_REGION = ap-northeast-1 (or your region)
- KB_S3_BUCKET = little-scholars-kb
- KB_S3_PREFIX = ls/kb/v1
- KB_ID = <KnowledgeBaseId>
- KB_MODEL_ARN = <Qwen ARN above>
- AWS credentials for the runner (user/role with AmazonBedrockFullAccess for runtime; AmazonS3FullAccess for ingestion only)

## Ingest content

```
python -m llm.ingest_bedrock_kb
```

This uploads:
- `s3://<bucket>/<prefix>/zh-HK/courses/*.md`, `.../policies/*.md`, etc.
- with object tags: `language`, `type`, `canonical`

## Sync KB

In Bedrock console: Knowledge Bases → Sync (or enable automatic sync).

## Run chat

- Ensure `main.py` includes the `llm.router`.
- POST `/chat`:

```
curl -X POST https://<your-domain>/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"中文語文課幾時上堂？收費點樣？","language":"zh-hk"}'
```

## Language strategy

- Separate docs by language (en, zh-HK, zh-CN) and filter retrieval by `language` metadata.
- The client auto-retries without the filter if your KB doesn’t expose S3 tags to filters.