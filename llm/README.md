# LLM RAG (Bedrock Knowledge Base) – Getting Started

This adds a Bedrock Knowledge Base (RAG) flow alongside Dialogflow. It reuses your Google Sheets–backed ContentStore and emits multilingual, topic-scoped Markdown docs to S3 for ingestion by a Bedrock Knowledge Base.

## What you’ll set up

1) AWS prerequisites
- Enable Amazon Bedrock in your region (recommended: ap-northeast-1 or us-east-1).
- Model access: Claude 3.5 Sonnet (for generation) and Titan Embeddings G1 – Text v2 (for KB).
- Create an S3 bucket (e.g., `s3://little-scholars-kb/`) with a prefix (e.g., `ls/kb/v1/`).
- Create a Knowledge Base that indexes the S3 prefix above, using Titan Embeddings v2.
  - Chunk size ~ 800–1200 chars, overlap 100–200 works well for chat Q&A.
  - Note the KnowledgeBaseId.

2) App environment
Set these env vars where your app/ingestor will run:
- INFO_SHEET_CATALOG_URL = <your published catalog CSV>  (already used by ContentStore)
- AWS_REGION = <e.g., ap-northeast-1>
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or use a role on the runner)
- KB_ID = <your Bedrock KnowledgeBaseId>
- KB_MODEL_ARN = <model ARN> (Claude 3.5 Sonnet recommended)
  Example: arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0
- KB_S3_BUCKET = <your bucket name>
- KB_S3_PREFIX = ls/kb/v1/

3) Install deps
- Ensure your image/venv has boto3 installed.

4) Ingest docs to S3
- Run: `python -m llm_agent.ingest_bedrock_kb`
- This will upload Markdown files like:
  - s3://.../ls/kb/v1/zh-HK/courses/Phonics.md
  - s3://.../ls/kb/v1/en/policies/absence_makeup.md
- Each object is tagged with language, type, and canonical (when relevant).

5) Hook up chat
- Include the router:
  - In `main.py`: `from llm_agent.router import router as llm_router` and `app.include_router(llm_router)`
- POST /chat with `{ "message": "...", "language": "zh-hk" }`.

## Language strategy

- We generate separate docs per language (en, zh-HK, zh-CN).
- Retrieval filters to the user’s language via metadata to keep answers consistent.
- If no docs are found for a language, you can fall back to English by omitting the filter.

## Testing

- After ingestion, in AWS console → Bedrock → Knowledge Bases, “Sync” the KB if auto-sync is off.
- Hit: `curl -X POST https://<your-domain>/chat -H "Content-Type: application/json" -d '{"message":"請問你們有Phonics課程嗎？適合幾歲的小朋友","language":"zh-hk"}'`
- You should receive a grounded answer with optional citations.

## Notes

- The ingestor pulls from your live Google Sheets via ContentStore (INFO_SHEET_CATALOG_URL).
- You can schedule ingestion (nightly) or trigger after sheet updates.
- Bedrock KB RnG supports metadata filters. This starter uses S3 object tags (language/type/course) so you can filter on `language`.