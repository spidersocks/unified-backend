# Content structure for KB docs (local and Google Drive)

We keep a single, human-friendly structure that mirrors S3:

content/
- en/
  - courses/
    - LanguageArts.md
  - institution/
    - contact.md
    - opening_hours.md
  - policies/
    - refund_transfer.md
  - marketing/
    - promotions.md
  - faq/
    - example.md
- zh-HK/
  - courses/
    - ChineseLanguageArts.md
  - institution/
    - contact.md
    - opening_hours.md
  - policies/
    - absence_makeup.md
  - marketing/
    - promotions.md
  - faq/
    - example.md
- zh-CN/
  - courses/
  - institution/
    - opening_hours.md
  - policies/
  - marketing/
  - faq/

Rules
- File type: Markdown (.md). Put human-friendly text (no mixed-language output inside one file).
- Frontmatter (YAML) at top:
  ---
  language: zh-HK
  type: course            # course | institution | policy | marketing | faq
  canonical: ChineseLanguageArts
  folder: courses         # top-level folder name, e.g., courses | institution | policies | marketing | faq
  aliases: term1; term2   # optional; improves retrieval recall
  ---
  Sidecar .metadata.json is generated from this frontmatter when syncing to S3.

- Canonical is the stable course key or logical name (e.g., contact, absence_makeup).
- Headings: use H1 for the doc title, H2/H3 for sections (e.g., 上課時間 / 收費).

S3 destination
- s3://$KB_S3_BUCKET/$KB_S3_PREFIX/<lang>/(courses|institution|policies|marketing|faq)/*.md
- A matching sidecar file is written next to each Markdown:
  <file>.md.metadata.json (required for KB filters)

Drive structure (optional, same shape)
- A Google Drive folder mirrors content/, e.g.:
  rootFolder/
    zh-HK/
      courses/
        ChineseLanguageArts (Google Doc)
      institution/
        contact (Google Doc)
        opening_hours (Google Doc)
      faq/
        example (Google Doc)

Sync flows
- Local → S3:
  python scripts/sync_kb_from_dir.py
- Google Drive → S3:
  python scripts/sync_kb_from_drive.py --root-folder-id <DRIVE_FOLDER_ID>

After syncing, start a KB ingestion job:
aws bedrock-agent start-ingestion-job --knowledge-base-id $KB_ID --data-source-id $KB_DATA_SOURCE_ID --region $AWS_REGION