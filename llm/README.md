# Little Scholars AI Assistant
Conversational help for parents — across English, Cantonese (zh‑HK), and Mandarin (zh‑CN) — powered by Amazon Bedrock and a curated Knowledge Base.

> “Short, useful answers when the docs cover it. Quiet and respectful hand‑off to humans when it’s an admin task.”


## ✨ What it does
- Answers parent questions from our official documentation:
  - Courses, syllabuses, ages, fees, materials
  - Opening hours and severe weather arrangements
  - Enrollment steps and forms
  - FAQs (e.g., phonics levels, writing pathways, GAPSK overview)
- Sends official documents when appropriate:
  - Enrollment form
  - Blooket homework instructions (PDF)
- Understands three languages:
  - English (en)
  - Traditional Chinese (zh‑HK, Hong Kong)
  - Simplified Chinese (zh‑CN, Mainland)

## 🔒 What it won’t do (by design)
Some requests must be handled by staff. In these cases the assistant stays silent so your team can follow up directly:
- Availability/time‑slot/timetable/start‑date checks
- Dated actions like reschedule/cancel/leave
- “Please tell/ask/notify the teacher…” (pass‑on requests)
- Private 1:1 pricing or quotations
- Student‑specific placement/level/suitability judgements (unless strictly asked about general policy)
- Terminal closings like “You’re welcome / 不客气 / 唔使客氣” (no reply)

Silence means: the system logged it for human follow‑up (WhatsApp auto‑ack may be sent).


---

## 🧭 How it works (in one minute)

1) Retrieve  
The bot queries an Amazon Bedrock Knowledge Base (vector search over our Markdown content). Retrieval is hard‑filtered by language.

2) Generate  
The answer is composed by the Bedrock model (Meta Llama 3 Instruct) using a strict prompt scaffold that enforces tone and formatting.

3) Guardrails  
- If context is insufficient or irrelevant, the model is instructed to output exactly “[NO_ANSWER]”.  
- The backend additionally silences:
  - Apologies/“no info” hedges without citations
  - Admin‑handled topics (availability, dated leave, pass‑on, private fees, placement judgement)
  - Terminal politeness closings

4) Comfort features  
- Deterministic opening‑hours answers when appropriate (with severe‑weather hints from the Hong Kong Observatory)  
- Enrollment and Blooket documents delivered when markers are detected  
- WhatsApp auto‑ack if the bot stays silent (immediately outside hours; short delay during hours)


---

## 🗂️ What’s in the Knowledge Base?
- content/en/... and mirrored content/zh‑HK/... and content/zh‑CN/...
- Each page carries frontmatter like:
  ```yaml
  ---
  language: en
  type: faq
  canonical: TuitionEnquiry
  folder: faq
  aliases: tuition; fee; price; cost
  ---
  ```
- The ingestion script uploads Markdown and sidecar metadata so the KB can filter by:
  - language (en / zh‑HK / zh‑CN)
  - type (course, policy, faq, institution, etc.)
  - canonical (stable identifier)


---

## 💬 Example questions
- “What are your opening hours on Saturday?”  
  → “Sat 09:00–16:00. Closed on Sundays and HK public holidays.”  
- “How long does phonics take to finish?”  
  → “8 levels, typically ~1.5 years, varies by practice.”  
- “Can you send the enrollment form?”  
  → Sends the PDF link automatically.
- “Please cancel next Friday 3pm.”  
  → [Silent — routed to staff for action]
- “You’re welcome.”  
  → [No reply — terminal closing]


---

## 🌦️ Opening hours & weather logic
- Mon–Fri 09:00–18:00; Sat 09:00–16:00; closed Sundays and HK public holidays
- Severe weather (Black Rain or Typhoon Signal No. 8+) triggers deterministic “closed/suspended” messaging with the current warning label, using HKO Open Data.


---

## 🧪 Product principles
- Short, helpful bullets; no filler
- No guessing — if the docs don’t state it, return nothing
- Never propose arrangements or confirm bookings
- Respect language preference automatically
- Keep parents’ data minimal and transient (see chat history retention policy in infra)


---

## 🚀 For developers (quick start)

### 1) Prerequisites
- AWS credentials with Bedrock + S3 access
- A Bedrock Knowledge Base (vector store) linked to your S3 bucket prefix
- Environment variables (minimum):
  - KB_ID — your Knowledge Base ID
  - LLM_MODEL_ID — generator model ID (e.g., meta.llama3-70b-instruct-v1:0)
  - KB_S3_BUCKET — S3 bucket for docs
  - KB_S3_PREFIX — prefix (e.g., ls/kb/v1)
  - AWS_REGION — e.g., ap-northeast-1 or us-east-1

Optional (WhatsApp, weather, diagnostics) are documented in llm/config.py.

### 2) Ingest (sync) content
Use the robust ingestion script that uploads Markdown + sidecars and can start a KB sync:
```bash
# Example (PowerShell/Unix):
export AWS_REGION=ap-northeast-1
export KB_S3_BUCKET=your-bucket
export KB_S3_PREFIX=ls/kb/v1
export KB_ID=kb-xxxxxxxxxxxxxxxx
export KB_DATA_SOURCE_ID=ds-xxxxxxxxxxxxxxxx   # if you want START_INGEST=true
export START_INGEST=true

python -m llm.ingest_kb_from_content
```
Tips:
- Sidecars auto‑generate from frontmatter when CREATE_SIDECAR_IF_MISSING=true
- Language/type/canonical come from frontmatter or inferred from path

### 3) Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Call the chat endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"中文語文課幾時上堂？收費點樣？","language":"zh-HK"}'
```

### 5) WhatsApp (optional)
Set:
- WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_GRAPH_VERSION
- WHATSAPP_TEST_NUMBERS: “+85212345678,+85287654321”
The webhook endpoint is /whatsapp_webhook (see llm/router.py for routing and guardrails).


---

## ⚙️ Notable guardrails (implemented in code)
- Admin Scheduling / Leave routing rules (content/en/faq/admin_scheduling_routing.md)
- No‑answer matrix (content/en/faq/no_answer_matrix.md)
- Minimal contact answers (phone/email only unless asked)
- Terminal closing detection (“You’re welcome / 不客气 / 唔使客氣” → no reply)
- Enrollment/Blooket markers to attach official documents
- Opening‑hours intent detector with weather/holiday awareness

If any rule requires silence, the final output is exactly “[NO_ANSWER]” (or an empty message over WhatsApp).


---

## 🧩 Architecture at a glance

```
User (Web / WhatsApp)
        │
        ▼
    FastAPI /chat  ──► Intent & guardrails (llm.intent)
        │                    │
        │                    ├─ Opening-hours context (llm.opening_hours + HKO)
        │                    └─ Extra keywords for retrieval hints
        ▼
  Bedrock KB (Retrieve) ──► Filter by language → top K chunks
        │
        ▼
  Bedrock LLM (Generate) ──► Llama 3 Instruct (LLM_MODEL_ID)
        │
        ▼
 Post-processing (llm.bedrock_kb_client)
  - Silence on [NO_ANSWER]/no-cite/hedges
  - Enforce admin rules
  - Attach forms (markers)
        │
        ▼
 Response (short, helpful, cited)
 or
 [NO_ANSWER] + human follow‑up (digest + auto‑ack)
```

---

## 📎 Tips for great KB results
- Keep each Markdown page focused with clear “aliases” in frontmatter
- Mirror content across en / zh‑HK / zh‑CN for language‑perfect answers
- Use concrete numbers and phrases (the model prefers specifics)
- Add assistant‑only notes when something should not be said to parents
- Prefer short bullets over long paragraphs

---

## 📣 Credits & license
- Built for Little Scholars Education Centre (Hong Kong)
- Uses Amazon Bedrock (Knowledge Bases + Foundation Models)
- See repository root for license and broader service integration

Have ideas to improve parent experience? PRs welcome — especially docs quality, better aliases, and edge‑case guardrails.