"""
Emit multilingual, topic-scoped Markdown docs from the ContentStore and upload to S3
for Bedrock Knowledge Base ingestion. Uses S3 object tags to label language/type/course.

Run:
  INFO_SHEET_CATALOG_URL=... KB_S3_BUCKET=... KB_S3_PREFIX=... KB_ID=... KB_MODEL_ARN=... \
  AWS_REGION=... python -m llm.ingest_bedrock_kb
"""
import sys
import boto3
from typing import Optional, List
from llm.config import SETTINGS
from dialogflow_app.content_store import ContentStore, norm_lang, CANONICAL_COURSES

s3 = boto3.client("s3", region_name=SETTINGS.aws_region)

def _put_md(key: str, body: str, tags: str):
    s3.put_object(
        Bucket=SETTINGS.kb_s3_bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="text/markdown; charset=utf-8",
        Tagging=tags
    )
    print(f"[UPLOAD] s3://{SETTINGS.kb_s3_bucket}/{key} tags={tags}")

def _h1(txt: str) -> str:
    return f"# {txt}\n\n"

def _sec(name: str, txt: Optional[str]) -> str:
    return f"### {name}\n{txt}\n\n" if txt else ""

def _lang_label(lang: str) -> str:
    l = norm_lang(lang)
    return "zh-HK" if l == "zh-HK" else ("zh-CN" if l == "zh-CN" else "en")

def _guess_display(store: ContentStore, canonical: str, lang: str) -> str:
    try:
        return store.format_course_name(canonical, lang)
    except Exception:
        return canonical

def build_course_doc(store: ContentStore, canonical: str, lang: str) -> Optional[str]:
    L = _lang_label(lang)
    title = _guess_display(store, canonical, L)

    desc = store.course_description(canonical, L)
    age = store.age_for_course(canonical, L)
    sched = store.schedule_for_course(canonical, L)
    size = store.class_size_for_course(canonical, L)
    fee = store.tuition_for_course(canonical, L)

    def after_colon(s: Optional[str]) -> Optional[str]:
        if not s:
            return s
        parts = s.split(":", 1)
        return parts[1].strip() if len(parts) == 2 else s

    age = after_colon(age)
    sched = after_colon(sched)
    size = after_colon(size)
    fee = after_colon(fee)

    if not any([desc, age, sched, size, fee]):
        return None

    if L == "zh-HK":
        parts = [
            _h1(title),
            _sec("簡介", desc),
            _sec("適合年齡", age),
            _sec("上課時間", sched),
            _sec("班級人數", size),
            _sec("收費", fee),
        ]
    elif L == "zh-CN":
        parts = [
            _h1(title),
            _sec("简介", desc),
            _sec("适合年龄", age),
            _sec("上课时间", sched),
            _sec("班级人数", size),
            _sec("收费", fee),
        ]
    else:
        parts = [
            _h1(title),
            _sec("Overview", desc),
            _sec("Target age", age),
            _sec("Schedule", sched),
            _sec("Class size", size),
            _sec("Tuition", fee),
        ]
    return "".join(parts)

def build_policy_doc(store: ContentStore, lang: str) -> List[tuple[str, str]]:
    L = _lang_label(lang)
    out: List[tuple[str, str]] = []

    intro = store.institution_intro(L)
    if intro:
        title = "機構介紹" if L == "zh-HK" else ("机构介绍" if L == "zh-CN" else "Institution Introduction")
        out.append(("institution/intro.md", _h1(title) + intro + "\n"))

    contact = store.contact_info(L)
    if contact:
        title = "聯絡資料" if L == "zh-HK" else ("联系资料" if L == "zh-CN" else "Contact Information")
        out.append(("institution/contact.md", _h1(title) + contact + "\n"))

    abs_mkup = store.policy_absence_makeup(L)
    if abs_mkup:
        title = "補課與請假政策" if L == "zh-HK" else ("补课与请假政策" if L == "zh-CN" else "Absence & Make-up Policy")
        out.append(("policies/absence_makeup.md", _h1(title) + abs_mkup + "\n"))

    refund = store.policy_refund(L)
    if refund:
        title = "退款與轉班政策" if L == "zh-HK" else ("退款与转班政策" if L == "zh-CN" else "Refund & Class Transfer Policy")
        out.append(("policies/refund_transfer.md", _h1(title) + refund + "\n"))

    promos = store.promotions(L)
    if promos:
        title = "優惠與活動" if L == "zh-HK" else ("优惠与活动" if L == "zh-CN" else "Promotions & Events")
        out.append(("marketing/promotions.md", _h1(title) + promos + "\n"))

    succ = store.success_stories(L)
    if succ:
        title = "成功個案" if L == "zh-HK" else ("成功个案" if L == "zh-CN" else "Success Stories")
        out.append(("marketing/success_stories.md", _h1(title) + succ + "\n"))

    return out

def main():
    if not (SETTINGS.kb_s3_bucket and SETTINGS.kb_s3_prefix and SETTINGS.kb_id and SETTINGS.kb_model_arn):
        print("[ERROR] Missing KB config. Please set KB_S3_BUCKET, KB_S3_PREFIX, KB_ID, KB_MODEL_ARN, AWS_REGION.", file=sys.stderr)
        sys.exit(2)

    store = ContentStore()  # reads INFO_SHEET_CATALOG_URL
    langs = SETTINGS.default_languages

    for L in langs:
        lang_dir = _lang_label(L)
        for course in sorted(CANONICAL_COURSES):
            doc = build_course_doc(store, course, L)
            if not doc:
                continue
            key = f"{SETTINGS.kb_s3_prefix}/{lang_dir}/courses/{course}.md"
            tags = f"language={lang_dir}&type=course&canonical={course}"
            _put_md(key, doc, tags)

        for rel_key, md in build_policy_doc(store, L):
            key = f"{SETTINGS.kb_s3_prefix}/{lang_dir}/{rel_key}"
            tags = f"language={lang_dir}&type={rel_key.split('/')[0]}"
            _put_md(key, md, tags)

    print("[DONE] Upload complete. If your KB is not auto-syncing, trigger a sync now.")

if __name__ == "__main__":
    main()