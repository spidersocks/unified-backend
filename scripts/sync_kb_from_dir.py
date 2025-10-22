import os, json, pathlib, sys
from typing import Optional, Tuple, Dict
import boto3

try:
    import yaml  # pip install pyyaml
except Exception:
    print("[ERROR] Missing dependency: pyyaml. pip install pyyaml", file=sys.stderr)
    sys.exit(2)

BUCKET = os.environ.get("KB_S3_BUCKET", "")
PREFIX = (os.environ.get("KB_S3_PREFIX", "ls/kb/v1") or "").strip("/")
CONTENT_ROOT = os.environ.get("KB_CONTENT_DIR", "content")

if not BUCKET:
    print("[ERROR] KB_S3_BUCKET env is required", file=sys.stderr)
    sys.exit(2)

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))

def put_obj(key: str, body: bytes, ctype: str):
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ctype)

def sidecar_key(md_key: str) -> str:
    return f"{md_key}.metadata.json"

def make_sidecar(attrs: Dict) -> bytes:
    # Convert frontmatter attrs into Bedrock KB metadataAttributes
    meta = {"metadataAttributes": {}}
    for k, v in attrs.items():
        if v is None:
            continue
        meta["metadataAttributes"][k] = {
            "value": {"type": "STRING", "stringValue": str(v)},
            "includeForEmbedding": True,
        }
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def parse_frontmatter(text: str) -> Tuple[Dict, str]:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm_text = text[3:end].strip()
            body = text[end + 4 :].lstrip("\n")
            data = yaml.safe_load(fm_text) or {}
            return data, body
    return {}, text

def infer_attrs_from_path(rel: str, fm: Dict) -> Dict:
    # content/<lang>/<type>/<name>.md
    parts = rel.split("/")
    attrs = dict(fm)
    if "language" not in attrs and len(parts) >= 2:
        attrs["language"] = parts[0]  # en | zh-HK | zh-CN
    if "type" not in attrs and len(parts) >= 3:
        attrs["type"] = parts[1]  # courses | institution | policies | marketing
        # normalize policy -> policy
        if attrs["type"] == "policies":
            attrs["type"] = "policy"
    if "canonical" not in attrs:
        fname = pathlib.Path(rel).stem  # name without .md
        attrs["canonical"] = fname
    return attrs

def main():
    root = pathlib.Path(CONTENT_ROOT)
    if not root.exists():
        print(f"[ERROR] Folder not found: {root.resolve()}", file=sys.stderr)
        sys.exit(2)

    uploaded = 0
    for md in root.glob("**/*.md"):
        rel = md.relative_to(root).as_posix()  # en/courses/X.md
        with md.open("r", encoding="utf-8") as f:
            txt = f.read()
        fm, body = parse_frontmatter(txt)
        attrs = infer_attrs_from_path(rel, fm)

        # Destination key in S3 mirrors ls/kb/v1/<rel>
        md_key = f"{PREFIX}/{rel}"
        put_obj(md_key, body.encode("utf-8"), "text/markdown; charset=utf-8")
        put_obj(sidecar_key(md_key), make_sidecar(attrs), "application/json")
        uploaded += 1
        print(f"[UPLOAD] s3://{BUCKET}/{md_key} + .metadata.json [{attrs}]")

    print(f"[DONE] Uploaded {uploaded} docs. Remember to start a KB ingestion job.", flush=True)

if __name__ == "__main__":
    main()