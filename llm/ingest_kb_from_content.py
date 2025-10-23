import os
import sys
import json
import pathlib
import boto3
from typing import Dict, Optional

# Env
AWS_REGION = os.environ.get("AWS_REGION")
BUCKET = os.environ.get("KB_S3_BUCKET", "")
PREFIX = (os.environ.get("KB_S3_PREFIX", "ls/kb/v1") or "").strip("/")
CONTENT_ROOT = os.environ.get("KB_CONTENT_DIR", "content")

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes")
CREATE_SIDECAR_IF_MISSING = os.environ.get("CREATE_SIDECAR_IF_MISSING", "false").lower() in ("1", "true", "yes")
WRITE_SIDECAR_LOCAL = os.environ.get("KB_WRITE_SIDECAR_LOCAL", "true").lower() in ("1","true","yes")

START_INGEST = os.environ.get("START_INGEST", "false").lower() in ("1", "true", "yes")
KB_ID = os.environ.get("KB_ID", "")
DS_ID = os.environ.get("KB_DATA_SOURCE_ID", "") or os.environ.get("DS_ID", "")

# Limits per AWS docs
MAX_METADATA_FILE_BYTES = 10 * 1024  # 10 KB

if not BUCKET:
    print("[ERROR] KB_S3_BUCKET is required", file=sys.stderr)
    sys.exit(2)

s3 = boto3.client("s3", region_name=AWS_REGION)
bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION) if START_INGEST else None

def ensure_prefix(p: str) -> str:
    return p if not p else p.strip("/")

def put_obj(key: str, body: bytes, ctype: str):
    if DRY_RUN:
        print(f"[DRY-RUN] PUT {ctype} s3://{BUCKET}/{key} ({len(body)} bytes)")
        return
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ctype)

def content_type_for(path: pathlib.Path) -> Optional[str]:
    name = path.name
    if name.endswith(".metadata.json"):
        return "application/json"
    if name.endswith(".md"):
        return "text/markdown; charset=utf-8"
    return None

def is_sidecar_file(path: pathlib.Path) -> bool:
    return path.name.endswith(".metadata.json")

def write_local_sidecar(md_path: pathlib.Path, sidecar_bytes: bytes):
    """
    Writes <file>.md.metadata.json next to the Markdown file.
    """
    sc_path = md_path.with_suffix(md_path.suffix + ".metadata.json")
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sc_path.write_bytes(sidecar_bytes)
    print(f"[LOCAL META] {sc_path.as_posix()}")

def make_sidecar_from_frontmatter(md_text: str) -> Optional[bytes]:
    """
    Minimal frontmatter parser: expects leading ---\\n ... \\n--- block.
    Produces Bedrock metadataAttributes JSON if language/type/canonical present (or any keys).
    No YAML dependency; parses simple key: value lines.
    """
    if not md_text.startswith("---"):
        return None
    end = md_text.find("\n---", 3)
    if end == -1:
        return None
    fm_text = md_text[3:end].strip()
    attrs: Dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if v:
            attrs[k] = v
    if not attrs:
        return None
    meta = {"metadataAttributes": {}}
    for k, v in attrs.items():
        meta["metadataAttributes"][k] = {
            "value": {"type": "STRING", "stringValue": str(v)},
            "includeForEmbedding": True,
        }
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def main():
    root = pathlib.Path(CONTENT_ROOT)
    if not root.exists():
        print(f"[ERROR] Folder not found: {root.resolve()}", file=sys.stderr)
        sys.exit(2)

    uploaded = 0
    skipped = 0
    warned = 0

    for file in root.glob("**/*"):
        if not file.is_file():
            continue

        ctype = content_type_for(file)
        if not ctype:
            # Skip non-md/json files
            skipped += 1
            continue

        rel = file.relative_to(root).as_posix()
        s3_key = "/".join([p for p in [ensure_prefix(PREFIX), rel] if p])

        data = file.read_bytes()

        # Validate metadata sidecar size and JSON
        if is_sidecar_file(file):
            if len(data) > MAX_METADATA_FILE_BYTES:
                print(f"[WARN] Sidecar too large (>10KB), skipping: {rel} ({len(data)} bytes)")
                skipped += 1
                warned += 1
                continue
            try:
                json.loads(data.decode("utf-8", errors="strict"))
            except Exception as e:
                print(f"[WARN] Invalid JSON in sidecar {rel}: {e}")
                warned += 1

            put_obj(s3_key, data, ctype)
            uploaded += 1
            print(f"[UPLOAD] s3://{BUCKET}/{s3_key}")
            continue

        # It's a Markdown file
        put_obj(s3_key, data, ctype)
        uploaded += 1
        print(f"[UPLOAD] s3://{BUCKET}/{s3_key}")

        # If requested, auto-generate sidecar when missing (and save locally too)
        if CREATE_SIDECAR_IF_MISSING:
            sidecar_path = root.joinpath(f"{rel}.metadata.json")
            if not sidecar_path.exists():
                sc = make_sidecar_from_frontmatter(data.decode("utf-8", errors="ignore"))
                if sc:
                    if len(sc) <= MAX_METADATA_FILE_BYTES:
                        sc_key = "/".join([p for p in [ensure_prefix(PREFIX), f"{rel}.metadata.json"] if p])
                        put_obj(sc_key, sc, "application/json")
                        uploaded += 1
                        print(f"[AUTOGEN META] s3://{BUCKET}/{sc_key}")

                        if WRITE_SIDECAR_LOCAL:
                            write_local_sidecar(file, sc)
                    else:
                        print(f"[WARN] Autogenerated sidecar >10KB, not uploaded: {rel}.metadata.json")
                        warned += 1
                else:
                    print(f"[INFO] No sidecar present and no usable frontmatter to autogenerate: {rel}")

    print(f"[DONE] Uploaded={uploaded}, Skipped={skipped}, Warnings={warned}")

    # Optionally trigger ingestion
    if START_INGEST:
        if not (KB_ID and DS_ID):
            print("[ERROR] START_INGEST=true requires KB_ID and KB_DATA_SOURCE_ID (or DS_ID) envs", file=sys.stderr)
            sys.exit(2)
        try:
            if DRY_RUN:
                print(f"[DRY-RUN] Would start ingestion: KB_ID={KB_ID}, DS_ID={DS_ID}")
            else:
                resp = bedrock_agent.start_ingestion_job(knowledgeBaseId=KB_ID, dataSourceId=DS_ID)
                job = resp.get("ingestionJob", {})
                job_id = job.get("ingestionJobId", "<unknown>")
                status = job.get("status", "<unknown>")
                print(f"[INGESTION] Started job id={job_id} status={status}")
        except Exception as e:
            print(f"[ERROR] Failed to start ingestion job: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(2)

if __name__ == "__main__":
    main()