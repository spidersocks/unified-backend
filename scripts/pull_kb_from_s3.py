import os, json, pathlib, sys
from typing import Dict, Optional
import boto3

try:
    import yaml  # pip install pyyaml
except Exception:
    print("[ERROR] Missing dependency: pyyaml. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

AWS_REGION = os.environ.get("AWS_REGION")
BUCKET = os.environ.get("KB_S3_BUCKET", "")
PREFIX = (os.environ.get("KB_S3_PREFIX", "ls/kb/v1") or "").strip("/")
DEST_DIR = os.environ.get("KB_CONTENT_DIR", "content")
WRITE_FRONTMATTER = (os.environ.get("PULL_WRITE_FRONTMATTER", "true").lower() in ("1","true","yes"))
KEEP_SIDECARS = (os.environ.get("PULL_KEEP_SIDECARS", "true").lower() in ("1","true","yes"))

if not BUCKET:
    print("[ERROR] KB_S3_BUCKET is required", file=sys.stderr)
    sys.exit(2)

s3 = boto3.client("s3", region_name=AWS_REGION)

def list_keys(bucket: str, prefix: str):
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj["Key"]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

def get_object_text(bucket: str, key: str) -> Optional[str]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8", errors="ignore")
    except Exception:
        return None

def get_sidecar_attrs(bucket: str, md_key: str) -> Dict[str,str]:
    sidecar_key = f"{md_key}.metadata.json"
    try:
        raw = get_object_text(bucket, sidecar_key)
        if raw:
            data = json.loads(raw)
            attrs = {}
            for k, spec in (data.get("metadataAttributes") or {}).items():
                v = (spec or {}).get("value") or {}
                sval = v.get("stringValue")
                if sval is not None:
                    attrs[k] = str(sval)
            return attrs
    except Exception:
        pass
    # Fallback to object tags
    try:
        tags = s3.get_object_tagging(Bucket=bucket, Key=md_key).get("TagSet", [])
        attrs = {}
        for t in tags:
            k = t.get("Key")
            v = t.get("Value")
            if k and v:
                attrs[k] = v
        return attrs
    except Exception:
        return {}

def write_file(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def main():
    root = pathlib.Path(DEST_DIR)
    count = 0
    for key in list_keys(BUCKET, PREFIX + "/"):
        if not key.endswith(".md"):
            # Optionally also save sidecars for completeness
            if KEEP_SIDECARS and key.endswith(".metadata.json"):
                rel = key[len(PREFIX)+1:] if key.startswith(PREFIX + "/") else key
                write_file(root.joinpath(rel), get_object_text(BUCKET, key) or "")
            continue

        body = get_object_text(BUCKET, key)
        if body is None:
            continue

        attrs = get_sidecar_attrs(BUCKET, key)
        rel = key[len(PREFIX)+1:] if key.startswith(PREFIX + "/") else key
        out_path = root.joinpath(rel)

        if WRITE_FRONTMATTER and attrs:
            # Prefer standard keys first
            ordered = {}
            for k in ("language", "type", "canonical"):
                if k in attrs:
                    ordered[k] = attrs.pop(k)
            # Keep any extras (e.g., folder)
            for k, v in sorted(attrs.items()):
                ordered[k] = v
            fm = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True).strip()
            text = f"---\n{fm}\n---\n{body.lstrip()}"
        else:
            text = body

        write_file(out_path, text)
        count += 1

        # Also save sidecar locally if it exists and KEEP_SIDECARS=true
        if KEEP_SIDECARS:
            sc_key = f"{key}.metadata.json"
            sc = get_object_text(BUCKET, sc_key)
            if sc is not None:
                sc_rel = f"{rel}.metadata.json"
                write_file(root.joinpath(sc_rel), sc)

        print(f"[PULLED] s3://{BUCKET}/{key} -> {out_path.as_posix()}")

    print(f"[DONE] Pulled {count} markdown files from s3://{BUCKET}/{PREFIX}/ into ./{DEST_DIR}")

if __name__ == "__main__":
    main()