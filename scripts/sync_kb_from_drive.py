"""
Sync KB docs from a Google Drive folder tree to S3, mirroring content/<lang>/<type>/<name>.md.
- Exports Google Docs to HTML, converts to Markdown (markdownify), preserves simple headings/lists.
- Infers metadata (language/type/canonical) from folder path and/or frontmatter if present.

Requirements:
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib markdownify pyyaml boto3

Auth:
  - Service Account: set GOOGLE_APPLICATION_CREDENTIALS to a JSON key file path, and share the Drive folder
    with that service account email (Viewer is enough for export).
"""
import os, io, sys, json, pathlib, re
from typing import Dict, Tuple
import boto3
import yaml
from markdownify import markdownify as md_from_html
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

BUCKET = os.environ.get("KB_S3_BUCKET", "")
PREFIX = (os.environ.get("KB_S3_PREFIX", "ls/kb/v1") or "").strip("/")
AWS_REGION = os.environ.get("AWS_REGION")
DRIVE_ROOT_ID = os.environ.get("DRIVE_ROOT_FOLDER_ID")  # or pass via CLI
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

if not BUCKET:
    print("[ERROR] KB_S3_BUCKET env is required", file=sys.stderr)
    sys.exit(2)

def drive_client():
    creds = None
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path and os.path.exists(key_path):
        creds = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
    else:
        print("[ERROR] GOOGLE_APPLICATION_CREDENTIALS not set or file missing.", file=sys.stderr)
        sys.exit(2)
    return build("drive", "v3", credentials=creds)

s3 = boto3.client("s3", region_name=AWS_REGION)

def put_obj(key: str, body: bytes, ctype: str):
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ctype)

def sidecar_key(md_key: str) -> str:
    return f"{md_key}.metadata.json"

def make_sidecar(attrs: Dict) -> bytes:
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

def infer_attrs_from_path(rel_parts: list, fm: Dict) -> Dict:
    # rel_parts like ["zh-HK","courses","ChineseLanguageArts.md"]
    attrs = dict(fm)
    if "language" not in attrs and len(rel_parts) >= 1:
        attrs["language"] = rel_parts[0]
    if "type" not in attrs and len(rel_parts) >= 2:
        t = rel_parts[1]
        attrs["type"] = "policy" if t == "policies" else t
    if "canonical" not in attrs and len(rel_parts) >= 3:
        name = pathlib.Path(rel_parts[-1]).stem
        attrs["canonical"] = name
    return attrs

def list_children(drive, folder_id: str):
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "files(id,name,mimeType)"
    page_token = None
    while True:
        resp = drive.files().list(q=q, fields=f"nextPageToken,{fields}", pageToken=page_token).execute()
        for f in resp.get("files", []):
            yield f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

def export_gdoc_to_md(drive, file_id: str) -> str:
    # Export Google Doc to HTML, convert to Markdown
    req = drive.files().export_media(fileId=file_id, mimeType="text/html")
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    html = buf.getvalue().decode("utf-8", errors="ignore")
    # Normalize headings a bit (Google exports strong tags sometimes)
    html = re.sub(r"<p><strong>([^<]+)</strong></p>", r"<h3>\1</h3>", html)
    md = md_from_html(html)
    return md.strip()

def walk_and_sync(drive, folder_id: str, rel_parts: list):
    for item in list_children(drive, folder_id):
        name = item["name"]
        mime = item["mimeType"]
        if mime == "application/vnd.google-apps.folder":
            # Recurse
            walk_and_sync(drive, item["id"], rel_parts + [name])
            continue

        # Only sync Google Docs or Markdown/plain text files
        if mime == "application/vnd.google-apps.document":
            body = export_gdoc_to_md(drive, item["id"])
        else:
            # Download as bytes and try to decode
            req = drive.files().get_media(fileId=item["id"])
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            body = buf.getvalue().decode("utf-8", errors="ignore")

        # Ensure .md extension
        fname = name if name.endswith(".md") else f"{name}.md"
        fm, md_body = parse_frontmatter(body)
        attrs = infer_attrs_from_path(rel_parts + [fname], fm)

        s3_key = "/".join([PREFIX] + rel_parts + [fname])
        put_obj(s3_key, md_body.encode("utf-8"), "text/markdown; charset=utf-8")
        put_obj(sidecar_key(s3_key), make_sidecar(attrs), "application/json")
        print(f"[UPLOAD] s3://{BUCKET}/{s3_key} + .metadata.json [{attrs}]")

def main():
    root_id = DRIVE_ROOT_ID
    if not root_id:
        # allow passing as CLI arg: --root-folder-id <id>
        try:
            idx = sys.argv.index("--root-folder-id")
            root_id = sys.argv[idx + 1]
        except Exception:
            print("[ERROR] Provide DRIVE_ROOT_FOLDER_ID env or --root-folder-id", file=sys.stderr)
            sys.exit(2)
    drive = drive_client()
    walk_and_sync(drive, root_id, [])
    print("[DONE] Drive → S3 sync complete. Start a KB ingestion job next.", flush=True)

if __name__ == "__main__":
    main()