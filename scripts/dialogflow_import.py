#!/usr/bin/env python
import os
import io
import zipfile
from google.cloud import dialogflow_v2 as dialogflow

"""
Imports (merges) your Dialogflow ES agent from the folder:
  dialogflow_app/data/LittleScholars_WhatsApp_Bot

Prereqs:
  - pip install google-cloud-dialogflow
  - GOOGLE_APPLICATION_CREDENTIALS set to a service account JSON
    with roles/dialogflow.admin on the target project
  - DIALOGFLOW_PROJECT_ID environment variable set to the agent's GCP project ID
"""

def make_zip_bytes(agent_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(agent_dir):
            for f in files:
                if f == ".DS_Store":
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, agent_dir)
                zf.write(full_path, arcname=rel_path)
    buf.seek(0)
    return buf.read()

def main():
    project_id = os.environ["DIALOGFLOW_PROJECT_ID"]
    agent_dir = os.environ.get(
        "AGENT_DIR",
        os.path.join("dialogflow_app", "data", "LittleScholars_WhatsApp_Bot"),
    )

    print(f"[INFO] Zipping agent from: {os.path.abspath(agent_dir)}")
    zip_bytes = make_zip_bytes(agent_dir)

    client = dialogflow.AgentsClient()
    parent = f"projects/{project_id}"

    print("[INFO] Importing agent (merge)...")
    op = client.import_agent(request={"parent": parent, "agent_content": zip_bytes})

    print("[INFO] Waiting for operation to complete...")
    op.result(timeout=600)
    print("[INFO] Import completed successfully.")

if __name__ == "__main__":
    main()