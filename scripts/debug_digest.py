#!/usr/bin/env python3
"""
Debug helper for the 5pm daily admin digest (Option B schema: PK=date, SK=session_id#ts).

Run from repo root with:
  PYTHONPATH=. python scripts/debug_digest.py --help
"""
import os
import sys
import json
import time
import argparse
import asyncio
from datetime import datetime
import pytz

def parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y", "on")

def main():
    # Load .env early so AWS_REGION and creds are available to boto3 when running standalone
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Admin Digest (5pm roundup) debug tool")
    ap.add_argument("--use-ddb", default=os.environ.get("USE_ADMIN_DIGEST_DDB", "true"),
                    help="Use DynamoDB (true/false). Default reads from USE_ADMIN_DIGEST_DDB env.")
    ap.add_argument("--table", default=os.environ.get("ADMIN_DIGEST_TABLE", "AdminDigestPending"),
                    help="DynamoDB table name (Option B schema).")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION"),
                    help="AWS region to use (sets AWS_REGION for this run).")
    ap.add_argument("--seed", action="store_true", help="Seed sample pending items for today.")
    ap.add_argument("--seed-count", type=int, default=2, help="How many sample items to seed (default 2).")
    ap.add_argument("--list", action="store_true", help="List unresolved items for today (dedup by session, latest only).")
    ap.add_argument("--json", dest="as_json", action="store_true", help="Output list in JSON when using --list.")
    ap.add_argument("--preview", action="store_true", help="Print the digest body that would be sent.")
    ap.add_argument("--send-now", action="store_true", help="Send the digest now to the director number via WhatsApp.")
    ap.add_argument("--director-number", default=os.environ.get("ADMIN_DIGEST_DIRECTOR_NUMBER"),
                    help="Override director number for send-now.")
    ap.add_argument("--resolve", default=None, help="Resolve all pending items for the given session_id (today).")
    ap.add_argument("--when", action="store_true", help="Print seconds until the next scheduled digest run.")
    ap.add_argument("--tz", default=os.environ.get("ADMIN_DIGEST_TZ", "Asia/Hong_Kong"), help="Timezone label (for --when).")
    args = ap.parse_args()

    # Configure environment BEFORE importing modules
    os.environ["USE_ADMIN_DIGEST_DDB"] = "true" if parse_bool(args.use_ddb) else "false"
    if args.table:
        os.environ["ADMIN_DIGEST_TABLE"] = args.table
    if args.region:
        os.environ["AWS_REGION"] = args.region

    # Lazy import after env is set
    try:
        from llm import admin_digest
        from llm.config import SETTINGS
    except Exception as e:
        print(f"[ERR] Failed to import admin_digest/SETTINGS: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    # Show effective region/table for sanity
    eff_region = os.environ.get("AWS_REGION") or SETTINGS.aws_region
    print(f"[INFO] Using table={os.environ.get('ADMIN_DIGEST_TABLE')} region={eff_region} use_ddb={parse_bool(args.use_ddb)}")

    if args.when:
        try:
            from llm import admin_digest as AD
            TZ = pytz.timezone(args.tz)
            now = datetime.now(TZ)
            secs = AD._next_run_delay_seconds(now)  # type: ignore[attr-defined]
            print(f"Seconds until next digest (tz={args.tz}): {int(secs)}")
        except Exception as e:
            print(f"[ERR] when: {e}", file=sys.stderr)

    # Optionally seed sample items
    if args.seed:
        now = time.time()
        samples = [
            ("+85251180001", "Hi, is there any class available Tue 4–5pm for K1?", "en", {"availability_request": True}),
            ("+85251180002", "請幫我轉告老師：Oscar 今天咳嗽。", "zh-HK", {"admin_action_request": True}),
            ("+85251180003", "Placement query: my daughter is K2 — is this class too young for her?", "en", {"placement_question": True}),
            ("+85251180004", "請問補課政策係點樣？", "zh-HK", {"has_policy_intent": True}),
            ("+85251180005", "We can’t attend on 11/5. Sorry!", "en", {"has_sched_verbs": True, "has_date_time": True}),
        ]
        count = max(1, min(args.seed_count, len(samples)))
        print(f"[SEED] Adding {count} pending item(s)")
        for i in range(count):
            sid, msg, lang, flags = samples[i]
            ts = now - (120 * (count - i))
            admin_digest.add_pending(session_id=sid, message=msg, lang=lang, flags=flags, ts=ts)
        print("[SEED] Done.")

    # Resolve for a specific session if requested
    if args.resolve:
        try:
            admin_digest.resolve_session(args.resolve)
            print(f"[OK] Resolved today's pendings for session_id={args.resolve}")
        except Exception as e:
            print(f"[ERR] resolve: {e}", file=sys.stderr)

    # List unresolved (dedup by session, latest only)
    items = None
    if args.list or args.preview or args.send_now:
        try:
            items = admin_digest.list_unresolved_today()
            if args.list:
                if args.as_json:
                    print(json.dumps(items, ensure_ascii=False, indent=2))
                else:
                    print(f"[LIST] {len(items)} unresolved sessions for today:")
                    for it in items:
                        print(f"- {it.get('session_id')} @ {it.get('ts')} | topic={ (it.get('flags') or {}) }")
            if args.preview:
                body = admin_digest._format_digest_body(items)  # type: ignore[attr-defined]
                print("\n[PREVIEW]\n" + body)
        except Exception as e:
            print(f"[ERR] list/preview: {e}", file=sys.stderr)

    # Send now (WhatsApp)
    if args.send_now:
        try:
            if items is None:
                items = admin_digest.list_unresolved_today()
            if not items:
                print("[SEND] No unresolved items for today. Nothing to send.")
            else:
                to = args.director_number or SETTINGS.admin_digest_director_number
                body = admin_digest._format_digest_body(items)  # type: ignore[attr-defined]

                async def _go():
                    ok = await admin_digest._send_whatsapp_text(to, body)  # type: ignore[attr-defined]
                    print(f"[SEND] Sent={ok} to {to}")

                asyncio.run(_go())
        except Exception as e:
            print(f"[ERR] send-now: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()