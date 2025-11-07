#!/usr/bin/env python3
"""
Debug helper for the 5pm daily admin digest. Lets you seed test data, preview, and (optionally) send digests.
Extended for edge-case topic labeling tests.
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

def test_edge_labeling():
    """
    Tests the topic labeling logic of admin_digest._topic_from_flags on edge-case messages.
    Prints a table of test input, flags, and the derived label.
    """
    print("\n[EDGE CASE LABELING TESTS]")
    try:
        from llm import intent
        from llm import admin_digest
    except Exception as e:
        print(f"[ERR] Failed to import: {e}", file=sys.stderr); return

    test_msgs = [
        # English - availability
        ("Is any class available after assessment for Owen?", "en"),
        # English - leave/reschedule
        ("Please cancel next Friday 3pm class.", "en"),
        # English - staff contact
        ("Can I schedule a call with the director?", "en"),
        # English - placement/policy explicit
        ("My son is in K1. Is this class too young for him? What is your placement policy?", "en"),
        # EN - pass-on but no request
        ("Please tell the teacher she is allergic to peanuts.", "en"),
        # EN - individualized homework
        ("He can't say G and J properly. How should I help?", "en"),
        # EN - generic fee
        ("How much does English class cost?", "en"),
        # EN - policy (should say Policy question)
        ("What is your make-up lesson policy?", "en"),
        # EN - polite thanks only (should be None/ignored)
        ("Thank you!", "en"),

        # Chinese (zh-HK) - leave notification
        ("星期三請假，唔好意思", "zh-HK"),
        # Chinese (zh-HK) - admin pass-on
        ("請幫我轉告老師：Oscar 今天咳嗽。", "zh-HK"),
        # Chinese (zh-HK) - schedule inquiry
        ("下星期有冇得補課？", "zh-HK"),
        # Chinese (zh-HK) - homework
        ("佢唔識讀 '狗' 同 '球'", "zh-HK"),
        # Chinese (zh-CN) - placement
        ("我家女儿是K3，这班是否太小？", "zh-CN"),
        # Chinese (zh-CN) - policy
        ("请问补课政策", "zh-CN"),
    ]

    results = []
    for msg, lang in test_msgs:
        try:
            flags = intent.classify_scheduling_context(msg, lang)
            label = admin_digest._topic_from_flags(flags)
            # Keep table neat: show minimal JSON for flags
            sig_flags = {k:v for k,v in flags.items() if v}
            results.append((msg, lang, sig_flags, label or "—"))
        except Exception as e:
            results.append((msg, lang, "ERR", str(e)))

    # Print a readable table
    print("{:<38} | {:<6} | {:<45} | {:<25}".format("Input", "Lang", "Active Flags", "Topic Label"))
    print("-"*120)
    for msg, lang, flags, label in results:
        print("{:<38} | {:<6} | {:<45} | {:<25}".format(
            msg[:36] + ("…" if len(msg) > 36 else ""), lang, json.dumps(flags, ensure_ascii=False), label)
        )
    print("[Done edge-case topic labeling tests]\n")

def main():
    # Load .env early so AWS_REGION and creds are available to boto3 when running standalone
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Admin Digest (5pm roundup) debug tool")
    ap.add_argument("--use-ddb", default=os.environ.get("USE_ADMIN_DIGEST_DDB", "true"),
                    help="Use DynamoDB (true/false).")
    ap.add_argument("--table", default=os.environ.get("ADMIN_DIGEST_TABLE", "AdminDigestPending"),
                    help="DynamoDB table name (Option B schema).")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION"),
                    help="AWS region to use.")
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
    ap.add_argument("--edge-labels", action="store_true", help="Run edge-case topic labeling tests.")
    args = ap.parse_args()

    os.environ["USE_ADMIN_DIGEST_DDB"] = "true" if parse_bool(args.use_ddb) else "false"
    if args.table:
        os.environ["ADMIN_DIGEST_TABLE"] = args.table
    if args.region:
        os.environ["AWS_REGION"] = args.region

    try:
        from llm import admin_digest
        from llm.config import SETTINGS
    except Exception as e:
        print(f"[ERR] Failed to import admin_digest/SETTINGS: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    eff_region = os.environ.get("AWS_REGION") or SETTINGS.aws_region
    print(f"[INFO] Using table={os.environ.get('ADMIN_DIGEST_TABLE')} region={eff_region} use_ddb={parse_bool(args.use_ddb)}")

    if args.edge_labels:
        test_edge_labeling()

    if args.when:
        try:
            from llm import admin_digest as AD
            TZ = pytz.timezone(args.tz)
            now = datetime.now(TZ)
            secs = AD._next_run_delay_seconds(now)  # type: ignore[attr-defined]
            print(f"Seconds until next digest (tz={args.tz}): {int(secs)}")
        except Exception as e:
            print(f"[ERR] when: {e}", file=sys.stderr)

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

    if args.resolve:
        try:
            admin_digest.resolve_session(args.resolve)
            print(f"[OK] Resolved today's pendings for session_id={args.resolve}")
        except Exception as e:
            print(f"[ERR] resolve: {e}", file=sys.stderr)

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