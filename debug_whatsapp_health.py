import os
import sys
import json
import requests
from dotenv import load_dotenv

# 1. Load environment variables from .env file BEFORE importing config
load_dotenv()

# 2. Now import settings (which reads os.environ)
try:
    from llm.config import SETTINGS
except ImportError:
    sys.path.append(os.getcwd())
    from llm.config import SETTINGS

def check_whatsapp_health():
    print("--- WhatsApp Health Check (Direct to Meta) ---")
    
    pid = SETTINGS.whatsapp_phone_number_id
    token = SETTINGS.whatsapp_access_token
    version = SETTINGS.whatsapp_graph_version or "v18.0"

    if not pid or not token:
        print("\n[ERROR] Missing credentials in .env.")
        print(f"ID: {pid}")
        print(f"Token: {'***' if token else 'None'}")
        return

    # --- Attempt 1: Treat ID as a Phone Number ID (Standard) ---
    url = f"https://graph.facebook.com/{version}/{pid}"
    params = {
        "fields": "status,quality_rating,code_verification_status,name_status",
        "access_token": token
    }

    print(f"\nChecking ID: {pid} ...")
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        
        # Case A: Success (It really was a Phone Number ID)
        if resp.status_code == 200:
            print("\n✅ SUCCESS: Found Phone Number Status")
            print(json.dumps(data, indent=2))
            if data.get("status") == "CONNECTED":
                print("\n>> STATUS: CONNECTED. If YCloud is 'Locked', check Coexistence/Mobile App conflict.")
            else:
                print(f"\n>> STATUS: {data.get('status')} (Review/Ban/Pending)")
            return

        # Case B: Error - Check if it is a WABA ID error
        err = data.get("error", {})
        msg = err.get("message", "")
        
        if "WhatsAppBusinessAccount" in msg:
            print("\n⚠️  MISCONFIGURATION DETECTED")
            print("You provided a Business Account ID (WABA ID), but the API needs a Phone Number ID.")
            print("Attempting to fetch associated phone numbers...\n")
            
            # --- Attempt 2: List Phone Numbers for this WABA ---
            waba_url = f"https://graph.facebook.com/{version}/{pid}/phone_numbers"
            waba_resp = requests.get(waba_url, params={"access_token": token}, timeout=15)
            waba_data = waba_resp.json()
            
            if waba_resp.status_code == 200:
                print("--- Available Phone Numbers ---")
                for item in waba_data.get("data", []):
                    print(f"Number: {item.get('display_phone_number')} | ID: {item.get('id')}  <-- USE THIS ID")
                    print(f"Status: {item.get('status')} | Quality: {item.get('quality_rating')}")
                    print("-" * 30)
                print("\n>> ACTION: Update WHATSAPP_PHONE_NUMBER_ID in your .env with the 'ID' shown above.")
            else:
                print(f"Failed to list numbers: {waba_resp.text}")
        else:
            print(f"\n❌ API Error: {resp.status_code}")
            print(json.dumps(data, indent=2))

    except Exception as e:
        print(f"\n[EXCEPTION] Request failed: {e}")

if __name__ == "__main__":
    check_whatsapp_health()