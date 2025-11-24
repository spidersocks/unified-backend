#!/usr/bin/env python3
"""
WhatsApp Health Status Check Utility

This script checks the health status of a WhatsApp Business Account via Meta's Graph API.
It retrieves the following information:
- status: Current account status (e.g., PENDING, CONNECTED, RESTRICTED)
- quality_rating: Account quality rating
- code_verification_status: Verification status
- platform_type: Platform type

Usage:
    python debug_whatsapp_health.py

Requirements:
    - WHATSAPP_PHONE_NUMBER_ID environment variable must be set
    - WHATSAPP_ACCESS_TOKEN environment variable must be set
"""

import sys
import json
import requests
from llm.config import SETTINGS


def check_whatsapp_health():
    """
    Check the health status of the WhatsApp Business Account.
    
    Makes a GET request to Meta's Graph API to retrieve account status information.
    """
    # Load credentials from config
    phone_number_id = SETTINGS.whatsapp_phone_number_id
    access_token = SETTINGS.whatsapp_access_token
    graph_version = SETTINGS.whatsapp_graph_version
    
    # Validate credentials are present
    if not phone_number_id:
        print("ERROR: WHATSAPP_PHONE_NUMBER_ID is not configured.", file=sys.stderr)
        print("Please set the WHATSAPP_PHONE_NUMBER_ID environment variable.", file=sys.stderr)
        sys.exit(1)
    
    if not access_token:
        print("ERROR: WHATSAPP_ACCESS_TOKEN is not configured.", file=sys.stderr)
        print("Please set the WHATSAPP_ACCESS_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)
    
    # Construct the Meta Graph API endpoint
    url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}"
    
    # Define fields to retrieve
    fields = "status,quality_rating,code_verification_status,platform_type"
    
    # Set up request parameters
    params = {
        "fields": fields,
        "access_token": access_token
    }
    
    print(f"Checking WhatsApp Business Account health...")
    print(f"Phone Number ID: {phone_number_id}")
    print(f"Graph API Version: {graph_version}")
    print(f"Endpoint: {url}")
    print(f"Fields: {fields}")
    print("-" * 80)
    
    try:
        # Make the GET request
        response = requests.get(url, params=params, timeout=30)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Print formatted JSON response
        print("\n✅ SUCCESS: WhatsApp Account Health Status Retrieved\n")
        print(json.dumps(data, indent=2))
        print("\n" + "-" * 80)
        
        # Provide interpretation of status field if present
        if "status" in data:
            status = data["status"]
            print(f"\nAccount Status: {status}")
            
            status_explanations = {
                "CONNECTED": "✅ Account is active and operational",
                "PENDING": "⚠️  Account is under review by Meta",
                "RESTRICTED": "🚫 Account has been restricted",
                "DELETED": "❌ Account has been deleted",
                "FLAGGED": "⚠️  Account has been flagged for review"
            }
            
            explanation = status_explanations.get(status, "Status explanation not available")
            print(f"Explanation: {explanation}")
        
        # Provide interpretation of quality_rating if present
        if "quality_rating" in data:
            quality = data["quality_rating"]
            print(f"\nQuality Rating: {quality}")
            
            quality_explanations = {
                "GREEN": "✅ High quality - no restrictions",
                "YELLOW": "⚠️  Medium quality - some restrictions may apply",
                "RED": "🚫 Low quality - significant restrictions",
                "UNKNOWN": "❓ Quality rating not yet determined"
            }
            
            explanation = quality_explanations.get(quality, "Quality explanation not available")
            print(f"Explanation: {explanation}")
        
        print("\n" + "=" * 80)
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP ERROR: {e}", file=sys.stderr)
        print(f"Status Code: {response.status_code}", file=sys.stderr)
        
        try:
            error_data = response.json()
            print("\nError Response:", file=sys.stderr)
            print(json.dumps(error_data, indent=2), file=sys.stderr)
        except:
            print(f"Response Text: {response.text}", file=sys.stderr)
        
        sys.exit(1)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ REQUEST ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON DECODE ERROR: {e}", file=sys.stderr)
        print(f"Response Text: {response.text}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    check_whatsapp_health()
