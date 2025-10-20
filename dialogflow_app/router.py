# C:\Users\sfont\unified_backend\dialogflow_app\router.py

from fastapi import APIRouter, Request
# Import the necessary functions
from dialogflow_app.data import get_course_info, get_display_name, get_course_list 

router = APIRouter(
    prefix="/dialogflow",
    tags=["Dialogflow Webhook"],
)

@router.post("/webhook")
async def dialogflow_webhook(request: Request):
    """
    Handles the POST request from Dialogflow Fulfillment.
    """
    try:
        req_json = await request.json()
    except Exception:
        # Should not happen with Dialogflow, but good practice
        return {"fulfillmentText": "Internal server error: Invalid request format."}

    # Extract core data
    query_result = req_json.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName")
    # Note: Dialogflow sends the languageCode in the queryResult object
    language_code = query_result.get("languageCode", "en").lower()
    
    fulfillment_text = ""

    # --- Intent Handling ---
    
    if intent_name == "Course_Inquiry":
        parameters = query_result.get("parameters", {})
        
        canonical_course_name = parameters.get("coursename")
        
        if not canonical_course_name:
            if language_code.startswith('zh'):
                fulfillment_text = "抱歉，我沒有識別出您詢問的課程名稱。"
            else:
                fulfillment_text = "I'm sorry, I didn't catch the name of the course you were asking about."
        else:
            # Get the user-friendly display name
            display_course_name = get_display_name(canonical_course_name, language_code)
            
            # Look up the specific course details in the correct language
            course_details = get_course_info(canonical_course_name, language_code)
            
            # Construct the final response using the user-friendly display name
            if language_code.startswith('en'):
                fulfillment_text = f"Details for the *{display_course_name}* course: {course_details}"
            elif language_code == 'zh-hk':
                fulfillment_text = f"*{display_course_name}* 課程詳情: {course_details}"
            elif language_code == 'zh-cn':
                fulfillment_text = f"*{display_course_name}* 课程详情: {course_details}"
            else:
                # Fallback language
                fulfillment_text = f"Details for the *{display_course_name}* course: {course_details}"


    elif intent_name == "Course_List":
        # Dynamic fulfillment for Course_List
        fulfillment_text = get_course_list(language_code)

    
    elif intent_name == "Policy_MakeUp_Quota":
        # Define the canonical name used in the Google Sheet for this policy
        POLICY_CANONICAL_NAME = "Policy_MakeUp_Quota" 
        
        # Retrieve the policy text using the existing data lookup function
        policy_text = get_course_info(POLICY_CANONICAL_NAME, language_code)
        
        # Check if the policy was found (get_course_info handles localization and fallback)
        if policy_text.startswith("Sorry, I could not find details"):
            # Provide a generic fallback if the data is missing entirely
            if language_code.startswith('zh'):
                fulfillment_text = "抱歉，目前找不到補課政策的詳細資訊，請聯繫我們的行政人員。"
            else:
                fulfillment_text = "Sorry, the makeup class policy details are currently unavailable. Please contact our administrative staff."
        else:
            fulfillment_text = policy_text

    else:
        # Fallback for unhandled intents
        if language_code.startswith('zh'):
            fulfillment_text = "抱歉，我目前無法處理這個請求。"
        else:
            fulfillment_text = "I received a request, but I don't have business logic for that intent yet."

    # Construct the final Dialogflow response object
    dialogflow_response = {
        "fulfillmentText": fulfillment_text,
        # Note: Dialogflow automatically handles the WhatsApp formatting from fulfillmentText
    }

    return dialogflow_response