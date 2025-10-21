# C:\Users\sfont\unified_backend\dialogflow_app\router.py

from fastapi import APIRouter, Request
from dialogflow_app.data import get_course_info, get_display_name, get_course_list
from dialogflow_app.info import get_info, get_admin_redirect

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
        return {"fulfillmentText": "Internal server error: Invalid request format."}

    query_result = req_json.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName")
    language_code = query_result.get("languageCode", "en").lower()
    parameters = query_result.get("parameters", {}) or {}

    fulfillment_text = ""

    if intent_name == "Course_Inquiry":
        canonical_course_name = parameters.get("coursename")

        if not canonical_course_name:
            if language_code.startswith('zh'):
                fulfillment_text = "抱歉，我沒有識別出您詢問的課程名稱。"
            else:
                fulfillment_text = "I'm sorry, I didn't catch the name of the course you were asking about."
        else:
            display_course_name = get_display_name(canonical_course_name, language_code)
            course_details = get_course_info(canonical_course_name, language_code)

            if language_code.startswith('en'):
                fulfillment_text = f"Details for the *{display_course_name}* course: {course_details}"
            elif language_code == 'zh-hk':
                fulfillment_text = f"*{display_course_name}* 課程詳情: {course_details}"
            elif language_code == 'zh-cn':
                fulfillment_text = f"*{display_course_name}* 课程详情: {course_details}"
            else:
                fulfillment_text = f"Details for the *{display_course_name}* course: {course_details}"

    elif intent_name == "Course_List":
        fulfillment_text = get_course_list(language_code)

    elif intent_name == "Policy_MakeUp_Quota":
        POLICY_CANONICAL_NAME = "Policy_MakeUp_Quota"
        policy_text = get_course_info(POLICY_CANONICAL_NAME, language_code)
        if policy_text.startswith("Sorry, I could not find details"):
            if language_code.startswith('zh'):
                fulfillment_text = "抱歉，目前找不到補課政策的詳細資訊，請聯繫我們的行政人員。"
            else:
                fulfillment_text = "Sorry, the makeup class policy details are currently unavailable. Please contact our administrative staff."
        else:
            fulfillment_text = policy_text

    # --- New granular info intents mapped to structured topics ---
    elif intent_name == "Info_Contact":
        fulfillment_text = get_info("ContactInformation", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_Social":
        fulfillment_text = get_info("SocialMedia", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_Intro":
        fulfillment_text = get_info("InstitutionIntroduction", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_WeatherPolicy":
        fulfillment_text = get_info("WeatherPolicy", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_PublicHoliday":
        fulfillment_text = get_info("PublicHolidaySchedule", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_TrialClass":
        fulfillment_text = get_info("TrialClassPolicy", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_MakeupPolicy":
        fulfillment_text = get_info("AbsenceAndMakeupPolicy", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_RefundPolicy":
        fulfillment_text = get_info("RefundAndClassTransferPolicy", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_ClassSize":
        fulfillment_text = get_info("ClassSize", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_Enrollment":
        fulfillment_text = get_info("EnrollmentProcess", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_Promotions":
        fulfillment_text = get_info("PromotionsAndEvents", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_SuccessStories":
        fulfillment_text = get_info("SuccessStories", language_code) or get_admin_redirect(language_code)

    # Course-qualified info
    elif intent_name == "Info_Age":
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("TargetStudentAge", language_code, coursename) or get_admin_redirect(language_code)

    elif intent_name == "Info_Schedule":
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("ClassSchedule", language_code, coursename) or get_admin_redirect(language_code)

    elif intent_name == "Info_Tuition":
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("TuitionAndPayment", language_code, coursename) or get_admin_redirect(language_code)

    # Keep generic catch-all if still present
    elif intent_name == "Info_Query":
        topic = parameters.get("topic")
        coursename = parameters.get("coursename")
        answer = get_info(topic, language_code, coursename)
        fulfillment_text = answer or get_admin_redirect(language_code)

    else:
        if language_code.startswith('zh'):
            fulfillment_text = "抱歉，我目前無法處理這個請求。"
        else:
            fulfillment_text = "I received a request, but I don't have business logic for that intent yet."

    return {
        "fulfillmentText": fulfillment_text,
    }