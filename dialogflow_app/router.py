# C:\Users\sfont\unified_backend\dialogflow_app\router.py

from fastapi import APIRouter, Request
from dialogflow_app.info import get_info, get_admin_redirect
from dialogflow_app.data import get_course_info, format_course_display

router = APIRouter(
    prefix="/dialogflow",
    tags=["Dialogflow Webhook"],
)

@router.post("/webhook")
async def dialogflow_webhook(request: Request):
    try:
        req_json = await request.json()
    except Exception:
        return {"fulfillmentText": "Internal server error: Invalid request format."}

    query_result = req_json.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName")
    language_code = (query_result.get("languageCode") or "en").lower()
    parameters = query_result.get("parameters", {}) or {}

    fulfillment_text = ""

    if intent_name == "Course_Inquiry":
        canonical_course_name = parameters.get("coursename")

        if not canonical_course_name:
            # Dialogflow should slot-fill and only call webhook once this is present.
            # Provide a minimal defensive prompt if it ever reaches here.
            if language_code.startswith("zh-hk"):
                return {"fulfillmentText": "想查詢邊個課程？例如：數學班（Clevercal）、英語拼音、Playgroups。"}
            elif language_code.startswith("zh-cn") or language_code == "zh":
                return {"fulfillmentText": "想查询哪个课程？例如：数学班（Clevercal）、英语拼音、Playgroups。"}
            else:
                return {"fulfillmentText": "Which course would you like details for? For example: Clevercal, Phonics, Playgroups."}

        display_course_name = format_course_display(canonical_course_name, language_code)
        course_details = get_course_info(canonical_course_name, language_code)

        if language_code.startswith('en'):
            fulfillment_text = f"Details for the *{display_course_name}* course: {course_details}"
        elif language_code == 'zh-hk':
            fulfillment_text = f"*{display_course_name}* 課程詳情: {course_details}"
        elif language_code == 'zh-cn':
            fulfillment_text = f"*{display_course_name}* 课程详情: {course_details}"
        else:
            fulfillment_text = f"Details for the *{display_course_name}* course: {course_details}"

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
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("ClassSize", language_code, coursename) or get_admin_redirect(language_code)

    elif intent_name == "Info_Enrollment":
        fulfillment_text = get_info("EnrollmentProcess", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_Promotions":
        fulfillment_text = get_info("PromotionsAndEvents", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_SuccessStories":
        fulfillment_text = get_info("SuccessStories", language_code) or get_admin_redirect(language_code)

    elif intent_name == "Info_Age":
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("TargetStudentAge", language_code, coursename) or get_admin_redirect(language_code)

    elif intent_name == "Info_Schedule":
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("ClassSchedule", language_code, coursename) or get_admin_redirect(language_code)

    elif intent_name == "Info_Tuition":
        coursename = parameters.get("coursename")
        fulfillment_text = get_info("TuitionAndPayment", language_code, coursename) or get_admin_redirect(language_code)

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

    return { "fulfillmentText": fulfillment_text }