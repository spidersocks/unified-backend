# C:\Users\sfont\unified_backend\dialogflow_app\router.py

from fastapi import APIRouter, Request
from dialogflow_app.info import get_info, get_admin_redirect
# Migrate to ContentStore accessors (course descriptions and display names)
from llm.content_store import get_course_info, format_course_display, STORE

router = APIRouter(
    prefix="/dialogflow",
    tags=["Dialogflow Webhook"],
)

def _after_colon(s: str) -> str:
    if not s:
        return s
    parts = s.split(":", 1)
    return parts[1].strip() if len(parts) == 2 else s

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

    if intent_name == "Course_Compare":
        course_a = parameters.get("courseA")
        course_b = parameters.get("courseB")

        if not course_a or not course_b:
            # Should be slot-filled by DF, but be defensive.
            if language_code.startswith("zh-hk"):
                return {"fulfillmentText": "想比較邊兩個課程？例如：Phonics 同 英語語文課。"}
            elif language_code.startswith("zh-cn") or language_code == "zh":
                return {"fulfillmentText": "想比较哪两个课程？例如：Phonics 和 英语语文课。"}
            else:
                return {"fulfillmentText": "Which two courses would you like to compare? For example: Phonics and Language Arts."}

        name_a = format_course_display(course_a, language_code)
        name_b = format_course_display(course_b, language_code)

        # Descriptions
        desc_a = get_course_info(course_a, language_code)
        desc_b = get_course_info(course_b, language_code)

        # Other aspects from ContentStore (strip leading course name labels)
        age_a = _after_colon(STORE.age_for_course(course_a, language_code) or "")
        age_b = _after_colon(STORE.age_for_course(course_b, language_code) or "")
        sched_a = _after_colon(STORE.schedule_for_course(course_a, language_code) or "")
        sched_b = _after_colon(STORE.schedule_for_course(course_b, language_code) or "")
        size_a = _after_colon(STORE.class_size_for_course(course_a, language_code) or "")
        size_b = _after_colon(STORE.class_size_for_course(course_b, language_code) or "")
        fee_a = _after_colon(STORE.tuition_for_course(course_a, language_code) or "")
        fee_b = _after_colon(STORE.tuition_for_course(course_b, language_code) or "")

        if language_code.startswith('zh-hk'):
            lines = [
                f"課程比較：{name_a} vs {name_b}",
                f"*{name_a}*",
                f"- 簡介：{desc_a}" if desc_a else "",
                f"- 適合年齡：{age_a}" if age_a else "",
                f"- 上課時間：{sched_a}" if sched_a else "",
                f"- 班級人數：{size_a}" if size_a else "",
                f"- 收費：{fee_a}" if fee_a else "",
                "",
                f"*{name_b}*",
                f"- 簡介：{desc_b}" if desc_b else "",
                f"- 適合年齡：{age_b}" if age_b else "",
                f"- 上課時間：{sched_b}" if sched_b else "",
                f"- 班級人數：{size_b}" if size_b else "",
                f"- 收費：{fee_b}" if fee_b else "",
            ]
        elif language_code.startswith('zh-cn') or language_code == 'zh':
            lines = [
                f"课程比较：{name_a} vs {name_b}",
                f"*{name_a}*",
                f"- 简介：{desc_a}" if desc_a else "",
                f"- 适合年龄：{age_a}" if age_a else "",
                f"- 上课时间：{sched_a}" if sched_a else "",
                f"- 班级人数：{size_a}" if size_a else "",
                f"- 收费：{fee_a}" if fee_a else "",
                "",
                f"*{name_b}*",
                f"- 简介：{desc_b}" if desc_b else "",
                f"- 适合年龄：{age_b}" if age_b else "",
                f"- 上课时间：{sched_b}" if sched_b else "",
                f"- 班级人数：{size_b}" if size_b else "",
                f"- 收费：{fee_b}" if fee_b else "",
            ]
        else:
            lines = [
                f"Comparison: {name_a} vs {name_b}",
                f"*{name_a}*",
                f"- Overview: {desc_a}" if desc_a else "",
                f"- Target age: {age_a}" if age_a else "",
                f"- Schedule: {sched_a}" if sched_a else "",
                f"- Class size: {size_a}" if size_a else "",
                f"- Tuition: {fee_a}" if fee_a else "",
                "",
                f"*{name_b}*",
                f"- Overview: {desc_b}" if desc_b else "",
                f"- Target age: {age_b}" if age_b else "",
                f"- Schedule: {sched_b}" if sched_b else "",
                f"- Class size: {size_b}" if size_b else "",
                f"- Tuition: {fee_b}" if fee_b else "",
            ]
        fulfillment_text = "\n".join([l for l in lines if l])

    elif intent_name == "Course_Inquiry":
        canonical_course_name = parameters.get("coursename")

        if not canonical_course_name:
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