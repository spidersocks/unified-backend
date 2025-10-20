import os
import json
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_JSON_PATH = os.path.join(BASE_DIR, "data", "little_scholars_info.json")

# Load Little Scholars info JSON once at import
INFO_DATA = {}
try:
    with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
        INFO_DATA = json.load(f)
    print("[INFO] Loaded little_scholars_info.json", flush=True)
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load little_scholars_info.json: {e}", flush=True)
    INFO_DATA = {}

def _norm_lang(lang_code: str) -> str:
    if not lang_code:
        return "en"
    lc = lang_code.lower()
    if lc.startswith("zh-hk"):
        return "zh-hk"
    if lc.startswith("zh-cn") or lc == "zh":
        return "zh-cn"
    return "en"

def _join_lines(lines):
    return "\n".join([line for line in lines if line and line.strip()])

def get_admin_redirect(lang_code: str) -> str:
    lang = _norm_lang(lang_code)
    contact = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("ContactInformation", {})
    phones = contact.get("Phone", [])
    email = contact.get("Email", "")
    address = contact.get("Address", "")

    # Try to extract WhatsApp number if present
    whatsapp = contact.get("SocialMedia", {}).get("WhatsApp", "")
    website = contact.get("SocialMedia", {}).get("OfficialWebsite", "")

    if lang == "zh-hk":
        msg = [
            "此問題超出機械人可解答範圍，請聯絡我們的職員：",
            f"電話：{'; '.join(phones) if phones else ''}".strip(),
            f"WhatsApp：{whatsapp}".strip() if whatsapp else "",
            f"電郵：{email}".strip() if email else "",
            f"地址：{address}".strip() if address else "",
            f"網站：{website}".strip() if website else "",
        ]
    elif lang == "zh-cn":
        msg = [
            "此问题超出机器人可解答范围，请联系我们的职员：",
            f"电话：{'; '.join(phones) if phones else ''}".strip(),
            f"WhatsApp：{whatsapp}".strip() if whatsapp else "",
            f"电邮：{email}".strip() if email else "",
            f"地址：{address}".strip() if address else "",
            f"网站：{website}".strip() if website else "",
        ]
    else:
        msg = [
            "This question is beyond the bot’s current scope. Please contact our staff:",
            f"Phone: {', '.join(phones) if phones else ''}".strip(),
            f"WhatsApp: {whatsapp}".strip() if whatsapp else "",
            f"Email: {email}".strip() if email else "",
            f"Address: {address}".strip() if address else "",
            f"Website: {website}".strip() if website else "",
        ]
    return _join_lines(msg)

def _institution_intro(lang: str) -> Optional[str]:
    intro = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("InstitutionIntroduction", {})
    full = intro.get("FullInstitutionName", "")
    phil = intro.get("FoundingPhilosophy", "")
    edu = intro.get("EducationalPhilosophy", "")
    if lang == "zh-hk":
        return _join_lines([
            f"機構名稱：{full}" if full else "",
            "我們的理念：" if phil else "",
            phil,
            "教育理念：" if edu else "",
            edu,
        ])
    elif lang == "zh-cn":
        return _join_lines([
            f"机构名称：{full}" if full else "",
            "我们的理念：" if phil else "",
            phil,
            "教育理念：" if edu else "",
            edu,
        ])
    else:
        return _join_lines([
            f"Institution: {full}" if full else "",
            "Founding philosophy:" if phil else "",
            phil,
            "Educational philosophy:" if edu else "",
            edu,
        ])

def _contact_info(lang: str) -> Optional[str]:
    contact = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("ContactInformation", {})
    phones = contact.get("Phone", [])
    email = contact.get("Email", "")
    address = contact.get("Address", "")
    social = contact.get("SocialMedia", {})

    if lang == "zh-hk":
        return _join_lines([
            "聯絡資料：",
            f"電話：{'; '.join(phones)}" if phones else "",
            f"電郵：{email}" if email else "",
            f"地址：{address}" if address else "",
            "社交平台：",
            f"- Facebook：{social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram：{social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp：{social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- 官方網站：{social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
            f"- 地圖：{social.get('MapLink','')}" if social.get("MapLink") else "",
        ])
    elif lang == "zh-cn":
        return _join_lines([
            "联系资料：",
            f"电话：{'; '.join(phones)}" if phones else "",
            f"电邮：{email}" if email else "",
            f"地址：{address}" if address else "",
            "社交平台：",
            f"- Facebook：{social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram：{social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp：{social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- 官网：{social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
            f"- 地图：{social.get('MapLink','')}" if social.get("MapLink") else "",
        ])
    else:
        return _join_lines([
            "Contact information:",
            f"Phone: {', '.join(phones)}" if phones else "",
            f"Email: {email}" if email else "",
            f"Address: {address}" if address else "",
            "Social:",
            f"- Facebook: {social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram: {social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp: {social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- Website: {social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
            f"- Map: {social.get('MapLink','')}" if social.get("MapLink") else "",
        ])

def _enrollment_process(lang: str) -> Optional[str]:
    steps = INFO_DATA.get("CourseDetails", {}).get("EnrollmentProcess", {}).get("Steps", [])
    if not steps:
        return None
    if lang == "zh-hk":
        header = "報名流程："
    elif lang == "zh-cn":
        header = "报名流程："
    else:
        header = "Enrollment process:"
    bullet = "\n".join([f"- {s}" for s in steps])
    return f"{header}\n{bullet}"

# Map canonical CourseName -> keys in JSON sections where necessary
COURSE_KEY_MAP = {
    "Playgroups": "Playgroups",
    "Phonics": "English",  # Fees/sections for English tracks
    "LanguageArts": "English",  # Use English group where applicable
    "Clevercal": "Clevercal (Math)",
    "Alludio": "Alludio (Games)",
    "ToddlerCharRecognition": "寶寶愛認字(普通話/ 廣東話)",
    "MandarinPinyin": "魔法拼音班",
    "ChineseLanguageArts": "中文語文課(普通話/ 廣東話)",
    "PrivateClass": "Private class 私人課",
}

def _course_age(lang: str, coursename: Optional[str]) -> Optional[str]:
    ages = INFO_DATA.get("CourseDetails", {}).get("TargetStudentAge", {})
    if not ages:
        return None
    if not coursename:
        # Summarize all
        lines = []
        for k, v in ages.items():
            lines.append(f"- {k}: {v}")
        title = "各課程年齡：" if lang == "zh-hk" else ("各课程年龄：" if lang == "zh-cn" else "Target ages:")
        return f"{title}\n" + "\n".join(lines)

    key = COURSE_KEY_MAP.get(coursename)
    if not key:
        return None
    val = ages.get(key)
    if not val:
        return None
    if lang == "zh-hk":
        return f"{key} 適合年齡：{val}"
    elif lang == "zh-cn":
        return f"{key} 适合年龄：{val}"
    else:
        return f"{key} target age: {val}"

def _course_schedule(lang: str, coursename: Optional[str]) -> Optional[str]:
    sched = INFO_DATA.get("CourseDetails", {}).get("ClassSchedule", {})
    if not sched:
        return None
    if not coursename:
        lines = []
        for k, v in sched.items():
            lines.append(f"- {k}: {v}")
        title = "上課時間：" if lang == "zh-hk" else ("上课时间：" if lang == "zh-cn" else "Class schedules:")
        return f"{title}\n" + "\n".join(lines)

    key = COURSE_KEY_MAP.get(coursename)
    if not key:
        return None
    val = sched.get(key)
    if not val:
        return None
    if lang == "zh-hk":
        return f"{key} 上課安排：{val}"
    elif lang == "zh-cn":
        return f"{key} 上课安排：{val}"
    else:
        return f"{key} schedule: {val}"

def _class_size(lang: str) -> Optional[str]:
    cs = INFO_DATA.get("CourseDetails", {}).get("ClassSize", {})
    if not cs:
        return None
    lines = [f"- {k}: {v}" for k, v in cs.items()]
    title = "班級人數：" if lang == "zh-hk" else ("班级人数：" if lang == "zh-cn" else "Class size:")
    return f"{title}\n" + "\n".join(lines)

def _tuition(lang: str, coursename: Optional[str]) -> Optional[str]:
    fees = INFO_DATA.get("CourseDetails", {}).get("TuitionAndPayment", {}).get("GroupClassFee", {})
    if not fees:
        return None

    disclaimer_en = "Note: Fees are indicative and may change. Please confirm with our staff."
    disclaimer_zh_hk = "備註：以上費用僅供參考，或有調整；以中心最新通知為準。"
    disclaimer_zh_cn = "备注：以上费用仅供参考，或有调整；以中心最新通知为准。"

    if coursename:
        key = COURSE_KEY_MAP.get(coursename)
        if key and key in fees:
            val = fees.get(key)
            if lang == "zh-hk":
                return _join_lines([f"{key} 收費：{val}", disclaimer_zh_hk])
            elif lang == "zh-cn":
                return _join_lines([f"{key} 收费：{val}", disclaimer_zh_cn])
            else:
                return _join_lines([f"{key} fee: {val}", disclaimer_en])

    # Otherwise list all
    lines = [f"- {k}: {v}" for k, v in fees.items()]
    title = "課程收費：" if lang == "zh-hk" else ("课程收费：" if lang == "zh-cn" else "Tuition:")
    body = f"{title}\n" + "\n".join(lines)
    if lang == "zh-hk":
        return _join_lines([body, disclaimer_zh_hk])
    elif lang == "zh-cn":
        return _join_lines([body, disclaimer_zh_cn])
    else:
        return _join_lines([body, disclaimer_en])

def _trial_class(lang: str) -> Optional[str]:
    txt = INFO_DATA.get("PoliciesAndFAQs", {}).get("TrialClassPolicy", {}).get("Description", "")
    return txt or None

def _absence_makeup_policy(lang: str) -> Optional[str]:
    pol = INFO_DATA.get("PoliciesAndFAQs", {}).get("AbsenceAndMakeupPolicy", {})
    if not pol:
        return None
    # Join all key points
    order = [
        "NotificationRequirement", "MakeupArrangement", "WrittenNotificationRequirement",
        "MedicalCertificateRequirement", "NoNoticeMissedClasses", "MakeupDeadline",
        "InstructorTimeGuarantee", "FreeMakeupQuota", "ExceedingFreeMakeup",
        "ShortTermAbsence", "LongTermAbsence", "CourseValidity"
    ]
    lines = [f"- {pol[k]}" for k in order if pol.get(k)]
    title = "補課與請假政策：" if lang == "zh-hk" else ("补课与请假政策：" if lang == "zh-cn" else "Absence and Make-up policy:")
    return f"{title}\n" + "\n".join(lines)

def _refund_transfer_policy(lang: str) -> Optional[str]:
    pol = INFO_DATA.get("PoliciesAndFAQs", {}).get("RefundAndClassTransferPolicy", {})
    if not pol:
        return None
    order = ["RefundPolicy", "ContinuationAssumption", "WithdrawalNotice", "ClassTransferInquiry"]
    lines = [f"- {pol[k]}" for k in order if pol.get(k)]
    title = "退款與轉班政策：" if lang == "zh-hk" else ("退款与转班政策：" if lang == "zh-cn" else "Refund and transfer policy:")
    return f"{title}\n" + "\n".join(lines)

def _common_obj(lang: str, key: str, zh_hk_title: str, zh_cn_title: str, en_title: str) -> Optional[str]:
    box = INFO_DATA.get("PoliciesAndFAQs", {}).get("CommonObjectionHandling", {})
    val = box.get(key)
    if not val:
        return None
    title = zh_hk_title if lang == "zh-hk" else (zh_cn_title if lang == "zh-cn" else en_title)
    return f"{title}\n- {val}"

def _promotions(lang: str) -> Optional[str]:
    txt = INFO_DATA.get("PoliciesAndFAQs", {}).get("PromotionsAndEvents", {}).get("Description", "")
    if not txt:
        # fallback to marketing section if populated
        txt = INFO_DATA.get("MarketingAndPromotionInfo", {}).get("PromotionsAndEvents", "")
    if not txt:
        return None
    return txt

def _success_stories(lang: str) -> Optional[str]:
    succ = INFO_DATA.get("PoliciesAndFAQs", {}).get("SuccessStories", {})
    if not succ:
        succ = INFO_DATA.get("MarketingAndPromotionInfo", {}).get("SuccessStories", {})
    if not succ:
        return None
    lines = [f"- {k}: {v}" for k, v in succ.items()]
    title = "成功個案：" if lang == "zh-hk" else ("成功个案：" if lang == "zh-cn" else "Success stories:")
    return f"{title}\n" + "\n".join(lines)

def _social_media(lang: str) -> Optional[str]:
    contact = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("ContactInformation", {})
    social = contact.get("SocialMedia", {})
    if not social:
        return None
    if lang == "zh-hk":
        title = "社交平台："
        lines = [
            f"- Facebook：{social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram：{social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp：{social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- 官方網站：{social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
        ]
    elif lang == "zh-cn":
        title = "社交平台："
        lines = [
            f"- Facebook：{social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram：{social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp：{social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- 官网：{social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
        ]
    else:
        title = "Social media:"
        lines = [
            f"- Facebook: {social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram: {social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp: {social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- Website: {social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
        ]
    return _join_lines([title] + [l for l in lines if l])

TOPIC_HANDLERS = {
    # Organization and contact
    "InstitutionIntroduction": lambda lang, _: _institution_intro(lang),
    "ContactInformation":      lambda lang, _: _contact_info(lang),
    "SocialMedia":             lambda lang, _: _social_media(lang),

    # Enrollment and logistics
    "EnrollmentProcess":       lambda lang, _: _enrollment_process(lang),
    "ClassSize":               lambda lang, _: _class_size(lang),

    # Course-specific
    "TargetStudentAge":        lambda lang, course: _course_age(lang, course),
    "ClassSchedule":           lambda lang, course: _course_schedule(lang, course),
    "TuitionAndPayment":       lambda lang, course: _tuition(lang, course),
    "Tuition":                 lambda lang, course: _tuition(lang, course),

    # Policies & FAQs
    "TrialClassPolicy":        lambda lang, _: _trial_class(lang),
    "AbsenceAndMakeupPolicy":  lambda lang, _: _absence_makeup_policy(lang),
    "RefundAndClassTransferPolicy": lambda lang, _: _refund_transfer_policy(lang),
    "TeacherNativeSpeaker":    lambda lang, _: _common_obj(lang, "TeacherNativeSpeaker",
                                                           "老師是否母語老師？", "老师是否母语老师？", "Are teachers native speakers?"),
    "PublicHolidaySchedule":   lambda lang, _: _common_obj(lang, "PublicHolidaySchedule",
                                                           "公眾假期安排", "公众假期安排", "Public holiday schedule"),
    "WeatherPolicy":           lambda lang, _: _common_obj(lang, "WeatherPolicy",
                                                           "天氣安排", "天气安排", "Weather policy"),
    "OtherCenters":            lambda lang, _: _common_obj(lang, "OtherCenters",
                                                           "有沒有其他分校？", "有没有其他分校？", "Do you have other centers?"),
    "PromotionsAndEvents":     lambda lang, _: _promotions(lang),
    "SuccessStories":          lambda lang, _: _success_stories(lang),
}

def get_info(topic: Optional[str], lang_code: str, coursename: Optional[str] = None) -> Optional[str]:
    """
    Returns an info text for a given topic and optional course name.
    topic: canonical value from @InfoTopic
    coursename: canonical value from @CourseName (optional)
    """
    if not topic:
        return None
    lang = _norm_lang(lang_code)
    handler = TOPIC_HANDLERS.get(topic)
    if not handler:
        return None
    try:
        return handler(lang, coursename)
    except Exception as e:
        print(f"[ERROR] get_info failed for topic={topic}, course={coursename}: {e}", flush=True)
        return None