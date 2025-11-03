import re
from typing import Tuple, Dict, Any, List

# Import the single source of truth for holiday keywords from opening_hours
# This prevents inconsistencies between intent detection and date parsing.
from llm.opening_hours import _HOLIDAY_KEYWORDS
from llm.config import SETTINGS  # NEW: use config flag instead of undefined `use_llm`

# Broad catch-all intent detection for opening hours, attendance, and arrangements
# Supports English, zh-HK (Traditional), zh-CN (Simplified)

# --- RESTRUCTURED TERMS FOR BETTER ACCURACY ---
# Strong terms are high-confidence indicators of opening-hours intent.
_EN_STRONG_TERMS = [
    r"\bopen(?:ing)?\b", r"\bhours?\b", r"\bclosed?\b", r"\bbusiness hours?\b",
    r"\battend(?:ing)?\s+(?:class|lesson)\b", r"go to class",
    r"\bpublic holiday\b", r"\bholiday\b",
]
# Weak terms are common but require other hints (like a time) to be confident.
_EN_WEAK_TERMS = [
    r"\btomorrow\b", r"\btoday\b",
    r"\b(?:next|this)\s+(?:week|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

_ZH_HK_STRONG_TERMS = [
    r"營業|營運|開放|開門|收(工|舖|店)|幾點(開|收)",
    r"上課|上堂|返學|返課",
    r"安排|改期",
    r"公眾假期|假期",
]
_ZH_HK_WEAK_TERMS = [
    r"今日|聽日|後日|下周|下星期|星期[一二三四五六日天]|周[一二三四五六日天]",
]

_ZH_CN_STRONG_TERMS = [
    r"营业|开放|开门|关门|几点(开|关)",
    r"上课|上学|上(?:不)?上课",
    r"安排|改期",
    r"公众假期|公休日|假期",
]
_ZH_CN_WEAK_TERMS = [
    r"今天|明天|后天|下周|星期[一二三四五六日天]|周[一二三四五六日天]",
]

# NEW: Availability / timetable / start-date inquiry markers (scheduling handled by admin)
_AVAIL_EN = [
    r"\bavailable\b", r"\bavailability\b", r"\bany (class|slot|time ?slot|timeslot)\b",
    r"\btimetable\b", r"\bschedule\b", r"\bwhat times\b", r"\bstart date\b",
    r"\bwhich time\b", r"\btime works\b", r"\bteacher availability\b",
]
_AVAIL_ZH_HK = [
    r"有冇(堂|時段|時間|位)", r"時間表", r"時間安排", r"檔期", r"可唔可以.*時間", r"幾時開始(上|開)課",
    r"老師(幾時|時間)有空|導師(幾時|時間)得閒|老師檔期|導師檔期",
]
_AVAIL_ZH_CN = [
    r"(有|有没有)(课|课程|时段|时间|名额)", r"时间表", r"课程安排", r"档期", r"可以.*时间", r"什么时候开始(上|开)课",
    r"(老师|教师)(什么时候|什么时间)有空|老师档期|教师档期",
]

# NEW: Post-assessment markers (indicates admin should check placements/timetable)
_POST_ASSESS_EN = [r"\bafter (the )?assessment\b", r"\bpost-?assessment\b", r"\bcompleted (the )?assessment\b"]
_POST_ASSESS_ZH_HK = [r"評估(之後|後)", r"完成(了)?評估", r"做完評估"]
_POST_ASSESS_ZH_CN = [r"评估(之后|后)", r"完成(了)?评估", r"做完评估"]

# NEW: Child/student reference markers to raise confidence it's a specific scheduling request
_STUDENT_REF_EN = [
    r"\bfor\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b",  # "for Owen", "for Owen Chan"
    r"\bmy (son|daughter|kid|child)\b", r"\bstudent\b", r"\bfor him\b", r"\bfor her\b",
]
_STUDENT_REF_ZH_HK = [r"為?(\S+)?(小朋友|小童|小孩|仔|女|學生)", r"我(個|的)?(仔|女|小朋友)", r"替.*(仔|女)"]
_STUDENT_REF_ZH_CN = [r"为?(\S+)?(小朋友|孩子|小孩|学生)", r"我(家)?(儿子|女儿|孩子)", r"替.*(儿子|女儿)"]

# Weather markers for adding the policy note
_WEATHER_MARKERS = [
    r"typhoon|rainstorm|t[13]|\bt8\b|black rain|amber|red",
    r"颱風|台风|風球|风球|黑雨|紅雨|红雨|黃雨|黄雨|三號|三号|一號|一号|八號|八号",
]

# Time-of-day hints (helps boost intent score)
_TIME_HINTS = [
    r"\b\d{1,2}:\d{2}\b", r"\b\d{1,2}\s*(am|pm)\b", r"\b(?:9|10|11|12|[1-8])\s*(?:am|pm)\b",
    r"[上下]午", r"\d點|\d点",
]

# --- FLATTENED HOLIDAY KEYWORDS FOR REGEX ---
# We now use the values from the _HOLIDAY_KEYWORDS dictionary in opening_hours.py
_ALL_HOLIDAY_KEYWORDS: List[str] = [keyword for sublist in _HOLIDAY_KEYWORDS.values() for keyword in sublist]
_HOLIDAY_TERMS_REGEX = [re.escape(term) for term in _ALL_HOLIDAY_KEYWORDS]


# Negative markers: if present, do NOT classify as opening-hours
# --- ADDED "homework" and related terms to prevent misclassification ---
_NEG_EN = [r"\b(tuition|fee|fees|price|cost)\b", r"\bclass\s*size\b", r"\bhomework\b", r"\bassignment\b"]
_NEG_ZH_HK = [r"學費|收費|費用|價錢|價格|班級人數|人數", r"功課|家課|作業"]
_NEG_ZH_CN = [r"学费|收费|费用|价钱|价格|班级人数|人数", r"功课|作业"]

def _score_regex(message: str, patterns: list[str]) -> Tuple[int, list[str]]:
    hits = []
    score = 0
    for pat in patterns:
        if re.search(pat, message, flags=re.IGNORECASE):
            hits.append(pat)
            score += 1
    return score, hits

def is_general_hours_query(message: str, lang: str) -> bool:
    """
    Returns True if the message is about opening hours/attendance intent,
    but does NOT contain any explicit date, weekday, relative-day, or named holiday marker.
    Used to distinguish 'What are your opening hours?' from 'Are you open on Christmas?'.
    """
    m = (message or "").lower()
    
    # Weekday and relative-day markers (in all languages)
    specific_date_patterns = [
        # Weekdays
        r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        # Relative days
        r"\b(today|tomorrow|yesterday|next week|this week)\b",
        # Chinese weekdays
        r"星期[一二三四五六日天]|周[一二三四五六日天]|週[一二三四五六日天]|礼拜[一二三四五六日天]|禮拜[一二三四五六日天]",
        # Chinese relative days
        r"今天|今日|明天|聽日|后天|後日|下周|下星期|本周|本星期",
        # Explicit numeric dates
        r"\d{1,2}\s*(月|日|号|號)",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}\b",
    ]
    for pat in specific_date_patterns:
        if re.search(pat, m, re.I):
            return False # Found a specific date marker

    # Check for named holidays by attempting to parse them
    # Import here to avoid circular imports
    from datetime import datetime
    import pytz
    from llm.opening_hours import _search_holiday_by_name
    
    HK_TZ = pytz.timezone("Asia/Hong_Kong")
    now = datetime.now(HK_TZ)
    
    # Try to parse the message as a holiday - if successful, it's specific
    try:
        holiday_match = _search_holiday_by_name(message, now)
        if holiday_match:
            return False  # Found a specific holiday
    except Exception:
        # If parsing fails, fall back to keyword matching
        pass
    
    # --- MAJOR FIX ---
    # Check for named holidays using keyword matching as fallback
    # We use the consolidated list imported from opening_hours.py
    for holiday_term in _ALL_HOLIDAY_KEYWORDS:
        # Use regex to match whole words for English terms to avoid partial matches like 'day' in 'today'
        if ' ' not in holiday_term and re.search(r'[a-zA-Z]', holiday_term):
             if re.search(r'\b' + re.escape(holiday_term) + r'\b', m, re.I):
                return False
        # For Chinese terms or multi-word English terms, simple substring search is fine
        elif holiday_term in m:
             return False

    # If no specific date, weekday, or holiday markers are found, it's a general query.
    return True

def detect_opening_hours_intent(message: str, lang: str) -> Tuple[bool, Dict[str, Any]]:
    """
    --- REVISED LOGIC ---
    Detects opening hours intent with higher precision by using strong/weak signals.
    """
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        strong_terms = _ZH_HK_STRONG_TERMS
        weak_terms = _ZH_HK_WEAK_TERMS
        neg_terms = _NEG_ZH_HK
    elif L.startswith("zh-cn") or L == "zh":
        strong_terms = _ZH_CN_STRONG_TERMS
        weak_terms = _ZH_CN_WEAK_TERMS
        neg_terms = _NEG_ZH_CN
    else:
        strong_terms = _EN_STRONG_TERMS
        weak_terms = _EN_WEAK_TERMS
        neg_terms = _NEG_EN

    strong_score, strong_hits = _score_regex(m, strong_terms)
    weak_score, weak_hits = _score_regex(m, weak_terms)
    time_score, time_hits = _score_regex(m, _TIME_HINTS)
    weather_score, weather_hits = _score_regex(m, _WEATHER_MARKERS)
    holiday_score, holiday_hits = _score_regex(m, _HOLIDAY_TERMS_REGEX)
    neg_score, neg_hits = _score_regex(m, neg_terms)

    # --- NEW SCORING RULE ---
    # An intent is triggered if:
    # 1. There are no negative markers (like 'homework', 'tuition fee').
    # AND
    # 2. At least one of the following is true:
    #    a) A strong signal is present ('open', 'hours', 'attend class').
    #    b) A named holiday is present ('Christmas', 'Lunar New Year').
    #    c) A weak signal ('today', 'Monday') is combined with a time hint ('9am').
    is_intent = neg_score == 0 and (
        strong_score > 0
        or holiday_score > 0
        or (weak_score > 0 and time_score > 0)
    )
    
    # The original score calculation is no longer used for the final decision,
    # but can be useful for debugging.
    total_score = strong_score + weak_score + time_score + holiday_score

    debug = {
        "score": total_score,
        "is_intent": is_intent,
        "strong_hits": strong_hits,
        "weak_hits": weak_hits,
        "time_hits": time_hits,
        "weather_hits": weather_hits,
        "holiday_hits": holiday_hits,
        "neg_hits": neg_hits,
        "used_llm": False,
        "llm_confidence": None,
    }

    # FIX: Replace undefined `use_llm` with configured flag (kept as placeholder for future use)
    if SETTINGS.opening_hours_use_llm_intent and not is_intent:
        # Reserved for future LLM assist
        pass

    return is_intent, debug

def mentions_weather(message: str) -> bool:
    for pat in _WEATHER_MARKERS:
        if re.search(pat, message or "", flags=re.IGNORECASE):
            return True
    return False

def mentions_attendance(message: str, lang: str) -> bool:
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return bool(re.search(r"上課|上堂|返學|返課", m))
    if L.startswith("zh-cn") or L == "zh":
        return bool(re.search(r"上课|上学", m))
    return bool(re.search(r"\battend(?:ing)?\s+(?:class|lesson)\b", m, flags=re.IGNORECASE))

# Lightweight “soft” classifiers used only to steer the prompt (not to gate)
_POLITENESS_ONLY_EN = re.compile(r"^\s*(thanks|thank you|ty|thx|appreciate(?: it)?|cheers)\s*[.!]*\s*$", re.I)
_POLITENESS_ONLY_ZH_HK = re.compile(r"^\s*(多謝|謝謝|唔該|唔該晒)\s*[！!。.\s]*$", re.I)
_POLITENESS_ONLY_ZH_CN = re.compile(r"^\s*(谢谢|多谢|辛苦了|麻烦了)\s*[！!。.\s]*$", re.I)

# Scheduling/leave verbs
_SCHED_ZH_HK = [r"請假", r"改期", r"改時間", r"改堂", r"取消", r"缺席", r"退堂"]
_SCHED_ZH_CN = [r"请假", r"改期", r"改时间", r"改堂", r"取消", r"缺席", r"退课"]
_SCHED_EN = [r"\breschedul(?:e|ing)\b", r"\bcancel(?:ling|ation)?\b", r"\btake\s+leave\b", r"\brequest\s+leave\b", r"\b(absent|absence)\b"]

# Date/time markers
_DATE_MARKERS = [
    r"\b\d{1,2}/\d{1,2}\b", r"\d{1,2}\s*(月|日|号|號)", r"(星期|周|週)[一二三四五六日天]",
    r"\b(mon|tue|wed|thu|fri|sat|sun)\b", r"\b(today|tomorrow)\b",
    r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b",
    r"今天|今日|明天|後日|后天|聽日|下(周|星期|週)",
]

# Policy markers (English and Chinese)
_POLICY_EN = [r"\bpolicy\b", r"\bwhat\s+is\s+the\s+policy\b", r"\brules?\b", r"\bhow\s+do(?:es)?\s+.*(reschedul|make-?up|absence)\b"]
_POLICY_ZH_HK = [r"政策", r"安排", r"規則", r"規定", r"補課政策", r"請假政策", r"改期政策", r"補課安排", r"請假安排", r"改期安排"]
_POLICY_ZH_CN = [r"政策", r"安排", r"规则", r"规定", r"补课政策", r"请假政策", r"改期政策", r"补课安排", r"请假安排", r"改期安排"]

def _score(m: str, pats: List[str]) -> int:
    return sum(bool(re.search(p, m, re.I)) for p in pats)

def is_politeness_only(message: str, lang: str) -> bool:
    m = (message or "").strip()
    L = (lang or "en").lower()
    if L.startswith("zh-hk"): return bool(_POLITENESS_ONLY_ZH_HK.match(m))
    if L.startswith("zh-cn") or L == "zh": return bool(_POLITENESS_ONLY_ZH_CN.match(m))
    return bool(_POLITENESS_ONLY_EN.match(m))

def classify_scheduling_context(message: str, lang: str) -> Dict[str, Any]:
    """
    Soft classification: returns booleans used to steer prompt only.
    - has_sched_verbs: mentions leave/reschedule/cancel
    - has_date_time: mentions specific date/weekday/time
    - has_policy_intent: asks for policy/arrangements/rules around reschedule/leave
    - availability_request: availability/timetable/slot/start-date/teacher availability
    - post_assessment: mentions 'after/completed assessment'
    - student_ref: refers to a specific child (name/pronoun/son/daughter)
    - politeness_only: message is pure politeness (no other content)
    """
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        base_sched = _score(m, _SCHED_ZH_HK) > 0
        policy = _score(m, _POLICY_ZH_HK) > 0
        avail = _score(m, _AVAIL_ZH_HK) > 0
        post = _score(m, _POST_ASSESS_ZH_HK) > 0
        student = _score(m, _STUDENT_REF_ZH_HK) > 0
    elif L.startswith("zh-cn") or L == "zh":
        base_sched = _score(m, _SCHED_ZH_CN) > 0
        policy = _score(m, _POLICY_ZH_CN) > 0
        avail = _score(m, _AVAIL_ZH_CN) > 0
        post = _score(m, _POST_ASSESS_ZH_CN) > 0
        student = _score(m, _STUDENT_REF_ZH_CN) > 0
    else:
        base_sched = _score(m, _SCHED_EN) > 0
        policy = _score(m, _POLICY_EN) > 0
        avail = _score(m, _AVAIL_EN) > 0
        post = _score(m, _POST_ASSESS_EN) > 0
        student = _score(m, _STUDENT_REF_EN) > 0

    date_time = _score(m, _DATE_MARKERS) > 0
    politeness = is_politeness_only(m, lang)

    # KEY: Treat availability+(post-assessment OR student-ref) as an admin scheduling request
    sched = base_sched or (avail and (post or student))

    return {
        "has_sched_verbs": sched,
        "has_date_time": date_time,
        "has_policy_intent": policy,
        "availability_request": avail,
        "post_assessment": post,
        "student_ref": student,
        "politeness_only": politeness,
    }