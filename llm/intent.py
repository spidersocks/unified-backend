import re
from typing import Tuple, Dict, Any, List

# Single source of truth for holiday keywords (keeps hours-intent consistent with date parsing)
from llm.opening_hours import _HOLIDAY_KEYWORDS
from llm.config import SETTINGS

# ============================================================
# Opening-hours intent detection and soft classifiers
# ============================================================

# Strong terms: high-confidence indicators of opening-hours intent
_EN_STRONG_TERMS = [
    r"\bopen(?:ing)?\b", r"\bhours?\b", r"\bclosed?\b", r"\bbusiness hours?\b",
    r"\bpublic holiday\b", r"\bholiday\b",
]
# Weak terms: require time or other hints to be confident
_EN_WEAK_TERMS = [
    r"\btomorrow\b", r"\btoday\b",
    r"\b(?:next|this)\s+(?:week|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

_ZH_HK_STRONG_TERMS = [
    r"營業|營運|開放|開門|收(工|舖|店)|幾點(開|收)",
    # Removed: r"安排|改期" (this is scheduling, not opening-hours)
    r"公眾假期|假期",
]
_ZH_HK_WEAK_TERMS = [
    r"今日|聽日|後日|下周|下星期|星期[一二三四五六日天]|周[一二三四五六日天]",
]

_ZH_CN_STRONG_TERMS = [
    r"营业|开放|开门|关门|几点(开|关)",
    # Removed: r"安排|改期" (this is scheduling, not opening-hours)
    r"公众假期|公休日|假期",
]
_ZH_CN_WEAK_TERMS = [
    r"今天|明天|后天|下周|星期[一二三四五六日天]|周[一二三四五六日天]",
]

# Time-of-day hints
_TIME_HINTS = [
    r"\b\d{1,2}:\d{2}\b", r"\b\d{1,2}\s*(am|pm)\b", r"\b(?:9|10|11|12|[1-8])\s*(?:am|pm)\b",
    r"[上下]午", r"\d點|\d点",
]

# Weather markers
_WEATHER_MARKERS = [
    r"typhoon|rainstorm|t[13]|\bt8\b|black rain|amber|red",
    r"颱風|台风|風球|风球|黑雨|紅雨|红雨|黃雨|黄雨|三號|三号|一號|一号|八號|八号",
]

# Negative markers to avoid misclassifying non-hours topics as hours
# Expanded with availability/scheduling phrasing to reduce false positives
# Add cost/time‑unit negatives so “per 1 hour / 50 mins” does NOT look like opening hours
_NEG_EN = [
    r"\b(tuition|fee|fees|price|cost)\b", r"\bclass\s*size\b", r"\bhomework\b", r"\bassignment\b",
    r"\b(time\s*slot|timeslot|slot)s?\b", r"\bavailability\b", r"\bavailable\b",
    r"\btimetable\b", r"\bschedule\b",
    r"\bstart(?:s)?\s+at\b", r"\bfit[s]?\b", r"\bsuit[s]?\b",
    r"\bfor\s+(?:my\s+(?:son|daughter|kid|child)|[A-Z][a-z]+)\b",
    # NEW: guard against billing/time-unit phrasing
    r"\bhow\s+much\b",
    r"\bper\s+(?:hour|hr|minute|min)s?\b",
    r"\b\d+\s*(?:hour|hours|hr|hrs|minute|minutes|min|mins)\b",
    r"\b(make-?ups?|makeups?)\b", r"\babsence\b", r"\bpolicy\b", r"\bquota\b",
]

_NEG_ZH_HK = [
    r"學費|收費|費用|價錢|價格|班級人數|人數", r"功課|家課|作業",
    r"(時段|檔期|時間表|時間安排)", r"(有冇|可唔可以).*(時段|時間)",
    r"(幾點|幾時)\s*開始", r"(適合|合適).*(仔|女|小朋友|學生|[A-Z][a-z]+)",
    r"補課|請假|政策|配額|額度",
]
_NEG_ZH_CN = [
    r"学费|收费|费用|价钱|价格|班级人数|人数", r"功课|作业",
    r"(时段|档期|时间表|课程安排)", r"(有|有没有).*(时段|时间|名额)",
    r"(几点|什么时候)\s*开始", r"(适合).*(儿子|女儿|孩子|学生|[A-Z][a-z]+)",
    r"补课|请假|政策|配额|额度",
]

# Flattened holiday keywords for regex fallback
_ALL_HOLIDAY_KEYWORDS: List[str] = [kw for group in _HOLIDAY_KEYWORDS.values() for kw in group]
_HOLIDAY_TERMS_REGEX = [re.escape(term) for term in _ALL_HOLIDAY_KEYWORDS]

def _score_regex(message: str, patterns: List[str]) -> Tuple[int, List[str]]:
    hits: List[str] = []
    score = 0
    for pat in patterns:
        if re.search(pat, message or "", flags=re.IGNORECASE):
            hits.append(pat)
            score += 1
    return score, hits

def detect_opening_hours_intent(message: str, lang: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect opening-hours intent with stronger precision using:
    - language-specific strong/weak terms
    - time hints
    - holiday keyword awareness
    - negative markers to exclude unrelated topics
    - HARD GUARD: If the message looks like availability/scheduling, force NOT opening-hours.
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

    is_intent = (neg_score == 0) and (
        strong_score > 0
        or holiday_score > 0
        or (weak_score > 0 and time_score > 0)
    )
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
        "overridden_by_sched": False,
    }

    # HARD GUARD: If this looks like availability/scheduling, force NOT opening-hours
    try:
        cls = classify_scheduling_context(message or "", lang or "en")
        if (cls.get("availability_request") or cls.get("has_sched_verbs")
            or cls.get("admin_action_request") or cls.get("staff_contact_request")
            or cls.get("individual_homework_request")):
            is_intent = False
            debug["is_intent"] = False
            debug["overridden_by_sched"] = True
        # NEW: If this is clearly a policy question, do NOT treat it as opening-hours
        if cls.get("has_policy_intent"):
            is_intent = False
            debug["is_intent"] = False
            debug["overridden_by_policy"] = True
    except Exception:
        pass

    if SETTINGS.opening_hours_use_llm_intent and not is_intent:
        # Reserved for optional future LLM-assisted intent confirmation
        pass

    return is_intent, debug

def is_general_hours_query(message: str, lang: str) -> bool:
    """
    True if the message is about opening hours but does NOT contain explicit dates,
    weekdays, relative days, or named holidays (i.e., 'What are your opening hours?').
    """
    m = (message or "").lower()
    specific_date_patterns = [
        r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(today|tomorrow|yesterday|next week|this week)\b",
        r"星期[一二三四五六日天]|周[一二三四五六日天]|週[一二三四五六日天]|礼拜[一二三四五六日天]|禮拜[一二三四五六日天]",
        r"今天|今日|明天|聽日|后天|後日|下周|下星期|本周|本星期",
        r"\d{1,2}\s*(月|日|号|號)",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}\b",
    ]
    for pat in specific_date_patterns:
        if re.search(pat, m, re.I):
            return False

    # Named-holiday presence makes it specific
    for holiday_term in _ALL_HOLIDAY_KEYWORDS:
        if ' ' not in holiday_term and re.search(r'[a-zA-Z]', holiday_term):
            if re.search(r'\b' + re.escape(holiday_term) + r'\b', m, re.I):
                return False
        elif holiday_term in m:
            return False

    return True

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

# ============================================================
# Soft classifiers for scheduling/leave/availability/homework/staff-contact
# ============================================================

# Availability / timetable / start-date (expanded to include seasonal/month-based asks and "slot" wording)
_AVAIL_EN = [
    r"\bavailable\b", r"\bavailability\b",
    r"\b(any|another)\s+(class|slot|time ?slot|timeslot)\b",
    r"\btime\s*slot(s)?\b", r"\bslot(s)?\b",
    r"\btimetable\b", r"\bschedule\b", r"\bwhat times\b", r"\bstart date\b",
    r"\bwhich time\b", r"\btime works\b", r"\bteacher availability\b",
    r"\bsummer\b", r"\bsummer (program|class|course|camp)s?\b", r"\bholiday\s*camp\b",
    r"\b(july|august)\b", r"\bterm\b", r"\bnext\s+term\b", r"\bsemester\b", r"\bsummer schedule\b",
    r"\b(after|post)\s+(cny|chinese new year|lunar new year)\b",
]
_AVAIL_ZH_HK = [
    r"有冇(堂|時段|時間|位)", r"時間表", r"時間安排", r"檔期", r"可唔可以.*時間", r"幾時開始(上|開)課",
    r"老師(幾時|時間)有空|導師(幾時|時間)得閒|老師檔期|導師檔期",
    r"(時段|時間檔|時間位)",
    r"暑期|暑假|夏令(營|营)|暑期班|夏季班|七月|八月|夏天|暑假課|暑期課",
    r"(下學期|過年之後|新年之後|農曆新年之後)",
]
_AVAIL_ZH_CN = [
    r"(有|有没有)(课|课程|时段|时间|名额)", r"时间表", r"课程安排", r"档期", r"可以.*时间", r"什么时候开始(上|开)课",
    r"(老师|教师)(什么时候|什么时间)有空|老师档期|教师档期",
    r"(时段|时间档|时间位)",
    r"暑期|暑假|夏令营|暑期班|夏季班|七月|八月|夏天|暑假课|暑期课",
    r"(下学期|过年之后|春节之后|农历新年之后)",
]

# Post-assessment markers
_POST_ASSESS_EN = [r"\bafter (the )?assessment\b", r"\bpost-?assessment\b", r"\bcompleted (the )?assessment\b"]
_POST_ASSESS_ZH_HK = [r"評估(之後|後)", r"完成(了)?評估", r"做完評估"]
_POST_ASSESS_ZH_CN = [r"评估(之后|后)", r"完成(了)?评估", r"做完评估"]

# Student-reference markers (helps detect specific-child requests)
_STUDENT_REF_EN = [
    r"\bfor\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b",
    r"\bmy (son|daughter|kid|child)\b", r"\bstudent\b", r"\bfor him\b", r"\bfor her\b",
    r"\b(?:fit|fits|suit|suits|work[s]?(?:\s*for)?)\s+[A-Z][a-z]+\b",
]
_STUDENT_REF_ZH_HK = [r"為?(\S+)?(小朋友|小童|小孩|仔|女|學生)", r"我(個|的)?(仔|女|小朋友)", r"替.*(仔|女)"]
_STUDENT_REF_ZH_CN = [r"为?(\S+)?(小朋友|孩子|小孩|学生)", r"我(家)?(儿子|女儿|孩子)", r"替.*(儿子|女儿)"]

# Extra reschedule/change-day phrasing boosters
_RESCHED_EXTRA_EN = [
    r"\b(change|move|switch|reschedul(?:e|ing)|rearrange)\s+(?:the\s+)?(class|lesson|time|date|day)\b",
    r"\b(can|could|may)\s+(?:he|she|my (son|daughter|kid|child)|[A-Z][a-z]+)\s+(?:come|attend)\s+(?:on|another)\s+(day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(another|a different)\s+day\b",
    r"\b(move|switch)\s+to\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]
_RESCHED_EXTRA_ZH_HK = [
    r"(改期|改時間|改堂|轉日|換日|移到|調到|調期|改去)",
    r"(可唔可以|得唔得|可否).*(星期[一二三四五六日天]|禮拜[一二三四五六日天]).*(上課|上堂|嚟)",
    r"(改|轉|換|移|調).*(去|到).*(星期[一二三四五六日天]|禮拜[一二三四五六日天])",
    r"(另一|另|第二)日|另一天|其他日子",
]
_RESCHED_EXTRA_ZH_CN = [
    r"(改期|改时间|改堂|换天|换日|调到|挪到|改到)",
    r"(能不能|可不可以|可以吗|是否可以).*(周[一二三四五六日天]|星期[一二三四五六日天]).*(上课|来)",
    r"(改|换|调|挪).*(去|到).*(周[一二三四五六日天]|星期[一二三四五六日天])",
    r"(另|换|别|其他)一[天日]|别的日子|其他日子",
]

# Scheduling/leave verbs (expanded with 'suspend/stop' variants and absence notifications)
_SCHED_ZH_HK = [
    r"請假", r"改期", r"改時間", r"改堂", r"取消", r"缺席", r"退堂",
    r"(暫\s*停|停止|停課|停堂|暫停安排|暫停上課)",
    r"(唔嚟|唔來|去唔到|來唔切|嚟唔切).*(上課|上堂)?",
    r"(缺席|未能出席)",
]
_SCHED_ZH_CN = [
    r"请假", r"改期", r"改时间", r"改堂", r"取消", r"缺席", r"退课",
    r"(暂\s*停|停止|停课|暂停安排|暂停上课)",
    r"(来不了|來不了|去不了|不能来|不能來|不能上课|不能上課)",
    r"(缺席|未能出席)",
]
_SCHED_EN = [
    r"\breschedul(?:e|ing)\b", r"\bcancel(?:ling|ation)?\b", r"\btake\s+leave\b",
    r"\brequest\s+leave\b", r"\b(absent|absence)\b", r"\b(suspend|put on hold|pause)\b",
    r"\b(can\'?t|cannot|won\'?t)\s+(attend|come|make it)\b",
    r"\bwon\'?t be able to attend\b",
    r"\bwill be away\b",
    r"\b(be|is|are) away\b",
]

# Date/time markers (for has_date_time) — expanded to include month names
_DATE_MARKERS = [
    r"\b\d{1,2}/\d{1,2}\b", r"\d{1,2}\s*(月|日|号|號)", r"(星期|周|週)[一二三四五六日天]",
    r"\b(mon|tue|wed|thu|fri|sat|sun)\b", r"\b(today|tomorrow)\b",
    r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b",
    r"今天|今日|明天|後日|后天|聽日|下(周|星期|週)",
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
]

# Policy markers
_POLICY_EN = [r"\bpolicy\b", r"\bwhat\s+is\s+the\s+policy\b", r"\brules?\b", r"\bhow\s+do(?:es)?\s+.*(reschedul|make-?up|absence)\b"]
_POLICY_ZH_HK = [r"政策", r"安排", r"規則", r"規定", r"補課政策", r"請假政策", r"改期政策", r"補課安排", r"請假安排", r"改期安排"]
_POLICY_ZH_CN = [r"政策", r"安排", r"规则", r"规定", r"补课政策", r"请假政策", r"改期政策", r"补课安排", r"请假安排", r"改期安排"]

# Politeness-only detectors
_POLITENESS_ONLY_EN = re.compile(r"^\s*(thanks|thank you|ty|thx|appreciate(?: it)?|cheers)\s*[.!]*\s*$", re.I)
_POLITENESS_ONLY_ZH_HK = re.compile(r"^\s*(多謝|謝謝|唔該|唔該晒)\s*[！!。.\s]*$", re.I)
_POLITENESS_ONLY_ZH_CN = re.compile(r"^\s*(谢谢|多谢|辛苦了|麻烦了)\s*[！!。.\s]*$", re.I)

def is_politeness_only(message: str, lang: str) -> bool:
    m = (message or "").strip()
    L = (lang or "en").lower()
    if L.startswith("zh-hk"): return bool(_POLITENESS_ONLY_ZH_HK.match(m))
    if L.startswith("zh-cn") or L == "zh": return bool(_POLITENESS_ONLY_ZH_CN.match(m))
    return bool(_POLITENESS_ONLY_EN.match(m))

def _score(m: str, pats: List[str]) -> int:
    return sum(bool(re.search(p, m or "", re.I)) for p in pats)

# Admin relay detection (pass/notify/ask teacher)
_ADMIN_RELAY_EN = [
    r"(?:please|kindly|could you|can you|would you|help(?: me)?(?: to)?)\s+(?:pass|forward|relay|send|share)\b.*\b(?:teacher|teachers|staff|instructor|coach)\b",
    r"(?:please|kindly|could you|can you|would you|help(?: me)?(?: to)?)\s+(?:tell|inform|notify)\b.*\b(?:teacher|teachers|staff|instructor|coach)\b",
    r"(?:please|kindly|could you|can you|would you|help(?: me)?(?: to)?)\s+ask\b.*\b(?:teacher|teachers|staff|instructor|coach)\b.*\bto\b",
    r"\bpass (?:the )?message\b.*\b(?:teacher|teachers|staff|instructor|coach)\b",
    r"\bleave (?:a )?note\b.*\b(?:teacher|teachers)\b",
]
# Negative guard to avoid “teacher qualifications” queries etc.
_ADMIN_RELAY_EN_NEG = [
    r"\bare\s+(?:your|the)\s+teachers?\b",
    r"\bwhat\s+(?:are|is)\s+(?:your|the)\s+teachers?\b",
    r"\bteacher(?:s)?\s+qualification|native\s+speaker",
]
_ADMIN_RELAY_ZH_HK = [
    r"(請|麻煩|幫)(我|手)?(問|轉告|通知)(老師|導師)",
    r"(同|跟)(老師|導師)講",
    r"(幫|請)將(訊息|信息|留言|說話|話)轉達(比|畀)?(老師|導師)",
    r"(請|幫|麻煩).*(轉達|代為轉告).*(老師|導師)",
    r"(請|幫|麻煩).*問(老師|導師).*(去|做|安排|提醒)",
]
_ADMIN_RELAY_ZH_CN = [
    r"(请|麻烦|帮)(我)?(问|转告|通知)(老师|教师)",
    r"(跟|和)(老师|教师)说",
    r"(帮|请)将(信息|留言|话)转达给(老师|教师)",
    r"(请|帮|麻烦).*(转达|代为转告).*(老师|教师)",
    r"(请|帮|麻烦).*问(老师|教师).*(去|做|安排|提醒)",
]

def _has_admin_action_request(message: str, lang: str) -> bool:
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return any(re.search(p, m) for p in _ADMIN_RELAY_ZH_HK)
    if L.startswith("zh-cn") or L == "zh":
        return any(re.search(p, m) for p in _ADMIN_RELAY_ZH_CN)
    if any(re.search(p, m, re.I) for p in _ADMIN_RELAY_EN_NEG):
        return False
    return any(re.search(p, m, re.I) for p in _ADMIN_RELAY_EN)

# Homework markers and individualized-advice patterns
_HOMEWORK_EN = [r"\bhomework\b", r"\bread(?:ing)?\s+assignment\b", r"\bworksheet\b", r"\bphonics\b", r"\bpronounc"]
_HOMEWORK_ZH_HK = [r"功課|家課|作業|閱讀作業|閱讀功課|閱讀任務|拼音|發音|讀音"]
_HOMEWORK_ZH_CN = [r"作业|功课|阅读作业|阅读功课|阅读任务|拼音|发音|读音"]

_INDIV_ADVICE_EN = [
    r"\bhow\s+should\b", r"\bhow\s+to\b", r"\bwhat\s+is\s+the\s+(best|ideal)\s+way\b",
    r"\bshould\s+(?:i|we|he|she|my (son|daughter|kid|child))\b",
    r"\b(adult|parent|teacher)\s+guidance\b", r"\bguide\s+(him|her|my (son|daughter|kid|child))\b",
    r"\bcan't\s+read\b", r"\bcan(?:not|'t)\s+pronounc",
]
_INDIV_ADVICE_ZH_HK = [
    r"(點樣|如何|應該|點做)", r"(需要|要唔要)(大人|家長|老師)指導",
    r"(唔識讀|唔識發音|讀唔到|讀唔出|發音有問題)", r"(指導|帶住|帶領)(佢|小朋友)",
]
_INDIV_ADVICE_ZH_CN = [
    r"(怎么|如何|应该|怎样|咋做)", r"(需要|要不要)(大人|家长|老师)指导",
    r"(不会读|不会发音|读不了|读不出|发音有问题)", r"(指导|带着|带领)(他|她|孩子|小朋友)",
]

# NEW: Staff role mentions and contact verbs (staff-contact requests)
_STAFF_ROLES_EN = [
    r"\b(course\s+)?director\b", r"\bprincipal\b", r"\bhead\s+(?:teacher|of[^\w]*\w+)\b",
    r"\bteacher\b", r"\binstructor\b", r"\btutor\b", r"\bstaff\b", r"\bconsultant\b", r"\badmin(?:istrator)?\b",
    r"\bms\.?\s+[A-Z][a-z]+\b", r"\bmr\.?\s+[A-Z][a-z]+\b", r"\bmrs\.?\s+[A-Z][a-z]+\b", r"\bmiss\s+[A-Z][a-z]+\b",
]
_STAFF_ROLES_ZH_HK = [r"主任|校長|老師|導師|顧問|職員|負責人|課程主任|教學主任|導師|老師"]
_STAFF_ROLES_ZH_CN = [r"主任|校长|老师|教师|顾问|职员|负责人|课程主任|教学主任|导师|老师"]

_CONTACT_VERBS_EN = [
    r"\barrang(e|ing|ement)\b", r"\bschedul(e|ing|e a)\b", r"\bset\s*up\b", r"\bbook\b", r"\borganis?e\b",
    r"\b(call|phone\s*call|video\s*call|zoom|teams|meeting|meet|chat)\b",
    r"\bspeak\s+(?:with|to)\b", r"\btalk\s+(?:with|to)\b",
]
_CONTACT_VERBS_ZH_HK = [r"安排|預約|約|約見|約電話|打電話|致電|通話|講電話|聯絡|見面|會面|約見面|約通話"]
_CONTACT_VERBS_ZH_CN = [r"安排|预约|约|约见|约电话|打电话|致电|通话|讲电话|联系|见面|会面|约见面|约通话"]

def classify_scheduling_context(message: str, lang: str) -> Dict[str, Any]:
    """
    Soft classification: returns booleans used to steer prompting only.
    - has_sched_verbs: leave/reschedule/cancel OR ANY availability inquiry (now includes generic "slot"/seasonal) OR availability + (post-assessment OR student-ref)
    - has_date_time: mentions specific date/weekday/time (now also month names)
    - has_policy_intent: asks about policy/rules around reschedule/leave
    - availability_request: availability/timetable/time-slots/teacher availability/start date (includes seasonal/month-based asks)
    - post_assessment: mentions 'after/completed assessment'
    - student_ref: refers to a specific child (name/pronoun/son/daughter)
    - politeness_only: message is pure thanks/politeness
    - admin_action_request: pass/relay/ask/notify a teacher/staff
    - individual_homework_request: student-specific homework/teaching guidance (requires teacher/admin) -> should be silenced
    - staff_contact_request: request to speak with/arrange a call/meeting with a specific staff role or named teacher/director -> should be silenced
    """
    m = message or ""
    L = (lang or "en").lower()

    if L.startswith("zh-hk"):
        base_sched = (_score(m, _SCHED_ZH_HK) > 0) or (_score(m, _RESCHED_EXTRA_ZH_HK) > 0)
        policy = _score(m, _POLICY_ZH_HK) > 0
        avail = _score(m, _AVAIL_ZH_HK) > 0
        post = _score(m, _POST_ASSESS_ZH_HK) > 0
        student = _score(m, _STUDENT_REF_ZH_HK) > 0
        hw = _score(m, _HOMEWORK_ZH_HK) > 0
        adv = _score(m, _INDIV_ADVICE_ZH_HK) > 0
        pron = bool(re.search(r"(佢|小朋友|學生)", m))
        staff_role = _score(m, _STAFF_ROLES_ZH_HK) > 0 or _score(m, _STAFF_ROLES_EN) > 0
        contact_verb = _score(m, _CONTACT_VERBS_ZH_HK) > 0
    elif L.startswith("zh-cn") or L == "zh":
        base_sched = (_score(m, _SCHED_ZH_CN) > 0) or (_score(m, _RESCHED_EXTRA_ZH_CN) > 0)
        policy = _score(m, _POLICY_ZH_CN) > 0
        avail = _score(m, _AVAIL_ZH_CN) > 0
        post = _score(m, _POST_ASSESS_ZH_CN) > 0
        student = _score(m, _STUDENT_REF_ZH_CN) > 0
        hw = _score(m, _HOMEWORK_ZH_CN) > 0
        adv = _score(m, _INDIV_ADVICE_ZH_CN) > 0
        pron = bool(re.search(r"(他|她|孩子|小朋友|学生)", m))
        staff_role = _score(m, _STAFF_ROLES_ZH_CN) > 0 or _score(m, _STAFF_ROLES_EN) > 0
        contact_verb = _score(m, _CONTACT_VERBS_ZH_CN) > 0
    else:
        base_sched = (_score(m, _SCHED_EN) > 0) or (_score(m, _RESCHED_EXTRA_EN) > 0)
        policy = _score(m, _POLICY_EN) > 0
        avail = _score(m, _AVAIL_EN) > 0
        post = _score(m, _POST_ASSESS_EN) > 0
        student = _score(m, _STUDENT_REF_EN) > 0
        hw = _score(m, _HOMEWORK_EN) > 0
        adv = _score(m, _INDIV_ADVICE_EN) > 0
        pron = bool(re.search(r"\b(he|him|his|she|her|hers|my (son|daughter|kid|child))\b", m, re.I))
        staff_role = _score(m, _STAFF_ROLES_EN) > 0
        contact_verb = _score(m, _CONTACT_VERBS_EN) > 0

    date_time = _score(m, _DATE_MARKERS) > 0
    politeness = is_politeness_only(m, lang)

    # Availability alone is admin-handled scheduling too (stronger rule)
    sched = base_sched or avail or (avail and (post or student))

    admin_action = _has_admin_action_request(m, L)
    individual_hw = bool(hw and (adv or student or pron))
    staff_contact = bool(staff_role and contact_verb)

    return {
        "has_sched_verbs": sched,
        "has_date_time": date_time,
        "has_policy_intent": policy,
        "availability_request": avail,
        "post_assessment": post,
        "student_ref": student,
        "politeness_only": politeness,
        "admin_action_request": admin_action,
        "individual_homework_request": individual_hw,
        "staff_contact_request": staff_contact,
    }