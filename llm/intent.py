import re
from typing import Tuple, Dict, Any

# Import the single source of truth for holiday keywords from opening_hours
# This prevents inconsistencies between intent detection and date parsing.
from llm.opening_hours import _HOLIDAY_KEYWORDS

# Broad catch-all intent detection for opening hours, attendance, and arrangements
# Supports English, zh-HK (Traditional), zh-CN (Simplified)

# NOTE: Keep terms focused on hours/closed/weekday/holiday/time. Avoid generic "class/lesson".
_EN_TERMS = [
    r"\bopen(?:ing)?\b", r"\bhours?\b", r"\bclosed?\b", r"\bbusiness hours?\b",
    r"\battend(?:ing)?\s+(?:class|lesson)\b",
    r"\btomorrow\b", r"\btoday\b",
    r"\b(?:next|this)\s+(?:week|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bpublic holiday\b", r"\bholiday\b",
]

_ZH_HK_TERMS = [
    r"營業|營運|開放|開門|收(工|舖|店)|幾點(開|收)",
    r"上課|上堂|返學|返課",   # removed generic 班/班級
    r"安排|改期",
    r"今日|聽日|後日|下周|下星期|星期[一二三四五六日天]|周[一二三四五六日天]",
    r"公眾假期|假期",
]

_ZH_CN_TERMS = [
    r"营业|开放|开门|关门|几点(开|关)",
    r"上课|上学|上(?:不)?上课",  # removed generic 课程/班级
    r"安排|改期",
    r"今天|明天|后天|下周|星期[一二三四五六日天]|周[一二三四五六日天]",
    r"公众假期|公休日|假期",
]

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

# --- REMOVED REDUNDANT HOLIDAY LISTS ---
# The _HOLIDAY_EN, _HOLIDAY_ZH_HK, and _HOLIDAY_ZH_CN lists were removed.
# We now use the keys from the _HOLIDAY_KEYWORDS dictionary in opening_hours.py
# as the single source of truth.
_HOLIDAY_TERMS_REGEX = [re.escape(term) for term in _HOLIDAY_KEYWORDS.keys()]


# Negative markers: if present, do NOT classify as opening-hours
_NEG_EN = [r"\b(tuition|fee|fees|price|cost)\b", r"\bclass\s*size\b"]
_NEG_ZH_HK = [r"學費|收費|費用|價錢|價格|班級人數|人數"]
_NEG_ZH_CN = [r"学费|收费|费用|价钱|价格|班级人数|人数"]

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
    m = message or ""
    # An intent check is no longer needed here; this function's purpose is to refine
    # a query already determined to have opening hours intent.

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

    # --- MAJOR FIX ---
    # Check for named holidays. If one is found, the query is specific.
    # We use the consolidated list imported from opening_hours.py
    for holiday_term in _HOLIDAY_KEYWORDS.keys():
        # Use regex to match whole words for English terms to avoid partial matches like 'day' in 'today'
        if holiday_term.isalpha() and re.search(r'\b' + re.escape(holiday_term) + r'\b', m, re.I):
             return False
        # For Chinese terms, simple substring search is fine
        elif not holiday_term.isalpha() and holiday_term in m:
             return False

    # If no specific date, weekday, or holiday markers are found, it's a general query.
    return True

def detect_opening_hours_intent(message: str, lang: str, use_llm: bool = True) -> Tuple[bool, Dict[str, Any]]:
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        base_terms = _ZH_HK_TERMS
        neg_terms = _NEG_ZH_HK
    elif L.startswith("zh-cn") or L == "zh":
        base_terms = _ZH_CN_TERMS
        neg_terms = _NEG_ZH_CN
    else:
        base_terms = _EN_TERMS
        neg_terms = _NEG_EN

    base_score, base_hits = _score_regex(m, base_terms)
    time_score, time_hits = _score_regex(m, _TIME_HINTS)
    weather_score, weather_hits = _score_regex(m, _WEATHER_MARKERS)
    # Use the consolidated holiday list for scoring
    holiday_score, holiday_hits = _score_regex(m, _HOLIDAY_TERMS_REGEX)
    neg_score, neg_hits = _score_regex(m, neg_terms)

    # Require at least one real signal (base/time/holiday), and suppress if negative terms are present.
    score = base_score + (1 if time_score else 0) + (1 if holiday_score else 0)
    is_intent = score >= 1 and neg_score == 0

    debug = {
        "score": score,
        "base_hits": base_hits,
        "time_hits": time_hits,
        "weather_hits": weather_hits,
        "holiday_hits": holiday_hits,
        "neg_hits": neg_hits,
        "used_llm": False,
        "llm_confidence": None,
    }

    if use_llm and not is_intent:
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