import re
from typing import Tuple, Dict, Any, List

# Import the single source of truth for holiday keywords from opening_hours
# This prevents inconsistencies between intent detection and date parsing.
from llm.opening_hours import _HOLIDAY_KEYWORDS

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

def detect_opening_hours_intent(message: str, lang: str, use_llm: bool = True) -> Tuple[bool, Dict[str, Any]]:
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