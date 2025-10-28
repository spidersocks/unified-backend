import re
from typing import Tuple, Dict, Any

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

# Holiday name patterns (EN / zh-HK / zh-CN)
_HOLIDAY_EN = [
    r"\blunar new year\b", r"\bchinese new year\b", r"\bsecond day of lunar new year\b", r"\bthird day of lunar new year\b",
    r"\bching ming\b", r"\btomb-?sweeping\b",
    r"\bchung yeung\b",
    r"\btuen ng\b", r"\bdragon boat\b",
    r"\bmid[- ]autumn\b", r"\bthe day following the chinese mid[- ]autumn festival\b",
    r"\bbuddha(?:'s)? birthday\b",
    r"\bnational day\b",
    r"\blabou?r day\b",
    r"\bhksar establishment day\b|\bestablishment day\b",
    r"\bgood friday\b", r"\beaster monday\b",
    r"\bchristmas\b|\bfirst weekday after christmas\b",
]
_HOLIDAY_ZH_HK = [
    r"農曆新年|年初[一二三]|新年",
    r"清明|清明節",
    r"重陽|重陽節",
    r"端午|端午節",
    r"中秋|中秋節|中秋節翌日",
    r"佛誕",
    r"國慶|國慶日",
    r"勞動節",
    r"回歸|香港特別行政區成立紀念日",
    r"耶穌受難日|復活節星期一",
    r"聖誕|聖誕節|聖誕節後首個工作天",
]
_HOLIDAY_ZH_CN = [
    r"农历新年|年初[一二三]|新年",
    r"清明|清明节",
    r"重阳|重阳节",
    r"端午|端午节",
    r"中秋|中秋节|中秋节翌日",
    r"佛诞",
    r"国庆|国庆日",
    r"劳动节",
    r"回归|香港特别行政区成立纪念日",
    r"耶稣受难日|复活节星期一",
    r"圣诞|圣诞节|圣诞节后第一个工作日",
]

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
    but does NOT contain any explicit date, weekday, or relative-day marker.
    Used to distinguish 'What are your opening hours?' from 'Are you open on Sunday?'.
    For EN, treat 'public holiday' or 'holiday' as general.
    """
    m = message or ""
    L = lang.lower() if lang else "en"
    is_intent, _ = detect_opening_hours_intent(m, lang, use_llm=True)
    if not is_intent:
        return False

    # Weekday and relative-day markers (in all languages)
    weekday_patterns = [
        r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(today|tomorrow|yesterday|next week|this week)\b",
        r"星期[一二三四五六日天]|周[一二三四五六日天]|週[一二三四五六日天]|礼拜[一二三四五六日天]|禮拜[一二三四五六日天]",
        r"今天|今日|明天|聽日|后天|後日|下周|下星期|本周|本星期"
    ]
    for pat in weekday_patterns:
        if re.search(pat, m, re.I):
            return False

    # Explicit date markers
    if re.search(r"\d{1,2}\s*(月|日|号|號)", m):
        return False
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b", m, re.I):
        return False
    if re.search(r"\b\d{1,2}/\d{1,2}\b", m):
        return False

    # For EN: treat 'public holiday' or 'holiday' as general unless a date is present
    if L == "en" and re.search(r"\b(public holiday|holiday|holidays?)\b", m, re.I):
        # If also contains "on <date>", not general
        if re.search(r"\bon\b.*\d{1,2}", m, re.I):
            return False
        return True

    return True

def detect_opening_hours_intent(message: str, lang: str, use_llm: bool = True) -> Tuple[bool, Dict[str, Any]]:
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        base_terms = _ZH_HK_TERMS
        neg_terms = _NEG_ZH_HK
        holiday_terms = _HOLIDAY_ZH_HK
    elif L.startswith("zh-cn") or L == "zh":
        base_terms = _ZH_CN_TERMS
        neg_terms = _NEG_ZH_CN
        holiday_terms = _HOLIDAY_ZH_CN
    else:
        base_terms = _EN_TERMS
        neg_terms = _NEG_EN
        holiday_terms = _HOLIDAY_EN

    base_score, base_hits = _score_regex(m, base_terms)
    time_score, time_hits = _score_regex(m, _TIME_HINTS)
    weather_score, weather_hits = _score_regex(m, _WEATHER_MARKERS)
    holiday_score, holiday_hits = _score_regex(m, holiday_terms)
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