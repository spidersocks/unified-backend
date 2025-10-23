import re
from typing import Tuple, Dict, Any

# Broad catch-all intent detection for opening hours, attendance, and arrangements
# Supports English, zh-HK (Traditional), zh-CN (Simplified)

_EN_TERMS = [
    r"\bopen(?:ing)?\b", r"\bhours?\b", r"\bclosed?\b", r"\bbusiness hours?\b",
    r"\battend (?:class|lesson)\b", r"\b(class|lesson)\b", r"\barrangements?\b",
    r"\btomorrow\b", r"\btoday\b", r"\bnext (?:week|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bpublic holiday\b", r"\bholiday\b",
]

_ZH_HK_TERMS = [
    r"營業|營運|開放|開門|收(工|舖|店)|幾點(開|收)",
    r"上課|上堂|返學|返(?:唔)?學|返(?:唔)?課|課堂|班",
    r"安排|改期",
    r"今日|聽日|後日|下周|下星期|星期[一二三四五六日天]|周[一二三四五六日天]",
    r"公眾假期|假期",
]

_ZH_CN_TERMS = [
    r"营业|开放|开门|关门|几点(开|关)",
    r"上课|上学|上(不)?上课|课程|班级",
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

def _score_regex(message: str, patterns: list[str]) -> Tuple[int, list[str]]:
    hits = []
    score = 0
    for pat in patterns:
        if re.search(pat, message, flags=re.IGNORECASE):
            hits.append(pat)
            score += 1
    return score, hits

def detect_opening_hours_intent(message: str, lang: str, use_llm: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (is_intent, debug).
    We prioritize high-recall regex over optional LLM assist. The LLM stage is a stub that can be
    wired later if desired; for now we rely on regex results only.
    """
    m = message or ""
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        base_terms = _ZH_HK_TERMS
    elif L.startswith("zh-cn") or L == "zh":
        base_terms = _ZH_CN_TERMS
    else:
        base_terms = _EN_TERMS

    base_score, base_hits = _score_regex(m, base_terms)
    time_score, time_hits = _score_regex(m, _TIME_HINTS)
    weather_score, weather_hits = _score_regex(m, _WEATHER_MARKERS)

    score = base_score + (1 if time_score else 0)
    is_intent = score >= 1  # extremely permissive to catch broad phrasing

    debug = {
        "score": score,
        "base_hits": base_hits,
        "time_hits": time_hits,
        "weather_hits": weather_hits,
        "used_llm": False,
        "llm_confidence": None,
    }

    # Optional LLM assist (stub). You can wire Bedrock here if needed.
    # Keep structure for future extension without changing callers.
    if use_llm and not is_intent:
        # If you later enable LLM, set debug["used_llm"]=True and update is_intent accordingly.
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
        return bool(re.search(r"上课|上学|课程|班级", m))
    return bool(re.search(r"\b(attend(ing)?|class|lesson)\b", m, flags=re.IGNORECASE))