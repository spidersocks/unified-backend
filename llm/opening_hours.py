from datetime import datetime, timedelta, time
from typing import Optional, Tuple, Dict, Any, Callable, List
import re

import pytz
try:
    import dateparser  # type: ignore
except Exception:
    dateparser = None

try:
    import holidays  # type: ignore
except Exception:
    holidays = None

from functools import lru_cache
from llm.hko import get_weather_hint_for_opening
from llm.config import SETTINGS

HK_TZ = pytz.timezone("Asia/Hong_Kong")
# opening_hours.py
if holidays is None:
    print("[OPENING_HOURS] ERROR: 'holidays' package not installed. HK public holiday resolution disabled.", flush=True)

def _log(msg):
    print(f"[OPENING_HOURS] {msg}", flush=True)

# Business hours
WEEKDAY_OPEN = time(9, 0)    # Mon-Fri
WEEKDAY_CLOSE = time(18, 0)
SAT_OPEN = time(9, 0)        # Sat
SAT_CLOSE = time(16, 0)

def _normalize_lang(lang: Optional[str]) -> str:
    l = (lang or "en").lower()
    if l.startswith("zh-hk"):
        return "zh-HK"
    if l.startswith("zh-cn") or l == "zh":
        return "zh-CN"
    return "en"

# Weekday patterns
_WD_PAT_EN = re.compile(
    r"\b(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)
_WD_MAP_EN = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}
_WD_PAT_ZH_HK = re.compile(r"(星期[一二三四五六日天]|周[一二三四五六日天]|週[一二三四五六日天]|礼拜[一二三四五六日天]|禮拜[一二三四五六日天])")
_WD_MAP_ZH = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# Relative day hints
_REL_ZH = {
    "今天": 0, "今日": 0,
    "明天": 1, "聽日": 1,
    "后天": 2, "後日": 2,
}
_REL_EN = {"today": 0, "tomorrow": 1, "day after tomorrow": 2}

# --- SINGLE SOURCE OF TRUTH FOR HOLIDAY KEYWORDS ---
_HOLIDAY_KEYWORDS = {
    "Lunar New Year": ["lunar new year", "chinese new year", "the first day of lunar new year", "年初一", "農曆新年", "农历新年"],
    "Second Day of Lunar New Year": ["the second day of lunar new year", "年初二"],
    "Third Day of Lunar New Year": ["the third day of lunar new year", "年初三"],
    "Ching Ming Festival": ["ching ming", "tomb-sweeping", "清明", "清明節", "清明节"],
    "Chung Yeung Festival": ["chung yeung", "重陽", "重阳", "重陽節", "重阳节"],
    "Tuen Ng Festival": ["tuen ng", "dragon boat", "端午", "端午節", "端午节"],
    "The day following the Chinese Mid-Autumn Festival": ["the day following the chinese mid-autumn festival", "中秋節翌日", "中秋节翌日"],
    "Mid-Autumn Festival": ["mid-autumn", "mid autumn", "中秋", "中秋節", "中秋节"],
    "Buddha's Birthday": ["buddha", "buddha's birthday", "佛誕", "佛诞"],
    "National Day": ["national day", "國慶", "国庆", "國慶日", "国庆日"],
    "Labour Day": ["labour day", "labor day", "勞動節", "劳动节"],
    "HKSAR Establishment Day": ["establishment day", "hksar establishment", "回歸", "回归", "香港特別行政區成立紀念日", "香港特别行政区成立纪念日"],
    "Good Friday": ["good friday", "耶穌受難日", "耶稣受难日"],
    "Easter Monday": ["easter monday", "復活節星期一", "复活节星期一"],
    "The first weekday after Christmas Day": ["the first weekday after christmas day", "聖誕節後首個工作天", "圣诞节后第一个工作日"],
    "Christmas Day": ["christmas", "christmas day", "聖誕", "圣诞", "聖誕節", "圣诞节"],
}

# --- NEW: DICTIONARY FOR SPECIAL DAYS THAT ARE NOT PUBLIC HOLIDAYS ---
# This allows us to parse days like "Christmas Eve" correctly.
# The value is a function that takes a year and returns a (month, day) tuple.
_SPECIAL_NAMED_DAYS: Dict[str, Callable[[int], Tuple[int, int]]] = {
    "christmas eve": lambda year: (12, 24),
    "平安夜": lambda year: (12, 24),
}

_MONTH_ABBR = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_WD_WORDS_EN = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

def summarize_user_date_intent(message: str, lang: Optional[str] = None) -> str:
    """
    Extracts and summarizes any explicit Month/Day and weekday mentions from the user's message.
    This is a neutral hint for the LLM when the intent is scheduling/action (not opening-hours).
    We do NOT compute 'open/closed' or any policy — just echo what was mentioned.
    """
    msg = (message or "")
    L = _normalize_lang(lang)

    # Month Day like "Nov 12" / "November 12"
    month_day_pat = re.compile(rf"\b{_MONTH_ABBR}[a-z]*\.?\s*\d{{1,2}}\b", re.I)
    month_day_hits = [m.group(0).strip() for m in month_day_pat.finditer(msg)]

    # Weekdays (EN and Chinese)
    wd_hits_en = []
    for m in re.finditer(_WD_PAT_EN, msg or ""):
        wd_hits_en.append(m.group(1).capitalize())
    wd_hits_zh = []
    for m in _WD_PAT_ZH_HK.finditer(msg or ""):
        wd_hits_zh.append(m.group(1))

    # Generic "holiday" mention (helps explain why they want to change)
    has_holiday_word = bool(re.search(r"\bholiday\b|公眾假期|公众假期|假期", msg, re.I))

    # Compose a localized, neutral summary
    if L == "zh-HK":
        parts = []
        if month_day_hits:
            parts.append(f"家長提及日期：{', '.join(month_day_hits)}")
        if wd_hits_en or wd_hits_zh:
            all_wd = []
            if wd_hits_en: all_wd.extend(wd_hits_en)
            if wd_hits_zh: all_wd.extend(wd_hits_zh)
            parts.append(f"提及星期：{', '.join(all_wd)}")
        if has_holiday_word:
            parts.append("訊息提到『假期』。")
        return "；".join(parts) or "家長訊息提及更改上課日期。"
    if L == "zh-CN":
        parts = []
        if month_day_hits:
            parts.append(f"家长提及日期：{', '.join(month_day_hits)}")
        if wd_hits_en or wd_hits_zh:
            all_wd = []
            if wd_hits_en: all_wd.extend(wd_hits_en)
            if wd_hits_zh: all_wd.extend(wd_hits_zh)
            parts.append(f"提及星期：{', '.join(all_wd)}")
        if has_holiday_word:
            parts.append("消息提到“假期”。")
        return "；".join(parts) or "家长消息提及更改上课日期。"
    # EN
    parts = []
    if month_day_hits:
        parts.append(f"User mentions date(s): {', '.join(month_day_hits)}")
    if wd_hits_en:
        parts.append(f"User mentions weekday(s): {', '.join(wd_hits_en)}")
    if has_holiday_word:
        parts.append("User mentions a holiday.")
    return " | ".join(parts) or "User mentions changing the lesson day."

def _looks_like_absolute_date(msg: str) -> bool:
    m = msg or ""
    if re.search(r"\b\d{1,2}/\d{1,2}\b", m, re.I):
        return True
    if re.search(rf"\b{_MONTH_ABBR}[a-z]*\.?\s*\d{{1,2}}\b", m, re.I):
        return True
    if re.search(r"\d{1,2}\s*(月|日|号|號)", m):
        return True
    if _WD_PAT_EN.search(m) or _WD_PAT_ZH_HK.search(m):
        return True
    return False

def _fmt_time(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

def _weekday_label(dt: datetime, L: str) -> str:
    wd = dt.weekday()
    if L == "zh-HK":
        return "星期" + "一二三四五六日"[wd]
    if L == "zh-CN":
        return "周" + "一二三四五六日"[wd]
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd]

def _fmt_date_human(dt: datetime, L: str) -> str:
    d = dt.astimezone(HK_TZ)
    if L == "zh-HK":
        return f"{d.month}月{d.day}日（{_weekday_label(d, L)}）"
    if L == "zh-CN":
        return f"{d.month}月{d.day}日（{_weekday_label(d, L)}）"
    return d.strftime("%a %d %b")

def _dow_window(dow: int) -> Tuple[Optional[time], Optional[time]]:
    if 0 <= dow <= 4:
        return WEEKDAY_OPEN, WEEKDAY_CLOSE
    if dow == 5:
        return SAT_OPEN, SAT_CLOSE
    return None, None  # Sunday closed

# ========== NEW: “Open now” helper ==========

def center_is_open_now(lang: Optional[str] = None) -> bool:
    """
    Returns True if RIGHT NOW is within operating hours and not a public holiday/Sunday
    and there is no severe weather closure (Black Rain or Typhoon Signal No. 8+ or Pre-8).
    """
    now = datetime.now(HK_TZ)
    open_t, close_t = _dow_window(now.weekday())
    if open_t is None or close_t is None:
        # Sunday
        return False

    # Public holiday check
    is_hol, _ = _is_public_holiday(now)
    if is_hol:
        return False

    # Severe weather closure (reuse HKO hint already used for opening answers)
    if SETTINGS.opening_hours_weather_enabled:
        severe = get_weather_hint_for_opening(_normalize_lang(lang))
        if severe:
            # We consider severe conditions a closure
            return False

    # Within hours?
    cur_t = now.time()
    return (open_t <= cur_t < close_t)

@lru_cache(maxsize=8)
def _hk_calendar(start_year: int, end_year: int):
    if not holidays:
        return None
    years = list(range(start_year, end_year + 1))
    try:
        return holidays.HK(years=years)  # type: ignore
    except Exception:
        return None

def _is_public_holiday(d: datetime) -> Tuple[bool, Optional[str]]:
    cal = _hk_calendar(d.year - 1, d.year + 1)
    if not cal:
        return False, None
    name = cal.get(d.date())
    if name:
        return True, str(name)
    return False, None

# --- NEW: export a simple checker for other modules ---
def is_hk_public_holiday(dt: datetime) -> bool:
    return _is_public_holiday(dt)[0]

def _names_match(a: str, b: str) -> bool:
    a = (a or "").lower()
    b = (b or "").lower()
    return a == b or a in b or b in a

def _detect_holiday_keyword(message: str) -> Optional[str]:
    m = (message or "").lower()
    for official_name, keywords_list in _HOLIDAY_KEYWORDS.items():
        for keyword in keywords_list:
            k = keyword.lower()
            if (' ' not in k and re.search(r'[a-zA-Z]', k)):
                if re.search(r'\b' + re.escape(k) + r'\b', m):
                    return official_name
            elif k in m:
                return official_name
    return None

# --- NEW: PARSER FOR SPECIAL NAMED DAYS ---
def _parse_special_named_day(message: str, base: datetime) -> Optional[datetime]:
    """
    Parses special, non-public-holiday dates like Christmas Eve.
    """
    m_normalized = (message or "").lower()
    for keyword, date_func in _SPECIAL_NAMED_DAYS.items():
        if keyword in m_normalized:
            # Check this year
            month, day = date_func(base.year)
            dt_this_year = HK_TZ.localize(datetime(base.year, month, day, 12, 0))
            if dt_this_year.date() >= base.date():
                return dt_this_year
            # If past, check next year
            month_next, day_next = date_func(base.year + 1)
            dt_next_year = HK_TZ.localize(datetime(base.year + 1, month_next, day_next, 12, 0))
            return dt_next_year
    return None

def _search_holiday_by_name(message: str, base: datetime) -> Optional[Tuple[datetime, str]]:
    """
    Finds the date of an OFFICIAL PUBLIC HOLIDAY mentioned in the message.
    Robust against calendar naming differences (e.g., 'Dragon Boat Festival' vs 'Tuen Ng Festival').
    Searches this year and next year for the next occurrence on/after 'base'.
    """
    cal_this = _hk_calendar(base.year, base.year)
    cal_next = _hk_calendar(base.year + 1, base.year + 1)
    m_normalized = (message or "").lower()

    # Build keyword -> canonical map and a fast detector of holiday keywords in the message
    keyword_to_official: Dict[str, str] = {}
    for official_name, keywords_list in _HOLIDAY_KEYWORDS.items():
        for keyword in keywords_list:
            keyword_to_official[keyword.lower()] = official_name

    # Detect a canonical holiday from the user's message (via any keyword, incl. Chinese)
    matched_official_name: Optional[str] = None
    for kw, off_name in keyword_to_official.items():
        if ' ' not in kw and re.search(r'[a-zA-Z]', kw):
            if re.search(r'\b' + re.escape(kw) + r'\b', m_normalized):
                matched_official_name = off_name
                break
        elif kw in m_normalized:
            matched_official_name = off_name
            break

    # Helper: expand canonical name to alternative calendar labels and synonyms
    def _candidate_calendar_synonyms(canonical: str) -> List[str]:
        c = canonical.lower()
        syns = {c}
        # Add all known keywords for this canonical holiday
        for s in _HOLIDAY_KEYWORDS.get(canonical, []):
            syns.add(s.lower())
        # Cross-link common alias cases where the official HK holiday name differs
        # - Mid-Autumn Festival -> The day following the Chinese Mid-Autumn Festival (public holiday)
        if c == "mid-autumn festival":
            syns.add("the day following the chinese mid-autumn festival")
        # - Lunar New Year (generic mention) -> First Day of LNY (public holiday)
        if c == "lunar new year":
            syns.add("the first day of lunar new year")
        # - Christmas (generic) -> Christmas Day
        if c == "christmas":
            syns.add("christmas day")
        return list(syns)

    # Helper: find the earliest future calendar match by synonyms, across this and next year
    def _find_future_match_by_synonyms(canonical_name: str) -> Optional[Tuple[datetime, str]]:
        if not (cal_this or cal_next):
            return None
        all_holidays = sorted((cal_this or {}).items()) + sorted((cal_next or {}).items())
        synonyms = _candidate_calendar_synonyms(canonical_name)
        future: List[Tuple[datetime, str]] = []
        for dt_obj, cal_name in all_holidays:
            cal_name_lc = str(cal_name).lower()
            if any(s in cal_name_lc for s in synonyms):
                dt_hk = HK_TZ.localize(datetime.combine(dt_obj, time(12, 0)))
                if dt_hk.date() >= base.date():
                    future.append((dt_hk, str(cal_name)))
        if future:
            return min(future, key=lambda x: x[0])
        return None

    # Primary path: if we detected a holiday keyword in the user message,
    # resolve it to a calendar date using synonym-aware matching.
    if matched_official_name:
        m = _find_future_match_by_synonyms(matched_official_name)
        if m:
            return m
        # If nothing matched, try next year explicitly (already included above), then give up.
        return None

    # Secondary path (calendar scan): If no keyword matched directly, scan all future holidays
    # and use the keyword lists to see if the user's message mentions any of them.
    if cal_this or cal_next:
        all_holidays = sorted((cal_this or {}).items()) + sorted((cal_next or {}).items())
        future_matches: List[Tuple[datetime, str]] = []
        for dt_obj, cal_name in all_holidays:
            dt_hk = HK_TZ.localize(datetime.combine(dt_obj, time(12, 0)))
            if dt_hk.date() < base.date():
                continue
            cal_name_lc = str(cal_name).lower()
            # Map calendar name -> canonical by checking if any canonical's keywords appear in the calendar label
            canonical_for_cal: Optional[str] = None
            for canonical, kw_list in _HOLIDAY_KEYWORDS.items():
                kw_list_lc = [kw.lower() for kw in kw_list]
                if (canonical.lower() in cal_name_lc) or any(kw in cal_name_lc for kw in kw_list_lc):
                    canonical_for_cal = canonical
                    break
            if not canonical_for_cal:
                continue

            # If the user's message mentions any keyword for that canonical holiday, we have a match
            for kw in _HOLIDAY_KEYWORDS.get(canonical_for_cal, []):
                kw_lc = kw.lower()
                if ' ' not in kw_lc and re.search(r'[a-zA-Z]', kw_lc):
                    if re.search(r'\b' + re.escape(kw_lc) + r'\b', m_normalized):
                        future_matches.append((dt_hk, str(cal_name)))
                        break
                elif kw_lc in m_normalized:
                    future_matches.append((dt_hk, str(cal_name)))
                    break

        if future_matches:
            return min(future_matches, key=lambda x: x[0])

    return None

# Minimal fixed-date fallback (only used when 'holidays' is unavailable)
_FIXED_GREGORIAN = {
    "Christmas Day": (12, 25),
    "Labour Day": (5, 1),
    "National Day": (10, 1),
    "HKSAR Establishment Day": (7, 1),
}

def _first_weekday_after(dt: datetime) -> datetime:
    # Mon-Fri as business weekdays; advance to next Mon-Fri strictly after dt
    d = dt + timedelta(days=1)
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d

def _fixed_gregorian_fallback(official_name: str, base: datetime) -> Optional[datetime]:
    name = (official_name or "").strip()
    y = base.year
    if name in _FIXED_GREGORIAN:
        m, d = _FIXED_GREGORIAN[name]
        cand = HK_TZ.localize(datetime(y, m, d, 12, 0))
        if cand.date() < base.date():
            cand = HK_TZ.localize(datetime(y + 1, m, d, 12, 0))
        return cand
    if name == "The first weekday after Christmas Day":
        day = HK_TZ.localize(datetime(y, 12, 25, 12, 0))
        cand = _first_weekday_after(day)
        if cand.date() < base.date():
            day_next = HK_TZ.localize(datetime(y + 1, 12, 25, 12, 0))
            cand = _first_weekday_after(day_next)
        return cand
    return None

_ORD_DAY_PAT = re.compile(
    r"\b(?:on\s+)?(?:the\s+)?((?:[12]?\d|3[01]))(st|nd|rd|th)\b",
    re.IGNORECASE
)
_TIME_PAT = re.compile(r"\b(\d{1,2}):(\d{2})\b|\b(\d{1,2})\s*(am|pm)\b", re.I)

_ZH_NUM = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
_ZH_TIME_PAT = re.compile(r"(上午|早上|中午|下午|晚上)?\s*([一二两三四五六七八九十〇零\d]{1,3})\s*(点|點|时|時)(半)?", re.IGNORECASE)

def _zh_num_to_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        v = int(s)
        return v if 0 <= v <= 23 else None
    total = 0
    if "十" in s:
        parts = s.split("十")
        tens = 1 if parts[0] == "" else _ZH_NUM.get(parts[0], -100)
        ones = _ZH_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] != "" else 0
        if tens < 0:
            return None
        total = tens * 10 + ones
    else:
        total = _ZH_NUM.get(s, -100)
    if 0 <= total <= 23:
        return total
    return None

def _parse_time_zh(msg: str) -> Optional[time]:
    m = _ZH_TIME_PAT.search(msg or "")
    if not m:
        return None
    period = (m.group(1) or "").strip()
    hour_raw = (m.group(2) or "").strip()
    half = bool(m.group(4))
    hh = _zh_num_to_int(hour_raw)
    if hh is None:
        return None
    if period in ("下午", "晚上"):
        if 1 <= hh <= 11:
            hh += 12
    elif period in ("上午", "早上"):
        if hh == 12:
            hh = 0
    mm = 30 if half else 0
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return time(hh, mm)
    return None

def _extract_day_of_month(msg: str) -> Optional[int]:
    # Only accept explicit ordinals (1st/2nd/3rd/…)
    m = _ORD_DAY_PAT.search(msg or "")
    if m:
        try:
            day = int(m.group(1))
            if 1 <= day <= 31:
                return day
        except Exception:
            return None
    return None

def _extract_weekday(msg: str, L: str) -> Optional[int]:
    if L == "en":
        m = _WD_PAT_EN.search(msg or "")
        if not m:
            return None
        key = m.group(1).lower()
        return _WD_MAP_EN.get(key)
    m = _WD_PAT_ZH_HK.search(msg or "")
    if not m:
        return None
    s = m.group(1)
    ch = s[-1]
    return _WD_MAP_ZH.get(ch)

def _parse_time(msg: str) -> Optional[time]:
    m = _TIME_PAT.search(msg or "")
    if m:
        if m.group(1) and m.group(2):
            hh = int(m.group(1)); mm = int(m.group(2))
        else:
            hh = int(m.group(3)); mm = 0
            ap = (m.group(4) or "").lower()
            if ap:
                if hh == 12:
                    hh = 0
                if ap == "pm":
                    hh += 12
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return time(hh, mm)
    return _parse_time_zh(msg or "")

def _relative_offset(message: str, L: str) -> Optional[int]:
    m = message or ""
    if L == "en":
        for k, off in _REL_EN.items():
            if k in m.lower():
                return off
    else:
        for k, off in _REL_ZH.items():
            if k in m:
                return off
    return None

def _parse_datetime(message: str, now: datetime, L: str) -> Optional[datetime]:
    # 1) Relative offsets first (today/tomorrow/etc.)
    rel = _relative_offset(message or "", L)
    if rel is not None:
        base = (now + timedelta(days=rel)).astimezone(HK_TZ)
        t = _parse_time(message or "")
        return base.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0) if t else base.replace(hour=12, minute=0, second=0, microsecond=0)

    # 2) Explicit day-of-month or weekday next
    dom = _extract_day_of_month(message or "")
    wd = _extract_weekday(message or "", L)
    t = _parse_time(message or "")
    if dom is not None or wd is not None:
        base = now + timedelta(days=1)
        for i in range(0, 60):
            cand = (base + timedelta(days=i)).astimezone(HK_TZ)
            if (dom is not None and cand.day != dom) or (wd is not None and cand.weekday() != wd):
                continue
            return cand.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0) if t else cand.replace(hour=12, minute=0, second=0, microsecond=0)

    # 3) Only then try dateparser, and only if it looks like an absolute date
    if dateparser and _looks_like_absolute_date(message or ""):
        settings = {
            "TIMEZONE": "Asia/Hong_Kong",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now,
            "NORMALIZE": True,
            "DATE_ORDER": "DMY",
        }
        langs = ["en"] if L == "en" else ["zh"]
        dt = dateparser.parse(message, settings=settings, languages=langs)
        if dt:
            dt = dt.astimezone(HK_TZ)
            t = _parse_time(message or "")
            return dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0) if t else dt.replace(hour=12, minute=0, second=0, microsecond=0)

    return None

def _next_open_window(start: datetime) -> Tuple[datetime, time, time]:
    cur = start
    for _ in range(14):
        open_t, close_t = _dow_window(cur.weekday())
        if open_t and close_t:
            closed_holiday, _ = _is_public_holiday(cur)
            if not closed_holiday:
                return cur, open_t, close_t
        cur = cur + timedelta(days=1)
        cur = cur.replace(hour=9, minute=0, second=0, microsecond=0)
    next_mon = start + timedelta(days=(7 - start.weekday()) % 7)
    return next_mon, WEEKDAY_OPEN, WEEKDAY_CLOSE

def _localize_holiday_name(name_en: str, L: str) -> str:
    name = (name_en or "").strip()
    if L == "zh-HK":
        mapping = {"Ching Ming": "清明節", "Chung Yeung": "重陽節", "Mid-Autumn": "中秋節", "Tuen Ng": "端午節", "Buddha": "佛誕", "National Day": "國慶日", "Christmas": "聖誕節", "Easter": "復活節", "The day following the Chinese Mid-Autumn Festival": "中秋節翌日", "The first weekday after Christmas Day": "聖誕節後首個工作天"}
    elif L == "zh-CN":
        mapping = {"Ching Ming": "清明节", "Chung Yeung": "重阳节", "Mid-Autumn": "中秋节", "Tuen Ng": "端午节", "Buddha": "佛诞", "National Day": "国庆日", "Christmas": "圣诞节", "Easter": "复活节", "The day following the Chinese Mid-Autumn Festival": "中秋节翌日", "The first weekday after Christmas Day": "圣诞节后第一个工作日"}
    else:
        return name
    for k, v in mapping.items():
        if k.lower() in name.lower(): return v
    return name

def _contains_time_of_day(message: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b", message, flags=re.IGNORECASE) or re.search(r"[一二两三四五六七八九十〇零]{1,3}\s*(点|點|时|時)", message))

# --- REVISED LOGIC WITH NEW PARSING ORDER ---

def _get_opening_facts(message: str, lang: Optional[str] = None, is_general: bool = False) -> Dict[str, Any]:
    """
    --- REWRITTEN FOR ROBUSTNESS ---
    A single source of truth for all date, holiday, and weather analysis.
    Uses a sequential, multi-attempt parsing strategy for maximum accuracy.
    """
    L = _normalize_lang(lang)
    now = datetime.now(HK_TZ)
    msg = message or ""

    dt = None
    holiday_reason = None
    parse_debug = {}
    holiday_kw_official = _detect_holiday_keyword(msg)
    parse_debug["holiday_keyword_detected"] = bool(holiday_kw_official)
    parse_debug["holiday_keyword_official"] = holiday_kw_official

    # --- Attempt 1: Parse special, non-holiday named days (e.g., Christmas Eve). ---
    dt = _parse_special_named_day(msg, now)
    if dt:
        parse_debug["matched_via"] = "special_named_day"

    # --- Attempt 2: If not a special day, search for an official public holiday. ---
    if dt is None:
        holiday_match = _search_holiday_by_name(msg, now)
        if holiday_match:
            dt, holiday_reason = holiday_match
            parse_debug["matched_via"] = "public_holiday"
            parse_debug["holiday_name"] = holiday_reason
    
    parse_debug["is_fallback_to_now"] = (dt is None)
    
    if dt is None:
        # Helpful debug: was a holiday keyword present in the message?
        all_kw = {kw.lower() for v in _HOLIDAY_KEYWORDS.values() for kw in v}
        msg_lc = (msg or "").lower()
        parse_debug["holiday_keyword_present"] = any(
            (re.search(r'\b'+re.escape(kw)+r'\b', msg_lc) if re.search(r'[a-zA-Z]', kw) else kw in msg_lc)
            for kw in all_kw
        )

    # --- Attempt 3: If still no date, use general-purpose date parsing. ---
    if dt is None:
        dt = _parse_datetime(msg, now, L)
        if dt:
            parse_debug["matched_via"] = "general_date_parsing"

    # --- Fallback: If all parsing fails, default to the current time. ---
    final_dt = dt or now
    if not final_dt.tzinfo: 
        final_dt = HK_TZ.localize(final_dt)

    parse_debug["final_datetime"] = final_dt.isoformat()
    parse_debug["is_fallback_to_now"] = (dt is None)

    # --- Post-processing and Fact Assembly ---
    weather_hint = get_weather_hint_for_opening(L) if SETTINGS.opening_hours_weather_enabled else None
    open_t, close_t = _dow_window(final_dt.weekday())
    
    # If we resolved a date but don't have a holiday reason yet, check if it's a holiday.
    # This covers cases like parsing "Dec 25" and then realizing it's Christmas.
    if holiday_reason is None:
        is_holiday, name = _is_public_holiday(final_dt)
        if is_holiday:
            holiday_reason = name

    return {
        "datetime": final_dt,
        "lang": L,
        "open_time": open_t,
        "close_time": close_t,
        "is_sunday": open_t is None and close_t is None,
        "is_holiday": bool(holiday_reason),
        "holiday_name": holiday_reason,
        "weather_hint": weather_hint,
        "asked_specific_time": _contains_time_of_day(msg),
        "is_general_query": is_general,
        "parse_debug": parse_debug,
    }

def extract_opening_context(message: str, lang: Optional[str] = None) -> str:
    facts = _get_opening_facts(message, lang, is_general=False)
    _log(f"Opening parse_debug: {facts.get('parse_debug')}")
    dt = facts["datetime"]
    L = facts["lang"]
    dbg = facts.get("parse_debug", {})
    suppress_hours = bool(dbg.get("holiday_keyword_detected") and dbg.get("is_fallback_to_now"))

    lines = []
    if facts["weather_hint"]:
        lines.append(f"Weather Status: {facts['weather_hint']}")
    lines.append(f"Resolved date: {dt.strftime('%Y-%m-%d')} ({_fmt_date_human(dt, L)})")
    if facts["is_holiday"]:
        lines.append(f"Public holiday: Yes ({facts['holiday_name']})")
    if facts["is_sunday"]:
        lines.append("Day: Sunday (center closed)")
    if (facts["open_time"] and facts["close_time"]
        and not facts["is_holiday"] and not facts["is_sunday"]
        and not suppress_hours):
        lines.append(f"Open hours: {_fmt_time(facts['open_time'])}–{_fmt_time(facts['close_time'])}")
    return "\n".join(lines)

def compute_opening_answer(message: str, lang: Optional[str] = None, brief: bool = False, is_general: bool = False) -> str:
    """
    Deterministic opening-hours answer using a unified facts object.
    Prioritizes closure reasons: 1. Weather, 2. Holiday, 3. Sunday.
    """
    facts = _get_opening_facts(message, lang, is_general=is_general)
    L = facts["lang"]
    dt = facts["datetime"]
    date_h = _fmt_date_human(dt, L)

    def canonical_line() -> str:
        if L == "zh-HK": return "我們的營業時間是：星期一至五 09:00–18:00；星期六 09:00–16:00；星期日及香港公眾假期休息。"
        if L == "zh-CN": return "我们的营业时间是：周一至周五 09:00–18:00；周六 09:00–16:00；周日及香港公众假期休息。"
        return "Our hours are: Mon–Fri 09:00–18:00; Sat 09:00–16:00; closed on Sundays and Hong Kong public holidays."

    # --- Priority 1: Severe Weather ---
    if facts["weather_hint"]:
        if L == "zh-HK": return f"{facts['weather_hint']}。因應惡劣天氣，中心暫停開放，所有課堂取消。"
        if L == "zh-CN": return f"{facts['weather_hint']}。因应恶劣天气，中心暂停开放，所有课程取消。"
        return f"{facts['weather_hint']}. Due to severe weather, the center is closed and all classes are suspended."

    base_next = dt.replace(hour=9, minute=0) + timedelta(days=1)
    nxt_day, n_open, n_close = _next_open_window(base_next)
    
    def next_open_line() -> str:
        if L == "zh-HK": return f"下一個開放時段為 {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。"
        if L == "zh-CN": return f"下一个开放时段为 {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。"
        return f"We will next be open on {_fmt_date_human(nxt_day, L)} from {_fmt_time(n_open)}–{_fmt_time(n_close)}."

    # --- Priority 2: Public Holiday ---
    if facts["is_holiday"]:
        hol_local = _localize_holiday_name(facts['holiday_name'], L)
        if L == "zh-HK": base = f"{date_h} 為香港公眾假期（{hol_local}），中心休息，課堂暫停。\n{next_open_line()}"
        elif L == "zh-CN": base = f"{date_h} 为香港公众假期（{hol_local}），中心休息，课程暂停。\n{next_open_line()}"
        else: base = f"The center is closed on {date_h} for the {hol_local} public holiday. Classes are suspended.\n{next_open_line()}"
        if facts["is_general_query"]: base += f"\n{canonical_line()}"
        return base

    # --- Priority 3: Sunday ---
    if facts["is_sunday"]:
        if L == "zh-HK": base = f"{date_h}（星期日）中心休息，課堂暫停。\n{next_open_line()}"
        elif L == "zh-CN": base = f"{date_h}（周日）中心休息，课程暂停。\n{next_open_line()}"
        else: base = f"The center is closed on {date_h} (Sunday). Classes are suspended.\n{next_open_line()}"
        if facts["is_general_query"]: base += f"\n{canonical_line()}"
        return base

    # --- Priority 4: Regular Open Day ---
    if facts["open_time"] and facts["close_time"]:
        open_str, close_str = _fmt_time(facts["open_time"]), _fmt_time(facts["close_time"])
        if L == "zh-HK": base = f"{date_h} 中心開放，時間為 {open_str}–{close_str}。"
        elif L == "zh-CN": base = f"{date_h} 中心开放，时间为 {open_str}–{close_str}。"
        else: base = f"The center is open on {date_h}. Hours are {open_str}–{close_str}."
        if facts["is_general_query"] and not facts["asked_specific_time"]: base += f"\n{canonical_line()}"
        return base

    # --- Fallback (should be rare) ---
    if L == "zh-HK": return "抱歉，未能解析該日期的營業安排。"
    if L == "zh-CN": return "抱歉，未能解析该日期的营业安排。"
    return "Sorry, I couldn’t resolve the opening arrangement for that date."