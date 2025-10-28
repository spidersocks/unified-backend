from datetime import datetime, timedelta, time
from typing import Optional, Tuple
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
from llm.config import SETTINGS  # NEW

HK_TZ = pytz.timezone("Asia/Hong_Kong")

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

# Weekday patterns (unchanged)
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

# Expanded Holiday keyword hints (English + HK/CN, including common variants)
_HOLIDAY_KEYWORDS = {
    # Lunar New Year and related days
    "lunar new year": "Lunar New Year",
    "chinese new year": "Lunar New Year",
    "the first day of lunar new year": "Lunar New Year",
    "the second day of lunar new year": "Second Day of Lunar New Year",
    "the third day of lunar new year": "Third Day of Lunar New Year",
    "年初一": "Lunar New Year",
    "年初二": "Second Day of Lunar New Year",
    "年初三": "Third Day of Lunar New Year",
    "農曆新年": "Lunar New Year", "农历新年": "Lunar New Year",

    # Ching Ming
    "ching ming": "Ching Ming Festival",
    "tomb-sweeping": "Ching Ming Festival",
    "清明": "Ching Ming Festival", "清明節": "Ching Ming Festival", "清明节": "Ching Ming Festival",

    # Chung Yeung
    "chung yeung": "Chung Yeung Festival",
    "重陽": "Chung Yeung Festival", "重阳": "Chung Yeung Festival", "重陽節": "Chung Yeung Festival", "重阳节": "Chung Yeung Festival",

    # Tuen Ng / Dragon Boat
    "tuen ng": "Tuen Ng Festival",
    "dragon boat": "Tuen Ng Festival",
    "端午": "Tuen Ng Festival", "端午節": "Tuen Ng Festival", "端午节": "Tuen Ng Festival",

    # Mid-Autumn
    "mid-autumn": "Mid-Autumn Festival", "mid autumn": "Mid-Autumn Festival",
    "the day following the chinese mid-autumn festival": "The day following the Chinese Mid-Autumn Festival",
    "中秋": "Mid-Autumn Festival", "中秋節": "Mid-Autumn Festival", "中秋节": "Mid-Autumn Festival",
    "中秋節翌日": "The day following the Chinese Mid-Autumn Festival", "中秋节翌日": "The day following the Chinese Mid-Autumn Festival",

    # Buddha's Birthday
    "buddha": "Buddha's Birthday", "buddha's birthday": "Buddha's Birthday",
    "佛誕": "Buddha's Birthday", "佛诞": "Buddha's Birthday",

    # National Day
    "national day": "National Day",
    "國慶": "National Day", "国庆": "National Day", "國慶日": "National Day", "国庆日": "National Day",

    # Labour Day
    "labour day": "Labour Day", "labor day": "Labour Day",
    "勞動節": "Labour Day", "劳动节": "Labour Day",

    # HKSAR Establishment Day
    "establishment day": "HKSAR Establishment Day", "hksar establishment": "HKSAR Establishment Day",
    "回歸": "HKSAR Establishment Day", "回归": "HKSAR Establishment Day",
    "香港特別行政區成立紀念日": "HKSAR Establishment Day", "香港特别行政区成立纪念日": "HKSAR Establishment Day",

    # Good Friday / Easter Monday
    "good friday": "Good Friday",
    "easter monday": "Easter Monday",
    "耶穌受難日": "Good Friday", "耶稣受难日": "Good Friday",
    "復活節星期一": "Easter Monday", "复活节星期一": "Easter Monday",

    # Christmas day + first weekday after
    "christmas": "Christmas Day", "christmas day": "Christmas Day",
    "the first weekday after christmas day": "The first weekday after Christmas Day",
    "聖誕": "Christmas Day", "圣诞": "Christmas Day",
    "聖誕節": "Christmas Day", "圣诞节": "Christmas Day",
    "聖誕節後首個工作天": "The first weekday after Christmas Day",
    "圣诞节后第一个工作日": "The first weekday after Christmas Day",
}

def _fmt_time(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

def _weekday_label(dt: datetime, L: str) -> str:
    wd = dt.weekday()
    if L == "zh-HK":
        return "星期" + "一二三四五六日"[wd] if wd < 6 else "星期日"
    if L == "zh-CN":
        return "周" + "一二三四五六日"[wd] if wd < 6 else "周日"
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

def _search_holiday_by_name(message: str, base: datetime) -> Optional[Tuple[datetime, str]]:
    cal_this = _hk_calendar(base.year, base.year)
    cal_next = _hk_calendar(base.year + 1, base.year + 1)
    if not (cal_this or cal_next):
        return None
    mlow = (message or "").lower()
    target_kw = None
    for k, en_kw in _HOLIDAY_KEYWORDS.items():
        if k in mlow or k in (message or ""):
            target_kw = en_kw.lower()
            break
    if not target_kw:
        return None
    
def is_general_hours_query(message: str, lang: str) -> bool:
    """
    Returns True if the message is about opening hours/attendance intent,
    but does NOT contain any explicit date, weekday, or relative-day marker.
    Used to distinguish 'What are your opening hours?' from 'Are you open on Sunday?'.
    For EN, treat 'public holiday' or 'holiday' as general.
    """
    from llm.intent import detect_opening_hours_intent
    from llm.opening_hours import _extract_day_of_month, _extract_weekday, _relative_offset
    m = message or ""
    L = lang.lower() if lang else "en"
    is_intent, _ = detect_opening_hours_intent(m, lang, use_llm=True)
    if not is_intent:
        return False
    # If it contains any explicit date/weekday/relative marker, it's not general
    if _extract_day_of_month(m) is not None:
        return False

def _extract_full_chinese_date(msg: str) -> Optional[Tuple[int, int]]:
    """
    Extracts (month, day) from patterns like '12月25號', '12月25日'
    """
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号號]", msg)
    if m:
        try:
            month = int(m.group(1))
            day = int(m.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return (month, day)
        except Exception:
            pass
    return None

# In _parse_datetime, before fallback:
    # 3) Heuristics by weekday or day-of-month
    dom = _extract_day_of_month(message or "")
    wd = _extract_weekday(message or "", L)
    t = _parse_time(message or "")
    # NEW: check for Chinese full date
    full_date = _extract_full_chinese_date(message or "")
    if full_date:
        month, day = full_date
        base = now
        year = base.year
        # If already past, roll to next year
        try:
            candidate = HK_TZ.localize(datetime(year, month, day, 12, 0))
            if candidate < now:
                candidate = HK_TZ.localize(datetime(year+1, month, day, 12, 0))
            if t:
                candidate = candidate.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            return candidate
        except Exception:
            pass
    if _extract_weekday(m, L) is not None:
        return False
    if _relative_offset(m, L) is not None:
        return False
    # crude: numbers + 月/日 in zh, or "<month> <day>" in en
    if L.startswith("zh") and re.search(r"\d{1,2}\s*(月|日|号|號)", m):
        return False
    if L == "en":
        # If message contains an explicit month+day (e.g., Dec 25, 25th December, 12/25), not general
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b", m, re.I):
            return False
        if re.search(r"\b\d{1,2}/\d{1,2}\b", m):
            return False
        # If message contains "public holiday" or "holiday" and NOT "on <date>"
        if re.search(r"\b(public holiday|holiday|holidays?)\b", m, re.I):
            # But if it also contains "on" + date, treat as specific
            if re.search(r"\bon\b.*\d{1,2}", m, re.I):
                return False
            return True
    # weekdays and relative days in any language
    if re.search(r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", m, re.I):
        return False
    if re.search(r"\b(today|tomorrow|yesterday|next week|this week|下周|下星期|本周|本星期|今日|明天|聽日|後日)\b", m, re.I):
        return False
    return True

    def find_in_calendar(cal, yr: int):
        if not cal:
            return None
        for dt, name in cal.items():
            if target_kw in str(name).lower():
                dt_hk = HK_TZ.localize(datetime(yr, dt.month, dt.day, 12, 0))
                return dt_hk, str(name)
        return None

    hit = find_in_calendar(cal_this, base.year)
    if hit:
        return hit
    return find_in_calendar(cal_next, base.year + 1)

_ORD_DAY_PAT = re.compile(r"\b(?:(?:the\s+)?)((?:[12]?\d|3[01]))(?:st|nd|rd|th)?\b", re.I)

# IMPORTANT: tighten time regex so plain integers (like in "10月") are NOT treated as times.
_TIME_PAT = re.compile(r"\b(\d{1,2}):(\d{2})\b|\b(\d{1,2})\s*(am|pm)\b", re.I)

_ZH_NUM = {"零":0,"〇":0,"一":1,"二":1,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
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
    """
    Parse a datetime in HK timezone.
    Priority:
      1) Explicit relative-day words (今天/明天/后天/聽日/後日 | today/tomorrow/day after tomorrow), noon by default.
      2) dateparser (if available); overlay explicit time if present; otherwise set to noon (12:00).
      3) Fallback via weekday or day-of-month heuristics; default time noon.
    """
    # 1) Relative-day words first (most reliable for zh)
    rel = _relative_offset(message or "", L)
    if rel is not None:
        base = (now + timedelta(days=rel)).astimezone(HK_TZ)
        t = _parse_time(message or "")
        if t:
            base = base.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        else:
            base = base.replace(hour=12, minute=0, second=0, microsecond=0)
        return base

    # 2) dateparser
    if dateparser:
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
            if not dt.tzinfo:
                dt = HK_TZ.localize(dt)
            else:
                dt = dt.astimezone(HK_TZ)
            t = _parse_time(message or "")
            if t:
                dt = dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            else:
                dt = dt.replace(hour=12, minute=0, second=0, microsecond=0)
            return dt

    # 3) Heuristics by weekday or day-of-month
    dom = _extract_day_of_month(message or "")
    wd = _extract_weekday(message or "", L)
    t = _parse_time(message or "")

    if dom is None and wd is None:
        return None

    base = now + timedelta(days=1)
    for i in range(0, 60):
        cand = (base + timedelta(days=i)).astimezone(HK_TZ)
        if dom is not None and cand.day != dom:
            continue
        if wd is not None and cand.weekday() != wd:
            continue
        if t:
            cand = cand.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        else:
            cand = cand.replace(hour=12, minute=0, second=0, microsecond=0)
        return cand

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
        mapping = {
            "Ching Ming": "清明節",
            "Chung Yeung": "重陽節",
            "Mid-Autumn": "中秋節",
            "Tuen Ng": "端午節",
            "Buddha": "佛誕",
            "National Day": "國慶日",
            "Christmas": "聖誕節",
            "Easter": "復活節",
            "The day following the Chinese Mid-Autumn Festival": "中秋節翌日",
            "The first weekday after Christmas Day": "聖誕節後首個工作天",
        }
        for k, v in mapping.items():
            if k.lower() in name.lower():
                return v
        return name
    if L == "zh-CN":
        mapping = {
            "Ching Ming": "清明节",
            "Chung Yeung": "重阳节",
            "Mid-Autumn": "中秋节",
            "Tuen Ng": "端午节",
            "Buddha": "佛诞",
            "National Day": "国庆日",
            "Christmas": "圣诞节",
            "Easter": "复活节",
            "The day following the Chinese Mid-Autumn Festival": "中秋节翌日",
            "The first weekday after Christmas Day": "圣诞节后第一个工作日",
        }
        for k, v in mapping.items():
            if k.lower() in name.lower():
                return v
        return name
    return name

def extract_opening_context(message: str, lang: Optional[str] = None) -> str:
    """
    Returns a context string with resolved attendance facts for the LLM.
    """
    L = _normalize_lang(lang)
    now = datetime.now(HK_TZ)
    dt = _parse_datetime(message or "", now, L) or now
    open_t, close_t = _dow_window(dt.weekday())
    is_holiday, holiday_name = _is_public_holiday(dt)
    is_sunday = open_t is None or close_t is None
    weather_hint = get_weather_hint_for_opening(L)
    context_lines = []
    context_lines.append(f"Resolved date: {dt.strftime('%Y-%m-%d')} ({_fmt_date_human(dt, L)})")
    if is_holiday:
        context_lines.append(f"Public holiday: Yes ({holiday_name})")
    if is_sunday:
        context_lines.append("Day: Sunday (center closed)")
    if open_t and close_t and not is_holiday and not is_sunday:
        context_lines.append(f"Open hours: {_fmt_time(open_t)}–{_fmt_time(close_t)}")
    if weather_hint:
        context_lines.append(f"Weather: {weather_hint}")
    return "\n".join(context_lines)

def _contains_time_of_day(message: str) -> bool:
    # Only treat as specific time if there is an explicit time (hh:mm, 3pm, 三点/三點).
    return bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b", message, flags=re.IGNORECASE) or
                re.search(r"[一二两三四五六七八九十〇零]{1,3}\s*(点|點|时|時)", message))

def compute_opening_answer(message: str, lang: Optional[str] = None, brief: bool = False) -> Optional[str]:
    """
    Deterministic opening-hours answer:
    - Uses explicit relative-day words (zh/en) and never inherits current clock time unless a time was given.
    - Weather hint is appended only when severe (Black Rain / Typhoon Signal No. 8+), and only if enabled.
    - Avoids mentioning “not a public holiday” unless asked; only mention holidays when applicable.
    """
    L = _normalize_lang(lang)
    now = datetime.now(HK_TZ)

    hol = _search_holiday_by_name(message or "", now)
    dt = None
    holiday_reason = None
    if hol:
        dt, holiday_reason = hol
    else:
        dt = _parse_datetime(message or "", now, L)

    if not dt:
        dt = now.replace(hour=12, minute=0, second=0, microsecond=0)

    if not dt.tzinfo:
        dt = HK_TZ.localize(dt)

    open_t, close_t = _dow_window(dt.weekday())
    is_sunday = open_t is None or close_t is None

    if holiday_reason is None:
        is_holiday, name = _is_public_holiday(dt)
        if is_holiday:
            holiday_reason = name

    asked_specific_time = _contains_time_of_day(message or "")
    date_h = _fmt_date_human(dt, L)

    def canonical_line() -> str:
        if L == "zh-HK":
            return "營業時間：星期一至五 09:00–18:00；星期六 09:00–16:00；香港公眾假期休息。"
        if L == "zh-CN":
            return "营业时间：周一至周五 09:00–18:00；周六 09:00–16:00；香港公众假期休息。"
        return "Hours: Mon–Fri 09:00–18:00; Sat 09:00–16:00; closed on Hong Kong public holidays."

    def maybe_weather_hint() -> Optional[str]:
        if not SETTINGS.opening_hours_weather_enabled:
            return None
        return get_weather_hint_for_opening(L)

    # Only include canonical_line for general queries
    from llm.intent import is_general_hours_query
    include_canonical = is_general_hours_query(message, L)

    # Holiday
    if holiday_reason:
        hol_local = _localize_holiday_name(holiday_reason, L)
        base_next = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            base = f"{date_h}為香港公眾假期（{hol_local}），中心休息。課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。"
            if include_canonical:
                base += f"\n{canonical_line()}"
        elif L == "zh-CN":
            base = f"{date_h}为香港公众假期（{hol_local}），中心休息。课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。"
            if include_canonical:
                base += f"\n{canonical_line()}"
        else:
            base = f"Closed on {date_h} due to Hong Kong public holiday: {hol_local}. Classes are suspended.\nNext open window: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}."
            if include_canonical:
                base += f"\n{canonical_line()}"
        hint = maybe_weather_hint()
        return base if not hint else f"{base}\n{hint}"