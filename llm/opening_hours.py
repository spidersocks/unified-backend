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

# Holiday keyword hints - expanded to cover common synonyms in English, zh-HK, zh-CN
# Maps user input keywords to the Chinese holiday labels returned by holidays.HK library
_HOLIDAY_KEYWORDS = {
    # Lunar New Year / Spring Festival -> 農曆年初一/初二/初三
    "lunar new year": "農曆年初",
    "chinese new year": "農曆年初",
    "spring festival": "農曆年初",
    "新年": "農曆年初",
    "農曆新年": "農曆年初",
    "农历新年": "農曆年初",
    "春節": "農曆年初",
    "春节": "農曆年初",
    "年初一": "農曆年初一",
    
    # Ching Ming -> 清明節
    "ching ming": "清明節",
    "qingming": "清明節",
    "清明": "清明節",
    "清明節": "清明節",
    "清明节": "清明節",
    
    # Good Friday -> 耶穌受難節 (in 2024)
    "good friday": "耶穌受難節",
    "耶穌受難節": "耶穌受難節",
    "耶稣受难日": "耶穌受難節",
    "受難節": "耶穌受難節",
    "受难节": "耶穌受難節",
    
    # Easter Monday -> 復活節星期一
    "easter monday": "復活節",
    "easter": "復活節",
    "復活節": "復活節",
    "复活节": "復活節",
    "復活節星期一": "復活節星期一",
    "复活节星期一": "復活節星期一",
    
    # Labour Day / Labor Day -> 勞動節
    "labour day": "勞動節",
    "labor day": "勞動節",
    "勞動節": "勞動節",
    "劳动节": "勞動節",
    
    # Buddha's Birthday -> 佛誕
    "buddha": "佛誕",
    "buddha's birthday": "佛誕",
    "佛誕": "佛誕",
    "佛诞": "佛誕",
    "佛誕日": "佛誕",
    "佛诞日": "佛誕",
    
    # Tuen Ng Festival / Dragon Boat Festival -> 端午節
    "tuen ng": "端午節",
    "dragon boat": "端午節",
    "duanwu": "端午節",
    "端午": "端午節",
    "端午節": "端午節",
    "端午节": "端午節",
    
    # HKSAR Establishment Day -> 香港特別行政區成立紀念日
    "hksar": "香港特別行政區成立紀念日",
    "establishment day": "香港特別行政區成立紀念日",
    "hong kong establishment": "香港特別行政區成立紀念日",
    "回歸": "香港特別行政區成立紀念日",
    "回归": "香港特別行政區成立紀念日",
    "回歸紀念日": "香港特別行政區成立紀念日",
    "回归纪念日": "香港特別行政區成立紀念日",
    "香港特別行政區成立紀念日": "香港特別行政區成立紀念日",
    "香港特别行政区成立纪念日": "香港特別行政區成立紀念日",
    "七一": "香港特別行政區成立紀念日",
    
    # National Day -> 國慶日
    "national day": "國慶日",
    "國慶": "國慶日",
    "国庆": "國慶日",
    "國慶日": "國慶日",
    "国庆日": "國慶日",
    "十一": "國慶日",
    
    # Mid-Autumn Festival and following day -> 中秋節翌日
    "mid-autumn": "中秋節",
    "mid autumn": "中秋節",
    "moon festival": "中秋節",
    "中秋": "中秋節",
    "中秋節": "中秋節",
    "中秋节": "中秋節",
    
    # Chung Yeung Festival -> 重陽節
    "chung yeung": "重陽節",
    "chong yang": "重陽節",
    "double ninth": "重陽節",
    "重陽": "重陽節",
    "重阳": "重陽節",
    "重陽節": "重陽節",
    "重阳节": "重陽節",
    
    # Christmas -> 聖誕節
    "christmas": "聖誕節",
    "xmas": "聖誕節",
    "聖誕": "聖誕節",
    "圣诞": "聖誕節",
    "聖誕節": "聖誕節",
    "圣诞节": "聖誕節",
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
    for k, ch_name in _HOLIDAY_KEYWORDS.items():
        if k in mlow or k in (message or ""):
            target_kw = ch_name
            break
    if not target_kw:
        return None

    def find_in_calendar(cal, yr: int):
        if not cal:
            return None
        for dt, name in cal.items():
            # Match against the Chinese name from holidays library
            # Support partial match since we might have "農曆年初" matching "農曆年初一/初二/初三"
            if target_kw in str(name) or str(name) in target_kw:
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

def _localize_holiday_name(name_zh: str, L: str) -> str:
    """
    The holidays library returns Chinese names by default.
    Convert to appropriate language if needed, or return as-is for Chinese locales.
    """
    name = (name_zh or "").strip()
    
    # For Chinese locales, convert traditional/simplified as needed
    if L == "zh-CN":
        # Convert traditional to simplified for common holiday names
        mapping = {
            "農曆年初": "农历年初",
            "清明節": "清明节",
            "勞動節": "劳动节",
            "佛誕": "佛诞",
            "端午節": "端午节",
            "香港特別行政區成立紀念日": "香港特别行政区成立纪念日",
            "國慶日": "国庆日",
            "中秋節": "中秋节",
            "中秋節翌日": "中秋节翌日",
            "重陽節": "重阳节",
            "聖誕節": "圣诞节",
            "聖誕節後": "圣诞节后",
        }
        for trad, simp in mapping.items():
            if trad in name:
                name = name.replace(trad, simp)
        return name
    
    if L == "en":
        # Convert to English
        mapping = {
            "一月一日": "New Year's Day",
            "農曆年初一": "Lunar New Year's Day",
            "農曆年初二": "The second day of Lunar New Year",
            "農曆年初三": "The third day of Lunar New Year",
            "清明節": "Ching Ming Festival",
            "勞動節": "Labour Day",
            "佛誕": "Buddha's Birthday",
            "端午節": "Tuen Ng Festival",
            "香港特別行政區成立紀念日": "HKSAR Establishment Day",
            "國慶日": "National Day",
            "中秋節翌日": "The day following the Chinese Mid-Autumn Festival",
            "重陽節": "Chung Yeung Festival",
            "聖誕節": "Christmas Day",
            "聖誕節後第一個周日": "The first weekday after Christmas Day",
        }
        for zh, en in mapping.items():
            if zh in name:
                return en
        return name
    
    # For zh-HK, return as-is (already in traditional Chinese)
    return name

def _contains_time_of_day(message: str) -> bool:
    # Only treat as specific time if there is an explicit time (hh:mm, 3pm, 三点/三點).
    return bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b", message, flags=re.IGNORECASE) or
                re.search(r"[一二两三四五六七八九十〇零]{1,3}\s*(点|點|时|時)", message))

def compute_opening_answer(message: str, lang: Optional[str] = None, brief: bool = False) -> Optional[str]:
    """
    Deterministic opening-hours answer:
    - Uses explicit relative-day words (zh/en) and never inherits current clock time unless a time was given.
    - Weather hint is appended only when severe (Black Rain / Typhoon Signal No. 8+), and only if enabled.
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
        # Could not parse a date; answer day-level about "today"
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
        # hko.get_weather_hint_for_opening already filters to severe only;
        # the flag allows global on/off control.
        return get_weather_hint_for_opening(L)

    # Holiday
    if holiday_reason:
        hol_local = _localize_holiday_name(holiday_reason, L)
        if brief:
            if L == "zh-HK":
                base = f"{date_h}因香港公眾假期（{hol_local}）休息。課堂暫停。"
            elif L == "zh-CN":
                base = f"{date_h}因香港公众假期（{hol_local}）休息。课程暂停。"
            else:
                base = f"Closed on {date_h} for Hong Kong public holiday: {hol_local}. Classes are suspended."
            hint = maybe_weather_hint()
            return base if not hint else f"{base}\n{hint}"
        base_next = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            base = f"{date_h}為香港公眾假期（{hol_local}），中心休息。課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        elif L == "zh-CN":
            base = f"{date_h}为香港公众假期（{hol_local}），中心休息。课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        else:
            base = f"Closed on {date_h} due to Hong Kong public holiday: {hol_local}. Classes are suspended.\nNext open window: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        hint = maybe_weather_hint()
        return base if not hint else f"{base}\n{hint}"

    # Sunday
    if is_sunday:
        if brief:
            if L == "zh-HK":
                base = f"{date_h}逢星期日休息，課堂暫停。"
            elif L == "zh-CN":
                base = f"{date_h}周日休息，课程暂停。"
            else:
                base = f"Closed on {date_h} (Sunday). Classes are suspended."
            hint = maybe_weather_hint()
            return base if not hint else f"{base}\n{hint}"
        base_next = dt + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            base = f"{date_h}逢星期日休息，課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        elif L == "zh-CN":
            base = f"{date_h}周日休息，课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        else:
            base = f"Closed on {date_h} (Sunday). Classes are suspended.\nNext open window: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        hint = maybe_weather_hint()
        return base if not hint else f"{base}\n{hint}"

    # Open day
    if asked_specific_time:
        t = dt.timetz()
        within = (open_t <= t.replace(tzinfo=None) <= close_t)
        if brief:
            if within:
                base = f"{date_h} {dt.strftime('%H:%M')} 照常上課。" if L == "zh-HK" else (f"{date_h} {dt.strftime('%H:%M')} 照常上课。" if L == "zh-CN" else f"Open on {date_h} at {dt.strftime('%H:%M')}.")
            else:
                if dt.time() < open_t:
                    nxt_day, n_open, n_close = dt, open_t, close_t
                else:
                    base_next = dt + timedelta(days=1)
                    nxt_day, n_open, n_close = _next_open_window(base_next)
                if L == "zh-HK":
                    base = f"{date_h} {dt.strftime('%H:%M')} 非開放時段。下一時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。"
                elif L == "zh-CN":
                    base = f"{date_h} {dt.strftime('%H:%M')} 非开放时段。下一时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。"
                else:
                    base = f"Closed at {dt.strftime('%H:%M')} on {date_h}. Next open: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}."
            hint = maybe_weather_hint()
            return base if not hint else f"{base}\n{hint}"

        if within:
            if L == "zh-HK":
                base = f"{date_h} {dt.strftime('%H:%M')} 為開放時段內（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。課堂如常進行。"
            elif L == "zh-CN":
                base = f"{date_h} {dt.strftime('%H:%M')} 在开放时段内（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。课程如常进行。"
            else:
                base = f"Open on {date_h} at {dt.strftime('%H:%M')} (within {_fmt_time(open_t)}–{_fmt_time(close_t)}). Classes proceed as usual."
        else:
            if dt.time() < open_t:
                nxt_day, n_open, n_close = dt, open_t, close_t
            else:
                base_next = dt + timedelta(days=1)
                nxt_day, n_open, n_close = _next_open_window(base_next)
            if L == "zh-HK":
                base = f"{date_h} {dt.strftime('%H:%M')} 不在開放時段內（當日：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。該時段不設課堂。下一個開放時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
            elif L == "zh-CN":
                base = f"{date_h} {dt.strftime('%H:%M')} 不在开放时段内（当日：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。该时段不设课程。下一个开放时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
            else:
                base = f"Closed at {dt.strftime('%H:%M')} on {date_h} (day window: {_fmt_time(open_t)}–{_fmt_time(close_t)}). No classes at that time. Next open window: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        hint = maybe_weather_hint()
        return base if not hint else f"{base}\n{hint}"

    # Day-level (no specific time)
    if brief:
        if L == "zh-HK":
            base = f"{date_h}中心開放（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。"
        elif L == "zh-CN":
            base = f"{date_h}中心开放（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。"
        else:
            base = f"Open on {date_h} ({_fmt_time(open_t)}–{_fmt_time(close_t)})."
        hint = maybe_weather_hint()
        return base if not hint else f"{base}\n{hint}"
    if L == "zh-HK":
        base = f"{date_h}中心開放（時段：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。課堂如常進行。\n{canonical_line()}"
    elif L == "zh-CN":
        base = f"{date_h}中心开放（时段：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。课程如常进行。\n{canonical_line()}"
    else:
        base = f"Open on {date_h} (window: {_fmt_time(open_t)}–{_fmt_time(close_t)}). Classes proceed as usual.\n{canonical_line()}"
    hint = maybe_weather_hint()
    return base if not hint else f"{base}\n{hint}"