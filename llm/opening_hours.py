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

# … [unchanged code omitted for brevity] …

def _dow_window(dow: int) -> Tuple[Optional[time], Optional[time]]:
    # Monday=0 ... Sunday=6
    if 0 <= dow <= 4:
        return WEEKDAY_OPEN, WEEKDAY_CLOSE
    if dow == 5:
        return SAT_OPEN, SAT_CLOSE
    return None, None  # Sunday closed

@lru_cache(maxsize=8)
def _hk_calendar(start_year: int, end_year: int):
    """
    Cache the HK holiday calendar for a range of years to avoid rebuilding on each call.
    """
    if not holidays:
        return None
    years = list(range(start_year, end_year + 1))
    try:
        return holidays.HK(years=years)  # type: ignore
    except Exception:
        return None

def _is_public_holiday(d: datetime) -> Tuple[bool, Optional[str]]:
    # Build once per (y-1..y+1) window; cached by @lru_cache
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
    # If holidays lib missing or failed, skip name search
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
_TIME_PAT = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)

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
    # zh
    m = _WD_PAT_ZH_HK.search(msg or "")
    if not m:
        return None
    ch = m.group(1)[2]  # e.g., 星期三 -> 三
    return _WD_MAP_ZH.get(ch)

def _parse_time(msg: str) -> Optional[time]:
    m = _TIME_PAT.search(msg or "")
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2) or "0")
    ap = (m.group(3) or "").lower()
    if ap:
        if hh == 12:
            hh = 0
        if ap == "pm":
            hh += 12
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return time(hh, mm)
    return None

def _parse_datetime(message: str, now: datetime, L: str) -> Optional[datetime]:
    # Try dateparser first with language hints and HK settings
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
            return dt

    # Fallback: manual extraction
    dom = _extract_day_of_month(message or "")
    wd = _extract_weekday(message or "", L)
    t = _parse_time(message or "")

    if dom is None and wd is None:
        return None

    # Start from tomorrow for "future" preference
    base = now + timedelta(days=1)
    # Find next date matching conditions within 60 days
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

def _contains_time_of_day(message: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b|\d\s*點|\d\s*点", message, flags=re.IGNORECASE))

def compute_opening_answer(message: str, lang: Optional[str] = None) -> Optional[str]:
    """
    Returns a localized, class-oriented answer string or None if we can't parse.
    """
    L = _normalize_lang(lang)
    now = datetime.now(HK_TZ)

    # 1) Holiday by name (e.g., 中秋節/Ching Ming)
    hol = _search_holiday_by_name(message or "", now)
    dt = None
    holiday_reason = None
    if hol:
        dt, holiday_reason = hol
    else:
        # 2) Parse relative/absolute date-time from text with fallbacks
        dt = _parse_datetime(message or "", now, L)

    # If nothing parsed, assume "today" without time for general arrangements-type questions
    if not dt:
        dt = now

    # Normalize to HK tz
    if not dt.tzinfo:
        dt = HK_TZ.localize(dt)

    # 3) Business window for the target datetime
    open_t, close_t = _dow_window(dt.weekday())
    is_sunday = open_t is None or close_t is None

    # 4) Public holiday check (if not already identified by name)
    if holiday_reason is None:
        is_holiday, name = _is_public_holiday(dt)
        if is_holiday:
            holiday_reason = name

    asked_specific_time = _contains_time_of_day(message or "")

    def canonical_line() -> str:
        if L == "zh-HK":
            return "營業時間：星期一至五 09:00–18:00；星期六 09:00–16:00；香港公眾假期休息。"
        if L == "zh-CN":
            return "营业时间：周一至周五 09:00–18:00；周六 09:00–16:00；香港公众假期休息。"
        return "Hours: Mon–Fri 09:00–18:00; Sat 09:00–16:00; closed on Hong Kong public holidays."

    date_h = _fmt_date_human(dt, L)

    # Holiday: closed, classes suspended
    if holiday_reason:
        hol_local = _localize_holiday_name(holiday_reason, L)
        base_next = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            return f"{date_h}為香港公眾假期（{hol_local}），中心休息。課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        if L == "zh-CN":
            return f"{date_h}为香港公众假期（{hol_local}），中心休息。课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        return f"Closed on {date_h} due to Hong Kong public holiday: {hol_local}. Classes are suspended.\nNext open window: { _fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"

    # Sunday: closed, classes suspended
    if is_sunday:
        base_next = dt + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            return f"{date_h}逢星期日休息，課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        if L == "zh-CN":
            return f"{date_h}周日休息，课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        return f"Closed on {date_h} (Sunday). Classes are suspended.\nNext open window: { _fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"

    # Open day
    # If time specified, check in-window
    if asked_specific_time:
        t = dt.timetz()
        within = (open_t <= t.replace(tzinfo=None) <= close_t)
        if within:
            if L == "zh-HK":
                return f"{date_h} {dt.strftime('%H:%M')} 為開放時段內（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。課堂如常進行。非香港公眾假期。"
            if L == "zh-CN":
                return f"{date_h} {dt.strftime('%H:%M')} 在开放时段内（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。课程如常进行。非香港公众假期。"
            return f"Open on {date_h} at {dt.strftime('%H:%M')} (within {_fmt_time(open_t)}–{_fmt_time(close_t)}). Classes proceed as usual. Not a Hong Kong public holiday."
        else:
            # Closed at that time; suggest next window
            if dt.time() < open_t:
                nxt_day, n_open, n_close = dt, open_t, close_t
            else:
                base_next = dt + timedelta(days=1)
                nxt_day, n_open, n_close = _next_open_window(base_next)
            if L == "zh-HK":
                return f"{date_h} {dt.strftime('%H:%M')} 不在開放時段內（當日：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。該時段不設課堂。下一個開放時段：{_fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
            if L == "zh-CN":
                return f"{date_h} {dt.strftime('%H:%M')} 不在开放时段内（当日：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。该时段不设课程。下一个开放时段：{_fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
            return f"Closed at {dt.strftime('%H:%M')} on {date_h} (day window: { _fmt_time(open_t)}–{_fmt_time(close_t)}). No classes at that time. Next open window: { _fmt_date_human(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"

    # Day-level answer (no specific time)
    if L == "zh-HK":
        return f"{date_h}中心開放（時段：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。課堂如常進行。非香港公眾假期。\n{canonical_line()}"
    if L == "zh-CN":
        return f"{date_h}中心开放（时段：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。课程如常进行。非香港公众假期。\n{canonical_line()}"
    return f"Open on {date_h} (window: { _fmt_time(open_t)}–{_fmt_time(close_t)}). Classes proceed as usual. Not a Hong Kong public holiday.\n{canonical_line()}"