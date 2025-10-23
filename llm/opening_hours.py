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
from llm.hko import get_weather_hint_for_opening  # NEW

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

# Weekday detection (English)
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

# Weekday detection (Chinese; supports 星期/周/週/礼拜/禮拜)
_WD_PAT_ZH_HK = re.compile(r"(星期[一二三四五六日天]|周[一二三四五六日天]|週[一二三四五六日天]|礼拜[一二三四五六日天]|禮拜[一二三四五六日天])")
_WD_MAP_ZH = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# Holiday keyword hints -> English name substrings used by holidays.HK names
_HOLIDAY_KEYWORDS = {
    "ching ming": "Ching Ming", "清明": "Ching Ming",
    "chung yeung": "Chung Yeung", "重陽": "Chung Yeung", "重阳": "Chung Yeung",
    "mid-autumn": "Mid-Autumn", "mid autumn": "Mid-Autumn", "中秋": "Mid-Autumn",
    "tuen ng": "Tuen Ng", "dragon boat": "Tuen Ng", "端午": "Tuen Ng",
    "buddha": "Buddha", "佛誕": "Buddha", "佛诞": "Buddha",
    "national day": "National Day", "國慶": "National Day", "国庆": "National Day",
    "christmas": "Christmas", "聖誕": "Christmas", "圣诞": "Christmas",
    "easter": "Easter", "復活": "Easter", "复活": "Easter",
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

from functools import lru_cache
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
    m = _WD_PAT_ZH_HK.search(msg or "")
    if not m:
        return None
    s = m.group(1)
    ch = s[-1]
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

def _contains_time_of_day(message: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b|\d\s*點|\d\s*点", message, flags=re.IGNORECASE))

def compute_opening_answer(message: str, lang: Optional[str] = None, brief: bool = False) -> Optional[str]:
    """
    Returns a localized answer string or None if we can't parse.
    If brief=True, return compact phrasing without boilerplate.
    Automatically appends a live HKO weather hint (if severe/relevant) at the end.
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
        dt = now

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
            hint = get_weather_hint_for_opening(L)
            return base if not hint else f"{base}\n{hint}"
        base_next = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            base = f"{date_h}為香港公眾假期（{hol_local}），中心休息。課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        elif L == "zh-CN":
            base = f"{date_h}为香港公众假期（{hol_local}），中心休息。课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        else:
            base = f"Closed on {date_h} due to Hong Kong public holiday: {hol_local}. Classes are suspended.\nNext open window: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        hint = get_weather_hint_for_opening(L)
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
            hint = get_weather_hint_for_opening(L)
            return base if not hint else f"{base}\n{hint}"
        base_next = dt + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            base = f"{date_h}逢星期日休息，課堂暫停。\n下一個開放時段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        elif L == "zh-CN":
            base = f"{date_h}周日休息，课程暂停。\n下一个开放时段：{_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        else:
            base = f"Closed on {date_h} (Sunday). Classes are suspended.\nNext open window: {_fmt_date_human(nxt_day, L)} {_fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        hint = get_weather_hint_for_opening(L)
        return base if not hint else f"{base}\n{hint}"

    # Open day
    if asked_specific_time:
        t = dt.timetz()
        within = (open_t <= t.replace(tzinfo=None) <= close_t)
        if brief:
            if within:
                if L == "zh-HK":
                    base = f"{date_h} {dt.strftime('%H:%M')} 照常上課。"
                elif L == "zh-CN":
                    base = f"{date_h} {dt.strftime('%H:%M')} 照常上课。"
                else:
                    base = f"Open on {date_h} at {dt.strftime('%H:%M')}."
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
            hint = get_weather_hint_for_opening(L)
            return base if not hint else f"{base}\n{hint}"
        # verbose path
        if within:
            if L == "zh-HK":
                base = f"{date_h} {dt.strftime('%H:%M')} 為開放時段內（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。課堂如常進行。非香港公眾假期."
            elif L == "zh-CN":
                base = f"{date_h} {dt.strftime('%H:%M')} 在开放时段内（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。课程如常进行。非香港公众假期。"
            else:
                base = f"Open on {date_h} at {dt.strftime('%H:%M')} (within {_fmt_time(open_t)}–{_fmt_time(close_t)}). Classes proceed as usual. Not a Hong Kong public holiday."
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
        hint = get_weather_hint_for_opening(L)
        return base if not hint else f"{base}\n{hint}"

    # Day-level (no specific time)
    if brief:
        if L == "zh-HK":
            base = f"{date_h}中心開放（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。"
        elif L == "zh-CN":
            base = f"{date_h}中心开放（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。"
        else:
            base = f"Open on {date_h} ({_fmt_time(open_t)}–{_fmt_time(close_t)})."
        hint = get_weather_hint_for_opening(L)
        return base if not hint else f"{base}\n{hint}"
    if L == "zh-HK":
        base = f"{date_h}中心開放（時段：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。課堂如常進行。非香港公眾假期。\n{canonical_line()}"
    elif L == "zh-CN":
        base = f"{date_h}中心开放（时段：{_fmt_time(open_t)}–{_fmt_time(close_t)}）。课程如常进行。非香港公众假期。\n{canonical_line()}"
    else:
        base = f"Open on {date_h} (window: {_fmt_time(open_t)}–{_fmt_time(close_t)}). Classes proceed as usual. Not a Hong Kong public holiday.\n{canonical_line()}"
    hint = get_weather_hint_for_opening(L)
    return base if not hint else f"{base}\n{hint}"