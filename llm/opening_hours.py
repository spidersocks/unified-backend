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

# Common holiday keyword mapping (for queries by name)
_HOLIDAY_KEYWORDS = {
    # zh-HK / zh-CN -> English keyword appearing in holidays.HK names
    "中秋": "Mid-Autumn",
    "中秋節": "Mid-Autumn",
    "中秋节": "Mid-Autumn",
    "清明": "Ching Ming",
    "清明節": "Ching Ming",
    "清明节": "Ching Ming",
    "重陽": "Chung Yeung",
    "重阳": "Chung Yeung",
    "重陽節": "Chung Yeung",
    "重阳节": "Chung Yeung",
    "端午": "Tuen Ng",
    "端午節": "Tuen Ng",
    "端午节": "Tuen Ng",
    "佛誕": "Buddha",
    "佛诞": "Buddha",
    "國慶": "National Day",
    "国庆": "National Day",
    "聖誕": "Christmas",
    "圣诞": "Christmas",
    "復活節": "Easter",
    "复活节": "Easter",
    # English direct
    "mid-autumn": "Mid-Autumn",
    "ching ming": "Ching Ming",
    "chung yeung": "Chung Yeung",
    "tuen ng": "Tuen Ng",
    "buddha": "Buddha",
    "national day": "National Day",
    "christmas": "Christmas",
    "easter": "Easter",
}

# Localized phrases
def _fmt_date(dt: datetime, L: str) -> str:
    if L == "zh-HK" or L == "zh-CN":
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")

def _fmt_time(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

def _dow_window(dow: int) -> Tuple[Optional[time], Optional[time]]:
    # Monday=0 ... Sunday=6
    if 0 <= dow <= 4:
        return WEEKDAY_OPEN, WEEKDAY_CLOSE
    if dow == 5:
        return SAT_OPEN, SAT_CLOSE
    return None, None  # Sunday closed

def _is_public_holiday(d: datetime) -> Tuple[bool, Optional[str]]:
    if not holidays:
        return False, None
    # Generate holidays for +/- 1 year window around the date
    yrs = {d.year - 1, d.year, d.year + 1}
    hk = holidays.HK(years=list(yrs))  # type: ignore
    # The holidays lib uses date() keys (no tz). Compare by date.
    name = hk.get(d.date())
    if name:
        return True, str(name)
    return False, None

def _search_holiday_by_name(message: str, base: datetime) -> Optional[Tuple[datetime, str]]:
    if not holidays:
        return None
    mlow = (message or "").lower()
    # Try to locate an English keyword or mapped zh keyword
    target_kw = None
    for k, en_kw in _HOLIDAY_KEYWORDS.items():
        if k in mlow:
            target_kw = en_kw.lower()
            break
    if not target_kw:
        # try zh chars in original text
        for k, en_kw in _HOLIDAY_KEYWORDS.items():
            if k in (message or ""):
                target_kw = en_kw.lower()
                break
    if not target_kw:
        return None
    # Look through current and next year for the first matching holiday
    for yr in [base.year, base.year + 1]:
        hk = holidays.HK(years=[yr])  # type: ignore
        for dt, name in hk.items():
            if target_kw in str(name).lower():
                # Use that date at 12:00 for clarity
                dt_hk = HK_TZ.localize(datetime(dt.year, dt.month, dt.day, 12, 0))
                return dt_hk, str(name)
    return None

def _parse_datetime(message: str, now: datetime) -> Optional[datetime]:
    if not dateparser:
        return None
    settings = {
        "TIMEZONE": "Asia/Hong_Kong",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": now,
        "PARSERS": ["relative-time", "absolute-time", "timestamp", "custom-formats"],
        "NORMALIZE": True,
    }
    # Try with languages hint; dateparser can auto-detect, but we attempt twice
    dt = dateparser.parse(message, settings=settings)
    return dt

def _next_open_window(start: datetime) -> Tuple[datetime, time, time]:
    # Find the next day with opening hours and not a holiday
    cur = start
    for _ in range(14):  # two weeks safety bound
        open_t, close_t = _dow_window(cur.weekday())
        if open_t and close_t:
            closed_holiday, _ = _is_public_holiday(cur)
            if not closed_holiday:
                return cur, open_t, close_t
        cur = cur + timedelta(days=1)
        cur = cur.replace(hour=9, minute=0, second=0, microsecond=0)
    # Fallback: return next Monday window
    next_mon = start + timedelta(days=(7 - start.weekday()) % 7)
    return next_mon, WEEKDAY_OPEN, WEEKDAY_CLOSE

def _localize_holiday_name(name_en: str, L: str) -> str:
    name = (name_en or "").strip()
    if L == "zh-HK":
        # Minimal mappings (fallback to English if unknown)
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
            "The first weekday after Christmas Day": "聖誕節後首個周日翌日",
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
    Returns a localized answer string or None if we can't parse.
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
        # 2) Parse relative/absolute date-time from text
        dt = _parse_datetime(message or "", now)

    # If nothing parsed, assume "today" without time for arrangements-style questions
    if not dt:
        dt = now

    # Normalize to HK tz
    if not dt.tzinfo:
        dt = HK_TZ.localize(dt)

    # 3) Business window for the target datetime
    open_t, close_t = _dow_window(dt.weekday())
    is_sunday = open_t is None or close_t is None

    # 4) Public holiday check
    if holiday_reason is None:
        is_holiday, name = _is_public_holiday(dt)
        if is_holiday:
            holiday_reason = name

    # 5) Evaluate time-of-day if present or assume at the provided time
    asked_specific_time = _contains_time_of_day(message or "")

    def canonical_line() -> str:
        if L == "zh-HK":
            return "營業時間：星期一至五 09:00–18:00；星期六 09:00–16:00；香港公眾假期休息。"
        if L == "zh-CN":
            return "营业时间：周一至周五 09:00–18:00；周六 09:00–16:00；香港公众假期休息。"
        return "Hours: Mon–Fri 09:00–18:00; Sat 09:00–16:00; closed on Hong Kong public holidays."

    # 6) Build answer
    date_str = _fmt_date(dt, L)
    if holiday_reason:
        hol_local = _localize_holiday_name(holiday_reason, L)
        # Closed; compute next open window
        base_next = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            ans = f"{date_str}因香港公眾假期（{hol_local}）休息。下一個開放時段：{_fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        elif L == "zh-CN":
            ans = f"{date_str}因香港公众假期（{hol_local}）休息。下一个开放时段：{_fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        else:
            ans = f"Closed on {date_str} due to a Hong Kong public holiday: {hol_local}. Next open window: { _fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        return ans

    if is_sunday:
        # Closed; find next open window (Monday)
        base_next = dt + timedelta(days=1)
        nxt_day, n_open, n_close = _next_open_window(base_next)
        if L == "zh-HK":
            ans = f"{date_str}逢星期日休息。下一個開放時段：{_fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        elif L == "zh-CN":
            ans = f"{date_str}周日休息。下一个开放时段：{_fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
        else:
            ans = f"Closed on {date_str} (Sunday). Next open window: { _fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"
        return ans

    # Within a normal open day
    # Check whether specified time is within hours; if no time asked, provide day window
    if asked_specific_time:
        t = dt.timetz()
        within = (open_t <= t.replace(tzinfo=None) <= close_t)
        if within:
            if L == "zh-HK":
                return f"{date_str} {dt.strftime('%H:%M')} 仍在開放時段內（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。\n{canonical_line()}"
            if L == "zh-CN":
                return f"{date_str} {dt.strftime('%H:%M')} 在开放时段内（{_fmt_time(open_t)}–{_fmt_time(close_t)}）。\n{canonical_line()}"
            return f"Yes — {date_str} at {dt.strftime('%H:%M')} is within opening hours ({_fmt_time(open_t)}–{_fmt_time(close_t)}).\n{canonical_line()}"
        else:
            # Closed at that time; suggest next window
            if dt.time() < open_t:
                # earlier than open — opens today
                nxt_day, n_open, n_close = dt, open_t, close_t
            else:
                # after close — next open window
                base_next = dt + timedelta(days=1)
                nxt_day, n_open, n_close = _next_open_window(base_next)
            if L == "zh-HK":
                return f"{date_str} {dt.strftime('%H:%M')} 不在開放時段內。當日時段：{_fmt_time(open_t)}–{_fmt_time(close_t)}。下一個開放時段：{_fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
            if L == "zh-CN":
                return f"{date_str} {dt.strftime('%H:%M')} 不在开放时段内。当日时段：{_fmt_time(open_t)}–{_fmt_time(close_t)}。下一个开放时段：{_fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}。\n{canonical_line()}"
            return f"Closed at {dt.strftime('%H:%M')} on {date_str}. Day window: { _fmt_time(open_t)}–{_fmt_time(close_t)}. Next open window: { _fmt_date(nxt_day, L)} { _fmt_time(n_open)}–{_fmt_time(n_close)}.\n{canonical_line()}"

    # No specific time, provide that day's window
    if L == "zh-HK":
        return f"{date_str}開放時段：{_fmt_time(open_t)}–{_fmt_time(close_t)}。\n{canonical_line()}"
    if L == "zh-CN":
        return f"{date_str}开放时段：{_fmt_time(open_t)}–{_fmt_time(close_t)}。\n{canonical_line()}"
    return f"{date_str} open window: { _fmt_time(open_t)}–{_fmt_time(close_t)}.\n{canonical_line()}"