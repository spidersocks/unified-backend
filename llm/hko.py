import os
import time
from typing import Optional, Tuple, Dict, Any, List
import requests

# HKO Open Data: Special Weather Tips (SWT)
# Docs: https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf
# Endpoint example:
#   https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=swt&lang=en|tc|sc

_HKO_SWT_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
_TIMEOUT = float(os.environ.get("HKO_HTTP_TIMEOUT_SECS", "4.0"))
_CACHE_TTL = int(os.environ.get("HKO_CACHE_TTL_SECS", "300"))
_STRICT_SEVERE_ONLY = os.environ.get("HKO_STRICT_SEVERE_ONLY", "true").lower() in ("1","true","yes")

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def _lang_code(lang: Optional[str]) -> str:
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return "tc"
    if L.startswith("zh-cn") or L == "zh":
        return "sc"
    return "en"

def _now() -> float:
    return time.time()

def _fetch_swt(lang: str) -> Optional[Dict[str, Any]]:
    # Cache by lang
    global _cache
    ent = _cache.get(lang)
    if ent and _now() - ent[0] <= _CACHE_TTL:
        return ent[1]

    try:
        resp = requests.get(
            _HKO_SWT_URL,
            params={"dataType": "swt", "lang": lang},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        _cache[lang] = (_now(), data)
        return data
    except Exception:
        # Don’t fail the whole tool if HKO is unreachable
        return None

def _flatten_items(payload: Dict[str, Any]) -> List[str]:
    # HKO returns either an array or { swt: [ ... ] }
    if payload is None:
        return []
    items = payload.get("swt")
    if isinstance(items, list):
        arr = items
    elif isinstance(payload, list):
        arr = payload
    else:
        arr = []
    texts: List[str] = []
    for it in arr:
        if isinstance(it, dict):
            # common fields: desc, details, content, title
            for k in ("desc", "details", "content", "title", "summary"):
                v = it.get(k)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
        elif isinstance(it, str):
            s = it.strip()
            if s:
                texts.append(s)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

# Relevant categories and severe markers
_SEVERE_MARKERS = [
    # EN
    "black rain", "black rainstorm", "typhoon signal no. 8", "no. 8 gale", "signal no. 8",
    # TC (zh-HK)
    "黑雨", "黑色暴雨", "八號", "八號風球",
    # SC (zh-CN)
    "黑雨", "黑色暴雨", "八号", "八号风球",
]

_CATEGORY_MARKERS = [
    # EN
    "thunderstorm", "rainstorm", "tropical cyclone", "strong wind",
    # TC
    "雷暴", "暴雨", "熱帶氣旋", "強風", "強烈季候風", "烈風",
    # SC
    "雷暴", "暴雨", "热带气旋", "强风", "强烈季候风", "烈风",
    # Common rain colors
    "amber", "red rain", "black rain", "紅雨", "紅色暴雨", "红雨", "红色暴雨", "黃色", "黄色",
]

def _contains_any(text: str, needles: List[str]) -> bool:
    low = (text or "").lower()
    return any(n in low for n in needles)

def get_weather_hint_for_opening(lang: Optional[str]) -> Optional[str]:
    """
    Returns a short, localized weather hint line for opening-hours answers,
    or None if there is nothing relevant to mention.
    Policy:
      - If STRICT_SEVERE_ONLY=true (default): only mention if Black Rain or Signal No. 8 is referenced.
      - Else: mention if SWT includes thunderstorm, rainstorm, strong wind, tropical cyclone.
    """
    lc = _lang_code(lang)
    data = _fetch_swt(lc)
    if not data:
        return None
    chunks = _flatten_items(data)
    if not chunks:
        return None

    # Choose the first relevant chunk
    chosen = None
    for t in chunks:
        if _STRICT_SEVERE_ONLY:
            if _contains_any(t, _SEVERE_MARKERS):
                chosen = t
                break
        else:
            if _contains_any(t, _SEVERE_MARKERS) or _contains_any(t, _CATEGORY_MARKERS):
                chosen = t
                break

    if not chosen:
        return None

    # Localize a simple prefix and trim the body
    body = chosen.strip().replace("\n", " ").strip()
    if len(body) > 180:
        body = body[:177] + "…"

    if lc == "tc":
        return f"天氣提示：{body}"
    if lc == "sc":
        return f"天气提示：{body}"
    return f"Weather tip: {body}"