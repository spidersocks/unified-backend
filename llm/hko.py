import os
import time
from typing import Optional, Tuple, Dict, Any, List
import requests

# HKO Open Data (docs):
# https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf
#
# Endpoints used:
# - Weather warnings in force (structured; includes codes):
#   https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warningInfo&lang=en|tc|sc
# - Special Weather Tips (textual fallback):
#   https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=swt&lang=en|tc|sc

_HKO_BASE_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
_TIMEOUT = float(os.environ.get("HKO_HTTP_TIMEOUT_SECS", "4.0"))
_CACHE_TTL = int(os.environ.get("HKO_CACHE_TTL_SECS", "300"))

# Simple in-memory cache keyed by (endpoint, lang)
_cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}

def _lang_code(lang: Optional[str]) -> str:
    L = (lang or "en").lower()
    if L.startswith("zh-hk"):
        return "tc"
    if L.startswith("zh-cn") or L == "zh":
        return "sc"
    return "en"

def _now() -> float:
    return time.time()

def _cached_get(params: Dict[str, str]) -> Optional[Any]:
    key = (params.get("dataType", ""), params.get("lang", "en"))
    ent = _cache.get(key)
    if ent and _now() - ent[0] <= _CACHE_TTL:
        return ent[1]
    try:
        resp = requests.get(_HKO_BASE_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        _cache[key] = (_now(), data)
        return data
    except Exception:
        return None

# -------- warningInfo (preferred) --------

# Treat these categories as "rain or strong wind related"
# We match by code/type/name fields (case-insensitive) to avoid brittle string lists.
_ALLOWED_WARNING_KEYWORDS = [
    # English
    "rain", "rainstorm", "thunder", "thunderstorm", "tropical cyclone", "typhoon", "monsoon", "strong wind", "gale",
    # Traditional Chinese
    "雨", "雷暴", "熱帶氣旋", "颱風", "季候風", "強風",
    # Simplified Chinese
    "雨", "雷暴", "热带气旋", "台风", "季候风", "强风",
]

def _contains_any(text: str, needles: List[str]) -> bool:
    low = (text or "").lower()
    return any(n.lower() in low for n in needles)

def _flatten_warning_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Accepts warningInfo JSON which may be an array or an object containing an array.
    Returns a list of dict warnings with at least code/name/type if available.
    """
    out: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for it in payload:
            if isinstance(it, dict):
                out.append(it)
    elif isinstance(payload, dict):
        # Try common container keys
        for k in ("warningInfo", "data", "details", "records", "warnings"):
            v = payload.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        out.append(it)
        # If none of the container keys matched, some feeds return dicts already shaped
        if not out:
            # Heuristic: treat top-level dict as a single warning if it looks like one
            if any(x in payload for x in ("code", "name", "type")):
                out.append(payload)  # type: ignore
    return out

def _pick_relevant_warning(warnings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Pick the first warning that is rain/wind related.
    We build a haystack string from name/type/code for robust matching across languages.
    """
    for w in warnings:
        name = str(w.get("name") or w.get("warningName") or "")
        typ  = str(w.get("type") or w.get("warningType") or "")
        code = str(w.get("code") or w.get("warningCode") or w.get("subtype") or "")
        hay = " ".join([name, typ, code]).strip()
        if _contains_any(hay, _ALLOWED_WARNING_KEYWORDS):
            return w
    return None

def _format_warning_line(w: Dict[str, Any], lc: str) -> str:
    # Prefer localized 'name', fall back to code if missing
    name = str(w.get("name") or w.get("warningName") or w.get("type") or w.get("warningType") or "").strip()
    code = str(w.get("code") or w.get("warningCode") or w.get("subtype") or "").strip()
    label = name if name else code if code else ""
    if lc == "tc":
        prefix = "天氣提示："
    elif lc == "sc":
        prefix = "天气提示："
    else:
        prefix = "Weather tip: "
    return f"{prefix}{label}".strip()

def _get_warning_hint_from_warninginfo(lang: Optional[str]) -> Optional[str]:
    lc = _lang_code(lang)
    data = _cached_get({"dataType": "warningInfo", "lang": lc})
    if not data:
        return None
    warnings = _flatten_warning_items(data)
    if not warnings:
        return None
    rel = _pick_relevant_warning(warnings)
    if not rel:
        return None
    return _format_warning_line(rel, lc)

# -------- SWT (fallback) --------

def _flatten_swt_items(payload: Any) -> List[str]:
    # HKO returns either an array or { swt: [ ... ] }
    if payload is None:
        return []
    items = None
    if isinstance(payload, dict):
        items = payload.get("swt")
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    texts: List[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                for k in ("desc", "details", "content", "title", "summary"):
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())
            elif isinstance(it, str) and it.strip():
                texts.append(it.strip())
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _pick_relevant_swt_text(chunks: List[str]) -> Optional[str]:
    for t in chunks:
        if _contains_any(t, _ALLOWED_WARNING_KEYWORDS):
            return t
    return None

def _get_weather_hint_from_swt(lang: Optional[str]) -> Optional[str]:
    lc = _lang_code(lang)
    data = _cached_get({"dataType": "swt", "lang": lc})
    if not data:
        return None
    chunks = _flatten_swt_items(data)
    if not chunks:
        return None
    chosen = _pick_relevant_swt_text(chunks)
    if not chosen:
        return None
    body = chosen.strip().replace("\n", " ").strip()
    if len(body) > 180:
        body = body[:177] + "…"
    if lc == "tc":
        return f"天氣提示：{body}"
    if lc == "sc":
        return f"天气提示：{body}"
    return f"Weather tip: {body}"

# -------- Public API --------

def get_weather_hint_for_opening(lang: Optional[str]) -> Optional[str]:
    """
    Returns a short, localized weather hint line for opening-hours answers,
    using HKO warning codes/categories (rain/thunderstorm/cyclone/monsoon/strong wind).
    Falls back to Special Weather Tips if no structured warnings are present.
    """
    hint = _get_warning_hint_from_warninginfo(lang)
    if hint:
        return hint
    return _get_weather_hint_from_swt(lang)