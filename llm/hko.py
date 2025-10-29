import os
import time
from typing import Optional, Tuple, Dict, Any, List
import requests

# HKO Open Data (docs):
# https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf
#
# Endpoints used:
# - Weather Warning Information (structured; includes codes + details):
#   https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warningInfo&lang=en|tc|sc
# - Weather Warning Summary (structured; compact, per-warning object):
#   https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en|tc|sc
# - Special Weather Tips (textual fallback):
#   https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=swt&lang=en|tc|sc
#
# Please include valid parameters in API requests. For details, refer to the
# Hong Kong Observatory Open Data API Documentation.

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
        resp = requests.get(
            _HKO_BASE_URL,
            params=params,
            timeout=_TIMEOUT,
            headers={"Accept": "application/json", "User-Agent": "decoders-hko/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        _cache[key] = (_now(), data)
        return data
    except Exception:
        return None

# Severe only (for opening-hours messages):
# - Black Rain (黑雨)
# - Typhoon Signal No. 8 / 9 / 10 (八號/九號/十號風球), including "Gale or Storm Signal"
# - "Pre-No. 8" special announcement (WTCPRE8)
_SEVERE_KEYWORDS = [
    # English
    "black rain", "black rainstorm", "black rainstorm warning", "black rainstorm signal",
    "typhoon signal no. 8", "t8", "no. 8", "no.8", "signal no. 8", "gale or storm signal",
    "typhoon signal no. 9", "t9", "no. 9", "no.9", "signal no. 9", "increasing gale or storm",
    "typhoon signal no. 10", "t10", "no. 10", "no.10", "signal no. 10", "hurricane signal", "hurricane force",
    "pre-no. 8 special announcement", "pre-8 announcement",

    # Traditional Chinese
    "黑雨", "黑色暴雨", "黑色暴雨警告", "黑色暴雨警告信號",
    "八號", "八號風球", "八號波", "八號烈風或暴風信號", "八號熱帶氣旋警告",
    "九號", "九號風球", "九號波", "九號烈風或暴風風力增強信號", "九號熱帶氣旋警告",
    "十號", "十號風球", "十號波", "十號颶風信號", "十號熱帶氣旋警告",
    "預警八號", "八號預警", "預先發出之八號熱帶氣旋警告信號",

    # Simplified Chinese
    "黑雨", "黑色暴雨", "黑色暴雨警告", "黑色暴雨警告信号",
    "八号", "八号风球", "八号波", "八号烈风或暴风信号", "八号热带气旋警告",
    "九号", "九号风球", "九号波", "九号烈风或暴风风力增强信号", "九号热带气旋警告",
    "十号", "十号风球", "十号波", "十号飓风信号", "十号热带气旋警告",
    "预警八号", "八号预警", "预先发出之八号热带气旋警告信号",
]

# Explicit code-based ranking (strongest first)
# Based on HKO docs: WTCSGNL subtype TC10/TC9/TC8{NE,SE,SW,NW}; WTCPRE8; WRAIN subtype WRAINB.
_TY_CODES_10 = {"TC10"}
_TY_CODES_9 = {"TC9"}
_TY_CODES_8 = {"TC8NE", "TC8SE", "TC8SW", "TC8NW"}
_PRE8_CODES = {"WTCPRE8"}
_BLACK_RAIN_CODES = {"WRAINB"}

def _contains_any(text: str, needles: List[str]) -> bool:
    low = (text or "").lower()
    return any(n.lower() in low for n in needles)

def _flatten_warning_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Accepts the response of dataType=warningInfo (usually {"details":[...]}),
    or warnsum (object with keys per warning), or already a list.
    Returns a flat list of dict records.
    """
    out: List[Dict[str, Any]] = []
    if payload is None:
        return out

    if isinstance(payload, list):
        for it in payload:
            if isinstance(it, dict):
                out.append(it)
        return out

    if isinstance(payload, dict):
        # warningInfo: "details": [ {...}, ... ]
        for k in ("warningInfo", "details", "data", "records", "warnings"):
            v = payload.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        out.append(it)

        # warnsum: object with warning-code keys, each containing an object
        # e.g. {"WTS": {...}, "WTCSGNL": {...}}
        if not out:
            for k, v in payload.items():
                if isinstance(v, dict) and any(x in v for x in ("code", "name", "type", "actionCode")):
                    rec = dict(v)
                    rec["_key"] = k
                    out.append(rec)

        # single object fallback
        if not out and any(x in payload for x in ("code", "name", "type", "warningStatementCode")):
            out.append(payload)  # type: ignore

    return out

def _normalize_warning_record(w: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a unified view for ranking/formatting.
    Fields:
      code: e.g. WRAINR / TC3 (may be absent)
      subtype: WRAINB / TC8NE / etc.
      wscode: warningStatementCode (for warningInfo details)
      name/type: displayable name or type
      contents_text: joined contents if present
    """
    code = str(w.get("code") or w.get("warningCode") or "").strip()
    subtype = str(w.get("subtype") or "").strip()
    wscode = str(w.get("warningStatementCode") or "").strip()
    name = str(w.get("name") or w.get("warningName") or "").strip()
    wtype = str(w.get("type") or w.get("warningType") or "").strip()
    contents = ""
    c = w.get("contents")
    if isinstance(c, list):
        contents = " ".join([str(x) for x in c if isinstance(x, str)]).strip()
    elif isinstance(c, str):
        contents = c.strip()

    # Best display label
    label = name or wtype or subtype or code or wscode

    return {
        "code": code,
        "subtype": subtype,
        "wscode": wscode,
        "name": name,
        "type": wtype,
        "contents_text": contents,
        "label": label,
        "_raw": w,
    }

def _severity_rank(nw: Dict[str, Any]) -> int:
    code = (nw.get("code") or "").upper()
    sub = (nw.get("subtype") or "").upper()
    wscode = (nw.get("wscode") or "").upper()
    hay = " ".join([
        nw.get("name") or "",
        nw.get("type") or "",
        code, sub, wscode,
        nw.get("contents_text") or "",
    ])

    # Code/subtype driven (highest confidence)
    if sub in _TY_CODES_10 or code in _TY_CODES_10:
        return 100
    if sub in _TY_CODES_9 or code in _TY_CODES_9:
        return 90
    if sub in _TY_CODES_8 or code in _TY_CODES_8:
        return 80
    if wscode in _PRE8_CODES or code in _PRE8_CODES or sub in _PRE8_CODES:
        return 70
    if sub in _BLACK_RAIN_CODES or code in _BLACK_RAIN_CODES:
        return 60

    # Textual keywords as fallback (e.g., WTCPRE8 may appear only in contents)
    if _contains_any(hay, _SEVERE_KEYWORDS):
        # Assign a conservative severity if detected only via text
        # Try to differentiate T8+/T9/T10 if text hints present
        low = hay.lower()
        if "no. 10" in low or "t10" in low or "颶風信號" in low or "飓风信号" in low:
            return 100
        if "no. 9" in low or "t9" in low or "增強信號" in low:
            return 90
        if "no. 8" in low or "t8" in low or "烈風或暴風信號" in low:
            return 80
        if "pre-no. 8" in low or "預警八號" in low or "预警八号" in low:
            return 70
        if "black rain" in low or "黑雨" in low:
            return 60
        return 50

    return 0

def _pick_severe_warning(warnings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_rank = 0
    for w in warnings:
        nw = _normalize_warning_record(w)
        rank = _severity_rank(nw)
        if rank > best_rank:
            best_rank = rank
            best = nw
    return best

def _format_warning_line(nw: Dict[str, Any], lc: str) -> str:
    label = (nw.get("label") or "").strip()
    # Refine label for common severe cases if possible
    code = (nw.get("code") or "").upper()
    sub = (nw.get("subtype") or "").upper()
    wscode = (nw.get("wscode") or "").upper()
    typ = (nw.get("type") or "").strip()

    if any(x in {code, sub, wscode} for x in _TY_CODES_10):
        label = typ or "Tropical Cyclone Warning Signal No. 10"
    elif any(x in {code, sub, wscode} for x in _TY_CODES_9):
        label = typ or "Tropical Cyclone Warning Signal No. 9"
    elif any(x in {code, sub, wscode} for x in _TY_CODES_8):
        label = typ or "Tropical Cyclone Warning Signal No. 8"
    elif any(x in {code, sub, wscode} for x in _BLACK_RAIN_CODES):
        # Prefer explicit "Black Rainstorm Warning Signal"
        if lc == "tc":
            label = "黑色暴雨警告信號"
        elif lc == "sc":
            label = "黑色暴雨警告信号"
        else:
            label = "Black Rainstorm Warning Signal"
    elif any(x in {code, sub, wscode} for x in _PRE8_CODES):
        if lc == "tc":
            label = "預先發出之八號熱帶氣旋警告信號"
        elif lc == "sc":
            label = "预先发出之八号热带气旋警告信号"
        else:
            label = "Pre-No. 8 Special Announcement"

    if lc == "tc":
        prefix = "天氣提示："
    elif lc == "sc":
        prefix = "天气提示："
    else:
        prefix = "Weather tip: "
    return f"{prefix}{label}".strip()

def _get_severe_from_warninginfo(lang: Optional[str]) -> Optional[str]:
    lc = _lang_code(lang)
    data = _cached_get({"dataType": "warningInfo", "lang": lc})
    if not data:
        return None
    warnings = _flatten_warning_items(data)
    if not warnings:
        return None
    rel = _pick_severe_warning(warnings)
    if not rel:
        return None
    return _format_warning_line(rel, lc)

def _get_severe_from_warnsum(lang: Optional[str]) -> Optional[str]:
    lc = _lang_code(lang)
    data = _cached_get({"dataType": "warnsum", "lang": lc})
    if not data:
        return None
    warnings = _flatten_warning_items(data)
    if not warnings:
        return None
    rel = _pick_severe_warning(warnings)
    if not rel:
        return None
    return _format_warning_line(rel, lc)

def _flatten_swt_items(payload: Any) -> List[str]:
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
    # Deduplicate
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _get_severe_from_swt(lang: Optional[str]) -> Optional[str]:
    lc = _lang_code(lang)
    data = _cached_get({"dataType": "swt", "lang": lc})
    if not data:
        return None
    chunks = _flatten_swt_items(data)
    if not chunks:
        return None
    for t in chunks:
        if _contains_any(t, _SEVERE_KEYWORDS):
            body = t.strip().replace("\n", " ").strip()
            if len(body) > 180:
                body = body[:177] + "…"
            if lc == "tc":
                return f"天氣提示：{body}"
            if lc == "sc":
                return f"天气提示：{body}"
            return f"Weather tip: {body}"
    return None

def get_weather_hint_for_opening(lang: Optional[str]) -> Optional[str]:
    """
    Return a short hint ONLY when a severe condition is in effect or explicitly noted:
      - Black Rainstorm Warning (WRAINB), or
      - Tropical Cyclone Signal No. 8/9/10 (TC8*/TC9/TC10), or
      - Pre-No. 8 Special Announcement (WTCPRE8).
    Prefer structured warnings (warningInfo or warnsum);
    fall back to SWT text if it explicitly mentions severe signals.
    """
    # 1) Prefer structured 'warningInfo'
    hint = _get_severe_from_warninginfo(lang)
    if hint:
        return hint
    # 2) Try compact 'warnsum'
    hint = _get_severe_from_warnsum(lang)
    if hint:
        return hint
    # 3) Fallback to textual SWT
    return _get_severe_from_swt(lang)