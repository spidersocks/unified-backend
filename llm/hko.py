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

# We ONLY surface severe conditions for opening-hours answers:
# Black Rain or Typhoon Signal No. 8 (or above).
_SEVERE_KEYWORDS = [
    # === English Keywords ===

    # --- Black Rainstorm and Its Direct Precursor ---
    "black rain", "black rainstorm", "black rainstorm warning", "black rainstorm signal",
    "red rain", "red rainstorm", "red rainstorm warning", "red rainstorm signal", # Immediate precursor to Black Rain

    # --- Typhoon Signals (T8 and above) ---
    "typhoon signal no. 8", "t8", "no. 8", "no.8", "signal no. 8",
    "gale or storm signal", # Official name for T8
    "typhoon signal no. 9", "t9", "no. 9", "no.9", "signal no. 9",
    "increasing gale or storm", # Part of the T9 warning
    "typhoon signal no. 10", "t10", "no. 10", "no.10", "signal no. 10",
    "hurricane signal", "hurricane force", # Official name/description for T10

    # --- Direct Precursors & Announcements for T8 ---
    "typhoon signal no. 3", "t3", "no. 3", "no.3", "signal no. 3", # The primary precursor to T8
    "strong wind signal", # Official name for T3
    "pre-no. 8 special announcement", "pre-8 announcement", # Definitive announcement that T8 is imminent

    # --- High-Threat Descriptive Terms ---
    "super typhoon", "severe typhoon", # Typhoon classifications suggesting a high signal is likely
    "landfall", "direct hit", # Actions suggesting a high signal is likely

    # --- Action Verbs (used with the signals above) ---
    "hoist", "issue", "raise", "consider hoisting", # Indicates a signal is or might become active
    "lower", "cancel", # Indicates a signal is ending


    # === Traditional Chinese (Cantonese / Hong Kong) Keywords ===

    # --- 暴雨警告 (Rainstorm Warnings) ---
    "黑雨", "黑色暴雨", "黑色暴雨警告", "黑色暴雨警告信號",
    "紅雨", "紅色暴雨", "紅色暴雨警告", "紅色暴雨警告信號", # 黑雨的直接前兆 (Direct precursor to Black Rain)

    # --- 熱帶氣旋警告 (Typhoon Signals, T8 and above) ---
    "八號", "八號風球", "八號波", # "波" (bo1) is a common colloquialism
    "八號烈風或暴風信號", "八號熱帶氣旋警告",
    "九號", "九號風球", "九號波",
    "九號烈風或暴風風力增強信號", "九號熱帶氣旋警告",
    "十號", "十號風球", "十號波",
    "十號颶風信號", "十號熱帶氣旋警告",

    # --- T8的直接前兆及預警 (Direct Precursors & Announcements for T8) ---
    "三號", "三號風球", "三號波", # 八號的主要前兆 (The primary precursor to T8)
    "三號強風信號",
    "預警八號", "八號預警", "預先發出之八號熱帶氣旋警告信號", # HKO's definitive pre-8 announcement

    # --- 高威脅描述性術語 (High-Threat Descriptive Terms) ---
    "超強颱風", "強颱風", # 颱風級別，預示可能懸掛高級別信號
    "登陸", "直吹", "正面吹襲", # 颱風路徑，預示可能懸掛高級別信號

    # --- 常用動詞 (Common Verbs) ---
    "懸掛", "掛波", "掛風球", "考慮懸掛", # 表示信號可能或將會生效
    "發出", "改發", # "發出" (issue), "改發" (change to)
    "取消", "除下", # 表示信號結束


    # === Simplified Chinese (Mandarin) Keywords ===

    # --- 暴雨警告 (Rainstorm Warnings) ---
    "黑雨", "黑色暴雨", "黑色暴雨警告", "黑色暴雨警告信号",
    "红雨", "红色暴雨", "红色暴雨警告", "红色暴雨警告信号", # 黑雨的直接前兆 (Direct precursor to Black Rain)

    # --- 热带气旋警告 (Typhoon Signals, T8 and above) ---
    "八号", "八号风球", "八号波",
    "八号烈风或暴风信号", "八号热带气旋警告",
    "九号", "九号风球", "九号波",
    "九号烈风或暴风风力增强信号", "九号热带气旋警告",
    "十号", "十号风球", "十号波",
    "十号飓风信号", "十号热带气旋警告",

    # --- T8的直接前兆及预警 (Direct Precursors & Announcements for T8) ---
    "三号", "三号风球", "三号波", # 八号的主要前兆 (The primary precursor to T8)
    "三号强风信号",
    "预警八号", "八号预警", "预先发出之八号热带气旋警告信号", # HKO's definitive pre-8 announcement

    # --- 高威胁描述性术语 (High-Threat Descriptive Terms) ---
    "超强台风", "强台风", # 台风级别，预示可能悬挂高级别信号
    "登陆", "直吹", "正面吹袭", # 台风路径，预示可能悬挂高级别信号

    # --- 常用动词 (Common Verbs) ---
    "悬挂", "挂球", "考虑悬挂", # 表示信号可能或将会生效
    "发出", "改发", # "发出" (issue), "改发" (change to)
    "取消", "除下", # 表示信号结束
]

def _contains_any(text: str, needles: List[str]) -> bool:
    low = (text or "").lower()
    return any(n.lower() in low for n in needles)

def _flatten_warning_items(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for it in payload:
            if isinstance(it, dict):
                out.append(it)
    elif isinstance(payload, dict):
        for k in ("warningInfo", "data", "details", "records", "warnings"):
            v = payload.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        out.append(it)
        if not out and any(x in payload for x in ("code", "name", "type")):
            out.append(payload)  # type: ignore
    return out

def _pick_severe_warning(warnings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for w in warnings:
        name = str(w.get("name") or w.get("warningName") or "")
        typ  = str(w.get("type") or w.get("warningType") or "")
        code = str(w.get("code") or w.get("warningCode") or w.get("subtype") or "")
        hay = " ".join([name, typ, code]).strip()
        if _contains_any(hay, _SEVERE_KEYWORDS):
            return w
    return None

def _format_warning_line(w: Dict[str, Any], lc: str) -> str:
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
    Only return a hint when there is a severe condition in effect or explicitly noted (Black Rain or T8+).
    Never return generic tides/monsoon/thunderstorm info for opening-hours answers.
    """
    # Prefer structured warnings; fall back to SWT text if it explicitly mentions severe signals.
    hint = _get_severe_from_warninginfo(lang)
    if hint:
        return hint
    return _get_severe_from_swt(lang)