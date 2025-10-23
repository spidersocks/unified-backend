import os
import re
import time
from functools import lru_cache
from typing import Optional, Tuple

# Optional libs (used if available)
try:
    # Fast local LID. If unavailable, we fall back to heuristics.
    import pycld3  # type: ignore
except Exception:
    pycld3 = None

try:
    # Script conversion helpers. Use the pure-Python reimplementation ('opencc' package).
    from opencc import OpenCC  # type: ignore
    _T2S = OpenCC("t2s")
    _S2T = OpenCC("s2t")
except Exception:
    _T2S = _S2T = None

import boto3

# Simple discriminator sets for Traditional vs Simplified when we only know "zh"
TRAD_ONLY = set("學體車國廣馬門風愛聽話醫龍書氣媽齡費號聯網臺灣灣課師資簡介聯絡資料")
SIMP_ONLY = set("学体车国广马门风爱听话医龙书气妈龄费号联网台湾湾课师资简介联络资料")

# Minimal in-memory session stickiness
_SESSION_LANG: dict[str, Tuple[str, float]] = {}
_SESSION_TTL_SECS = int(os.environ.get("LANG_SESSION_TTL_SECS", "3600"))

def _now() -> float:
    return time.time()

def _contains_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s or "")

def _normalize_lang_tag(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    t = tag.strip().lower()
    if t.startswith("en"):
        return "en"
    # Traditional variants
    if t in ("zh-hk", "zh_hk", "zh-tw", "zh_tw", "zh-mo", "zh-hant", "zh_hant", "zh-hant-hk"):
        return "zh-HK"
    # Simplified variants (default for generic zh)
    if t in ("zh-cn", "zh_cn", "zh-sg", "zh_hans", "zh-hans", "zh-hans-cn", "zh-hans-sg", "zh"):
        return "zh-CN"
    return None

def _parse_accept_language(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    best = None
    best_q = -1.0
    for part in header.split(","):
        m = re.match(r"^\s*([^;]+)(?:;q=([0-9.]+))?\s*$", part)
        if not m:
            continue
        tag = m.group(1).strip()
        q = float(m.group(2)) if m.group(2) else 1.0
        mapped = _normalize_lang_tag(tag)
        if mapped and q > best_q:
            best, best_q = mapped, q
    return best

def _score_script(text: str) -> Tuple[int, int]:
    trad = sum(1 for ch in text if ch in TRAD_ONLY)
    simp = sum(1 for ch in text if ch in SIMP_ONLY)
    return trad, simp

def _opencc_similarity(a: str, b: str) -> float:
    # Simple char-level similarity ratio in [0,1]
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))

def _score_opencc(text: str) -> Tuple[float, float]:
    if not _T2S or not _S2T or not text:
        return 0.0, 0.0
    try:
        s2t = _S2T.convert(text)
        t2s = _T2S.convert(text)
        # If text is Traditional, s2t(text) ~ text; if Simplified, t2s(text) ~ text
        return _opencc_similarity(s2t, text), _opencc_similarity(t2s, text)
    except Exception:
        return 0.0, 0.0

def _decide_zh_variant(text: str) -> str:
    trad_cnt, simp_cnt = _score_script(text)
    trad_sim, simp_sim = _score_opencc(text)
    # Weighted vote
    trad_score = trad_cnt + 2.0 * trad_sim
    simp_score = simp_cnt + 2.0 * simp_sim
    if trad_score > simp_score:
        return "zh-HK"
    if simp_score > trad_score:
        return "zh-CN"
    # Tie-breaker: default to HK for CJK (safer for your audience)
    return "zh-HK"

def _detect_with_comprehend(text: str) -> Optional[str]:
    if os.environ.get("USE_COMPREHEND_LID", "").lower() not in ("1", "true", "yes"):
        return None
    if not text:
        return None
    try:
        client = boto3.client("comprehend", region_name=os.environ.get("AWS_REGION"))
        resp = client.detect_dominant_language(Text=text[:4500])
        langs = sorted(resp.get("Languages", []), key=lambda x: x.get("Score", 0), reverse=True)
        if not langs:
            return None
        code = (langs[0].get("LanguageCode") or "").lower()
        if code == "en":
            return "en"
        if code in ("zh-tw", "zh_tw"):
            return "zh-HK"
        if code.startswith("zh"):
            return _decide_zh_variant(text)
        return None
    except Exception:
        return None

def _detect_with_cld3(text: str) -> Optional[str]:
    if not pycld3:
        return None
    try:
        res = pycld3.get_language(text or "")
        if not res or not res.is_reliable:
            return None
        lang = (res.language or "").lower()
        if lang == "en":
            return "en"
        if lang == "zh":
            return _decide_zh_variant(text)
        return None
    except Exception:
        return None

@lru_cache(maxsize=2048)
def detect_language(message: str, accept_language: Optional[str] = None) -> str:
    """
    Returns one of: 'en' | 'zh-HK' | 'zh-CN'
    New priority:
      1) Strong content signal: if message contains CJK, pick zh-HK vs zh-CN from the text.
      2) AWS Comprehend (optional) or CLD3 if installed.
      3) Accept-Language header as a hint (used only when content is ambiguous).
      4) Default to English.
    """
    msg = message or ""
    # 1) Text content takes precedence
    if _contains_cjk(msg):
        return _decide_zh_variant(msg)

    # 2) ML detectors if available
    v = _detect_with_comprehend(msg)
    if v in ("en", "zh-HK", "zh-CN"):
        return v
    v = _detect_with_cld3(msg)
    if v in ("en", "zh-HK", "zh-CN"):
        return v

    # 3) Only now consider Accept-Language
    hinted = _parse_accept_language(accept_language)
    if hinted in ("en", "zh-HK", "zh-CN"):
        return hinted

    # 4) Default
    return "en"

def remember_session_language(session_id: Optional[str], lang: str):
    if not session_id or not lang:
        return
    _SESSION_LANG[session_id] = (lang, _now())

def get_session_language(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    entry = _SESSION_LANG.get(session_id)
    if not entry:
        return None
    lang, ts = entry
    if _now() - ts > _SESSION_TTL_SECS:
        _SESSION_LANG.pop(session_id, None)
        return None
    return lang