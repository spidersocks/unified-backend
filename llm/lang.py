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

# --- Robust CJK detection with expanded ranges
def _contains_cjk(s: str) -> bool:
    # Includes CJK Unified, Extension A, Symbols/Punctuation
    return any(
        ("\u4e00" <= ch <= "\u9fff") or
        ("\u3400" <= ch <= "\u4DBF") or
        ("\u20000" <= ch <= "\u2A6DF") or
        ("\u3000" <= ch <= "\u303F")
        for ch in s or ""
    )

# --- Known Chinese greetings and short markers
CHINESE_GREETINGS = {
    "你好", "您好", "嗨", "哈囉", "哈啰", "早安", "晚安", "早上好", "您好呀"
}

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
    Priority:
      1) If the message is a known Chinese greeting or contains any CJK, treat as Chinese. For CJK, pick zh-HK vs zh-CN from the script.
      2) If mixed CJK and Latin, prefer Chinese.
      3) ML detectors (Comprehend/CLD3) if available.
      4) Accept-Language header as a hint (only if content is ambiguous).
      5) Default to English.
    """
    msg = (message or "").strip()
    # 1) Exact Chinese greeting
    if msg in CHINESE_GREETINGS:
        return "zh-HK"
    # 2) Any CJK presence: treat as Chinese (even if mixed or short)
    if _contains_cjk(msg):
        return _decide_zh_variant(msg)
    # 3) ML detectors if available
    v = _detect_with_comprehend(msg)
    if v in ("en", "zh-HK", "zh-CN"):
        return v
    v = _detect_with_cld3(msg)
    if v in ("en", "zh-HK", "zh-CN"):
        return v
    # 4) Accept-Language header as a hint
    hinted = _parse_accept_language(accept_language)
    if hinted in ("en", "zh-HK", "zh-CN"):
        return hinted
    # 5) Default
    return "en"