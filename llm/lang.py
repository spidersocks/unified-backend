"""
Centralized language detection pipeline for /chat endpoint.

Supports English (en), Traditional Chinese (zh-HK), and Simplified Chinese (zh-CN).

Detection strategy (in order):
1. Accept-Language header hint mapping
2. AWS Comprehend (if USE_COMPREHEND_LID=true)
3. pycld3 library (if available)
4. Lightweight heuristic fallback

Session stickiness: remembers language by session_id with configurable TTL.
"""

import os
import time
from typing import Optional, Dict, Tuple
from datetime import datetime

# Optional imports - gracefully handle if not installed
try:
    import gcld3
    PYCLD3_AVAILABLE = True
except ImportError:
    PYCLD3_AVAILABLE = False

try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Environment configuration
USE_COMPREHEND_LID = os.environ.get("USE_COMPREHEND_LID", "false").lower() in ("1", "true", "yes")
LANG_SESSION_TTL_SECS = int(os.environ.get("LANG_SESSION_TTL_SECS", "3600"))

# Session memory: {session_id: (language, timestamp)}
_session_memory: Dict[str, Tuple[str, float]] = {}

# Discriminator character sets for Traditional vs Simplified Chinese
TRAD_ONLY = set("學體車國廣馬門風愛聽話醫龍書氣媽齡費號聯網臺灣灣課師資簡介聯絡資料")
SIMP_ONLY = set("学体车国广马门风爱听话医龙书气妈龄费号联网台湾湾课师资简介联络资料")

def _contains_cjk(s: str) -> bool:
    """Check if string contains CJK characters."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

def _parse_accept_language(accept_lang: Optional[str]) -> Optional[str]:
    """
    Parse Accept-Language header and map to our supported languages.
    Returns 'en', 'zh-HK', 'zh-CN', or None if not specific enough.
    
    Examples:
      "en-US" -> "en"
      "zh-HK" -> "zh-HK"
      "zh-CN" -> "zh-CN"
      "zh-TW" -> "zh-HK"
      "zh" -> None (not specific enough)
    """
    if not accept_lang:
        return None
    
    # Parse first language tag (ignore quality values for simplicity)
    lang = accept_lang.split(',')[0].strip().split(';')[0].strip().lower()
    
    if lang.startswith('en'):
        return 'en'
    if lang in ('zh-hk', 'zh-tw', 'zh-mo'):
        return 'zh-HK'
    if lang in ('zh-cn', 'zh-sg'):
        return 'zh-CN'
    
    return None

def _resolve_zh_variant(text: str) -> str:
    """
    Resolve Chinese variant using script analysis.
    
    Strategy:
    1. Count discriminator characters (Traditional vs Simplified)
    2. If available, use OpenCC similarity analysis
    3. Default tie-break to zh-HK for CJK content
    
    Returns 'zh-HK' or 'zh-CN'.
    """
    trad_count = sum(1 for ch in text if ch in TRAD_ONLY)
    simp_count = sum(1 for ch in text if ch in SIMP_ONLY)
    
    # Clear winner from discriminator characters
    if trad_count > simp_count:
        return 'zh-HK'
    if simp_count > trad_count:
        return 'zh-CN'
    
    # If OpenCC is available, use similarity analysis
    if OPENCC_AVAILABLE and len(text.strip()) > 0:
        try:
            # Convert to Traditional and Simplified to see which is closer to original
            cc_t2s = opencc.OpenCC('t2s')  # Traditional to Simplified
            cc_s2t = opencc.OpenCC('s2t')  # Simplified to Traditional
            
            simplified = cc_t2s.convert(text)
            traditionalized = cc_s2t.convert(text)
            
            # Count how many characters changed
            # If text is already Traditional, t2s will change many chars
            # If text is already Simplified, s2t will change many chars
            changes_when_simplifying = sum(1 for a, b in zip(text, simplified) if a != b)
            changes_when_traditionalizing = sum(1 for a, b in zip(text, traditionalized) if a != b)
            
            if changes_when_simplifying > changes_when_traditionalizing:
                return 'zh-HK'  # Text is already more Traditional
            elif changes_when_traditionalizing > changes_when_simplifying:
                return 'zh-CN'  # Text is already more Simplified
        except Exception:
            pass  # Fall through to default
    
    # Default to zh-HK for CJK content (safer for HK audience)
    return 'zh-HK'

def _detect_with_comprehend(text: str) -> Optional[str]:
    """
    Use AWS Comprehend to detect language.
    Returns 'en', 'zh-HK', 'zh-CN', or None if detection fails.
    """
    if not BOTO3_AVAILABLE or not USE_COMPREHEND_LID:
        return None
    
    try:
        client = boto3.client('comprehend')
        response = client.detect_dominant_language(Text=text[:5000])  # API limit
        
        if not response.get('Languages'):
            return None
        
        # Get the dominant language
        dominant = response['Languages'][0]
        lang_code = dominant.get('LanguageCode', '').lower()
        
        if lang_code.startswith('en'):
            return 'en'
        if lang_code in ('zh', 'zh-tw'):
            # Need to resolve variant
            return _resolve_zh_variant(text)
        if lang_code == 'zh-cn':
            return 'zh-CN'
        
        # If we got a language but it's not one we support, fall through
        return None
        
    except Exception:
        return None

def _detect_with_pycld3(text: str) -> Optional[str]:
    """
    Use pycld3 to detect language.
    Returns 'en', 'zh-HK', 'zh-CN', or None if detection fails.
    """
    if not PYCLD3_AVAILABLE:
        return None
    
    try:
        detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        result = detector.FindLanguage(text=text)
        
        if not result or not result.is_reliable:
            return None
        
        lang = result.language.lower()
        
        if lang.startswith('en'):
            return 'en'
        if lang in ('zh', 'zh-hant', 'zh-hans'):
            # Need to resolve variant
            return _resolve_zh_variant(text)
        
        return None
        
    except Exception:
        return None

def _heuristic_detect(text: str) -> str:
    """
    Lightweight heuristic fallback for language detection.
    
    Returns 'en', 'zh-HK', or 'zh-CN'.
    """
    if not text or not _contains_cjk(text):
        return 'en'
    
    return _resolve_zh_variant(text)

def detect_language(message: str, accept_language: Optional[str] = None) -> str:
    """
    Detect language for a message using multi-tier detection strategy.
    
    Args:
        message: The text message to analyze
        accept_language: Optional Accept-Language header value
    
    Returns:
        One of: 'en', 'zh-HK', 'zh-CN'
    
    Detection order:
        1. Accept-Language header hint (if specific)
        2. AWS Comprehend (if USE_COMPREHEND_LID=true)
        3. pycld3 library (if available)
        4. Heuristic fallback (always succeeds)
    """
    # Try Accept-Language hint first
    lang_hint = _parse_accept_language(accept_language)
    if lang_hint:
        return lang_hint
    
    # Try AWS Comprehend
    if USE_COMPREHEND_LID:
        lang = _detect_with_comprehend(message)
        if lang:
            return lang
    
    # Try pycld3
    if PYCLD3_AVAILABLE:
        lang = _detect_with_pycld3(message)
        if lang:
            return lang
    
    # Fallback to heuristic
    return _heuristic_detect(message)

def remember_session_language(session_id: Optional[str], lang: str) -> None:
    """
    Remember the detected/used language for a session.
    
    Args:
        session_id: Session identifier (if None, does nothing)
        lang: Language code to remember ('en', 'zh-HK', 'zh-CN')
    """
    if not session_id:
        return
    
    _session_memory[session_id] = (lang, time.time())

def get_session_language(session_id: Optional[str]) -> Optional[str]:
    """
    Retrieve remembered language for a session, if still valid (within TTL).
    
    Args:
        session_id: Session identifier (if None, returns None)
    
    Returns:
        Language code if found and valid, None otherwise
    """
    if not session_id:
        return None
    
    if session_id not in _session_memory:
        return None
    
    lang, timestamp = _session_memory[session_id]
    
    # Check if TTL expired
    if time.time() - timestamp > LANG_SESSION_TTL_SECS:
        del _session_memory[session_id]
        return None
    
    return lang

def _cleanup_expired_sessions() -> None:
    """
    Internal cleanup function to remove expired session entries.
    Called periodically to prevent memory leak.
    """
    current_time = time.time()
    expired = [
        sid for sid, (_, timestamp) in _session_memory.items()
        if current_time - timestamp > LANG_SESSION_TTL_SECS
    ]
    for sid in expired:
        del _session_memory[sid]
