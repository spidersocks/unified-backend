import os, json, pathlib, re, threading
from typing import Dict, Set, List, Tuple, Optional

# Build once at import time, thread-safe
_LOCK = threading.Lock()
_BUILT = False

# tokens_by_lang: lang -> set(tokens)
_TOKENS_BY_LANG: Dict[str, Set[str]] = {"en": set(), "zh-HK": set(), "zh-CN": set()}

# For future use: map token -> list of (lang, canonical) that mention it
_TOKEN_CANONICALS: Dict[str, List[Tuple[str, str]]] = {}

_CONTENT_ROOT = os.environ.get("KB_CONTENT_DIR", "content")

# Simple splitter for alias strings like "term1; term2; term-3"
_SPLIT_PAT = re.compile(r"[;,\|/]+|\s{2,}")

def _add_token(lang: str, token: str, canonical: Optional[str] = None):
    t = (token or "").strip()
    if not t:
        return
    t_lower = t.lower()
    # Ignore 1-char tokens (too noisy), except digits/letters combos length>=2
    if len(t_lower) < 2:
        return
    _TOKENS_BY_LANG.setdefault(lang, set()).add(t_lower)
    if canonical:
        _TOKEN_CANONICALS.setdefault(t_lower, []).append((lang, canonical))

def _extract_alias_tokens(alias_value: str) -> List[str]:
    """
    Accepts the 'aliases' stringValue from sidecar frontmatter; splits into discrete tokens/phrases.
    We keep multi-word phrases intact and also add simple wordpieces for EN.
    """
    if not alias_value:
        return []
    raw_parts = [p.strip() for p in _SPLIT_PAT.split(alias_value) if p.strip()]
    tokens: Set[str] = set()
    for p in raw_parts:
        tokens.add(p)
        # For English phrases, also add hyphen/space split pieces to help recall
        if re.search(r"[A-Za-z]", p):
            for w in re.split(r"[\s\-_/]+", p):
                w = w.strip()
                if len(w) >= 3:
                    tokens.add(w)
    return sorted(tokens)

def _load_sidecar(path: pathlib.Path) -> Optional[dict]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return json.loads(txt)
    except Exception:
        return None

def build_index(content_root: Optional[str] = None):
    """
    Build the in-process tags index from local sidecars.
    More robust root resolution:
      1) KB_CONTENT_DIR (if exists)
      2) repo-relative: <repo>/content (based on this file's location)
    """
    global _BUILT
    with _LOCK:
        if _BUILT:
            return
        # 1) KB_CONTENT_DIR or param
        root = pathlib.Path(content_root or _CONTENT_ROOT)
        # 2) Fallback to package-relative content/
        if not root.exists():
            pkg_root = pathlib.Path(__file__).resolve().parents[1] / "content"
            if pkg_root.exists():
                root = pkg_root
        if not root.exists():
            # Could not find content; keep empty index but don't crash
            _BUILT = True
            return

        for sc in root.glob("**/*.metadata.json"):
            data = _load_sidecar(sc)
            if not data:
                continue
            meta = data.get("metadataAttributes") or {}
            def sval(k: str) -> Optional[str]:
                v = meta.get(k) or {}
                vv = (v.get("value") or {}).get("stringValue")
                return str(vv) if vv is not None else None

            lang = (sval("language") or "en").strip()
            # Normalize language tags
            L = "zh-HK" if lang.lower().startswith("zh-hk") else ("zh-CN" if lang.lower().startswith("zh-cn") or lang.lower()=="zh" else "en")
            canonical = sval("canonical") or ""
            aliases = sval("aliases") or ""
            # Add canonical as a token too (helps when users ask by proper name)
            if canonical:
                _add_token(L, canonical, canonical)
            # Add alias tokens
            for tok in _extract_alias_tokens(aliases):
                _add_token(L, tok, canonical if canonical else None)
        _BUILT = True

def find_matching_tags(message: str, lang: str, limit: int = 12) -> List[str]:
    """
    Returns a de-duplicated subset of tokens present in the message (case-insensitive),
    prioritized by longer tokens first to capture multi-word phrases.
    """
    if not _BUILT:
        build_index()
    msg = (message or "").lower()
    L = "zh-HK" if lang.lower().startswith("zh-hk") else ("zh-CN" if lang.lower().startswith("zh-cn") or lang.lower()=="zh" else "en")

    candidates = list(_TOKENS_BY_LANG.get(L, set()))
    # Fallback to EN tag hints if nothing in target lang
    if not candidates and L != "en":
        candidates = list(_TOKENS_BY_LANG.get("en", set()))

    # Prefer longer phrases first; then stable sort by alpha
    candidates.sort(key=lambda x: (-len(x), x))
    hits: List[str] = []
    seen = set()
    for tok in candidates:
        # Simple substring presence for EN and zh; avoids complex segmentation
        if tok in msg and tok not in seen:
            hits.append(tok)
            seen.add(tok)
            if len(hits) >= limit:
                break
    return hits

def debug_snapshot() -> Dict[str, int]:
    if not _BUILT:
        build_index()
    return {k: len(v) for k, v in _TOKENS_BY_LANG.items()}