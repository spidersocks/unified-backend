# Deprecated: logic moved into ContentStore. This module now acts as a thin shim.
# It preserves existing imports (get_course_info, format_course_display, get_course_list, DISPLAY_NAMES)
# while delegating to dialogflow_app.content_store.

from dialogflow_app.content_store import (
    STORE as _STORE,
    get_course_info,
    get_course_list,
    format_course_display,
)

# Optional: keep a small static map for legacy callers that imported DISPLAY_NAMES.
# New code should rely on the display_names sheet via ContentStore.
DISPLAY_NAMES = {
    "Playgroups": {"en": "Playgroups", "zh-hk": "幼兒班", "zh-cn": "幼儿班"},
    "Phonics": {"en": "Phonics", "zh-hk": "英語拼音", "zh-cn": "英语拼音"},
    "LanguageArts": {"en": "Language Arts", "zh-hk": "英語語文課", "zh-cn": "英语语文课"},
    "Clevercal": {"en": "Clevercal Math", "zh-hk": "數學班 (Clevercal)", "zh-cn": "数学班 (Clevercal)"},
    "Alludio": {"en": "Alludio Educational Games", "zh-hk": "教育遊戲 (Alludio)", "zh-cn": "教育游戏 (Alludio)"},
    "ToddlerCharRecognition": {"en": "Chinese Character Recognition", "zh-hk": "寶寶愛認字", "zh-cn": "宝宝爱认字"},
    "MandarinPinyin": {"en": "Mandarin Pinyin", "zh-hk": "魔法拼音班", "zh-cn": "魔法拼音班"},
    "ChineseLanguageArts": {"en": "Chinese Language Arts", "zh-hk": "中文語文課", "zh-cn": "中文语文课"},
    "PrivateClass": {"en": "Private Class", "zh-hk": "私人課", "zh-cn": "私人课"},
}

# Expose the underlying store for advanced callers (optional)
CONTENT_STORE = _STORE