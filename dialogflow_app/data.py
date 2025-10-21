import os
import pandas as pd
import time

try:
    from dialogflow_app.content_store import ContentStore
    _CONTENT_STORE = ContentStore()
except Exception:
    _CONTENT_STORE = None

# --- Configuration ---
# Use an environment variable to store the Google Sheet CSV export URL
GOOGLE_SHEET_URL = os.environ.get("COURSE_DATA_SHEET_URL")

# Global variable to hold the loaded course data dictionary
COURSE_DETAILS = {}

# --- New: Display Name Mapping ---
# Maps the canonical name (Dialogflow entity value) to user-friendly names
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


def load_course_data():
    """
    Loads course data from the Google Sheet CSV export URL and transforms it 
    into a nested dictionary.
    """
    global COURSE_DETAILS
    
    start_time = time.time()
    print(f"[INFO] Attempting to load course data from Google Sheet URL...", flush=True)

    if not GOOGLE_SHEET_URL:
        print("[CRITICAL ERROR] COURSE_DATA_SHEET_URL environment variable is not set. Using empty data.", flush=True)
        COURSE_DETAILS = {}
        return

    try:
        # Read the CSV directly from the URL using pandas
        # This requires the sheet to be published to the web as CSV
        df = pd.read_csv(GOOGLE_SHEET_URL)
        
        # Ensure the canonical_name column exists and is set as the index
        if 'canonical_name' not in df.columns:
            raise ValueError("Data source must contain a 'canonical_name' column.")

        df = df.set_index('canonical_name').fillna('')

        # Convert the DataFrame structure into the required nested dictionary format:
        # { canonical_name: { lang_code: description, ... }, ... }
        COURSE_DETAILS = df.T.to_dict()
        
        load_duration = time.time() - start_time
        print(f"[INFO] Successfully loaded {len(COURSE_DETAILS)} courses in {load_duration:.2f} seconds.", flush=True)

    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load course data from Google Sheet: {e}", flush=True)
        COURSE_DETAILS = {}

# Execute the loading function immediately when the module is imported
load_course_data()


def get_course_info(course_name: str, lang_code: str) -> str:
    """Retrieves the course description based on the canonical name and language."""
    
    # Use the globally loaded data
    details = COURSE_DETAILS.get(course_name)
    
    if not details:
        # Course not found in the loaded data
        return f"Sorry, I could not find details for the course: {course_name}."

    # Normalize language code (e.g., zh-HK -> zh-hk, en-US -> en)
    lang_code = lang_code.lower()
    
    # Dialogflow often sends 'zh-cn' or 'zh-hk' but sometimes sends 'zh' as languageCode.
    # We need to handle the specific codes used in the sheet columns: 'en', 'zh-HK', 'zh-CN'
    
    # Map common Dialogflow codes to sheet column names
    lang_map = {
        'en': 'en',
        'zh-cn': 'zh-CN',
        'zh-hk': 'zh-HK',
        'zh': 'zh-CN' # Default Chinese fallback if only 'zh' is sent
    }
    
    target_column = lang_map.get(lang_code, 'en') # Default to English if code is unknown
    
    # 1. Try the mapped language column
    if target_column in details and details[target_column]:
        return details[target_column]
    
    # 2. Fallback to English (if the target language column was empty)
    if 'en' in details and details['en']:
        return details['en']
        
    # 3. Course exists, but no description found for any language
    return f"Details for {course_name} are currently unavailable."


def get_display_name(canonical_name: str, lang_code: str) -> str:
    # Prefer ContentStore if available
    if _CONTENT_STORE:
        try:
            return _CONTENT_STORE._get_display_name(canonical_name, lang_code)
        except Exception:
            pass
    # existing fallback mapping
    if lang_code.startswith('zh-hk'):
        code = 'zh-hk'
    elif lang_code.startswith('zh-cn') or lang_code.startswith('zh'):
        code = 'zh-cn'
    else:
        code = 'en'
    return DISPLAY_NAMES.get(canonical_name, {}).get(code, canonical_name)

def format_course_display(canonical_name: str, lang_code: str) -> str:
    """
    Returns localized course name, and includes an alias in parentheses to avoid confusion:
      - For zh requests: zh name (EN name)
      - For en requests: EN name (HK name if available, else CN)
    """
    if _CONTENT_STORE:
        try:
            return _CONTENT_STORE.format_course_name(canonical_name, lang_code)
        except Exception:
            pass
    # Fallback using our static DISPLAY_NAMES map
    en = DISPLAY_NAMES.get(canonical_name, {}).get('en', canonical_name)
    zh_hk = DISPLAY_NAMES.get(canonical_name, {}).get('zh-hk', "")
    zh_cn = DISPLAY_NAMES.get(canonical_name, {}).get('zh-cn', "")

    if lang_code.startswith('en'):
        alt = zh_hk or zh_cn
        return f"{en} ({alt})" if alt and alt != en else en
    else:
        primary = zh_hk if lang_code.startswith('zh-hk') else (zh_cn or zh_hk)
        primary = primary or en
        alt = en if en and en != primary else ""
        return f"{primary} ({alt})" if alt else primary

def get_course_list(lang_code: str) -> str:
    """
    Generates a dynamic, localized list of available courses.
    """
    
    # 1. Get localized display names
    course_names = []
    for canonical_name in COURSE_DETAILS.keys():
        # Exclude courses that shouldn't be listed in the main curriculum overview
        if canonical_name in ["PrivateClass"]:
            continue
        
        # Get the localized display name
        display_name = get_display_name(canonical_name, lang_code)
        course_names.append(f"*{display_name}*") # Use WhatsApp markdown for emphasis

    # 2. Format the list based on language
    if lang_code.startswith('en'):
        course_list_str = ", ".join(course_names)
        
        # Replace the last comma with "and" for natural English flow
        last_comma_index = course_list_str.rfind(',')
        if last_comma_index > 0:
            # Handle the Oxford comma case for better readability
            course_list_str = course_list_str[:last_comma_index] + ", and" + course_list_str[last_comma_index + 1:]
        elif len(course_names) == 2:
            # Handle list of two items (e.g., A and B)
            course_list_str = " and ".join(course_names)
            
        response = (
            f"We currently offer programs in {course_list_str}. "
            f"Which area are you interested in?"
        )
    else: # Chinese (zh-hk, zh-cn)
        # Use the Chinese comma (、) for listing items
        course_list_str = "、".join(course_names)
        
        if lang_code == 'zh-hk':
            response = (
                f"我們提供{course_list_str}課程。請問您對哪個範疇感興趣？"
            )
        else: # zh-cn (or default zh)
            response = (
                f"我们提供{course_list_str}课程。请问您对哪个范畴感兴趣？"
            )
            
    return response