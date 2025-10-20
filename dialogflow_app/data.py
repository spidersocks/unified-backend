# C:\Users\sfont\unified_backend\dialogflow_app\data.py

import os
import pandas as pd
import time # Import time for measuring load duration

# --- Configuration ---
# Use an environment variable to store the Google Sheet CSV export URL
GOOGLE_SHEET_URL = os.environ.get("COURSE_DATA_SHEET_URL")

# Global variable to hold the loaded course data dictionary
COURSE_DETAILS = {}

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