# news_app/router.py

import pandas as pd
import os
import numpy as np
from typing import List, Optional
import traceback
import json

from fastapi import APIRouter, Query, HTTPException

# Define the router for this submodule
router = APIRouter(
    prefix="/news",
    tags=["News Analysis"],
)

# --- Define file paths ---
BASE_DIR = os.path.dirname(__file__)
NEWS_CHUNKS_FILE = os.path.join(BASE_DIR, "data/news_chunks_w_umap.parquet")
TOPWORDS_PARQUET = os.path.join(BASE_DIR, "data/topwords_by_topic.parquet")
AVG_SENTIMENT_FILE = os.path.join(BASE_DIR, "data/avg_sentiment_by_source_topic.parquet")
STANCE_Z_CSV = os.path.join(BASE_DIR, "data/stance_z_agg.csv")

# --- Define constant column names ---
ALL_TOPICS_COL = "all_topics"
SOURCE_COL = "source_type"

# --- Data Loading Functions ---
def load_news_df():
    return pd.read_parquet(NEWS_CHUNKS_FILE, columns=["all_topics", "source_type"])

def load_topwords_df():
    return pd.read_parquet(TOPWORDS_PARQUET, columns=["source_type", "source_name", "topic", "top_words", "top_words_plain"])

def load_avg_sentiment_df():
    return pd.read_parquet(AVG_SENTIMENT_FILE)

def load_stance_z_df():
    return pd.read_csv(STANCE_Z_CSV, usecols=["topic", "source_type", "stance_score_z"])

# --- Load all dataframes into memory ---
news_df = load_news_df()
topwords_df = load_topwords_df()
avg_sentiment_df = load_avg_sentiment_df()

# === Helper Functions ===
def get_subtopic_distribution(drill_path=None):
    try:
        if not drill_path:
            subdf = news_df[news_df[ALL_TOPICS_COL].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)].copy()
            subdf['label'] = subdf[ALL_TOPICS_COL].apply(lambda x: x[-1])
            group = subdf.groupby([SOURCE_COL, 'label']).size().reset_index(name='count')
            group['source_total'] = group.groupby(SOURCE_COL)['count'].transform('sum')
            group['proportion'] = group['count'] / group['source_total']
            order = group.groupby('label')['count'].sum().sort_values(ascending=False).index
            group['label'] = pd.Categorical(group['label'], categories=order, ordered=True)
            group = group.sort_values(['label', SOURCE_COL])
            return group.to_dict(orient='records')

        if not isinstance(drill_path, (list, tuple)) or len(drill_path) < 1:
            return []
        drill_path = list(reversed(drill_path))
        def matches_drill_path(all_topics):
            if not isinstance(all_topics, (list, np.ndarray)) or len(all_topics) < len(drill_path):
                return False
            return all_topics[-len(drill_path):].tolist() == drill_path
        subset = news_df[news_df[ALL_TOPICS_COL].apply(matches_drill_path)].copy()
        if subset.empty: return []
        def get_next_subtopic(all_topics):
            idx = len(all_topics) - len(drill_path)
            return all_topics[idx - 1] if idx > 0 else None
        subset['label'] = subset[ALL_TOPICS_COL].apply(get_next_subtopic)
        subset = subset.dropna(subset=['label'])
        if subset.empty: return []
        group = subset.groupby([SOURCE_COL, 'label']).size().reset_index(name='count')
        group['source_total'] = group.groupby(SOURCE_COL)['count'].transform('sum')
        group['proportion'] = group['count'] / group['source_total']
        order = group.groupby('label')['count'].sum().sort_values(ascending=False).index
        group['label'] = pd.Categorical(group['label'], categories=order, ordered=True)
        group = group.sort_values(['label', SOURCE_COL])
        return group.to_dict(orient='records')
    except Exception as e:
        print(f"[ERROR] in get_subtopic_distribution: {e}")
        return []

def format_topic_label(topic_key: str) -> str:
    if topic_key.startswith("topic_"): label = topic_key[len("topic_"):]
    else: label = topic_key
    return label.replace("_", " ").strip().capitalize()

def _safe_float(x):
    try:
        # Convert numpy number types and pandas NA to float
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float('nan')
        return float(x)
    except Exception:
        # If unparseable, skip by returning NaN
        return float('nan')

def _is_nan(x):
    try:
        return isinstance(x, float) and np.isnan(x)
    except Exception:
        return False

def process_word_list(word_data):
    """
    Normalize top_words into a list of dicts: [{'text': str, 'value': float}, ...]
    Robust to numpy arrays, pandas objects, and mixed inner item types.
    """
    try:
        # Normalize container
        if isinstance(word_data, np.ndarray):
            word_data = word_data.tolist()
        # Some parquet backends may store lists as tuples
        if isinstance(word_data, tuple):
            word_data = list(word_data)

        if not isinstance(word_data, (list, tuple)):
            return []

        out = []
        for idx, item in enumerate(word_data):
            try:
                # Already a mapping
                if isinstance(item, dict):
                    if 'text' in item and 'value' in item:
                        text = str(item['text'])
                        value = _safe_float(item['value'])
                        if not _is_nan(value):
                            out.append({'text': text, 'value': float(value)})
                        continue
                    else:
                        # Unexpected dict shape; skip
                        continue

                # Typical pair-like cases
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = str(item[0])
                    value = _safe_float(item[1])
                    if not _is_nan(value):
                        out.append({'text': text, 'value': float(value)})
                    continue

                # Fallback: try to coerce to sequence
                try:
                    seq = list(item)
                    if len(seq) >= 2:
                        text = str(seq[0])
                        value = _safe_float(seq[1])
                        if not _is_nan(value):
                            out.append({'text': text, 'value': float(value)})
                        continue
                except Exception:
                    pass

                # If we get here, item is unrecognized; log a small sample
                if idx < 3:
                    print(f"[DEBUG] Unrecognized top_words item type: {type(item)} repr={repr(item)[:200]}", flush=True)
            except Exception as inner_e:
                print(f"[DEBUG] Error processing item in top_words: {inner_e} type={type(item)}", flush=True)
                continue

        return out
    except Exception as e:
        print(f"[DEBUG] process_word_list failed: {e}", flush=True)
        return []

def _sanitize_results_for_json(results):
    """
    Ensure all values are plain Python types before FastAPI encodes them.
    """
    for r in results:
        # Force strings for source fields
        r['source_type'] = str(r.get('source_type', ''))
        r['source_name'] = str(r.get('source_name', ''))
        r['topic'] = str(r.get('topic', ''))
        # Sanitize top_words
        cleaned = []
        for tw in r.get('top_words', []):
            try:
                cleaned.append({'text': str(tw.get('text', '')), 'value': float(_safe_float(tw.get('value', float('nan'))))})
            except Exception as e:
                print(f"[DEBUG] Failed to sanitize tw entry: {e} entry={repr(tw)[:200]}", flush=True)
        r['top_words'] = cleaned
        # Sanitize top_words_plain if present
        if 'top_words_plain' in r:
            twp = r['top_words_plain']
            if isinstance(twp, (list, tuple, np.ndarray)):
                r['top_words_plain'] = [str(x) for x in list(twp)]
            elif twp is None:
                r['top_words_plain'] = None
            else:
                r['top_words_plain'] = [str(twp)]
    return results

# === API Endpoints ===

@router.get("/api/topics/")
def get_broad_topics():
    return {"topics": get_subtopic_distribution()}

PATH_SEPARATOR = "||"

@router.get("/api/topics/drilldown/")
def get_subtopics(path: str = Query("")):
    drill_path = [p.strip() for p in path.split(PATH_SEPARATOR) if p.strip()]
    return {"topics": get_subtopic_distribution(drill_path or None)}

@router.get("/api/stance/zscores/")
def get_stance_z_data(topics: Optional[str] = Query(None)):
    df = load_stance_z_df()
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
        df = df[df["topic"].isin(topic_list)]
    return {"data": df.to_dict(orient="records")}

@router.get("/api/wordcloud/topwords/")
def get_topwords(source_type: Optional[str] = Query(None), source_name: Optional[str] = Query(None), topic: Optional[str] = Query(None)):
    try:
        df = topwords_df.copy()
        if source_type: df = df[df["source_type"] == source_type]
        if source_name: df = df[df["source_name"] == source_name]
        if topic: df = df[df["topic"] == topic]

        # Debug: log counts after filtering
        print(f"[DEBUG] topwords filter -> source_type={source_type} source_name={source_name} topic={topic} rows={len(df)}", flush=True)

        if df.empty or 'top_words' not in df.columns:
            print("[DEBUG] No rows or missing 'top_words' column after filtering.", flush=True)
            return {"data": []}

        # Inspect the raw cell for your failing case explicitly
        if source_type == "podcast" and source_name == "Club Shay Shay" and topic == "topic_india":
            try:
                raw_vals = df['top_words'].tolist()
                print(f"[DEBUG] Raw top_words sample count={len(raw_vals)}", flush=True)
                if raw_vals:
                    sample = raw_vals[0]
                    # Avoid huge spam
                    print(f"[DEBUG] Raw first top_words type={type(sample)} preview={repr(sample)[:500]}", flush=True)
                    if isinstance(sample, (list, tuple, np.ndarray)):
                        for i, it in enumerate(list(sample)[:5]):
                            print(f"[DEBUG] Inner item[{i}] type={type(it)} preview={repr(it)[:200]}", flush=True)
            except Exception as e:
                print(f"[DEBUG] Failed inspecting raw top_words: {e}", flush=True)

        # Robust normalization
        df['top_words'] = df['top_words'].apply(process_word_list)

        # Filter out rows where processing resulted in an empty list
        df = df[df['top_words'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

        if df.empty:
            print("[DEBUG] All rows dropped after normalization; returning empty.", flush=True)
            return {"data": []}

        response_cols = ['source_type', 'source_name', 'topic', 'top_words', 'top_words_plain']
        results = df[response_cols].to_dict(orient='records')

        # Extra validation logging: detect suspicious entries
        for ri, r in enumerate(results[:5]):
            bad_items = [tw for tw in r.get('top_words', []) if not isinstance(tw, dict) or not ('text' in tw and 'value' in tw)]
            if bad_items:
                print(f"[DEBUG] Suspicious top_words in result[{ri}] count={len(bad_items)} first={repr(bad_items[0])[:200]}", flush=True)

        # Sanitize to plain Python types
        results = _sanitize_results_for_json(results)

        # Try a pre-encode to surface any remaining issues with a helpful log
        try:
            json.dumps({"data": results})
        except Exception as e:
            print(f"[DEBUG] json.dumps pre-encode failed: {e}", flush=True)
            # Log types of first problematic entry
            if results:
                first = results[0]
                print(f"[DEBUG] First result types: "
                      f"source_type={type(first.get('source_type'))} "
                      f"source_name={type(first.get('source_name'))} "
                      f"topic={type(first.get('topic'))} "
                      f"top_words_type={type(first.get('top_words'))}", flush=True)
                if isinstance(first.get('top_words'), list) and first['top_words']:
                    print(f"[DEBUG] First top_word entry type={type(first['top_words'][0])} value={repr(first['top_words'][0])[:200]}", flush=True)

        # Final debug for the specific failing filter
        if source_type == "podcast" and source_name == "Club Shay Shay" and topic == "topic_india":
            print(f"[DEBUG] Returning results count={len(results)} first_item_preview={repr(results[0])[:500]}", flush=True)

        return {"data": results}

    except Exception as e:
        print(f"[ERROR] in get_topwords endpoint: {e}")
        traceback.print_exc()  # Print full stack trace to logs
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

@router.get("/api/wordcloud/sentiment/")
def get_sentiment(source_type: Optional[str] = Query(None), source_name: Optional[str] = Query(None), topic: Optional[str] = Query(None)):
    df = avg_sentiment_df.copy()
    if source_type: df = df[df["source_type"] == source_type]
    if source_name: df = df[df["source_name"] == source_name]
    if topic: df = df[df["topic"] == topic]
    return {"data": df.to_dict(orient="records")}

@router.get("/api/wordcloud/options/")
def get_wordcloud_options(source_type: Optional[str] = Query(None)):
    df = topwords_df.copy()
    if source_type: df = df[df["source_type"] == source_type]
    sources_df = df[["source_type", "source_name"]].drop_duplicates().sort_values(by=["source_type", "source_name"])
    sources = sources_df.to_dict(orient="records")
    topics = sorted([t for t in df["topic"].dropna().unique().tolist()])
    if "all_topics" in topics:
        topics = ["all_topics"] + [t for t in topics if t != "all_topics"]
    return {"sources": sources, "topics": topics}

@router.get("/api/wordcloud/common_topics/")
def get_common_topics(left_source_type: str = Query(...), left_source_name: str = Query(...), right_source_type: str = Query(...), right_source_name: str = Query(...)):
    df = topwords_df.copy()
    left_topics = set(df[(df["source_type"] == left_source_type) & (df["source_name"] == left_source_name)]["topic"].dropna().unique())
    right_topics = set(df[(df["source_type"] == right_source_type) & (df["source_name"] == right_source_name)]["topic"].dropna().unique())
    common = sorted(list(left_topics & right_topics))
    
    topic_keys = common
    if "all_topics" in topic_keys:
        topic_keys = ["all_topics"] + [t for t in topic_keys if t != "all_topics"]
    
    topics = [{"key": key, "label": "All topics" if key == "all_topics" else format_topic_label(key)} for key in topic_keys]
    return {"topics": topics}