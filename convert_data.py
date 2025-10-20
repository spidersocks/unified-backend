# convert_data.py

import pandas as pd
import os
import ast

print("Starting data conversion process...")

# --- 1. Optimize and Convert News App Data ---

NEWS_APP_DIR = "news_app/data"
print(f"\nProcessing directory: {NEWS_APP_DIR}")

# File 1: news_chunks_w_umap.csv
try:
    csv_path = os.path.join(NEWS_APP_DIR, "news_chunks_w_umap.csv")
    parquet_path = os.path.join(NEWS_APP_DIR, "news_chunks_w_umap.parquet")
    
    print(f"Reading {csv_path}...")
    df_news = pd.read_csv(csv_path)
    
    # Optimization: Convert string columns with low cardinality to 'category'
    df_news['source_type'] = df_news['source_type'].astype('category')
    
    # The 'all_topics' column is a string representation of a list. Let's fix it.
    if df_news['all_topics'].dtype == 'object':
        df_news['all_topics'] = df_news['all_topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    print(f"Saving optimized data to {parquet_path}...")
    df_news.to_parquet(parquet_path, index=False)
    print("...Done.")

except FileNotFoundError:
    print(f"[WARNING] {csv_path} not found. Skipping.")


# File 2: avg_sentiment_by_source_topic.csv
try:
    csv_path = os.path.join(NEWS_APP_DIR, "avg_sentiment_by_source_topic.csv")
    parquet_path = os.path.join(NEWS_APP_DIR, "avg_sentiment_by_source_topic.parquet")
    
    print(f"Reading {csv_path}...")
    df_sentiment = pd.read_csv(csv_path)

    # Optimization: Convert multiple columns to 'category'
    for col in ['source_type', 'source_name', 'topic', 'sentiment_label']:
        if col in df_sentiment.columns:
            df_sentiment[col] = df_sentiment[col].astype('category')
            
    print(f"Saving optimized data to {parquet_path}...")
    df_sentiment.to_parquet(parquet_path, index=False)
    print("...Done.")

except FileNotFoundError:
    print(f"[WARNING] {csv_path} not found. Skipping.")


# --- 2. Optimize and Convert Run Calculator App Data ---

RUN_APP_DIR = "run_calculator_app/tables"
print(f"\nProcessing directory: {RUN_APP_DIR}")

if os.path.exists(RUN_APP_DIR):
    for filename in os.listdir(RUN_APP_DIR):
        if filename.endswith(".csv"):
            csv_path = os.path.join(RUN_APP_DIR, filename)
            parquet_path = os.path.join(RUN_APP_DIR, filename.replace(".csv", ".parquet"))
            
            print(f"Reading {csv_path}...")
            df_run = pd.read_csv(csv_path)
            
            # All columns in these files are numeric, so we just need to convert the format
            print(f"Saving optimized data to {parquet_path}...")
            df_run.to_parquet(parquet_path, index=False)
            print("...Done.")
else:
    print(f"[WARNING] Directory {RUN_APP_DIR} not found. Skipping.")

print("\nConversion complete!")