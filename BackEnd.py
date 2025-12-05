from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import os
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


CSV_FILENAME = "cleaned_steam_games.csv"
EMBED_FILENAME = "steam_embeddings.npy"   # where embeddings are cached
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

dataset_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading dataset from {CSV_FILENAME}...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, CSV_FILENAME)
    embed_path = os.path.join(current_dir, EMBED_FILENAME)

    if not os.path.exists(csv_path):
        print("❌ CSV not found.")
        dataset_context["df"] = pd.DataFrame()
        yield
        return

    # Load CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df['about_the_game'] = df['about_the_game'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['short_desc'] = df['about_the_game'].astype(str).str.slice(0, 300)

    descriptions = df['short_desc'].tolist()

    # Load or compute embeddings
    if os.path.exists(embed_path):
        try:
            print("🔵 Found cached embeddings — loading...")
            embeddings = np.load(embed_path)
            print("✅ Loaded cached embeddings successfully.")
        except Exception as e:
            print("⚠️ Could not load cached embeddings. Recomputing...", e)
            embeddings = compute_and_cache_embeddings(descriptions, embed_path)
    else:
        print("🟡 No cached embeddings found — computing now...")
        embeddings = compute_and_cache_embeddings(descriptions, embed_path)

    dataset_context["df"] = df
    dataset_context["embeddings"] = embeddings

    print(f"🚀 API ready. Total games loaded: {len(df)}")
    yield
    dataset_context.clear()


def compute_and_cache_embeddings(descriptions, embed_path):
    """
    Computes SentenceTransformer embeddings, saves to disk,
    and returns the numpy matrix.
    """
    print("⬇️ Loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("⚙️ Encoding descriptions (this happens only once)...")
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True
    ).astype(np.float32)

    print(f"💾 Saving embeddings to {embed_path} ...")
    np.save(embed_path, embeddings)

    print("✅ Embeddings saved.")
    return embeddings

# --- HELPER FUNCTIONS ---
def clean_price(price_val):
    try:
        if pd.isna(price_val): return 0.0
        s = str(price_val).lower().strip()
        if 'free' in s: return 0.0
        s = ''.join(c for c in s if c.isdigit() or c == '.')
        return float(s) if s else 0.0
    except:
        return 0.0

# FASTAPI SETUP
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Recommendation API is running."}


@app.get("/games")
def get_games(limit: int = 10, search: str = None):
    df = dataset_context.get("df")
    if df is None or df.empty: return []

    if search:
        filtered_df = df[df['name'].astype(str).str.contains(search, case=False, na=False)]
    else:
        filtered_df = df

    subset = filtered_df.head(limit).where(pd.notnull(filtered_df), None)
    return subset[['name']].to_dict(orient="records")


@app.get("/recommend")
def get_recommendation(game_name: str, price_filter: Optional[str] = Query(None)):
    df = dataset_context.get("df")
    embeddings = dataset_context.get("embeddings")

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Exact match first
    matches = df[df['name'].str.lower() == game_name.lower()]

    # Partial match fallback
    if matches.empty:
        matches = df[df['name'].str.lower().str.contains(game_name.lower(), regex=False)]

    if matches.empty:
        raise HTTPException(status_code=404, detail="Game not found")

    idx = matches.index[0]
    matched_name = df.iloc[idx]['name']

    # Compute similarity
    target_vec = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(target_vec, embeddings)[0]

    # Get candidates (Top 100 most similar)
    # We still fetch a larger pool to allow filtering, but we will sort properly later
    top_idx = similarities.argsort()[-100:-1][::-1]
    candidate_df = df.iloc[top_idx].copy()

    # 3. APPLY PRICE FILTER (Single Option)
    if price_filter and price_filter != 'any':
        candidate_df['numeric_price'] = candidate_df['price'].apply(clean_price)
        
        if price_filter == 'free':
            candidate_df = candidate_df[candidate_df['numeric_price'] == 0]
        elif price_filter == 'under_5':
            candidate_df = candidate_df[candidate_df['numeric_price'] < 5]
        elif price_filter == 'under_10':
            candidate_df = candidate_df[candidate_df['numeric_price'] < 10]
        elif price_filter == 'under_30':
            candidate_df = candidate_df[candidate_df['numeric_price'] < 30]
        elif price_filter == 'under_50':
            candidate_df = candidate_df[candidate_df['numeric_price'] < 50]
        elif price_filter == 'above_50':
            candidate_df = candidate_df[candidate_df['numeric_price'] >= 50]

    # 4. RETURN TOP 5 (DETERMINISTIC)
    # We use .head(5) to always return the absolute best matches that survived the filter
    results = candidate_df.head(5)
    
    results = results.where(pd.notnull(results), None)

    # --- ID COLUMN DETECTION ---
    possible_id_cols = ['appid', 'app_id', 'steam_appid', 'id']
    app_id_col = next((col for col in possible_id_cols if col in df.columns), None)
    
    cols_to_return = ['name', 'genres', 'about_the_game', 'price', 'header_image']
    if app_id_col:
        cols_to_return.append(app_id_col)

    response_data = results[cols_to_return].to_dict(orient="records")
    
    # Standardize 'app_id' key
    for item in response_data:
        if app_id_col and app_id_col in item:
            item['app_id'] = item.pop(app_id_col)

    return {
        "source_game": matched_name,
        "recommendations": response_data
    }