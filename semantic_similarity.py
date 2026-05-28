"""
Semantic Similarity with OpenAI Embeddings
===========================================
This script teaches semantic similarity using real song lyrics and OpenAI embeddings.

HOW TO RUN
----------
Default (uses built-in queries and droplingizer lyrics):
  python semantic_similarity.py

Custom queries via CLI:
  python semantic_similarity.py --query "heartbreak and rain" "summer party"

Custom CSV + custom queries:
  python semantic_similarity.py my_lyrics.csv --query "lost and alone"

Edit queries directly in the script (see CUSTOM_QUERIES near the bottom of the file).

CONCEPT: SEMANTIC SIMILARITY
------------------------------
Unlike Bag of Words (which counts words), semantic similarity captures *meaning*.
It uses an embedding model to convert each text into a dense vector of numbers —
a point in a high-dimensional space where similar meanings cluster together.

Example:
  "I'm so happy today"   → [0.21, -0.54, 0.88, ...]
  "I feel very joyful"   → [0.19, -0.51, 0.85, ...]   ← close in space → similar!
  "The car needs repair" → [-0.73, 0.12, -0.34, ...]  ← far away → dissimilar

The model used here is OpenAI's text-embedding-3-small:
  - Input: a string of text
  - Output: a vector of 1536 numbers representing its meaning
  - Trained on vast amounts of text, so it understands synonyms, themes, and context

SIMILARITY MEASURE: COSINE SIMILARITY (same as BoW, different vectors)
------------------------------------------------------------------------
We still use cosine similarity to compare vectors. The key difference is that
these vectors encode *meaning*, not just word frequency.

COST NOTE
---------
Each song's lyrics are sent to the OpenAI API once. Embeddings are cached locally
in a .pkl file so re-running the script doesn't re-embed the same songs.
"""

import argparse
import pickle
import sys
from pathlib import Path

try:
    import pandas as pd
    from dotenv import load_dotenv
    from openai import OpenAI
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ModuleNotFoundError as error:
    project_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    print(f"Missing Python package: {error.name}")
    print(f"  {project_python} {Path(__file__).resolve()}")
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "input-data" / "droplingizer_lyrics.csv"
EMBED_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_DIR / ".env")

parser = argparse.ArgumentParser(description="Semantic similarity over song lyrics using OpenAI embeddings")
parser.add_argument(
    "csv",
    nargs="?",
    default=DEFAULT_CSV,
    help="Path to a lyrics CSV file (default: droplingizer_lyrics.csv)",
)
parser.add_argument(
    "--query",
    nargs="+",
    metavar="PHRASE",
    help='One or more query phrases, e.g. --query "heartbreak and rain" "summer party"',
)
args = parser.parse_args()

csv_path = Path(args.csv)
CACHE_FILE = csv_path.with_suffix(".embeddings.pkl")

client = OpenAI()

# =============================================================================
# STEP 1: Load data
# =============================================================================

print("=" * 60)
print("STEP 1: Loading lyrics data")
print("=" * 60)

df = pd.read_csv(args.csv)
df = df[df["lyrics_found"] == True][["song_title", "primary_artist", "plain_lyrics"]].dropna()
df = df.drop_duplicates(subset=["song_title", "primary_artist"]).reset_index(drop=True)

print(f"Songs with lyrics: {len(df)}")
example = df.sample(n=1).iloc[0]
print(f"Example: '{example['song_title']}' by {example['primary_artist']}\n")


# =============================================================================
# STEP 2: Embed lyrics with OpenAI
# =============================================================================

print("=" * 60)
print("STEP 2: Generating embeddings")
print("=" * 60)
print(f"""
Model: {EMBED_MODEL}
Each lyric is sent to OpenAI and returned as a vector of 1536 numbers.
Embeddings are cached locally so we don't re-pay for songs we've already embedded.
""")

# Load existing cache
cache: dict[str, list[float]] = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

song_keys = [f"{row['song_title']}|{row['primary_artist']}" for _, row in df.iterrows()]
to_embed = [i for i, key in enumerate(song_keys) if key not in cache]

if to_embed:
    print(f"Embedding {len(to_embed)} new songs (cached: {len(df) - len(to_embed)})...")
    # Batch in groups of 100 to stay within API limits
    BATCH = 100
    for batch_start in range(0, len(to_embed), BATCH):
        batch_indices = to_embed[batch_start : batch_start + BATCH]
        texts = [df.iloc[i]["plain_lyrics"] for i in batch_indices]
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        for j, embedding_obj in enumerate(response.data):
            key = song_keys[batch_indices[j]]
            cache[key] = embedding_obj.embedding
        print(f"  Embedded {min(batch_start + BATCH, len(to_embed))}/{len(to_embed)}")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print("  Cache saved.\n")
else:
    print(f"All {len(df)} songs loaded from cache.\n")

embeddings = np.array([cache[key] for key in song_keys])
print(f"Embedding matrix shape: {embeddings.shape}  ({embeddings.shape[0]} songs × {embeddings.shape[1]} dimensions)\n")


# =============================================================================
# STEP 3: Compare two songs — BoW vs Semantic
# =============================================================================

print("=" * 60)
print("STEP 3: Comparing two songs — what changes with embeddings?")
print("=" * 60)
print("""
We'll pick two pairs of songs and show how semantic similarity can differ
from what Bag of Words would give you.
""")

PAIRS = [
    ("Say Something", "Wish You The Best"),
    ("Beautiful Things", "Moral of the Story (feat. Niall Horan) - Bonus Track"),
]

for title_a, title_b in PAIRS:
    match_a = df[df["song_title"].str.lower() == title_a.lower()]
    match_b = df[df["song_title"].str.lower() == title_b.lower()]
    if match_a.empty or match_b.empty:
        continue
    idx_a, idx_b = match_a.index[0], match_b.index[0]
    sim = cosine_similarity([embeddings[idx_a]], [embeddings[idx_b]])[0][0]
    song_a = df.iloc[idx_a]
    song_b = df.iloc[idx_b]
    print(f"  '{song_a['song_title']}' by {song_a['primary_artist']}")
    print(f"  '{song_b['song_title']}' by {song_b['primary_artist']}")
    print(f"  Semantic similarity: {sim:.4f}\n")


# =============================================================================
# STEP 4: Find most semantically similar songs for a query song
# =============================================================================

print("=" * 60)
print("STEP 4: Finding the most semantically similar songs")
print("=" * 60)

DEMO_SONGS = [
    "Beautiful Things",
    "Punchline",
]


def find_similar(title: str, top_n: int = 5):
    matches = df[df["song_title"].str.lower() == title.lower()]
    if matches.empty:
        print(f"\n  '{title}' not found in dataset — skipping.\n")
        return
    idx = matches.index[0]
    row = df.iloc[idx]

    sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    sims[idx] = -1  # exclude itself
    top_indices = sims.argsort()[-top_n:][::-1]

    print(f"\n  Query: '{row['song_title']}' by {row['primary_artist']}")
    print(f"  {'Rank':<6} {'Similarity':>10}  Song")
    print(f"  {'----':<6} {'----------':>10}  ----")
    for rank, i in enumerate(top_indices, 1):
        song = df.iloc[i]
        print(f"  {rank:<6} {sims[i]:>10.4f}  '{song['song_title']}' by {song['primary_artist']}")


for title in DEMO_SONGS:
    find_similar(title)

# =============================================================================
# STEP 5: Artist comparison — top 2 artists by song count
# =============================================================================

top_artists = df["primary_artist"].value_counts().head(2).index.tolist()
artist_a, artist_b = top_artists[0], top_artists[1]

print("=" * 60)
print(f"STEP 5: Artist comparison — {artist_a} vs {artist_b}")
print("=" * 60)
print(f"""
Do two artists write about similar themes? We pick the two artists with the
most songs in the dataset ({artist_a}, {artist_b}) and compare every combination.
""")

songs_a = df[df["primary_artist"] == artist_a].reset_index()
songs_b = df[df["primary_artist"] == artist_b].reset_index()

col_width = 16
header = f"  {'':30}" + "".join(f"{t[:col_width]:>{col_width}}" for t in songs_b["song_title"])
print(header)
print("  " + "-" * (30 + col_width * len(songs_b)))

for _, row_a in songs_a.iterrows():
    row_str = f"  {row_a['song_title'][:28]:<30}"
    for _, row_b in songs_b.iterrows():
        sim = cosine_similarity([embeddings[row_a["index"]]], [embeddings[row_b["index"]]])[0][0]
        row_str += f"{sim:>{col_width}.4f}"
    print(row_str)

# =============================================================================
# STEP 6: Most similar and most different pairs across all songs
# =============================================================================

print()
print("=" * 60)
print("STEP 6: Most similar and most different song pairs")
print("=" * 60)
print("""
Looking at all possible pairs in the dataset — which songs are closest
in meaning, and which are furthest apart?
""")

all_sims = cosine_similarity(embeddings)
n = len(df)

pairs = [
    (all_sims[i, j], i, j)
    for i in range(n)
    for j in range(i + 1, n)
]

pairs.sort(key=lambda x: x[0])
most_different = pairs[:5]
most_similar = pairs[-5:][::-1]

print("  Top 5 most SIMILAR pairs:")
for rank, (sim, i, j) in enumerate(most_similar, 1):
    print(f"    {rank}. {sim:.4f}  '{df.iloc[i]['song_title']}' by {df.iloc[i]['primary_artist']}")
    print(f"             '{df.iloc[j]['song_title']}' by {df.iloc[j]['primary_artist']}")

print()
print("  Top 5 most DIFFERENT pairs:")
for rank, (sim, i, j) in enumerate(most_different, 1):
    print(f"    {rank}. {sim:.4f}  '{df.iloc[i]['song_title']}' by {df.iloc[i]['primary_artist']}")
    print(f"             '{df.iloc[j]['song_title']}' by {df.iloc[j]['primary_artist']}")

# =============================================================================
# STEP 7: Custom text query
# =============================================================================

print()
print("=" * 60)
print("STEP 7: Custom text query")
print("=" * 60)


def query_songs(phrase: str, top_n: int = 5):
    response = client.embeddings.create(model=EMBED_MODEL, input=[phrase])
    query_vec = np.array(response.data[0].embedding).reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_n:][::-1]
    print(f"\n  Top {top_n} songs matching '{phrase}':")
    print(f"  {'Rank':<6} {'Similarity':>10}  Song")
    print(f"  {'----':<6} {'----------':>10}  ----")
    for rank, i in enumerate(top_indices, 1):
        song = df.iloc[i]
        print(f"  {rank:<6} {sims[i]:>10.4f}  '{song['song_title']}' by {song['primary_artist']}")
    print()


# ── Edit this list to add your own queries ────────────────────────────────────
CUSTOM_QUERIES = [
    "work", "love is all around"
]
# ──────────────────────────────────────────────────────────────────────────────

queries = args.query or CUSTOM_QUERIES
print(f"Queries: {queries}\n")
for phrase in queries:
    query_songs(phrase)

print()
print("=" * 60)
print("KEY TAKEAWAY")
print("=" * 60)
print("""
Semantic similarity understands *meaning*, not just vocabulary.
  - Synonyms ("joyful" / "happy") are treated as similar
  - Theme and emotion are captured across different words
  - It works even when songs share zero words in common

The trade-off vs Bag of Words:
  - More powerful and accurate
  - Requires an API call (costs money, needs internet)
  - Harder to explain WHY two songs are similar (the vector is a black box)
""")
