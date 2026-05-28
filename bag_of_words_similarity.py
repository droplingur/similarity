"""
Bag of Words (BoW) Similarity
==============================
CONCEPT: BAG OF WORDS
----------------------
A "bag of words" represents a piece of text as a set of word counts, ignoring
grammar and word order. Two texts are "similar" if they use the same words
with similar frequencies.

Example:
  Text A: "I love love music"  → {"i": 1, "love": 2, "music": 1}
  Text B: "I love songs"       → {"i": 1, "love": 1, "songs": 1}
  Text C: "Dogs love bones"    → {"dogs": 1, "love": 1, "bones": 1}

  A and B share more words → more similar than A and C.

SIMILARITY MEASURE: COSINE SIMILARITY
--------------------------------------
We measure similarity using cosine similarity, which compares the angle between
two word-count vectors. It ranges from 0 (nothing in common) to 1 (identical
word distribution).
"""

import argparse
from pathlib import Path
import sys

try:
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as error:
    project_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    print(f"Missing Python package: {error.name}")
    print()
    print("Run this script with the similarity project's virtual environment:")
    print(f"  {project_python} {Path(__file__).resolve()}")
    print()
    print("Or activate it first:")
    print(f"  cd {Path(__file__).resolve().parent}")
    print("  source .venv/bin/activate")
    print("  python bag_of_words_similarity.py")
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "input-data" / "droplingizer_lyrics.csv"

parser = argparse.ArgumentParser(description="Bag of Words similarity over song lyrics")
parser.add_argument(
    "csv",
    nargs="?",
    default=DEFAULT_CSV,
    help="Path to a lyrics CSV file (default: droplingizer_lyrics.csv)",
)
args = parser.parse_args()

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
# STEP 2: Build the Bag of Words matrix
# =============================================================================

print("=" * 60)
print("STEP 2: Building the Bag of Words matrix")
print("=" * 60)
print("""
CountVectorizer does three things:
  1. Builds a vocabulary of all unique words across all lyrics
  2. Removes common English stop words (the, a, is, ...) — they carry no meaning
  3. Represents each song as a vector of word counts
""")

vectorizer = CountVectorizer(stop_words="english", lowercase=True, max_features=10_000)
bow_matrix = vectorizer.fit_transform(df["plain_lyrics"])
vocabulary = vectorizer.get_feature_names_out()
word_to_index = {word: i for i, word in enumerate(vocabulary)}

print(f"Vocabulary size (after stop-word removal): {len(vocabulary)} unique words")
print(f"Matrix shape: {bow_matrix.shape[0]} songs × {bow_matrix.shape[1]} words\n")

# Show top 20 most common words across all lyrics
word_counts = bow_matrix.sum(axis=0).A1
top_words = sorted(zip(vocabulary, word_counts), key=lambda x: -x[1])[:20]
print("Top 20 most common words across all songs:")
for word, count in top_words:
    print(f"  {word:<20} {int(count):>6} occurrences")
print()


# =============================================================================
# STEP 3: Show example count vectors for two songs side by side
# =============================================================================

print("=" * 60)
print("STEP 3: Example — count vectors for two songs")
print("=" * 60)

song_a_idx = 0
song_b_idx = 1
song_a = df.iloc[song_a_idx]
song_b = df.iloc[song_b_idx]

vec_a = bow_matrix[song_a_idx].toarray()[0]
vec_b = bow_matrix[song_b_idx].toarray()[0]

# Find words that appear in either song
nonzero = set(vocabulary[i] for i in vec_a.nonzero()[0]) | set(vocabulary[i] for i in vec_b.nonzero()[0])
shared = sorted(nonzero)[:30]  # show up to 30 words

print(f"\n  {'Word':<20} {'Song A':>8} {'Song B':>8}")
print(f"  {'----':<20} {'------':>8} {'------':>8}")
for word in shared:
    idx = word_to_index[word]
    print(f"  {word:<20} {int(vec_a[idx]):>8} {int(vec_b[idx]):>8}")

score = cosine_similarity(bow_matrix[song_a_idx], bow_matrix[song_b_idx])[0][0]
print(f"\n  Song A: '{song_a['song_title']}' by {song_a['primary_artist']}")
print(f"  Song B: '{song_b['song_title']}' by {song_b['primary_artist']}")
print(f"  Cosine similarity: {score:.4f}  (0 = nothing in common, 1 = identical distribution)\n")


# =============================================================================
# STEP 4: Find most similar songs for a query song
# =============================================================================

print("=" * 60)
print("STEP 4: Finding the most similar songs (BoW)")
print("=" * 60)

DEMO_SONGS = [
    "Beautiful Things",
    "Punchline",
]


def common_words(song_idx_a: int, song_idx_b: int, top_n: int = 8) -> str:
    vec_a = bow_matrix[song_idx_a].toarray()[0]
    vec_b = bow_matrix[song_idx_b].toarray()[0]
    words_a = set(vocabulary[vec_a.nonzero()[0]])
    words_b = set(vocabulary[vec_b.nonzero()[0]])
    shared_words = words_a & words_b

    ranked_words = sorted(
        shared_words,
        key=lambda word: min(vec_a[word_to_index[word]], vec_b[word_to_index[word]]),
        reverse=True,
    )
    return ", ".join(ranked_words[:top_n]) if ranked_words else "none"


def find_similar(title: str, top_n: int = 5):
    matches = df[df["song_title"].str.lower() == title.lower()]
    if matches.empty:
        print(f"\n  '{title}' not found in dataset — skipping.\n")
        return
    idx = matches.index[0]
    row = df.iloc[idx]

    sims = cosine_similarity(bow_matrix[idx], bow_matrix).flatten()
    sims[idx] = -1  # exclude itself
    top_indices = sims.argsort()[-top_n:][::-1]

    print(f"\n  Query: '{row['song_title']}' by {row['primary_artist']}")
    print(f"  {'Rank':<6} {'Similarity':>10}  Song")
    print(f"  {'----':<6} {'----------':>10}  ----")
    for rank, i in enumerate(top_indices, 1):
        sim = sims[i]
        song = df.iloc[i]
        words = common_words(idx, i)
        print(
            f"  {rank:<6} {sim:>10.4f}  "
            f"'{song['song_title']}' by {song['primary_artist']} "
            f"(common words: {words})"
        )


for title in DEMO_SONGS:
    find_similar(title)

print()
print("=" * 60)
print("KEY TAKEAWAY")
print("=" * 60)
print("""
Bag of Words captures TOPIC similarity — songs with similar lyrical themes
(love, night, dance) score high. But it ignores:
  - Word order: "dog bites man" and "man bites dog" are identical
  - Word meaning: "happy" and "joyful" are treated as different words

Next step → Semantic Similarity, which understands meaning.
""")
