# Similarity

Two Python scripts that explain **how computers measure similarity between texts** — using real song lyrics as data.

1. **Bag of Words** — counts words, ignores meaning
2. **Semantic Similarity** — pretends to understand meaning using AI embeddings

---

## Setup

**Requirements:** Python 3.12, a terminal

```bash
# Clone / open this folder, then create the virtual environment
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Script 1 — Bag of Words

**File:** `bag_of_words_similarity.py`

### What it teaches

A **bag of words** represents a text as a set of word counts — it throws away grammar and word order and just asks: *which words appear, and how often?*

```
"I love love music"  →  {"love": 2, "music": 1, "i": 1}
"I love songs"       →  {"love": 1, "songs": 1, "i": 1}
```

Two texts are *similar* if they use the same words with similar frequencies. We measure this with **cosine similarity** (0 = nothing in common, 1 = identical).

### What the script shows

| Step | What happens |
|------|-------------|
| 1 | Load lyrics from CSV |
| 2 | Build the Bag of Words matrix — vocabulary size, top words |
| 3 | Show word count vectors for two songs side by side |
| 4 | Find the 5 most similar songs for a query song |

### How to run

```bash
# Default — uses the Droplingizer playlist
python bag_of_words_similarity.py

# Use a different lyrics CSV
python bag_of_words_similarity.py path/to/other_lyrics.csv
```

### Limitations

Bag of Words misses two important things:
- **Word order** — "dog bites man" and "man bites dog" are identical
- **Word meaning** — "happy" and "joyful" are treated as completely different words

That's where Script 2 comes in.

---

## Script 2 — Semantic Similarity

**File:** `semantic_similarity.py`

### What it teaches

Instead of counting words, **semantic similarity** asks: *what does this text mean?*

It uses OpenAI's embedding model (`text-embedding-3-small`) to convert each lyric into a **vector of 1536 numbers** — a point in a high-dimensional space where similar meanings cluster together.

```
"I'm so happy today"   →  [0.21, -0.54, 0.88, ...]
"I feel very joyful"   →  [0.19, -0.51, 0.85, ...]  ← close → similar!
"The car needs repair" →  [-0.73, 0.12, -0.34, ...]  ← far → dissimilar
```

Key insight: **one lyric = one API call = one vector**, regardless of how long the text is. A single word and ten pages of lyrics both return 1536 numbers. The model reads the full text before producing that vector, so context and word order influence the result.

> **Cost note:** Each lyric is embedded once and cached locally (`.embeddings.pkl` next to your CSV). Re-running the script is free — you only pay for new songs.

### What the script shows

| Step | What happens |
|------|-------------|
| 1 | Load lyrics from CSV |
| 2 | Generate embeddings via OpenAI API (cached after first run) |
| 3 | Compare specific song pairs — see how semantic scores differ from BoW |
| 4 | Find the 5 most semantically similar songs for query songs |
| 5 | Artist comparison — cross-compare every song between the two artists with the most songs |
| 6 | Most similar and most different pairs across the entire dataset |
| 7 | Custom query — type any phrase and find matching songs |

### How to run

```bash
# Default — runs built-in queries
python semantic_similarity.py

# Pass your own query phrases
python semantic_similarity.py --query "heartbreak and rain" "summer party" "missing someone"

# Use a different CSV
python semantic_similarity.py path/to/other_lyrics.csv --query "lost and alone"
```

To set your own default queries **without using the CLI**, open `semantic_similarity.py` and edit the `CUSTOM_QUERIES` list near the bottom of the file:

```python
CUSTOM_QUERIES = [
    "heartbreak and rain",
    "fun summer party",
    # add your own here
]
```

### Setup: OpenAI API key

The script needs an OpenAI API key. Create a `.env` file in this folder:

```
OPENAI_API_KEY=sk-...your key here...
```

---

## Input data format

Both scripts expect a CSV with at least these columns:

| Column | Description |
|--------|-------------|
| `song_title` | Title of the song |
| `primary_artist` | Main artist name |
| `plain_lyrics` | Full lyrics as plain text |
| `lyrics_found` | Boolean — rows where this is `False` are skipped |

The default dataset (`input-data/droplingizer_lyrics.csv`) is a Droplingizer playlist with ~150 songs that have lyrics.

---

## Bag of Words vs Semantic — when does it matter?

| | Bag of Words | Semantic |
|---|---|---|
| "happy" ≈ "joyful" | ❌ different words | ✅ similar meaning |
| Speed | Fast, no API needed | Slower, needs OpenAI |
| Explainability | Easy — show shared words | Hard — vector is a black box |
| Cross-language | ❌ | ✅ (partially) |
| Cost | Free | ~$0.00002 per 1K tokens |

Both approaches use **cosine similarity** to compare vectors — the difference is entirely in what the vectors represent.
