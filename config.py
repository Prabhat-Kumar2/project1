"""
Central configuration for the Personalised Content Recommendation system.

All tunable constants and file paths live here so the pipeline, API and
dashboard share a single source of truth. The demonstration limits
(NEWS_SAMPLE_SIZE, NUM_USERS) are configurable and can be raised for larger
runs without any structural change to the code.
"""
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Raw Kaggle "Global News Dataset" input. The preprocessing module reads this
# file. Point RAW_NEWS_PATH at whichever raw CSV you want to ingest; by default
# it uses data.csv from the supplied archive folder.
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
RAW_NEWS_PATH = os.environ.get("RAW_NEWS_PATH", os.path.join(ARCHIVE_DIR, "data.csv"))

# Generated artefacts (the "data store").
PROCESSED_NEWS_PATH = os.path.join(DATA_DIR, "processed_news.csv")
USER_SIMULATION_PATH = os.path.join(DATA_DIR, "user_simulation.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# --------------------------------------------------------------------------- #
# Pipeline tunables
# --------------------------------------------------------------------------- #
# Number of valid articles kept after cleaning.
#   0 (default)  -> use the ENTIRE dataset (all records) for embeddings.
#   N > 0        -> cap to the first N valid records (fast demo runs).
# The SDD used 100 for fast iteration; set a small value here if you want a
# quick run instead of processing the full ~100k-article corpus.
NEWS_SAMPLE_SIZE = int(os.environ.get("NEWS_SAMPLE_SIZE", "0"))

# Synthetic user population.
NUM_USERS = int(os.environ.get("NUM_USERS", "1000"))
MIN_HISTORY = 3
MAX_HISTORY = 10

# Default number of recommendations returned.
TOP_N = 5

# Reproducibility.
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# Pre-trained Sentence-Transformer used for 384-dim article embeddings.
# If sentence-transformers / torch are unavailable the engine transparently
# falls back to a TF-IDF + SVD embedding so the pipeline always runs.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FALLBACK_EMBEDDING_DIM = 384

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_URL = os.environ.get("API_URL", f"http://{API_HOST}:{API_PORT}")


def ensure_data_dir() -> str:
    """Create the data/ directory if it does not exist and return its path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR
