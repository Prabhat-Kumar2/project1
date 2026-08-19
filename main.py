"""
Offline pipeline orchestration (SDD 9.9).

Runs the full offline pipeline in sequence:
    1. Data preprocessing        -> data/processed_news.csv
    2. Sentiment analysis        -> adds 'sentiment' column
    3. User simulation           -> data/user_simulation.csv
    4. Embedding generation      -> data/embeddings.npy  (cached)

After this completes the REST API and dashboard can be started.

Usage:
    python main.py            # run the pipeline
    python main.py --force    # also regenerate embeddings even if cached
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.preprocessing import clean_news_data
from src.sentiment_analysis import analyze_sentiment
from src.user_simulator import simulate_users
from src.recommendation import RecommendationEngine


def run_pipeline(force_embeddings: bool = False) -> None:
    print("=" * 70)
    print(" Personalised Content Recommendation - Offline Pipeline")
    print("=" * 70)

    # --- Stage 0: guard for the raw dataset ------------------------------- #
    if not os.path.exists(config.RAW_NEWS_PATH):
        print("\n[ERROR] Raw dataset not found at:")
        print(f"        {config.RAW_NEWS_PATH}")
        print("Place the Kaggle Global News dataset there, or set the "
              "RAW_NEWS_PATH environment variable, then re-run.")
        sys.exit(1)

    cap = "ALL records (full dataset)" if not config.NEWS_SAMPLE_SIZE or config.NEWS_SAMPLE_SIZE <= 0 else config.NEWS_SAMPLE_SIZE
    print(f"\nRaw dataset      : {config.RAW_NEWS_PATH}")
    print(f"Article cap      : {cap}")
    print(f"Simulated users  : {config.NUM_USERS}")
    print(f"Embedding model  : {config.EMBEDDING_MODEL}\n")

    # --- Stage 1: preprocessing ------------------------------------------ #
    print("[1/4] Preprocessing raw news data ...")
    clean_news_data()

    # --- Stage 2: sentiment ---------------------------------------------- #
    # Train the supervised sentiment model once (if labelled rating.csv exists
    # and no model is cached yet), then label articles with it.
    sentiment_model_path = os.path.join(config.DATA_DIR, "sentiment_model.joblib")
    rating_csv = os.path.join(config.ARCHIVE_DIR, "rating.csv")
    if not os.path.exists(sentiment_model_path) and os.path.exists(rating_csv):
        print("\n[2/4] Training supervised sentiment model (first run) ...")
        try:
            from train_sentiment import train_sentiment_model
            train_sentiment_model()
        except Exception as exc:
            print(f"[main] Sentiment training skipped ({exc}); will use TextBlob.")

    print("\n[2/4] Running sentiment analysis ...")
    analyze_sentiment()

    # --- Stage 3: user simulation ---------------------------------------- #
    print("\n[3/4] Simulating users ...")
    simulate_users()

    # --- Stage 4: embeddings --------------------------------------------- #
    print("\n[4/4] Generating / loading article embeddings ...")
    engine = RecommendationEngine()
    engine.generate_embeddings(force=force_embeddings)

    print("\n" + "=" * 70)
    print(" Pipeline complete. Artifacts written to:", config.DATA_DIR)
    print("   - processed_news.csv")
    print("   - user_simulation.csv")
    print("   - embeddings.npy")
    print("\n Next steps:")
    print("   1. Start the API      : uvicorn app.api:app --reload")
    print("   2. Start the dashboard: streamlit run app/dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline(force_embeddings="--force" in sys.argv)
