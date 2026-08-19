"""
Sentiment analysis (SDD 9.3).

analyze_sentiment() computes a TextBlob polarity score for each article's text
and maps it to a categorical label:

    polarity > 0.1                -> Positive
    -0.1 <= polarity <= 0.1       -> Neutral
    polarity < -0.1               -> Negative

The resulting 'sentiment' column is merged back into processed_news.csv.
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# TextBlob polarity is scored on the first N characters of each article so that
# sentiment tagging stays fast even on a full ~100k-article corpus (lexicon
# scoring cost grows with text length; the opening of a news article carries the
# dominant sentiment signal).
SENTIMENT_TEXT_CHARS = 500

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except Exception:  # pragma: no cover - handled gracefully at runtime
    _HAS_TEXTBLOB = False


def _polarity(text: str) -> float:
    """Return TextBlob polarity in [-1, 1]; 0.0 if TextBlob is unavailable."""
    if not _HAS_TEXTBLOB:
        return 0.0
    try:
        return TextBlob(str(text)[:SENTIMENT_TEXT_CHARS]).sentiment.polarity
    except Exception:
        return 0.0


def label_from_polarity(polarity: float) -> str:
    """Map a polarity score to a categorical sentiment label."""
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


_MODEL_PATH = os.path.join(config.DATA_DIR, "sentiment_model.joblib")


def _load_trained_model():
    """Load the supervised sentiment model if it has been trained, else None."""
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        import joblib
        return joblib.load(_MODEL_PATH)
    except Exception as exc:  # pragma: no cover
        print(f"[sentiment] Could not load trained model ({exc}); using TextBlob.")
        return None


def analyze_sentiment(
    processed_path: str = config.PROCESSED_NEWS_PATH,
) -> pd.DataFrame:
    """
    Add a 'sentiment' column to processed_news.csv and re-persist it.

    Uses the trained supervised classifier (data/sentiment_model.joblib) when
    available - it is markedly more accurate - and falls back to TextBlob's
    lexicon polarity otherwise.

    Returns the DataFrame with the sentiment column populated.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"processed_news.csv not found at '{processed_path}'. "
            f"Run preprocessing first."
        )

    df = pd.read_csv(processed_path)

    model = _load_trained_model()
    if model is not None:
        df["sentiment"] = model.predict(df["title"].astype(str).values)
        method = "trained model"
    else:
        if not _HAS_TEXTBLOB:
            print("[sentiment] WARNING: TextBlob not installed; labelling all "
                  "articles 'Neutral'. Install textblob for real sentiment labels.")
        df["sentiment"] = df["text"].map(lambda t: label_from_polarity(_polarity(t)))
        method = "TextBlob lexicon"

    df.to_csv(processed_path, index=False)
    counts = df["sentiment"].value_counts().to_dict()
    print(f"[sentiment] Labelled {len(df)} articles via {method} -> {counts}")
    return df


if __name__ == "__main__":
    analyze_sentiment()
