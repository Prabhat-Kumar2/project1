"""
Train a supervised sentiment classifier (accuracy improvement).

Instead of the generic TextBlob lexicon, we train a lightweight but strong
TF-IDF + Logistic Regression classifier on the LABELLED data in rating.csv
(its `title_sentiment` column is ground truth). This learns news-specific
sentiment patterns and markedly beats the lexicon baseline.

Outputs:
  data/sentiment_model.joblib    - the fitted pipeline (deployed for labelling)
  data/sentiment_metrics.json    - honest held-out test metrics (for evaluate.py)

Usage:  python train_sentiment.py
"""
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

LABELS = ["Positive", "Neutral", "Negative"]
MODEL_PATH = os.path.join(config.DATA_DIR, "sentiment_model.joblib")
METRICS_PATH = os.path.join(config.DATA_DIR, "sentiment_metrics.json")


def _load_labelled() -> pd.DataFrame:
    path = os.path.join(config.ARCHIVE_DIR, "rating.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"rating.csv not found at {path}; cannot train.")
    df = pd.read_csv(path, usecols=["title", "title_sentiment"], dtype=str, on_bad_lines="skip")
    df = df.dropna(subset=["title", "title_sentiment"])
    df["title"] = df["title"].str.strip()
    df["label"] = df["title_sentiment"].str.strip().str.capitalize()
    df = df[(df["title"] != "") & (df["label"].isin(LABELS))]
    return df[["title", "label"]]


def train_sentiment_model() -> dict:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    df = _load_labelled()
    print(f"[train_sentiment] Loaded {len(df):,} labelled titles "
          f"-> {df['label'].value_counts().to_dict()}")

    X, y = df["title"].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000, C=4.0, class_weight="balanced",
        )),
    ])

    print("[train_sentiment] Training TF-IDF + Logistic Regression ...")
    pipe.fit(X_tr, y_tr)

    preds = pipe.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_te, preds, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_te, preds, labels=LABELS)

    metrics = {
        "available": True,
        "method": "supervised (TF-IDF + Logistic Regression, held-out test)",
        "accuracy": round(acc, 4),
        "samples": int(len(y_te)),
        "per_class": {
            LABELS[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
    }

    # Refit on ALL data for deployment (more data = better labelling).
    pipe.fit(X, y)
    config.ensure_data_dir()
    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_sentiment] Held-out accuracy = {acc:.3f} on {len(y_te):,} titles")
    print(f"[train_sentiment] Saved model  -> {MODEL_PATH}")
    print(f"[train_sentiment] Saved metrics-> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_sentiment_model()
