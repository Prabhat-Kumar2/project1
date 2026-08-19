"""
Model accuracy / quality evaluation.

Because this is a content-based recommender with no real user click logs, we
evaluate the *model* on objective, ground-truth-backed signals:

  1. Embedding retrieval accuracy  (Precision@K, category as ground truth)
       "Given an article, are its nearest neighbours in the same topic?"
       Directly measures whether the embeddings capture topical meaning.

  2. Recommendation quality        (category consistency, diversity, coverage)
       Are the top-N recommendations on-topic vs the reader's history, are they
       varied, and how much of the catalogue can the engine surface?

  3. Sentiment accuracy            (vs labelled rating.csv title_sentiment)
       Real ground truth: compare our TextBlob labels to the dataset's own
       title_sentiment column (accuracy, per-class precision/recall, confusion).

Results are printed and saved to data/evaluation_report.json so the dashboard
can display them.

Usage:  python evaluate.py
"""
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.recommendation import RecommendationEngine


# --------------------------------------------------------------------------- #
# 1. Embedding retrieval accuracy
# --------------------------------------------------------------------------- #
def evaluate_embedding_retrieval(engine, sample=500, ks=(1, 5, 10), seed=config.RANDOM_SEED):
    emb = engine._ensure_embeddings().astype(np.float32)
    cats = engine.df["category"].astype(str).values
    n = emb.shape[0]
    sample = min(sample, n)

    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    rng = random.Random(seed)
    idxs = rng.sample(range(n), sample)

    maxk = max(ks)
    sims = norm[idxs] @ norm.T  # (sample, n)
    prec = {k: [] for k in ks}
    for r, i in enumerate(idxs):
        row = sims[r]
        row[i] = -np.inf  # exclude the query itself
        top = np.argpartition(-row, maxk)[:maxk]
        top = top[np.argsort(-row[top])]
        same = (cats[top] == cats[i]).astype(float)
        for k in ks:
            prec[k].append(float(same[:k].mean()))

    p = pd.Series(cats).value_counts(normalize=True).values
    random_baseline = float((p ** 2).sum())  # chance two random articles share a topic
    return {
        **{f"precision@{k}": round(float(np.mean(prec[k])), 4) for k in ks},
        "random_baseline": round(random_baseline, 4),
        "sampled_articles": sample,
    }


# --------------------------------------------------------------------------- #
# 2. Recommendation quality (category consistency, diversity, coverage)
# --------------------------------------------------------------------------- #
def evaluate_recommendations(engine, users, sample=300, top_n=10, seed=config.RANDOM_SEED):
    import ast

    cats = engine.df["category"].astype(str).values
    n = len(engine.df)
    rng = random.Random(seed)

    rows = users.sample(min(sample, len(users)), random_state=seed)
    consistency, diversity, baseline = [], [], []
    recommended_ids = set()

    prevalence = pd.Series(cats).value_counts(normalize=True).to_dict()

    for _, u in rows.iterrows():
        try:
            hist = [int(x) for x in ast.literal_eval(str(u["history_indices"]))]
        except Exception:
            continue
        hist = [h for h in hist if 0 <= h < n]
        if not hist:
            continue
        hist_cats = set(cats[hist])
        recs = engine.get_top_n_recommendations(hist, top_n=top_n)
        if not recs:
            continue
        rec_cats = [r.get("category", "General") for r in recs]
        consistency.append(np.mean([c in hist_cats for c in rec_cats]))
        diversity.append(len(set(rec_cats)) / len(rec_cats))
        baseline.append(sum(prevalence.get(c, 0.0) for c in hist_cats))
        for r in recs:
            recommended_ids.add(r["title"])

    return {
        "category_consistency": round(float(np.mean(consistency)), 4) if consistency else None,
        "consistency_baseline": round(float(np.mean(baseline)), 4) if baseline else None,
        "diversity@N": round(float(np.mean(diversity)), 4) if diversity else None,
        "catalogue_coverage": round(len(recommended_ids) / max(n, 1), 4),
        "top_n": top_n,
        "sampled_users": len(consistency),
    }


# --------------------------------------------------------------------------- #
# 3. Sentiment accuracy vs labelled rating.csv
# --------------------------------------------------------------------------- #
def evaluate_sentiment(sample=5000, seed=config.RANDOM_SEED):
    # Prefer the honest held-out metrics saved by the supervised trainer.
    metrics_path = os.path.join(config.DATA_DIR, "sentiment_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    path = os.path.join(config.ARCHIVE_DIR, "rating.csv")
    if not os.path.exists(path):
        return {"available": False, "reason": "rating.csv not found"}

    from src.sentiment_analysis import _HAS_TEXTBLOB, _polarity, label_from_polarity

    if not _HAS_TEXTBLOB:
        return {"available": False, "reason": "TextBlob not installed"}

    df = pd.read_csv(path, usecols=["title", "title_sentiment"], dtype=str, on_bad_lines="skip")
    df = df.dropna(subset=["title", "title_sentiment"])
    df["truth"] = df["title_sentiment"].str.strip().str.capitalize()
    df = df[df["truth"].isin(["Positive", "Neutral", "Negative"])]
    if len(df) > sample:
        df = df.sample(sample, random_state=seed)

    preds = df["title"].map(lambda t: label_from_polarity(_polarity(t)))
    truth = df["truth"]

    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    labels = ["Positive", "Neutral", "Negative"]
    acc = float(accuracy_score(truth, preds))
    prec, rec, f1, support = precision_recall_fscore_support(
        truth, preds, labels=labels, zero_division=0
    )
    cm = confusion_matrix(truth, preds, labels=labels)

    return {
        "available": True,
        "accuracy": round(acc, 4),
        "samples": int(len(df)),
        "per_class": {
            labels[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        },
        "labels": labels,
        "confusion_matrix": cm.tolist(),  # rows = truth, cols = predicted
    }


# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print(" Model Accuracy / Quality Evaluation")
    print("=" * 70)

    engine = RecommendationEngine()
    engine.generate_embeddings()
    users = pd.read_csv(config.USER_SIMULATION_PATH)

    report = {
        "corpus_articles": int(len(engine.df)),
        "embedding_backend": engine.backend,
    }

    print("\n[1/3] Embedding retrieval accuracy (topic as ground truth) ...")
    t = time.time()
    report["embedding_retrieval"] = evaluate_embedding_retrieval(engine)
    print(f"      done in {time.time()-t:.1f}s -> {report['embedding_retrieval']}")

    print("\n[2/3] Recommendation quality (consistency / diversity / coverage) ...")
    t = time.time()
    report["recommendation_quality"] = evaluate_recommendations(engine, users)
    print(f"      done in {time.time()-t:.1f}s -> {report['recommendation_quality']}")

    print("\n[3/3] Sentiment accuracy vs labelled rating.csv ...")
    t = time.time()
    report["sentiment"] = evaluate_sentiment()
    print(f"      done in {time.time()-t:.1f}s")
    s = report["sentiment"]
    if s.get("available"):
        print(f"      accuracy={s['accuracy']:.3f} on {s['samples']} labelled titles")
    else:
        print(f"      skipped: {s.get('reason')}")

    config.ensure_data_dir()
    out = os.path.join(config.DATA_DIR, "evaluation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f" Report saved -> {out}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    main()
