"""
Recommendation engine (SDD 9.5 & 9.6).

The RecommendationEngine encapsulates:
  * model loading,
  * embedding generation + disk caching (embeddings.npy),
  * similarity-based top-N recommendation.

Primary embedding backend is the pre-trained Sentence-Transformer
'all-MiniLM-L6-v2' (384-dim). If sentence-transformers / torch are not
installed, the engine transparently falls back to a TF-IDF + Truncated SVD
embedding so the whole pipeline still runs end-to-end.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _cosine_similarity_matrix(profile: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single profile vector and every row of matrix."""
    profile = profile.astype(np.float32)
    matrix = matrix.astype(np.float32)
    p_norm = np.linalg.norm(profile) + 1e-10
    m_norm = np.linalg.norm(matrix, axis=1) + 1e-10
    return (matrix @ profile) / (m_norm * p_norm)


class RecommendationEngine:
    """Content-based recommendation engine over article embeddings."""

    def __init__(
        self,
        processed_path: str = config.PROCESSED_NEWS_PATH,
        embeddings_path: str = config.EMBEDDINGS_PATH,
        model_name: str = config.EMBEDDING_MODEL,
    ):
        if not os.path.exists(processed_path):
            raise FileNotFoundError(
                f"processed_news.csv not found at '{processed_path}'. "
                f"Run the offline pipeline (main.py) first."
            )
        self.processed_path = processed_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name

        # Load only the light columns needed for serving. The heavy 'text' body
        # is read lazily (see _corpus) and only when embeddings must be built,
        # so the API stays memory-efficient on a full ~50k+ article corpus.
        header = list(pd.read_csv(processed_path, nrows=0).columns)
        light = [c for c in ("title", "category", "url", "description", "sentiment") if c in header]
        self.df = pd.read_csv(processed_path, usecols=light)
        if "sentiment" not in self.df.columns:
            self.df["sentiment"] = "Neutral"
        self.embeddings: np.ndarray | None = None
        self.backend = None  # 'sentence-transformer' or 'tfidf'

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def _corpus(self) -> list[str]:
        """Concatenate title + text for every article (text read lazily)."""
        text = pd.read_csv(self.processed_path, usecols=["text"])["text"].astype(str)
        titles = self.df["title"].astype(str).reset_index(drop=True)
        text = text.reset_index(drop=True)
        return (titles + ". " + text).tolist()

    def _encode_sentence_transformer(self, corpus: list[str]) -> np.ndarray | None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None
        try:
            print(f"[recommendation] Loading Sentence-Transformer '{self.model_name}' ...")
            model = SentenceTransformer(self.model_name)
            emb = model.encode(
                corpus, batch_size=64, show_progress_bar=True, convert_to_numpy=True
            )
            self.backend = "sentence-transformer"
            return emb.astype(np.float32)
        except Exception as exc:  # network/model load failure
            print(f"[recommendation] Sentence-Transformer unavailable ({exc}); "
                  f"falling back to TF-IDF embeddings.")
            return None

    def _encode_tfidf(self, corpus: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("[recommendation] Building TF-IDF + SVD fallback embeddings ...")
        tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
        matrix = tfidf.fit_transform(corpus)
        dim = min(config.FALLBACK_EMBEDDING_DIM, matrix.shape[1] - 1, max(matrix.shape[0] - 1, 1))
        dim = max(dim, 2)
        svd = TruncatedSVD(n_components=dim, random_state=config.RANDOM_SEED)
        emb = svd.fit_transform(matrix)
        self.backend = "tfidf"
        return emb.astype(np.float32)

    def generate_embeddings(self, force: bool = False) -> np.ndarray:
        """
        Load embeddings from cache, or generate and cache them.

        Regenerates when the cache is missing, `force` is True, or the cached
        row count no longer matches the number of articles (stale cache).
        """
        if not force and os.path.exists(self.embeddings_path):
            cached = np.load(self.embeddings_path)
            if cached.shape[0] == len(self.df):
                self.embeddings = cached.astype(np.float32)
                self.backend = self.backend or "cache"
                print(f"[recommendation] Loaded cached embeddings "
                      f"{self.embeddings.shape} from {self.embeddings_path}")
                return self.embeddings
            print("[recommendation] Cached embeddings are stale "
                  f"(cache={cached.shape[0]} rows, corpus={len(self.df)}); regenerating.")

        corpus = self._corpus()
        emb = self._encode_sentence_transformer(corpus)
        if emb is None:
            emb = self._encode_tfidf(corpus)

        self.embeddings = emb
        config.ensure_data_dir()
        np.save(self.embeddings_path, emb)
        print(f"[recommendation] Generated embeddings {emb.shape} "
              f"(backend={self.backend}) -> {self.embeddings_path}")
        return emb

    def _ensure_embeddings(self) -> np.ndarray:
        if self.embeddings is None:
            self.generate_embeddings()
        return self.embeddings

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _short_description(self, row) -> str:
        """Return a brief (<=300 char) description for an article row."""
        desc = row.get("description", "")
        if not isinstance(desc, str) or not desc.strip():
            desc = str(row.get("text", ""))
        desc = desc.strip()
        return desc[:300] + ("..." if len(desc) > 300 else "")

    def _article_payload(self, idx: int) -> dict:
        """Base recommendation payload (title, sentiment, category, url, desc)."""
        row = self.df.iloc[idx]
        url = row.get("url", "")
        url = str(url).strip() if isinstance(url, str) or pd.notna(url) else ""
        return {
            "title": str(row["title"]),
            "sentiment": str(row.get("sentiment", "Neutral")),
            "category": str(row.get("category", "General")),
            "url": url if url and url.lower() != "nan" else "",
            "description": self._short_description(row),
        }

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #
    def get_top_n_recommendations(
        self, history_indices: list[int], top_n: int = config.TOP_N
    ) -> list[dict]:
        """
        Return top-N recommendations for a user given their reading history.

        Steps: fetch history embeddings -> mean-pool into a profile vector ->
        cosine similarity vs all articles -> rank -> drop already-seen -> top-N.
        Each item is {"title": str, "sentiment": str}.
        """
        embeddings = self._ensure_embeddings()
        n = embeddings.shape[0]

        # Keep only valid indices.
        valid = [i for i in history_indices if isinstance(i, (int, np.integer)) and 0 <= i < n]
        if not valid:
            # Cold-start fallback: return the first top_n articles.
            return [self._article_payload(i) for i in range(min(top_n, n))]

        profile = embeddings[valid].mean(axis=0)
        scores = _cosine_similarity_matrix(profile, embeddings)

        # Rank descending, exclude seen articles.
        order = np.argsort(-scores)
        seen = set(valid)
        recommendations = []
        for idx in order:
            if idx in seen:
                continue
            payload = self._article_payload(int(idx))
            payload["score"] = round(float(scores[idx]), 4)
            recommendations.append(payload)
            if len(recommendations) >= top_n:
                break
        return recommendations

    # ------------------------------------------------------------------ #
    # New-user (cold-start) recommendation from an interest profile
    # ------------------------------------------------------------------ #
    def category_counts(self) -> dict:
        """Return {category: article_count} sorted by frequency (desc)."""
        return self.df["category"].astype(str).value_counts().to_dict()

    def recommend_for_preferences(
        self,
        categories: list[str] | None = None,
        preferred_sentiment: str = "Any",
        interest_focus: float = 0.7,
        sentiment_weight: float = 0.3,
        top_n: int = config.TOP_N,
        exclude_indices: list[int] | None = None,
    ) -> list[dict]:
        """
        Recommend articles for a brand-new user with no reading history, using
        a self-declared interest profile instead of past behaviour.

        Parameters
        ----------
        categories : areas of interest selected by the user. Their articles are
            mean-pooled into a topical profile vector.
        preferred_sentiment : 'Positive' | 'Neutral' | 'Negative' | 'Any'.
        interest_focus : weight (0..1) on topical similarity to the chosen areas.
        sentiment_weight : weight (0..1) on matching the preferred sentiment.
        top_n : number of items to return.
        exclude_indices : optional article indices to skip.

        Score = (interest_focus * topic_similarity + sentiment_weight *
        sentiment_match) / (interest_focus + sentiment_weight), with a small
        bonus for articles that literally belong to a chosen category.
        """
        embeddings = self._ensure_embeddings()
        df = self.df
        n = len(df)
        exclude = set(exclude_indices or [])

        cats = [str(c) for c in (categories or []) if str(c).strip()]
        cat_series = df["category"].astype(str)

        # --- Topical profile from chosen categories ---------------------- #
        if cats:
            idx = np.where(cat_series.isin(cats).values)[0]
            profile = embeddings[idx].mean(axis=0) if len(idx) else embeddings.mean(axis=0)
        else:
            profile = embeddings.mean(axis=0)

        sim = _cosine_similarity_matrix(profile, embeddings)      # [-1, 1]
        sim_norm = (sim + 1.0) / 2.0                              # [0, 1]

        # --- Sentiment preference component ------------------------------ #
        sent_series = df["sentiment"].astype(str)
        if preferred_sentiment and preferred_sentiment != "Any":
            sent_match = (sent_series.values == preferred_sentiment).astype(np.float32)
        else:
            sent_match = np.full(n, 0.5, dtype=np.float32)

        cat_match = cat_series.isin(cats).values.astype(np.float32) if cats else np.zeros(n, np.float32)

        wf = max(float(interest_focus), 0.0)
        ws = max(float(sentiment_weight), 0.0)
        total = wf + ws
        if total <= 0:
            final = sim_norm.copy()
        else:
            final = (wf * sim_norm + ws * sent_match) / total
        final = final + 0.05 * cat_match  # nudge exact-category articles up

        order = np.argsort(-final)
        specific = preferred_sentiment not in (None, "", "Any")
        recs: list[dict] = []
        for i in order:
            if i in exclude:
                continue
            payload = self._article_payload(int(i))
            payload["score"] = round(float(final[i]), 4)
            payload["topic_match"] = round(float(sim_norm[i]), 4)
            payload["sentiment_match"] = bool(sent_match[i] >= 1.0) if specific else None
            recs.append(payload)
            if len(recs) >= top_n:
                break
        return recs

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #
    def analytics_summary(self, top_k_categories: int = 10) -> dict:
        """Server-side analytics over the processed corpus."""
        df = self.df
        sentiment_counts = df["sentiment"].astype(str).value_counts().to_dict()
        category_counts = df["category"].astype(str).value_counts().head(20).to_dict()

        top_cats = list(df["category"].astype(str).value_counts().head(top_k_categories).index)
        sub = df[df["category"].astype(str).isin(top_cats)]
        crosstab = pd.crosstab(sub["category"].astype(str), sub["sentiment"].astype(str))
        sentiment_by_category = {
            cat: {sent: int(crosstab.loc[cat, sent]) for sent in crosstab.columns}
            for cat in crosstab.index
        }

        return {
            "articles": int(len(df)),
            "categories": int(df["category"].nunique()),
            "sentiment_counts": {k: int(v) for k, v in sentiment_counts.items()},
            "category_counts": {k: int(v) for k, v in category_counts.items()},
            "sentiment_by_category": sentiment_by_category,
        }


if __name__ == "__main__":
    engine = RecommendationEngine()
    engine.generate_embeddings()
    print("Sample recommendations for history [0, 1, 2]:")
    for rec in engine.get_top_n_recommendations([0, 1, 2], top_n=5):
        print(" -", rec)
