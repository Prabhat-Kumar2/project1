"""
Data preprocessing (SDD 9.2).

clean_news_data() loads the raw Kaggle news CSV, standardises the text column,
validates required fields, removes missing/duplicate records, caps the corpus
to a configurable size, normalises whitespace and persists processed_news.csv.
"""
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Candidate columns (in priority order) that may hold the article body. The
# first one present in the raw file is mapped to the standard 'text' column.
_TEXT_COLUMN_CANDIDATES = ["full_content", "content", "article", "article_body", "description"]

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_whitespace(value: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""
    if not isinstance(value, str):
        value = str(value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def _vclean(series: pd.Series) -> pd.Series:
    """Vectorised whitespace normalisation (fast for large corpora)."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(_WHITESPACE_RE, " ", regex=True)
        .str.strip()
    )


def clean_news_data(
    raw_path: str = config.RAW_NEWS_PATH,
    output_path: str = config.PROCESSED_NEWS_PATH,
    sample_size: int = config.NEWS_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Clean the raw news dataset and write processed_news.csv.

    When ``sample_size`` is 0 (or None/negative) the ENTIRE dataset is used;
    otherwise the corpus is capped to the first ``sample_size`` valid records.

    Returns the cleaned DataFrame (title, text, category, url, description).
    Raises FileNotFoundError if the raw dataset is missing.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_path}'. Place the Kaggle Global "
            f"News dataset there or set the RAW_NEWS_PATH environment variable."
        )

    use_all = not sample_size or sample_size <= 0

    # --- Inspect the header to pick only the columns we actually need ----- #
    header_cols = list(pd.read_csv(raw_path, nrows=0).columns)
    text_source = next((c for c in _TEXT_COLUMN_CANDIDATES if c in header_cols), None)
    if text_source is None:
        raise ValueError(
            "Raw dataset has no recognised content column "
            f"(expected one of {_TEXT_COLUMN_CANDIDATES})."
        )
    if "title" not in header_cols:
        raise ValueError("Raw dataset must contain a 'title' column.")

    usecols = ["title", text_source]
    for extra in ("category", "url", "description"):
        if extra in header_cols and extra not in usecols:
            usecols.append(extra)

    # --- Read (full file for all-data mode, else a bounded buffer) -------- #
    read_kwargs = dict(usecols=usecols, dtype=str, on_bad_lines="skip")
    if use_all:
        print(f"[preprocessing] Reading FULL dataset (all records) from {raw_path} ...")
        df = pd.read_csv(raw_path, **read_kwargs)
    else:
        # Buffer leaves head-room for rows dropped during cleaning/de-dup.
        read_rows = max(sample_size * 6, 5000)
        df = pd.read_csv(raw_path, nrows=read_rows, **read_kwargs)

    # --- Standardise the text column -------------------------------------- #
    if text_source != "text":
        df = df.rename(columns={text_source: "text"})

    if "category" not in df.columns:
        df["category"] = "General"
    if "url" not in df.columns:
        df["url"] = ""
    if "description" not in df.columns:
        df["description"] = ""

    df = df[["title", "text", "category", "url", "description"]]

    # --- Drop missing and duplicate records (vectorised) ------------------ #
    df = df.dropna(subset=["title", "text"]).copy()
    df["title"] = _vclean(df["title"])
    df["text"] = _vclean(df["text"])
    df["category"] = _vclean(df["category"]).replace("", "General")
    df["url"] = _vclean(df["url"])
    df["description"] = _vclean(df["description"])

    # Remove rows that became empty after normalisation.
    df = df[(df["title"] != "") & (df["text"] != "")]
    df = df.drop_duplicates(subset=["title"])

    # Fall back to a snippet of the article body when no description exists.
    empty_desc = df["description"] == ""
    df.loc[empty_desc, "description"] = df.loc[empty_desc, "text"].str.slice(0, 300)

    # --- Cap the corpus (skipped in all-data mode) ------------------------ #
    if not use_all:
        df = df.head(sample_size)
    df = df.reset_index(drop=True)

    # --- Persist ---------------------------------------------------------- #
    config.ensure_data_dir()
    df.to_csv(output_path, index=False)
    scope = "ALL records" if use_all else f"first {sample_size}"
    print(f"[preprocessing] Wrote {len(df):,} cleaned articles ({scope}) -> {output_path}")
    return df


if __name__ == "__main__":
    clean_news_data()
