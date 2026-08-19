# Personalised Content Recommendation using Embeddings

End-to-end news recommendation system (PGCP-BDA Group 13). It ingests the
Kaggle **Global News Dataset**, cleans and sentiment-tags articles, simulates a
population of 1000 readers, generates semantic embeddings, and serves
personalised top-N recommendations through a REST API and a Streamlit dashboard.

## Architecture

```
Kaggle CSV -> preprocessing -> sentiment -> processed_news.csv
                                         -> user_simulation.csv (1000 users)
processed_news.csv -> embeddings (all-MiniLM-L6-v2) -> embeddings.npy
Recommendation Engine (cosine similarity) -> FastAPI -> Streamlit dashboard
```

## Project layout

```
project/
├── config.py                # paths + tunable constants
├── main.py                  # offline pipeline orchestration
├── requirements.txt
├── archive/                 # raw Kaggle dataset (data.csv, rating.csv, raw-data.csv)
├── data/                    # generated artifacts
│   ├── processed_news.csv
│   ├── user_simulation.csv
│   └── embeddings.npy
├── src/
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   ├── user_simulator.py
│   └── recommendation.py
└── app/
    ├── api.py               # FastAPI REST service
    └── dashboard.py         # Streamlit UI
```

## Setup

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # first run only, for sentiment
```

## Run

1. **Offline pipeline** (build model artifacts):
   ```bash
   python main.py
   ```
2. **REST API**:
   ```bash
   uvicorn app.api:app --reload
   ```
   Interactive docs at http://127.0.0.1:8000/docs
3. **Dashboard** (in a second terminal):
   ```bash
   streamlit run app/dashboard.py
   ```

## Configuration

Edit `config.py` or set environment variables:

| Setting            | Default            | Meaning                                  |
|--------------------|--------------------|------------------------------------------|
| `RAW_NEWS_PATH`    | `archive/data.csv` | Raw Kaggle CSV to ingest                 |
| `NEWS_SAMPLE_SIZE` | `0`                | Articles kept after cleaning (`0` = use ALL ~53k) |
| `NUM_USERS`        | `1000`             | Number of simulated users                |
| `EMBEDDING_MODEL`  | `all-MiniLM-L6-v2` | Sentence-Transformer model               |

## API

`POST /recommend/?top_n=5`

```json
{ "history_indices": [3, 17, 42] }
```

Response:

```json
{ "recommendations": [ { "title": "...", "sentiment": "Positive", "category": "...", "score": 0.83 } ] }
```

> If `sentence-transformers`/`torch` cannot be loaded, the engine automatically
> falls back to TF-IDF + SVD embeddings so the pipeline still runs.
