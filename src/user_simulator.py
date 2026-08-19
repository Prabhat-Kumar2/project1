"""
User simulator (SDD 9.4).

simulate_users() generates a population of synthetic users, each with a random
reading history of MIN_HISTORY..MAX_HISTORY article indices drawn (without
replacement) from the processed article set. The result is stored as
user_simulation.csv with columns: user_id, history_indices.
"""
import os
import random
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def simulate_users(
    processed_path: str = config.PROCESSED_NEWS_PATH,
    output_path: str = config.USER_SIMULATION_PATH,
    num_users: int = config.NUM_USERS,
    min_history: int = config.MIN_HISTORY,
    max_history: int = config.MAX_HISTORY,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create num_users synthetic users with randomised reading histories.

    Returns a DataFrame with columns user_id and history_indices (stored as a
    string representation of a list of integers).
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"processed_news.csv not found at '{processed_path}'. "
            f"Run preprocessing first."
        )

    articles = pd.read_csv(processed_path)
    n_articles = len(articles)
    if n_articles == 0:
        raise ValueError("processed_news.csv is empty; cannot simulate users.")

    rng = random.Random(seed)
    # A user cannot read more distinct articles than exist in the corpus.
    upper = min(max_history, n_articles)
    lower = min(min_history, upper)

    rows = []
    for i in range(1, num_users + 1):
        k = rng.randint(lower, upper)
        history = sorted(rng.sample(range(n_articles), k))
        rows.append({"user_id": f"User_{i}", "history_indices": str(history)})

    df = pd.DataFrame(rows)
    config.ensure_data_dir()
    df.to_csv(output_path, index=False)
    print(f"[user_simulator] Simulated {len(df)} users "
          f"(history {lower}-{upper} articles each) -> {output_path}")
    return df


if __name__ == "__main__":
    simulate_users()
