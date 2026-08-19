"""
Streamlit dashboard (SDD 9.8 & 10.2).

Primary user-facing interface. It offers three modes:

  * Existing User   - pick a simulated user, inspect their reading history and
                      get behaviour-based recommendations (via the REST API).
  * New User        - a brand-new visitor picks their favourite topics and the
                      kind of news they enjoy, and gets cold-start
                      recommendations tailored to that profile.
  * Insights        - plain-language, interactive analytics about the news.

Backend connection failures are surfaced with a clear message rather than
crashing the app.

Run:  streamlit run app/dashboard.py
"""
import ast
import json
import os
import sys
from collections import Counter

import pandas as pd
import requests
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Personalised Content Recommendation", page_icon="📰", layout="wide")

SENTIMENT_COLOURS = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

# Plain-language names for the sentiment labels, so non-technical users get it.
FRIENDLY_SENTIMENT = {
    "Positive": "🟢 Positive (upbeat)",
    "Neutral": "⚪ Neutral (factual)",
    "Negative": "🔴 Negative (concerning)",
}


def friendly_index(series: pd.Series) -> pd.Series:
    """Rename a Series index of sentiment labels to plain-language names."""
    return series.rename(index={k: FRIENDLY_SENTIMENT.get(k, k) for k in series.index})


@st.cache_data
def load_data():
    users = pd.read_csv(config.USER_SIMULATION_PATH) if os.path.exists(config.USER_SIMULATION_PATH) else None
    articles = None
    if os.path.exists(config.PROCESSED_NEWS_PATH):
        # Load only the light columns the UI needs (skip the heavy 'text' body)
        # so the dashboard stays responsive on a full ~100k-article corpus.
        header = pd.read_csv(config.PROCESSED_NEWS_PATH, nrows=0).columns
        wanted = [c for c in ("title", "category", "sentiment", "url", "description") if c in header]
        articles = pd.read_csv(config.PROCESSED_NEWS_PATH, usecols=wanted)
    return users, articles


def parse_history(value) -> list[int]:
    try:
        return [int(x) for x in ast.literal_eval(str(value))]
    except Exception:
        return []


def sentiment_badge(sentiment: str) -> str:
    return f"{SENTIMENT_COLOURS.get(sentiment, '⚪')} {sentiment}"


def call_api(path: str, payload: dict, top_n: int):
    """POST to the API; returns (recs, error_message)."""
    try:
        resp = requests.post(
            f"{config.API_URL}{path}",
            params={"top_n": top_n},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("recommendations", []), None
    except requests.exceptions.ConnectionError:
        return None, (
            "Could not connect to the backend API. Start it with:\n\n"
            "`python -m uvicorn app.api:app --host 127.0.0.1 --port 8000`"
        )
    except Exception as exc:
        return None, f"Request failed: {exc}"


def render_recommendations(recs: list[dict], title: str) -> None:
    if not recs:
        st.warning("No recommendations returned.")
        return
    st.success(f"{title} ({len(recs)} items)")
    for i, rec in enumerate(recs, start=1):
        url = (rec.get("url") or "").strip()
        heading = f"[{rec['title']}]({url})" if url else rec["title"]

        with st.container(border=True):
            st.markdown(f"**{i}. {heading}**")

            description = (rec.get("description") or "").strip()
            if description:
                st.caption(description)

            bits = [
                sentiment_badge(rec.get("sentiment", "Neutral")),
                f"_{rec.get('category', 'General')}_",
            ]
            # Show a friendly "match" percentage instead of raw internal scores.
            match = rec.get("topic_match")
            if match is None:
                match = rec.get("score")
            if match is not None:
                pct = max(0, min(100, round(float(match) * 100)))
                bits.append(f"🎯 {pct}% match")
            if rec.get("sentiment_match") is True:
                bits.append("✅ matches your mood")
            st.markdown(" · ".join(bits))

            if url:
                st.markdown(f"🔗 [Read full article]({url})")


# ====================================================================== #
# Tabs
# ====================================================================== #
def tab_existing_user(users: pd.DataFrame, articles: pd.DataFrame, top_n: int) -> None:
    st.subheader("Existing user - history-based recommendations")
    user_id = st.selectbox("Select a simulated user", users["user_id"].tolist())
    history = parse_history(users.loc[users["user_id"] == user_id, "history_indices"].iloc[0])

    with st.expander(f"📖 View Reading History ({len(history)} articles)"):
        for idx in history:
            if 0 <= idx < len(articles):
                row = articles.iloc[idx]
                st.markdown(
                    f"- **[{idx}] {row['title']}**  \n"
                    f"  _{row.get('category', 'General')}_ · "
                    f"{sentiment_badge(str(row.get('sentiment', 'Neutral')))}"
                )

    if st.button("🚀 Generate Recommendations", type="primary", key="existing_btn"):
        with st.spinner("Contacting recommendation service ..."):
            recs, err = call_api("/recommend/", {"history_indices": history}, top_n)
        if err:
            st.error(f"❌ {err}")
        else:
            render_recommendations(recs, f"Top recommendations for {user_id}")


def tab_new_user(articles: pd.DataFrame, top_n: int) -> None:
    st.subheader("New here? Tell us what you like")
    st.caption("Pick your favourite topics and the kind of news you enjoy - we'll build your feed instantly.")

    cat_counts = articles["category"].astype(str).value_counts()
    cat_options = cat_counts.index.tolist()
    default_cats = cat_options[:3]

    with st.form("new_user_form"):
        interests = st.multiselect(
            "🎯 Topics you're interested in",
            options=cat_options,
            default=default_cats,
            help="Choose one or more topics you'd like to read about.",
        )
        preferred_sentiment = st.radio(
            "🙂 What kind of news do you prefer?",
            options=["Any", "Positive", "Neutral", "Negative"],
            horizontal=True,
            help="Positive = upbeat · Neutral = factual · Negative = serious/concerning · Any = a mix.",
        )
        submitted = st.form_submit_button("✨ Show my recommendations", type="primary")

    if submitted:
        if not interests:
            st.warning("Please pick at least one topic to get the best results.")
        # Sensible built-in weighting so the reader doesn't need to tune anything:
        # focus mostly on the chosen topics, and let mood matter only when the
        # reader actually expresses a preference.
        payload = {
            "categories": interests,
            "preferred_sentiment": preferred_sentiment,
            "interest_focus": 0.7,
            "sentiment_weight": 0.0 if preferred_sentiment == "Any" else 0.4,
        }
        with st.spinner("Finding news you'll like ..."):
            recs, err = call_api("/recommend/preferences", payload, top_n)
        if err:
            st.error(f"❌ {err}")
        else:
            topics = ", ".join(interests) if interests else "all topics"
            mood = "any mood" if preferred_sentiment == "Any" else f"{preferred_sentiment.lower()} news"
            st.info(f"Showing **{mood}** about **{topics}**")
            render_recommendations(recs, "Picked for you")


def tab_analytics(users: pd.DataFrame, articles: pd.DataFrame) -> None:
    st.subheader("📊 Insights — a simple look at the news")
    st.caption("An easy, plain-English summary of what's in the news collection and what readers look at.")

    n_articles = len(articles)
    n_users = len(users)
    n_topics = articles["category"].nunique()
    avg_hist = users["history_indices"].map(lambda v: len(parse_history(v))).mean()
    cat_col = articles["category"].astype(str)

    # --- Headline numbers ------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📰 News articles", f"{n_articles:,}",
              help="Total number of news stories in the collection.")
    c2.metric("👥 Sample readers", f"{n_users:,}",
              help="Simulated readers used to test the recommendations.")
    c3.metric("🏷️ Topics", f"{n_topics:,}",
              help="Different subject areas the news covers.")
    c4.metric("📖 Stories per reader", f"{avg_hist:.1f}",
              help="On average, how many stories each sample reader has looked at.")

    st.divider()

    # --- Mood of the news ------------------------------------------------- #
    if "sentiment" in articles.columns:
        st.markdown("### 🙂 How does the news feel?")
        st.caption("Every article is tagged by its tone. Here's the overall mix.")
        counts = articles["sentiment"].value_counts()
        total = int(counts.sum()) or 1
        m1, m2, m3 = st.columns(3)
        for col, key, label in [
            (m1, "Positive", "🟢 Positive / upbeat"),
            (m2, "Neutral", "⚪ Neutral / factual"),
            (m3, "Negative", "🔴 Negative / concerning"),
        ]:
            val = int(counts.get(key, 0))
            col.metric(label, f"{val:,}", f"{val / total * 100:.0f}% of all news")
        st.bar_chart(friendly_index(counts))

    st.divider()

    # --- Explore a topic (interactive) ------------------------------------ #
    st.markdown("### 🔍 Explore a topic")
    st.caption("Choose a topic to see how many stories it has, how it feels, and a few example headlines.")
    topic_list = cat_col.value_counts().index.tolist()
    chosen = st.selectbox("Pick a topic", topic_list)
    sub = articles[cat_col == chosen]

    left, right = st.columns([1, 2])
    with left:
        st.metric(f"Stories about “{chosen}”", f"{len(sub):,}")
        if "sentiment" in sub.columns and len(sub):
            top_mood = sub["sentiment"].value_counts().idxmax()
            st.write(f"Usual tone: **{FRIENDLY_SENTIMENT.get(top_mood, top_mood)}**")
    with right:
        if "sentiment" in sub.columns and len(sub):
            st.bar_chart(friendly_index(sub["sentiment"].value_counts()))

    with st.expander(f"See example headlines about “{chosen}”"):
        for _, row in sub.head(5).iterrows():
            url = str(row.get("url", "")).strip()
            title = row["title"]
            line = f"[{title}]({url})" if url and url.lower() != "nan" else title
            st.markdown(f"- {line}  \n  {sentiment_badge(str(row.get('sentiment', 'Neutral')))}")

    st.divider()

    # --- Most common topics (interactive) --------------------------------- #
    st.markdown("### 🏆 Most common topics")
    st.caption("The subject areas with the most news stories.")
    how_many = st.slider("How many topics to show?", 5, 25, 10, key="common_topics")
    st.bar_chart(articles["category"].value_counts().head(how_many))

    st.divider()

    # --- What readers look at most ---------------------------------------- #
    st.markdown("### 👀 What do readers look at most?")
    st.caption("Topics that show up most often in the sample readers' reading history.")
    counter: Counter = Counter()
    for v in users["history_indices"]:
        for idx in parse_history(v):
            if 0 <= idx < n_articles:
                counter[cat_col.iloc[idx]] += 1
    if counter:
        how_many2 = st.slider("How many to show?", 5, 25, 10, key="read_topics")
        st.bar_chart(pd.Series(dict(counter.most_common(how_many2))))

    st.divider()

    # --- How much do readers read ----------------------------------------- #
    st.markdown("### 📚 How much do readers read?")
    st.caption("How many readers have looked at 3 stories, 4 stories, and so on.")
    lengths = users["history_indices"].map(lambda v: len(parse_history(v)))
    dist = lengths.value_counts().sort_index()
    dist.index = [f"{i} stories" for i in dist.index]
    st.bar_chart(dist)


def tab_accuracy() -> None:
    st.subheader("✅ How good is the model?")
    st.caption("Simple accuracy checks that show how well the system understands the news.")

    report_path = os.path.join(config.DATA_DIR, "evaluation_report.json")
    if not os.path.exists(report_path):
        st.info(
            "No accuracy report yet. Run this once to generate it:\n\n"
            "`python evaluate.py`"
        )
        return

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    st.caption(f"Based on {rep.get('corpus_articles', 0):,} articles.")

    # --- Topic understanding (embedding retrieval) ----------------------- #
    er = rep.get("embedding_retrieval", {})
    if er:
        st.markdown("### 🧠 Does it understand topics?")
        st.caption(
            "We take an article and find the 5 most similar ones. This score is how "
            "often those matches are about the **same topic**. Higher is better."
        )
        p5 = er.get("precision@5", 0) * 100
        base = er.get("random_baseline", 0) * 100
        lift = (er.get("precision@5", 0) / er["random_baseline"]) if er.get("random_baseline") else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Top-5 topic match", f"{p5:.0f}%")
        c2.metric("Random guessing", f"{base:.1f}%", help="What you'd get by picking articles at random.")
        c3.metric("Better than random", f"{lift:.0f}x", help="How many times better than random guessing.")
        st.progress(min(er.get("precision@5", 0), 1.0))

    st.divider()

    # --- Recommendation quality ------------------------------------------ #
    rq = rep.get("recommendation_quality", {})
    if rq:
        st.markdown("### 🎯 Are the recommendations on-topic and varied?")
        cc = (rq.get("category_consistency") or 0) * 100
        base = (rq.get("consistency_baseline") or 0) * 100
        div = (rq.get("diversity@N") or 0) * 100
        cov = (rq.get("catalogue_coverage") or 0) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("On-topic picks", f"{cc:.0f}%",
                  help=f"Share of recommendations matching the reader's topics (random ≈ {base:.0f}%).")
        c2.metric("Variety", f"{div:.0f}%",
                  help="How mixed the recommended topics are (higher = more varied).")
        c3.metric("Catalogue reach", f"{cov:.0f}%",
                  help="How much of the whole news collection the engine can surface.")

    st.divider()

    # --- Sentiment accuracy ---------------------------------------------- #
    s = rep.get("sentiment", {})
    if s.get("available"):
        st.markdown("### 🙂 Is the mood label correct?")
        st.caption(
            "We compare our upbeat / factual / concerning labels against the dataset's "
            "own labelled answers. This is the share we get right."
        )
        c1, c2 = st.columns(2)
        c1.metric("Mood accuracy", f"{s.get('accuracy', 0) * 100:.0f}%")
        c2.metric("Checked on", f"{s.get('samples', 0):,} headlines")

        pc = s.get("per_class", {})
        if pc:
            friendly = {"Positive": "🟢 Positive", "Neutral": "⚪ Neutral", "Negative": "🔴 Negative"}
            table = pd.DataFrame(
                {
                    "Mood": [friendly.get(k, k) for k in pc],
                    "Correct when predicted (precision)": [f"{v['precision']*100:.0f}%" for v in pc.values()],
                    "Found (recall)": [f"{v['recall']*100:.0f}%" for v in pc.values()],
                    "Examples": [v["support"] for v in pc.values()],
                }
            )
            st.dataframe(table, hide_index=True, use_container_width=True)
    elif s:
        st.info(f"Sentiment accuracy not available: {s.get('reason', 'unknown')}")

    st.caption("Re-run `python evaluate.py` any time to refresh these numbers.")


# ====================================================================== #
# Main
# ====================================================================== #
def main() -> None:
    st.title("📰 Personalised Content Recommendation")
    st.caption("Content-based news recommendations using sentence embeddings")

    users, articles = load_data()
    if users is None or articles is None:
        st.error(
            "Pipeline artifacts not found. Run `python main.py` first to generate "
            "processed_news.csv and user_simulation.csv."
        )
        return

    st.sidebar.header("Settings")
    top_n = st.sidebar.slider("Number of recommendations", 1, 20, config.TOP_N)
    st.sidebar.markdown(f"**API endpoint:** `{config.API_URL}`")
    st.sidebar.caption("Start the API with:\n\n`python -m uvicorn app.api:app --port 8000`")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Existing User", "🆕 New User", "📊 Insights", "✅ Accuracy"]
    )
    with tab1:
        tab_existing_user(users, articles, top_n)
    with tab2:
        tab_new_user(articles, top_n)
    with tab3:
        tab_analytics(users, articles)
    with tab4:
        tab_accuracy()


if __name__ == "__main__":
    main()
