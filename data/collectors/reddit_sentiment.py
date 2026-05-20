"""
data/collectors/reddit_sentiment.py

Collects Reddit mentions and sentiment for each ticker.

NOTE ON API ACCESS:
Reddit significantly restricted API access in 2023. New accounts now require
manual approval via a support form. Until credentials are approved, this
collector degrades gracefully — it logs a warning and returns neutral sentiment
values so nothing downstream breaks.

When credentials become available:
1. Add REDDIT_CLIENT_ID and REDDIT_SECRET to .env
2. This collector activates automatically — no other changes needed

Subreddits monitored: r/investing, r/SecurityAnalysis, r/stocks, r/wallstreetbets
Sentiment engine: VADER (Valence Aware Dictionary and sEntiment Reasoner)

Why VADER over a transformer model (e.g. FinBERT)?
VADER is rule-based, runs instantly, and requires no GPU. FinBERT would give
more accurate financial sentiment but takes seconds per document — too slow
for processing thousands of Reddit posts daily on a laptop. The spec explicitly
calls out VADER/TextBlob as sufficient for headline-level sentiment.
"""

import os
from datetime import datetime, timedelta, date


from dotenv import load_dotenv
load_dotenv()

from config.settings import REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT, DATA, FEATURES

import nltk
# Download VADER lexicon if not already present
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


_sia = SentimentIntensityAnalyzer()


def _credentials_available() -> bool:
    return bool(REDDIT_CLIENT_ID and REDDIT_SECRET
                and REDDIT_CLIENT_ID != "your_client_id_here")


def _neutral_features() -> dict:
    """
    Return a dict of neutral/zero sentiment features.
    Used when Reddit credentials are unavailable.
    Zero-fill is the correct missing value strategy for sentiment:
    absence of data = neutral stance, not the prior day's sentiment.
    """
    return {
        "reddit_mention_count_7d":  0,
        "reddit_mention_change":    0.0,
        "reddit_sentiment_avg_7d":  0.0,
        "reddit_sentiment_std_7d":  0.0,
        "reddit_bullish_ratio":     0.0,
        "reddit_post_score_avg":    0.0,
        "reddit_comment_ratio":     0.0,
        "wsb_mention_flag":         0,
    }


def collect(ticker: str, days: int = None) -> dict:
    """
    Collect Reddit sentiment features for a ticker over the last N days.

    Returns a flat dict mapping feature names to values.
    These feed directly into the sentiment_features table.

    VADER compound score ranges from -1 (most negative) to +1 (most positive).
    Threshold for "bullish": compound > 0.05 (VADER's recommended neutral boundary).
    """
    days = days or FEATURES["sentiment_short_window"]

    if not _credentials_available():
        print(f"    Reddit: credentials not available — returning neutral sentiment")
        return _neutral_features()

    try:
        import praw
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

        subreddits  = DATA["reddit_subreddits"]
        now_ts      = datetime.utcnow()
        cutoff_ts   = (now_ts - timedelta(days=days)).timestamp()      # start of current window
        prev_cutoff = (now_ts - timedelta(days=days * 2)).timestamp()  # start of previous window
        mentions    = []

        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                # time_filter="month" gives ~30 days of history so we can compute
                # both the current window (last `days` days) and the prior window
                # (days*2 to days ago) for week-over-week change.
                for post in subreddit.search(ticker, time_filter="month", limit=200):
                    if post.created_utc < prev_cutoff:
                        continue
                    text = post.title + " " + (post.selftext or "")
                    sentiment = _sia.polarity_scores(text)
                    mentions.append({
                        "subreddit":          sub_name,
                        "created_utc":        post.created_utc,
                        "score":              post.score,
                        "num_comments":       post.num_comments,
                        "sentiment_compound": sentiment["compound"],
                    })
            except Exception as e:
                print(f"    Reddit r/{sub_name}: {e}")
                continue

        if not mentions:
            return _neutral_features()

        import pandas as pd
        df = pd.DataFrame(mentions)

        # Split into current window (last `days` days) and previous window
        current = df[df["created_utc"] >= cutoff_ts]
        previous = df[(df["created_utc"] >= prev_cutoff) & (df["created_utc"] < cutoff_ts)]
        prev_count = len(previous)

        return {
            "reddit_mention_count_7d":  len(current),
            "reddit_mention_change":    float(len(current) - prev_count),
            "reddit_sentiment_avg_7d":  float(current["sentiment_compound"].mean()) if len(current) else 0.0,
            "reddit_sentiment_std_7d":  float(current["sentiment_compound"].std())  if len(current) else 0.0,
            "reddit_bullish_ratio":     float((current["sentiment_compound"] > 0.05).mean()) if len(current) else 0.0,
            "reddit_post_score_avg":    float(current["score"].mean())        if len(current) else 0.0,
            "reddit_comment_ratio":     float(current["num_comments"].mean()) if len(current) else 0.0,
            "wsb_mention_flag":         int("wallstreetbets" in current["subreddit"].values) if len(current) else 0,
        }

    except Exception as e:
        print(f"    Reddit collect failed for {ticker}: {e}")
        return _neutral_features()
