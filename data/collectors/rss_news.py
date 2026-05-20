"""
data/collectors/rss_news.py

Collects news headlines from RSS feeds and scores them with VADER sentiment.

RSS (Really Simple Syndication) is a standardized XML feed format.
No authentication required — news outlets publish them publicly.
feedparser handles the XML parsing; we do sentiment on the headlines.

Sources:
- Yahoo Finance (ticker-specific)
- Seeking Alpha (ticker-specific)
- Reuters Business (market-wide context)
- CNBC (market-wide context)
- MarketWatch (market-wide context)

Why RSS over scraping?
RSS feeds are explicitly published for machine consumption — no ToS violations,
no bot detection, no JavaScript rendering needed. The tradeoff is that RSS
feeds only carry headlines and summaries, not full article text. VADER on
headlines is less accurate than full-article NLP, but it's fast and free.
"""

import os
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime


import feedparser
import pandas as pd
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
load_dotenv()

from config.settings import DATA, FEATURES

_sia = SentimentIntensityAnalyzer()

# Threshold below which we flag as "highly negative"
NEGATIVE_THRESHOLD = -0.5


def _parse_date(entry) -> datetime | None:
    """Parse the published date from an RSS entry. Returns None if unparseable."""
    for field in ("published", "updated", "created"):
        val = entry.get(field)
        if val:
            try:
                return parsedate_to_datetime(val)
            except Exception:
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    pass
    return None


def _score_text(text: str) -> float:
    """Run VADER on a string, return compound score (-1 to +1)."""
    return _sia.polarity_scores(text)["compound"]


def fetch_ticker_news(ticker: str, days: int = None) -> list[dict]:
    """
    Fetch news articles for a specific ticker from RSS feeds.
    Returns a list of article dicts with title, date, sentiment, source.
    """
    days   = days or FEATURES["sentiment_short_window"]
    cutoff = datetime.utcnow() - timedelta(days=days)
    articles = []

    feeds = [
        f.format(ticker=ticker)
        for f in DATA["rss_feeds"]
    ]

    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:30]:
                pub_date = _parse_date(entry)

                # If we can't parse the date, include it anyway (conservative)
                if pub_date and pub_date.replace(tzinfo=None) < cutoff:
                    continue

                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                text    = title + " " + summary

                articles.append({
                    "title":     title,
                    "source":    feed_url,
                    "date":      pub_date,
                    "sentiment": _score_text(text),
                })
        except Exception as e:
            # RSS feeds go down, return empty XML, etc. — don't crash pipeline
            print(f"    RSS feed failed ({feed_url}): {e}")
            continue

    return articles


def fetch_market_news() -> list[dict]:
    """
    Fetch market-wide news from broad RSS feeds (Reuters, CNBC, MarketWatch).
    Used for overall market sentiment context — not ticker-specific.
    """
    broad_feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    ]
    articles = []
    for feed_url in broad_feeds:
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:20]:
                title = entry.get("title", "")
                articles.append({
                    "title":     title,
                    "source":    feed_url,
                    "date":      _parse_date(entry),
                    "sentiment": _score_text(title),
                })
        except Exception:
            continue
    return articles


def compute_news_features(ticker: str) -> dict:
    """
    Aggregate raw RSS articles into the news sentiment features
    defined in the spec and stored in sentiment_features.

    news_sentiment_momentum = 7d avg - 30d avg.
    Positive momentum means sentiment is improving recently.
    """
    # 7-day window (short)
    articles_7d  = fetch_ticker_news(ticker, days=7)
    # 30-day window (long) — re-fetch with wider window
    articles_30d = fetch_ticker_news(ticker, days=30)

    def _neutral():
        return {
            "news_volume_7d":          0,
            "news_volume_change":      0.0,
            "news_sentiment_avg_7d":   0.0,
            "news_sentiment_30d":      0.0,
            "news_sentiment_momentum": 0.0,
            "negative_news_flag":      0,
        }

    if not articles_7d and not articles_30d:
        return _neutral()

    scores_7d  = [a["sentiment"] for a in articles_7d]
    scores_30d = [a["sentiment"] for a in articles_30d]

    avg_7d  = sum(scores_7d)  / len(scores_7d)  if scores_7d  else 0.0
    avg_30d = sum(scores_30d) / len(scores_30d) if scores_30d else 0.0

    # Week-over-week volume change: compare 7d count to prior 7d
    # (articles_30d includes the 7d window, so prior_7d = 30d_count - 7d_count)
    prior_7d_count = max(0, len(articles_30d) - len(articles_7d))

    return {
        "news_volume_7d":          len(articles_7d),
        "news_volume_change":      float(len(articles_7d) - prior_7d_count),
        "news_sentiment_avg_7d":   round(avg_7d,  4),
        "news_sentiment_30d":      round(avg_30d, 4),
        "news_sentiment_momentum": round(avg_7d - avg_30d, 4),
        "negative_news_flag":      int(
            any(s < NEGATIVE_THRESHOLD for s in scores_7d)
        ),
    }
