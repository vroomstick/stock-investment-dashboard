"""
data/feature_store/sentiment.py

Computes all sentiment features defined in spec Section 7D and stores
them in the sentiment_features table.

Three signal sources combined here:
  1. Reddit sentiment   (from data/collectors/reddit_sentiment.py)
  2. News RSS sentiment (from data/collectors/rss_news.py)
  3. SEC-derived signals: insider transactions + 8-K flags + activist filings
     (queried directly from sec_filings and insider_transactions tables)

Why merge these three here instead of in the collectors?
  Separation of concerns. Collectors fetch and parse raw data.
  The feature store's job is to aggregate raw signals into model features.
  This also lets us re-run just feature computation without re-fetching data.

Missing value strategy (per feature_config.yaml): zero-fill.
Sentiment features that can't be computed default to 0 / neutral — because
absence of news/Reddit activity is itself informative (low attention stock)
and zero is a reasonable neutral value for signed sentiment scores.
"""

from datetime import date, timedelta
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from data.collectors.reddit_sentiment import collect as collect_reddit
from data.collectors.rss_news import compute_news_features
from database.db import get_db, fetch_all, fetch_one


# ---------------------------------------------------------------------------
# SEC-derived sentiment signals
# These come from data already stored by the SEC EDGAR collector.
# We query the database — no new API calls needed.
# ---------------------------------------------------------------------------

def _insider_signals(stock_id: int, as_of_date: str) -> dict:
    """
    Compute insider sentiment features from the insider_transactions table.

    insider_net_sentiment = (buys - sells) / total_transactions
    Range: -1 (all selling) to +1 (all buying)

    Insider buying is the most reliable signal in the system:
    - Legal requirement to disclose within 2 business days
    - CEO buying $2M of their own stock with personal money = high conviction
    - Unlike sell signals (diversification, taxes), buys are always voluntary

    30-day window captures the most recent and relevant activity.
    """
    cutoff = (
        date.fromisoformat(as_of_date) - timedelta(days=30)
    ).isoformat()

    rows = fetch_all(
        """SELECT transaction_type, total_value
           FROM insider_transactions
           WHERE stock_id = ?
             AND transaction_date >= ?""",
        (stock_id, cutoff)
    )

    buys  = [r for r in rows if r["transaction_type"] == "P"]
    sells = [r for r in rows if r["transaction_type"] == "S"]
    total = len(rows)

    buy_count  = len(buys)
    sell_count = len(sells)
    net_sent   = (buy_count - sell_count) / total if total > 0 else 0.0
    dollar_vol = sum(r["total_value"] for r in buys if r["total_value"])

    return {
        "insider_buy_count_30d":  buy_count,
        "insider_sell_count_30d": sell_count,
        "insider_net_sentiment":  round(net_sent, 4),
        "insider_dollar_volume":  dollar_vol,
    }


def _activist_signals(stock_id: int, as_of_date: str) -> dict:
    """
    Detect activist investor activity from SC 13D/G filings.

    SC 13D: filed when an investor acquires >5% and intends to influence management.
    SC 13G: passive holder >5%. SC 13D is the aggressive/activist version.

    activist_stake_change: we use the filing count as a proxy since parsing
    stake percentages from XBRL for each filing is out of scope here.
    A positive value means new filings appeared (stake being built).
    """
    cutoff = (
        date.fromisoformat(as_of_date) - timedelta(days=90)
    ).isoformat()

    rows = fetch_all(
        """SELECT form_type FROM sec_filings
           WHERE stock_id = ?
             AND filed_date >= ?
             AND form_type IN ('SC 13D', 'SC 13D/A', 'SC 13G', 'SC 13G/A')""",
        (stock_id, cutoff)
    )

    activist_flag = int(any(r["form_type"] in ("SC 13D", "SC 13D/A") for r in rows))
    stake_change  = float(len(rows))  # number of new activist filings as proxy

    return {
        "activist_filing_flag": activist_flag,
        "activist_stake_change": stake_change,
    }


def _material_event_signals(stock_id: int, as_of_date: str) -> dict:
    """
    Count 8-K filings (material events) and check for negative flags.

    8-K count in 30 days: many 8-Ks can indicate volatility (M&A, exec turnover).
    negative_8k_flag: 1 if any recent 8-K contained red-flag keywords
    (restatement, investigation, going concern, etc.) as detected by
    the sec_edgar collector's is_negative_8k() function.
    """
    cutoff = (
        date.fromisoformat(as_of_date) - timedelta(days=30)
    ).isoformat()

    rows = fetch_all(
        """SELECT is_negative_8k FROM sec_filings
           WHERE stock_id = ?
             AND filed_date >= ?
             AND form_type = '8-K'""",
        (stock_id, cutoff)
    )

    count = len(rows)
    # is_negative_8k is now stored by sec_edgar.py after keyword-scanning each 8-K.
    # Flag is 1 if any filing in the window contains red-flag keywords.
    neg_flag = int(any(r["is_negative_8k"] for r in rows))

    return {
        "material_event_count_30d": count,
        "negative_8k_flag":         neg_flag,
    }


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store(stock_id: int, as_of_date: str, features: dict):
    """Upsert sentiment features for one stock on one date."""
    cols         = ", ".join(features.keys())
    placeholders = ", ".join(["?"] * len(features))
    values       = list(features.values())

    with get_db() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO sentiment_features
                (stock_id, date, {cols})
                VALUES (?, ?, {placeholders})""",
            [stock_id, as_of_date] + values,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(stock_id: int, ticker: str, as_of_date: str = None):
    """
    Compute and store all sentiment features for one stock.
    Called by data/pipeline.py for each stock in the universe.

    Order of operations:
      1. Reddit (live API or neutral fallback if credentials missing)
      2. News RSS (live fetch + VADER scoring)
      3. SEC signals (queried from already-stored DB data)
    """
    as_of_date = as_of_date or date.today().isoformat()

    print(f"  {ticker}: computing sentiment features...")

    reddit_features = collect_reddit(ticker)
    news_features   = compute_news_features(ticker)
    insider_features = _insider_signals(stock_id, as_of_date)
    activist_features = _activist_signals(stock_id, as_of_date)
    material_features = _material_event_signals(stock_id, as_of_date)

    features = {}
    features.update(reddit_features)
    features.update(news_features)
    features.update(insider_features)
    features.update(activist_features)
    features.update(material_features)

    store(stock_id, as_of_date, features)

    non_null = sum(1 for v in features.values() if v is not None)
    print(f"  {ticker}: {len(features)} sentiment features computed ({non_null} non-null)")
