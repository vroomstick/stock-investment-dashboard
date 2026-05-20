"""
config/settings.py

Central configuration for the entire project. Every hardcoded value —
API endpoints, thresholds, hyperparameters, risk limits — lives here.

Why centralize config?
- When Manuj says "change the stop-loss from 15% to 12%", you change one line
- When you're debugging, you know exactly where every magic number came from
- Nothing sensitive (API keys, passwords) is hardcoded — they're loaded from
  environment variables so they never accidentally end up in git

HOW TO SET YOUR API KEYS (do this before running anything):
  1. Copy .env.example to .env  (cp .env.example .env)
  2. Fill in your real keys in .env
  3. .env is in .gitignore — it will never be committed
"""

import os
from pathlib import Path

# Project root: the directory that contains this config/ folder
PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# API Keys — loaded from environment variables, never hardcoded
# ---------------------------------------------------------------------------
# os.getenv() returns None if the variable isn't set, or the default if given.
# We'll validate these are set when the relevant collector is first run.

FRED_API_KEY     = os.getenv("FRED_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET    = os.getenv("REDDIT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "stock-dashboard-mlspec")
EMAIL_ADDRESS    = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD   = os.getenv("EMAIL_PASSWORD", "")    # Gmail App Password
EMAIL_RECIPIENT  = os.getenv("EMAIL_RECIPIENT", "")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = PROJECT_ROOT / "database" / "stock_dashboard.db"


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------
SEC = {
    # Base URLs — no authentication required, but User-Agent header is mandatory
    "submissions_url":   "https://data.sec.gov/submissions/CIK{cik}.json",
    "company_facts_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
    "company_concept_url": "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json",
    "frames_url":        "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/{unit}/{period}.json",
    "full_text_search":  "https://efts.sec.gov/LATEST/search-index",

    # Required by SEC — must identify yourself or get blocked
    "user_agent": "MLSpec Research admin@mlspec.com",

    # Rate limit: 10 requests/second max
    # We stay safely under at 8/sec to avoid 503s
    "rate_limit_per_sec": 8,
    "retry_attempts": 3,
    "retry_backoff_base": 2,   # exponential backoff: 2^attempt seconds

    # Filing types we care about (used to filter the submissions response)
    "target_form_types": ["4", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
                          "8-K", "10-Q", "10-K"],
}


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------
DATA = {
    # How many years of price history to fetch on initial load
    "price_history_years": 5,

    # How many years back to train on
    "training_start_date": "2019-01-01",

    # Reddit subreddits to monitor for sentiment
    "reddit_subreddits": ["investing", "SecurityAnalysis", "stocks", "wallstreetbets"],

    # News RSS feeds — {TICKER} is replaced at runtime
    "rss_feeds": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
        "https://seekingalpha.com/api/sa/combined/{ticker}.xml",
    ],

    # Macro series to pull from FRED
    "fred_series": {
        "GS10":     "treasury_10y",
        "GS2":      "treasury_2y",
        "FEDFUNDS": "fed_funds_rate",
        "CPIAUCSL": "cpi",
        "UNRATE":   "unemployment_rate",
        "GDP":      "gdp",
        "VIXCLS":   "vix_level",
        "BAA":      "baa_yield",
        "AAA":      "aaa_yield",
        "ICSA":     "initial_claims",
        "M2SL":     "m2_money_supply",
        "DTWEXBGS": "dollar_index",
    },

    # Sector ETFs for sector rotation features
    "sector_etfs": {
        "XLK": "technology",
        "XLF": "financials",
        "XLV": "healthcare",
        "XLE": "energy",
        "XLI": "industrials",
        "XLP": "consumer_staples",
        "XLY": "consumer_discretionary",
    },
}


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
FEATURES = {
    # Lookback windows for technical indicators (trading days)
    "sma_windows":  [10, 20, 50, 100, 200],
    "ema_windows":  [10, 20, 50],
    "rsi_windows":  [7, 14],
    "atr_window":   14,
    "momentum_windows": [10, 30, 90],

    # Number of days in the LSTM sequence input
    "lstm_sequence_length": 60,

    # Features used as LSTM inputs (temporal features only — point-in-time
    # fundamentals don't make sense in a daily sequence)
    "lstm_feature_subset": [
        "adj_close", "volume", "rsi_14", "macd", "macd_signal",
        "bollinger_pct_b", "realized_vol_20d", "volume_ratio_30d",
        "momentum_10d", "momentum_30d", "relative_strength_vs_spy",
        "insider_buy_count_30d", "insider_sell_count_30d",
        "reddit_sentiment_avg_7d", "news_sentiment_avg_7d",
        "vix_level", "yield_curve_slope", "credit_spread",
    ],

    # Source table for each LSTM feature.
    # The sequence builder JOINs multiple tables; this map tells it which
    # table each column lives in:
    #   "price"     → daily_prices         (per stock, per date)
    #   "technical" → technical_features   (per stock, per date)
    #   "sentiment" → sentiment_features   (per stock, per date)
    #   "macro"     → macro_features       (date only — same value all stocks)
    "lstm_feature_sources": {
        "adj_close":                "price",
        "volume":                   "price",
        "rsi_14":                   "technical",
        "macd":                     "technical",
        "macd_signal":              "technical",
        "bollinger_pct_b":          "technical",
        "realized_vol_20d":         "technical",
        "volume_ratio_30d":         "technical",
        "momentum_10d":             "technical",
        "momentum_30d":             "technical",
        "relative_strength_vs_spy": "technical",
        "insider_buy_count_30d":    "sentiment",
        "insider_sell_count_30d":   "sentiment",
        "reddit_sentiment_avg_7d":  "sentiment",
        "news_sentiment_avg_7d":    "sentiment",
        "vix_level":                "macro",
        "yield_curve_slope":        "macro",
        "credit_spread":            "macro",
    },

    # Sentiment lookback windows (calendar days)
    "sentiment_short_window": 7,
    "sentiment_long_window":  30,
    "insider_window_days":    30,
    "activist_window_days":   90,
}


# ---------------------------------------------------------------------------
# ML Models
# ---------------------------------------------------------------------------

# 4-class return bucket boundaries (fractional, e.g. -0.05 = -5%)
RETURN_BUCKETS = [-float("inf"), -0.05, 0.05, 0.15, float("inf")]
RETURN_BUCKET_LABELS = [0, 1, 2, 3]   # big_loss, flat, moderate_gain, strong_gain
RETURN_BUCKET_MIDPOINTS = [-0.10, 0.00, 0.10, 0.25]  # used for expected return calc

# Prediction horizons in trading days
PREDICTION_HORIZONS = {
    "short":  63,   # ~3 months
    "medium": 126,  # ~6 months
}
PRIMARY_HORIZON = "short"   # what the model is trained to predict

XGBOOST_PARAMS = {
    "objective":           "multi:softprob",
    "num_class":           4,
    "max_depth":           6,
    "learning_rate":       0.05,
    "n_estimators":        500,
    "min_child_weight":    5,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "reg_alpha":           0.1,   # L1 regularization
    "reg_lambda":          1.0,   # L2 regularization
    "eval_metric":         "mlogloss",
    "early_stopping_rounds": 50,
    "random_state":        42,
}

LSTM_PARAMS = {
    "input_size":   len(FEATURES["lstm_feature_subset"]),
    "hidden_size":  128,
    "num_layers":   2,
    "num_classes":  4,
    "dropout":      0.3,
    "learning_rate": 1e-3,
    "batch_size":   64,
    "max_epochs":   100,
    "patience":     10,   # early stopping patience
}

ENSEMBLE = {
    "xgb_weight":  0.6,
    "lstm_weight": 0.4,
}

# Walk-forward cross-validation settings
WALKFORWARD = {
    "train_months": 36,   # rolling training window
    "val_months":   3,    # hyperparameter tuning
    "test_months":  3,    # held-out evaluation
    "gap_months":   1,    # gap between train end and test start (prevents leakage)
}


# ---------------------------------------------------------------------------
# Scoring & Signal Generation
# ---------------------------------------------------------------------------
SCORING = {
    # Composite score = bull_prob*50 + expected_return_component*30 + confidence*20
    "bull_prob_weight":           50,
    "expected_return_weight":     30,
    "expected_return_normalizer": 25,  # divide expected_return_pct by this to get 0-1
    "confidence_weight":          20,

    # Action thresholds
    "strong_buy_score":     75,
    "strong_buy_confidence": 0.60,
    "buy_score":            60,
    "buy_confidence":       0.50,
    "watch_score":          45,
    "sell_score":           25,
    "sell_confidence":      0.60,
}


# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
RISK = {
    # Portfolio-level limits
    "max_positions":          15,
    "max_position_pct":       0.08,   # 8% max per single position
    "base_position_pct":      0.06,   # 6% base position size
    "min_cash_reserve_pct":   0.10,   # always keep 10% cash
    "max_sector_pct":         0.20,   # 20% max in any one sector
    "max_correlation":        0.70,   # max correlation between any two positions
    "target_beta_low":        0.80,
    "target_beta_high":       1.20,

    # Trade-level stops
    "stop_loss_pct":          -0.15,  # -15% from entry
    "profit_take_1_pct":       0.25,  # sell 25% of position at +25%
    "profit_take_1_size":      0.25,
    "profit_take_2_pct":       0.50,  # sell 50% of remaining at +50%
    "profit_take_2_size":      0.50,

    # Time-based exit
    "max_hold_days":          180,    # 6 months
    "time_exit_min_score":     60,    # re-evaluate if score < 60 at 6mo

    # Model-based exit triggers
    "exit_score_threshold":   40,     # was 60+, now below 40 = sell
    "exit_confidence_threshold": 0.35,

    # Volatility targeting for position sizing
    "target_vol_per_position": 0.15,  # 15% annualized vol target
    "vol_multiplier_cap":      1.50,

    # Portfolio drawdown trigger
    "max_portfolio_drawdown":  -0.15, # stop new buys if portfolio down 15% from peak

    # Transaction cost model
    "slippage_bps": 10,   # 10 basis points = 0.1% per trade
}


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
BACKTEST = {
    "initial_capital":   100_000,
    "max_positions":     RISK["max_positions"],
    "rebalance_freq":    "weekly",   # how often to re-evaluate positions
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
REPORT = {
    "schedule_day":    "sunday",
    "schedule_hour":   20,           # 8 PM
    "smtp_host":       "smtp.gmail.com",
    "smtp_port":       587,
    "top_n_picks":     10,           # number of stocks to highlight in report
    "alert_score_drop": 20,          # alert if score drops this many points
    "model_accuracy_alert_drop": 0.10,  # alert if accuracy drops 10%+ week over week
}
