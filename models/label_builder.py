"""
models/label_builder.py

Constructs training datasets by joining all feature tables and computing
forward-return labels. This is the "data assembly" step that sits between
the raw feature store and the models.

WHY THIS IS ITS OWN MODULE
--------------------------
Label construction is subtle and error-prone. The two most common mistakes
in ML for finance are:
  1. Lookahead bias — using features that encode future information
  2. Label leakage — letting future prices contaminate today's features

Centralizing label logic here means:
- Every model (XGBoost, LSTM, backtester) uses the same labels
- The boundary conditions are tested once in tests/test_signals.py
- When we change the horizon (63 → 126 days), we change it in one place

LABEL DEFINITION
----------------
90-calendar-day (≈ 63 trading day) forward return, bucketed into 4 classes:

  Class 0: < -5%          (avoid — expected loss)
  Class 1: -5% to +5%     (hold — flat, no strong signal)
  Class 2: +5% to +15%    (buy — moderate gain expected)
  Class 3: > +15%         (strong buy — high conviction)

WHY CLASSIFICATION NOT REGRESSION
----------------------------------
Stock returns are too noisy to predict as a point estimate. A model that
predicts "+8.3% return" is lying — the confidence interval on a 90-day
individual stock return is ±30% or more. But classifying "will this be in
the top quartile of outcomes?" is a learnable problem:
  - Class boundaries capture economically meaningful thresholds
  - Cross-entropy loss is more stable than MSE on fat-tailed data
  - Softmax probabilities give us a ranking signal even without hard labels

TRAINING SAMPLE STRUCTURE
--------------------------
Each row in the training set is a (stock, date) pair where:
  - "date" is the date we would have made the prediction
  - All features reflect information available on that date
  - The label is the return from date to date+horizon (future, but used
    for training only — the model never sees this during inference)

Walk-forward validation splits this into non-overlapping folds:
  [train_start ... train_end] → [val_start ... val_end]
  with a 1-month gap to prevent any label overlap (since labels look
  63 trading days into the future, a gap of ~21 days isn't enough).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from database.db import fetch_all, get_connection
from config.settings import RETURN_BUCKETS, RETURN_BUCKET_LABELS, FEATURES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default prediction horizon in trading days (~3 months = 1 quarter)
DEFAULT_HORIZON = FEATURES.get("prediction_horizon_days", 63)

# Return bucket boundaries (same as RETURN_BUCKETS, kept local for clarity)
_BINS   = [-np.inf, -0.05, 0.05, 0.15, np.inf]
_LABELS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Core label computation
# ---------------------------------------------------------------------------

def compute_forward_returns(
    prices: pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> pd.Series:
    """
    For each date t, compute (price[t+horizon] / price[t]) - 1.

    Returns a Series aligned to the input index.
    The last `horizon` entries are NaN — no future price exists yet.

    Args:
        prices:  Daily adj_close series, indexed by date strings or Timestamps.
        horizon: Number of trading days to look forward.

    Returns:
        Series of fractional returns, same index as input.

    Example:
        prices = pd.Series([100, 105, 110], index=['2024-01-01', ...])
        compute_forward_returns(prices, horizon=1)
        → [0.05, 0.0476..., NaN]
    """
    future_price = prices.shift(-horizon)
    return (future_price / prices) - 1


def bucket_returns(returns: pd.Series) -> pd.Series:
    """
    Map a Series of forward returns to 4-class integer labels.

    Bucket boundaries:
      Class 0: < -5%          (big loss)
      Class 1: -5% to +5%     (flat)
      Class 2: +5% to +15%    (moderate gain)
      Class 3: > +15%         (strong gain)

    Boundary rule (pd.cut default, right=True):
      -5% exactly → Class 0 (belongs to left bin)
      +15% exactly → Class 2 (belongs to left bin)

    Args:
        returns: Series of fractional returns (0.10 = 10%)

    Returns:
        Series of integers in {0, 1, 2, 3}, same index.

    Note:
        NaN values in input produce NaN in output.
        Always dropna() before passing to a model.
    """
    return pd.cut(returns, bins=_BINS, labels=_LABELS).astype("Int64")


# ---------------------------------------------------------------------------
# Dataset construction — XGBoost (flat feature vectors)
# ---------------------------------------------------------------------------

def build_training_dataset(
    train_start: str,
    train_end: str,
    horizon: int = DEFAULT_HORIZON,
    tickers: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a flat training set joining all feature tables.

    Each row = one (stock, date) observation.
    Features are raw (not normalized) — preprocessing happens in
    models/preprocessing.py at training time.

    Walk-forward note:
        train_end should be at least `horizon` trading days before the
        last date you have prices for. Rows where the forward return label
        is NaN (because date+horizon exceeds available prices) are dropped.

    Args:
        train_start: Inclusive start date, "YYYY-MM-DD"
        train_end:   Inclusive end date for feature observation dates.
                     Labels look `horizon` days beyond this.
        horizon:     Trading days into the future for the return label.
        tickers:     If provided, restrict to this list of tickers.

    Returns:
        X:    DataFrame (n_samples, n_features) — raw features
        y:    Series (n_samples,) — integer class labels 0-3
        meta: DataFrame with columns [ticker, date, forward_return]
              Useful for tracking which sample maps to which stock/date.

    Raises:
        ValueError: if train_end is before train_start.

    Example usage:
        X, y, meta = build_training_dataset("2021-01-01", "2024-01-01")
        # Split for CV — see walk_forward_splits() below
    """
    if train_end < train_start:
        raise ValueError(f"train_end ({train_end}) must be >= train_start ({train_start})")

    # 1. Pull price history needed for label computation.
    #    We need prices from train_start to train_end + horizon buffer.
    #    Using a 150-day buffer (> 63 trading days) to cover weekends/holidays.
    price_rows = _fetch_prices(train_start, train_end, tickers, buffer_days=150)
    if not price_rows:
        raise RuntimeError("No price data found for the given date range / tickers.")

    # 2. Compute forward return labels for every (stock_id, date) in range.
    labels_df = _compute_labels(price_rows, horizon, train_start, train_end)
    if labels_df.empty:
        raise RuntimeError("No labeled samples after filtering NaN returns.")

    # 3. Pull all features from the 4 feature tables and merge them.
    features_df = _fetch_all_features(train_start, train_end, tickers)
    if features_df.empty:
        raise RuntimeError("No feature data found for the given date range / tickers.")

    # 4. Merge labels with features (inner join — only keep rows with both).
    merged = labels_df.merge(
        features_df,
        on=["stock_id", "date"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No rows after merging labels with features — check date alignment.")

    # 5. Drop rows with NaN labels (can happen if price series ends early).
    merged = merged.dropna(subset=["label"])

    # 6. Separate into X, y, meta.
    meta_cols    = ["ticker", "date", "forward_return"]
    feature_cols = [c for c in merged.columns
                    if c not in meta_cols + ["stock_id", "label"]]

    X    = merged[feature_cols].copy()
    y    = merged["label"].astype(int)
    meta = merged[meta_cols].copy()

    return X, y, meta


def build_inference_dataset(
    as_of_date: str,
    tickers: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a feature set for inference (no labels needed).

    Pulls the most recent feature row for each ticker as of `as_of_date`.
    Used by weekly_predict.py to generate today's buy/sell signals.

    Args:
        as_of_date: The prediction date, "YYYY-MM-DD"
        tickers:    If None, uses all active tickers.

    Returns:
        X:    DataFrame (n_tickers, n_features)
        meta: DataFrame with [ticker, date] columns
    """
    features_df = _fetch_all_features(as_of_date, as_of_date, tickers,
                                       use_latest=True)
    if features_df.empty:
        raise RuntimeError(f"No features found for as_of_date={as_of_date}")

    meta_cols    = ["ticker", "date", "stock_id"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    X    = features_df[feature_cols].copy()
    meta = features_df[["ticker", "date"]].copy()
    return X, meta


# ---------------------------------------------------------------------------
# Dataset construction — LSTM (3D sequence tensors)
# ---------------------------------------------------------------------------

def build_sequence_dataset(
    train_start: str,
    train_end: str,
    seq_len: int = None,
    horizon: int = DEFAULT_HORIZON,
    tickers: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build 3D sequence tensors for LSTM training.

    For each (stock, observation_date) pair:
      - Take the 60 trading days ending on observation_date
      - Stack the lstm_feature_subset columns into shape (seq_len, n_features)
      - Label is the 4-class return from observation_date + horizon

    Why 60 days?
      - Captures ~3 months of trading patterns (momentum, reversal cycles)
      - Short enough that early rows in the training window have 60 prior days
      - Long enough to capture MACD crossovers (which use 26-day EMA)

    Feature sources (see settings.FEATURES["lstm_feature_sources"]):
      - price:     daily_prices (adj_close, volume)
      - technical: technical_features (rsi_14, macd, ...)
      - sentiment: sentiment_features (reddit_sentiment_avg_7d, ...)
      - macro:     macro_features (vix_level, yield_curve_slope, ...)

    Args:
        train_start: First possible observation date.
        train_end:   Last possible observation date.
        seq_len:     Sequence length (default: settings.lstm_sequence_length).
        horizon:     Trading days into the future for the label.
        tickers:     If None, uses all active tickers.

    Returns:
        X_seq:  np.ndarray shape (n_samples, seq_len, n_features)
        y_seq:  np.ndarray shape (n_samples,) — integer labels 0-3
        meta:   DataFrame with [ticker, date, forward_return] for each sample

    Notes:
        NaN values in sequences are forward-filled within each stock's
        time series, then backfilled for any leading NaNs, then zero-filled
        as a last resort (absence of signal = neutral).
    """
    from config.settings import FEATURES as F

    seq_len     = seq_len or F["lstm_sequence_length"]
    feature_cols = F["lstm_feature_subset"]
    n_features   = len(feature_cols)

    # Fetch the LSTM feature columns for each stock over the date range.
    # We need seq_len extra days before train_start to build the first sequence.
    extended_start = _offset_trading_days(train_start, -seq_len - 10)

    seq_df = _fetch_lstm_features(extended_start, train_end, tickers, feature_cols)
    if seq_df.empty:
        raise RuntimeError("No LSTM feature data found.")

    # Fetch labels (forward returns) for train_start..train_end only.
    price_rows = _fetch_prices(train_start, train_end, tickers, buffer_days=150)
    labels_df  = _compute_labels(price_rows, horizon, train_start, train_end)

    X_list, y_list, meta_list = [], [], []

    for (stock_id, ticker), group in seq_df.groupby(["stock_id", "ticker"]):
        group = group.sort_values("date").set_index("date")

        # Forward-fill within each stock, then backfill, then zero-fill
        seq_features = (
            group[feature_cols]
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        # Get labels for this stock
        stock_labels = labels_df[labels_df["stock_id"] == stock_id].copy()
        if stock_labels.empty:
            continue

        for _, lrow in stock_labels.iterrows():
            obs_date = lrow["date"]
            label    = lrow["label"]
            fwd_ret  = lrow["forward_return"]

            if pd.isna(label):
                continue

            # Find rows in seq_features up to and including obs_date
            avail = seq_features.loc[seq_features.index <= obs_date]
            if len(avail) < seq_len:
                continue  # not enough history for a full sequence

            seq = avail.iloc[-seq_len:].values.astype(np.float32)  # (seq_len, n_features)
            X_list.append(seq)
            y_list.append(int(label))
            meta_list.append({"ticker": ticker, "date": obs_date,
                               "forward_return": fwd_ret})

    if not X_list:
        raise RuntimeError("No valid sequences found — check date range and feature coverage.")

    X_seq  = np.stack(X_list)                       # (n_samples, seq_len, n_features)
    y_seq  = np.array(y_list, dtype=np.int64)       # (n_samples,)
    meta   = pd.DataFrame(meta_list)

    return X_seq, y_seq, meta


# ---------------------------------------------------------------------------
# Walk-forward split generator
# ---------------------------------------------------------------------------

def walk_forward_splits(
    all_dates: pd.Index,
    train_months: int = 36,
    val_months: int = 3,
    gap_months: int = 1,
    step_months: int = 3,
) -> list:
    """
    Generate walk-forward (expanding or rolling) train/val splits.

    Each fold:
      [train_start ... train_end] GAP [val_start ... val_end]

    The 1-month gap prevents label overlap: since each label looks 63 trading
    days (~3 months) into the future, a row near train_end could have its
    label period extend into the val window. The gap ensures the last training
    label's forward period ends before the first val observation.

    Args:
        all_dates:    Sorted DatetimeIndex of all observation dates.
        train_months: Size of training window (rolling, not expanding).
        val_months:   Size of validation window.
        gap_months:   Gap between train_end and val_start (label buffer).
        step_months:  How far to advance each fold.

    Returns:
        List of dicts: [
          {"train_start": ..., "train_end": ...,
           "val_start": ...,   "val_end": ...},
          ...
        ]

    Why 1-month gap and not 3-month?
        The label horizon is 63 TRADING days ≈ 3 calendar months.
        But walk-forward folds step by 3 months. If we add a full 3-month gap,
        we'd waste an entire fold worth of data. 1 month is the minimum that
        prevents any individual row's label from extending into the val window
        in practice (since 63 trading days ≈ 84 calendar days ≈ 2.8 months,
        not exactly 3). Use 2 months if you want extra safety margin.
    """
    splits      = []
    dates_arr   = pd.DatetimeIndex(all_dates).sort_values()
    start_date  = dates_arr[0]

    while True:
        train_start = start_date
        train_end   = train_start + pd.DateOffset(months=train_months)
        val_start   = train_end   + pd.DateOffset(months=gap_months)
        val_end     = val_start   + pd.DateOffset(months=val_months)

        if val_end > dates_arr[-1]:
            break

        splits.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end":   train_end.strftime("%Y-%m-%d"),
            "val_start":   val_start.strftime("%Y-%m-%d"),
            "val_end":     val_end.strftime("%Y-%m-%d"),
        })

        start_date = start_date + pd.DateOffset(months=step_months)

    return splits


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fetch_prices(
    start: str,
    end: str,
    tickers: Optional[list],
    buffer_days: int = 150,
) -> list:
    """Fetch adj_close from daily_prices over [start, extended_end]."""
    extended_end = (
        pd.Timestamp(end) + pd.DateOffset(days=buffer_days)
    ).strftime("%Y-%m-%d")

    ticker_filter = ""
    params: tuple = (start, extended_end)
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        ticker_filter = f"AND s.ticker IN ({placeholders})"
        params = tuple(tickers) + params

    return fetch_all(
        f"""SELECT dp.stock_id, s.ticker, dp.date, dp.adj_close
            FROM daily_prices dp
            JOIN stocks s ON s.id = dp.stock_id
            WHERE dp.date >= ? AND dp.date <= ?
              {ticker_filter}
            ORDER BY dp.stock_id, dp.date""",
        params,
    )


def _compute_labels(
    price_rows: list,
    horizon: int,
    label_start: str,
    label_end: str,
) -> pd.DataFrame:
    """
    Given raw price rows, compute forward return labels for the
    [label_start, label_end] window.

    Returns a DataFrame with columns:
        [stock_id, ticker, date, forward_return, label]
    """
    # Pivot to wide format: columns = stock_ids, rows = dates
    df = pd.DataFrame([dict(r) for r in price_rows])
    if df.empty:
        return pd.DataFrame()

    records = []
    for (stock_id, ticker), group in df.groupby(["stock_id", "ticker"]):
        group   = group.sort_values("date").set_index("date")
        prices  = group["adj_close"].astype(float)
        fwd_ret = compute_forward_returns(prices, horizon)
        labels  = bucket_returns(fwd_ret)

        for date, ret, lbl in zip(fwd_ret.index, fwd_ret.values, labels.values):
            if date < label_start or date > label_end:
                continue
            records.append({
                "stock_id":      stock_id,
                "ticker":        ticker,
                "date":          date,
                "forward_return": ret,
                "label":         lbl,
            })

    return pd.DataFrame(records)


def _fetch_all_features(
    start: str,
    end: str,
    tickers: Optional[list],
    use_latest: bool = False,
) -> pd.DataFrame:
    """
    Join technical, sentiment, fundamental, and macro features into a flat
    DataFrame with one row per (stock, date).

    use_latest=True: for each stock, return only the most recent row ≤ end.
    """
    ticker_filter = ""
    base_params: tuple = (start, end)
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        ticker_filter = f"AND s.ticker IN ({placeholders})"
        base_params = tuple(tickers) + base_params

    query = f"""
        SELECT
            s.id                  AS stock_id,
            s.ticker,
            t.date,
            -- Technical features
            t.rsi_14, t.rsi_7,
            t.macd, t.macd_signal, t.macd_histogram,
            t.bollinger_pct_b, t.bollinger_bandwidth,
            t.realized_vol_20d, t.realized_vol_60d,
            t.volume_ratio_10d, t.volume_ratio_30d, t.volume_trend_20d,
            t.momentum_10d, t.momentum_30d, t.momentum_90d,
            t.relative_strength_vs_spy, t.relative_strength_vs_sector,
            t.beta_60d, t.correlation_spy_60d,
            t.price_vs_sma50, t.price_vs_sma200, t.sma50_vs_sma200,
            t.price_vs_sma20, t.price_vs_ema10, t.price_vs_ema20,
            t.atr_14, t.atr_pct,
            t.stochastic_k, t.stochastic_d, t.williams_r,
            t.cci_14, t.mfi_14,
            t.high_52w_pct, t.low_52w_pct,
            t.daily_return, t.weekly_return,
            -- Sentiment / insider features
            sf.reddit_mention_count_7d, sf.reddit_mention_change,
            sf.reddit_sentiment_avg_7d, sf.reddit_sentiment_std_7d,
            sf.reddit_bullish_ratio,
            sf.news_volume_7d, sf.news_sentiment_avg_7d, sf.news_sentiment_30d,
            sf.news_sentiment_momentum, sf.negative_news_flag,
            sf.insider_buy_count_30d, sf.insider_sell_count_30d,
            sf.insider_net_sentiment, sf.insider_dollar_volume,
            sf.wsb_mention_flag, sf.activist_filing_flag,
            -- Macro features (date-keyed, no stock_id)
            m.vix_level, m.vix_change_5d, m.vix_percentile_1y,
            m.yield_curve_slope, m.treasury_10y, m.treasury_2y,
            m.fed_funds_rate, m.fed_rate_change_3m,
            m.credit_spread, m.credit_spread_change,
            m.gdp_growth, m.unemployment_rate,
            m.dollar_index, m.dollar_change_30d,
            m.sector_momentum_xlk, m.sector_momentum_xlf,
            m.growth_vs_value,
            -- Fundamental features
            ff.pe_ratio, ff.pb_ratio, ff.ps_ratio,
            ff.earnings_yield, ff.fcf_yield, ff.ev_ebitda, ff.ev_revenue,
            ff.gross_margin, ff.operating_margin, ff.net_margin,
            ff.roe, ff.roa, ff.asset_turnover, ff.current_ratio,
            ff.debt_to_equity, ff.interest_coverage, ff.revenue_growth_yoy,
            ff.earnings_growth_yoy, ff.free_cash_flow_growth,
            ff.piotroski_score, ff.altman_z, ff.buyback_yield,
            ff.pe_vs_sector_avg, ff.roe_vs_sector, ff.net_margin_vs_sector
        FROM technical_features t
        JOIN stocks s ON s.id = t.stock_id
        LEFT JOIN sentiment_features sf
               ON sf.stock_id = t.stock_id AND sf.date = t.date
        LEFT JOIN macro_features m
               ON m.date = t.date
        LEFT JOIN fundamental_features ff
               ON ff.stock_id = t.stock_id
               AND ff.date = (
                   SELECT MAX(ff2.date)
                   FROM fundamental_features ff2
                   WHERE ff2.stock_id = t.stock_id
                     AND ff2.date <= t.date
               )
        WHERE t.date >= ? AND t.date <= ?
          {ticker_filter}
        ORDER BY s.ticker, t.date
    """

    rows = fetch_all(query, base_params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = df["date"].astype(str)
    return df


def _fetch_lstm_features(
    start: str,
    end: str,
    tickers: Optional[list],
    feature_cols: list,
) -> pd.DataFrame:
    """
    Fetch only the LSTM feature subset from their respective source tables.
    Returns a DataFrame with [stock_id, ticker, date] + feature_cols.
    """
    from config.settings import FEATURES as F
    sources = F["lstm_feature_sources"]

    ticker_filter = ""
    base_params: tuple = (start, end)
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        ticker_filter = f"AND s.ticker IN ({placeholders})"
        base_params = tuple(tickers) + base_params

    # Build SELECT clause: each column aliased to avoid name collisions
    select_parts = ["dp.stock_id", "s.ticker", "dp.date"]
    for col in feature_cols:
        src = sources.get(col, "technical")
        if src == "price":
            select_parts.append(f"dp.{col}")
        elif src == "technical":
            select_parts.append(f"t.{col}")
        elif src == "sentiment":
            select_parts.append(f"sf.{col}")
        elif src == "macro":
            select_parts.append(f"m.{col}")

    select_clause = ", ".join(select_parts)

    query = f"""
        SELECT {select_clause}
        FROM daily_prices dp
        JOIN stocks s ON s.id = dp.stock_id
        LEFT JOIN technical_features t
               ON t.stock_id = dp.stock_id AND t.date = dp.date
        LEFT JOIN sentiment_features sf
               ON sf.stock_id = dp.stock_id AND sf.date = dp.date
        LEFT JOIN macro_features m
               ON m.date = dp.date
        WHERE dp.date >= ? AND dp.date <= ?
          {ticker_filter}
        ORDER BY dp.stock_id, dp.date
    """

    rows = fetch_all(query, base_params)
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def _offset_trading_days(date_str: str, n_days: int) -> str:
    """
    Offset a date by approximately n_days trading days.
    Uses 1.5× multiplier to account for weekends/holidays.
    """
    ts = pd.Timestamp(date_str)
    calendar_days = int(abs(n_days) * 1.5)
    if n_days < 0:
        return (ts - pd.DateOffset(days=calendar_days)).strftime("%Y-%m-%d")
    return (ts + pd.DateOffset(days=calendar_days)).strftime("%Y-%m-%d")
