-- =============================================================================
-- Stock Investment Dashboard — SQLite Schema
-- =============================================================================
-- Design principle: separate tables per data domain, joined by (stock_id, date)
-- This mirrors the spec's feature store architecture (fundamental / technical /
-- macro / sentiment) and makes it easy to update one category without touching
-- the others.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- stocks
-- Master list of every ticker we track. All other tables foreign-key into this.
-- CIK is the SEC's unique company identifier — needed for EDGAR API calls.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stocks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL UNIQUE,
    cik          TEXT,               -- 10-digit padded, e.g. "0000320193" (Apple)
    company_name TEXT,
    sector       TEXT,
    sector_etf   TEXT,               -- e.g. "XLK" for Technology, "XLF" for Financials
    industry     TEXT,
    is_active    INTEGER DEFAULT 1,  -- 1 = tracking, 0 = dropped from universe
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- daily_prices
-- Raw OHLCV data from yfinance. Stored separately from computed features
-- because (a) it's the raw input, and (b) features are recomputed on top of it.
-- adj_close accounts for splits and dividends — always use this for returns.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_prices (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id  INTEGER NOT NULL,
    date      TEXT    NOT NULL,  -- YYYY-MM-DD
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    adj_close REAL,
    volume    INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id),
    UNIQUE(stock_id, date)
);


-- -----------------------------------------------------------------------------
-- sec_filings
-- Log of every SEC filing we've fetched and processed.
-- accession_number is SEC's globally unique filing ID (e.g. 0000320193-23-000077).
-- raw_json stores the full API response so we can reparse without re-fetching.
-- processed flag lets daily_collect.py skip already-parsed filings.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sec_filings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id         INTEGER NOT NULL,
    accession_number TEXT    NOT NULL UNIQUE,
    form_type        TEXT    NOT NULL,  -- '4', '8-K', 'SC 13D', '10-Q', etc.
    filed_date       TEXT    NOT NULL,
    period_of_report TEXT,
    raw_json         TEXT,              -- full API response, stored for reprocessing
    processed        INTEGER DEFAULT 0, -- 0 = raw, 1 = features extracted
    is_negative_8k   INTEGER DEFAULT 0, -- 1 = negative keyword hit (restatement, investigation, etc.)
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id)
);


-- -----------------------------------------------------------------------------
-- insider_transactions
-- Parsed Form 4 data. Each row is one transaction by one insider.
-- transaction_type: 'P' = purchase (bullish signal), 'S' = sale (bearish signal).
-- This feeds directly into the sentiment feature: insider_buy_count_30d etc.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insider_transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id            INTEGER NOT NULL,
    filing_id           INTEGER,
    transaction_date    TEXT,
    insider_name        TEXT,
    insider_title       TEXT,
    transaction_type    TEXT,   -- 'P' (purchase) or 'S' (sale)
    shares              REAL,
    price_per_share     REAL,
    total_value         REAL,
    shares_owned_after  REAL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id),
    FOREIGN KEY (filing_id) REFERENCES sec_filings(id)
);


-- -----------------------------------------------------------------------------
-- fundamental_features
-- 80-100 features per stock per quarter-end date.
-- Source: yfinance quarterly financials + SEC XBRL API.
-- One row per (stock, quarter). The ML model uses the most recent quarter's
-- values as a point-in-time snapshot.
-- Piotroski F-Score: 9 binary sub-components (0 or 1 each), summed to 0-9.
-- Altman Z-Score: bankruptcy prediction composite — below 1.8 is danger zone.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamental_features (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id                  INTEGER NOT NULL,
    date                      TEXT    NOT NULL,  -- quarter-end date YYYY-MM-DD

    -- Profitability
    roe                       REAL,  -- Return on Equity: Net Income / Equity
    roa                       REAL,  -- Return on Assets: Net Income / Total Assets
    roic                      REAL,  -- Return on Invested Capital
    gross_margin              REAL,  -- Gross Profit / Revenue
    operating_margin          REAL,  -- Operating Income / Revenue
    net_margin                REAL,  -- Net Income / Revenue
    gross_margin_trend        REAL,  -- QoQ change in gross margin
    operating_margin_trend    REAL,  -- QoQ change in operating margin

    -- Valuation
    pe_ratio                  REAL,  -- Price / Trailing EPS
    forward_pe                REAL,  -- Price / Forward EPS
    pb_ratio                  REAL,  -- Price / Book Value
    ps_ratio                  REAL,  -- Price / Revenue per share
    ev_ebitda                 REAL,  -- Enterprise Value / EBITDA
    peg_ratio                 REAL,  -- P/E / Earnings Growth Rate
    pe_vs_sector_avg          REAL,  -- Stock P/E relative to sector average

    -- Growth
    revenue_growth_yoy        REAL,  -- (Rev_now - Rev_1yr_ago) / Rev_1yr_ago
    revenue_growth_qoq        REAL,  -- Quarter-over-quarter revenue change
    earnings_growth_yoy       REAL,  -- YoY EPS growth
    revenue_acceleration      REAL,  -- Current growth rate minus prior growth rate
    book_value_growth         REAL,  -- YoY change in book value per share
    fcf_growth                REAL,  -- YoY change in free cash flow

    -- Quality & Balance Sheet Strength
    debt_to_equity            REAL,  -- Total Debt / Shareholders' Equity
    current_ratio             REAL,  -- Current Assets / Current Liabilities
    interest_coverage         REAL,  -- EBIT / Interest Expense
    piotroski_f_score         INTEGER, -- Sum of 9 binary signals below (0-9)
    altman_z_score            REAL,  -- Bankruptcy risk composite
    accruals_ratio            REAL,  -- (Net Income - Op CF) / Total Assets
    cash_vs_earnings_quality  REAL,  -- Operating CF / Net Income

    -- Piotroski F-Score sub-components (each 0 or 1)
    p_positive_net_income         INTEGER,
    p_positive_operating_cf       INTEGER,
    p_roa_increasing              INTEGER,
    p_cf_greater_than_ni          INTEGER,
    p_debt_ratio_decreasing       INTEGER,
    p_current_ratio_increasing    INTEGER,
    p_no_new_shares               INTEGER,
    p_gross_margin_increasing     INTEGER,
    p_asset_turnover_increasing   INTEGER,

    -- Efficiency
    asset_turnover            REAL,  -- Revenue / Total Assets
    inventory_turnover        REAL,  -- COGS / Avg Inventory
    receivables_turnover      REAL,  -- Revenue / Avg Receivables
    days_sales_outstanding    REAL,  -- 365 / Receivables Turnover

    -- Extended Valuation
    ev_revenue                REAL,  -- Enterprise Value / Revenue
    price_to_fcf              REAL,  -- Market Cap / Free Cash Flow
    earnings_yield            REAL,  -- EPS / Price (inverse of P/E)
    fcf_yield                 REAL,  -- FCF per share / Price
    dividend_yield            REAL,  -- Annual Dividend / Price
    price_to_tangible_book    REAL,  -- Price / Tangible Book Value per share

    -- Extended Quality / Liquidity
    quick_ratio               REAL,  -- (Current Assets - Inventory) / Current Liabilities
    cash_ratio                REAL,  -- Cash / Current Liabilities
    net_debt                  REAL,  -- Total Debt - Cash
    net_debt_ebitda           REAL,  -- Net Debt / EBITDA
    fcf_to_debt               REAL,  -- Free Cash Flow / Total Debt
    interest_coverage_ttm     REAL,  -- EBIT(TTM) / Interest Expense(TTM)

    -- Extended Profitability
    ebitda_margin             REAL,  -- EBITDA / Revenue
    fcf_margin                REAL,  -- Free Cash Flow / Revenue
    capex_ratio               REAL,  -- CapEx / Revenue
    rd_intensity              REAL,  -- R&D Expense / Revenue
    return_on_tangible_equity REAL,  -- Net Income / Tangible Equity

    -- Scale / TTM Aggregates
    revenue_ttm               REAL,  -- Trailing 12-month revenue
    gross_profit_ttm          REAL,  -- Trailing 12-month gross profit
    eps_ttm                   REAL,  -- Trailing 12-month EPS
    shares_outstanding        REAL,  -- Shares outstanding (for position sizing)

    -- Sector-Relative (normalized peer comparisons)
    roe_vs_sector             REAL,  -- Stock ROE / Sector median ROE
    roa_vs_sector             REAL,  -- Stock ROA / Sector median ROA
    net_margin_vs_sector      REAL,  -- Stock net margin / Sector median net margin
    revenue_growth_vs_sector  REAL,  -- Stock revenue growth / Sector median growth

    -- Additional Growth
    net_income_growth         REAL,  -- YoY change in net income
    operating_cf_growth       REAL,  -- YoY change in operating cash flow
    buyback_yield             REAL,  -- Share repurchases / Market Cap

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id),
    UNIQUE(stock_id, date)
);


-- -----------------------------------------------------------------------------
-- technical_features
-- 60-80 features per stock per trading day.
-- Source: computed locally from daily_prices using pandas.
-- Updated daily by the data pipeline.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS technical_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id    INTEGER NOT NULL,
    date        TEXT    NOT NULL,

    -- Moving Averages & Trends
    sma_10               REAL,
    sma_20               REAL,
    sma_50               REAL,
    sma_100              REAL,
    sma_200              REAL,
    ema_10               REAL,
    ema_20               REAL,
    ema_50               REAL,
    price_vs_sma50       REAL,  -- (Price - SMA50) / SMA50
    price_vs_sma200      REAL,  -- (Price - SMA200) / SMA200
    sma50_vs_sma200      REAL,  -- Golden/death cross signal
    sma50_slope          REAL,  -- 10-day slope of SMA50
    sma200_slope         REAL,

    -- Momentum Indicators
    rsi_14               REAL,  -- 14-day Relative Strength Index
    rsi_7                REAL,  -- Short-term RSI
    macd                 REAL,  -- EMA12 - EMA26
    macd_signal          REAL,  -- 9-day EMA of MACD
    macd_histogram       REAL,  -- MACD - Signal
    stochastic_k         REAL,  -- %K stochastic oscillator
    stochastic_d         REAL,  -- %D (3-day SMA of %K)
    williams_r           REAL,  -- Williams %R
    momentum_10d         REAL,  -- Price / Price_10d_ago - 1
    momentum_30d         REAL,
    momentum_90d         REAL,

    -- Volatility
    bollinger_upper      REAL,
    bollinger_lower      REAL,
    bollinger_pct_b      REAL,  -- Where price sits within the bands (0-1)
    bollinger_bandwidth  REAL,  -- Band width relative to SMA20
    atr_14               REAL,  -- Average True Range (absolute volatility)
    realized_vol_20d     REAL,  -- Annualized 20-day realized volatility
    realized_vol_60d     REAL,
    vol_ratio            REAL,  -- 20d vol / 60d vol (expanding vs contracting)

    -- Volume
    volume_ratio_10d     REAL,  -- Today's volume / 10-day avg
    volume_ratio_30d     REAL,
    volume_trend_20d     REAL,  -- Linear regression slope of volume
    obv                  REAL,  -- On-Balance Volume
    obv_trend            REAL,  -- 20-day slope of OBV
    accumulation_distribution REAL,
    volume_price_trend   REAL,

    -- Relative Strength
    relative_strength_vs_spy    REAL,  -- Stock return / SPY return (20d rolling)
    relative_strength_vs_sector REAL,
    beta_60d                    REAL,  -- 60-day beta vs SPY
    correlation_spy_60d         REAL,

    -- Extended Price vs MA
    price_vs_sma10              REAL,  -- (Price - SMA10) / SMA10
    price_vs_sma20              REAL,  -- (Price - SMA20) / SMA20
    price_vs_ema10              REAL,  -- (Price - EMA10) / EMA10
    price_vs_ema20              REAL,  -- (Price - EMA20) / EMA20
    price_vs_ema50              REAL,  -- (Price - EMA50) / EMA50
    ema50_vs_sma200             REAL,  -- (EMA50 - SMA200) / SMA200

    -- Extended Momentum / Rate of Change
    momentum_5d                 REAL,  -- 5-day return
    momentum_180d               REAL,  -- 180-day return (6-month)
    roc_5d                      REAL,  -- Rate of change: (Price/Price[-5]) - 1
    roc_10d                     REAL,
    roc_20d                     REAL,

    -- 52-Week Levels
    high_52w_pct                REAL,  -- (Price - 52w High) / 52w High (how far from high)
    low_52w_pct                 REAL,  -- (Price - 52w Low) / 52w Low (how far above low)

    -- Short-term Returns
    daily_return                REAL,  -- Single-day return
    weekly_return               REAL,  -- 5-day return

    -- Extended Oscillators
    cci_14                      REAL,  -- Commodity Channel Index (14-day)
    mfi_14                      REAL,  -- Money Flow Index (14-day, volume-weighted RSI)
    atr_pct                     REAL,  -- ATR as % of price (normalized ATR)

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id),
    UNIQUE(stock_id, date)
);


-- -----------------------------------------------------------------------------
-- macro_features
-- 30-40 features per date — shared across ALL stocks (no stock_id).
-- Source: FRED API. Updated daily/weekly depending on FRED release cadence.
-- One row per date. When building the ML feature vector for a stock, we join
-- on the most recent available macro date (FRED data lags by days/weeks).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro_features (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    date                 TEXT NOT NULL UNIQUE,

    -- Interest Rates & Yield Curve
    treasury_10y         REAL,  -- 10-Year Treasury Yield (GS10)
    treasury_2y          REAL,  -- 2-Year Treasury Yield (GS2)
    yield_curve_slope    REAL,  -- GS10 - GS2 (negative = inverted = recession signal)
    fed_funds_rate       REAL,  -- Federal Funds Rate (FEDFUNDS)
    fed_rate_change_3m   REAL,  -- 3-month delta in fed funds rate
    credit_spread        REAL,  -- BAA - AAA spread (risk appetite proxy)
    credit_spread_change REAL,  -- 1-month delta

    -- Economic Health
    gdp_growth           REAL,  -- QoQ annualized GDP growth
    unemployment_rate    REAL,
    unemployment_change  REAL,  -- 3-month delta
    cpi_yoy              REAL,  -- Year-over-year inflation
    cpi_change_3m        REAL,  -- 3-month inflation momentum
    initial_claims       REAL,  -- Weekly jobless claims (ICSA)
    initial_claims_4wk_avg REAL,

    -- Market Regime
    vix_level            REAL,  -- VIX (VIXCLS)
    vix_change_5d        REAL,
    vix_percentile_1y    REAL,  -- Where current VIX sits in 1-year distribution
    market_breadth       REAL,  -- % of S&P 500 stocks above their 200d MA
    dollar_index         REAL,  -- USD strength (DTWEXBGS)
    dollar_change_30d    REAL,
    m2_growth            REAL,  -- YoY money supply growth (M2SL)

    -- Sector ETF Momentum (30-day returns)
    sector_momentum_xlk  REAL,  -- Technology 30d return
    sector_momentum_xlf  REAL,  -- Financials 30d return
    sector_momentum_xlv  REAL,  -- Healthcare 30d return
    sector_momentum_xle  REAL,  -- Energy 30d return
    sector_momentum_xli  REAL,  -- Industrials 30d return
    sector_momentum_xlp  REAL,  -- Consumer Staples 30d return
    sector_momentum_xly  REAL,  -- Consumer Discretionary 30d return
    growth_vs_value      REAL,  -- IWF vs IWD relative performance
    -- Sector relative flow: sector return minus SPY return (outperformance)
    sector_relative_flow_xlk  REAL,
    sector_relative_flow_xlf  REAL,
    sector_relative_flow_xlv  REAL,
    sector_relative_flow_xle  REAL,
    sector_relative_flow_xli  REAL,
    sector_relative_flow_xlp  REAL,
    sector_relative_flow_xly  REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- sentiment_features
-- 20-30 features per stock per day.
-- Source: Reddit (PRAW), news RSS feeds (feedparser), SEC Form 4 / 8-K.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id    INTEGER NOT NULL,
    date        TEXT    NOT NULL,

    -- Reddit Sentiment
    reddit_mention_count_7d  INTEGER,  -- Mentions across target subreddits
    reddit_mention_change    REAL,     -- Week-over-week change
    reddit_sentiment_avg_7d  REAL,     -- Avg VADER compound score (-1 to +1)
    reddit_sentiment_std_7d  REAL,     -- Std dev (high = disagreement)
    reddit_bullish_ratio     REAL,     -- % of posts with positive sentiment
    reddit_post_score_avg    REAL,     -- Avg upvote score
    reddit_comment_ratio     REAL,     -- Comments per post
    wsb_mention_flag         INTEGER,  -- 1 = mentioned on r/wallstreetbets

    -- News Sentiment
    news_volume_7d           INTEGER,
    news_volume_change       REAL,
    news_sentiment_avg_7d    REAL,
    news_sentiment_30d       REAL,
    news_sentiment_momentum  REAL,    -- 7d sentiment - 30d sentiment
    negative_news_flag       INTEGER, -- 1 = highly negative article detected

    -- Institutional Sentiment (derived from SEC filings)
    insider_buy_count_30d    INTEGER,
    insider_sell_count_30d   INTEGER,
    insider_net_sentiment    REAL,    -- (Buys - Sells) / Total transactions
    insider_dollar_volume    REAL,    -- Total $ value of insider buys
    activist_filing_flag     INTEGER, -- 1 = new 13D/13G in last 90 days
    activist_stake_change    REAL,    -- 13D/A stake size change
    material_event_count_30d INTEGER, -- Number of 8-K filings in 30 days
    negative_8k_flag         INTEGER, -- 1 = restatement, investigation, exec departure

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id),
    UNIQUE(stock_id, date)
);


-- -----------------------------------------------------------------------------
-- predictions
-- Model output per stock per prediction date.
-- Stores raw probabilities from both models plus ensemble, and derived signals.
-- model_version lets us track which model generated a prediction (for auditing).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id        INTEGER NOT NULL,
    prediction_date TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,  -- e.g. 'xgb_v1_lstm_v1_ensemble'

    -- Ensemble probabilities (4 classes: big_loss, flat, moderate_gain, strong_gain)
    prob_big_loss        REAL,
    prob_flat            REAL,
    prob_moderate_gain   REAL,
    prob_strong_gain     REAL,

    -- XGBoost raw probabilities
    xgb_prob_big_loss      REAL,
    xgb_prob_flat          REAL,
    xgb_prob_moderate_gain REAL,
    xgb_prob_strong_gain   REAL,

    -- LSTM raw probabilities
    lstm_prob_big_loss      REAL,
    lstm_prob_flat          REAL,
    lstm_prob_moderate_gain REAL,
    lstm_prob_strong_gain   REAL,

    -- Derived signals
    predicted_class     INTEGER,  -- 0=big_loss, 1=flat, 2=moderate_gain, 3=strong_gain
    confidence          REAL,     -- Probability assigned to the predicted class
    expected_return_pct REAL,     -- Weighted expected return across buckets
    bull_probability    REAL,     -- P(>5% gain) = prob_moderate + prob_strong
    score               INTEGER,  -- 0-100 composite score
    action              TEXT,     -- 'STRONG BUY' / 'BUY' / 'WATCH' / 'NEUTRAL' / 'SELL'

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id),
    UNIQUE(stock_id, prediction_date, model_version)
);


-- -----------------------------------------------------------------------------
-- positions
-- Portfolio holdings — both open and closed.
-- exit_reason captures WHY we exited (for post-trade analysis).
-- entry_score / entry_confidence let us compare model conviction at entry
-- vs. what the model thinks now (for thesis health tracking in Section 14).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id              INTEGER NOT NULL,
    entry_date            TEXT    NOT NULL,
    exit_date             TEXT,                 -- NULL if still open
    entry_price           REAL    NOT NULL,
    exit_price            REAL,
    shares                REAL    NOT NULL,
    position_size_usd     REAL,
    entry_score           INTEGER,              -- Model score at time of entry
    entry_confidence      REAL,
    entry_expected_return REAL,
    status                TEXT DEFAULT 'open',  -- 'open' or 'closed'
    exit_reason           TEXT,                 -- 'stop_loss' / 'profit_target' /
                                                -- 'model_signal' / 'time_exit' / 'manual'
    realized_return_pct   REAL,                 -- Populated on close
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id)
);


-- -----------------------------------------------------------------------------
-- backtest_results
-- One row per date per backtest run. run_id groups all rows from one run.
-- Storing SPY side-by-side lets us compute alpha at any date without
-- needing to re-fetch benchmark data.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,  -- UUID grouping a full backtest run
    date              TEXT NOT NULL,
    portfolio_value   REAL,
    cash              REAL,
    invested          REAL,
    spy_value         REAL,          -- SPY benchmark at same starting capital
    alpha             REAL,          -- portfolio_return - spy_return
    n_positions       INTEGER,
    daily_return      REAL,
    cumulative_return REAL,
    sharpe_ratio      REAL,          -- rolling Sharpe as of this date
    max_drawdown      REAL,          -- rolling peak-to-trough as of this date
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, date)
);


-- =============================================================================
-- Indexes
-- Without these, every lookup scans the entire table row-by-row.
-- With them, SQLite builds a B-tree for fast lookups on those columns.
-- Rule of thumb: index every column you filter or join on frequently.
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_prices_stock_date       ON daily_prices(stock_id, date);
CREATE INDEX IF NOT EXISTS idx_fundamental_stock_date  ON fundamental_features(stock_id, date);
CREATE INDEX IF NOT EXISTS idx_fundamental_sector_date ON fundamental_features(date);  -- sector-relative lookups
CREATE INDEX IF NOT EXISTS idx_technical_stock_date    ON technical_features(stock_id, date);
CREATE INDEX IF NOT EXISTS idx_sentiment_stock_date    ON sentiment_features(stock_id, date);
CREATE INDEX IF NOT EXISTS idx_macro_date              ON macro_features(date);
CREATE INDEX IF NOT EXISTS idx_predictions_stock_date  ON predictions(stock_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_positions_stock_status  ON positions(stock_id, status);
CREATE INDEX IF NOT EXISTS idx_filings_stock_type_date ON sec_filings(stock_id, form_type, filed_date);
CREATE INDEX IF NOT EXISTS idx_filings_negative        ON sec_filings(stock_id, is_negative_8k, filed_date);
CREATE INDEX IF NOT EXISTS idx_insider_stock_date      ON insider_transactions(stock_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_backtest_run            ON backtest_results(run_id, date);
CREATE INDEX IF NOT EXISTS idx_stocks_sector           ON stocks(sector, is_active);  -- sector grouping

-- =============================================================================
-- pipeline_runs
-- One row per pipeline execution. Tracks per-source success/failure counts,
-- staleness warnings, and QA results. Used for monitoring and debugging.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT    NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          TEXT    DEFAULT 'running',  -- running | success | partial | failed
    stocks_total    INTEGER DEFAULT 0,
    stocks_ok       INTEGER DEFAULT 0,
    stocks_failed   INTEGER DEFAULT 0,
    price_rows      INTEGER DEFAULT 0,
    sec_filings     INTEGER DEFAULT 0,
    fund_rows       INTEGER DEFAULT 0,
    tech_rows       INTEGER DEFAULT 0,
    sent_rows       INTEGER DEFAULT 0,
    macro_updated   INTEGER DEFAULT 0,
    stale_tickers   TEXT,   -- JSON list of tickers with stale price data
    qa_warnings     TEXT,   -- JSON list of QA warnings
    error_log       TEXT    -- JSON list of error messages
);

-- =============================================================================
-- Schema version table
-- Tracks applied migrations so future schema changes can be incremental.
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    description TEXT    NOT NULL,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
