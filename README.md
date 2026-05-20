# Stock Investment Dashboard — ML-First

An end-to-end machine learning system that predicts 3–6 month stock returns using 200+ features across fundamental analysis, technical indicators, macroeconomic data, and sentiment signals.

## Architecture

```
Data Sources → Collectors → Feature Store (SQLite) → ML Models → Signals → Reports
     ↑                              ↑                      ↑
  FRED API               daily_prices              XGBoost (tabular)
  SEC EDGAR              fundamental_features      LSTM (sequential)
  yfinance               technical_features        Ensemble (60/40)
  Reddit/RSS             macro_features
                         sentiment_features
```

**Two models, one ensemble:**
- **XGBoost**: 200+ feature snapshot per stock per day. Handles non-linear interactions between fundamental, technical, macro, and sentiment signals.
- **LSTM**: 60-day sequence of 18 temporal features per stock. Captures momentum patterns and sequential signals that a point-in-time snapshot misses.
- **Ensemble**: 60% XGBoost + 40% LSTM weighted average.

---

## Setup

### 1. Create the environment

```bash
conda create -n mlspec python=3.11
conda activate mlspec
pip install -r requirements.txt
pip install -e .      # installs project as editable package (no sys.path hacks)
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in:
#   FRED_API_KEY     — free at https://fred.stlouisfed.org/docs/api/api_key.html
#   REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT (optional)
```

### 3. Initialize the database

```bash
python scripts/setup_db.py
```

This creates `database/stock_dashboard.db` and seeds 30 large-cap stocks + SPY + 7 sector ETFs.

### 4. Run the pipeline

```bash
# Full daily run
python scripts/daily_collect.py

# Specific date (for historical backfill)
python scripts/daily_collect.py --date 2025-01-15

# Quick test — skip FRED API call
python scripts/daily_collect.py --skip-macro --tickers AAPL MSFT
```

---

## Daily Cron Setup

Run after market close (6:30 PM ET, weekdays):

```cron
# Edit with: crontab -e
30 18 * * 1-5 cd /Users/varun/Documents/MLspec && \
  conda run -n mlspec python scripts/daily_collect.py >> logs/cron.log 2>&1
```

Log files rotate daily: `logs/daily_collect_YYYY-MM-DD.log`

Exit codes:
- `0` — all stocks processed successfully
- `1` — partial success (some stocks failed, majority completed)
- `2` — critical failure (pipeline could not run)

---

## Feature Store

| Category | Table | # Features | Update Frequency |
|---|---|---|---|
| Fundamental | `fundamental_features` | 69 | Quarterly (earnings) |
| Technical | `technical_features` | 61 | Daily |
| Macro | `macro_features` | 36 | Daily/Weekly (FRED cadence) |
| Sentiment | `sentiment_features` | 22 | Daily |
| **Total** | | **188** | |

Key features by category:

**Fundamental:** ROE, ROA, ROIC, Piotroski F-Score (9 components), Altman Z-Score, P/E, EV/EBITDA, FCF yield, net debt/EBITDA, sector-relative valuations

**Technical:** RSI (7, 14), MACD + histogram, Bollinger %B, ATR, OBV, MFI, CCI, Stochastic %K/%D, 52-week high/low proximity, beta (60d), sector relative strength

**Macro:** Yield curve slope (10Y–2Y), VIX percentile, credit spread (BAA–AAA), GDP growth, CPI YoY, sector ETF momentum vs SPY

**Sentiment:** Reddit VADER scores, RSS news sentiment momentum, insider buy/sell counts, Form 4 dollar volume, 8-K negative event flag

---

## Database Schema

```
stocks               — 30 large-caps + SPY + 7 sector ETFs
daily_prices         — 5yr OHLCV history per stock
sec_filings          — Form 4, 8-K, SC 13D/G filing log
insider_transactions — Parsed Form 4 buy/sell transactions
fundamental_features — 69 features per stock per quarter
technical_features   — 61 features per stock per day
macro_features       — 36 features per date (stock-agnostic)
sentiment_features   — 22 features per stock per day
predictions          — Model output per stock per date
positions            — Active portfolio positions
backtest_results     — Walk-forward backtest performance
pipeline_runs        — Per-run health metrics and QA results
schema_version       — Migration tracking
```

---

## Model Training

```bash
# Weekly retrain (Sundays)
python scripts/weekly_train.py

# Generate predictions + report
python scripts/weekly_predict.py
```

Walk-forward cross-validation: train on rolling 2-year windows, test on next quarter. 1-month gap between train end and test start to prevent lookahead bias.

Labels: 4-class classification on 90-day forward returns
- Class 0: < -5% (big loss)
- Class 1: -5% to +5% (flat)
- Class 2: +5% to +15% (moderate gain)
- Class 3: > +15% (strong gain)

---

## Schema Migrations

Schema changes are tracked in the `schema_version` table. To apply a migration:

```bash
python scripts/migrate.py --version 4 --sql "ALTER TABLE ..."
```

Current schema version: **3**

| Version | Description |
|---|---|
| 1 | Initial schema |
| 2 | Add `sector_etf` to stocks, `is_negative_8k` to sec_filings, new indexes |
| 3 | Add `pipeline_runs` table |

---

## Data Sources

All free, no paid subscriptions required:

| Source | Data | Library |
|---|---|---|
| SEC EDGAR | Form 4 (insider transactions), 8-K, SC 13D, XBRL financials | `requests` |
| yfinance | OHLCV prices, fundamentals | `yfinance` |
| FRED | 12 macro series (VIX, yield curve, CPI, GDP) | `fredapi` |
| Reddit | Wallstreetbets/investing sentiment | `praw` + VADER |
| RSS feeds | Reuters, CNBC, MarketWatch headlines | `feedparser` + VADER |

SEC rate limit: 8 req/sec (limit is 10, we run at 80% for safety). Exponential backoff on 503s.

---

## Project Structure

```
config/           — settings.py, stock_universe.json, feature_config.yaml
data/
  collectors/     — one file per data source
  feature_store/  — transforms raw data into ML features
  pipeline.py     — daily orchestrator
database/         — schema.sql, db.py
logs/             — rotating daily logs (gitignored)
models/           — xgboost, lstm, ensemble, preprocessing, backtester
notebooks/        — EDA and model analysis (01_eda, 02_features, 03_models, 04_backtest)
reports/          — weekly report generator + alert engine
scripts/          — cron entry points
signals/          — scoring, position sizing, risk management, exit engine
tests/            — unit tests
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Notes

- All API keys are loaded from `.env` — never committed to git
- `NOTES.md` is a private learning document — gitignored
- Model binaries (`.json`, `.pkl`, `.pt`) are gitignored; retrain from scratch or store separately
- The project uses `pip install -e .` for clean cross-module imports — no `sys.path` hacks
