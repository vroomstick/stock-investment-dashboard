# Stock Investment Dashboard — CLAUDE.md

You are working on an ML-powered stock investment prediction system. Follow the spec exactly. Do not add features, libraries, or patterns not in the spec.

## Project Overview

XGBoost + LSTM ensemble trained on 200+ features from SEC EDGAR, yfinance, FRED, and Reddit to predict 3-6 month stock returns for ~100 large-cap US stocks. Generates weekly buy/sell signals with confidence scoring.

## Tech Stack (do not change)

- Python 3.10+
- SQLite (database/stock_dashboard.db)
- XGBoost (tabular model)
- PyTorch (LSTM sequential model)
- pandas, numpy (data manipulation)
- yfinance (price data, no API key)
- fredapi (macro data, free API key in .env)
- VADER/nltk (sentiment analysis)
- PRAW (Reddit API, credentials in .env)
- feedparser (RSS news)
- SHAP (feature importance)
- matplotlib, seaborn (visualization)
- jinja2 + smtplib (weekly reports)

## Project Structure

```
stock-dashboard/
├── config/           # settings.py, stock_universe.json, feature_config.yaml
├── data/
│   ├── collectors/   # sec_edgar.py, price_data.py, macro_data.py, reddit_sentiment.py, rss_news.py
│   └── feature_store/ # fundamental.py, technical.py, macro.py, sentiment.py
├── models/           # xgboost_model.py, lstm_model.py, ensemble.py, backtester.py, etc.
├── signals/          # scoring.py, position_sizer.py, risk_manager.py, exit_engine.py
├── reports/          # weekly_report.py, alert_engine.py
├── scripts/          # daily_collect.py, weekly_train.py, weekly_predict.py, setup_db.py
├── database/         # db.py, schema.sql
├── notebooks/        # 01_eda, 02_feature_analysis, 03_model_comparison, 04_backtest_results
└── tests/            # test_features.py, test_models.py, test_signals.py
```

## Key Architecture Decisions

- Feature store in SQLite: 200+ features per stock per day
- Walk-forward validation, NEVER random splits (financial data is time-ordered)
- 4-class target: big_loss (<-5%), flat (-5% to +5%), moderate_gain (+5% to +15%), strong_gain (>+15%)
- XGBoost gets 60% ensemble weight, LSTM gets 40%
- LSTM input: 60-day sequences of ~40 temporal features
- All data sources are free ($0 budget constraint)

## Coding Conventions

- Every file starts with a docstring explaining what it does and why
- Use type hints on all function signatures
- Handle missing data source-specifically: fundamentals forward-fill then sector median, technicals forward-fill then zero, sentiment zero-fill, macro forward-fill
- SEC EDGAR rate limit: 10 req/sec with exponential backoff
- Never use look-ahead data — all features must be point-in-time
- Store raw data separately from computed features
- Log errors, don't crash the pipeline for one stock's failure

## What NOT to Build

- No web UI/dashboard (CLI + email + Jupyter only)
- No real-time streaming (daily batch is sufficient)
- No cloud infrastructure (local laptop)
- No paid data sources
- No transformer/BERT models for NLP
- No reinforcement learning

## Running the Project

```bash
# First time setup
python scripts/setup_db.py

# Daily data collection
python scripts/daily_collect.py

# Weekly training
python scripts/weekly_train.py

# Weekly predictions
python scripts/weekly_predict.py
```

## Environment Variables (.env)

```
FRED_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=stock-dashboard
SEC_USER_AGENT=YourName youremail@example.com
```

## Agent Workflow

Before writing any code, read agent-workspace/LOOP.md for the execution loop. Before starting work, read agent-workspace/CHECKLIST.md for the current task list and agent-workspace/DEVLOG.md for recent context. Always orient against the PRD before starting a new task.
