"""
data/collectors/price_data.py

Fetches price history and fundamental data from yfinance for all stocks.

yfinance is a Python wrapper around Yahoo Finance's unofficial API.
No API key required — it scrapes Yahoo's backend.

Two kinds of data:
1. Daily OHLCV (Open/High/Low/Close/Volume) — stored in daily_prices
2. Quarterly fundamentals (income statement, balance sheet, cash flow)
   — passed to the fundamental feature store for feature computation

Why store raw prices separately from computed features?
Because features are derived from prices. If we add a new indicator later,
we recompute from the stored raw prices without re-fetching from Yahoo.
"""

import os
import time
from datetime import date, timedelta


import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config.settings import DATA
from database.db import get_db, fetch_one, executemany


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_price_history(ticker: str, years: int = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV history from yfinance.

    adj_close (adjusted close) accounts for stock splits and dividends.
    Always use adj_close for return calculations — raw close will give
    you a false 50% drop on a 2-for-1 split day.

    Returns a DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    Index: DatetimeIndex
    """
    years = years or DATA["price_history_years"]
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{years}y", auto_adjust=False)

    if hist.empty:
        print(f"    {ticker}: no price data returned from yfinance")
        return pd.DataFrame()

    return hist


def store_price_history(stock_id: int, ticker: str, hist: pd.DataFrame):
    """
    Upsert daily OHLCV rows into daily_prices.
    INSERT OR IGNORE skips dates already in the database (idempotent).
    """
    if hist.empty:
        return 0

    rows = []
    for dt, row in hist.iterrows():
        date_str = dt.strftime("%Y-%m-%d")
        rows.append((
            stock_id,
            date_str,
            round(float(row.get("Open",  0) or 0), 4),
            round(float(row.get("High",  0) or 0), 4),
            round(float(row.get("Low",   0) or 0), 4),
            round(float(row.get("Close", 0) or 0), 4),
            round(float(row.get("Adj Close", row.get("Close", 0)) or 0), 4),
            int(row.get("Volume", 0) or 0),
        ))

    executemany(
        """INSERT OR IGNORE INTO daily_prices
           (stock_id, date, open, high, low, close, adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def get_latest_stored_date(stock_id: int) -> str | None:
    """Return the most recent date already in daily_prices for this stock."""
    row = fetch_one(
        "SELECT MAX(date) as max_date FROM daily_prices WHERE stock_id = ?",
        (stock_id,)
    )
    return row["max_date"] if row else None


def fetch_and_store_prices(stock_id: int, ticker: str, as_of_date: str = None):
    """
    Incremental price update: fetch only what's missing.

    On first run: fetches full history (5 years).
    On subsequent runs: fetches only since last stored date.
    as_of_date: used as the "today" anchor for staleness checks so that
                historical pipeline reruns don't compare against actual today.
    """
    anchor = date.fromisoformat(as_of_date) if as_of_date else date.today()
    latest = get_latest_stored_date(stock_id)

    if latest is None:
        print(f"  {ticker}: initial load ({DATA['price_history_years']}yr history)...")
        hist = fetch_price_history(ticker)
    else:
        days_missing = (anchor - date.fromisoformat(latest)).days
        if days_missing <= 1:
            print(f"  {ticker}: prices up to date ({latest})")
            return
        print(f"  {ticker}: fetching {days_missing} missing days since {latest}...")
        t = yf.Ticker(ticker)
        hist = t.history(start=latest, end=anchor.isoformat(), auto_adjust=False)

    count = store_price_history(stock_id, ticker, hist)
    print(f"  {ticker}: stored {count} price rows")


# ---------------------------------------------------------------------------
# Fundamental data (raw — passed to feature store for computation)
# ---------------------------------------------------------------------------

def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data from yfinance.

    Returns a dict with three DataFrames (quarterly statements) and
    one dict (company info). The feature store computes ratios from these.

    Why not compute features here?
    Separation of concerns. This collector's job is to get data.
    The feature store's job is to compute features from that data.
    Mixing them makes both harder to test and debug.
    """
    t = yf.Ticker(ticker)

    fundamentals = {
        "info":       {},
        "financials": pd.DataFrame(),   # Income statement (quarterly)
        "balance":    pd.DataFrame(),   # Balance sheet (quarterly)
        "cashflow":   pd.DataFrame(),   # Cash flow statement (quarterly)
    }

    try:
        fundamentals["info"] = t.info or {}
    except Exception as e:
        print(f"    {ticker}: could not fetch info — {e}")

    try:
        fundamentals["financials"] = t.quarterly_financials
    except Exception as e:
        print(f"    {ticker}: could not fetch quarterly financials — {e}")

    try:
        fundamentals["balance"] = t.quarterly_balance_sheet
    except Exception as e:
        print(f"    {ticker}: could not fetch balance sheet — {e}")

    try:
        fundamentals["cashflow"] = t.quarterly_cashflow
    except Exception as e:
        print(f"    {ticker}: could not fetch cash flow — {e}")

    return fundamentals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_stock(stock_id: int, ticker: str, as_of_date: str = None):
    """
    Fetch and store price history for one stock.
    Fundamental data is returned (not stored — the feature store handles that).
    Called by data/pipeline.py.
    """
    try:
        fetch_and_store_prices(stock_id, ticker, as_of_date)
    except Exception as e:
        print(f"  {ticker}: price fetch failed — {e}")


def get_fundamentals(ticker: str) -> dict:
    """
    Public interface for the fundamental feature store to call.
    Returns raw fundamental data for a single ticker.
    """
    try:
        return fetch_fundamentals(ticker)
    except Exception as e:
        print(f"  {ticker}: fundamental fetch failed — {e}")
        return {"info": {}, "financials": pd.DataFrame(),
                "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}
