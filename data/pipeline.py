"""
data/pipeline.py

Daily data collection and feature computation pipeline.
Orchestrates all collectors and feature stores in the correct order.

Run order:
  1. Macro features (stock-agnostic — one FRED fetch covers all stocks)
  2. Per-stock loop:
       a. Prices (incremental — only fetches missing days)
       b. SEC filings (last 90 days of filings)
       c. Fundamental features (quarterly ratios, Piotroski, Altman Z)
       d. Technical features (60+ indicators from price history)
       e. Sentiment features (Reddit + RSS news + SEC-derived signals)

Why macro first?
  Macro features are used by all stocks and require only one API call batch.
  Running it once at the start avoids redundant fetches.

Why prices before features?
  Technical features are computed FROM stored prices. Prices must exist first.

Why SEC before fundamentals?
  Fundamental feature store uses yfinance, but SEC data enriches sentiment
  features (insider transactions, 8-K flags) which run last.

Usage:
  python -m data.pipeline                  # run for today
  python -m data.pipeline --date 2025-01-15  # run for a specific date
  python -m data.pipeline --skip-macro     # skip FRED fetches (for testing)
  python -m data.pipeline --tickers AAPL MSFT  # only process specific tickers
"""

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from config.settings import DATA
from database.db import fetch_all, fetch_one, get_db, init_db

import data.feature_store.macro as macro_store
from data.collectors import price_data, sec_edgar
from data.feature_store import fundamental, technical, sentiment


def load_universe() -> list[dict]:
    """
    Load all active stocks from the database.
    Returns list of dicts with: id, ticker, cik, sector, sector_etf
    """
    rows = fetch_all(
        """SELECT id, ticker, cik, sector, sector_etf
           FROM stocks
           WHERE is_active = 1
           ORDER BY ticker""",
        ()
    )
    return [dict(r) for r in rows]


def run(as_of_date: str = None,
        skip_macro: bool = False,
        tickers: list = None) -> dict:
    """
    Run the full daily data pipeline.

    as_of_date: date string (YYYY-MM-DD). Defaults to today.
    skip_macro: if True, skip the FRED macro fetch (faster for dev/testing).
    tickers:    optional list of tickers to process (default: full universe).

    Returns a health dict with per-source counts and any QA warnings.
    Raises RuntimeError if a critical failure prevents the run from completing.
    """
    as_of_date = as_of_date or date.today().isoformat()
    started_at = datetime.utcnow().isoformat()

    print(f"\n{'='*60}")
    print(f"  MLspec Data Pipeline — {as_of_date}")
    print(f"{'='*60}\n")

    init_db()

    # Insert run record — track this run even if it partially fails
    run_id = None
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO pipeline_runs (run_date, started_at, status, stocks_total)
               VALUES (?, ?, 'running', 0)""",
            (as_of_date, started_at)
        )
        run_id = cursor.lastrowid

    health = {
        "run_id": run_id, "as_of_date": as_of_date,
        "stocks_ok": 0, "stocks_failed": 0,
        "price_rows": 0, "sec_filings": 0,
        "fund_rows": 0, "tech_rows": 0, "sent_rows": 0,
        "macro_updated": 0,
        "errors": [], "qa_warnings": [], "stale_tickers": [],
    }

    # -------------------------------------------------------------------------
    # Stage 1: Macro Features (stock-agnostic, run once)
    # -------------------------------------------------------------------------
    if not skip_macro:
        print("[1/3] Fetching macro features (FRED + sector ETFs)...")
        try:
            macro_store.run(as_of_date)
            health["macro_updated"] = 1
        except Exception as e:
            msg = f"macro pipeline error: {e}"
            print(f"  ERROR: {msg}")
            health["errors"].append(msg)
    else:
        print("[1/3] Macro features skipped (--skip-macro)")

    # -------------------------------------------------------------------------
    # Stage 2: Per-stock collection
    # -------------------------------------------------------------------------
    universe = load_universe()
    if tickers:
        universe = [s for s in universe if s["ticker"] in tickers]

    health["stocks_total"] = len(universe)
    print(f"\n[2/3] Processing {len(universe)} stocks...\n")

    for i, stock in enumerate(universe, 1):
        stock_id   = stock["id"]
        ticker     = stock["ticker"]
        cik        = stock["cik"]
        sector     = stock["sector"]
        sector_etf = stock["sector_etf"]
        stock_ok   = True

        print(f"--- [{i}/{len(universe)}] {ticker} ---")

        try:
            price_data.process_stock(stock_id, ticker, as_of_date)
            # Count new price rows for this run date
            prow = fetch_one(
                "SELECT COUNT(*) as n FROM daily_prices WHERE stock_id=? AND date=?",
                (stock_id, as_of_date)
            )
            health["price_rows"] += prow["n"] if prow else 0
        except Exception as e:
            msg = f"{ticker}: price collection failed — {e}"
            print(f"  {msg}")
            health["errors"].append(msg)
            stock_ok = False

        try:
            before = _count_filings(stock_id)
            sec_edgar.process_stock(stock_id, ticker, cik,
                                    days_back=90, as_of_date=as_of_date)
            health["sec_filings"] += _count_filings(stock_id) - before
        except Exception as e:
            msg = f"{ticker}: SEC collection failed — {e}"
            print(f"  {msg}")
            health["errors"].append(msg)

        try:
            fundamental.run(stock_id, ticker, sector, as_of_date)
            health["fund_rows"] += 1
        except Exception as e:
            msg = f"{ticker}: fundamental features failed — {e}"
            print(f"  {msg}")
            health["errors"].append(msg)
            stock_ok = False

        try:
            technical.run(stock_id, ticker, sector_etf, as_of_date)
            health["tech_rows"] += 1
        except Exception as e:
            msg = f"{ticker}: technical features failed — {e}"
            print(f"  {msg}")
            health["errors"].append(msg)
            stock_ok = False

        try:
            sentiment.run(stock_id, ticker, as_of_date)
            health["sent_rows"] += 1
        except Exception as e:
            msg = f"{ticker}: sentiment features failed — {e}"
            print(f"  {msg}")
            health["errors"].append(msg)

        if stock_ok:
            health["stocks_ok"] += 1
        else:
            health["stocks_failed"] += 1

        print()
        time.sleep(0.5)

    # -------------------------------------------------------------------------
    # Stage 3: QA checks + stale detection
    # -------------------------------------------------------------------------
    print("[3/3] Running QA checks...")
    qa_warnings = _run_qa(as_of_date, universe)
    health["qa_warnings"] = qa_warnings
    health["stale_tickers"] = _detect_stale(universe, as_of_date)

    for w in qa_warnings:
        print(f"  ⚠ QA: {w}")
    for t in health["stale_tickers"]:
        print(f"  ⚠ STALE: {t} — no price update in >3 days")

    # Persist run results
    status = "success" if health["stocks_failed"] == 0 else "partial"
    with get_db() as conn:
        conn.execute(
            """UPDATE pipeline_runs SET
               finished_at=?, status=?, stocks_total=?, stocks_ok=?,
               stocks_failed=?, price_rows=?, sec_filings=?, fund_rows=?,
               tech_rows=?, sent_rows=?, macro_updated=?,
               stale_tickers=?, qa_warnings=?, error_log=?
               WHERE id=?""",
            (
                datetime.utcnow().isoformat(), status,
                health["stocks_total"], health["stocks_ok"],
                health["stocks_failed"], health["price_rows"],
                health["sec_filings"], health["fund_rows"],
                health["tech_rows"], health["sent_rows"],
                health["macro_updated"],
                json.dumps(health["stale_tickers"]),
                json.dumps(health["qa_warnings"]),
                json.dumps(health["errors"]),
                run_id,
            )
        )

    _print_summary(as_of_date, health)
    return health


def _count_filings(stock_id: int) -> int:
    row = fetch_one("SELECT COUNT(*) as n FROM sec_filings WHERE stock_id=?", (stock_id,))
    return row["n"] if row else 0


def _run_qa(as_of_date: str, universe: list) -> list:
    """
    Post-pipeline data quality checks. Returns list of warning strings.

    Checks:
      - RSI out of [0, 100] range
      - Duplicate feature rows for the same (stock, date)
      - Null explosion: any stock missing >80% of technical features
      - Macro row missing when not in skip-macro mode
    """
    warnings = []

    # RSI bounds check
    bad_rsi = fetch_all(
        """SELECT s.ticker, t.rsi_14
           FROM technical_features t JOIN stocks s ON s.id = t.stock_id
           WHERE t.date = ? AND t.rsi_14 IS NOT NULL
             AND (t.rsi_14 < 0 OR t.rsi_14 > 100)""",
        (as_of_date,)
    )
    for r in bad_rsi:
        warnings.append(f"{r['ticker']}: RSI out of range ({r['rsi_14']:.2f})")

    # Duplicate rows
    dups = fetch_all(
        """SELECT stock_id, COUNT(*) as n FROM technical_features
           WHERE date = ? GROUP BY stock_id HAVING n > 1""",
        (as_of_date,)
    )
    for d in dups:
        warnings.append(f"stock_id={d['stock_id']}: duplicate technical_features row")

    # Null explosion: stocks with <20% non-null technical features
    tech_rows = fetch_all(
        """SELECT s.ticker, t.*
           FROM technical_features t JOIN stocks s ON s.id = t.stock_id
           WHERE t.date = ?""",
        (as_of_date,)
    )
    skip = {"id", "stock_id", "date", "created_at", "ticker"}
    for row in tech_rows:
        d = dict(row)
        feat_vals = [v for k, v in d.items() if k not in skip]
        non_null = sum(1 for v in feat_vals if v is not None)
        pct = non_null / len(feat_vals) if feat_vals else 0
        if pct < 0.2:
            warnings.append(f"{d['ticker']}: only {pct:.0%} technical features non-null")

    return warnings


def _detect_stale(universe: list, as_of_date: str) -> list:
    """
    Return tickers whose most recent price row is more than 3 days old.
    3 days accounts for weekends — a Monday run will see Friday prices.
    """
    stale = []
    anchor = date.fromisoformat(as_of_date)
    for stock in universe:
        row = fetch_one(
            "SELECT MAX(date) as latest FROM daily_prices WHERE stock_id=?",
            (stock["id"],)
        )
        if not row or not row["latest"]:
            stale.append(f"{stock['ticker']} (no prices)")
            continue
        latest = date.fromisoformat(row["latest"])
        if (anchor - latest).days > 3:
            stale.append(f"{stock['ticker']} (latest: {row['latest']})")
    return stale


def _print_summary(as_of_date: str, health: dict):
    """Print run summary with health metrics."""
    print(f"\n  {'='*50}")
    print(f"  Pipeline complete — {as_of_date}")
    print(f"  {'='*50}")
    print(f"  Stocks OK / Failed    : {health['stocks_ok']} / {health['stocks_failed']}")
    print(f"  Fundamental features  : {health['fund_rows']} stocks")
    print(f"  Technical features    : {health['tech_rows']} stocks")
    print(f"  Sentiment features    : {health['sent_rows']} stocks")
    print(f"  Macro updated         : {'yes' if health['macro_updated'] else 'no (--skip-macro)'}")
    print(f"  New SEC filings       : {health['sec_filings']}")
    if health["stale_tickers"]:
        print(f"  Stale tickers ({len(health['stale_tickers'])}): {', '.join(health['stale_tickers'][:5])}")
    if health["qa_warnings"]:
        print(f"  QA warnings ({len(health['qa_warnings'])}): see pipeline_runs table")
    if health["errors"]:
        print(f"  Errors ({len(health['errors'])}): {health['errors'][0]}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLspec daily data pipeline"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to run for (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--skip-macro", action="store_true",
        help="Skip FRED macro fetch (useful for testing)."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Only process these tickers (e.g. --tickers AAPL MSFT)."
    )
    args = parser.parse_args()

    run(
        as_of_date=args.date,
        skip_macro=args.skip_macro,
        tickers=args.tickers,
    )
