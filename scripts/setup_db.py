"""
scripts/setup_db.py

One-time setup script. Run this before anything else:
    conda activate mlspec
    python scripts/setup_db.py

What it does:
1. Creates the SQLite database and all tables (via schema.sql)
2. Seeds the stocks table from config/stock_universe.json
"""

import os
import json

# Make project root importable from scripts/

from dotenv import load_dotenv
load_dotenv()  # Load .env before importing settings (settings reads env vars)

from database.db import init_db, executemany, fetch_all
from config.settings import PROJECT_ROOT


def seed_stocks():
    path = PROJECT_ROOT / "config" / "stock_universe.json"
    with open(path) as f:
        stocks = json.load(f)["stocks"]

    rows = [
        (
            s["ticker"],
            s.get("cik"),
            s["company"],
            s["sector"],
            s.get("sector_etf"),
            s.get("is_active", 1),   # default active; ETFs and benchmarks may set 0
        )
        for s in stocks
    ]

    executemany(
        """INSERT OR IGNORE INTO stocks (ticker, cik, company_name, sector, sector_etf, is_active)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def main():
    print("=" * 50)
    print("Stock Dashboard — First-Time Setup")
    print("=" * 50)

    print("\n[1/2] Initializing database...")
    init_db()

    print("[2/2] Seeding stock universe...")
    count = seed_stocks()

    rows = fetch_all(
        "SELECT ticker, sector FROM stocks ORDER BY sector, ticker"
    )

    current_sector = None
    for r in rows:
        if r["sector"] != current_sector:
            current_sector = r["sector"]
            print(f"\n  {current_sector}:")
        print(f"    {r['ticker']}")

    print(f"\nDone. {count} stocks loaded into database.")
    print(f"Database: {PROJECT_ROOT / 'database' / 'stock_dashboard.db'}\n")


if __name__ == "__main__":
    main()
