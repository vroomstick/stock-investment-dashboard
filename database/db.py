"""
database/db.py

Single source of truth for database access. Every module in this project
imports get_connection() from here — nothing else opens the database directly.

Why centralize this?
- One place to change the DB path, timeout, or connection settings
- Foreign keys are enabled once here, not scattered across every file
- Row factory is set once here so every query returns dicts, not tuples
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

# Path to the SQLite database file, in the same directory as this module.
DB_PATH = str(Path(__file__).parent / "stock_dashboard.db")


def get_connection() -> sqlite3.Connection:
    """
    Open and return a connection to the SQLite database.

    Two settings applied to every connection:
    1. foreign_keys = ON  — SQLite disables FK enforcement by default.
       Without this, you could insert a prediction for a stock_id that
       doesn't exist in the stocks table and SQLite would allow it.
    2. row_factory = sqlite3.Row  — makes query results act like dicts.
       Without this, cursor.fetchall() returns plain tuples: (1, 'AAPL', ...).
       With this, you get row['ticker'] instead of row[1]. Much safer and
       more readable throughout the codebase.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """
    Context manager for database access. Use this for all write operations.

    Usage:
        with get_db() as conn:
            conn.execute("INSERT INTO stocks ...")
        # connection is automatically committed and closed

    Why a context manager?
    - Guarantees the connection is always closed, even if an exception occurs
    - Auto-commits on success, auto-rolls back on exception
    - Prevents connection leaks (leaving connections open is a common bug)
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """
    Run schema.sql against the database. Safe to call multiple times —
    all CREATE TABLE statements use IF NOT EXISTS.

    Called once by scripts/setup_db.py on first run.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    with get_db() as conn:
        with open(schema_path) as f:
            conn.executescript(f.read())
    print(f"Database initialized at: {DB_PATH}")


# ---------------------------------------------------------------------------
# Query helpers
# These are thin wrappers used throughout the project to avoid repeating
# boilerplate. They don't hide complexity — they just remove noise.
# ---------------------------------------------------------------------------

def fetch_one(query: str, params: tuple = ()) -> sqlite3.Row | None:
    """Run a SELECT and return the first row, or None if no results."""
    conn = get_connection()
    try:
        return conn.execute(query, params).fetchone()
    finally:
        conn.close()


def fetch_all(query: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Run a SELECT and return all rows as a list."""
    conn = get_connection()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def execute(query: str, params: tuple = ()) -> None:
    """Run a single INSERT / UPDATE / DELETE and commit."""
    with get_db() as conn:
        conn.execute(query, params)


def executemany(query: str, params_list: list[tuple]) -> None:
    """
    Run the same INSERT / UPDATE for a list of parameter tuples.
    Much faster than calling execute() in a loop — SQLite batches these
    into a single transaction.

    Example:
        rows = [('AAPL', '0000320193'), ('MSFT', '0000789019')]
        executemany("INSERT OR IGNORE INTO stocks (ticker, cik) VALUES (?,?)", rows)
    """
    with get_db() as conn:
        conn.executemany(query, params_list)


def get_stock_id(ticker: str) -> int | None:
    """Return the integer primary key for a ticker, or None if not found."""
    row = fetch_one("SELECT id FROM stocks WHERE ticker = ?", (ticker,))
    return row["id"] if row else None


def get_all_active_tickers() -> list[str]:
    """Return list of all tickers currently in the active universe."""
    rows = fetch_all("SELECT ticker FROM stocks WHERE is_active = 1 ORDER BY ticker")
    return [r["ticker"] for r in rows]
