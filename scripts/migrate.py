"""
scripts/migrate.py

Schema migration runner. Applies a SQL statement and records the new version
in the schema_version table.

Usage:
  python scripts/migrate.py --version 4 --sql "ALTER TABLE stocks ADD COLUMN foo TEXT"
  python scripts/migrate.py --version 4 --file migrations/v4.sql

Design:
  - Idempotent: refuses to apply a version that is already recorded.
  - The schema_version table records every applied migration with a timestamp.
  - SQL is executed in a single transaction; if it fails, the version is not recorded.
  - Current version is the MAX(version) in schema_version.

Schema version history (keep this comment up to date):
  1 — Initial schema
  2 — Add sector_etf to stocks, is_negative_8k to sec_filings, new indexes
  3 — Add pipeline_runs table
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from database.db import get_db, fetch_one


def current_version() -> int:
    """Return the highest schema version recorded, or 0 if none."""
    row = fetch_one("SELECT MAX(version) as v FROM schema_version", ())
    return int(row["v"]) if row and row["v"] is not None else 0


def apply_migration(version: int, sql: str, description: str = ""):
    """
    Execute sql and record version in schema_version.
    Raises SystemExit on failure — never leaves a partial migration.
    """
    cur = current_version()

    if version <= cur:
        print(f"  Already at version {cur}. Version {version} already applied.")
        sys.exit(0)

    if version != cur + 1:
        print(f"  ERROR: current version is {cur}. "
              f"Migrations must be applied sequentially — next expected: {cur + 1}, got: {version}.")
        sys.exit(1)

    print(f"  Applying migration v{version}...")
    print(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")

    try:
        with get_db() as conn:
            conn.executescript(sql)
            conn.execute(
                """INSERT INTO schema_version (version, description, applied_at)
                   VALUES (?, ?, ?)""",
                (version, description or f"Migration v{version}", datetime.utcnow().isoformat()),
            )
        print(f"  Migration v{version} applied successfully.")
        print(f"  Current schema version: {version}")
    except Exception as e:
        print(f"  ERROR: migration failed — {e}")
        print("  Database was not modified (transaction rolled back).")
        sys.exit(2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLspec schema migration runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrate.py --version 4 --sql "ALTER TABLE stocks ADD COLUMN exchange TEXT"
  python scripts/migrate.py --version 4 --file migrations/v4.sql
  python scripts/migrate.py --status
        """,
    )
    parser.add_argument(
        "--version", type=int, default=None,
        help="Version number to apply (must be current_version + 1)",
    )
    parser.add_argument(
        "--sql", type=str, default=None,
        help="SQL statement(s) to execute",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a .sql file to execute",
    )
    parser.add_argument(
        "--description", type=str, default="",
        help="Human-readable description of this migration",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current schema version and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.status:
        ver = current_version()
        print(f"Current schema version: {ver}")
        # Show full history
        with get_db() as conn:
            rows = conn.execute(
                "SELECT version, description, applied_at FROM schema_version ORDER BY version"
            ).fetchall()
        if rows:
            print("\nMigration history:")
            for r in rows:
                print(f"  v{r[0]}  {r[2][:19]}  {r[1]}")
        sys.exit(0)

    if args.version is None:
        print("ERROR: --version is required (or use --status to check current version).")
        sys.exit(1)

    if args.sql and args.file:
        print("ERROR: specify --sql or --file, not both.")
        sys.exit(1)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        sql = path.read_text()
    elif args.sql:
        sql = args.sql
    else:
        print("ERROR: --sql or --file is required.")
        sys.exit(1)

    apply_migration(args.version, sql, args.description)


if __name__ == "__main__":
    main()
