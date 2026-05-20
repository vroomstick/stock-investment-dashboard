"""
scripts/daily_collect.py

Cron entry point for the daily data collection pipeline.
Wraps data/pipeline.py with:
  - Rotating log files (logs/daily_collect_YYYY-MM-DD.log)
  - Exit code 0 = success, 1 = partial failures, 2 = critical failure
  - CLI flags mirroring data/pipeline.run()
  - Retry summary in output

Cron setup (runs at 6:30 PM ET daily, after market close):
  30 18 * * 1-5 cd /Users/varun/Documents/MLspec && \
    conda run -n mlspec python scripts/daily_collect.py >> logs/cron.log 2>&1

Or with full path:
  30 18 * * 1-5 /Users/varun/.conda/envs/mlspec/bin/python \
    /Users/varun/Documents/MLspec/scripts/daily_collect.py

Usage:
  python scripts/daily_collect.py
  python scripts/daily_collect.py --date 2025-06-01
  python scripts/daily_collect.py --skip-macro
  python scripts/daily_collect.py --tickers AAPL MSFT NVDA
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def setup_logging(run_date: str) -> logging.Logger:
    """
    Set up rotating daily log file + console output.
    Log files live in logs/ directory (gitignored).
    Each day gets its own file: logs/daily_collect_2025-06-01.log
    """
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"daily_collect_{run_date}.log"

    logger = logging.getLogger("daily_collect")
    logger.setLevel(logging.INFO)

    # File handler — full record for debugging
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))

    # Console handler — same output goes to stdout (captured by cron >> cron.log)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="MLspec daily data collection")
    parser.add_argument("--date",       type=str,  default=None,
                        help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip FRED macro fetch")
    parser.add_argument("--tickers",    nargs="+", default=None,
                        help="Limit to specific tickers")
    return parser.parse_args()


def main():
    args   = parse_args()
    run_date = args.date or date.today().isoformat()
    logger = setup_logging(run_date)

    logger.info(f"{'='*60}")
    logger.info(f"  MLspec daily_collect.py — {run_date}")
    logger.info(f"{'='*60}")

    try:
        from data.pipeline import run as pipeline_run
    except Exception as e:
        logger.error(f"CRITICAL: failed to import pipeline — {e}")
        sys.exit(2)

    try:
        health = pipeline_run(
            as_of_date=run_date,
            skip_macro=args.skip_macro,
            tickers=args.tickers,
        )
    except Exception as e:
        logger.error(f"CRITICAL: pipeline raised unhandled exception — {e}")
        sys.exit(2)

    # Exit codes:
    # 0 = all stocks processed successfully
    # 1 = partial success (some stocks failed, rest completed)
    # 2 = critical failure (already handled above)
    if health["stocks_failed"] == 0 and not health["errors"]:
        logger.info(f"Run complete. Status: SUCCESS")
        sys.exit(0)
    elif health["stocks_ok"] > 0:
        logger.warning(
            f"Run complete. Status: PARTIAL — "
            f"{health['stocks_failed']} stocks failed, "
            f"{health['stocks_ok']} ok"
        )
        sys.exit(1)
    else:
        logger.error("Run complete. Status: FAILED — 0 stocks processed successfully")
        sys.exit(2)


if __name__ == "__main__":
    main()
