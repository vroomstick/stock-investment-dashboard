"""
scripts/weekly_predict.py

Weekly inference pipeline. Loads the latest trained models and generates
buy/sell signals for all active tickers as of today.

WHAT HAPPENS EACH WEEK
-----------------------
1. Load the versioned XGBoost + LSTM models and preprocessor from disk.
2. Build an inference dataset: pull today's features for all active tickers.
3. Preprocess using TRAINING-time parameters (no refitting).
4. Run XGBoost and LSTM predictions → 4-class probability vectors.
5. Ensemble: blend probabilities (0.6 XGB + 0.4 LSTM).
6. Compute expected returns and position sizes.
7. Write predictions to the 'predictions' database table.
8. Print a summary of the top buy signals.

IMPORTANT: POINT-IN-TIME DISCIPLINE
-------------------------------------
This script only uses features that were available before the prediction date.
The daily_collect.py job runs at 6:30 PM ET (after market close), so by the
time weekly_predict.py runs (Sunday evening), all Friday features are in the DB.

We NEVER use prices or news from after the prediction date. The inference
dataset uses as_of_date = today, and all features in the DB are labeled with
the date they were collected (not the date they were computed for a future date).

PREDICTIONS TABLE SCHEMA
-------------------------
See database/schema.sql for the full CREATE TABLE statement. Key columns:
  stock_id, date, predicted_class, prob_class_0..3,
  expected_return, conviction, strong_buy, strong_sell,
  model_version, created_at

Usage:
  python scripts/weekly_predict.py
  python scripts/weekly_predict.py --date 2024-06-07
  python scripts/weekly_predict.py --version v3
  python scripts/weekly_predict.py --top-n 20   (print top 20 signals)
  python scripts/weekly_predict.py --dry-run     (predict but don't write to DB)
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("weekly_predict")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="MLspec weekly predictions")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Prediction date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Model version to load (default: latest)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Limit to specific tickers (default: all active)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Print top N buy signals (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate predictions but don't write to DB",
    )
    return parser.parse_args()


def _find_latest_version() -> str:
    """Find the most recently saved model version by scanning artifacts/."""
    artifacts = ROOT / "models" / "artifacts"
    if not artifacts.exists():
        raise FileNotFoundError(
            f"No model artifacts found at {artifacts}. "
            "Run weekly_train.py first."
        )
    xgb_files = sorted(artifacts.glob("xgb_*.pkl"))
    if not xgb_files:
        raise FileNotFoundError(
            "No XGBoost model files found. Run weekly_train.py first."
        )
    # Most recently modified file
    latest = max(xgb_files, key=lambda p: p.stat().st_mtime)
    # Extract version from filename: xgb_{version}.pkl
    version = latest.stem.replace("xgb_", "")
    return version


def main():
    args   = parse_args()
    logger = setup_logging()

    as_of_date = args.date or date.today().isoformat()
    version    = args.version or _find_latest_version()

    sep = "=" * 60
    logger.info(sep)
    logger.info(f"  MLspec weekly_predict.py — date={as_of_date}, version={version}")
    logger.info(sep)

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    logger.info("Loading models...")

    from models.xgboost_model import XGBoostClassifier
    from models.preprocessing import FeaturePreprocessor
    from models.ensemble import EnsembleModel

    xgb_model    = XGBoostClassifier.load(f"xgb_{version}")
    preprocessor = FeaturePreprocessor.load(f"preprocessor_{version}")

    # Try to load LSTM — fall back to XGB-only if not available
    lstm_model = None
    try:
        from models.lstm_model import LSTMClassifier
        lstm_model = LSTMClassifier.load(f"lstm_{version}")
        logger.info("  Loaded XGBoost + LSTM models.")
    except FileNotFoundError:
        logger.info("  LSTM model not found — running XGBoost only.")
    except ImportError:
        logger.info("  PyTorch not installed — running XGBoost only.")

    ensemble = EnsembleModel(xgb_model, lstm_model)

    # ------------------------------------------------------------------
    # 2. Build inference dataset
    # ------------------------------------------------------------------
    logger.info(f"Building inference dataset for {as_of_date}...")

    from models.label_builder import build_inference_dataset, build_sequence_dataset

    try:
        X_flat, meta = build_inference_dataset(as_of_date, args.tickers)
    except Exception as e:
        logger.error(f"Inference dataset build failed: {e}")
        sys.exit(2)

    logger.info(f"  {len(X_flat)} stocks × {X_flat.shape[1]} features")

    # ------------------------------------------------------------------
    # 3. Preprocess (apply training parameters — no refitting)
    # ------------------------------------------------------------------
    X_flat_pp = preprocessor.transform(X_flat.copy())

    # ------------------------------------------------------------------
    # 4. Build sequence data for LSTM (if available)
    # ------------------------------------------------------------------
    X_seq = None
    if lstm_model is not None:
        try:
            # Build the last 60 days of sequences ending on as_of_date
            from config.settings import FEATURES
            seq_start = (
                __import__("pandas").Timestamp(as_of_date)
                - __import__("pandas").DateOffset(days=100)
            ).strftime("%Y-%m-%d")

            X_seq_all, _, meta_seq = build_sequence_dataset(
                train_start=seq_start,
                train_end=as_of_date,
                tickers=args.tickers,
            )
            # Filter to only the most recent sequence per ticker
            latest_idx = (
                __import__("pandas").DataFrame(meta_seq)
                .groupby("ticker")["date"]
                .idxmax()
                .values
            )
            X_seq = X_seq_all[latest_idx]
            meta_seq_latest = meta_seq.iloc[latest_idx].reset_index(drop=True)

            # Align X_flat and X_seq on ticker order
            import pandas as pd
            seq_tickers = meta_seq_latest["ticker"].values
            flat_tickers = meta["ticker"].values
            # Only keep tickers present in both
            common = set(flat_tickers) & set(seq_tickers)
            flat_mask = meta["ticker"].isin(common)
            seq_mask  = meta_seq_latest["ticker"].isin(common)

            X_flat_pp = X_flat_pp[flat_mask].reset_index(drop=True)
            meta       = meta[flat_mask].reset_index(drop=True)
            X_seq      = X_seq[seq_mask.values]

            logger.info(f"  LSTM sequences built for {len(X_seq)} stocks.")
        except Exception as e:
            logger.warning(f"  LSTM sequence build failed ({e}) — XGB only.")
            X_seq = None

    # ------------------------------------------------------------------
    # 5. Generate ensemble predictions
    # ------------------------------------------------------------------
    logger.info("Generating predictions...")
    signals = ensemble.predict(X_flat_pp, X_seq, meta=meta)

    summary = ensemble.signal_summary(signals)
    logger.info(
        f"  Universe: {summary['n_stocks']} stocks  |  "
        f"Strong buy: {summary['strong_buy_count']} "
        f"({summary['strong_buy_pct']:.1f}%)  |  "
        f"Strong sell: {summary['strong_sell_count']} "
        f"({summary['strong_sell_pct']:.1f}%)"
    )

    # Add prediction date
    signals["date"]          = as_of_date
    signals["model_version"] = version

    # ------------------------------------------------------------------
    # 6. Print top signals
    # ------------------------------------------------------------------
    import pandas as pd
    top_buys = (
        signals[signals["strong_buy"]]
        .sort_values("expected_return", ascending=False)
        .head(args.top_n)
    )

    logger.info(f"\n  Top {args.top_n} buy signals:")
    if len(top_buys) > 0:
        display_cols = ["ticker", "expected_return", "predicted_class",
                        "conviction", "prob_class_3", "prob_class_2"]
        display_cols = [c for c in display_cols if c in top_buys.columns]
        logger.info("\n" + top_buys[display_cols].to_string(index=False))
    else:
        logger.info("  No strong buy signals today.")

    # ------------------------------------------------------------------
    # 7. Write to DB
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("Dry run — skipping DB write.")
    else:
        logger.info("Writing predictions to database...")
        _write_predictions(signals, as_of_date, version, logger)

    logger.info(sep)
    logger.info("  Prediction run complete.")
    logger.info(sep)


def _write_predictions(
    signals,
    as_of_date: str,
    version: str,
    logger: logging.Logger,
) -> None:
    """
    Upsert predictions into the predictions table.

    Uses INSERT OR REPLACE to handle re-running the script on the same date.
    """
    from database.db import get_db, get_stock_id

    rows = []
    for _, row in signals.iterrows():
        stock_id = get_stock_id(row.get("ticker", ""))
        if stock_id is None:
            continue
        p0 = float(row["prob_class_0"])
        p1 = float(row["prob_class_1"])
        p2 = float(row["prob_class_2"])
        p3 = float(row["prob_class_3"])
        rows.append((
            stock_id,
            as_of_date,
            int(row["predicted_class"]),
            p0, p1, p2, p3,
            float(row["expected_return"]) * 100,  # store as %, schema says expected_return_pct
            float(row["conviction"]),
            p2 + p3,                               # bull_probability = P(>5% gain)
            version,
        ))

    if not rows:
        logger.warning("  No predictions to write (no valid stock_ids found).")
        return

    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO predictions
               (stock_id, prediction_date, predicted_class,
                prob_big_loss, prob_flat, prob_moderate_gain, prob_strong_gain,
                expected_return_pct, confidence,
                bull_probability,
                model_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    logger.info(f"  Wrote {len(rows)} predictions to DB.")


if __name__ == "__main__":
    main()
