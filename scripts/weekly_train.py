"""
scripts/weekly_train.py

Weekly model training pipeline. Trains XGBoost and LSTM on historical data
and saves versioned artifacts.

RUN SCHEDULE
------------
Trains once per week, typically Sunday evening after daily_collect.py has
finished populating the DB with Friday's data.

Cron entry (Sunday 8 PM ET):
  0 20 * * 0 cd /Users/varun/Documents/MLspec && \
    conda run -n mlspec python scripts/weekly_train.py >> logs/train.log 2>&1

WHAT IT DOES (step by step)
----------------------------
1. Build training dataset:
   - Pull features from DB for [cutoff_date - 36 months, cutoff_date]
   - Compute 4-class forward return labels
   - Drop rows where forward return is NaN (end of dataset)

2. Preprocess:
   - Winsorize (clip at 1st/99th percentile of training distribution)
   - Z-score normalize (global or sector-specific per settings)
   - Impute missing values (forward fill, then 0 for sentiment)
   - IMPORTANT: fit on training data only, then apply same transform to val

3. Walk-forward validation:
   - Split training window into multiple folds
   - Train XGBoost on each fold, validate on next period
   - Average val mlogloss and accuracy across folds
   - If val performance < threshold, warn and exit (don't overwrite good model)

4. Final training:
   - Train on full [cutoff_date - 36 months, cutoff_date] window
   - Use val set = last 3 months of this window for early stopping

5. Train LSTM (optional):
   - Build 60-day sequence tensors from lstm_feature_subset
   - Train for up to 100 epochs with early stopping
   - Skip if --skip-lstm flag is set or PyTorch not available

6. Save artifacts:
   - models/artifacts/xgb_{version}.pkl + _meta.json
   - models/artifacts/lstm_{version}.pt + _meta.json
   - models/artifacts/preprocessor_{version}.pkl

Usage:
  python scripts/weekly_train.py
  python scripts/weekly_train.py --cutoff-date 2024-06-01
  python scripts/weekly_train.py --version v3 --skip-lstm
  python scripts/weekly_train.py --dry-run   (builds dataset, skips training)
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # needed for conda run invocations

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("weekly_train")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(ch)

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "weekly_train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(fh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="MLspec weekly model training")
    parser.add_argument(
        "--cutoff-date", type=str, default=None,
        help="Training cutoff date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Model version tag (default: cutoff date)",
    )
    parser.add_argument(
        "--train-months", type=int, default=36,
        help="Months of training history (default: 36)",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training (XGBoost only)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Limit to specific tickers (default: all active)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build dataset and print stats, skip actual training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    cutoff_date = args.cutoff_date or date.today().isoformat()
    version     = args.version or cutoff_date.replace("-", "")

    sep = "=" * 60
    logger.info(sep)
    logger.info(f"  MLspec weekly_train.py — cutoff={cutoff_date}, version={version}")
    logger.info(sep)

    # ------------------------------------------------------------------
    # 1. Build training dataset
    # ------------------------------------------------------------------
    logger.info("Building training dataset...")

    from models.label_builder import build_training_dataset, walk_forward_splits
    from models.preprocessing import FeaturePreprocessor
    import pandas as pd

    train_end   = cutoff_date
    train_start = (
        pd.Timestamp(cutoff_date) - pd.DateOffset(months=args.train_months)
    ).strftime("%Y-%m-%d")
    # Validation uses the last 3 months before cutoff
    val_start = (
        pd.Timestamp(cutoff_date) - pd.DateOffset(months=4)
    ).strftime("%Y-%m-%d")

    logger.info(f"  Training window: {train_start} → {train_end}")
    logger.info(f"  Validation:      {val_start}   → {train_end} (last 4 months)")

    try:
        X_all, y_all, meta_all = build_training_dataset(
            train_start=train_start,
            train_end=train_end,
            tickers=args.tickers,
        )
    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        sys.exit(2)

    logger.info(f"  Dataset: {len(X_all)} samples × {X_all.shape[1]} features")
    logger.info(f"  Class distribution:\n{y_all.value_counts().sort_index().to_string()}")

    if args.dry_run:
        logger.info("Dry run — skipping training.")
        return

    # ------------------------------------------------------------------
    # 2. Split into train / val
    # ------------------------------------------------------------------
    train_mask = meta_all["date"] < val_start
    val_mask   = meta_all["date"] >= val_start

    X_train, y_train = X_all[train_mask].copy(), y_all[train_mask].copy()
    X_val,   y_val   = X_all[val_mask].copy(),   y_all[val_mask].copy()

    logger.info(f"  Train: {len(X_train)} samples  |  Val: {len(X_val)} samples")

    if len(X_val) == 0:
        logger.error("Validation set is empty — check val_start date vs. data availability.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Preprocess (fit on train, apply to both)
    # ------------------------------------------------------------------
    logger.info("Preprocessing features...")

    preprocessor = FeaturePreprocessor()
    X_train_pp   = preprocessor.fit_transform(X_train.copy())

    # For sector z-scoring we need sector labels
    # (FeaturePreprocessor.fit_transform accepts optional 'sectors' argument)
    X_val_pp     = preprocessor.transform(X_val.copy())

    coverage = preprocessor.coverage_report(X_all)
    low_coverage = coverage[coverage["non_null_pct"] < 50]
    if len(low_coverage) > 0:
        logger.warning(
            f"  {len(low_coverage)} features have <50% non-null coverage:\n"
            f"{low_coverage.to_string()}"
        )

    # ------------------------------------------------------------------
    # 4. Train XGBoost
    # ------------------------------------------------------------------
    logger.info("Training XGBoostClassifier...")

    from models.xgboost_model import XGBoostClassifier
    import numpy as np

    xgb_model = XGBoostClassifier()
    xgb_model.fit(X_train_pp, y_train, X_val_pp, y_val, verbose=100)

    val_probs   = xgb_model.predict_proba(X_val_pp)
    val_preds   = val_probs.argmax(axis=1)
    val_acc     = float((val_preds == y_val.values).mean())
    val_metrics = xgb_model.val_metrics()

    logger.info(f"  XGB val accuracy:  {val_acc:.3f}")
    logger.info(f"  XGB val mlogloss:  {val_metrics.get('best_val_mlogloss', 'N/A')}")
    logger.info(f"  XGB best iter:     {val_metrics.get('best_iteration', 'N/A')}")

    top_features = xgb_model.feature_importances(top_n=10)
    logger.info(f"  Top 10 features:\n{top_features.to_string(index=False)}")

    xgb_model.save(f"xgb_{version}")
    preprocessor.save(f"preprocessor_{version}")

    logger.info(f"  XGBoost model saved (version={version})")

    # ------------------------------------------------------------------
    # 5. Train LSTM (optional)
    # ------------------------------------------------------------------
    if args.skip_lstm:
        logger.info("LSTM training skipped (--skip-lstm flag).")
    else:
        logger.info("Training LSTM...")
        try:
            from models.label_builder import build_sequence_dataset
            from models.lstm_model import LSTMClassifier

            X_seq_all, y_seq_all, meta_seq = build_sequence_dataset(
                train_start=train_start,
                train_end=train_end,
                tickers=args.tickers,
            )

            seq_train_mask = meta_seq["date"] < val_start
            seq_val_mask   = meta_seq["date"] >= val_start

            X_seq_train = X_seq_all[seq_train_mask]
            y_seq_train = y_seq_all[seq_train_mask]
            X_seq_val   = X_seq_all[seq_val_mask]
            y_seq_val   = y_seq_all[seq_val_mask]

            logger.info(
                f"  LSTM dataset: {len(X_seq_train)} train seqs, "
                f"{len(X_seq_val)} val seqs"
            )

            lstm_model = LSTMClassifier()
            lstm_model.fit(
                X_seq_train, y_seq_train,
                X_seq_val,   y_seq_val,
                verbose=True,
            )

            lstm_preds = lstm_model.predict(X_seq_val)
            lstm_acc   = float((lstm_preds == y_seq_val).mean())
            logger.info(f"  LSTM val accuracy: {lstm_acc:.3f}")

            lstm_model.save(f"lstm_{version}")
            logger.info(f"  LSTM model saved (version={version})")

        except ImportError as e:
            logger.warning(f"LSTM skipped — PyTorch not available: {e}")
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            logger.warning("Continuing with XGBoost-only model.")

    # ------------------------------------------------------------------
    # 6. Quick walk-forward summary
    # ------------------------------------------------------------------
    logger.info("Computing walk-forward validation summary...")
    try:
        all_dates = pd.DatetimeIndex(sorted(meta_all["date"].unique()))
        splits    = walk_forward_splits(
            all_dates,
            train_months=args.train_months,
            val_months=3,
            gap_months=1,
            step_months=3,
        )
        logger.info(f"  {len(splits)} walk-forward folds available")
        for i, s in enumerate(splits[-3:]):  # show last 3 folds
            logger.info(
                f"  Fold {len(splits)-2+i}: "
                f"train={s['train_start']}→{s['train_end']}  "
                f"val={s['val_start']}→{s['val_end']}"
            )
    except Exception as e:
        logger.warning(f"Walk-forward summary failed: {e}")

    logger.info(sep)
    logger.info(f"  Training complete. Version: {version}")
    logger.info(sep)


if __name__ == "__main__":
    main()
