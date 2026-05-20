"""
models/xgboost_model.py

XGBoost multiclass classifier for 4-class return prediction.

WHY XGBOOST (NOT A NEURAL NETWORK) AS THE FIRST MODEL
-------------------------------------------------------
XGBoost excels on tabular data with mixed feature types — exactly what we
have: technical ratios, fundamental metrics, sentiment scores, macro levels.
Neural networks need large datasets and careful architecture choices; for
200 stocks × 3 years of weekly observations, XGBoost is more robust.

XGBoost also gives us:
  - Feature importances out of the box (which features actually drive predictions)
  - Robustness to missing values (trees handle NaN natively)
  - Fast training (~30 seconds for our dataset size)
  - No data normalization required (trees are invariant to monotonic scaling)

We still preprocess (winsorize + z-score) because:
  - It makes feature importances comparable across features
  - The preprocessor must be applied consistently across XGB and LSTM
  - SHAP values are more interpretable on normalized features

MODEL ARCHITECTURE
------------------
    Input: 188-dimensional feature vector (one row per stock, per date)
           All feature groups flattened: technical + fundamental + macro + sentiment

    XGBoost: multi:softprob objective
      - Outputs a 4-dimensional probability vector [p0, p1, p2, p3] summing to 1.0
      - NOT a hard class label — we keep probabilities for ensemble weighting
      - num_class=4, objective="multi:softprob"

    Hyperparameters (from settings.XGBOOST_PARAMS):
      max_depth=6         — moderate depth, captures interactions without overfitting
      learning_rate=0.05  — slow learning with many trees (500 estimators)
      min_child_weight=5  — min samples per leaf, prevents memorizing tiny patterns
      subsample=0.8       — row subsampling (reduces variance, like bagging)
      colsample_bytree=0.8 — column subsampling (like Random Forest)
      reg_alpha=0.1       — L1 regularization (sparsifies feature weights)
      reg_lambda=1.0      — L2 regularization (shrinks all weights)
      early_stopping_rounds=50 — stop if val loss doesn't improve for 50 rounds

WALK-FORWARD TRAINING
---------------------
For a given cutoff date D, we train on [D - 36 months, D] and validate on
[D + 1 month gap, D + 4 months]. This is NOT a single train/test split —
we run multiple folds (see label_builder.walk_forward_splits) and average
validation metrics across folds to get an unbiased performance estimate.

    Fold structure:
        [------36m train------] [1m gap] [--3m val--]
                                    ↑
                         no-lookahead boundary
                         (gap ensures last training label's forward
                          period doesn't overlap first val observation)

FEATURE IMPORTANCES
-------------------
After training, call .feature_importances() to see which features drive
predictions most. This is useful for:
  - Debugging: if "date" or "stock_id" shows up, something is wrong
  - Research: identifying leading indicators for the 4 return classes
  - Pruning: removing features with near-zero importance to simplify the model
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


class XGBoostClassifier:
    """
    Wrapper around xgboost.XGBClassifier with project-specific conventions.

    Responsibilities:
      1. Train on (X, y) with optional eval set for early stopping
      2. Predict probability vectors (not hard class labels)
      3. Report feature importances and val metrics
      4. Save/load model + metadata together as a versioned artifact

    Usage:
        model = XGBoostClassifier()
        model.fit(X_train, y_train, X_val, y_val)
        probs = model.predict_proba(X_test)  # shape (n, 4)
        model.save("xgb_v1")
    """

    def __init__(self, params: dict = None):
        """
        Args:
            params: Override XGBOOST_PARAMS from settings. Useful for
                    hyperparameter search without touching settings.py.
        """
        from config.settings import XGBOOST_PARAMS
        self.params         = {**XGBOOST_PARAMS, **(params or {})}
        self._model         = None
        self._feature_names: list = []
        self._val_metrics:   dict = {}
        self.is_fitted      = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
        verbose: int = 50,
    ) -> "XGBoostClassifier":
        """
        Train the model.

        Args:
            X_train: Feature DataFrame (n_samples, n_features)
            y_train: Integer class labels 0-3
            X_val:   Validation features (used for early stopping)
            y_val:   Validation labels
            verbose: Print eval every N rounds (0 = silent)

        Returns:
            self (for method chaining)

        Why early stopping on val loss?
            Without it, with 500 trees and lr=0.05, we'll memorize training
            data. The val set gives us an out-of-sample loss to monitor.
            We stop when val mlogloss hasn't improved for 50 rounds.
            The best iteration is automatically selected.

        Important:
            X_val/y_val must be from a FUTURE time window (after train_end).
            Using a random hold-out from the same period would leak
            time-series patterns (e.g., regime similarities across dates).
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self._feature_names = list(X_train.columns)

        fit_kwargs = {
            "verbose": verbose,
        }

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]

        # Pop early_stopping_rounds from params to pass separately
        params = dict(self.params)
        early_stop = params.pop("early_stopping_rounds", 50)

        self._model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=early_stop,
        )
        self._model.fit(
            X_train.values,
            y_train.values,
            **fit_kwargs,
        )

        self.is_fitted = True

        # Store best iteration and val loss
        if X_val is not None and self._model.best_score is not None:
            self._val_metrics = {
                "best_iteration": self._model.best_iteration,
                "best_val_mlogloss": self._model.best_score,
            }

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability matrix of shape (n_samples, 4).

        Column j = probability of class j for each sample.

        Example:
            probs = model.predict_proba(X_test)
            probs[0]  → [0.05, 0.15, 0.55, 0.25]  # class 2 most likely
            probs.argmax(axis=1)  → predicted class per sample

        The ensemble does NOT use argmax — it uses the raw probability
        vector to compute weighted expected return across all 4 classes.
        """
        self._check_fitted()
        return self._model.predict_proba(X.values)  # shape (n, 4)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class labels (argmax of probabilities)."""
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importances(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.

        Uses XGBoost's built-in 'gain' importance (total gain across all
        splits using that feature) — more informative than 'weight' (split count).

        Args:
            top_n: Return only the top N features.

        Returns:
            DataFrame with columns [feature, importance]
        """
        self._check_fitted()
        scores = self._model.get_booster().get_score(importance_type="gain")
        df = pd.DataFrame(
            {"feature": list(scores.keys()), "importance": list(scores.values())}
        ).sort_values("importance", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def val_metrics(self) -> dict:
        """Return validation metrics from training (best iteration, best loss)."""
        return dict(self._val_metrics)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, version: str) -> None:
        """
        Save model to models/artifacts/xgb_{version}.pkl

        Also saves a JSON sidecar with metadata (params, feature names,
        val metrics) for auditability — so we can reconstruct what a model
        was trained on without loading the pickle.

        Args:
            version: Version string, e.g. "v1", "2024-01-01"
        """
        self._check_fitted()
        out_dir = ROOT / "models" / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save model binary
        pkl_path = out_dir / f"xgb_{version}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self._model, f)

        # Save metadata sidecar
        meta = {
            "version":       version,
            "feature_names": self._feature_names,
            "params":        {k: v for k, v in self.params.items()
                              if isinstance(v, (int, float, str, bool))},
            "val_metrics":   self._val_metrics,
        }
        meta_path = out_dir / f"xgb_{version}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"XGBoostClassifier saved → {pkl_path}")

    @classmethod
    def load(cls, version: str) -> "XGBoostClassifier":
        """
        Load a saved model.

        Args:
            version: Same string used in .save()

        Returns:
            Fitted XGBoostClassifier instance.
        """
        out_dir = ROOT / "models" / "artifacts"
        pkl_path  = out_dir / f"xgb_{version}.pkl"
        meta_path = out_dir / f"xgb_{version}_meta.json"

        if not pkl_path.exists():
            raise FileNotFoundError(f"No XGBoost model found at {pkl_path}")

        instance = cls.__new__(cls)

        with open(pkl_path, "rb") as f:
            instance._model = pickle.load(f)

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            instance.params          = meta.get("params", {})
            instance._feature_names  = meta.get("feature_names", [])
            instance._val_metrics    = meta.get("val_metrics", {})
        else:
            instance.params         = {}
            instance._feature_names = []
            instance._val_metrics   = {}

        instance.is_fitted = True
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted or self._model is None:
            raise RuntimeError(
                "Model is not fitted. Call .fit() before predict/save."
            )
