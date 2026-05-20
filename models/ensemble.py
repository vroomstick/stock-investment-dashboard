"""
models/ensemble.py

Ensemble combiner for XGBoost + LSTM probability outputs.

WHY ENSEMBLE?
-------------
XGBoost and LSTM make errors in different situations:

  XGBoost is good at:
    - Fundamental-driven calls (cheap stock with improving earnings)
    - Cross-sectional signals (stock looks good vs. peers)
    - Identifying sector-level themes

  LSTM is good at:
    - Momentum and trend continuation/reversal timing
    - Identifying specific technical patterns (MACD crossovers, RSI extremes)
    - Reacting to news sentiment sequences (multi-day developing stories)

When they agree, we have high-conviction signals.
When they disagree, the blend reduces extreme bets on either model's errors.

Empirically (in academic finance and industry), ensembles of diverse models
consistently outperform any single model by 10-20% on risk-adjusted returns.

HOW THE ENSEMBLE WORKS
-----------------------
Both models output probability vectors: [p0, p1, p2, p3] summing to 1.0.

    final_probs = w_xgb * xgb_probs + w_lstm * lstm_probs

    where w_xgb + w_lstm = 1.0 (from settings.ENSEMBLE)

This is a simple weighted average of probability distributions, which is
valid because both are proper probability distributions (non-negative, sum to 1).

We do NOT average hard class labels. Averaging [2, 3] → "2.5" is meaningless.
Averaging [[0.05, 0.15, 0.55, 0.25], [0.03, 0.10, 0.40, 0.47]] → proper blend.

WHY 60/40 (XGBoost/LSTM)?
--------------------------
Default weights are 0.6 XGB / 0.4 LSTM because:
  - XGBoost has more features (188 vs 18) and generally higher accuracy
  - LSTM needs more data to train well; with 3 years × 200 stocks we're
    at the lower bound of where LSTMs are reliable
  - As we accumulate more data, we can tune these weights via grid search
    on walk-forward validation Sharpe ratio

EXPECTED RETURN COMPUTATION
----------------------------
After blending, we compute the expected 90-day return:

    E[R] = sum_k(p_k * midpoint_k)

    where midpoint_k is the representative return for class k:
      Class 0: -10% (representative loss)
      Class 1:   0% (flat)
      Class 2: +10% (moderate gain)
      Class 3: +25% (strong gain)

This converts probability vectors → a single scalar we can rank stocks by.

SIGNAL CONSTRUCTION
-------------------
The final output for each stock is:
    {
        "expected_return": float,      # E[R] in fractional form
        "class_probs": [p0,p1,p2,p3],  # full probability vector
        "predicted_class": int,        # argmax(probs)
        "conviction": float,           # max(probs) — how concentrated is the dist?
        "strong_buy": bool,            # p2 + p3 > 0.6 (top two classes dominate)
        "strong_sell": bool,           # p0 > 0.4 (loss class dominates)
    }
"""

from typing import Optional
import numpy as np
import pandas as pd

from config.settings import ENSEMBLE, RETURN_BUCKET_MIDPOINTS


# ---------------------------------------------------------------------------
# Ensemble combiner
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Combines XGBoost and LSTM probability outputs into final predictions.

    Can operate in XGB-only mode if LSTM is unavailable (e.g., no PyTorch,
    or training data too sparse for reliable LSTM training).

    Usage:
        ensemble = EnsembleModel(xgb_model, lstm_model)
        signals  = ensemble.predict(X_flat, X_seq)
        # signals is a DataFrame with [ticker, expected_return, ...]

        # XGB-only mode:
        ensemble = EnsembleModel(xgb_model, lstm_model=None)
        signals  = ensemble.predict(X_flat)
    """

    def __init__(
        self,
        xgb_model,
        lstm_model       = None,
        xgb_weight: float = None,
        lstm_weight: float = None,
    ):
        """
        Args:
            xgb_model:   Fitted XGBoostClassifier instance.
            lstm_model:  Fitted LSTMClassifier instance, or None for XGB-only.
            xgb_weight:  Weight for XGBoost (default: settings.ENSEMBLE.xgb_weight)
            lstm_weight: Weight for LSTM (default: settings.ENSEMBLE.lstm_weight)
        """
        self.xgb_model  = xgb_model
        self.lstm_model = lstm_model

        if xgb_weight is not None and lstm_weight is not None:
            assert abs(xgb_weight + lstm_weight - 1.0) < 1e-9, \
                "xgb_weight + lstm_weight must equal 1.0"
            self.xgb_weight  = xgb_weight
            self.lstm_weight = lstm_weight
        else:
            self.xgb_weight  = ENSEMBLE["xgb_weight"]
            self.lstm_weight = ENSEMBLE["lstm_weight"]

        self._midpoints = np.array(RETURN_BUCKET_MIDPOINTS, dtype=float)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def predict(
        self,
        X_flat: pd.DataFrame,
        X_seq:  Optional[np.ndarray] = None,
        meta:   Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate ensemble signals for a batch of stocks.

        Args:
            X_flat: Feature DataFrame for XGBoost (n_stocks, n_features)
            X_seq:  3D array for LSTM (n_stocks, seq_len, n_features).
                    If None, runs XGB-only.
            meta:   DataFrame with [ticker, date] aligned to X_flat rows.
                    If provided, included in output.

        Returns:
            DataFrame with one row per stock, columns:
              ticker (if meta provided), date (if meta provided),
              expected_return, predicted_class, conviction,
              strong_buy, strong_sell,
              prob_class_0, prob_class_1, prob_class_2, prob_class_3

        Why include all 4 probabilities in output?
            The position sizer needs the full distribution, not just the
            expected value. A stock with E[R]=5% but p0=0.3 is riskier than
            one with E[R]=5% but p0=0.05. Kelly fraction uses the full dist.
        """
        probs = self.predict_proba(X_flat, X_seq)  # (n, 4)

        expected_return = probs @ self._midpoints   # (n,) weighted sum
        predicted_class = probs.argmax(axis=1)      # (n,)
        conviction      = probs.max(axis=1)         # (n,) max probability

        n = len(probs)
        results = {
            "expected_return":  expected_return,
            "predicted_class":  predicted_class,
            "conviction":       conviction,
            "strong_buy":       (probs[:, 2] + probs[:, 3]) > 0.60,
            "strong_sell":      probs[:, 0] > 0.40,
            "prob_class_0":     probs[:, 0],
            "prob_class_1":     probs[:, 1],
            "prob_class_2":     probs[:, 2],
            "prob_class_3":     probs[:, 3],
        }

        df = pd.DataFrame(results)

        if meta is not None:
            for col in reversed(["date", "ticker"]):
                if col in meta.columns:
                    df.insert(0, col, meta[col].values)

        return df

    def predict_proba(
        self,
        X_flat: pd.DataFrame,
        X_seq:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return blended probability matrix (n_stocks, 4).

        If lstm_model is None or X_seq is None, returns XGBoost probabilities
        only (with the xgb_weight effectively = 1.0).

        Implementation note:
            We re-normalize after blending to ensure exact sum-to-1 per row.
            Floating point rounding can cause tiny deviations (1.000000001).
        """
        xgb_probs = self.xgb_model.predict_proba(X_flat)  # (n, 4)

        if self.lstm_model is not None and X_seq is not None:
            lstm_probs = self.lstm_model.predict_proba(X_seq)  # (n, 4)

            if len(xgb_probs) != len(lstm_probs):
                raise ValueError(
                    f"XGBoost ({len(xgb_probs)}) and LSTM ({len(lstm_probs)}) "
                    "must have the same number of samples."
                )

            blended = (
                self.xgb_weight  * xgb_probs +
                self.lstm_weight * lstm_probs
            )
        else:
            blended = xgb_probs

        # Normalize rows to sum to exactly 1.0
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / row_sums

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def signal_summary(self, signals: pd.DataFrame) -> dict:
        """
        Summarize the distribution of signals across a universe.

        Useful for sanity checking: if 80% of stocks are "strong buy",
        something is probably wrong with the model or features.

        Args:
            signals: Output of .predict()

        Returns:
            Dict with counts and percentages per class + coverage stats.
        """
        n = len(signals)
        return {
            "n_stocks":         n,
            "strong_buy_count": int(signals["strong_buy"].sum()),
            "strong_buy_pct":   float(signals["strong_buy"].mean()) * 100,
            "strong_sell_count": int(signals["strong_sell"].sum()),
            "strong_sell_pct":  float(signals["strong_sell"].mean()) * 100,
            "class_distribution": {
                f"class_{k}": int((signals["predicted_class"] == k).sum())
                for k in range(4)
            },
            "avg_expected_return": float(signals["expected_return"].mean()),
            "avg_conviction":      float(signals["conviction"].mean()),
        }

    def is_xgb_only(self) -> bool:
        """True if running without LSTM (single model mode)."""
        return self.lstm_model is None
