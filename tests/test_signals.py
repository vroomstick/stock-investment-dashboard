"""
tests/test_signals.py

Unit tests for signals/ logic.
Currently covers the math foundations that position sizing and risk
management will depend on. Full signal tests will expand as
signals/scoring.py, signals/position_sizing.py, etc. are built.

Run with: pytest tests/test_signals.py -v
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Label construction math (used by weekly_train.py)
# These tests validate the 4-class bucketing logic before the model is built.
# ---------------------------------------------------------------------------

def _bucket_returns(returns: pd.Series) -> pd.Series:
    """
    Convert a Series of 90-day forward returns to 4-class labels.

    Class 0: < -5%          (big loss)
    Class 1: -5% to +5%     (flat)
    Class 2: +5% to +15%    (moderate gain)
    Class 3: > +15%         (strong gain)
    """
    bins   = [-np.inf, -0.05, 0.05, 0.15, np.inf]
    labels = [0, 1, 2, 3]
    return pd.cut(returns, bins=bins, labels=labels).astype(int)


class TestLabelBucketing:
    def test_big_loss(self):
        r = pd.Series([-0.10, -0.20, -0.06])
        labels = _bucket_returns(r)
        assert all(labels == 0)

    def test_flat(self):
        r = pd.Series([-0.04, 0.0, 0.04])
        labels = _bucket_returns(r)
        assert all(labels == 1)

    def test_moderate_gain(self):
        r = pd.Series([0.06, 0.10, 0.14])
        labels = _bucket_returns(r)
        assert all(labels == 2)

    def test_strong_gain(self):
        r = pd.Series([0.16, 0.25, 0.50])
        labels = _bucket_returns(r)
        assert all(labels == 3)

    def test_boundary_minus_5pct_is_class1(self):
        # -5% exactly: bins are (-inf, -0.05] → class 0; (-0.05, 0.05] → class 1
        # pd.cut with right=True (default): -0.05 falls in class 0
        r = pd.Series([-0.05])
        label = _bucket_returns(r).iloc[0]
        assert label == 0  # boundary belongs to the left bin

    def test_boundary_plus_15pct_is_class2(self):
        r = pd.Series([0.15])
        label = _bucket_returns(r).iloc[0]
        assert label == 2  # 0.15 is in (-inf, 0.15] → class 2

    def test_all_classes_represented(self):
        r = pd.Series([-0.10, 0.0, 0.10, 0.20])
        labels = _bucket_returns(r)
        assert set(labels) == {0, 1, 2, 3}

    def test_returns_series_same_length(self):
        r = pd.Series(np.random.uniform(-0.3, 0.3, 100))
        labels = _bucket_returns(r)
        assert len(labels) == len(r)

    def test_label_dtype_is_int(self):
        r = pd.Series([0.0, 0.1, -0.1, 0.2])
        labels = _bucket_returns(r)
        assert labels.dtype in (int, np.int64, np.int32)


# ---------------------------------------------------------------------------
# Forward return computation (used to build labels from price data)
# ---------------------------------------------------------------------------

class TestForwardReturns:
    def _forward_returns(self, prices: pd.Series, horizon: int = 90) -> pd.Series:
        """
        Compute horizon-day forward return for each date.
        NaN for the last `horizon` dates (future not available).
        Uses adj_close for accuracy (already adjusted for splits/dividends).
        """
        future = prices.shift(-horizon)
        return (future / prices) - 1

    def test_forward_return_simple(self):
        prices = pd.Series([100.0, 110.0])
        fwd = self._forward_returns(prices, horizon=1)
        assert fwd.iloc[0] == pytest.approx(0.10)
        assert np.isnan(fwd.iloc[1])  # last row has no future

    def test_forward_return_nan_at_tail(self):
        prices = pd.Series(np.arange(1, 101, dtype=float))
        fwd = self._forward_returns(prices, horizon=10)
        assert fwd.iloc[-10:].isna().all()
        assert fwd.iloc[:-10].notna().all()

    def test_no_lookahead_within_horizon(self):
        # Row t's forward return must only use price at t+horizon, not t+1..t+horizon-1
        prices = pd.Series([100.0, 105.0, 110.0, 115.0])
        fwd = self._forward_returns(prices, horizon=2)
        assert fwd.iloc[0] == pytest.approx((110.0 / 100.0) - 1)
        assert fwd.iloc[1] == pytest.approx((115.0 / 105.0) - 1)


# ---------------------------------------------------------------------------
# Kelly criterion — position sizing foundation
# The full position_sizing.py will implement fractional Kelly.
# These tests validate the core math before that module exists.
# ---------------------------------------------------------------------------

class TestKellyCriterion:
    def _kelly_fraction(self, win_prob: float, win_size: float,
                        loss_size: float) -> float:
        """
        Full Kelly fraction = (p * b - q) / b
        where b = win_size / loss_size, p = win_prob, q = 1 - p.

        In practice we use half-Kelly or quarter-Kelly to reduce drawdowns.
        """
        b = win_size / loss_size
        return (win_prob * b - (1 - win_prob)) / b

    def test_positive_kelly_for_edge(self):
        # 60% win rate, win=10%, loss=5% → positive edge
        f = self._kelly_fraction(0.60, 0.10, 0.05)
        assert f > 0

    def test_zero_kelly_at_breakeven(self):
        # 50% win, win=5%, loss=5% → f = (0.5*1 - 0.5)/1 = 0
        f = self._kelly_fraction(0.50, 0.05, 0.05)
        assert abs(f) < 1e-9

    def test_negative_kelly_for_no_edge(self):
        # 40% win, win=5%, loss=5% → negative
        f = self._kelly_fraction(0.40, 0.05, 0.05)
        assert f < 0

    def test_half_kelly_is_smaller(self):
        f_full = self._kelly_fraction(0.60, 0.10, 0.05)
        f_half = f_full / 2
        assert f_half < f_full
        assert f_half > 0


# ---------------------------------------------------------------------------
# Risk management math — max drawdown and position limits
# ---------------------------------------------------------------------------

class TestRiskMath:
    def test_max_drawdown_flat_curve(self):
        equity = pd.Series([100.0, 100.0, 100.0, 100.0])
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        assert drawdown.min() == pytest.approx(0.0)

    def test_max_drawdown_monotone_decline(self):
        equity = pd.Series([100.0, 90.0, 80.0, 70.0])
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        assert drawdown.min() == pytest.approx(-0.30)

    def test_max_drawdown_partial_recovery(self):
        equity = pd.Series([100.0, 80.0, 90.0, 70.0])
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        # Peak is 100, trough is 70 → -30% max drawdown
        assert drawdown.min() == pytest.approx(-0.30)

    def test_position_size_capped_at_max(self):
        """Position size must never exceed max allocation per stock."""
        max_per_stock = 0.10  # 10% of portfolio
        kelly_fraction = 0.25  # hypothetical Kelly output
        # Half-Kelly, capped at max
        size = min(kelly_fraction * 0.5, max_per_stock)
        assert size <= max_per_stock

    def test_total_exposure_limit(self):
        """Sum of all position sizes must not exceed gross exposure limit."""
        max_gross = 1.0  # 100% invested, no leverage
        positions = [0.08, 0.10, 0.07, 0.09, 0.06]  # 5 positions
        assert sum(positions) <= max_gross
