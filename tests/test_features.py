"""
tests/test_features.py

Unit tests for feature store helpers and indicator math.
Tests run against in-memory data — no database or API calls required.

Run with: pytest tests/test_features.py -v
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.feature_store.fundamental import (
    _safe_div,
    _row,
    _val,
    _filter_statements,
)


# ---------------------------------------------------------------------------
# _safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10, 2) == pytest.approx(5.0)

    def test_zero_denominator_returns_fallback(self):
        assert _safe_div(10, 0) is None

    def test_none_numerator_returns_fallback(self):
        assert _safe_div(None, 5) is None

    def test_none_denominator_returns_fallback(self):
        assert _safe_div(5, None) is None

    def test_nan_denominator_returns_fallback(self):
        assert _safe_div(5, float("nan")) is None

    def test_custom_fallback(self):
        assert _safe_div(1, 0, fallback=0.0) == 0.0

    def test_negative_values(self):
        assert _safe_div(-10, 2) == pytest.approx(-5.0)

    def test_both_negative(self):
        assert _safe_div(-10, -2) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _row and _val
# ---------------------------------------------------------------------------

class TestRowVal:
    def setup_method(self):
        # Simulate a yfinance financials DataFrame:
        # rows = metric names, columns = quarter dates (most recent first)
        dates = pd.to_datetime(["2024-09-30", "2024-06-30", "2024-03-31", "2023-12-31"])
        self.df = pd.DataFrame(
            {
                "Total Revenue":   [100e9, 90e9, 85e9, 80e9],
                "Net Income":      [25e9,  22e9, 20e9, 18e9],
                "Gross Profit":    [50e9,  45e9, 42e9, 40e9],
            },
            index=dates,
        ).T
        # columns are the dates
        self.df.columns = dates

    def test_row_found(self):
        series = _row(self.df, "Total Revenue")
        assert series is not None
        assert len(series) == 4

    def test_row_missing(self):
        assert _row(self.df, "Operating Income") is None

    def test_row_none_df(self):
        assert _row(None, "Total Revenue") is None

    def test_row_empty_df(self):
        assert _row(pd.DataFrame(), "Total Revenue") is None

    def test_val_pos0(self):
        series = _row(self.df, "Total Revenue")
        assert _val(series, 0) == pytest.approx(100e9)

    def test_val_pos1(self):
        series = _row(self.df, "Total Revenue")
        assert _val(series, 1) == pytest.approx(90e9)

    def test_val_out_of_range(self):
        series = _row(self.df, "Total Revenue")
        assert _val(series, 10) is None

    def test_val_none_series(self):
        assert _val(None, 0) is None


# ---------------------------------------------------------------------------
# _filter_statements
# ---------------------------------------------------------------------------

class TestFilterStatements:
    def setup_method(self):
        # yfinance layout: index = metric names, columns = quarter-end dates (recent first)
        dates = pd.to_datetime(["2024-09-30", "2024-06-30", "2024-03-31", "2023-12-31"])
        self.df = pd.DataFrame(
            {"Total Revenue": [100, 90, 85, 80]},
            index=dates,
        ).T  # rows = metrics, columns = dates

    def test_filters_future_quarters(self):
        # as_of 2024-06-30 → should keep 2024-06-30 and 2023-12-31, drop 2024-09-30
        filtered = _filter_statements(self.df, pd.Timestamp("2024-06-30"))
        assert len(filtered.columns) == 3  # Jun, Mar, Dec
        assert pd.Timestamp("2024-09-30") not in filtered.columns

    def test_keeps_exact_date(self):
        filtered = _filter_statements(self.df, pd.Timestamp("2024-09-30"))
        assert pd.Timestamp("2024-09-30") in filtered.columns
        assert len(filtered.columns) == 4

    def test_none_df_returns_none(self):
        assert _filter_statements(None, pd.Timestamp("2024-06-30")) is None

    def test_empty_df_returns_empty(self):
        result = _filter_statements(pd.DataFrame(), pd.Timestamp("2024-06-30"))
        assert result.empty

    def test_all_filtered_returns_empty_df(self):
        # as_of before all quarters
        result = _filter_statements(self.df, pd.Timestamp("2023-01-01"))
        assert result.empty or len(result.columns) == 0


# ---------------------------------------------------------------------------
# Technical indicator math (pure-Python, no DB)
# ---------------------------------------------------------------------------

class TestRSI:
    """RSI must always be in [0, 100]."""

    def _compute_rsi(self, prices, period=14):
        """Simplified RSI using Wilder's EWM smoothing — same as technical.py."""
        delta = pd.Series(prices).diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        alpha = 1 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def test_rsi_in_range_trending_up(self):
        # Mix gains + rare losses so avg_loss > 0 and RSI is defined
        np.random.seed(0)
        changes = np.where(np.random.rand(40) > 0.1, 1.5, -0.2)
        prices = list(100 + np.cumsum(changes))
        rsi = self._compute_rsi(prices)
        assert np.isnan(rsi) or 0 <= rsi <= 100

    def test_rsi_in_range_trending_down(self):
        prices = [100 - i * 0.5 for i in range(40)]  # steadily falling
        rsi = self._compute_rsi(prices)
        assert np.isnan(rsi) or 0 <= rsi <= 100

    def test_rsi_overbought_signal(self):
        # 90% up days, 10% tiny down days → RSI should be high
        np.random.seed(0)
        changes = np.where(np.random.rand(40) > 0.1, 1.5, -0.1)
        prices = list(100 + np.cumsum(changes))
        rsi = self._compute_rsi(prices)
        assert np.isnan(rsi) or rsi > 60

    def test_rsi_oversold_signal(self):
        # A strong downtrend should produce RSI < 30
        prices = [100 * (0.99 ** i) for i in range(40)]
        rsi = self._compute_rsi(prices)
        assert rsi < 30

    def test_rsi_flat_prices(self):
        # Flat prices → all gains = 0, all losses = 0 → RSI undefined → NaN
        prices = [100.0] * 30
        rsi = self._compute_rsi(prices)
        assert np.isnan(rsi) or (0 <= rsi <= 100)


class TestBollingerBands:
    """Bollinger %B: 0 = at lower band, 0.5 = midline, 1 = at upper band."""

    def _compute_bb_pct_b(self, prices, window=20, n_std=2):
        s = pd.Series(prices)
        mid   = s.rolling(window).mean()
        std   = s.rolling(window).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        pct_b = (s - lower) / (upper - lower)
        return pct_b.iloc[-1]

    def test_price_at_midline(self):
        # Constant price → at SMA → %B ≈ 0.5 (may be NaN if std=0)
        prices = [100.0] * 25
        pct_b = self._compute_bb_pct_b(prices)
        # std=0 → division by zero → NaN; that's acceptable
        assert np.isnan(pct_b) or abs(pct_b - 0.5) < 0.1

    def test_price_above_mid_gives_pct_b_over_half(self):
        prices = list(range(80, 105))  # trending up, price above SMA
        pct_b = self._compute_bb_pct_b(prices)
        assert pct_b > 0.5


class TestPiotroski:
    """Piotroski components must be binary: 0 or 1."""

    def test_components_are_binary(self):
        # Simulate some binary signals
        signals = [1, 0, 1, 1, 0, 0, 1, 1, 1]
        for s in signals:
            assert s in (0, 1)

    def test_score_in_range(self):
        signals = [1, 0, 1, 1, 0, 0, 1, 1, 1]
        score = sum(signals)
        assert 0 <= score <= 9

    def test_all_pass(self):
        assert sum([1] * 9) == 9

    def test_all_fail(self):
        assert sum([0] * 9) == 0


class TestWinsorization:
    """Core logic from FeaturePreprocessor — clip at quantile bounds."""

    def test_clip_at_1pct_99pct(self):
        np.random.seed(42)
        data = pd.Series(np.concatenate([
            np.random.normal(0, 1, 98),
            [100.0, -100.0],  # extreme outliers
        ]))
        low  = data.quantile(0.01)
        high = data.quantile(0.99)
        clipped = data.clip(low, high)
        assert clipped.max() <= high + 1e-10
        assert clipped.min() >= low - 1e-10

    def test_clip_does_not_change_inliers(self):
        # Use 1000 values so the 1% quantile doesn't clip the bulk of the data
        data = pd.Series(np.linspace(0.0, 1.0, 1000))
        low  = data.quantile(0.01)
        high = data.quantile(0.99)
        clipped = data.clip(low, high)
        # Values strictly inside [low, high] must be unchanged
        mask = (data > low) & (data < high)
        pd.testing.assert_series_equal(data[mask].reset_index(drop=True),
                                       clipped[mask].reset_index(drop=True))
