"""
tests/test_models.py

Unit tests for models/preprocessing.py (FeaturePreprocessor).

Key invariants tested:
  1. fit_transform produces z-scored output (mean≈0, std≈1 globally)
  2. transform uses training params — does NOT refit on val/test data
  3. Winsorization clips before z-scoring (tested via clip_bounds)
  4. Sector z-scoring groups stocks separately
  5. save/load round-trip preserves all fit parameters exactly
  6. coverage_report returns per-feature null rates

Run with: pytest tests/test_models.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.preprocessing import FeaturePreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n=100, seed=42):
    """
    Create a synthetic feature DataFrame with known properties.
    Includes one feature in SECTOR_NORM_FEATURES (pe_ratio) and
    several global features (rsi_14, momentum_30d, etc.).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pe_ratio":      rng.normal(20, 5,   n),   # sector-normed
        "rsi_14":        rng.uniform(20, 80, n),   # global
        "momentum_30d":  rng.normal(0, 0.05, n),   # global
        "atr_14":        rng.normal(5, 2,    n),   # global
        "volume_ratio":  rng.normal(1, 0.3,  n),   # global
    })


def _make_sectors(n=100, seed=42):
    """Return a Series of sector labels aligned with the DataFrame."""
    rng = np.random.default_rng(seed)
    sectors = ["Technology", "Financials", "Healthcare"]
    return pd.Series(rng.choice(sectors, size=n), name="sector")


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_output_shape_matches_input(self):
        df = _make_df()
        pp = FeaturePreprocessor()
        out = pp.fit_transform(df.copy())
        assert out.shape == df.shape

    def test_global_features_mean_near_zero(self):
        df = _make_df(n=500)
        pp = FeaturePreprocessor()
        out = pp.fit_transform(df.copy())
        for col in ["rsi_14", "momentum_30d", "atr_14", "volume_ratio"]:
            assert abs(out[col].mean()) < 0.05, f"{col} mean not near 0"

    def test_global_features_std_near_one(self):
        df = _make_df(n=500)
        pp = FeaturePreprocessor()
        out = pp.fit_transform(df.copy())
        for col in ["rsi_14", "momentum_30d", "atr_14", "volume_ratio"]:
            assert abs(out[col].std() - 1.0) < 0.1, f"{col} std not near 1"

    def test_is_fitted_set_after_fit(self):
        pp = FeaturePreprocessor()
        assert not pp.is_fitted
        pp.fit_transform(_make_df().copy())
        assert pp.is_fitted

    def test_clip_bounds_populated(self):
        df = _make_df()
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy())
        assert len(pp.clip_bounds) == len(df.columns)
        for col in df.columns:
            low, high = pp.clip_bounds[col]
            assert low <= high

    def test_means_and_stds_populated_for_global_features(self):
        df = _make_df()
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy())
        for col in ["rsi_14", "momentum_30d"]:
            assert col in pp.means
            assert col in pp.stds
            assert pp.stds[col] > 0

    def test_sector_normed_feature_not_in_global_means(self):
        df = _make_df()
        sectors = _make_sectors()
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy(), sectors=sectors)
        # pe_ratio uses sector z-scoring → should NOT be in global means
        assert "pe_ratio" not in pp.means

    def test_sector_normed_feature_in_sector_params(self):
        df = _make_df()
        sectors = _make_sectors()
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy(), sectors=sectors)
        assert "pe_ratio" in pp.sector_means
        assert "pe_ratio" in pp.sector_stds


# ---------------------------------------------------------------------------
# transform (val/test — must use training params only)
# ---------------------------------------------------------------------------

class TestTransform:
    def test_raises_if_not_fitted(self):
        pp = FeaturePreprocessor()
        with pytest.raises(RuntimeError):
            pp.transform(_make_df())

    def test_transform_uses_training_mean(self):
        """
        If we fit on data with mean 20 for rsi_14, then transform a value
        of 20, the output should be ≈ 0.
        """
        train = pd.DataFrame({"rsi_14": [20.0] * 100})
        # Add tiny variance so std > 0
        train["rsi_14"] += np.linspace(-0.01, 0.01, 100)
        pp = FeaturePreprocessor()
        pp.fit_transform(train.copy())

        val = pd.DataFrame({"rsi_14": [20.0]})
        out = pp.transform(val)
        assert abs(out["rsi_14"].iloc[0]) < 0.5  # near 0

    def test_transform_does_not_refit_on_outlier(self):
        """
        An outlier in val data should NOT shift the z-score params.
        The val outlier gets clipped to train clip_bounds, then z-scored
        with training mean/std — so training mean/std are unchanged.
        """
        train = _make_df(n=200)
        pp = FeaturePreprocessor()
        pp.fit_transform(train.copy())
        train_means = dict(pp.means)

        val = _make_df(n=10)
        val["rsi_14"] = 9999.0  # extreme outlier in val
        pp.transform(val)

        # Means must be identical to training means
        assert pp.means == train_means

    def test_new_column_falls_back_to_zero_mean_one_std(self):
        """
        A feature in val that wasn't in training gets mean=0, std=1 fallback.
        """
        train = pd.DataFrame({"rsi_14": np.random.normal(50, 10, 100)})
        pp = FeaturePreprocessor()
        pp.fit_transform(train.copy())

        # val has an extra column not seen in training
        val = pd.DataFrame({"rsi_14": [50.0], "unknown_feature": [100.0]})
        out = pp.transform(val)
        # unknown_feature: mean=0, std=1 → (100 - 0) / 1 = 100
        assert out["unknown_feature"].iloc[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

class TestWinsorization:
    def test_winsorization_clips_outliers(self):
        data = pd.DataFrame({
            "rsi_14": list(np.random.normal(50, 10, 98)) + [9999.0, -9999.0]
        })
        pp = FeaturePreprocessor(winsor_quantile=0.01)
        out = pp.fit_transform(data.copy())

        low, high = pp.clip_bounds["rsi_14"]
        # After clipping + z-scoring, no value should correspond to 9999
        # The clipped values map to max/min of clipped distribution
        raw_clipped_max = (high - pp.means.get("rsi_14", high)) / pp.stds.get("rsi_14", 1)
        raw_clipped_min = (low  - pp.means.get("rsi_14", low))  / pp.stds.get("rsi_14", 1)
        assert out["rsi_14"].max() <= raw_clipped_max + 1e-9
        assert out["rsi_14"].min() >= raw_clipped_min - 1e-9

    def test_clip_bounds_are_training_quantiles(self):
        data = pd.DataFrame({"rsi_14": np.arange(100, dtype=float)})
        pp = FeaturePreprocessor(winsor_quantile=0.05)
        pp.fit_transform(data.copy())
        low, high = pp.clip_bounds["rsi_14"]
        assert low  == pytest.approx(data["rsi_14"].quantile(0.05),  abs=1)
        assert high == pytest.approx(data["rsi_14"].quantile(0.95),  abs=1)


# ---------------------------------------------------------------------------
# Sector z-scoring
# ---------------------------------------------------------------------------

class TestSectorZScore:
    def test_each_sector_has_separate_mean(self):
        n = 300
        rng = np.random.default_rng(0)
        # Tech PE centered at 30, Financials at 10
        pe = np.where(
            np.arange(n) < 150,
            rng.normal(30, 3, n),
            rng.normal(10, 2, n),
        )
        df = pd.DataFrame({"pe_ratio": pe})
        sectors = pd.Series(["Technology"] * 150 + ["Financials"] * 150)

        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy(), sectors=sectors)

        tech_mean = pp.sector_means["pe_ratio"]["Technology"]
        fin_mean  = pp.sector_means["pe_ratio"]["Financials"]
        assert abs(tech_mean - 30) < 2
        assert abs(fin_mean  - 10) < 2

    def test_sector_output_mean_near_zero_per_sector(self):
        n = 300
        rng = np.random.default_rng(1)
        pe = np.where(np.arange(n) < 150, rng.normal(30, 3, n), rng.normal(10, 2, n))
        df = pd.DataFrame({"pe_ratio": pe})
        sectors = pd.Series(["Technology"] * 150 + ["Financials"] * 150)

        pp = FeaturePreprocessor()
        out = pp.fit_transform(df.copy(), sectors=sectors)

        tech_out = out.loc[sectors == "Technology", "pe_ratio"]
        fin_out  = out.loc[sectors == "Financials",  "pe_ratio"]
        assert abs(tech_out.mean()) < 0.1
        assert abs(fin_out.mean())  < 0.1


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

class TestImputation:
    def test_piotroski_columns_filled_with_zero(self):
        df = pd.DataFrame({
            "p_positive_net_income": [1.0, None, 0.0],
            "rsi_14":                [50.0, 60.0, 40.0],
        })
        pp = FeaturePreprocessor()
        out = pp.fit_transform(df)
        assert out["p_positive_net_income"].isna().sum() == 0

    def test_sentiment_columns_filled_with_zero(self):
        df = pd.DataFrame({
            "reddit_sentiment_avg_7d": [0.3, None, -0.1],
            "rsi_14":                  [50.0, 60.0, 40.0],
        })
        pp = FeaturePreprocessor()
        out = pp.fit_transform(df)
        assert out["reddit_sentiment_avg_7d"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip_preserves_means(self, tmp_path, monkeypatch):
        # Monkeypatch ROOT so save goes to tmp_path
        import models.preprocessing as pp_module
        monkeypatch.setattr(pp_module, "ROOT", tmp_path)
        (tmp_path / "models").mkdir()

        df = _make_df(n=200)
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy())
        pp.save("test_v1")

        pp2 = FeaturePreprocessor.load("test_v1")
        assert pp2.means == pp.means
        assert pp2.stds  == pp.stds

    def test_round_trip_preserves_clip_bounds(self, tmp_path, monkeypatch):
        import models.preprocessing as pp_module
        monkeypatch.setattr(pp_module, "ROOT", tmp_path)
        (tmp_path / "models").mkdir()

        df = _make_df(n=200)
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy())
        pp.save("test_v2")

        pp2 = FeaturePreprocessor.load("test_v2")
        assert pp2.clip_bounds == pp.clip_bounds

    def test_loaded_preprocessor_is_fitted(self, tmp_path, monkeypatch):
        import models.preprocessing as pp_module
        monkeypatch.setattr(pp_module, "ROOT", tmp_path)
        (tmp_path / "models").mkdir()

        df = _make_df()
        pp = FeaturePreprocessor()
        pp.fit_transform(df.copy())
        pp.save("test_v3")

        pp2 = FeaturePreprocessor.load("test_v3")
        assert pp2.is_fitted


# ---------------------------------------------------------------------------
# coverage_report
# ---------------------------------------------------------------------------

class TestCoverageReport:
    def test_returns_dataframe(self):
        df = _make_df()
        df.loc[0, "rsi_14"] = None  # introduce one null
        pp = FeaturePreprocessor()
        report = pp.coverage_report(df)
        assert isinstance(report, pd.DataFrame)

    def test_sorted_by_non_null_pct_ascending(self):
        df = _make_df()
        df["rsi_14"] = np.nan  # all null — use np.nan, not None, for float columns
        pp = FeaturePreprocessor()
        report = pp.coverage_report(df)
        assert report["non_null_pct"].iloc[0] == pytest.approx(0.0)

    def test_fully_populated_column_is_100pct(self):
        df = pd.DataFrame({"rsi_14": [50.0] * 100})
        pp = FeaturePreprocessor()
        report = pp.coverage_report(df)
        assert report.loc[report["feature"] == "rsi_14", "non_null_pct"].iloc[0] == 100.0
