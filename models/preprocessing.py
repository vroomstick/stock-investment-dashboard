"""
models/preprocessing.py

Feature preprocessing pipeline: winsorization, z-score normalization,
and missing-value imputation. Keeps raw features in the DB untouched —
this module transforms them at training/inference time.

Design principles:
  1. Fit on TRAINING data only — never on val/test (prevents leakage).
  2. Store fit parameters (means, stds, clip bounds) so the same transform
     can be applied to new daily predictions without refitting.
  3. Raw features stay raw in the DB — if we change the normalization
     strategy, we refit the preprocessor, not the feature tables.

Normalization strategies (per feature_config.yaml):
  - Fundamental features: z-score WITHIN sector (peer-relative)
  - Technical features:   z-score GLOBALLY (already ratio-based)
  - Macro features:       z-score GLOBALLY (one value per date, universal)
  - Sentiment features:   z-score GLOBALLY (signed scores, universal meaning)

Winsorization:
  Clip outliers to the [1%, 99%] percentile range before z-scoring.
  Financial data has extreme values (e.g., P/E of 1000 for near-zero earnings).
  Without winsorization, one outlier dominates the z-score normalization.

Missing value imputation (training time only):
  - Fundamental: forward-fill within each stock, then fill with sector median.
  - Technical:   forward-fill within each stock, then fill with 0 (neutral).
  - Sentiment:   fill with 0 (absence of signal = neutral).
  - Macro:       forward-fill across dates (FRED releases lag by days/weeks).

Versioning:
  Preprocessor state is saved to disk with a version tag matching the model.
  This ensures prediction-time transforms match training-time transforms exactly.
"""

import json
import pickle
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class FeaturePreprocessor:
    """
    Stateful preprocessing pipeline.

    Usage pattern:
      # Training:
      pp = FeaturePreprocessor()
      X_train = pp.fit_transform(df_train, sectors=sectors_train)
      pp.save("v1")

      # Inference / validation:
      pp = FeaturePreprocessor.load("v1")
      X_new = pp.transform(df_new, sectors=sectors_new)

    Attributes populated after fit():
      clip_bounds: dict[feature] -> (low, high)  — 1%/99% clip values
      means:       dict[feature] -> float          — mean for z-scoring
      stds:        dict[feature] -> float          — std for z-scoring
      sector_means: dict[sector][feature] -> float — for within-sector z-score
      sector_stds:  dict[sector][feature] -> float
    """

    # Features that use within-sector z-scoring (everything else is global)
    SECTOR_NORM_FEATURES = {
        "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda",
        "peg_ratio", "pe_vs_sector_avg", "ev_revenue", "price_to_fcf",
        "earnings_yield", "fcf_yield", "price_to_tangible_book",
        "roe", "roa", "roic", "gross_margin", "operating_margin",
        "net_margin", "ebitda_margin", "fcf_margin",
        "revenue_growth_yoy", "earnings_growth_yoy",
        "debt_to_equity", "current_ratio", "quick_ratio",
        "net_debt_ebitda", "altman_z_score", "piotroski_f_score",
        "asset_turnover", "inventory_turnover",
    }

    def __init__(self, winsor_quantile: float = 0.01):
        """
        winsor_quantile: clip at this percentile on each side (default 1%).
        """
        self.winsor_quantile = winsor_quantile
        self.clip_bounds:  dict = {}
        self.means:        dict = {}
        self.stds:         dict = {}
        self.sector_means: dict = {}  # {sector: {feature: mean}}
        self.sector_stds:  dict = {}
        self.feature_cols: list = []
        self.is_fitted:    bool = False

    def fit_transform(self, df: pd.DataFrame,
                      sectors: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit on training data and return transformed DataFrame.
        df: feature matrix (rows = samples, cols = features)
        sectors: Series of sector labels aligned with df index.
        """
        self.feature_cols = list(df.columns)
        df = df.copy()

        # Step 1: Winsorize (fit clip bounds on training data)
        for col in df.columns:
            low  = df[col].quantile(self.winsor_quantile)
            high = df[col].quantile(1 - self.winsor_quantile)
            self.clip_bounds[col] = (float(low), float(high))
            df[col] = df[col].clip(low, high)

        # Step 2: Impute missing values (training-time fill)
        df = self._impute(df)

        # Step 3: Z-score normalization
        for col in df.columns:
            if col in self.SECTOR_NORM_FEATURES and sectors is not None:
                df[col] = self._fit_sector_zscore(df[col], sectors, col)
            else:
                mean = df[col].mean()
                std  = df[col].std()
                self.means[col] = float(mean)
                self.stds[col]  = float(std) if std > 0 else 1.0
                df[col] = (df[col] - mean) / self.stds[col]

        self.is_fitted = True
        return df

    def transform(self, df: pd.DataFrame,
                  sectors: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply fitted transforms to new data (val/test/inference).
        Does NOT refit — uses parameters learned from training set.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() before transform()")

        df = df.copy()

        # Winsorize using training clip bounds
        for col in df.columns:
            if col in self.clip_bounds:
                low, high = self.clip_bounds[col]
                df[col] = df[col].clip(low, high)

        # Impute
        df = self._impute(df)

        # Z-score using training means/stds
        for col in df.columns:
            if col in self.SECTOR_NORM_FEATURES and sectors is not None:
                df[col] = self._apply_sector_zscore(df[col], sectors, col)
            else:
                mean = self.means.get(col, 0.0)
                std  = self.stds.get(col, 1.0)
                df[col] = (df[col] - mean) / std

        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values. Strategy depends on column type.
        Binary Piotroski components get filled with 0 (unknown = not confirmed).
        Sentiment gets 0 (neutral). All others get 0 after forward-fill.
        The proper forward-fill within each stock happens at query time
        (fetch most recent non-null value per feature).
        """
        p_cols = [c for c in df.columns if c.startswith("p_")]
        sent_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in ("reddit_", "news_", "insider_",
                                              "activist_", "wsb_", "negative_",
                                              "material_"))
        ]

        df[p_cols]    = df[p_cols].fillna(0)
        df[sent_cols] = df[sent_cols].fillna(0)
        df            = df.fillna(0)  # remaining: global 0 fill after winsorize
        return df

    def _fit_sector_zscore(self, series: pd.Series, sectors: pd.Series,
                            col: str) -> pd.Series:
        """Fit within-sector z-score and return transformed values."""
        if col not in self.sector_means:
            self.sector_means[col] = {}
            self.sector_stds[col]  = {}

        result = series.copy()
        for sector in sectors.unique():
            mask = sectors == sector
            vals = series[mask]
            mean = float(vals.mean())
            std  = float(vals.std()) if vals.std() > 0 else 1.0
            self.sector_means[col][sector] = mean
            self.sector_stds[col][sector]  = std
            result[mask] = (vals - mean) / std

        return result

    def _apply_sector_zscore(self, series: pd.Series, sectors: pd.Series,
                              col: str) -> pd.Series:
        """Apply pre-fitted sector z-score params."""
        result = series.copy()
        sm = self.sector_means.get(col, {})
        ss = self.sector_stds.get(col, {})

        for sector in sectors.unique():
            mask = sectors == sector
            mean = sm.get(sector, 0.0)
            std  = ss.get(sector, 1.0)
            result[mask] = (series[mask] - mean) / std

        return result

    def save(self, version: str):
        """Persist fit parameters to disk."""
        path = ROOT / "models" / f"preprocessor_{version}.pkl"
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Preprocessor saved: {path}")

    @classmethod
    def load(cls, version: str) -> "FeaturePreprocessor":
        """Load a previously fitted preprocessor."""
        path = ROOT / "models" / f"preprocessor_{version}.pkl"
        with open(path, "rb") as f:
            pp = pickle.load(f)
        print(f"  Preprocessor loaded: {path}")
        return pp

    def coverage_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame showing % non-null per feature before imputation.
        Useful for monitoring feature completeness over time.
        """
        report = pd.DataFrame({
            "feature": df.columns,
            "non_null_pct": (df.notna().mean() * 100).round(1).values,
            "null_count": df.isna().sum().values,
            "mean": df.mean().round(4).values,
            "std": df.std().round(4).values,
            "p01": df.quantile(0.01).round(4).values,
            "p99": df.quantile(0.99).round(4).values,
        })
        return report.sort_values("non_null_pct")
