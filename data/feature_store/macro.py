"""
data/feature_store/macro.py

Macro feature store entry point.

The macro data collector (data/collectors/macro_data.py) already handles
the full pipeline: fetching from FRED, computing derived features, and
storing to the macro_features table. This module is the canonical entry
point called by data/pipeline.py so that the pipeline doesn't need to
know which collector module does the work.

Macro features are stock-agnostic — one row per date, shared across
all stocks. At training time, each stock's feature vector joins the
most recent available macro row by date.
"""

from data.collectors.macro_data import run as _macro_run


def run(as_of_date: str = None):
    """
    Fetch all macro features and store them for as_of_date.
    Delegates entirely to the macro_data collector.
    Called by data/pipeline.py once per daily run (not per stock).
    """
    _macro_run(as_of_date)
