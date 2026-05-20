"""
data/collectors/macro_data.py

Fetches macroeconomic data from FRED (Federal Reserve Economic Data).
Computes derived macro features and stores them in macro_features.

FRED publishes hundreds of economic time series. We use 12 core series
(defined in settings.py) and derive additional features from them
(yield curve slope, credit spread, rate of change, etc.)

FRED data quirks:
- Different series have different release cadences (daily, weekly, monthly, quarterly)
- Data lags: GDP is released weeks after the quarter ends
- Missing values: FRED forward-fills internally but we still check
- All series are stored as floats; units vary by series
"""

import os
from datetime import date, timedelta


import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from fredapi import Fred

from config.settings import FRED_API_KEY, DATA
from database.db import get_db, fetch_one


# ---------------------------------------------------------------------------
# FRED client
# ---------------------------------------------------------------------------

def _get_fred() -> Fred:
    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=FRED_API_KEY)


def fetch_series(fred: Fred, series_id: str, years_back: int = 6,
                 anchor: str = None) -> pd.Series:
    """
    Fetch a FRED series going back N years from anchor date.

    anchor: ISO date string to anchor the history window (default: today).
            Pass as_of_date here for historical backfills so the window
            is correctly centered on the target date, not today.
            If as_of_date is more than years_back years ago, without this
            the start date would be too recent and we'd miss the data.

    Forward-fill: if GDP is released monthly, the daily series has NaN
    on non-release days. Forward-filling carries the last known value
    forward — correct for most macro indicators.
    """
    anchor_date = date.fromisoformat(anchor) if anchor else date.today()
    start = (anchor_date - timedelta(days=365 * years_back)).isoformat()
    series = fred.get_series(series_id, observation_start=start)
    return series.ffill()


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_macro_features(as_of_date: str = None) -> dict:
    """
    Fetch all FRED series and compute macro features for a given date.
    Defaults to today if as_of_date is None.

    Returns a flat dict of feature_name -> float value.
    This dict maps directly to columns in the macro_features table.
    """
    fred = _get_fred()
    as_of = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()

    def latest(series: pd.Series) -> float:
        """Get the most recent value on or before as_of_date."""
        s = series[series.index <= as_of]
        return float(s.iloc[-1]) if not s.empty else None

    def delta(series: pd.Series, days: int) -> float:
        """Compute change over N calendar days."""
        s = series[series.index <= as_of]
        s_past = series[series.index <= as_of - timedelta(days=days)]
        if s.empty or s_past.empty:
            return None
        return float(s.iloc[-1]) - float(s_past.iloc[-1])

    def pct_change(series: pd.Series, periods: int = 12) -> float:
        """Year-over-year percent change (periods = 12 months for monthly data)."""
        s = series[series.index <= as_of]
        if len(s) < periods + 1:
            return None
        return float((s.iloc[-1] - s.iloc[-periods]) / abs(s.iloc[-periods]))

    def percentile(series: pd.Series, window_days: int = 365) -> float:
        """Where does the latest value sit in the distribution over the window?"""
        s = series[series.index <= as_of]
        window = s[s.index >= as_of - timedelta(days=window_days)]
        if window.empty:
            return None
        current = s.iloc[-1]
        return float((window < current).mean())

    print("  Fetching FRED series...")

    # Fetch raw series — anchor to as_of_date so the 6yr history window is
    # always relative to the target date, not today. Critical for backfills:
    # without this, a 2020 backfill run would compute start = today - 6yr,
    # which might still include the right range, but a run older than 6yr
    # would miss data entirely. Anchoring is always correct.
    anchor = as_of_date  # str or None — fetch_series handles both
    gs10    = fetch_series(fred, "GS10",     anchor=anchor)
    gs2     = fetch_series(fred, "GS2",      anchor=anchor)
    fedfunds= fetch_series(fred, "FEDFUNDS", anchor=anchor)
    cpi     = fetch_series(fred, "CPIAUCSL", anchor=anchor)
    unrate  = fetch_series(fred, "UNRATE",   anchor=anchor)
    vix     = fetch_series(fred, "VIXCLS",   anchor=anchor)
    baa     = fetch_series(fred, "BAA",      anchor=anchor)
    aaa     = fetch_series(fred, "AAA",      anchor=anchor)
    icsa    = fetch_series(fred, "ICSA",     anchor=anchor)
    m2      = fetch_series(fred, "M2SL",     anchor=anchor)
    dollar  = fetch_series(fred, "DTWEXBGS", anchor=anchor)

    credit_spread = baa - aaa  # BAA - AAA: wider = more risk aversion

    print("  Computing derived features...")

    features = {
        # --- Interest Rates ---
        "treasury_10y":       latest(gs10),
        "treasury_2y":        latest(gs2),
        "yield_curve_slope":  latest(gs10) - latest(gs2) if latest(gs10) and latest(gs2) else None,
        "fed_funds_rate":     latest(fedfunds),
        "fed_rate_change_3m": delta(fedfunds, 90),
        "credit_spread":      latest(credit_spread),
        "credit_spread_change": delta(credit_spread, 30),

        # --- Economic Health ---
        "unemployment_rate":      latest(unrate),
        "unemployment_change":    delta(unrate, 90),
        "cpi_yoy":                pct_change(cpi, 12),
        "cpi_change_3m":          delta(cpi, 90),
        "initial_claims":         latest(icsa),
        "initial_claims_4wk_avg": float(
            icsa[icsa.index <= as_of].iloc[-4:].mean()
        ) if len(icsa[icsa.index <= as_of]) >= 4 else latest(icsa),
        "m2_growth":              pct_change(m2, 12),
        "dollar_index":           latest(dollar),
        "dollar_change_30d":      delta(dollar, 30),

        # --- Market Regime ---
        "vix_level":          latest(vix),
        "vix_change_5d":      delta(vix, 5),
        "vix_percentile_1y":  percentile(vix, 365),
        "market_breadth":     _compute_market_breadth(),

        # GDP growth fetched separately (quarterly, different handling)
        "gdp_growth": _fetch_gdp_growth(fred, as_of),
    }

    # Sector ETF momentum/relative flow placeholders — computed separately
    # by compute_sector_momentum() and merged in run()
    sector_defaults = {
        "sector_momentum_xlk": None,
        "sector_momentum_xlf": None,
        "sector_momentum_xlv": None,
        "sector_momentum_xle": None,
        "sector_momentum_xli": None,
        "sector_momentum_xlp": None,
        "sector_momentum_xly": None,
        "growth_vs_value":     None,
    }
    features.update(sector_defaults)

    return features


def _compute_market_breadth() -> float | None:
    """
    % of S&P 500 stocks trading above their 200-day moving average.
    A proxy for how broadly the rally/selloff is distributed.

    > 70%: broad bull market (healthy)
    < 30%: broad bear market (risk-off)
    40-60%: mixed / rotating market

    We use all active non-benchmark stocks in our universe as the sample.
    Fetching all 500 S&P constituents daily would hit rate limits; our
    universe of large-caps is a sufficient proxy for the signal.
    """
    import yfinance as yf
    from database.db import get_all_active_tickers

    sample = [t for t in get_all_active_tickers() if t != "SPY"]
    above_200 = 0
    valid = 0

    for ticker in sample:
        try:
            hist = yf.Ticker(ticker).history(period="1y")["Close"]
            if len(hist) >= 200:
                sma200 = hist.rolling(200).mean().iloc[-1]
                if hist.iloc[-1] > sma200:
                    above_200 += 1
                valid += 1
        except Exception:
            continue

    return round(above_200 / valid, 4) if valid > 0 else None


def _fetch_gdp_growth(fred: Fred, as_of: pd.Timestamp) -> float | None:
    """
    GDP growth rate: quarter-over-quarter annualized change.
    GDP is released quarterly with a ~1 month lag, so we forward-fill.
    """
    try:
        gdp = fetch_series(fred, "GDP")
        s = gdp[gdp.index <= as_of]
        if len(s) < 2:
            return None
        qoq = (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2])
        return float(qoq * 4)  # annualize: multiply quarterly rate by 4
    except Exception:
        return None


def compute_sector_momentum(etf_tickers: list, window_days: int = 30) -> dict:
    """
    Compute 30-day momentum for each sector ETF using yfinance.
    Called from pipeline.py after price data is updated.

    Momentum = (current price / price N days ago) - 1
    """
    import yfinance as yf

    results = {}
    etf_map = {
        "XLK": "sector_momentum_xlk",
        "XLF": "sector_momentum_xlf",
        "XLV": "sector_momentum_xlv",
        "XLE": "sector_momentum_xle",
        "XLI": "sector_momentum_xli",
        "XLP": "sector_momentum_xlp",
        "XLY": "sector_momentum_xly",
    }

    for ticker, col in etf_map.items():
        try:
            hist = yf.Ticker(ticker).history(period="3mo")["Close"]
            if len(hist) >= window_days:
                momentum = float(hist.iloc[-1] / hist.iloc[-window_days] - 1)
                results[col] = round(momentum, 6)
            else:
                results[col] = None
        except Exception:
            results[col] = None

    # Growth vs Value: IWF (growth) vs IWD (value)
    try:
        iwf = yf.Ticker("IWF").history(period="3mo")["Close"]
        iwd = yf.Ticker("IWD").history(period="3mo")["Close"]
        if len(iwf) >= window_days and len(iwd) >= window_days:
            growth_ret = float(iwf.iloc[-1] / iwf.iloc[-window_days] - 1)
            value_ret  = float(iwd.iloc[-1] / iwd.iloc[-window_days] - 1)
            results["growth_vs_value"] = round(growth_ret - value_ret, 6)
        else:
            results["growth_vs_value"] = None
    except Exception:
        results["growth_vs_value"] = None

    # Sector relative flow: each sector ETF return vs SPY return
    # Positive = sector outperforming the broad market
    sector_relative_flows = {}
    try:
        spy = yf.Ticker("SPY").history(period="3mo")["Close"]
        spy_ret = float(spy.iloc[-1] / spy.iloc[-window_days] - 1) if len(spy) >= window_days else None

        for ticker, col in etf_map.items():
            flow_col = col.replace("sector_momentum_", "sector_relative_flow_")
            if spy_ret is not None and results.get(col) is not None:
                sector_relative_flows[flow_col] = round(results[col] - spy_ret, 6)
            else:
                sector_relative_flows[flow_col] = None
    except Exception:
        for ticker, col in etf_map.items():
            sector_relative_flows[col.replace("sector_momentum_", "sector_relative_flow_")] = None

    results.update(sector_relative_flows)
    return results


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store_macro_features(features: dict, as_of_date: str = None):
    """
    Insert or replace macro features for a given date.
    Uses INSERT OR REPLACE so running daily doesn't create duplicates.
    """
    date_str = as_of_date or date.today().isoformat()

    # Build column list and values dynamically from the features dict
    cols = ", ".join(features.keys())
    placeholders = ", ".join(["?"] * len(features))
    values = list(features.values())

    with get_db() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO macro_features (date, {cols})
                VALUES (?, {placeholders})""",
            [date_str] + values,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(as_of_date: str = None):
    """
    Fetch all macro features and store them.
    Called daily by data/pipeline.py.
    """
    date_str = as_of_date or date.today().isoformat()
    print(f"  Collecting macro features for {date_str}...")

    features = compute_macro_features(date_str)
    sector   = compute_sector_momentum(list(DATA["sector_etfs"].keys()))
    features.update(sector)

    store_macro_features(features, date_str)
    print(f"  Macro features stored: {len(features)} values")
