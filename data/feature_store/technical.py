"""
data/feature_store/technical.py

Computes all technical features defined in spec Section 7B and stores
them in the technical_features table.

Source: daily_prices table (already fetched and stored by price_data.py)

We read price data from our own database rather than re-fetching from
yfinance on every run. This is faster and ensures consistency — the
exact same prices used for features are the ones we trained on.

Feature categories (per spec Section 7B):
  - Moving Averages & Trends    (13 features)
  - Momentum Indicators         (11 features)
  - Volatility                  (8 features)
  - Volume                      (7 features)
  - Relative Strength           (4 features)
  Total: 43 features matching schema columns exactly

All formulas match Appendix B of the spec exactly.
Normalization: z-score globally — happens at training time, not here.
Missing value strategy: forward-fill (series-level, handled naturally
  because we compute from a full price history).
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from database.db import get_db, fetch_all, fetch_one


# ---------------------------------------------------------------------------
# Load price history from DB
# ---------------------------------------------------------------------------

def _load_prices(stock_id: int, as_of_date: str) -> pd.DataFrame:
    """
    Load daily OHLCV from the database up to as_of_date.
    Returns a DataFrame indexed by date (ascending), with columns:
    open, high, low, close, adj_close, volume.

    We use adj_close for all price-based calculations.
    Adjusted close corrects for stock splits and dividends so that
    historical returns are accurate (a 2-for-1 split looks like a 50%
    drop in raw close but is flat in adj_close).
    """
    rows = fetch_all(
        """SELECT date, open, high, low, close, adj_close, volume
           FROM daily_prices
           WHERE stock_id = ? AND date <= ?
           ORDER BY date ASC""",
        (stock_id, as_of_date)
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def _load_ticker_prices(ticker: str, as_of_date: str) -> pd.Series:
    """Load adj_close for any ticker by symbol (used for SPY and sector ETFs)."""
    row = fetch_one("SELECT id FROM stocks WHERE ticker = ?", (ticker,))
    if not row:
        return pd.Series(dtype=float)
    rows = fetch_all(
        """SELECT date, adj_close FROM daily_prices
           WHERE stock_id = ? AND date <= ?
           ORDER BY date ASC""",
        (row["id"], as_of_date)
    )
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["adj_close"]


# ---------------------------------------------------------------------------
# Safe scalar helper
# ---------------------------------------------------------------------------

def _s(value) -> Optional[float]:
    """Convert a pandas scalar to float, or None if NaN/missing."""
    try:
        v = float(value)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Linear regression slope helper
# ---------------------------------------------------------------------------

def _slope(series: pd.Series, window: int) -> Optional[float]:
    """
    Linear regression slope of the last `window` values.
    Normalized by dividing by the mean so the slope is in %/day units,
    comparable across stocks at different price levels.
    """
    if len(series) < window:
        return None
    s = series.iloc[-window:].values
    if np.any(np.isnan(s)):
        return None
    x = np.arange(window, dtype=float)
    slope = np.polyfit(x, s, 1)[0]
    mean  = np.mean(s)
    return float(slope / mean) if mean != 0 else None


# ---------------------------------------------------------------------------
# Moving Averages & Trends — Spec Section 7B
# ---------------------------------------------------------------------------

def _moving_averages(close: pd.Series) -> dict:
    """
    Spec features:
      sma_10, sma_20, sma_50, sma_100, sma_200
      ema_10, ema_20, ema_50
      price_vs_sma50, price_vs_sma200, sma50_vs_sma200
      sma50_slope, sma200_slope

    sma50_slope: 10-day linear regression slope of SMA50.
    Slope > 0 = uptrend, Slope < 0 = downtrend.
    More robust than comparing two SMA values because it captures
    the trend direction over a window, not just a snapshot.
    """
    price = _s(close.iloc[-1])

    sma10  = _s(close.rolling(10).mean().iloc[-1])
    sma20  = _s(close.rolling(20).mean().iloc[-1])
    sma50  = _s(close.rolling(50).mean().iloc[-1])
    sma100 = _s(close.rolling(100).mean().iloc[-1])
    sma200 = _s(close.rolling(200).mean().iloc[-1])

    ema10  = _s(close.ewm(span=10, adjust=False).mean().iloc[-1])
    ema20  = _s(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50  = _s(close.ewm(span=50, adjust=False).mean().iloc[-1])

    price_vs_sma50  = (price - sma50)  / sma50  if price and sma50  else None
    price_vs_sma200 = (price - sma200) / sma200 if price and sma200 else None
    sma50_vs_sma200 = (sma50 - sma200) / sma200 if sma50  and sma200 else None

    sma50_series  = close.rolling(50).mean().dropna()
    sma200_series = close.rolling(200).mean().dropna()
    sma50_slope   = _slope(sma50_series, 10)
    sma200_slope  = _slope(sma200_series, 10)

    return {
        "sma_10":          sma10,
        "sma_20":          sma20,
        "sma_50":          sma50,
        "sma_100":         sma100,
        "sma_200":         sma200,
        "ema_10":          ema10,
        "ema_20":          ema20,
        "ema_50":          ema50,
        "price_vs_sma50":  price_vs_sma50,
        "price_vs_sma200": price_vs_sma200,
        "sma50_vs_sma200": sma50_vs_sma200,
        "sma50_slope":     sma50_slope,
        "sma200_slope":    sma200_slope,
    }


# ---------------------------------------------------------------------------
# Momentum Indicators — Spec Section 7B + Appendix B
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> Optional[float]:
    """
    RSI using Wilder's smoothing (EWM with alpha=1/period).
    Formula: RS = avg_gain / avg_loss; RSI = 100 - 100 / (1 + RS)
    Wilder's smoothing (not simple rolling average) is the correct method —
    it gives exponentially more weight to recent gains/losses.
    """
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - 100 / (1 + rs)
    return _s(rsi.iloc[-1])


def _momentum(close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
    """
    Spec features:
      rsi_14, rsi_7
      macd, macd_signal, macd_histogram
      stochastic_k, stochastic_d
      williams_r
      momentum_10d, momentum_30d, momentum_90d
    """
    rsi_14 = _rsi(close, 14)
    rsi_7  = _rsi(close, 7)

    # MACD: EMA12 - EMA26; signal = 9-day EMA of MACD line
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
    macd       = _s(macd_line.iloc[-1])
    macd_signal  = _s(macd_sig.iloc[-1])
    macd_histogram = (macd - macd_signal) if (macd is not None and macd_signal is not None) else None

    # Stochastic %K / %D (14-day)
    # %K = (Close - LowestLow14) / (HighestHigh14 - LowestLow14) * 100
    # %D = 3-day SMA of %K (signal line)
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    hl_range = high14 - low14
    pct_k  = ((close - low14) / hl_range * 100).where(hl_range != 0)
    pct_d  = pct_k.rolling(3).mean()
    stoch_k = _s(pct_k.iloc[-1])
    stoch_d = _s(pct_d.iloc[-1])

    # Williams %R (14-day): range -100 (oversold) to 0 (overbought)
    # %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100
    will_r = None
    hl14 = _s(hl_range.iloc[-1])
    if hl14 and hl14 != 0:
        will_r = float((high14.iloc[-1] - close.iloc[-1]) / hl14 * -100)

    # Simple momentum: price return over N trading days
    price = close.iloc[-1]
    mom10 = _s(price / close.iloc[-10] - 1) if len(close) >= 10 else None
    mom30 = _s(price / close.iloc[-30] - 1) if len(close) >= 30 else None
    mom90 = _s(price / close.iloc[-90] - 1) if len(close) >= 90 else None

    return {
        "rsi_14":         rsi_14,
        "rsi_7":          rsi_7,
        "macd":           macd,
        "macd_signal":    macd_signal,
        "macd_histogram": macd_histogram,
        "stochastic_k":   stoch_k,
        "stochastic_d":   stoch_d,
        "williams_r":     will_r,
        "momentum_10d":   mom10,
        "momentum_30d":   mom30,
        "momentum_90d":   mom90,
    }


# ---------------------------------------------------------------------------
# Volatility — Spec Section 7B
# ---------------------------------------------------------------------------

def _volatility(close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
    """
    Spec features:
      bollinger_upper, bollinger_lower, bollinger_pct_b, bollinger_bandwidth
      atr_14, realized_vol_20d, realized_vol_60d, vol_ratio

    ATR (Average True Range) measures absolute volatility including gaps:
      True Range = max(H-L, |H-PrevClose|, |L-PrevClose|)
    ATR = EWM(14) of True Range — Wilder's smoothing again.

    Bollinger %B: where price sits within the bands.
    0 = at lower band, 1 = at upper band, >1 = above upper band.

    vol_ratio = 20d vol / 60d vol:
    > 1: volatility is expanding (recent is more volatile than baseline)
    < 1: volatility is contracting (market calming down)
    """
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    price  = _s(close.iloc[-1])
    bb_u   = _s(bb_upper.iloc[-1])
    bb_l   = _s(bb_lower.iloc[-1])
    bb_mid = _s(sma20.iloc[-1])

    bb_pct_b    = (price - bb_l) / (bb_u - bb_l) if (
        price is not None and bb_u is not None and bb_l is not None
        and bb_u != bb_l
    ) else None
    bb_bandwidth = (bb_u - bb_l) / bb_mid if (
        bb_u is not None and bb_l is not None and bb_mid
    ) else None

    # True Range using prev_close = close.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = _s(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    # Realized volatility = annualized std of log returns
    log_ret = np.log(close / close.shift(1))
    rv20 = _s(log_ret.tail(20).std() * np.sqrt(252))
    rv60 = _s(log_ret.tail(60).std() * np.sqrt(252))
    vol_ratio = (rv20 / rv60) if (rv20 and rv60 and rv60 != 0) else None

    return {
        "bollinger_upper":     bb_u,
        "bollinger_lower":     bb_l,
        "bollinger_pct_b":     bb_pct_b,
        "bollinger_bandwidth": bb_bandwidth,
        "atr_14":              atr14,
        "realized_vol_20d":    rv20,
        "realized_vol_60d":    rv60,
        "vol_ratio":           vol_ratio,
    }


# ---------------------------------------------------------------------------
# Volume Indicators — Spec Section 7B
# ---------------------------------------------------------------------------

def _volume_indicators(close: pd.Series, high: pd.Series, low: pd.Series,
                       volume: pd.Series) -> dict:
    """
    Spec features:
      volume_ratio_10d, volume_ratio_30d, volume_trend_20d
      obv, obv_trend
      accumulation_distribution, volume_price_trend

    OBV (On-Balance Volume): cumulative volume flowing in/out.
    Price up → add full volume; Price down → subtract full volume.
    OBV rising while price flat = accumulation (bullish divergence).

    AD Line (Accumulation/Distribution):
    Weights volume by where price closed within the day's range.
    Close near high = more buying pressure. Close near low = more selling.
    Formula: MFM = ((Close-Low) - (High-Close)) / (High-Low)
             AD  = cumsum(MFM * Volume)

    VPT (Volume Price Trend):
    VPT = cumsum(Volume * daily_pct_change)
    Combines price change magnitude with volume, unlike OBV which ignores magnitude.
    """
    vol = volume.astype(float)

    v_today    = _s(vol.iloc[-1])
    avg_vol10  = _s(vol.rolling(10).mean().iloc[-1])
    avg_vol30  = _s(vol.rolling(30).mean().iloc[-1])

    vol_ratio_10d = (v_today / avg_vol10) if (v_today and avg_vol10 and avg_vol10 != 0) else None
    vol_ratio_30d = (v_today / avg_vol30) if (v_today and avg_vol30 and avg_vol30 != 0) else None
    vol_trend_20d = _slope(vol, 20)

    # OBV
    direction  = np.sign(close.diff()).fillna(0)
    obv_series = (direction * vol).cumsum()
    obv_val    = _s(obv_series.iloc[-1])
    obv_trend  = _slope(obv_series, 20)

    # AD Line
    hl = high - low
    mfm       = ((close - low) - (high - close)) / hl.replace(0, np.nan)
    ad_series = (mfm * vol).fillna(0).cumsum()
    ad_val    = _s(ad_series.iloc[-1])

    # VPT
    price_roc  = close.pct_change().fillna(0)
    vpt_series = (price_roc * vol).cumsum()
    vpt_val    = _s(vpt_series.iloc[-1])

    return {
        "volume_ratio_10d":          vol_ratio_10d,
        "volume_ratio_30d":          vol_ratio_30d,
        "volume_trend_20d":          vol_trend_20d,
        "obv":                       obv_val,
        "obv_trend":                 obv_trend,
        "accumulation_distribution": ad_val,
        "volume_price_trend":        vpt_val,
    }


# ---------------------------------------------------------------------------
# Relative Strength — Spec Section 7B
# ---------------------------------------------------------------------------

def _relative_strength(close: pd.Series, spy: pd.Series,
                       sector_close: Optional[pd.Series]) -> dict:
    """
    Spec features:
      relative_strength_vs_spy, relative_strength_vs_sector
      beta_60d, correlation_spy_60d

    Relative strength = stock's 20d return / benchmark's 20d return.
    RS > 1: stock outperforming. RS < 1: underperforming.

    Beta = Cov(stock_returns, spy_returns) / Var(spy_returns)
    Interpretation: Beta=1.5 means stock moves 1.5% per 1% SPY move.
    Beta > 1: aggressive/growth. Beta < 1: defensive.

    We align series by date index before computing — different stocks
    may have slightly different trading days (e.g., foreign exchanges).
    """
    stock_ret = close.pct_change()
    spy_ret   = spy.pct_change()

    # 20-day relative strength vs SPY
    rs_spy = None
    if len(close) >= 20:
        spy_aligned = spy.reindex(close.index, method="ffill")
        if len(spy_aligned) >= 20:
            stock_20d = _s(close.iloc[-1] / close.iloc[-20] - 1)
            spy_20d   = _s(spy_aligned.iloc[-1] / spy_aligned.iloc[-20] - 1)
            if stock_20d is not None and spy_20d is not None and spy_20d != 0:
                rs_spy = stock_20d / spy_20d

    # 20-day relative strength vs sector ETF
    rs_sector = None
    if sector_close is not None and len(sector_close) >= 20 and len(close) >= 20:
        sec_aligned = sector_close.reindex(close.index, method="ffill")
        stock_20d = _s(close.iloc[-1] / close.iloc[-20] - 1)
        sec_20d   = _s(sec_aligned.iloc[-1] / sec_aligned.iloc[-20] - 1)
        if stock_20d is not None and sec_20d is not None and sec_20d != 0:
            rs_sector = stock_20d / sec_20d

    # Beta and correlation: align on common trading days
    combined = pd.concat([stock_ret, spy_ret], axis=1, join="inner").dropna()
    combined.columns = ["stock", "spy"]

    beta        = None
    correlation = None
    if len(combined) >= 60:
        window = combined.tail(60)
        cov    = window["stock"].cov(window["spy"])
        var    = window["spy"].var()
        if var and var != 0:
            beta = float(cov / var)
        correlation = _s(window["stock"].corr(window["spy"]))

    return {
        "relative_strength_vs_spy":    rs_spy,
        "relative_strength_vs_sector": rs_sector,
        "beta_60d":                    beta,
        "correlation_spy_60d":         correlation,
    }


# ---------------------------------------------------------------------------
# Extended Technical Features — filling 60-80 feature target
# ---------------------------------------------------------------------------

def _extended_technical(close: pd.Series, high: pd.Series, low: pd.Series,
                         volume: pd.Series) -> dict:
    """
    Additional technical features beyond the core spec tables.

    CCI (Commodity Channel Index):
      Measures how far price is from its statistical mean.
      CCI > +100: overbought / strong trend. CCI < -100: oversold.
      Formula: (Typical Price - SMA(TP,14)) / (0.015 * Mean Deviation)

    MFI (Money Flow Index):
      Volume-weighted RSI. Uses typical price * volume as "money flow."
      MFI > 80: overbought. MFI < 20: oversold.
      Better than RSI for detecting institutional accumulation/distribution
      because it incorporates volume — pure price RSI misses volume conviction.

    52-week high/low proximity:
      high_52w_pct = (Price - 52w High) / 52w High — negative means below peak
      low_52w_pct  = (Price - 52w Low)  / 52w Low  — positive means above trough
      These capture where in the annual range the stock is trading.
      A stock near its 52w high has price momentum; near its low may be a
      value opportunity or falling knife — context from other features decides.

    ROC (Rate of Change): same as simple momentum but expressed as a ratio.
    atr_pct: ATR divided by price = volatility as % of price. Comparable
    across stocks at different price levels (ATR of $5 means different things
    for a $10 stock vs a $500 stock).
    """
    price = _s(close.iloc[-1])

    # --- Extended price vs MA ---
    sma10  = close.rolling(10).mean()
    sma20  = close.rolling(20).mean()
    ema10  = close.ewm(span=10, adjust=False).mean()
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    sma200 = close.rolling(200).mean()

    p10  = _s(sma10.iloc[-1])
    p20  = _s(sma20.iloc[-1])
    e10  = _s(ema10.iloc[-1])
    e20  = _s(ema20.iloc[-1])
    e50  = _s(ema50.iloc[-1])
    s200 = _s(sma200.iloc[-1])

    price_vs_sma10  = (price - p10)  / p10  if (price and p10)  else None
    price_vs_sma20  = (price - p20)  / p20  if (price and p20)  else None
    price_vs_ema10  = (price - e10)  / e10  if (price and e10)  else None
    price_vs_ema20  = (price - e20)  / e20  if (price and e20)  else None
    price_vs_ema50  = (price - e50)  / e50  if (price and e50)  else None
    ema50_vs_sma200 = (e50 - s200)   / s200 if (e50 and s200)   else None

    # --- Extended momentum / ROC ---
    mom5   = _s(close.iloc[-1] / close.iloc[-5]   - 1) if len(close) >= 5   else None
    mom180 = _s(close.iloc[-1] / close.iloc[-180] - 1) if len(close) >= 180 else None
    roc5   = mom5    # ROC and momentum are the same formula; kept separately for naming
    roc10  = _s(close.iloc[-1] / close.iloc[-10]  - 1) if len(close) >= 10  else None
    roc20  = _s(close.iloc[-1] / close.iloc[-20]  - 1) if len(close) >= 20  else None

    # --- 52-week high/low proximity ---
    window_252 = min(252, len(close))
    high_52w = _s(high.iloc[-window_252:].max())
    low_52w  = _s(low.iloc[-window_252:].min())
    high_52w_pct = (price - high_52w) / high_52w if (price and high_52w) else None
    low_52w_pct  = (price - low_52w)  / low_52w  if (price and low_52w)  else None

    # --- Short-term returns ---
    daily_return  = _s(close.pct_change().iloc[-1])
    weekly_return = _s(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else None

    # --- CCI (14-day) ---
    tp  = (high + low + close) / 3          # Typical Price
    tp_sma14  = tp.rolling(14).mean()
    mean_dev  = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_series = (tp - tp_sma14) / (0.015 * mean_dev.replace(0, np.nan))
    cci_14 = _s(cci_series.iloc[-1])

    # --- MFI (14-day) ---
    # Typical Price * Volume = Raw Money Flow
    # Separate into positive (up days) and negative (down days)
    tp_change  = tp.diff()
    raw_mf     = tp * volume.astype(float)
    pos_mf     = raw_mf.where(tp_change > 0, 0.0)
    neg_mf     = raw_mf.where(tp_change < 0, 0.0)
    pos_sum    = pos_mf.rolling(14).sum()
    neg_sum    = neg_mf.rolling(14).sum().replace(0, np.nan)
    mfr        = pos_sum / neg_sum          # Money Flow Ratio
    mfi_series = 100 - 100 / (1 + mfr)
    mfi_14     = _s(mfi_series.iloc[-1])

    # --- ATR as % of price ---
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    atr_pct = _s(atr / price) if (price and price != 0) else None

    return {
        "price_vs_sma10":  price_vs_sma10,
        "price_vs_sma20":  price_vs_sma20,
        "price_vs_ema10":  price_vs_ema10,
        "price_vs_ema20":  price_vs_ema20,
        "price_vs_ema50":  price_vs_ema50,
        "ema50_vs_sma200": ema50_vs_sma200,
        "momentum_5d":     mom5,
        "momentum_180d":   mom180,
        "roc_5d":          roc5,
        "roc_10d":         roc10,
        "roc_20d":         roc20,
        "high_52w_pct":    high_52w_pct,
        "low_52w_pct":     low_52w_pct,
        "daily_return":    daily_return,
        "weekly_return":   weekly_return,
        "cci_14":          cci_14,
        "mfi_14":          mfi_14,
        "atr_pct":         atr_pct,
    }


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store(stock_id: int, as_of_date: str, features: dict):
    """Upsert technical features for one stock on one date."""
    cols         = ", ".join(features.keys())
    placeholders = ", ".join(["?"] * len(features))
    values       = list(features.values())

    with get_db() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO technical_features
                (stock_id, date, {cols})
                VALUES (?, ?, {placeholders})""",
            [stock_id, as_of_date] + values,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(stock_id: int, ticker: str, sector_etf: str, as_of_date: str = None):
    """
    Compute and store all technical features for one stock.
    Called by data/pipeline.py for each stock in the universe.

    sector_etf: the ETF ticker for this stock's sector (e.g., "XLK" for Technology).
                Used to compute relative strength vs. sector.
    as_of_date: the date through which to compute features. Defaults to today.
    """
    as_of_date = as_of_date or date.today().isoformat()

    print(f"  {ticker}: computing technical features...")

    df = _load_prices(stock_id, as_of_date)
    if df.empty or len(df) < 20:
        print(f"  {ticker}: insufficient price history — need at least 20 days")
        return

    close  = df["adj_close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    features = {}
    features.update(_moving_averages(close))
    features.update(_momentum(close, high, low))
    features.update(_volatility(close, high, low))
    features.update(_volume_indicators(close, high, low, volume))

    # Load SPY and sector ETF for relative strength
    spy = _load_ticker_prices("SPY", as_of_date)

    sector_close = None
    if sector_etf and sector_etf != ticker:
        sector_close = _load_ticker_prices(sector_etf, as_of_date)
        if sector_close.empty:
            sector_close = None

    features.update(_relative_strength(close, spy, sector_close))
    features.update(_extended_technical(close, high, low, volume))

    store(stock_id, as_of_date, features)

    non_null = sum(1 for v in features.values() if v is not None)
    print(f"  {ticker}: {len(features)} features computed ({non_null} non-null)")
