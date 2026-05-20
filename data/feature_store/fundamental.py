"""
data/feature_store/fundamental.py

Computes all fundamental features defined in spec Section 7A and stores
them in the fundamental_features table.

Source: yfinance (via price_data collector) + SEC XBRL (via sec_edgar collector)

Feature categories (per spec):
  - Profitability Ratios   (8 features)
  - Valuation Ratios       (7 features)
  - Growth Metrics         (6 features)
  - Quality & Strength     (7 features + 9 Piotroski components)
  - Efficiency Metrics     (4 features)
  Total: ~41 base features, extended toward 80-100 with derived variants

All formulas match Appendix B of the spec exactly.
Missing value strategy (per feature_config.yaml): forward-fill then sector-median.
Normalization: z-score within sector — happens at training time, not here.
"""

import os
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from data.collectors.price_data import get_fundamentals
from database.db import get_db, fetch_all, fetch_one


# ---------------------------------------------------------------------------
# Safe arithmetic helpers
# Fundamental data is full of None, 0, and NaN. Every division needs a guard.
# ---------------------------------------------------------------------------

def _safe_div(numerator, denominator, fallback=None):
    """Divide two values, returning fallback on zero/None/NaN."""
    try:
        if denominator is None or denominator == 0 or pd.isna(denominator):
            return fallback
        if numerator is None or pd.isna(numerator):
            return fallback
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return fallback


def _row(df: pd.DataFrame, label: str) -> Optional[pd.Series]:
    """
    Safely retrieve a row from a yfinance financial DataFrame.
    yfinance DataFrames have metric names as the index. Columns are dates
    (most recent first). Returns the Series or None if not found.
    """
    if df is None or df.empty:
        return None
    if label in df.index:
        return df.loc[label]
    return None


def _val(series: Optional[pd.Series], pos: int = 0) -> Optional[float]:
    """Get the value at position `pos` from a Series, or None."""
    if series is None or len(series) <= pos:
        return None
    v = series.iloc[pos]
    return float(v) if pd.notna(v) else None


def _filter_statements(df: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Filter a yfinance financial statement DataFrame to only columns whose
    quarter-end date is on or before as_of_ts.

    yfinance DataFrames have pd.Timestamp columns (most recent first).
    Filtering prevents using financial data not yet published on as_of_date,
    which would introduce lookahead bias in historical backfills.
    """
    if df is None or df.empty:
        return df
    cols = [c for c in df.columns if pd.Timestamp(c) <= as_of_ts]
    return df[cols] if cols else pd.DataFrame()


def _get_historical_price(stock_id: int, as_of_date: str) -> Optional[float]:
    """
    Fetch the most recent adj_close on or before as_of_date from daily_prices.

    Used to compute price-based ratios (P/E, P/B, P/S) without lookahead bias.
    yfinance's info dict always returns today's price; for historical backfills
    we replace it with the actual price on (or just before) the target date.
    Returns None if no price data exists for this stock yet.
    """
    row = fetch_one(
        """SELECT adj_close FROM daily_prices
           WHERE stock_id = ? AND date <= ?
           ORDER BY date DESC LIMIT 1""",
        (stock_id, as_of_date),
    )
    return float(row["adj_close"]) if row and row["adj_close"] else None


# ---------------------------------------------------------------------------
# Profitability Ratios — Spec Section 7A
# Formula source: Appendix B + spec table
# ---------------------------------------------------------------------------

def _profitability(info: dict, fin: pd.DataFrame, bal: pd.DataFrame,
                   cf: pd.DataFrame) -> dict:
    """
    Spec features:
      roe, roa, roic, gross_margin, operating_margin, net_margin,
      gross_margin_trend, operating_margin_trend
    """
    rev   = _row(fin, "Total Revenue")
    gp    = _row(fin, "Gross Profit")
    op_in = _row(fin, "Operating Income")
    ni    = _row(fin, "Net Income")
    ta    = _row(bal, "Total Assets")
    eq    = _row(bal, "Stockholders Equity")
    cl    = _row(bal, "Current Liabilities")
    cash  = _row(bal, "Cash And Cash Equivalents")
    op_cf = _row(cf,  "Operating Cash Flow")

    # ROE and ROA from yfinance info (pre-calculated, more accurate)
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")

    # ROIC = NOPAT / Invested Capital
    # NOPAT = Operating Income * (1 - effective tax rate)
    # Invested Capital = Total Assets - Current Liabilities - Cash
    tax_rate = info.get("effectiveTaxRate") or 0.21  # fallback to US corporate rate
    nopat = _val(op_in) * (1 - tax_rate) if _val(op_in) is not None else None
    invested_cap = None
    if _val(ta) and _val(cl) is not None and _val(cash) is not None:
        invested_cap = _val(ta) - _val(cl) - _val(cash)
    roic = _safe_div(nopat, invested_cap)

    # Margins — Appendix B formulas
    gross_margin      = _safe_div(_val(gp), _val(rev))
    gross_margin_prev = _safe_div(_val(gp, 1), _val(rev, 1))
    gross_margin_trend = (
        gross_margin - gross_margin_prev
        if gross_margin is not None and gross_margin_prev is not None
        else None
    )

    operating_margin      = _safe_div(_val(op_in), _val(rev))
    operating_margin_prev = _safe_div(_val(op_in, 1), _val(rev, 1))
    operating_margin_trend = (
        operating_margin - operating_margin_prev
        if operating_margin is not None and operating_margin_prev is not None
        else None
    )

    net_margin = _safe_div(_val(ni), _val(rev))

    return {
        "roe":                    roe,
        "roa":                    roa,
        "roic":                   roic,
        "gross_margin":           gross_margin,
        "operating_margin":       operating_margin,
        "net_margin":             net_margin,
        "gross_margin_trend":     gross_margin_trend,
        "operating_margin_trend": operating_margin_trend,
    }


# ---------------------------------------------------------------------------
# Valuation Ratios — Spec Section 7A
# ---------------------------------------------------------------------------

def _valuation(info: dict, sector_pe_avg: Optional[float]) -> dict:
    """
    Spec features:
      pe_ratio, forward_pe, pb_ratio, ps_ratio, ev_ebitda, peg_ratio,
      pe_vs_sector_avg

    Most valuation ratios come directly from yfinance info — they're
    already computed by Yahoo using market price + financial data.
    pe_vs_sector_avg requires knowing the sector average, passed in
    from the pipeline after all stocks in the sector have been processed.
    """
    pe = info.get("trailingPE")

    return {
        "pe_ratio":        pe,
        "forward_pe":      info.get("forwardPE"),
        "pb_ratio":        info.get("priceToBook"),
        "ps_ratio":        info.get("priceToSalesTrailing12Months"),
        "ev_ebitda":       info.get("enterpriseToEbitda"),
        "peg_ratio":       info.get("pegRatio"),
        "pe_vs_sector_avg": _safe_div(pe, sector_pe_avg),
    }


# ---------------------------------------------------------------------------
# Growth Metrics — Spec Section 7A + Appendix B
# ---------------------------------------------------------------------------

def _growth(fin: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame) -> dict:
    """
    Spec features:
      revenue_growth_yoy, revenue_growth_qoq, earnings_growth_yoy,
      revenue_acceleration, book_value_growth, fcf_growth

    Appendix B formulas:
      revenue_growth_yoy = (rev[0] - rev[4]) / abs(rev[4])   # Q0 vs Q-4 (1 year ago)
      revenue_growth_qoq = (rev[0] - rev[1]) / abs(rev[1])   # Q0 vs Q-1
      earnings_growth_yoy = (ni[0] - ni[4]) / abs(ni[4])
      revenue_acceleration = current_qoq_growth - prior_qoq_growth
    """
    rev  = _row(fin, "Total Revenue")
    ni   = _row(fin, "Net Income")
    eq   = _row(bal, "Stockholders Equity")
    fcf  = _row(cf,  "Free Cash Flow")

    # Revenue growth — Appendix B
    revenue_growth_yoy = None
    if rev is not None and len(rev) > 4:
        revenue_growth_yoy = _safe_div(_val(rev, 0) - _val(rev, 4), abs(_val(rev, 4)))

    revenue_growth_qoq = None
    if rev is not None and len(rev) > 1:
        revenue_growth_qoq = _safe_div(_val(rev, 0) - _val(rev, 1), abs(_val(rev, 1)))

    # Revenue acceleration = current QoQ - prior QoQ
    revenue_acceleration = None
    if rev is not None and len(rev) > 2:
        prior_qoq = _safe_div(_val(rev, 1) - _val(rev, 2), abs(_val(rev, 2)))
        if revenue_growth_qoq is not None and prior_qoq is not None:
            revenue_acceleration = revenue_growth_qoq - prior_qoq

    # Earnings growth YoY — Appendix B
    earnings_growth_yoy = None
    if ni is not None and len(ni) > 4:
        earnings_growth_yoy = _safe_div(_val(ni, 0) - _val(ni, 4), abs(_val(ni, 4)))

    # Book value growth YoY (equity per share proxy)
    book_value_growth = None
    if eq is not None and len(eq) > 4:
        book_value_growth = _safe_div(_val(eq, 0) - _val(eq, 4), abs(_val(eq, 4)))

    # FCF growth YoY
    fcf_growth = None
    if fcf is not None and len(fcf) > 4:
        fcf_growth = _safe_div(_val(fcf, 0) - _val(fcf, 4), abs(_val(fcf, 4)))

    return {
        "revenue_growth_yoy":   revenue_growth_yoy,
        "revenue_growth_qoq":   revenue_growth_qoq,
        "earnings_growth_yoy":  earnings_growth_yoy,
        "revenue_acceleration": revenue_acceleration,
        "book_value_growth":    book_value_growth,
        "fcf_growth":           fcf_growth,
    }


# ---------------------------------------------------------------------------
# Quality & Strength — Spec Section 7A
# Includes Piotroski F-Score (9 binary components) and Altman Z-Score
# ---------------------------------------------------------------------------

def _quality(info: dict, fin: pd.DataFrame, bal: pd.DataFrame,
             cf: pd.DataFrame) -> dict:
    """
    Spec features:
      debt_to_equity, current_ratio, interest_coverage, piotroski_f_score,
      altman_z_score, accruals_ratio, cash_vs_earnings_quality
      + 9 Piotroski binary components
    """
    rev     = _row(fin, "Total Revenue")
    ni      = _row(fin, "Net Income")
    ebit    = _row(fin, "EBIT")
    int_exp = _row(fin, "Interest Expense")
    gp      = _row(fin, "Gross Profit")
    ta      = _row(bal, "Total Assets")
    ca      = _row(bal, "Current Assets")
    cl      = _row(bal, "Current Liabilities")
    eq      = _row(bal, "Stockholders Equity")
    ltd     = _row(bal, "Long Term Debt")
    op_cf   = _row(cf,  "Operating Cash Flow")
    shares  = _row(bal, "Share Issued")

    ta0  = _val(ta, 0)
    ta1  = _val(ta, 1)
    ni0  = _val(ni, 0)
    ni1  = _val(ni, 1)  # prior quarter
    cf0  = _val(op_cf, 0)
    ca0  = _val(ca, 0)
    cl0  = _val(cl, 0)
    eq0  = _val(eq, 0)

    # --- Core quality ratios ---
    total_debt  = info.get("totalDebt") or 0
    debt_to_equity  = _safe_div(total_debt, eq0)
    current_ratio   = _safe_div(ca0, cl0)
    interest_coverage = _safe_div(_val(ebit, 0), abs(_val(int_exp, 0))) if _val(int_exp, 0) else None
    accruals_ratio  = _safe_div(
        (ni0 - cf0) if ni0 is not None and cf0 is not None else None, ta0
    )
    cash_vs_earnings_quality = _safe_div(cf0, ni0)

    # --- Piotroski F-Score (spec: 9 binary signals, 0 or 1 each) ---
    # 1. Positive net income
    p1 = int(ni0 > 0) if ni0 is not None else 0
    # 2. Positive operating cash flow
    p2 = int(cf0 > 0) if cf0 is not None else 0
    # 3. ROA increasing (compare current quarter ROA to prior quarter ROA)
    roa0 = _safe_div(ni0, ta0)
    roa1 = _safe_div(ni1, ta1)
    p3 = int(roa0 > roa1) if roa0 is not None and roa1 is not None else 0
    # 4. Cash flow > net income (quality of earnings — cash is harder to manipulate)
    p4 = int(cf0 > ni0) if cf0 is not None and ni0 is not None else 0
    # 5. Long-term debt ratio decreasing
    ltd0 = _val(ltd, 0)
    ltd1 = _val(ltd, 1)
    ltd_ratio0 = _safe_div(ltd0, ta0)
    ltd_ratio1 = _safe_div(ltd1, ta1)
    p5 = int(ltd_ratio0 < ltd_ratio1) if ltd_ratio0 is not None and ltd_ratio1 is not None else 0
    # 6. Current ratio increasing
    cr0 = _safe_div(_val(ca, 0), _val(cl, 0))
    cr1 = _safe_div(_val(ca, 1), _val(cl, 1))
    p6 = int(cr0 > cr1) if cr0 is not None and cr1 is not None else 0
    # 7. No new shares issued (dilution is bad)
    sh0 = _val(shares, 0)
    sh1 = _val(shares, 1)
    p7 = int(sh0 <= sh1) if sh0 is not None and sh1 is not None else 0
    # 8. Gross margin increasing
    gm0 = _safe_div(_val(gp, 0), _val(rev, 0))
    gm1 = _safe_div(_val(gp, 1), _val(rev, 1))
    p8 = int(gm0 > gm1) if gm0 is not None and gm1 is not None else 0
    # 9. Asset turnover increasing (revenue / total assets, efficiency improving)
    at0 = _safe_div(_val(rev, 0), ta0)
    at1 = _safe_div(_val(rev, 1), ta1)
    p9 = int(at0 > at1) if at0 is not None and at1 is not None else 0

    piotroski = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

    # --- Altman Z-Score (bankruptcy prediction) ---
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings / Total Assets
    # X3 = EBIT / Total Assets
    # X4 = Market Cap / Total Liabilities
    # X5 = Revenue / Total Assets
    altman_z = None
    try:
        wc   = (ca0 or 0) - (cl0 or 0)
        re   = _val(_row(bal, "Retained Earnings"), 0) or 0
        ebit0 = _val(ebit, 0) or 0
        mktcap = info.get("marketCap") or 0
        tl   = info.get("totalDebt") or 0
        rev0 = _val(rev, 0) or 0

        if ta0 and ta0 != 0:
            x1 = wc   / ta0
            x2 = re   / ta0
            x3 = ebit0 / ta0
            x4 = _safe_div(mktcap, tl) or 0
            x5 = rev0 / ta0
            altman_z = round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5, 4)
    except Exception:
        altman_z = None

    return {
        "debt_to_equity":          debt_to_equity,
        "current_ratio":           current_ratio,
        "interest_coverage":       interest_coverage,
        "piotroski_f_score":       piotroski,
        "altman_z_score":          altman_z,
        "accruals_ratio":          accruals_ratio,
        "cash_vs_earnings_quality": cash_vs_earnings_quality,
        # Piotroski components (binary)
        "p_positive_net_income":       p1,
        "p_positive_operating_cf":     p2,
        "p_roa_increasing":            p3,
        "p_cf_greater_than_ni":        p4,
        "p_debt_ratio_decreasing":     p5,
        "p_current_ratio_increasing":  p6,
        "p_no_new_shares":             p7,
        "p_gross_margin_increasing":   p8,
        "p_asset_turnover_increasing": p9,
    }


# ---------------------------------------------------------------------------
# Efficiency Metrics — Spec Section 7A
# ---------------------------------------------------------------------------

def _efficiency(fin: pd.DataFrame, bal: pd.DataFrame) -> dict:
    """
    Spec features:
      asset_turnover, inventory_turnover, receivables_turnover,
      days_sales_outstanding
    """
    rev  = _row(fin, "Total Revenue")
    cogs = _row(fin, "Cost Of Revenue")
    ta   = _row(bal, "Total Assets")
    inv  = _row(bal, "Inventory")
    rec  = _row(bal, "Net Receivables")

    # Asset turnover = Revenue / Total Assets
    asset_turnover = _safe_div(_val(rev, 0), _val(ta, 0))

    # Inventory turnover = COGS / Average Inventory (Q0 and Q1)
    avg_inv = None
    if _val(inv, 0) is not None and _val(inv, 1) is not None:
        avg_inv = (_val(inv, 0) + _val(inv, 1)) / 2
    inventory_turnover = _safe_div(_val(cogs, 0), avg_inv)

    # Receivables turnover = Revenue / Average Receivables
    avg_rec = None
    if _val(rec, 0) is not None and _val(rec, 1) is not None:
        avg_rec = (_val(rec, 0) + _val(rec, 1)) / 2
    receivables_turnover = _safe_div(_val(rev, 0), avg_rec)

    # Days Sales Outstanding = 365 / Receivables Turnover
    days_sales_outstanding = _safe_div(365, receivables_turnover)

    return {
        "asset_turnover":         asset_turnover,
        "inventory_turnover":     inventory_turnover,
        "receivables_turnover":   receivables_turnover,
        "days_sales_outstanding": days_sales_outstanding,
    }


# ---------------------------------------------------------------------------
# Extended features — valuation, quality, profitability, sector-relative
# ---------------------------------------------------------------------------

def _extended(info: dict, fin: pd.DataFrame, bal: pd.DataFrame,
               cf: pd.DataFrame, sector: str) -> dict:
    """
    Additional features beyond the core spec tables, filling out the
    80-100 fundamental feature target.

    Sector-relative features compare this stock to its sector peers using
    the most recently stored values in the DB — same pattern as pe_vs_sector_avg.
    If fewer than 3 peers have data, sector-relative features are None.
    """
    rev     = _row(fin, "Total Revenue")
    gp      = _row(fin, "Gross Profit")
    ni      = _row(fin, "Net Income")
    op_in   = _row(fin, "Operating Income")
    ebit    = _row(fin, "EBIT")
    int_exp = _row(fin, "Interest Expense")
    ca      = _row(bal, "Current Assets")
    cl      = _row(bal, "Current Liabilities")
    inv     = _row(bal, "Inventory")
    cash    = _row(bal, "Cash And Cash Equivalents")
    eq      = _row(bal, "Stockholders Equity")
    ltd     = _row(bal, "Long Term Debt")
    op_cf   = _row(cf,  "Operating Cash Flow")
    fcf     = _row(cf,  "Free Cash Flow")
    capex   = _row(cf,  "Capital Expenditure")
    rd      = _row(fin, "Research And Development")

    rev0  = _val(rev,  0)
    gp0   = _val(gp,   0)
    ni0   = _val(ni,   0)
    ca0   = _val(ca,   0)
    cl0   = _val(cl,   0)
    inv0  = _val(inv,  0)
    cash0 = _val(cash, 0)
    eq0   = _val(eq,   0)
    cf0   = _val(op_cf, 0)
    fcf0  = _val(fcf,  0)

    total_debt = info.get("totalDebt") or 0
    mktcap     = info.get("marketCap")
    shares     = info.get("sharesOutstanding")
    price      = info.get("currentPrice") or info.get("regularMarketPrice")
    ebitda     = info.get("ebitda")

    # --- Extended valuation ---
    ev         = info.get("enterpriseValue")
    ev_revenue = _safe_div(ev, rev0)

    # TTM FCF from info if available, else from cf statement
    fcf_ttm = info.get("freeCashflow") or fcf0
    price_to_fcf = _safe_div(mktcap, fcf_ttm)
    earnings_yield = _safe_div(info.get("trailingEps"), price) if price else None
    fcf_per_share  = _safe_div(fcf_ttm, shares) if shares else None
    fcf_yield      = _safe_div(fcf_per_share, price) if price else None
    dividend_yield = info.get("dividendYield")

    # Tangible book = equity - intangible assets - goodwill
    goodwill      = _val(_row(bal, "Goodwill"), 0) or 0
    intangibles   = _val(_row(bal, "Other Intangible Assets"), 0) or 0
    tangible_book = (eq0 or 0) - goodwill - intangibles
    tangible_bv_per_share = _safe_div(tangible_book, shares) if shares else None
    price_to_tangible_book = _safe_div(price, tangible_bv_per_share)

    # --- Extended quality / liquidity ---
    # Quick ratio = (Current Assets - Inventory) / Current Liabilities
    quick_assets = (ca0 - inv0) if (ca0 is not None and inv0 is not None) else ca0
    quick_ratio  = _safe_div(quick_assets, cl0)
    cash_ratio   = _safe_div(cash0, cl0)

    net_debt     = (total_debt - (cash0 or 0)) if total_debt else None
    net_debt_ebitda = _safe_div(net_debt, ebitda) if ebitda else None
    fcf_to_debt  = _safe_div(fcf_ttm, total_debt) if total_debt else None

    # Interest coverage TTM — use EBIT from most recent quarter annualized
    ebit0 = _val(ebit, 0)
    int0  = _val(int_exp, 0)
    interest_coverage_ttm = _safe_div(ebit0 * 4, abs(int0) * 4) if (ebit0 and int0) else None

    # --- Extended profitability ---
    ebitda_margin = _safe_div(ebitda, rev0 * 4) if (ebitda and rev0) else None
    fcf_margin    = _safe_div(fcf_ttm, rev0 * 4) if (fcf_ttm and rev0) else None
    capex0        = _val(capex, 0)
    capex_ratio   = _safe_div(abs(capex0), rev0) if capex0 else None
    rd0           = _val(rd, 0)
    rd_intensity  = _safe_div(abs(rd0), rev0) if rd0 else None

    # Return on tangible equity
    return_on_tangible_equity = _safe_div(ni0, tangible_book) if tangible_book else None

    # --- TTM aggregates (sum of 4 quarters) ---
    def _ttm(series):
        if series is None or len(series) < 4:
            return None
        vals = [_val(series, i) for i in range(4)]
        if all(v is not None for v in vals):
            return sum(vals)
        return None

    revenue_ttm     = _ttm(rev)
    gross_profit_ttm = _ttm(gp)
    eps_ttm         = info.get("trailingEps")
    shares_outstanding = shares

    # --- Additional growth ---
    # Net income growth YoY (Q0 vs Q4)
    ni4 = _val(ni, 4)
    net_income_growth = _safe_div(ni0 - ni4, abs(ni4)) if (ni0 and ni4) else None

    # Operating CF growth YoY
    cf4 = _val(op_cf, 4)
    operating_cf_growth = _safe_div(cf0 - cf4, abs(cf4)) if (cf0 and cf4) else None

    # Buyback yield = share repurchases / market cap
    buybacks = _val(_row(cf, "Repurchase Of Capital Stock"), 0)
    buyback_yield = _safe_div(abs(buybacks), mktcap) if (buybacks and mktcap) else None

    # --- Sector-relative features ---
    _SECTOR_MEDIAN_COLS = frozenset({"roe", "roa", "net_margin", "revenue_growth_yoy"})

    def _sector_median(col: str) -> Optional[float]:
        if col not in _SECTOR_MEDIAN_COLS:
            raise ValueError(f"_sector_median: column '{col}' not in whitelist")
        rows = fetch_all(
            f"""SELECT f.{col}
               FROM fundamental_features f
               JOIN stocks s ON s.id = f.stock_id
               WHERE s.sector = ?
                 AND f.{col} IS NOT NULL
                 AND f.date = (
                   SELECT MAX(f2.date) FROM fundamental_features f2
                   WHERE f2.stock_id = f.stock_id
                 )""",
            (sector,)
        )
        vals = [r[col] for r in rows if r[col] is not None]
        return float(np.median(vals)) if len(vals) >= 3 else None

    roe_val = info.get("returnOnEquity")
    roa_val = info.get("returnOnAssets")
    nm_val  = _safe_div(ni0, rev0)
    rg_val  = _safe_div(_val(rev, 0) - _val(rev, 4), abs(_val(rev, 4))) if (rev is not None and len(rev) > 4) else None

    roe_med = _sector_median("roe")
    roa_med = _sector_median("roa")
    nm_med  = _sector_median("net_margin")
    rg_med  = _sector_median("revenue_growth_yoy")

    roe_vs_sector            = _safe_div(roe_val, roe_med) if roe_val else None
    roa_vs_sector            = _safe_div(roa_val, roa_med) if roa_val else None
    net_margin_vs_sector     = _safe_div(nm_val,  nm_med)  if nm_val  else None
    revenue_growth_vs_sector = _safe_div(rg_val,  rg_med)  if rg_val  else None

    return {
        # Extended valuation
        "ev_revenue":                ev_revenue,
        "price_to_fcf":              price_to_fcf,
        "earnings_yield":            earnings_yield,
        "fcf_yield":                 fcf_yield,
        "dividend_yield":            dividend_yield,
        "price_to_tangible_book":    price_to_tangible_book,
        # Extended quality
        "quick_ratio":               quick_ratio,
        "cash_ratio":                cash_ratio,
        "net_debt":                  net_debt,
        "net_debt_ebitda":           net_debt_ebitda,
        "fcf_to_debt":               fcf_to_debt,
        "interest_coverage_ttm":     interest_coverage_ttm,
        # Extended profitability
        "ebitda_margin":             ebitda_margin,
        "fcf_margin":                fcf_margin,
        "capex_ratio":               capex_ratio,
        "rd_intensity":              rd_intensity,
        "return_on_tangible_equity": return_on_tangible_equity,
        # TTM aggregates
        "revenue_ttm":               revenue_ttm,
        "gross_profit_ttm":          gross_profit_ttm,
        "eps_ttm":                   eps_ttm,
        "shares_outstanding":        shares_outstanding,
        # Sector-relative
        "roe_vs_sector":             roe_vs_sector,
        "roa_vs_sector":             roa_vs_sector,
        "net_margin_vs_sector":      net_margin_vs_sector,
        "revenue_growth_vs_sector":  revenue_growth_vs_sector,
        # Additional growth
        "net_income_growth":         net_income_growth,
        "operating_cf_growth":       operating_cf_growth,
        "buyback_yield":             buyback_yield,
    }


# ---------------------------------------------------------------------------
# Sector average PE — needed for pe_vs_sector_avg
# ---------------------------------------------------------------------------

def get_sector_avg_pe(sector: str) -> Optional[float]:
    """
    Compute the average trailing PE ratio for all stocks in the same sector
    using the most recently stored fundamental features.

    Called by the pipeline after all stocks have been processed so every
    stock can be compared to its peers. Falls back to None if insufficient data.
    """
    rows = fetch_all(
        """SELECT f.pe_ratio
           FROM fundamental_features f
           JOIN stocks s ON s.id = f.stock_id
           WHERE s.sector = ?
             AND f.pe_ratio IS NOT NULL
             AND f.pe_ratio > 0
             AND f.date = (
               SELECT MAX(f2.date) FROM fundamental_features f2
               WHERE f2.stock_id = f.stock_id
             )""",
        (sector,)
    )
    values = [r["pe_ratio"] for r in rows if r["pe_ratio"]]
    return float(np.mean(values)) if len(values) >= 3 else None


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store(stock_id: int, as_of_date: str, features: dict):
    """
    Upsert fundamental features for a stock on a given date.
    INSERT OR REPLACE handles the case where we re-run for the same quarter.
    """
    cols = ", ".join(features.keys())
    placeholders = ", ".join(["?"] * len(features))
    values = list(features.values())

    with get_db() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO fundamental_features
                (stock_id, date, {cols})
                VALUES (?, ?, {placeholders})""",
            [stock_id, as_of_date] + values,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(stock_id: int, ticker: str, sector: str, as_of_date: str = None):
    """
    Compute and store all fundamental features for one stock.
    Called by data/pipeline.py for each stock in the universe.

    as_of_date: the anchor date for point-in-time feature computation.
                Defaults to today (live mode). For historical backfills,
                pass the target date so no future data leaks in.

    Point-in-time design:
      - fin/bal/cf are filtered to quarters whose end date <= as_of_date,
        preventing use of financial statements not yet published.
      - Price-based ratios (P/E, P/B, P/S) are computed from the historical
        adj_close in daily_prices rather than yfinance's current-day price.
      - forward_pe, ev_ebitda, peg_ratio remain yfinance current-day values
        (analyst estimates — no historical equivalent available from yfinance).
    """
    as_of_date = as_of_date or date.today().isoformat()
    as_of_ts   = pd.Timestamp(as_of_date)

    print(f"  {ticker}: computing fundamental features...")

    raw  = get_fundamentals(ticker)
    info = raw["info"]
    # Filter to statements available on as_of_date — prevents lookahead bias
    fin  = _filter_statements(raw["financials"], as_of_ts)
    bal  = _filter_statements(raw["balance"],    as_of_ts)
    cf   = _filter_statements(raw["cashflow"],   as_of_ts)

    if not info and (fin is None or fin.empty):
        print(f"  {ticker}: no fundamental data available — skipping")
        return

    # Get sector average PE for pe_vs_sector_avg
    sector_pe_avg = get_sector_avg_pe(sector)

    features = {}
    features.update(_profitability(info, fin, bal, cf))
    features.update(_valuation(info, sector_pe_avg))
    features.update(_growth(fin, bal, cf))
    features.update(_quality(info, fin, bal, cf))
    features.update(_efficiency(fin, bal))
    features.update(_extended(info, fin, bal, cf, sector))

    # Override price-based ratios with historical price from daily_prices.
    # yfinance info always returns today's price; for a backfill run on
    # 2024-01-15, this would embed 2026 prices into 2024 training rows.
    hist_price = _get_historical_price(stock_id, as_of_date)
    if hist_price is not None:
        shares = info.get("sharesOutstanding")
        mktcap_hist = hist_price * shares if shares else None

        ni  = _row(fin, "Net Income")
        rev = _row(fin, "Total Revenue")
        eq  = _row(bal, "Stockholders Equity")
        fcf_s = _row(cf, "Free Cash Flow")

        def _ttm4(series):
            if series is None or len(series) < 4:
                return None
            vals = [_val(series, i) for i in range(4)]
            return sum(vals) if all(v is not None for v in vals) else None

        eps_ttm = _safe_div(_ttm4(ni), shares)
        rev_ttm = _ttm4(rev)
        bvps    = _safe_div(_val(eq, 0), shares) if shares else None
        fcf_ttm = _ttm4(fcf_s) or info.get("freeCashflow")

        goodwill    = _val(_row(bal, "Goodwill"), 0) or 0
        intangibles = _val(_row(bal, "Other Intangible Assets"), 0) or 0
        tang_book   = (_val(eq, 0) or 0) - goodwill - intangibles
        tang_bvps   = _safe_div(tang_book, shares) if shares else None

        pe_hist  = _safe_div(hist_price, eps_ttm) if eps_ttm and eps_ttm > 0 else None
        pb_hist  = _safe_div(hist_price, bvps)     if bvps    and bvps    > 0 else None
        ps_hist  = _safe_div(mktcap_hist, rev_ttm) if rev_ttm              else None
        ey_hist  = _safe_div(eps_ttm, hist_price)
        fcf_ps   = _safe_div(fcf_ttm, shares) if shares else None
        fy_hist  = _safe_div(fcf_ps, hist_price)
        ptf_hist = _safe_div(mktcap_hist, fcf_ttm) if fcf_ttm else None
        ptb_hist = _safe_div(hist_price, tang_bvps)

        features.update({
            "pe_ratio":               pe_hist,
            "pb_ratio":               pb_hist,
            "ps_ratio":               ps_hist,
            "pe_vs_sector_avg":       _safe_div(pe_hist, sector_pe_avg),
            "earnings_yield":         ey_hist,
            "fcf_yield":              fy_hist,
            "price_to_fcf":           ptf_hist,
            "price_to_tangible_book": ptb_hist,
        })

    store(stock_id, as_of_date, features)

    non_null = sum(1 for v in features.values() if v is not None)
    print(f"  {ticker}: {len(features)} features computed ({non_null} non-null)")
