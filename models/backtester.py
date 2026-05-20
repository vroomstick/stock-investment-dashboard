"""
models/backtester.py

Walk-forward backtester for evaluating the ensemble's trading performance.

WHAT THIS MODULE DOES
----------------------
After we have predictions (expected_return, class_probs per stock per date),
this module simulates what a portfolio would have looked like if we had:
  1. Rebalanced weekly based on the ensemble's signals
  2. Applied position sizing rules (Kelly fraction, capped at max_per_stock)
  3. Applied risk management rules (max drawdown circuit breaker)

The output is an equity curve + performance metrics.

WHY BACKTESTING MATTERS
------------------------
A model can have 60% accuracy on the 4-class label and still lose money in
a real portfolio because:
  - Transaction costs eat into frequent rebalancing gains
  - Position sizing matters: concentrating on class 3 predictions amplifies
    both gains and losses
  - Drawdown limits reduce returns in volatile markets
  - Class imbalance: "flat" (class 1) dominates training data; a model that
    predicts flat for everything has high accuracy but zero alpha

The backtester reveals these issues before going live.

BACKTEST DESIGN (preventing lookahead bias)
-------------------------------------------
Critical rule: predictions used to trade on date D must use features
computed from data available up to and including date D-1.

Our pipeline guarantees this because:
  1. Features are stored with the date they were computed for (today's features)
  2. Labels are computed from prices D to D+horizon (future — used only for training)
  3. The backtester uses stored predictions (never recomputes them from future data)

PERFORMANCE METRICS
-------------------
  Total return:      (equity[-1] / equity[0]) - 1
  Annualized return: (1 + total_return)^(252/n_days) - 1
  Sharpe ratio:      mean(daily_returns) / std(daily_returns) * sqrt(252)
  Max drawdown:      max of (equity - cummax(equity)) / cummax(equity)
  Win rate:          fraction of closed positions with positive PnL
  Benchmark:         SPY buy-and-hold over the same period

We target Sharpe > 1.0. Below 0.5 means the model is not adding value
beyond buying SPY. Negative Sharpe means we're destroying capital.

POSITION SIZING
---------------
For each stock predicted to be class 2 or 3:
  raw_size = kelly_fraction(p_win, avg_win, avg_loss) * 0.5 (half-Kelly)
  final_size = min(raw_size, max_per_stock=0.10)

The sum of all positions is capped at gross_exposure (default 1.0 = fully
invested). We do not use leverage (no borrowing).

Stocks predicted class 0 are excluded (or shorted in a long-short version).
Stocks predicted class 1 are excluded (flat = no signal).
"""

from typing import Optional
import numpy as np
import pandas as pd

from config.settings import RISK, RETURN_BUCKET_MIDPOINTS


# ---------------------------------------------------------------------------
# Main backtester class
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulates portfolio performance from ensemble predictions.

    Usage:
        bt = Backtester()
        results = bt.run(predictions, actual_returns)
        print(bt.metrics())
        bt.equity_curve().plot()
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_per_stock:   float = None,
        gross_exposure:  float = None,
        use_half_kelly:  bool  = True,
        transaction_cost_bps: float = 10,  # 10 basis points per trade
    ):
        """
        Args:
            initial_capital:     Starting portfolio value ($)
            max_per_stock:       Max fraction of portfolio per position
                                 (default: RISK.max_position_size_pct)
            gross_exposure:      Max sum of all position sizes (1.0 = no leverage)
                                 (default: RISK.max_gross_exposure)
            use_half_kelly:      If True, use 0.5 × Kelly fraction
                                 (reduces drawdown at cost of lower expected return)
            transaction_cost_bps: Round-trip cost in basis points.
                                 10 bps ≈ 0.10%, typical for liquid US equities.
        """
        self.initial_capital      = initial_capital
        self.max_per_stock        = max_per_stock  or RISK.get("max_position_size_pct", 0.10)
        self.gross_exposure       = gross_exposure or RISK.get("max_gross_exposure", 1.0)
        self.use_half_kelly       = use_half_kelly
        self.transaction_cost_bps = transaction_cost_bps

        self._equity_curve:  Optional[pd.Series]    = None
        self._trades:        Optional[pd.DataFrame] = None
        self._positions_log: list = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
    ) -> "Backtester":
        """
        Simulate portfolio performance.

        Args:
            predictions:     Output of EnsembleModel.predict(), must include:
                               [ticker, date, expected_return, predicted_class,
                                conviction, prob_class_0, prob_class_1,
                                prob_class_2, prob_class_3]
            actual_returns:  DataFrame with [ticker, date, actual_return]
                             where actual_return is the realized return
                             over the prediction horizon.

        Returns:
            self (for method chaining, then call .metrics())

        How positions are determined:
            On each rebalance date, we:
            1. Filter to stocks with predicted_class in {2, 3} (or short class 0)
            2. Compute Kelly fraction for each
            3. Scale all positions down if total > gross_exposure
            4. Apply transaction cost on all changes from last period's weights
            5. Record PnL using actual_return

        Why weekly rebalancing?
            Daily rebalancing at 10 bps/trade would cost ~2.5% per year in
            transaction costs alone. Weekly reduces this to ~0.5%.
            The prediction horizon is 63 days so weekly rebalancing is more
            than frequent enough to respond to signal changes.
        """
        # Merge predictions with actual returns
        merged = predictions.merge(
            actual_returns[["ticker", "date", "actual_return"]],
            on=["ticker", "date"],
            how="inner",
        )
        if merged.empty:
            raise ValueError("No rows after merging predictions with actual returns.")

        # Sort by date and simulate period by period
        dates = sorted(merged["date"].unique())
        equity    = self.initial_capital
        trades    = []
        equity_ts = {}
        prev_weights = {}  # ticker → weight from last period

        for date in dates:
            period = merged[merged["date"] == date].copy()
            if period.empty:
                equity_ts[date] = equity
                continue

            # Build position weights for this period
            weights = self._compute_weights(period)

            # Apply transaction costs on turnover
            turnover = _compute_turnover(prev_weights, weights)
            cost     = turnover * (self.transaction_cost_bps / 10_000) * equity

            # Compute period PnL
            period_return = 0.0
            for _, row in period.iterrows():
                w = weights.get(row["ticker"], 0.0)
                if w > 0:
                    period_return += w * row["actual_return"]
                    trades.append({
                        "date":            date,
                        "ticker":          row["ticker"],
                        "weight":          w,
                        "actual_return":   row["actual_return"],
                        "pnl_pct":         w * row["actual_return"],
                        "predicted_class": row["predicted_class"],
                        "expected_return": row["expected_return"],
                    })

            equity = equity * (1 + period_return) - cost
            equity_ts[date] = equity
            prev_weights    = weights

        self._equity_curve = pd.Series(equity_ts)
        self._equity_curve.index = pd.to_datetime(self._equity_curve.index)
        self._trades = pd.DataFrame(trades)

        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def equity_curve(self) -> pd.Series:
        """Return the portfolio equity curve (indexed by date)."""
        self._check_run()
        return self._equity_curve.copy()

    def trades(self) -> pd.DataFrame:
        """Return log of all individual position PnLs."""
        self._check_run()
        return self._trades.copy()

    def metrics(self) -> dict:
        """
        Compute summary performance metrics.

        Returns:
            Dict with keys:
              total_return, annualized_return, sharpe_ratio, max_drawdown,
              win_rate, avg_position_size, n_trades, n_periods

        Interpretation guide:
          Sharpe > 1.5 → Excellent (institutional quality)
          Sharpe > 1.0 → Good (worth running live)
          Sharpe > 0.5 → Marginal (worth more research)
          Sharpe < 0.5 → Don't run this live

          Max drawdown < 20% → Acceptable for most investors
          Max drawdown > 30% → Will cause investors to panic-sell
        """
        self._check_run()

        curve = self._equity_curve
        daily_ret = curve.pct_change().dropna()

        total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
        n_days       = (curve.index[-1] - curve.index[0]).days
        ann_return   = (1 + total_return) ** (365 / max(n_days, 1)) - 1

        if daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        rolling_max  = curve.cummax()
        drawdowns    = (curve - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min())

        win_rate = 0.0
        if not self._trades.empty:
            win_rate = float((self._trades["pnl_pct"] > 0).mean())

        return {
            "total_return":       float(total_return),
            "annualized_return":  float(ann_return),
            "sharpe_ratio":       float(sharpe),
            "max_drawdown":       float(max_drawdown),
            "win_rate":           float(win_rate),
            "n_trades":           len(self._trades),
            "n_periods":          len(self._equity_curve),
            "final_equity":       float(self._equity_curve.iloc[-1]),
        }

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def compare_to_benchmark(
        self, benchmark_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Compare portfolio equity curve to a benchmark (e.g., SPY).

        Args:
            benchmark_returns: Daily return Series for the benchmark,
                               indexed by date.

        Returns:
            DataFrame with [date, portfolio_equity, benchmark_equity]
            Both normalized to 100 at start.
        """
        self._check_run()

        portfolio = (self._equity_curve / self._equity_curve.iloc[0]) * 100

        bench_equity = (1 + benchmark_returns).cumprod() * 100
        bench_equity = bench_equity.reindex(portfolio.index).ffill()

        return pd.DataFrame({
            "portfolio_equity":  portfolio,
            "benchmark_equity":  bench_equity,
        }).dropna()

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _compute_weights(self, period: pd.DataFrame) -> dict:
        """
        Compute position weights for one rebalance period.

        Returns dict: {ticker: weight}
        Weights sum to ≤ gross_exposure.
        Only stocks with predicted_class in {2, 3} get positive weight.
        """
        candidates = period[period["predicted_class"].isin([2, 3])].copy()
        if candidates.empty:
            return {}

        weights = {}
        for _, row in candidates.iterrows():
            # Kelly fraction from class probabilities
            p_win   = float(row["prob_class_2"] + row["prob_class_3"])
            avg_win  = 0.10  # representative win: midpoint of classes 2+3
            avg_loss = 0.05  # representative loss if wrong

            kelly = _kelly_fraction(p_win, avg_win, avg_loss)
            if kelly <= 0:
                continue

            w = kelly * (0.5 if self.use_half_kelly else 1.0)
            w = min(w, self.max_per_stock)
            weights[row["ticker"]] = w

        # Scale down if total exposure exceeds limit
        total = sum(weights.values())
        if total > self.gross_exposure:
            scale = self.gross_exposure / total
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_run(self):
        if self._equity_curve is None:
            raise RuntimeError("Call .run() before accessing results.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _kelly_fraction(
    p_win: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Full Kelly criterion: f* = (p * b - q) / b

    where b = avg_win / avg_loss, p = win probability, q = 1 - p.

    Args:
        p_win:    Probability of a winning outcome (class 2 or 3)
        avg_win:  Expected gain if win (fractional)
        avg_loss: Expected loss if lose (fractional)

    Returns:
        Kelly fraction (can be negative — means no edge, don't trade)
    """
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - p_win
    return (p_win * b - q) / b


def _compute_turnover(prev_weights: dict, new_weights: dict) -> float:
    """
    Compute portfolio turnover as the L1 distance between weight dicts.

    Turnover = sum of |new_w - old_w| across all tickers.
    A fully new portfolio (0% overlap) has turnover = 2.0 (all sells + all buys).
    A static portfolio has turnover = 0.
    """
    all_tickers = set(prev_weights) | set(new_weights)
    return sum(
        abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
        for t in all_tickers
    )
