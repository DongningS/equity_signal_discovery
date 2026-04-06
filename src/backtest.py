"""
src/backtest.py
───────────────
Simple long-only quintile backtest.

CRITICAL DISCLAIMER
───────────────────
This backtest is a simplified simulation for methodology demonstration only.
It does NOT model:
  - Transaction costs (configurable placeholder only)
  - Market impact or slippage
  - Borrowing costs for short positions
  - Liquidity constraints or position limits
  - Tax effects
  - Actual execution prices (assumes close-to-close)
  - Survivorship-free index membership

Results should be read as a directional sanity check on signal quality,
NOT as an estimate of achievable real-world performance.

Approach
--------
  Monthly rebalancing. Equal-weight positions in top-quantile stocks.
  Compared against SPY total return as benchmark.
  Excess return series used to compute Sharpe, max drawdown.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src import config as cfg
from src.metrics import max_drawdown, sharpe_ratio

log = logging.getLogger(__name__)


def run_backtest(
    scores: pd.Series,
    prices: pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
    top_pct: float = 0.20,
    rebalance_freq: str = "ME",
    transaction_cost_bps: float = 0.0,
) -> dict:
    """
    Run a simple equal-weighted long-only backtest on top-quantile stocks.

    Parameters
    ----------
    scores       : (Date, Ticker) MultiIndex Series of model ranking scores
                   (all OOS scores concatenated from walk-forward)
    prices       : (Date, Ticker) MultiIndex DataFrame with 'close' column
    benchmark    : DataFrame with 'spy_return' column (daily)
    top_pct      : Fraction of universe to hold (default top 20%)
    rebalance_freq : Pandas offset alias for rebalancing frequency
    transaction_cost_bps : One-way transaction cost in basis points

    Returns
    -------
    dict with:
        portfolio_returns : monthly return Series
        benchmark_returns : monthly SPY return Series
        turnover          : monthly average turnover
        cumulative        : cumulative return comparison DataFrame
        stats             : summary statistics dict
    """
    tc = transaction_cost_bps / 10_000

    # Extract close prices wide
    if isinstance(prices.index, pd.MultiIndex):
        close = prices["close"].unstack("Ticker")
    else:
        close = prices

    # Daily returns for each ticker
    daily_ret = close.pct_change()

    # Get rebalance dates — end of each month within the scores date range
    score_dates = pd.DatetimeIndex(scores.index.get_level_values("Date").unique()).sort_values()
    start_date, end_date = score_dates[0], score_dates[-1]

    rebal_dates = pd.date_range(
        start=start_date, end=end_date, freq=rebalance_freq
    ).normalize()
    rebal_dates = rebal_dates[rebal_dates.isin(close.index)]

    if len(rebal_dates) < 2:
        raise ValueError("Insufficient rebalance dates — check score date range and frequency")

    port_returns: dict[pd.Timestamp, float] = {}
    turnover_series: dict[pd.Timestamp, float] = {}
    prev_holdings: set[str] = set()

    for i in range(len(rebal_dates) - 1):
        rebal_date  = rebal_dates[i]
        next_rebal  = rebal_dates[i + 1]

        # Find the latest score date at or before rebalancing date
        available_score_dates = score_dates[score_dates <= rebal_date]
        if len(available_score_dates) == 0:
            continue
        score_date = available_score_dates[-1]

        # Get scores for this rebalance date
        try:
            date_scores = scores.xs(score_date, level="Date")
        except KeyError:
            continue

        # Select top-pct tickers by score
        valid_scores = date_scores.dropna()
        if len(valid_scores) < 5:
            continue
        n_hold = max(1, int(len(valid_scores) * top_pct))
        holdings = set(valid_scores.nlargest(n_hold).index.tolist())

        # Filter to tickers with price data
        holdings = holdings.intersection(set(close.columns))
        if not holdings:
            continue

        # Compute turnover: fraction of portfolio that changed
        if prev_holdings:
            unchanged = len(prev_holdings & holdings) / len(prev_holdings | holdings)
            turnover = 1 - unchanged
        else:
            turnover = 1.0
        turnover_series[rebal_date] = turnover

        # Holding period returns (next_rebal exclusive)
        hold_slice = daily_ret.loc[
            (daily_ret.index > rebal_date) & (daily_ret.index <= next_rebal),
            list(holdings)
        ]

        if hold_slice.empty:
            continue

        # Equal-weight portfolio return (compound over period)
        port_daily = hold_slice.mean(axis=1)

        # Apply one-way transaction cost at rebalancing
        if tc > 0 and turnover > 0:
            port_daily.iloc[0] -= tc * turnover

        # Compound over the holding period
        period_ret = (1 + port_daily).prod() - 1
        port_returns[next_rebal] = period_ret
        prev_holdings = holdings

    portfolio = pd.Series(port_returns, name="portfolio").sort_index()
    portfolio.index = pd.DatetimeIndex(portfolio.index)

    # Benchmark monthly returns
    if benchmark is not None and "spy_return" in benchmark.columns:
        spy_daily = benchmark["spy_return"].dropna()
        spy_monthly = spy_daily.resample("ME").apply(lambda r: (1 + r).prod() - 1)
        spy_monthly = spy_monthly.reindex(portfolio.index, method="nearest")
        spy_monthly.name = "spy"
    else:
        spy_monthly = pd.Series(np.nan, index=portfolio.index, name="spy")

    # Cumulative returns
    cumulative = pd.DataFrame({
        "portfolio": (1 + portfolio).cumprod(),
        "spy":       (1 + spy_monthly.fillna(0)).cumprod(),
    })

    # Excess returns over benchmark
    excess = portfolio - spy_monthly.reindex(portfolio.index).fillna(0)

    # Summary statistics
    stats = {
        "total_return_pct":       round((cumulative["portfolio"].iloc[-1] - 1) * 100, 2),
        "spy_total_return_pct":   round((cumulative["spy"].iloc[-1] - 1) * 100, 2),
        "annualised_return_pct":  round(
            ((cumulative["portfolio"].iloc[-1]) ** (12 / len(portfolio)) - 1) * 100, 2
        ),
        "monthly_sharpe":         round(sharpe_ratio(portfolio, annualise_factor=12), 3),
        "excess_sharpe":          round(sharpe_ratio(excess, annualise_factor=12), 3),
        "max_drawdown_pct":       round(max_drawdown(cumulative["portfolio"]) * 100, 2),
        "hit_rate_vs_spy":        round((portfolio > spy_monthly.reindex(portfolio.index)).mean(), 3),
        "avg_monthly_turnover":   round(pd.Series(turnover_series).mean(), 3),
        "n_months":               len(portfolio),
        "disclaimer":             (
            "PRE-COST GROSS SIMULATION ONLY. No transaction costs, slippage, "
            "or market impact modelled. Survivorship bias present. "
            "Do not interpret as evidence of a viable trading strategy."
        ),
    }

    log.warning(
        "⚠  BACKTEST DISCLAIMER: Results are pre-cost gross simulation only. "
        "See stats['disclaimer'] for full warning."
    )

    return {
        "portfolio_returns": portfolio,
        "benchmark_returns": spy_monthly,
        "excess_returns":    excess,
        "turnover":          pd.Series(turnover_series).sort_index(),
        "cumulative":        cumulative,
        "stats":             stats,
    }


def turnover_adjusted_spread(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float = 10.0,
) -> pd.Series:
    """
    Approximate post-cost returns by deducting turnover-weighted cost.
    Demonstrates the impact of transaction costs on signal viability.
    """
    tc = cost_bps / 10_000
    aligned_turnover = turnover.reindex(portfolio_returns.index).fillna(0)
    return portfolio_returns - tc * aligned_turnover
