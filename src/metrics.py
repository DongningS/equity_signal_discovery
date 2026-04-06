"""
src/metrics.py
──────────────
Ranking quality metrics for signal evaluation.

These metrics are the language of quantitative research:
  IC     — Information Coefficient (Spearman rank correlation)
  ICIR   — IC Information Ratio (mean IC / std IC) — stability measure
  Top-K  — spread between top and bottom quintile returns
  Hit    — fraction of periods where top-k beats bottom-k

Why not accuracy?
  Classification accuracy rewards the majority class. In a balanced
  quintile setup that's meaningless. IC measures the degree to which
  our score ordering tracks realized returns — exactly the right question.

A realistic target for public-data, no-transaction-cost signals:
  IC     : 0.02–0.06 per period
  ICIR   : 0.3–0.8 (stability)
  Top-K  : small positive spread, hit rate > 52%

Anything materially higher on public data should be scrutinised for
look-ahead leakage or data-mining bias.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Signal Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def information_coefficient(
    scores: pd.Series,
    realized_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Compute IC between predicted scores and realized forward returns.

    Parameters
    ----------
    scores            : Model-predicted ranking scores (any scale)
    realized_returns  : Actual forward returns for the same (Date, Ticker) index
    method            : 'spearman' (default, robust) or 'pearson' (linear)

    Returns
    -------
    IC value ∈ [-1, 1]. NaN if fewer than 5 observations.

    Notes
    -----
    Spearman IC is preferred because it is robust to outlier returns and
    does not assume linear relationship between score and return.
    """
    mask = scores.notna() & realized_returns.notna()
    s = scores[mask]
    r = realized_returns[mask]

    if len(s) < 5:
        return np.nan

    if method == "spearman":
        corr, _ = spearmanr(s, r)
    elif method == "pearson":
        corr, _ = pearsonr(s, r)
    else:
        raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")

    return float(corr)


def rolling_ic(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    date_level: str = "Date",
    method: str = "spearman",
) -> pd.Series:
    """
    Compute IC per cross-sectional date.

    Parameters
    ----------
    scores  : (Date, Ticker) MultiIndex Series or DataFrame with single score column
    returns : (Date, Ticker) MultiIndex Series of realized forward returns

    Returns
    -------
    pd.Series with DatetimeIndex and IC value per period.
    """
    if isinstance(scores, pd.DataFrame):
        if scores.shape[1] != 1:
            raise ValueError("scores DataFrame must have exactly 1 column")
        scores = scores.iloc[:, 0]

    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("returns DataFrame must have exactly 1 column")
        returns = returns.iloc[:, 0]

    # Align index
    common = scores.index.intersection(returns.index)
    sc = scores.loc[common]
    rt = returns.loc[common]

    dates = sc.index.get_level_values(date_level).unique().sort_values()
    ic_values: dict[pd.Timestamp, float] = {}

    for date in dates:
        sc_date = sc.xs(date, level=date_level)
        rt_date = rt.xs(date, level=date_level)
        ic_values[date] = information_coefficient(sc_date, rt_date, method=method)

    return pd.Series(ic_values, name="IC")


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summarise IC series with standard quant metrics.

    Returns
    -------
    dict with: mean_ic, std_ic, icir, t_stat, pct_positive, min_ic, max_ic
    """
    clean = ic_series.dropna()
    if len(clean) < 3:
        return {"error": "insufficient data"}

    mean_ic = clean.mean()
    std_ic  = clean.std()
    icir    = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat  = mean_ic / (std_ic / np.sqrt(len(clean))) if std_ic > 0 else np.nan

    return {
        "n_periods":    len(clean),
        "mean_ic":      round(mean_ic, 4),
        "std_ic":       round(std_ic, 4),
        "icir":         round(icir, 4),
        "t_stat":       round(t_stat, 3),
        "pct_positive": round((clean > 0).mean(), 3),
        "min_ic":       round(clean.min(), 4),
        "max_ic":       round(clean.max(), 4),
        "median_ic":    round(clean.median(), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Quintile / Top-K Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def topk_spread(
    scores: pd.Series,
    realized_returns: pd.Series,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
) -> float:
    """
    Mean return of top-k% minus mean return of bottom-k% stocks.

    This is a gross spread — no transaction costs. Interpret as a
    directional quality check, not a backtest P&L.
    """
    mask = scores.notna() & realized_returns.notna()
    sc = scores[mask]
    rt = realized_returns[mask]

    n = len(sc)
    if n < 10:
        return np.nan

    top_n    = max(1, int(n * top_pct))
    bottom_n = max(1, int(n * bottom_pct))

    top_tickers    = sc.nlargest(top_n).index
    bottom_tickers = sc.nsmallest(bottom_n).index

    top_ret    = rt.loc[top_tickers].mean()
    bottom_ret = rt.loc[bottom_tickers].mean()

    return float(top_ret - bottom_ret)


def rolling_topk_spread(
    scores: pd.Series,
    returns: pd.Series,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    date_level: str = "Date",
) -> pd.Series:
    """Compute top-k spread per period, return as pd.Series."""
    common = scores.index.intersection(returns.index)
    sc = scores.loc[common]
    rt = returns.loc[common]

    dates = sc.index.get_level_values(date_level).unique().sort_values()
    spreads: dict[pd.Timestamp, float] = {}

    for date in dates:
        sc_d = sc.xs(date, level=date_level)
        rt_d = rt.xs(date, level=date_level)
        spreads[date] = topk_spread(sc_d, rt_d, top_pct=top_pct, bottom_pct=bottom_pct)

    return pd.Series(spreads, name="topk_spread")


def quintile_returns(
    scores: pd.Series,
    realized_returns: pd.Series,
    n_bins: int = 5,
) -> pd.Series:
    """
    Average return per quintile bin (1 = lowest score, n_bins = highest).

    Returns pd.Series indexed 1..n_bins.
    """
    mask = scores.notna() & realized_returns.notna()
    sc = scores[mask]
    rt = realized_returns[mask]

    if len(sc) < n_bins * 2:
        return pd.Series(dtype=float)

    labels = pd.qcut(sc, q=n_bins, labels=False) + 1
    return rt.groupby(labels).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Stability Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def hit_rate(spread_series: pd.Series) -> float:
    """Fraction of periods where top-k outperformed bottom-k (spread > 0)."""
    clean = spread_series.dropna()
    if len(clean) == 0:
        return np.nan
    return float((clean > 0).mean())


def ic_decay(
    scores_history: dict[pd.Timestamp, pd.Series],
    returns_history: dict[pd.Timestamp, pd.Series],
    horizons: list[int] = (5, 10, 21, 42, 63),
) -> pd.DataFrame:
    """
    Compute IC at multiple forward horizons to measure signal persistence.

    A signal that has IC=0.04 at 21 days but IC=0.01 at 63 days tells
    us the information decays quickly — consistent with price-based momentum.

    Parameters
    ----------
    scores_history  : {date: cross-sectional score series}
    returns_history : {horizon_days: {date: cross-sectional return series}}

    Returns
    -------
    DataFrame: rows = dates, cols = horizons
    """
    rows = []
    for h in horizons:
        if h not in returns_history:
            continue
        ic_at_h = {}
        for date, scores in scores_history.items():
            if date not in returns_history[h]:
                continue
            ic_at_h[date] = information_coefficient(scores, returns_history[h][date])
        rows.append({"horizon": h, **ic_summary(pd.Series(ic_at_h))})

    return pd.DataFrame(rows).set_index("horizon")


def sharpe_ratio(returns: pd.Series, annualise_factor: int = 12) -> float:
    """
    Compute annualised Sharpe ratio of a monthly return series.
    (Used for backtest evaluation, not signal IC.)
    """
    clean = returns.dropna()
    if len(clean) < 3:
        return np.nan
    mean  = clean.mean()
    std   = clean.std()
    if std == 0:
        return np.nan
    return float(mean / std * np.sqrt(annualise_factor))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a cumulative return series."""
    roll_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / roll_max - 1
    return float(drawdown.min())


# ═══════════════════════════════════════════════════════════════════════════════
# Full Walk-Forward Evaluation Report
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_walk_forward(
    all_scores: pd.Series,
    all_returns: pd.Series,
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    top_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Run IC and top-k metrics across all OOS folds from walk-forward CV.

    Parameters
    ----------
    all_scores  : (Date, Ticker) MultiIndex Series of OOS model scores
    all_returns : (Date, Ticker) MultiIndex Series of realized returns
    fold_dates  : List of (test_start, test_end) tuples per fold

    Returns
    -------
    DataFrame with one row per fold:
        fold, test_start, test_end, ic, topk_spread, hit_rate, n_obs
    """
    rows = []
    for i, (t_start, t_end) in enumerate(fold_dates):
        mask_dates = (
            (all_scores.index.get_level_values("Date") >= t_start) &
            (all_scores.index.get_level_values("Date") <= t_end)
        )
        sc_fold = all_scores[mask_dates]
        rt_fold = all_returns[mask_dates]

        dates = sc_fold.index.get_level_values("Date").unique().sort_values()
        ic_vals   = []
        spread_vals = []

        for date in dates:
            sc_d = sc_fold.xs(date, level="Date")
            rt_d = rt_fold.xs(date, level="Date")
            ic_vals.append(information_coefficient(sc_d, rt_d))
            spread_vals.append(topk_spread(sc_d, rt_d, top_pct=top_pct))

        ic_s     = pd.Series(ic_vals)
        spread_s = pd.Series(spread_vals)

        rows.append({
            "fold":         i,
            "test_start":   t_start.date(),
            "test_end":     t_end.date(),
            "mean_ic":      round(ic_s.dropna().mean(), 4),
            "icir":         round(ic_s.dropna().mean() / ic_s.dropna().std(), 4)
                            if ic_s.dropna().std() > 0 else np.nan,
            "topk_spread":  round(spread_s.dropna().mean(), 4),
            "hit_rate":     round(hit_rate(spread_s), 3),
            "n_periods":    ic_s.notna().sum(),
        })

    return pd.DataFrame(rows)
