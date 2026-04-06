"""
src/targets.py
──────────────
Compute forward return targets and cross-sectional rank labels.

All targets are defined at observation time t, referencing returns
from t+1 to t+forward_days+1, meaning:
  - We observe features at close of day t
  - We "enter" positions at open of day t+1 (approximated as close t+1)
  - We "exit" at close of day t+forward_days+1

This convention is conservative — it avoids same-day close-to-close
look-ahead that inflates paper IC.

Target types
------------
  rank_target   : cross-sectional pct rank of fwd return ∈ [0, 1]
  quintile_label: integer 1–5 (1 = bottom, 5 = top)
  top_label     : binary 1 = top quintile, 0 = otherwise
  bottom_label  : binary 1 = bottom quintile, 0 = otherwise
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

log = logging.getLogger(__name__)


def forward_returns(
    prices: pd.DataFrame,
    forward_days: int = 21,
) -> pd.DataFrame:
    """
    Compute forward raw returns.

    Returns wide DataFrame (Date × Ticker) where each cell is the
    future N-day log return from t+1 to t+forward_days+1.
    """
    if isinstance(prices.index, pd.MultiIndex):
        close = prices["close"].unstack("Ticker")
    else:
        close = prices

    # Log return: more symmetric, better for ranking
    # Shifted by 1 to exclude same-day (t→t) return
    fwd = np.log(close.shift(-forward_days - 1) / close.shift(-1))
    fwd.index.name = "Date"
    return fwd


def rank_target(
    prices: pd.DataFrame,
    forward_days: int = 21,
    method: str = "average",
) -> pd.DataFrame:
    """
    Cross-sectional rank of forward returns, mapped to [0, 1].

    Higher rank = better expected return relative to peers.
    NaN where insufficient forward data exists (near the end of the sample).
    """
    fwd = forward_returns(prices, forward_days=forward_days)

    # Rank cross-sectionally per date
    ranked = fwd.rank(axis=1, pct=True, method=method, na_option="keep")
    return ranked


def quintile_labels(
    prices: pd.DataFrame,
    forward_days: int = 21,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Assign stocks to quintiles based on cross-sectional forward return rank.

    Returns integer labels 1 (worst) to n_bins (best).
    NaN for rows with insufficient coverage.
    """
    fwd = forward_returns(prices, forward_days=forward_days)

    def _quintile_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < n_bins:
            return pd.Series(np.nan, index=row.index)
        labels = pd.qcut(valid, q=n_bins, labels=False) + 1  # 1-indexed
        return labels.reindex(row.index)

    quintiles = fwd.apply(_quintile_row, axis=1)
    return quintiles


def build_target_panel(
    prices: pd.DataFrame,
    forward_days: int = 21,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Build a long-form (Date, Ticker) target DataFrame with all label types.

    Columns
    -------
    fwd_return      : raw log forward return
    rank_target     : cross-sectional pct rank [0, 1]
    quintile_label  : integer 1–n_bins
    top_label       : 1 if quintile == n_bins else 0
    bottom_label    : 1 if quintile == 1 else 0
    """
    log.info(f"Computing {forward_days}d forward return targets")

    fwd  = forward_returns(prices, forward_days=forward_days).stack().rename("fwd_return")
    rank = rank_target(prices, forward_days=forward_days).stack().rename("rank_target")
    qlab = quintile_labels(prices, forward_days=forward_days, n_bins=n_bins).stack().rename("quintile_label")

    target = pd.concat([fwd, rank, qlab], axis=1)
    target.index.names = ["Date", "Ticker"]
    target["top_label"]    = (target["quintile_label"] == n_bins).astype(float)
    target["bottom_label"] = (target["quintile_label"] == 1).astype(float)

    # Drop the last forward_days trading days — no valid forward return exists
    target = target.dropna(subset=["fwd_return"])

    log.info(
        f"Targets: {target.shape[0]:,} observations, "
        f"{target.index.get_level_values('Date').nunique()} dates, "
        f"{target.index.get_level_values('Ticker').nunique()} tickers"
    )
    return target


def align_features_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    drop_na_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner-join features and targets on (Date, Ticker), drop rows
    where more than `drop_na_threshold` fraction of features are NaN.

    Returns (X, y) ready for model training.
    """
    # Align on common index
    common_idx = features.index.intersection(targets.index)
    X = features.loc[common_idx]
    y = targets.loc[common_idx]

    # Drop rows with too many missing features
    n_feat = X.shape[1]
    min_non_null = int(n_feat * (1 - drop_na_threshold))
    valid_mask = X.notna().sum(axis=1) >= min_non_null
    X = X[valid_mask]
    y = y[valid_mask]

    # Ensure consistent index
    X = X.sort_index()
    y = y.sort_index()

    log.info(
        f"Aligned dataset: {X.shape[0]:,} rows × {X.shape[1]} features | "
        f"{y.notna().sum().min():,} min non-null targets"
    )
    return X, y
