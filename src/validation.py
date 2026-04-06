"""
src/validation.py
─────────────────
Walk-forward (time-series) cross-validation for financial data.

Why not sklearn's KFold?
  KFold randomly shuffles observations, creating future→past leakage in
  time-series data. Walk-forward guarantees all training data precedes
  all test data, respecting the temporal ordering of the universe.

Two strategies
--------------
  Expanding window: training set grows with each fold (more data = better)
  Rolling window  : fixed-length training window (more stable comparisons)

Both include a configurable gap between train end and test start to
approximate real-world latency in signal computation and execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def train_months(self) -> float:
        return (self.train_end - self.train_start).days / 30.44

    @property
    def test_months(self) -> float:
        return (self.test_end - self.test_start).days / 30.44

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold:02d}: "
            f"train [{self.train_start.date()} → {self.train_end.date()}] "
            f"({self.train_months:.1f}mo) | "
            f"test  [{self.test_start.date()} → {self.test_end.date()}] "
            f"({self.test_months:.1f}mo)"
        )


class WalkForwardSplitter:
    """
    Generates (train, test) index pairs for walk-forward validation.

    Parameters
    ----------
    min_train_months : Minimum months of data before first test period
    test_months      : Length of each OOS test window in months
    gap_days         : Trading-day gap between train end and test start
                       (approximates signal delay + execution latency)
    strategy         : 'expanding' (grows train window) or 'rolling' (fixed)
    max_train_months : Only used when strategy='rolling'

    Example
    -------
    >>> splitter = WalkForwardSplitter(min_train_months=24, test_months=3)
    >>> for split in splitter.split(X, dates):
    ...     X_train = X.iloc[split.train_idx]
    ...     X_test  = X.iloc[split.test_idx]
    """

    def __init__(
        self,
        min_train_months: int = 24,
        test_months: int = 3,
        gap_days: int = 5,
        strategy: str = "expanding",
        max_train_months: int | None = None,
    ) -> None:
        self.min_train_months  = min_train_months
        self.test_months       = test_months
        self.gap_days          = gap_days
        self.strategy          = strategy
        self.max_train_months  = max_train_months

        if strategy not in ("expanding", "rolling"):
            raise ValueError(f"strategy must be 'expanding' or 'rolling', got '{strategy}'")
        if strategy == "rolling" and max_train_months is None:
            raise ValueError("max_train_months required when strategy='rolling'")

    def split(
        self,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex | None = None,
    ) -> Iterator[SplitInfo]:
        """
        Yield SplitInfo objects for each walk-forward fold.

        Parameters
        ----------
        X : Feature matrix with DatetimeIndex or MultiIndex (Date, Ticker)
        dates : Optional explicit date index; inferred from X.index if None
        """
        if dates is None:
            if isinstance(X.index, pd.MultiIndex):
                dates_arr = pd.DatetimeIndex(X.index.get_level_values("Date").unique())
            else:
                dates_arr = pd.DatetimeIndex(X.index.unique())
        else:
            dates_arr = pd.DatetimeIndex(dates)

        dates_arr = dates_arr.sort_values()
        n_dates = len(dates_arr)

        # Find the earliest allowable test start
        min_train_delta = pd.DateOffset(months=self.min_train_months)
        earliest_test = dates_arr[0] + min_train_delta
        test_delta    = pd.DateOffset(months=self.test_months)

        fold = 0
        test_start = earliest_test

        while test_start <= dates_arr[-1]:
            test_end = test_start + test_delta - pd.Timedelta(days=1)
            test_end = min(test_end, dates_arr[-1])

            # Gap between train end and test start
            train_end_dt = test_start - pd.Timedelta(days=self.gap_days + 1)
            train_end_dt = max(train_end_dt, dates_arr[0])

            if self.strategy == "expanding":
                train_start_dt = dates_arr[0]
            else:  # rolling
                train_start_dt = train_end_dt - pd.DateOffset(months=self.max_train_months)
                train_start_dt = max(train_start_dt, dates_arr[0])

            # Convert date boundaries to positional indices
            train_dates_mask = (dates_arr >= train_start_dt) & (dates_arr <= train_end_dt)
            test_dates_mask  = (dates_arr >= test_start)     & (dates_arr <= test_end)

            if not train_dates_mask.any() or not test_dates_mask.any():
                test_start += test_delta
                continue

            train_dates = dates_arr[train_dates_mask]
            test_dates  = dates_arr[test_dates_mask]

            # Map date arrays back to integer positions in X
            if isinstance(X.index, pd.MultiIndex):
                all_dates = pd.DatetimeIndex(X.index.get_level_values("Date"))
                train_idx = np.where(all_dates.isin(train_dates))[0]
                test_idx  = np.where(all_dates.isin(test_dates))[0]
            else:
                all_dates = pd.DatetimeIndex(X.index)
                train_idx = np.where(all_dates.isin(train_dates))[0]
                test_idx  = np.where(all_dates.isin(test_dates))[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                test_start += test_delta
                continue

            split = SplitInfo(
                fold=fold,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_idx=train_idx,
                test_idx=test_idx,
            )
            log.debug(split)
            yield split

            fold += 1
            test_start += test_delta

    def n_splits(self, X: pd.DataFrame) -> int:
        """Count the number of folds without consuming the generator."""
        return sum(1 for _ in self.split(X))

    def summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame summarising all folds for inspection."""
        rows = []
        for s in self.split(X):
            rows.append({
                "fold":          s.fold,
                "train_start":   s.train_start.date(),
                "train_end":     s.train_end.date(),
                "train_months":  round(s.train_months, 1),
                "test_start":    s.test_start.date(),
                "test_end":      s.test_end.date(),
                "test_months":   round(s.test_months, 1),
                "train_n":       len(s.train_idx),
                "test_n":        len(s.test_idx),
            })
        return pd.DataFrame(rows)
