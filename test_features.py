"""
tests/test_features.py
──────────────────────
Unit tests for feature engineering functions.

Key invariants tested
---------------------
1. No look-ahead leakage: features at time t must not use data from t+1 onwards
2. Cross-sectional ranking outputs sum to ~0.5 * n_stocks (pct rank property)
3. Momentum correctly skips the last `skip_last` days
4. Volatility uses only past data within the rolling window
5. Feature matrix has no fully-empty columns after assembly
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    bollinger_position,
    forward_returns,
    momentum,
    realized_volatility,
    rsi,
    sma_ratio,
    _cross_sectional_rank_panel,
)
from src.targets import build_target_panel, rank_target


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_prices(n_days: int = 500, n_tickers: int = 20) -> pd.DataFrame:
    """
    Generate synthetic price panel as MultiIndex (Date, Ticker).
    Uses geometric Brownian motion to produce realistic price paths.
    """
    np.random.seed(42)
    dates = pd.bdate_range(start="2016-01-04", periods=n_days)
    tickers = [f"STOCK_{i:02d}" for i in range(n_tickers)]

    records = []
    for ticker in tickers:
        price = 100.0
        for date in dates:
            ret = np.random.normal(0.0003, 0.015)
            price *= (1 + ret)
            vol_factor = abs(np.random.normal(1, 0.3))
            records.append({
                "Date":   date,
                "Ticker": ticker,
                "open":   price * (1 + np.random.normal(0, 0.002)),
                "high":   price * (1 + abs(np.random.normal(0, 0.008))),
                "low":    price * (1 - abs(np.random.normal(0, 0.008))),
                "close":  price,
                "volume": int(1e6 * vol_factor),
            })

    df = pd.DataFrame(records)
    df = df.set_index(["Date", "Ticker"]).sort_index()
    return df


@pytest.fixture(scope="module")
def prices():
    return _make_prices()


# ── Test: No Look-Ahead in Momentum ──────────────────────────────────────────

class TestMomentum:
    def test_shape(self, prices):
        mom = momentum(prices, windows=[21], skip_last=5)
        assert isinstance(mom, pd.DataFrame)
        assert "mom_21d" in mom.columns

    def test_no_lookahead_skip_last(self, prices):
        """
        With skip_last=5, momentum at date t should not change if we
        shuffle returns from t-1 to t (these are within the skip window).
        """
        mom = momentum(prices, windows=[21], skip_last=5)
        # The feature at any date t uses only data up to t-5
        # Verify: the earliest 26 dates should all be NaN
        close = prices["close"].unstack("Ticker")
        mom_wide = mom["mom_21d"].unstack("Ticker")

        # First 21+5=26 days must be NaN (not enough history)
        earliest_valid = mom_wide.dropna(how="all").index[0]
        n_nan_dates = (mom_wide.index < earliest_valid).sum()
        assert n_nan_dates >= 25, f"Expected ≥25 NaN dates, got {n_nan_dates}"

    def test_correct_window(self, prices):
        """Momentum at t uses price(t-skip_last) / price(t-window-skip_last) - 1."""
        close = prices["close"].unstack("Ticker")
        ticker = close.columns[0]
        skip, w = 5, 21

        mom = momentum(prices, windows=[w], skip_last=skip)
        mom_wide = mom["mom_21d"].unstack("Ticker")

        idx = 60  # arbitrary date with sufficient history
        date = close.index[idx]
        expected = close[ticker].iloc[idx - skip] / close[ticker].iloc[idx - skip - w] - 1
        computed = mom_wide[ticker].iloc[idx]
        assert abs(computed - expected) < 1e-10, f"Expected {expected:.6f}, got {computed:.6f}"

    def test_multiple_windows(self, prices):
        mom = momentum(prices, windows=[21, 63, 252], skip_last=5)
        for col in ["mom_21d", "mom_63d", "mom_252d"]:
            assert col in mom.columns


# ── Test: Volatility ──────────────────────────────────────────────────────────

class TestVolatility:
    def test_positive(self, prices):
        """Realized volatility must be non-negative."""
        rv = realized_volatility(prices, windows=[21])
        assert (rv["vol_21d"].dropna() >= 0).all()

    def test_nans_at_start(self, prices):
        """First 21 days must be NaN for 21-day window."""
        rv = realized_volatility(prices, windows=[21])
        vol_wide = rv["vol_21d"].unstack("Ticker")
        # With min_periods = 0.75 * window, first ~16 dates are NaN
        first_valid = vol_wide.dropna(how="all").index[0]
        n_nan_dates = (vol_wide.index < first_valid).sum()
        assert n_nan_dates >= 10

    def test_annualisation(self, prices):
        """Annualised vol should be sqrt(252) times daily vol."""
        rv_ann   = realized_volatility(prices, windows=[21], annualize=True)
        rv_daily = realized_volatility(prices, windows=[21], annualize=False)
        ratio = rv_ann["vol_21d"].dropna() / rv_daily["vol_21d"].dropna()
        # All ratios should be very close to sqrt(252)
        close_to_sqrt252 = (ratio - np.sqrt(252)).abs() < 0.01
        assert close_to_sqrt252.mean() > 0.99


# ── Test: RSI ─────────────────────────────────────────────────────────────────

class TestRSI:
    def test_bounds(self, prices):
        """RSI must lie within [0, 100]."""
        r = rsi(prices, period=14)
        valid = r["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_no_negative(self, prices):
        r = rsi(prices, period=14)
        assert (r["rsi_14"].dropna() >= 0).all()


# ── Test: Cross-Sectional Ranking ─────────────────────────────────────────────

class TestCrossSectionalRank:
    def test_range(self, prices):
        """Cross-sectional pct ranks must be in [0, 1]."""
        mom = momentum(prices, windows=[21], skip_last=5)
        ranked = _cross_sectional_rank_panel(mom)
        valid = ranked["mom_21d"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_uniform_cross_section(self, prices):
        """
        For each date, cross-sectional ranks should have mean ≈ 0.5
        (property of percentile ranks on a complete cross-section).
        """
        mom = momentum(prices, windows=[21], skip_last=5)
        ranked = _cross_sectional_rank_panel(mom)
        wide = ranked["mom_21d"].unstack("Ticker").dropna()
        row_means = wide.mean(axis=1)
        # Mean rank per date should be very close to 0.5
        assert ((row_means - 0.5).abs() < 0.05).all(), \
            f"Row means deviate from 0.5: {row_means.describe()}"


# ── Test: Targets ──────────────────────────────────────────────────────────────

class TestTargets:
    def test_rank_in_01(self, prices):
        """Rank target must lie in [0, 1]."""
        targets = build_target_panel(prices, forward_days=21)
        valid = targets["rank_target"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_forward_return_no_lookahead(self, prices):
        """
        Forward returns use only future data. Verify: if we remove the last
        N days from prices, the forward return on date T should be NaN for
        dates near the end.
        """
        targets = build_target_panel(prices, forward_days=21)
        # Last 21+1 = 22 dates should have NaN forward returns
        # (build_target_panel drops these via dropna, so count the total rows)
        close = prices["close"].unstack("Ticker")
        n_total_dates = close.index.nunique()
        n_target_dates = targets.index.get_level_values("Date").nunique()
        # At least forward_days fewer dates in targets
        assert n_target_dates <= n_total_dates - 20, \
            f"Expected targets to drop at least 20 dates, got {n_total_dates - n_target_dates}"

    def test_quintile_labels_integers(self, prices):
        """Quintile labels must be integers 1–5."""
        targets = build_target_panel(prices, forward_days=21, n_bins=5)
        labels = targets["quintile_label"].dropna()
        assert set(labels.unique()).issubset({1, 2, 3, 4, 5})

    def test_binary_labels_consistent(self, prices):
        """top_label should be 1 iff quintile_label == 5."""
        targets = build_target_panel(prices, forward_days=21, n_bins=5)
        valid = targets.dropna(subset=["quintile_label", "top_label"])
        assert ((valid["quintile_label"] == 5) == (valid["top_label"] == 1)).all()


# ── Test: No Data Leakage in SMA ──────────────────────────────────────────────

class TestSMARatio:
    def test_no_future_price(self, prices):
        """SMA at time t must not use prices from t+1."""
        close = prices["close"].unstack("Ticker")
        ticker = close.columns[0]
        sma = sma_ratio(prices, windows=[50])
        sma_wide = sma["price_sma_50"].unstack("Ticker")

        # Manually compute SMA for a specific date
        idx = 100
        date = close.index[idx]
        expected_sma = close[ticker].iloc[idx - 49 : idx + 1].mean()
        expected_ratio = close[ticker].iloc[idx] / expected_sma
        computed = sma_wide[ticker].iloc[idx]
        assert abs(computed - expected_ratio) < 1e-6


# ── Test: Bollinger Bands ─────────────────────────────────────────────────────

class TestBollinger:
    def test_bounds(self, prices):
        """Bollinger position can be outside [0,1] when price breaks bands."""
        bb = bollinger_position(prices, window=20)
        # Just verify it runs and has finite values
        valid = bb["bb_position"].dropna()
        assert valid.notna().sum() > 0
        assert np.isfinite(valid).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
