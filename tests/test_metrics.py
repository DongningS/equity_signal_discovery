"""
tests/test_metrics.py
─────────────────────
Unit tests for signal evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    hit_rate,
    ic_summary,
    information_coefficient,
    max_drawdown,
    quintile_returns,
    sharpe_ratio,
    topk_spread,
)


class TestIC:
    def test_perfect_positive_correlation(self):
        """IC = 1.0 when scores and returns are perfectly positively correlated."""
        scores  = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(information_coefficient(scores, returns) - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        """IC = -1.0 for perfectly inverse relationship."""
        scores  = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(information_coefficient(scores, returns) + 1.0) < 1e-6

    def test_no_correlation(self):
        """IC ≈ 0 for uncorrelated random data (probabilistic — uses fixed seed)."""
        np.random.seed(0)
        scores  = pd.Series(np.random.randn(1000))
        returns = pd.Series(np.random.randn(1000))
        ic = information_coefficient(scores, returns)
        assert abs(ic) < 0.15, f"IC={ic:.4f} unexpectedly high for uncorrelated data"

    def test_nan_handling(self):
        """IC computed only on overlapping non-NaN rows; result is finite (not NaN) when enough valid pairs remain."""
        # With 4 valid pairs (NaN in scores filters one row), IC is computable
        scores  = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        returns = pd.Series([1.0, 2.0, 3.0,    4.0, 5.0, 6.0, 7.0])
        ic = information_coefficient(scores, returns)
        assert not np.isnan(ic), f"Expected finite IC, got NaN. Only {scores.notna().sum()} valid rows."

    def test_insufficient_data(self):
        """IC returns NaN when fewer than 5 valid observations."""
        ic = information_coefficient(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
        )
        assert np.isnan(ic)

    def test_pearson_method(self):
        """Pearson IC supported."""
        scores  = pd.Series(np.arange(10, dtype=float))
        returns = pd.Series(np.arange(10, dtype=float))
        ic = information_coefficient(scores, returns, method="pearson")
        assert abs(ic - 1.0) < 1e-6

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            information_coefficient(
                pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
                pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
                method="kendall",
            )


class TestICSummary:
    def test_structure(self):
        ic_s = pd.Series([0.03, 0.05, -0.01, 0.04, 0.02, 0.06, -0.02])
        s = ic_summary(ic_s)
        for key in ["n_periods", "mean_ic", "std_ic", "icir", "t_stat", "pct_positive"]:
            assert key in s, f"Missing key: {key}"

    def test_icir_sign(self):
        """ICIR > 0 when mean IC > 0."""
        ic_s = pd.Series([0.03, 0.04, 0.05, 0.02, 0.06])
        s = ic_summary(ic_s)
        assert s["icir"] > 0

    def test_insufficient_data(self):
        ic_s = pd.Series([0.03, 0.05])
        s = ic_summary(ic_s)
        assert "error" in s


class TestTopKSpread:
    def test_positive_spread_when_sorted(self):
        """
        Top stocks (highest scores) have higher returns → positive spread.
        """
        scores  = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05,
                             0.06, 0.07, 0.08, 0.09, 0.10])
        spread = topk_spread(scores, returns, top_pct=0.2, bottom_pct=0.2)
        assert spread > 0

    def test_negative_spread_when_reversed(self):
        """
        Top stocks have lower returns → negative spread (bad signal).
        """
        scores  = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05,
                             0.06, 0.07, 0.08, 0.09, 0.10])
        spread = topk_spread(scores, returns, top_pct=0.2, bottom_pct=0.2)
        assert spread < 0

    def test_nan_for_small_input(self):
        """Spread is NaN when fewer than 10 valid observations."""
        scores  = pd.Series([1.0, 2.0, 3.0])
        returns = pd.Series([0.01, 0.02, 0.03])
        assert np.isnan(topk_spread(scores, returns))


class TestHitRate:
    def test_all_positive(self):
        assert hit_rate(pd.Series([0.01, 0.02, 0.005])) == 1.0

    def test_all_negative(self):
        assert hit_rate(pd.Series([-0.01, -0.02, -0.005])) == 0.0

    def test_mixed(self):
        hr = hit_rate(pd.Series([0.01, -0.02, 0.03, -0.04, 0.05]))
        assert abs(hr - 0.6) < 1e-6

    def test_empty(self):
        assert np.isnan(hit_rate(pd.Series([], dtype=float)))


class TestSharpe:
    def test_positive_sharpe(self):
        """Positive mean return with nonzero std → positive Sharpe."""
        np.random.seed(1)
        returns = pd.Series(0.01 + np.random.randn(24) * 0.002)
        assert sharpe_ratio(returns) > 0

    def test_zero_std(self):
        """Constant return series has zero std → NaN Sharpe (avoid div/0)."""
        result = sharpe_ratio(pd.Series([0.0] * 24))
        assert np.isnan(result)
        # Constant non-zero also has zero std → NaN
        result2 = sharpe_ratio(pd.Series([0.01] * 24))
        assert np.isnan(result2)

    def test_negative_sharpe(self):
        """Negative mean return → negative Sharpe."""
        np.random.seed(1)
        returns = pd.Series(-0.01 + np.random.randn(24) * 0.002)
        assert sharpe_ratio(returns) < 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing returns → zero drawdown."""
        cum = pd.Series([1.0, 1.01, 1.02, 1.03, 1.05])
        assert max_drawdown(cum) == 0.0

    def test_known_drawdown(self):
        """Peak at 1.1, trough at 0.9 → drawdown = (0.9/1.1 - 1) ≈ -18.2%."""
        cum = pd.Series([1.0, 1.1, 1.05, 0.9, 0.95, 1.0])
        dd = max_drawdown(cum)
        assert abs(dd - (0.9 / 1.1 - 1)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
