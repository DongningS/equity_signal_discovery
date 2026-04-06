"""
src/plotting.py
───────────────
Publication-quality charts for signal discovery and evaluation.

All functions return matplotlib Figure objects so callers can save
or display as needed. Style is kept clean — suitable for a portfolio
or research write-up without being flashy.

Usage
-----
    from src.plotting import plot_rolling_ic, plot_topk_spread, plot_shap_summary
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Style defaults ─────────────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#185FA5",
    "teal":   "#0F6E56",
    "red":    "#A32D2D",
    "amber":  "#BA7517",
    "gray":   "#5F5E5A",
    "light":  "#D3D1C7",
}

plt.rcParams.update({
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.8,
})

US_RECESSION_BANDS = [
    ("2007-12-01", "2009-06-30"),  # GFC
    ("2020-02-01", "2020-04-30"),  # COVID
]


def _add_recession_bands(ax: plt.Axes, alpha: float = 0.12) -> None:
    """Add light gray shading for US recession periods."""
    for start, end in US_RECESSION_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="gray", alpha=alpha, label=None)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IC Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rolling_ic(
    ic_series: pd.Series,
    model_name: str = "Model",
    rolling_window: int = 4,
) -> plt.Figure:
    """
    Bar chart of per-period IC with rolling mean overlay and zero line.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    clean = ic_series.dropna()

    colors = [PALETTE["teal"] if v > 0 else PALETTE["red"] for v in clean]
    ax.bar(clean.index, clean.values, color=colors, alpha=0.7, width=15, label="IC per period")

    rolling = clean.rolling(rolling_window, min_periods=1).mean()
    ax.plot(rolling.index, rolling.values,
            color=PALETTE["blue"], linewidth=2, label=f"{rolling_window}-period rolling IC")
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.8, linestyle="--")

    mean_ic  = clean.mean()
    ax.axhline(mean_ic, color=PALETTE["amber"], linewidth=1.5, linestyle=":",
               label=f"Mean IC = {mean_ic:.4f}")

    _add_recession_bands(ax)
    ax.set_title(f"{model_name} — Rolling Information Coefficient (IC)")
    ax.set_ylabel("Spearman IC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ic_comparison(
    ic_dict: dict[str, pd.Series],
) -> plt.Figure:
    """
    Compare IC series from multiple models side by side.
    """
    n = len(ic_dict)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = list(PALETTE.values())
    for ax, (name, ic_s), color in zip(axes, ic_dict.items(), colors):
        clean = ic_s.dropna()
        ax.bar(clean.index, clean.values, color=color, alpha=0.6, width=15)
        ax.axhline(0, color=PALETTE["gray"], linewidth=0.8, linestyle="--")
        mean_ic = clean.mean()
        icir = mean_ic / clean.std() if clean.std() > 0 else 0
        ax.set_title(f"{name}  |  Mean IC = {mean_ic:.4f}  |  ICIR = {icir:.3f}")
        ax.set_ylabel("IC")
        _add_recession_bands(ax)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    fig.suptitle("Model IC Comparison — Walk-Forward OOS", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_ic_heatmap(
    feature_ic: pd.DataFrame,
    title: str = "Feature IC Heatmap (by period)",
) -> plt.Figure:
    """
    Heatmap of per-feature IC across rolling periods.

    Parameters
    ----------
    feature_ic : DataFrame with index=dates, columns=feature names, values=IC
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("Install seaborn: pip install seaborn")

    fig, ax = plt.subplots(figsize=(max(12, len(feature_ic.columns) * 0.6), 6))
    sns.heatmap(
        feature_ic.T,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-0.15,
        vmax=0.15,
        linewidths=0.3,
        cbar_kws={"shrink": 0.6, "label": "IC"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel("Feature")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Top-K Spread Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_topk_spread(
    spread_series: pd.Series,
    model_name: str = "Model",
) -> plt.Figure:
    """Monthly top-quintile vs bottom-quintile return spread."""
    fig, ax = plt.subplots(figsize=(12, 4))
    clean = spread_series.dropna()

    colors = [PALETTE["teal"] if v > 0 else PALETTE["red"] for v in clean]
    ax.bar(clean.index, clean.values * 100, color=colors, alpha=0.75, width=15)
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.8, linestyle="--")

    mean_spread = clean.mean()
    hit = (clean > 0).mean()
    ax.axhline(mean_spread * 100, color=PALETTE["blue"], linewidth=1.5, linestyle=":",
               label=f"Mean spread = {mean_spread*100:.2f}%  |  Hit rate = {hit:.1%}")

    _add_recession_bands(ax)
    ax.set_title(f"{model_name} — Top-20% vs Bottom-20% Monthly Return Spread")
    ax.set_ylabel("Return Spread (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_quintile_returns(
    quintile_ret: pd.Series,
    model_name: str = "Model",
) -> plt.Figure:
    """Bar chart of average return per quintile bin."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE["red"], PALETTE["amber"], PALETTE["gray"],
              PALETTE["teal"], PALETTE["blue"]]
    bars = ax.bar(quintile_ret.index, quintile_ret.values * 100,
                  color=colors[:len(quintile_ret)], alpha=0.85, width=0.6)
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.8, linestyle="--")
    ax.set_xticks(quintile_ret.index)
    ax.set_xticklabels([f"Q{i}" for i in quintile_ret.index])
    ax.set_title(f"{model_name} — Average Return by Quintile")
    ax.set_xlabel("Quintile (Q1 = lowest score, Q5 = highest)")
    ax.set_ylabel("Avg Forward Return (%)")

    for bar, val in zip(bars, quintile_ret.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Backtest Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_returns(
    cumulative: pd.DataFrame,
    stats: dict,
) -> plt.Figure:
    """Cumulative return comparison: portfolio vs benchmark."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    ax1.plot(cumulative.index, cumulative["portfolio"],
             color=PALETTE["blue"], linewidth=2, label="Top-quintile portfolio")
    if "spy" in cumulative.columns:
        ax1.plot(cumulative.index, cumulative["spy"],
                 color=PALETTE["gray"], linewidth=1.5, linestyle="--", label="SPY")

    _add_recession_bands(ax1)
    ax1.set_ylabel("Cumulative Return (1 = start)")
    ax1.legend()
    ax1.set_title(
        f"Pre-cost Gross Simulation  |  "
        f"Total: {stats['total_return_pct']}%  |  "
        f"Sharpe: {stats['monthly_sharpe']}  |  "
        f"Max DD: {stats['max_drawdown_pct']}%"
    )

    # Drawdown
    roll_max = cumulative["portfolio"].cummax()
    drawdown = (cumulative["portfolio"] / roll_max - 1) * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4,
                     color=PALETTE["red"], label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30, ha="right")

    # Disclaimer box
    ax2.text(
        0.01, -0.45, stats["disclaimer"],
        transform=ax2.transAxes, fontsize=7, color=PALETTE["gray"],
        style="italic", ha="left",
    )

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feature Importance / SHAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    model_name: str = "Model",
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    top = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(top.index, top.values, color=PALETTE["blue"], alpha=0.75)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 15,
) -> plt.Figure:
    """
    SHAP beeswarm summary plot.
    Requires shap installed.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    fig, ax = plt.subplots(figsize=(10, max(4, max_display * 0.4)))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.gcf().set_size_inches(10, max(4, max_display * 0.4))
    plt.tight_layout()
    return plt.gcf()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EDA Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_return_distribution(
    returns: pd.Series,
    ticker_or_label: str = "Universe",
) -> plt.Figure:
    """Histogram + KDE of returns with normal overlay."""
    try:
        import seaborn as sns
        from scipy import stats as scipy_stats
    except ImportError:
        raise ImportError("Install seaborn and scipy")

    fig, ax = plt.subplots(figsize=(9, 4))
    clean = returns.dropna()
    sns.histplot(clean, bins=80, stat="density", color=PALETTE["blue"],
                 alpha=0.5, ax=ax, label="Observed")
    mu, sigma = clean.mean(), clean.std()
    x = np.linspace(clean.quantile(0.01), clean.quantile(0.99), 200)
    ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma),
            color=PALETTE["red"], linewidth=1.5, label="Normal fit")
    ax.set_title(f"{ticker_or_label} — Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    stats_text = f"μ={mu:.4f}  σ={sigma:.4f}  skew={clean.skew():.2f}  kurt={clean.kurtosis():.2f}"
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color=PALETTE["gray"])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_universe_coverage(prices: pd.DataFrame) -> plt.Figure:
    """Show data availability (non-NaN days) per ticker."""
    if isinstance(prices.index, pd.MultiIndex):
        coverage = prices["close"].unstack("Ticker").notna().sum()
    else:
        coverage = prices.notna().sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    coverage.sort_values().plot(kind="barh", ax=ax, color=PALETTE["teal"], alpha=0.7)
    ax.axvline(coverage.mean(), color=PALETTE["red"], linestyle="--", linewidth=1.2,
               label=f"Mean: {coverage.mean():.0f} days")
    ax.set_title("Universe Data Coverage — Trading Days per Ticker")
    ax.set_xlabel("Number of trading days with data")
    ax.legend()
    fig.tight_layout()
    return fig
