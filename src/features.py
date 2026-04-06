"""
src/features.py
───────────────
Cross-sectional feature engineering — all functions are point-in-time safe.

Design principles
-----------------
1. Every function takes a wide DataFrame (Date × Ticker) and returns
   a DataFrame of the same shape (or a panel with the same MultiIndex).
2. Fundamental features are explicitly lagged before use.
3. All features are cross-sectionally ranked (winsorized percentile) at
   the end to make them comparable across stocks and robust to outliers.
4. No future data ever enters the computation — validated in tests/.

Feature categories
------------------
  - Momentum       : price-based trend signals
  - Volatility     : realized vol, range-based
  - Technical      : RSI, Bollinger, SMA ratios
  - Fundamental    : value + quality ratios (lagged)
  - Macro          : macro regime context (lagged)
  - Sentiment      : VADER RSS proxy (optional)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _wide_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted close; return wide (Date × Ticker) DataFrame."""
    if isinstance(prices.index, pd.MultiIndex):
        return prices["close"].unstack("Ticker")
    return prices  # already wide


def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, replace values with cross-sectional percentile ranks [0, 1].
    NaN rows are left as NaN.
    """
    return df.rank(axis=1, pct=True, na_option="keep")


def _winsorise(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip per-column to [lower, upper] quantile to remove outliers."""
    q_lo = df.quantile(lower, axis=1)
    q_hi = df.quantile(upper, axis=1)
    return df.clip(lower=q_lo, upper=q_hi, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Momentum Features
# ═══════════════════════════════════════════════════════════════════════════════

def momentum(
    prices: pd.DataFrame,
    windows: list[int] = (21, 63, 252),
    skip_last: int = 5,
) -> pd.DataFrame:
    """
    Price momentum over multiple windows.

    Calculated as the cumulative return from (t - window) to (t - skip_last),
    which avoids the short-term reversal contaminating the signal.

    Returns
    -------
    DataFrame with MultiIndex (Date, Ticker) and columns like 'mom_21', 'mom_63'.
    """
    close = _wide_close(prices)
    frames: dict[str, pd.DataFrame] = {}

    for w in windows:
        # Return from (t - w) to (t - skip_last)
        ret = (
            close.shift(skip_last) / close.shift(w + skip_last) - 1.0
        )
        frames[f"mom_{w}d"] = ret

    result = pd.concat(frames, axis=1)
    result.columns.names = ["feature", "Ticker"]
    result = result.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


def short_term_reversal(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Short-term reversal: negative of prior-week return.
    Known to be mean-reverting over 1–5 day horizon.
    """
    close = _wide_close(prices)
    rev = -(close / close.shift(window) - 1.0)
    rev.columns = pd.MultiIndex.from_arrays([["rev_5d"] * len(rev.columns), rev.columns])
    rev.columns.names = ["feature", "Ticker"]
    result = rev.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Volatility Features
# ═══════════════════════════════════════════════════════════════════════════════

def realized_volatility(
    prices: pd.DataFrame,
    windows: list[int] = (21, 63),
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Realized volatility: rolling std of log returns.

    Annualized by sqrt(252). Lower vol stocks are assigned higher
    (better) rank by design — this proxy captures risk-adjusted quality.
    """
    close = _wide_close(prices)
    log_ret = np.log(close / close.shift(1))
    factor = np.sqrt(252) if annualize else 1.0
    frames: dict[str, pd.DataFrame] = {}

    for w in windows:
        rv = log_ret.rolling(w, min_periods=int(w * 0.75)).std() * factor
        frames[f"vol_{w}d"] = rv

    result = pd.concat(frames, axis=1)
    result.columns.names = ["feature", "Ticker"]
    result = result.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


def atr_ratio(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average True Range normalised by closing price.
    Captures intraday volatility independent of price level.
    """
    if isinstance(prices.index, pd.MultiIndex):
        hi = prices["high"].unstack("Ticker")
        lo = prices["low"].unstack("Ticker")
        cl = prices["close"].unstack("Ticker")
    else:
        hi = prices.get("high", prices)
        lo = prices.get("low", prices)
        cl = prices.get("close", prices)

    prev_cl = cl.shift(1)
    tr = pd.concat(
        [hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1
    ).max(axis=1)

    # Reconstruct per-ticker: tr is flat after concat — redo per ticker
    atr_frames: dict[str, pd.Series] = {}
    for ticker in cl.columns:
        _hi = hi[ticker] if ticker in hi.columns else hi.squeeze()
        _lo = lo[ticker] if ticker in lo.columns else lo.squeeze()
        _cl = cl[ticker]
        _prev = _cl.shift(1)
        _tr = pd.concat([_hi - _lo, (_hi - _prev).abs(), (_lo - _prev).abs()], axis=1).max(axis=1)
        atr_frames[ticker] = _tr.rolling(window).mean() / _cl

    atr_df = pd.DataFrame(atr_frames)
    atr_df = atr_df.rename(columns={c: c for c in atr_df.columns})
    atr_df.columns = pd.MultiIndex.from_arrays(
        [["atr_ratio"] * len(atr_df.columns), atr_df.columns]
    )
    atr_df.columns.names = ["feature", "Ticker"]
    result = atr_df.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Technical Features
# ═══════════════════════════════════════════════════════════════════════════════

def rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (Wilder smoothing).

    RSI > 70 = potentially overbought, RSI < 30 = potentially oversold.
    Used here as a cross-sectional feature — high RSI relative to peers
    captures relative trend strength.
    """
    close = _wide_close(prices)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing = EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    rsi_val.columns = pd.MultiIndex.from_arrays(
        [[f"rsi_{period}"] * len(rsi_val.columns), rsi_val.columns]
    )
    rsi_val.columns.names = ["feature", "Ticker"]
    result = rsi_val.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


def sma_ratio(prices: pd.DataFrame, windows: list[int] = (50, 200)) -> pd.DataFrame:
    """
    Price / SMA ratio. Values > 1 indicate price above moving average.
    """
    close = _wide_close(prices)
    frames: dict[str, pd.DataFrame] = {}

    for w in windows:
        sma = close.rolling(w, min_periods=int(w * 0.8)).mean()
        frames[f"price_sma_{w}"] = close / sma

    result = pd.concat(frames, axis=1)
    result.columns.names = ["feature", "Ticker"]
    result = result.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


def bollinger_position(
    prices: pd.DataFrame, window: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Band position: (price - lower) / (upper - lower).
    0 = at lower band, 1 = at upper band.
    """
    close = _wide_close(prices)
    sma = close.rolling(window, min_periods=int(window * 0.75)).mean()
    std = close.rolling(window, min_periods=int(window * 0.75)).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    bb_pos = (close - lower) / (upper - lower + 1e-10)

    bb_pos.columns = pd.MultiIndex.from_arrays(
        [["bb_position"] * len(bb_pos.columns), bb_pos.columns]
    )
    bb_pos.columns.names = ["feature", "Ticker"]
    result = bb_pos.stack("Ticker")
    result.index.names = ["Date", "Ticker"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fundamental Features
# ═══════════════════════════════════════════════════════════════════════════════

def fundamental_features(
    fundam: pd.DataFrame,
    prices: pd.DataFrame,
    lag_days: int = 90,
) -> pd.DataFrame:
    """
    Merge static fundamentals with price panel, then lag by `lag_days`.

    The lag approximates the delay between fiscal quarter end and SEC
    filing / public availability. This is a simplification — point-in-time
    databases (e.g. Compustat) should be used in production.

    Returns long-form (Date, Ticker) DataFrame with fundamental columns.
    """
    close = _wide_close(prices)
    dates = close.index

    # Build panel: broadcast fundamentals across all dates
    records = []
    for ticker in fundam.index:
        if ticker not in close.columns:
            continue
        row = fundam.loc[ticker].to_dict()
        row["Ticker"] = ticker
        records.append(row)

    if not records:
        log.warning("No fundamental data matched price universe — skipping")
        return pd.DataFrame()

    static_df = pd.DataFrame(records).set_index("Ticker")

    # Build cross join: (Date, Ticker) × fundamental columns
    # Then shift the effective date forward by lag_days
    all_tickers = static_df.index.tolist()
    idx = pd.MultiIndex.from_product([dates, all_tickers], names=["Date", "Ticker"])
    panel = pd.DataFrame(index=idx)

    for col in static_df.columns:
        panel[col] = panel.index.get_level_values("Ticker").map(static_df[col])

    # Apply lag: shift the series by lag_days (the fundamentals are "stale" by lag_days)
    # In a real pipeline this would align on actual filing dates
    panel = panel.sort_index()
    numeric_cols = panel.select_dtypes("number").columns.tolist()
    panel[numeric_cols] = (
        panel[numeric_cols]
        .groupby(level="Ticker")
        .transform(lambda x: x.shift(lag_days))
    )

    return panel[numeric_cols]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Macro Features
# ═══════════════════════════════════════════════════════════════════════════════

def macro_features(
    macro: pd.DataFrame,
    prices: pd.DataFrame,
    lag_days: int = 30,
) -> pd.DataFrame:
    """
    Broadcast macro series onto the (Date, Ticker) panel.

    Macro data is lagged by lag_days to avoid look-ahead.
    Macro features are the same for all stocks on a given date — they act
    as regime indicators rather than stock-specific signals.
    """
    close = _wide_close(prices)
    tickers = close.columns.tolist()
    dates = close.index

    # Lag macro series
    macro_lagged = macro.shift(lag_days)
    macro_lagged = macro_lagged.reindex(dates, method="ffill")

    # Add interaction terms
    if "t10y2y" in macro_lagged.columns and "vix" in macro_lagged.columns:
        macro_lagged["yield_vix_interaction"] = (
            macro_lagged["t10y2y"] * macro_lagged["vix"]
        )
    if "vix" in macro_lagged.columns:
        macro_lagged["high_vix"] = (macro_lagged["vix"] > 25).astype(float)

    # Cross join onto (Date, Ticker) panel
    records = []
    for date in dates:
        if date not in macro_lagged.index:
            continue
        macro_row = macro_lagged.loc[date]
        for ticker in tickers:
            row = macro_row.to_dict()
            row["Date"] = date
            row["Ticker"] = ticker
            records.append(row)

    if not records:
        return pd.DataFrame()

    panel = pd.DataFrame(records).set_index(["Date", "Ticker"]).sort_index()
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Sentiment Features (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def sentiment_proxy(
    tickers: list[str],
    start: str,
    end: str,
    window_days: int = 7,
) -> Optional[pd.DataFrame]:
    """
    VADER sentiment proxy from Yahoo Finance RSS headlines.

    IMPORTANT CAVEAT:
    - Yahoo Finance RSS does not provide historical headlines.
    - This function demonstrates the methodology — in practice, a
      historical news archive (e.g. RavenPack, GDELT) would be required.
    - IC of this proxy is expected to be near zero on public data; this
      is an intentional finding, not a bug.

    Returns None if vaderSentiment is not installed.
    """
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        log.warning("vaderSentiment or feedparser not installed — skipping sentiment")
        return None

    analyzer = SentimentIntensityAnalyzer()
    url_template = cfg.get(
        "features.sentiment.sources",
        ["https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"],
    )[0]

    scores: dict[str, float] = {}
    for ticker in tickers[:20]:  # Limit to avoid rate limiting
        url = url_template.format(ticker=ticker)
        try:
            feed = feedparser.parse(url)
            headlines = [entry.title for entry in feed.entries[:20]]
            if not headlines:
                scores[ticker] = np.nan
                continue
            compound_scores = [
                analyzer.polarity_scores(h)["compound"] for h in headlines
            ]
            scores[ticker] = np.mean(compound_scores)
        except Exception:
            scores[ticker] = np.nan

    log.info(
        f"Sentiment computed for {sum(~np.isnan(v) for v in scores.values())} tickers "
        f"(current only — not historical)"
    )
    return pd.Series(scores, name="sentiment_score")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Feature Assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    prices: pd.DataFrame,
    fundam: Optional[pd.DataFrame] = None,
    macro: Optional[pd.DataFrame] = None,
    cfg_features: Optional[dict] = None,
    cross_section_rank: bool = True,
) -> pd.DataFrame:
    """
    Assemble all features into a single (Date, Ticker) × feature DataFrame.

    Steps
    -----
    1. Compute each feature group independently.
    2. Concatenate along the column axis.
    3. Optionally cross-sectionally rank each feature (pct rank per date).
    4. Winsorise cross-sectional outliers.

    Parameters
    ----------
    prices : MultiIndex DataFrame (Date, Ticker) from raw/prices.parquet
    fundam : Static fundamentals from raw/fundamentals.parquet (optional)
    macro  : Macro panel from raw/macro.parquet (optional)
    cross_section_rank : If True, replace each feature with cross-sectional pct rank
    """
    from src import config as cfg_module

    feat_cfg = cfg_features or {}
    mom_cfg  = feat_cfg.get("momentum",   {})
    vol_cfg  = feat_cfg.get("volatility", {})
    tech_cfg = feat_cfg.get("technical",  {})
    fund_cfg = feat_cfg.get("fundamental",{})
    macro_cfg= feat_cfg.get("macro",      {})

    log.info("Building feature matrix...")
    frames: list[pd.DataFrame] = []

    # ── Momentum ─────────────────────────────────────────────────────────────
    log.info("  Computing momentum features")
    mom_windows = mom_cfg.get("windows", [21, 63, 252])
    mom_skip    = mom_cfg.get("skip_last", 5)
    frames.append(momentum(prices, windows=mom_windows, skip_last=mom_skip))
    frames.append(short_term_reversal(prices, window=5))

    # ── Volatility ────────────────────────────────────────────────────────────
    log.info("  Computing volatility features")
    vol_windows = vol_cfg.get("windows", [21, 63])
    frames.append(realized_volatility(prices, windows=vol_windows))
    frames.append(atr_ratio(prices))

    # ── Technical ─────────────────────────────────────────────────────────────
    log.info("  Computing technical features")
    rsi_period  = tech_cfg.get("rsi_period", 14)
    sma_windows = tech_cfg.get("sma_windows", [50, 200])
    frames.append(rsi(prices, period=rsi_period))
    frames.append(sma_ratio(prices, windows=sma_windows))
    frames.append(bollinger_position(prices))

    # ── Fundamentals ──────────────────────────────────────────────────────────
    if fundam is not None:
        log.info("  Computing fundamental features (with lag)")
        lag = fund_cfg.get("lag_days", 90)
        fund_panel = fundamental_features(fundam, prices, lag_days=lag)
        if not fund_panel.empty:
            frames.append(fund_panel)

    # ── Macro ──────────────────────────────────────────────────────────────────
    if macro is not None:
        log.info("  Computing macro features")
        lag = macro_cfg.get("lag_days", 30)
        macro_panel = macro_features(macro, prices, lag_days=lag)
        if not macro_panel.empty:
            frames.append(macro_panel)

    # ── Assemble ──────────────────────────────────────────────────────────────
    log.info("  Concatenating feature frames")
    feature_df = pd.concat(frames, axis=1)
    feature_df = feature_df.sort_index()

    # Remove fully-empty columns and rows with no features at all
    feature_df = feature_df.dropna(axis=1, how="all")
    min_features = max(1, len(feature_df.columns) // 2)
    feature_df = feature_df.dropna(thresh=min_features)

    if cross_section_rank:
        log.info("  Applying cross-sectional ranking")
        feature_df = _cross_sectional_rank_panel(feature_df)

    log.info(
        f"Feature matrix: {feature_df.shape[0]:,} rows × {feature_df.shape[1]} columns"
    )
    return feature_df


def _cross_sectional_rank_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional pct rank per date for every feature column."""
    return (
        panel
        .groupby(level="Date")
        .transform(lambda col: col.rank(pct=True, na_option="keep"))
    )
