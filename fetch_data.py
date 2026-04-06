"""
data/fetch_data.py
──────────────────
Fetch and cache all raw data needed by the pipeline:
  1. OHLCV price data        (yfinance)
  2. Basic fundamentals      (yfinance .info)
  3. Macro series            (FRED)
  4. SPY benchmark           (yfinance)

All outputs are written to data/raw/ as parquet files.
Run this once before any notebook or model training.

Usage
-----
    python data/fetch_data.py [--start 2012-01-01] [--end 2023-12-31]

Notes
-----
- yfinance rate-limits at ~2000 calls/hour. Batching + sleep handles this.
- Fundamentals from yfinance .info are POINT-IN-TIME UNSAFE — they reflect
  current values, not historical. Treat as a rough cross-sectional proxy only.
  For production, replace with EDGAR SEC filings (see notebooks/01_eda.ipynb).
- FRED data requires a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path so src/ imports work when run from data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = cfg.PROJECT_ROOT / cfg.get("data.raw_dir", "data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Price Data ─────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 50,
    sleep_sec: float = 2.0,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for all tickers.
    Returns a MultiIndex DataFrame: (Date, Ticker) → OHLCV columns.
    """
    log.info(f"Fetching prices for {len(tickers)} tickers ({start} → {end})")

    frames: list[pd.DataFrame] = []
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for i, batch in enumerate(batches):
        log.info(f"  Batch {i+1}/{len(batches)}: {batch[:5]}...")
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                log.warning(f"  Empty response for batch {i+1}")
                continue

            # yf returns MultiIndex columns (field, ticker) when multiple tickers
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.stack(level=1)  # → (Date, Ticker) index
                raw.index.names = ["Date", "Ticker"]
            else:
                # Single ticker — add ticker level
                ticker = batch[0]
                raw = raw.copy()
                raw["Ticker"] = ticker
                raw = raw.set_index(["Ticker"], append=True)
                raw.index.names = ["Date", "Ticker"]

            frames.append(raw)
        except Exception as exc:
            log.error(f"  Batch {i+1} failed: {exc}")

        if i < len(batches) - 1:
            time.sleep(sleep_sec)

    if not frames:
        raise RuntimeError("No price data fetched. Check tickers and date range.")

    prices = pd.concat(frames).sort_index()

    # Standardise column names
    prices.columns = [c.lower().replace(" ", "_") for c in prices.columns]
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in prices.columns]
    prices = prices[keep_cols]

    # Drop tickers with insufficient data (< 252 trading days)
    ticker_counts = prices.groupby("Ticker").size()
    valid_tickers = ticker_counts[ticker_counts >= 252].index
    prices = prices[prices.index.get_level_values("Ticker").isin(valid_tickers)]

    log.info(
        f"Prices fetched: {prices.index.get_level_values('Ticker').nunique()} tickers, "
        f"{prices.index.get_level_values('Date').nunique()} dates"
    )
    return prices


# ── 2. Fundamentals ───────────────────────────────────────────────────────────

FUNDAMENTAL_FIELDS = {
    "trailingPE":          "pe_ratio",
    "priceToBook":         "pb_ratio",
    "debtToEquity":        "debt_to_equity",
    "returnOnEquity":      "return_on_equity",
    "grossMargins":        "gross_margin",
    "revenueGrowth":       "revenue_growth_qoq",
    "earningsGrowth":      "earnings_growth",
    "currentRatio":        "current_ratio",
    "quickRatio":          "quick_ratio",
    "operatingMargins":    "operating_margin",
}


def fetch_fundamentals(tickers: list[str], sleep_sec: float = 0.5) -> pd.DataFrame:
    """
    Fetch static fundamentals from yfinance .info.

    IMPORTANT LIMITATION: These values are NOT point-in-time safe.
    yfinance returns current values, not historical snapshots.
    This is adequate for a portfolio project demonstrating methodology,
    but must be replaced with EDGAR / point-in-time data in production.
    """
    log.info(f"Fetching fundamentals for {len(tickers)} tickers (current snapshot)")

    records: list[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            row = {"Ticker": ticker}
            for yf_key, our_key in FUNDAMENTAL_FIELDS.items():
                row[our_key] = info.get(yf_key)
            records.append(row)
        except Exception as exc:
            log.warning(f"  {ticker}: {exc}")

        if (i + 1) % 20 == 0:
            log.info(f"  {i+1}/{len(tickers)} done")
            time.sleep(sleep_sec)

    df = pd.DataFrame(records).set_index("Ticker")

    # Clip extreme outliers (raw yfinance sometimes returns garbage values)
    for col in df.select_dtypes("number").columns:
        q_lo, q_hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=q_lo, upper=q_hi)

    log.info(f"Fundamentals: {df.shape[0]} tickers, {df.notna().sum().min()} min non-null per field")
    return df


# ── 3. Macro Data (FRED) ─────────────────────────────────────────────────────

def fetch_macro(start: str, end: str) -> pd.DataFrame:
    """
    Fetch macro series from FRED.
    Requires FRED_API_KEY in .env

    Returns daily DataFrame (forward-filled) with columns:
        fedfunds, t10y2y, vix, cpi_yoy, unrate
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    fred_key = cfg.fred_api_key()
    fred = Fred(api_key=fred_key)

    series_map = cfg.get("data.fred_series", {})
    frames: dict[str, pd.Series] = {}

    for our_name, fred_id in series_map.items():
        log.info(f"  FRED: {fred_id} → {our_name}")
        try:
            s = fred.get_series(fred_id, observation_start=start, observation_end=end)
            s.name = our_name
            frames[our_name] = s
        except Exception as exc:
            log.warning(f"  FRED {fred_id} failed: {exc}")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"

    # Compute CPI YoY if raw CPI was fetched
    if "cpi_yoy" in macro.columns:
        macro["cpi_yoy"] = macro["cpi_yoy"].pct_change(12) * 100

    # Reindex to daily and forward-fill (macro releases are monthly/weekly)
    date_range = pd.date_range(start=start, end=end, freq="B")
    macro = macro.reindex(date_range, method="ffill")
    macro.index.name = "Date"

    log.info(f"Macro: {macro.shape[1]} series, {macro.shape[0]} business days")
    return macro


# ── 4. Benchmark ─────────────────────────────────────────────────────────────

def fetch_benchmark(start: str, end: str) -> pd.DataFrame:
    """Download SPY as benchmark. Returns daily adjusted close."""
    log.info("Fetching SPY benchmark")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    spy = spy[["Close"]].rename(columns={"Close": "spy_close"})
    spy.index.name = "Date"
    spy["spy_return"] = spy["spy_close"].pct_change()
    return spy


# ── Main ──────────────────────────────────────────────────────────────────────

def main(start: str, end: str) -> None:
    tickers = cfg.tickers()

    if not tickers:
        # Fallback demo universe if config parsing fails
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "JNJ",
            "V", "PG", "UNH", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
            "AVGO", "COST", "MCD", "WMT", "BAC", "TMO", "CSCO", "ACN", "ABT",
            "DHR", "TXN", "NEE", "LIN", "PM", "CRM", "ADBE", "NKE", "RTX",
            "HON", "QCOM", "UPS", "AMGN", "SBUX", "IBM", "GE", "CAT", "BA",
            "GS", "MS", "BLK", "AXP", "MDT",
        ]

    log.info(f"Universe: {len(tickers)} tickers")

    # Prices
    prices = fetch_prices(tickers, start=start, end=end)
    out = RAW_DIR / "prices.parquet"
    prices.to_parquet(out)
    log.info(f"Saved → {out}")

    # Fundamentals
    try:
        fundam = fetch_fundamentals(tickers)
        out = RAW_DIR / "fundamentals.parquet"
        fundam.to_parquet(out)
        log.info(f"Saved → {out}")
    except Exception as exc:
        log.warning(f"Fundamentals skipped: {exc}")

    # Macro
    try:
        macro = fetch_macro(start=start, end=end)
        out = RAW_DIR / "macro.parquet"
        macro.to_parquet(out)
        log.info(f"Saved → {out}")
    except EnvironmentError as exc:
        log.warning(f"Macro skipped (no FRED key): {exc}")
    except Exception as exc:
        log.warning(f"Macro skipped: {exc}")

    # Benchmark
    benchmark = fetch_benchmark(start=start, end=end)
    out = RAW_DIR / "benchmark.parquet"
    benchmark.to_parquet(out)
    log.info(f"Saved → {out}")

    log.info("Data fetch complete. Run notebooks in order: 01 → 02 → 03 → 04 → 05")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw equity signal data")
    parser.add_argument("--start", default=cfg.get("universe.start_date", "2012-01-01"))
    parser.add_argument("--end",   default=cfg.get("universe.end_date",   "2023-12-31"))
    args = parser.parse_args()
    main(args.start, args.end)
