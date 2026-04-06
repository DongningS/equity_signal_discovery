# Cross-Sectional Signal Discovery in Equity Markets

> A disciplined, reproducible framework for discovering and evaluating predictive signals for stock ranking — built on public data, honest about assumptions, focused on methodology over performance claims.

---

## Table of Contents

1. [Conceptual Foundations](#conceptual-foundations)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Data Sources & Limitations](#data-sources--limitations)
5. [Methodology](#methodology)
6. [Evaluation Framework](#evaluation-framework)
7. [Key Results (Expected Range)](#key-results-expected-range)
8. [Limitations](#limitations)
9. [Running the Tests](#running-the-tests)

---

## Conceptual Foundations

Before running any code, these three ideas are worth understanding clearly.
They explain every design decision in this project.

---

### What is a Signal?

A **signal** is a feature that has statistically reliable, repeatable correlation with future asset returns — distinguished from noise by its *consistency across time periods and market regimes*, not by its magnitude in any single window.

The equity market produces an enormous amount of data. Almost all of it is noise. A feature that shows IC = 0.08 in one year and IC = -0.02 in the next is not a signal — it is an artifact of in-sample fitting. A feature that shows IC = 0.03 consistently across 40 rolling quarters is a signal, even though it looks unimpressive in isolation.

Signal discovery is fundamentally about separating small, persistent effects from large, random ones.

---

### Why Ranking Over Prediction?

Predicting the absolute return of a stock is effectively impossible with public data. The information required — earnings surprises, deal announcements, macro shocks — is either not available ahead of time or is already priced in.

**Ranking is a weaker, more achievable goal.** Instead of asking "will AAPL return +8% next month?", we ask "will AAPL likely outperform the average stock in my universe next month?"

This reframing matters for several reasons:

- Cross-sectional ranking removes common market factors (the "tide that lifts all boats") and focuses on relative performance
- Ranking targets are more stationary than absolute return targets
- A model that correctly ranks 55% of pairs is potentially useful for portfolio construction, even though it "predicts" almost nothing about absolute returns
- Most institutional portfolio construction (factor investing, systematic equity) is fundamentally a ranking problem

This project uses **cross-sectional percentile rank of forward returns** as the primary target.

---

### Why Robustness Over Peak Performance?

In-sample performance is close to meaningless in financial machine learning. A model that achieves IC = 0.15 by fitting to the training window will almost always revert toward IC ≈ 0 out of sample.

**Robustness is measured by ICIR (IC Information Ratio):**

```
ICIR = Mean(IC across periods) / Std(IC across periods)
```

A model with IC = 0.05 every single period (ICIR ≈ ∞) is infinitely more valuable than a model with IC = 0.20 in three periods and IC = -0.10 in the other three (ICIR ≈ 1.0, unreliable).

This project explicitly optimises for:
1. Walk-forward OOS consistency (not in-sample fit)
2. ICIR > mean IC as the primary quality metric
3. Conservative model complexity (max_depth=4, min_child_weight=20)
4. No hyperparameter search inside the CV loop

The goal is a signal that a quant researcher would be willing to trade on for the next 12 months — not one that looked good in a backtest.

---

## Project Structure

```
equity-signal-discovery/
├── data/
│   ├── raw/                    # gitignored — fetched by fetch_data.py
│   ├── processed/              # parquet feature and target files
│   └── fetch_data.py           # data download script
├── notebooks/
│   ├── 01_eda.ipynb            # universe + return distribution
│   ├── 02_feature_eng.ipynb    # signal construction + individual IC
│   ├── 03_baseline_models.ipynb # ridge + logistic baselines
│   ├── 04_tree_models.ipynb    # xgboost + lightgbm + shap
│   ├── 05_evaluation.ipynb     # walk-forward IC, spread, hit rate
│   └── 06_backtest_lite.ipynb  # gross pre-cost portfolio simulation
├── src/
│   ├── config.py               # config loader (all params from config.yaml)
│   ├── features.py             # all feature engineering functions
│   ├── targets.py              # forward return and rank targets
│   ├── validation.py           # walk-forward CV splitter
│   ├── metrics.py              # IC, ICIR, top-k spread, hit rate
│   ├── models.py               # Ridge, Logistic, XGBoost, LightGBM + runner
│   ├── backtest.py             # simple quintile backtest simulation
│   └── plotting.py             # all visualisation functions
├── tests/
│   ├── test_features.py        # no look-ahead, feature correctness
│   └── test_metrics.py         # IC, spread, and stability metric tests
├── config.yaml                 # all tuneable parameters
├── requirements.txt
└── .env.example                # FRED_API_KEY template
```

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/yourusername/equity-signal-discovery
cd equity-signal-discovery

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your FRED API key
# Free key: https://fred.stlouisfed.org/docs/api/api_key.html
```

### 3. Fetch Data

```bash
python data/fetch_data.py --start 2012-01-01 --end 2023-12-31
```

This fetches:
- Adjusted OHLCV prices for ~50 liquid US equities (yfinance)
- Static fundamentals snapshot (yfinance — see limitations)
- Macro series: Fed Funds, yield spread, VIX, CPI, unemployment (FRED)
- SPY benchmark (yfinance)

Expected runtime: 5–15 minutes depending on rate limiting.

### 4. Run Notebooks in Order

```
01_eda.ipynb            → understand the data
02_feature_eng.ipynb    → compute and inspect features
03_baseline_models.ipynb → establish linear baselines
04_tree_models.ipynb    → tree models + SHAP
05_evaluation.ipynb     → THE CORE — walk-forward IC evaluation
06_backtest_lite.ipynb  → optional gross portfolio simulation
```

### 5. Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## Data Sources & Limitations

| Source | Data | Limitation |
|--------|------|-----------|
| yfinance | OHLCV prices, basic fundamentals | **Survivorship bias** — static ticker list. Delistings not reflected. Fundamentals are current snapshots, not historical. |
| FRED | Macro: Fed Funds, VIX, CPI, yield spread, unemployment | Monthly/weekly frequency. Requires free API key. |
| Yahoo RSS | News headlines for sentiment proxy | **Not historical.** Demonstrates methodology only. Real use requires RavenPack/GDELT archive. |

### Survivorship Bias (Important)

The universe is a static list of S&P 500 constituents. Companies that were added to the index post-2012 are included from 2012 if they were publicly traded. Companies that were delisted during the period are excluded. This **inflates apparent returns** and should be clearly stated in any write-up.

A production system would use a point-in-time index membership database.

### Fundamental Data (Important)

`yfinance.info` returns **current** fundamental values, not historical ones. A stock's current P/E is used as a proxy for its historical P/E. This is a significant simplification. The 90-day lag applied in `features.py` partially mitigates look-ahead but does not fully address the point-in-time issue.

Production-grade alternative: SEC EDGAR API with quarterly filing dates, or Compustat point-in-time database.

---

## Methodology

### Feature Engineering

All features are computed in a point-in-time safe manner:
- Price-based features use only data available at time t
- Fundamental features are lagged by 90 days (approximate filing delay)
- Macro features are lagged by 30 days
- All features are cross-sectionally ranked (percentile) to make them comparable across stocks and robust to outliers

| Feature | Description | Lag |
|---------|-------------|-----|
| `mom_21d` | 21-day momentum (t-26 to t-6) | none |
| `mom_63d` | 63-day momentum | none |
| `mom_252d` | 12-month momentum (skip last 5d) | none |
| `rev_5d` | 5-day short-term reversal | none |
| `vol_21d` | 21-day annualised realised vol | none |
| `vol_63d` | 63-day annualised realised vol | none |
| `atr_ratio` | ATR normalised by price | none |
| `rsi_14` | RSI (Wilder, 14-period) | none |
| `price_sma_50` | Price / 50-day SMA ratio | none |
| `price_sma_200` | Price / 200-day SMA ratio | none |
| `bb_position` | Bollinger Band position [0,1] | none |
| `pe_ratio` | Trailing P/E ratio | 90 days |
| `pb_ratio` | Price-to-book | 90 days |
| `debt_to_equity` | D/E ratio | 90 days |
| `return_on_equity` | ROE | 90 days |
| `gross_margin` | Gross margin % | 90 days |
| `revenue_growth_qoq` | Revenue growth QoQ | 90 days |
| `fedfunds` | Federal Funds Rate | 30 days |
| `t10y2y` | 10yr-2yr yield spread | 30 days |
| `vix` | VIX level | 30 days |
| `cpi_yoy` | CPI year-over-year | 30 days |

### Target Variable

**Primary target:** Cross-sectional percentile rank of 21-day forward log return.

```python
rank_target = forward_return.rank(axis=1, pct=True)  # per date, per stock
```

Higher rank (closer to 1.0) = better relative return within the universe that month.

Binary classification target (for logistic baseline): top quintile (rank > 0.80) vs all others.

### Models

**Baseline: Ridge Regression**
Linear model on rank target. Uses RidgeCV for alpha selection. Establishes the floor — tree models should beat this if non-linear interactions matter.

**Baseline: Logistic Regression**
Binary classifier (top quintile). Probability of top-quintile membership used as ranking score.

**Primary: XGBoost**
Pointwise rank regression (`reg:squarederror` on rank target). Conservative depth (4) and `min_child_weight=20` to prevent overfitting on small cross-sections. Early stopping on held-out 20% of training period.

**Alternative: LightGBM**
Identical structure to XGBoost. Generally faster. Comparison between the two gives a free robustness check.

### Walk-Forward Validation

Expanding window training with 3-month OOS test periods. 5-day gap between train end and test start.

```
Train: [2012-01 → 2013-12]  Test: [2014-01 → 2014-03]
Train: [2012-01 → 2014-03]  Test: [2014-04 → 2014-06]
Train: [2012-01 → 2014-06]  Test: [2014-07 → 2014-09]
...
```

No hyperparameter tuning inside the loop. Parameters set once in `config.yaml`.

---

## Evaluation Framework

The hierarchy of metrics, ordered by importance:

### 1. ICIR (Primary — stability)

```
ICIR = mean(IC) / std(IC)  across rolling periods
```

Target: ICIR > 0.4 across all OOS folds. This indicates the signal is reliable enough to trust in future periods.

### 2. Mean IC (Magnitude)

Spearman rank correlation between model scores and realised forward returns, averaged across all OOS dates.

Target on public data: IC ∈ [0.02, 0.06]. Higher values with public data should be investigated for leakage.

### 3. Top-K Spread

Average return of top-20% minus bottom-20% ranked stocks, per month. Viewed as directional quality check only — no transaction costs.

### 4. Hit Rate

Fraction of months where top quintile outperforms bottom quintile. Target: > 52%.

---

## Key Results (Expected Range)

These are the realistic outcomes a careful implementation should produce. Numbers outside this range in either direction warrant investigation.

| Metric | Expected Range | If Higher | If Lower |
|--------|---------------|-----------|----------|
| Mean OOS IC | 0.02 – 0.06 | Check for leakage | Normal — most public signals are weak |
| ICIR | 0.3 – 0.8 | Surprisingly stable — double-check CV | Signal is noisy — expected |
| Hit rate | 50% – 58% | Suspiciously consistent | Expected in high-volatility periods |
| Top-K spread | +0.1% – +1.0%/mo (gross) | Check survivorship bias | Possible in low-dispersion regimes |

XGBoost will likely show modest improvement over Ridge in mean IC but similar or lower ICIR — an instructive finding about model complexity versus signal stability.

Sentiment proxy IC will likely be near zero — this is the expected and honest result.

---

## Limitations

These are not afterthoughts. They define the boundary of what this project claims.

**1. Survivorship Bias**
The static ticker universe excludes all companies that were delisted, merged, or removed from the index during 2012–2023. This inflates universe quality and makes returns look better than they are.

**2. No Transaction Costs**
All spreads and backtest returns are gross of any costs. Real-world costs (bid-ask, market impact, rebalancing turnover) would reduce — and potentially eliminate — the apparent edge.

**3. Fundamental Data Not Point-in-Time**
`yfinance.info` returns current fundamental values. Historical fundamentals from a proper point-in-time database (Compustat, EDGAR with filing dates) would likely reduce IC from fundamental features.

**4. Small Universe**
50–100 stocks is a very small cross-section for a ranking model. IC estimates have high variance with small N. A production system would use 500–3000 stocks.

**5. Feature Selection Bias**
The features included were chosen based on well-known academic literature. Even so, testing 20+ features on the same dataset introduces mild data-mining bias that the walk-forward structure does not fully eliminate.

**6. No Short-Selling**
The backtest is long-only. The bottom-quintile information (stocks to avoid or short) is not captured in performance.

---

## Running the Tests

Tests validate two critical properties:

1. **No look-ahead leakage**: features at time t do not use data from t+1 or later
2. **Metric correctness**: IC, ICIR, spread, and hit rate compute correctly on known inputs

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only feature tests
pytest tests/test_features.py -v

# Run only metric tests
pytest tests/test_metrics.py -v
```

---

## Environment Variables

```bash
# .env.example
FRED_API_KEY=your_fred_api_key_here
```

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

---

## Academic References

The features and evaluation framework in this project are grounded in published research:

- Fama & French (1992) — Size and value factors
- Jegadeesh & Titman (1993) — Momentum in stock returns
- Asness, Moskowitz & Pedersen (2013) — Value and momentum everywhere
- Grinold & Kahn (2000) — *Active Portfolio Management* — IC, ICIR framework
- López de Prado (2018) — *Advances in Financial Machine Learning* — walk-forward CV, feature importance

---

## Licence

MIT. Use freely for learning and portfolio projects. Not financial advice.
