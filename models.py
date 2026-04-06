"""
src/models.py
─────────────
Model training and prediction for equity signal ranking.

Architecture
------------
  BaselineRankModel  : Ridge regression on rank targets
  BaselineClassifier : Logistic regression for top/bottom quintile
  XGBoostRanker      : Gradient boosted trees (pointwise rank regression)
  LightGBMRanker     : LightGBM alternative
  WalkForwardRunner  : Orchestrates all models across CV folds

Design choices
--------------
- Models are trained on cross-sectionally ranked features (already done
  in features.py) and cross-sectional rank targets.
- sklearn Pipelines include StandardScaler for linear models (important
  for regularisation) and Imputer for any remaining NaNs.
- XGBoost/LGBM handle NaN natively — no imputation needed.
- SHAP values computed on a held-out window for interpretability.
- No hyperparameter tuning is done inside the walk-forward loop to avoid
  leakage — parameters are set once in config.yaml.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.metrics import (
    evaluate_walk_forward,
    ic_summary,
    information_coefficient,
    rolling_ic,
    rolling_topk_spread,
)
from src.validation import WalkForwardSplitter

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Result Container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold: int
    model_name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    oos_scores: pd.Series          # (Date, Ticker) → predicted score
    oos_returns: pd.Series         # (Date, Ticker) → realized return
    ic: float = np.nan
    topk_spread: float = np.nan
    feature_importance: Optional[pd.Series] = None


@dataclass
class WalkForwardResults:
    model_name: str
    fold_results: list[FoldResult] = field(default_factory=list)

    @property
    def ic_series(self) -> pd.Series:
        """Per-fold mean IC as pd.Series."""
        return pd.Series(
            {f.fold: f.ic for f in self.fold_results},
            name="ic"
        )

    @property
    def all_oos_scores(self) -> pd.Series:
        """Concatenate all OOS scores across folds."""
        return pd.concat([f.oos_scores for f in self.fold_results]).sort_index()

    @property
    def all_oos_returns(self) -> pd.Series:
        return pd.concat([f.oos_returns for f in self.fold_results]).sort_index()

    def summary(self) -> dict:
        return ic_summary(self.ic_series)


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineRankModel:
    """
    Ridge regression on cross-sectional rank target.

    Baseline: if tree models cannot beat this, the non-linearity of XGBoost
    provides no value — a signal that should prompt feature re-examination.
    """

    def __init__(self, alphas: list[float] | None = None) -> None:
        alphas = alphas or cfg.get("models.ridge.alphas", [0.01, 0.1, 1.0, 10.0, 100.0])
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")),
        ])
        self.name = "Ridge (Baseline)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineRankModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    @property
    def best_alpha(self) -> float:
        return self.pipeline.named_steps["model"].alpha_


class BaselineClassifier:
    """
    Logistic regression for binary top/bottom quintile classification.
    Gives probability of being in the top quintile — used as ranking score.
    """

    def __init__(self, C_values: list[float] | None = None) -> None:
        C = C_values or [1.0]
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(
                C=C[0], max_iter=1000, solver="lbfgs", class_weight="balanced"
            )),
        ])
        self.name = "Logistic (Baseline)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineClassifier":
        """y should be binary top_label or bottom_label."""
        self.pipeline.fit(X, y)
        return self

    def predict_proba_positive(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of belonging to top quintile — used as ranking score."""
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba_positive(X)


class XGBoostRanker:
    """
    XGBoost gradient boosted trees for cross-sectional rank regression.

    Uses pointwise rank regression (obj=reg:squarederror on rank targets).
    Early stopping is applied on a 20% validation split from training data.
    """

    def __init__(self, params: dict | None = None) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost")

        p = params or {}
        self.model = xgb.XGBRegressor(
            objective         = p.get("objective",         "reg:squarederror"),
            n_estimators      = p.get("n_estimators",      300),
            max_depth         = p.get("max_depth",         4),
            learning_rate     = p.get("learning_rate",     0.05),
            subsample         = p.get("subsample",         0.8),
            colsample_bytree  = p.get("colsample_bytree",  0.8),
            min_child_weight  = p.get("min_child_weight",  20),
            early_stopping_rounds = p.get("early_stopping_rounds", 30),
            eval_metric       = p.get("eval_metric",       "rmse"),
            tree_method       = "hist",
            verbosity         = 0,
            random_state      = 42,
        )
        self.name = "XGBoost"
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRanker":
        self._feature_names = list(X.columns)

        # 20% validation split (time-sorted — take last 20% of train)
        n = len(X)
        n_val = max(1, int(n * 0.2))
        X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self._feature_names, name="importance").sort_values(ascending=False)

    def shap_values(self, X: pd.DataFrame, sample_n: int = 500) -> tuple:
        """
        Compute SHAP values for a sample of rows.
        Returns (shap_values, X_sample).
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        sample = X.sample(min(sample_n, len(X)), random_state=42)
        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(sample)
        return shap_vals, sample


class LightGBMRanker:
    """
    LightGBM alternative to XGBoost — generally faster, similar performance.
    Include both to compare and demonstrate ensemble thinking.
    """

    def __init__(self, params: dict | None = None) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("Install lightgbm: pip install lightgbm")

        p = params or {}
        self.model = lgb.LGBMRegressor(
            objective         = p.get("objective",     "regression"),
            n_estimators      = p.get("n_estimators",  300),
            max_depth         = p.get("max_depth",     4),
            learning_rate     = p.get("learning_rate", 0.05),
            subsample         = p.get("subsample",     0.8),
            colsample_bytree  = p.get("colsample_bytree", 0.8),
            min_child_samples = p.get("min_child_samples", 20),
            verbosity         = -1,
            random_state      = 42,
        )
        self.name = "LightGBM"
        self._feature_names: list[str] = []
        self._callbacks = None
        try:
            import lightgbm as lgb
            self._callbacks = [lgb.early_stopping(
                p.get("early_stopping_rounds", 30), verbose=False
            ), lgb.log_evaluation(-1)]
        except Exception:
            pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMRanker":
        self._feature_names = list(X.columns)
        n = len(X)
        n_val = max(1, int(n * 0.2))
        X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=self._callbacks,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self._feature_names, name="importance").sort_values(ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Runner
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardRunner:
    """
    Orchestrate walk-forward training and evaluation for all models.

    Usage
    -----
    >>> runner = WalkForwardRunner(X, y_rank, y_binary)
    >>> results = runner.run_all()
    >>> comparison = runner.compare_models(results)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y_rank: pd.Series,
        y_binary: pd.Series | None = None,
        splitter: WalkForwardSplitter | None = None,
    ) -> None:
        self.X        = X
        self.y_rank   = y_rank
        self.y_binary = y_binary
        self.splitter = splitter or WalkForwardSplitter(
            min_train_months = cfg.get("validation.min_train_months", 24),
            test_months      = cfg.get("validation.test_months", 3),
            gap_days         = cfg.get("validation.gap_days", 5),
            strategy         = cfg.get("validation.strategy", "expanding"),
        )

        # Validate alignment
        assert X.index.equals(y_rank.index), "X and y_rank index must match"

    def _get_dates(self) -> pd.DatetimeIndex:
        if isinstance(self.X.index, pd.MultiIndex):
            return pd.DatetimeIndex(self.X.index.get_level_values("Date").unique()).sort_values()
        return pd.DatetimeIndex(self.X.index.unique()).sort_values()

    def run_model(
        self,
        model_class,
        model_kwargs: dict | None = None,
        target: str = "rank",  # 'rank' or 'binary'
    ) -> WalkForwardResults:
        """
        Run a single model class across all walk-forward folds.

        Parameters
        ----------
        model_class  : One of the model classes defined above
        model_kwargs : Constructor arguments for the model
        target       : 'rank' for rank regression, 'binary' for classification
        """
        model_kwargs = model_kwargs or {}
        y = self.y_rank if target == "rank" else (self.y_binary or self.y_rank)

        results = WalkForwardResults(model_name=model_class(**model_kwargs).name)
        n_splits = self.splitter.n_splits(self.X)
        log.info(f"Running {results.model_name} over {n_splits} folds...")

        for split in self.splitter.split(self.X):
            # Slice data for this fold
            X_train = self.X.iloc[split.train_idx]
            y_train = y.iloc[split.train_idx]
            X_test  = self.X.iloc[split.test_idx]
            y_test  = self.y_rank.iloc[split.test_idx]   # always eval on rank

            # Drop NaN target rows from training
            valid_train = y_train.notna()
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]

            if len(X_train) < 100:
                log.warning(f"  Fold {split.fold}: insufficient training data ({len(X_train)} rows), skipping")
                continue

            # Instantiate fresh model per fold
            model = model_class(**model_kwargs)

            try:
                model.fit(X_train, y_train)
                scores = model.predict(X_test)
            except Exception as exc:
                log.error(f"  Fold {split.fold}: model failed — {exc}")
                continue

            oos_scores  = pd.Series(scores, index=X_test.index, name="score")
            oos_returns = y_test.rename("fwd_return")

            # IC per fold (mean over dates in the test window)
            ic_s = rolling_ic(oos_scores, oos_returns)
            fold_ic = ic_s.mean()

            # Feature importance (if available)
            feat_imp = None
            if hasattr(model, "feature_importance"):
                try:
                    feat_imp = model.feature_importance()
                except Exception:
                    pass

            fold_result = FoldResult(
                fold        = split.fold,
                model_name  = results.model_name,
                train_start = split.train_start,
                train_end   = split.train_end,
                test_start  = split.test_start,
                test_end    = split.test_end,
                oos_scores  = oos_scores,
                oos_returns = oos_returns,
                ic          = fold_ic,
                feature_importance = feat_imp,
            )
            results.fold_results.append(fold_result)

            log.info(f"  Fold {split.fold:02d} [{split.test_start.date()} → {split.test_end.date()}]: IC={fold_ic:.4f}")

        log.info(f"{results.model_name}: {results.summary()}")
        return results

    def run_all(self) -> dict[str, WalkForwardResults]:
        """Run Ridge, Logistic, XGBoost, and LightGBM models."""
        results: dict[str, WalkForwardResults] = {}

        # Baselines
        results["ridge"]    = self.run_model(BaselineRankModel, target="rank")
        if self.y_binary is not None:
            results["logistic"] = self.run_model(BaselineClassifier, target="binary")

        # Tree models
        try:
            xgb_params = cfg.load_config().get("models", {}).get("xgboost", {})
            results["xgboost"]  = self.run_model(XGBoostRanker, {"params": xgb_params}, target="rank")
        except Exception as exc:
            log.warning(f"XGBoost skipped: {exc}")

        try:
            lgb_params = cfg.load_config().get("models", {}).get("lightgbm", {})
            results["lightgbm"] = self.run_model(LightGBMRanker, {"params": lgb_params}, target="rank")
        except Exception as exc:
            log.warning(f"LightGBM skipped: {exc}")

        return results

    @staticmethod
    def compare_models(results: dict[str, WalkForwardResults]) -> pd.DataFrame:
        """
        Return a comparison table of model IC statistics across all folds.
        """
        rows = []
        for name, res in results.items():
            s = res.summary()
            s["model"] = name
            rows.append(s)
        df = pd.DataFrame(rows).set_index("model")
        return df.sort_values("icir", ascending=False)
