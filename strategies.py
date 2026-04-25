"""
Strategy interface for the CSI500 walk-forward framework.

A `Strategy` is anything that, given the full feature panel and an `as_of`
date, returns a portfolio (Series of weights indexed by stock_code) plus an
optional dict of diagnostics (IC, model params, ...).

The framework is responsible for:
  - building features once (so strategies don't pay that cost N times)
  - sweeping through as_of dates with no leakage
  - scoring the realized return of every portfolio against CSI500
  - aggregating results across windows

Strategies are responsible for:
  - using ONLY rows with date < as_of  (the framework will warn if violated)
  - using a training cutoff that respects FORWARD_HORIZON to avoid label leak
  - returning weights that satisfy the competition rules (>=30 names, <=10%,
    sum to 1).  `build_portfolio` from this module is the standard helper.

To add a new strategy, subclass `Strategy` (or write a function with the same
signature) and register it in `STRATEGIES` at the bottom of this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    training_frame, prediction_frame,
)

MIN_STOCKS = 30
MAX_WEIGHT = 0.10
DEFAULT_TOP_K = 50
DEFAULT_EMBARGO = 5  # >= FORWARD_HORIZON to keep train labels out of pred features


# ----------------------------------------------------------------------------
# Helpers shared by all strategies
# ----------------------------------------------------------------------------

def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged across dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Standard top-K rank-weighted portfolio with iterative 10% cap.

    Identical to the baseline.  Use this so portfolio construction is held
    constant when comparing different scoring models.
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


# ----------------------------------------------------------------------------
# Strategy interface
# ----------------------------------------------------------------------------

@dataclass
class StrategyResult:
    weights: pd.Series                 # index: stock_code, values sum to 1
    diagnostics: dict = field(default_factory=dict)  # e.g. {"val_ic": 0.07, ...}


class Strategy(Protocol):
    name: str

    def fit_predict(
        self,
        panel: pd.DataFrame,
        as_of: pd.Timestamp,
        top_k: int = DEFAULT_TOP_K,
    ) -> StrategyResult: ...


def _train_val_split(panel: pd.DataFrame, as_of: pd.Timestamp,
                     val_days: int, embargo: int):
    """Split training frame into train / embargo / val with no leakage.

    Layout:
        ... <- train ->  | <- embargo (discarded) -> | <- val (val_days) -> | as_of (excluded)
    Training rows have `date <= train_end`, val rows have `date >= val_start`,
    AND we cap the entire training pool at `as_of - embargo` so that targets
    (which look 5 days into the future) cannot peek past `as_of`.
    """
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - embargo)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())
    if len(all_dates) < val_days + embargo + 20:
        raise RuntimeError(
            f"Not enough history before {as_of.date()} for a clean train/val split "
            f"(have {len(all_dates)} dates, need >= {val_days + embargo + 20})"
        )
    val_start = pd.Timestamp(all_dates[-val_days])
    train_end = pd.Timestamp(all_dates[-(val_days + embargo + 1)])
    train_df = pool[pool["date"] <= train_end]
    val_df = pool[pool["date"] >= val_start]
    return train_df, val_df, train_end, val_start


# ----------------------------------------------------------------------------
# Reference strategy: XGBoost (mirrors baseline_xgboost.py)
# ----------------------------------------------------------------------------

@dataclass
class XGBStrategy:
    name: str = "xgb_baseline"
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 10.0
    reg_lambda: float = 1.0
    val_days: int = 10
    embargo: int = DEFAULT_EMBARGO
    seed: int = 0

    def _model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            tree_method="hist",
            n_jobs=-1,
            early_stopping_rounds=30,
            random_state=self.seed,
        )

    def fit_predict(self, panel, as_of, top_k=DEFAULT_TOP_K):
        train_df, val_df, train_end, val_start = _train_val_split(
            panel, as_of, self.val_days, self.embargo
        )

        model = self._model()
        model.fit(
            train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
            eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
            verbose=False,
        )

        val_pred = model.predict(val_df[FEATURE_COLUMNS])
        val_ic = rank_ic(
            val_df[TARGET_COLUMN].to_numpy(),
            val_pred,
            val_df["date"].to_numpy(),
        )

        pred_df = prediction_frame(panel, as_of=as_of)
        if pred_df.empty:
            raise RuntimeError(f"No prediction rows on {as_of.date()}")

        pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
        scores = pred_df.set_index("stock_code")["score"]
        weights = build_portfolio(scores, top_k=top_k)

        return StrategyResult(
            weights=weights,
            diagnostics={
                "val_ic": val_ic,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_pred": len(pred_df),
                "train_end": train_end.date().isoformat(),
                "val_start": val_start.date().isoformat(),
                "best_iter": int(getattr(model, "best_iteration", -1) or -1),
            },
        )


# Register strategies here so the CLI can look them up by --strategy <name>.
STRATEGIES: dict[str, Callable[[], Strategy]] = {
    "xgb_baseline": lambda: XGBStrategy(),
}
