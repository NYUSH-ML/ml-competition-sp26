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
    build_features as build_features_v1,
)
import features_v2

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

    def build_panel(
        self,
        prices: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the feature panel this strategy needs.  Done once per run."""
        ...

    def fit_predict(
        self,
        panel: pd.DataFrame,
        as_of: pd.Timestamp,
        top_k: int = DEFAULT_TOP_K,
    ) -> StrategyResult: ...


def _train_val_split(panel: pd.DataFrame, as_of: pd.Timestamp,
                     val_days: int, embargo: int,
                     training_frame_fn=training_frame):
    """Split training frame into train / embargo / val with no leakage.

    Layout:
        ... <- train ->  | <- embargo (discarded) -> | <- val (val_days) -> | as_of (excluded)
    Training rows have `date <= train_end`, val rows have `date >= val_start`,
    AND we cap the entire training pool at `as_of - embargo` so that targets
    (which look 5 days into the future) cannot peek past `as_of`.

    `training_frame_fn` lets v1 / v2 strategies plug in their own NA-drop logic.
    """
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - embargo)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    pool = training_frame_fn(panel, max_date=train_cutoff)
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

    # Subclasses override these three to switch feature sets.
    feature_columns: tuple = tuple(FEATURE_COLUMNS)
    target_column: str = TARGET_COLUMN

    def build_panel(self, prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        return build_features_v1(prices)

    def _training_frame_fn(self):
        return training_frame

    def _prediction_frame_fn(self):
        return prediction_frame

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
        feats = list(self.feature_columns)
        tgt = self.target_column
        train_df, val_df, train_end, val_start = _train_val_split(
            panel, as_of, self.val_days, self.embargo,
            training_frame_fn=self._training_frame_fn(),
        )

        model = self._model()
        model.fit(
            train_df[feats], train_df[tgt],
            eval_set=[(val_df[feats], val_df[tgt])],
            verbose=False,
        )

        val_pred = model.predict(val_df[feats])
        val_ic = rank_ic(
            val_df[tgt].to_numpy(),
            val_pred,
            val_df["date"].to_numpy(),
        )

        pred_df = self._prediction_frame_fn()(panel, as_of=as_of)
        if pred_df.empty:
            raise RuntimeError(f"No prediction rows on {as_of.date()}")

        pred_df = pred_df.assign(score=model.predict(pred_df[feats]))
        scores = pred_df.set_index("stock_code")["score"]
        weights = build_portfolio(scores, top_k=top_k)

        return StrategyResult(
            weights=weights,
            diagnostics={
                "val_ic": val_ic,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_pred": len(pred_df),
                "n_features": len(feats),
                "train_end": train_end.date().isoformat(),
                "val_start": val_start.date().isoformat(),
                "best_iter": int(getattr(model, "best_iteration", -1) or -1),
            },
        )


# ----------------------------------------------------------------------------
# v2: same XGBoost recipe over the richer feature set in features_v2.py
# ----------------------------------------------------------------------------

@dataclass
class XGBStrategyV2(XGBStrategy):
    name: str = "xgb_v2"
    feature_columns: tuple = tuple(features_v2.ALL_FEATURES)
    target_column: str = features_v2.TARGET_COLUMN
    # A bit more capacity for the wider feature set
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.04
    colsample_bytree: float = 0.7

    def build_panel(self, prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        return features_v2.build_features(prices, index_df)

    def _training_frame_fn(self):
        return features_v2.training_frame

    def _prediction_frame_fn(self):
        return features_v2.prediction_frame


# ----------------------------------------------------------------------------
# LambdaRank: optimise pairwise ordering directly (objective rank:pairwise)
# ----------------------------------------------------------------------------

@dataclass
class XGBRankerStrategy(XGBStrategyV2):
    """LambdaRank-style XGBoost over the v2 feature set.

    Key differences from regression:
      - objective is `rank:pairwise`: model learns *ordering* of stocks within
        each day (one query = one trading day), which is exactly what a
        top-K portfolio cares about.
      - Train rows are sorted by date and we pass `group=` = rows-per-day so
        pairs only form within the same date.
      - Target is the raw 5d forward return (rank:pairwise only uses ordering,
        so the magnitude does not affect loss).
    """
    name: str = "xgb_ranker_v2"
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    colsample_bytree: float = 0.7

    def _ranker(self) -> "xgb.XGBRanker":
        return xgb.XGBRanker(
            objective="rank:pairwise",
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
        feats = list(self.feature_columns)
        tgt = self.target_column
        train_df, val_df, train_end, val_start = _train_val_split(
            panel, as_of, self.val_days, self.embargo,
            training_frame_fn=self._training_frame_fn(),
        )

        # Pairwise loss requires rows grouped by query (here: by trading day).
        train_df = train_df.sort_values("date")
        val_df = val_df.sort_values("date")
        group_train = train_df.groupby("date").size().to_numpy()
        group_val = val_df.groupby("date").size().to_numpy()

        model = self._ranker()
        model.fit(
            train_df[feats], train_df[tgt],
            group=group_train,
            eval_set=[(val_df[feats], val_df[tgt])],
            eval_group=[group_val],
            verbose=False,
        )

        val_pred = model.predict(val_df[feats])
        val_ic = rank_ic(
            val_df[tgt].to_numpy(), val_pred, val_df["date"].to_numpy()
        )

        pred_df = self._prediction_frame_fn()(panel, as_of=as_of)
        if pred_df.empty:
            raise RuntimeError(f"No prediction rows on {as_of.date()}")

        pred_df = pred_df.assign(score=model.predict(pred_df[feats]))
        scores = pred_df.set_index("stock_code")["score"]
        weights = build_portfolio(scores, top_k=top_k)

        return StrategyResult(
            weights=weights,
            diagnostics={
                "val_ic": val_ic,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_pred": len(pred_df),
                "n_features": len(feats),
                "train_end": train_end.date().isoformat(),
                "val_start": val_start.date().isoformat(),
                "best_iter": int(getattr(model, "best_iteration", -1) or -1),
                "objective": "rank:pairwise",
            },
        )


# Register strategies here so the CLI can look them up by --strategy <name>.
STRATEGIES: dict[str, Callable[[], Strategy]] = {
    "xgb_baseline": lambda: XGBStrategy(),
    "xgb_v2": lambda: XGBStrategyV2(),
    "xgb_ranker_v2": lambda: XGBRankerStrategy(),
}
