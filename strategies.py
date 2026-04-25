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
import lightgbm as lgb
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


def cluster_neutralize_scores(
    scores: pd.Series,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    n_clusters: int = 10,
    lookback_days: int = 60,
    seed: int = 42,
) -> pd.Series:
    """De-mean each stock's score by its cluster mean.

    We don't have a pre-computed sector map, so we cluster stocks by their
    realised return pattern in the trailing window before `as_of`.  Stocks
    that move together end up in the same bucket; subtracting the bucket
    mean removes systematic over-/under-pricing of any single style or
    sector so the model's rank signal can't lever the whole portfolio onto
    one factor.

    Returns a new Series with the same index as `scores`; if there is not
    enough history to fit clusters we return the original scores unchanged.
    """
    from sklearn.cluster import KMeans

    end = pd.Timestamp(as_of)
    start = end - pd.Timedelta(days=lookback_days * 2)  # generous wall-clock
    win = prices[(prices["date"] < end) & (prices["date"] >= start)]
    if win.empty:
        return scores

    # Build (stocks x days) returns matrix.  Use pct_chg if present else compute.
    if "pct_chg" in win.columns:
        ret_col = "pct_chg"
    else:
        win = win.sort_values(["stock_code", "date"]).copy()
        win["pct_chg"] = win.groupby("stock_code")["close"].pct_change() * 100.0
        ret_col = "pct_chg"

    pivot = (
        win.pivot_table(index="stock_code", columns="date", values=ret_col)
        .dropna(axis=0, thresh=int(lookback_days * 0.7))
        .fillna(0.0)
    )
    if len(pivot) < n_clusters * 3:
        return scores

    common = pivot.index.intersection(scores.index)
    if len(common) < n_clusters * 3:
        return scores

    X = pivot.loc[common].to_numpy()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = pd.Series(km.fit_predict(X), index=common, name="cluster")

    # Demean scores within each cluster.
    s = scores.copy()
    s_in = s.loc[common]
    means = s_in.groupby(labels).transform("mean")
    s.loc[common] = s_in - means
    return s


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

    def fit_predict_scores(self, panel, as_of):
        """Train and return (scores: Series[stock_code -> float], diag: dict).

        Pure prediction step — no portfolio construction.  Used both by
        `fit_predict` (which then calls build_portfolio) and by the ensemble
        strategy (which rank-averages scores from multiple base learners).
        """
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
        diag = {
            "val_ic": val_ic,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_pred": len(pred_df),
            "n_features": len(feats),
            "train_end": train_end.date().isoformat(),
            "val_start": val_start.date().isoformat(),
            "best_iter": int(getattr(model, "best_iteration", -1) or -1),
        }
        return scores, diag

    def fit_predict(self, panel, as_of, top_k=DEFAULT_TOP_K):
        scores, diag = self.fit_predict_scores(panel, as_of)
        weights = build_portfolio(scores, top_k=top_k)
        return StrategyResult(weights=weights, diagnostics=diag)


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
        # Bind self.target_column so multi-target ensembles can train each
        # member on its own forward target without polluting the shared API.
        from functools import partial
        return partial(features_v2.training_frame, target=self.target_column)

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

        # Pairwise loss requires rows grouped by query (= trading day) AND
        # XGBoost 2.x demands non-negative integer "relevance degree" labels
        # in [0, 31] when NDCG with exponential gain is the eval metric.
        # Bin the continuous 5d-forward-return target into 32 quantile buckets
        # PER DAY -> always satisfies the [0, 31] constraint while preserving
        # intra-query ordering (which is all pairwise/NDCG losses care about).
        N_BUCKETS = 32
        def _bucketize(g):
            r = g.rank(method="first", ascending=True) - 1  # 0..n-1
            n = len(g)
            return (r * N_BUCKETS // n).clip(upper=N_BUCKETS - 1).astype(int)

        train_df = train_df.sort_values(["date", "stock_code"]).copy()
        val_df = val_df.sort_values(["date", "stock_code"]).copy()
        train_df["_rel"] = train_df.groupby("date")[tgt].transform(_bucketize)
        val_df["_rel"] = val_df.groupby("date")[tgt].transform(_bucketize)
        group_train = train_df.groupby("date").size().to_numpy()
        group_val = val_df.groupby("date").size().to_numpy()

        model = self._ranker()
        model.fit(
            train_df[feats], train_df["_rel"],
            group=group_train,
            eval_set=[(val_df[feats], val_df["_rel"])],
            eval_group=[group_val],
            verbose=False,
        )

        val_pred = model.predict(val_df[feats])
        # Score with rank IC against the *real* continuous return so the
        # number is comparable to regression strategies.
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


# ----------------------------------------------------------------------------
# LightGBM regression over v2 features — second base learner for ensembling
# ----------------------------------------------------------------------------

@dataclass
class LGBStrategyV2:
    """LightGBM-based v2 strategy.  Same train/val split & target as XGBStrategyV2,
    but a different boosting library + a different default tree shape so it makes
    a useful ensemble partner (different bias).
    """
    name: str = "lgb_v2"
    feature_columns: tuple = tuple(features_v2.ALL_FEATURES)
    target_column: str = features_v2.TARGET_COLUMN

    n_estimators: int = 800
    num_leaves: int = 64           # leaf-wise growth -> different shape from xgb
    max_depth: int = -1
    learning_rate: float = 0.03
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_data_in_leaf: int = 80
    reg_lambda: float = 1.0
    seed: int = 42
    val_days: int = 30
    embargo: int = DEFAULT_EMBARGO

    def build_panel(self, prices, index_df):
        return features_v2.build_features(prices, index_df)

    def _training_frame_fn(self):
        return features_v2.training_frame

    def _prediction_frame_fn(self):
        return features_v2.prediction_frame

    def _model(self):
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            min_data_in_leaf=self.min_data_in_leaf,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit_predict_scores(self, panel, as_of):
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
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(0)],
        )

        val_pred = model.predict(val_df[feats])
        val_ic = rank_ic(val_df[tgt].to_numpy(), val_pred, val_df["date"].to_numpy())

        pred_df = self._prediction_frame_fn()(panel, as_of=as_of)
        if pred_df.empty:
            raise RuntimeError(f"No prediction rows on {as_of.date()}")
        pred_df = pred_df.assign(score=model.predict(pred_df[feats]))
        scores = pred_df.set_index("stock_code")["score"]
        diag = {
            "val_ic": val_ic,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_pred": len(pred_df),
            "n_features": len(feats),
            "train_end": train_end.date().isoformat(),
            "val_start": val_start.date().isoformat(),
            "best_iter": int(getattr(model, "best_iteration_", -1) or -1),
        }
        return scores, diag

    def fit_predict(self, panel, as_of, top_k=DEFAULT_TOP_K):
        scores, diag = self.fit_predict_scores(panel, as_of)
        weights = build_portfolio(scores, top_k=top_k)
        return StrategyResult(weights=weights, diagnostics=diag)


# ----------------------------------------------------------------------------
# Ensemble: rank-average of multiple base learners
# ----------------------------------------------------------------------------

@dataclass
class EnsembleStrategy:
    """Train each base learner independently on the same train/val split,
    convert each model's CSI500-cross-section predictions into pct-ranks
    (scale-free), average those ranks (optionally weighted), and feed the
    averaged ranks into the standard top-K portfolio builder.

    Why rank-average vs score-average?  Different boosters return scores on
    different magnitude scales (XGB regression target, LGB target, ranker
    logits).  Ranks normalise that out and are robust to outlier predictions.
    """
    name: str = "ensemble_v2"
    members: tuple = field(default_factory=lambda: (
        XGBStrategyV2(), LGBStrategyV2(),
    ))
    weights: tuple = field(default_factory=lambda: (1.0, 1.0))

    def build_panel(self, prices, index_df):
        # All current members use the v2 feature set, so build it once.
        return features_v2.build_features(prices, index_df)

    def fit_predict(self, panel, as_of, top_k=DEFAULT_TOP_K):
        if len(self.members) != len(self.weights):
            raise ValueError("members and weights must be same length")

        wsum = float(sum(self.weights))
        per_member_ranks: list[pd.Series] = []
        per_member_diag: dict[str, dict] = {}
        member_ics: list[float] = []

        for m in self.members:
            scores, diag = m.fit_predict_scores(panel, as_of)
            # Convert each member's raw scores into pct-ranks (0..1) across
            # the SAME prediction-day cross-section, so a name with prob 0.85
            # of being top-decile gets the same magnitude regardless of which
            # booster produced it.
            ranks = scores.rank(method="average", pct=True)
            per_member_ranks.append(ranks)
            per_member_diag[m.name] = diag
            ic = diag.get("val_ic")
            if ic is not None and not (isinstance(ic, float) and np.isnan(ic)):
                member_ics.append(float(ic))

        # Weighted average rank.  Outer-join the indexes so a stock missing
        # from one member's prediction set just contributes zero from that
        # member (effectively penalising it).
        all_codes = sorted(set().union(*[r.index for r in per_member_ranks]))
        agg = pd.Series(0.0, index=all_codes)
        for w, r in zip(self.weights, per_member_ranks):
            agg = agg.add(r.reindex(all_codes).fillna(0.0) * w, fill_value=0.0)
        agg = agg / wsum

        weights = build_portfolio(agg, top_k=top_k)

        diag = {
            "n_members": len(self.members),
            "members": [m.name for m in self.members],
            "member_weights": list(self.weights),
            "mean_member_ic": float(np.mean(member_ics)) if member_ics else float("nan"),
            # Surface as `val_ic` so walkforward report shows it.
            "val_ic": float(np.mean(member_ics)) if member_ics else float("nan"),
        }
        for n, d in per_member_diag.items():
            for k, v in d.items():
                diag[f"{n}.{k}"] = v
        return StrategyResult(weights=weights, diagnostics=diag)


# ----------------------------------------------------------------------------
# Cluster-neutralised wrapper: same underlying scores, but de-meaned by
# return-correlation cluster before portfolio construction.
# ----------------------------------------------------------------------------

@dataclass
class ClusterNeutralWrapper:
    """Wraps a base strategy and neutralises its raw scores against
    return-correlation clusters before building the top-K portfolio.

    Uses the *panel itself* as the price source (since the v2 panel already
    has stock_code/date/pct_chg).  This keeps the wrapper drop-in
    compatible with the walk-forward framework.
    """
    name: str = "xgb_v2_neutral"
    base: "Strategy" = field(default_factory=lambda: XGBStrategyV2())
    n_clusters: int = 10
    lookback_days: int = 60

    def build_panel(self, prices, index_df):
        return self.base.build_panel(prices, index_df)

    def fit_predict(self, panel, as_of, top_k=DEFAULT_TOP_K):
        scores, diag = self.base.fit_predict_scores(panel, as_of)
        # Reuse the panel as the price source — has stock_code/date/pct_chg.
        cols = ["stock_code", "date"]
        ret_col = "pct_chg" if "pct_chg" in panel.columns else None
        if ret_col is None:
            # Fallback: derive from close.
            prices = panel[["stock_code", "date", "close"]].copy()
        else:
            prices = panel[cols + [ret_col]].copy()
        neutralised = cluster_neutralize_scores(
            scores, prices, as_of,
            n_clusters=self.n_clusters,
            lookback_days=self.lookback_days,
        )
        weights = build_portfolio(neutralised, top_k=top_k)
        diag = {**diag,
                "neutral_n_clusters": self.n_clusters,
                "neutral_lookback": self.lookback_days,
                "neutral_n_neutralised": int(neutralised.notna().sum())}
        return StrategyResult(weights=weights, diagnostics=diag)


# Register strategies here so the CLI can look them up by --strategy <name>.
STRATEGIES: dict[str, Callable[[], Strategy]] = {
    "xgb_baseline": lambda: XGBStrategy(),
    "xgb_v2": lambda: XGBStrategyV2(),
    "xgb_ranker_v2": lambda: XGBRankerStrategy(),
    "lgb_v2": lambda: LGBStrategyV2(),
    "ensemble_v2": lambda: EnsembleStrategy(),
    # Multi-seed bagging of xgb_v2: same model, different random seeds.
    # Lowers variance without sacrificing mean, since each member has ~equal
    # alpha but different idiosyncratic noise.
    "xgb_v2_bag": lambda: EnsembleStrategy(
        name="xgb_v2_bag",
        members=(
            XGBStrategyV2(seed=42),
            XGBStrategyV2(seed=7),
            XGBStrategyV2(seed=123),
            XGBStrategyV2(seed=2026),
            XGBStrategyV2(seed=314),
        ),
        weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    # XGB-heavy ensemble: 3 xgb seeds + 1 lgb -> retains LGB's diversity but
    # lets the (better) xgb dominate the consensus.
    "xgb_heavy_ensemble": lambda: EnsembleStrategy(
        name="xgb_heavy_ensemble",
        members=(
            XGBStrategyV2(seed=42),
            XGBStrategyV2(seed=7),
            XGBStrategyV2(seed=123),
            LGBStrategyV2(seed=42),
        ),
        weights=(1.0, 1.0, 1.0, 1.0),
    ),
    # Multi-target ensemble: same model & features, 4 different forward
    # targets.  This diversifies y-side noise (which seed bagging cannot do).
    # Weights favour the canonical 5d target (matches the eval window best).
    "xgb_v2_multi_target": lambda: EnsembleStrategy(
        name="xgb_v2_multi_target",
        members=(
            XGBStrategyV2(target_column="target_5d"),
            XGBStrategyV2(target_column="target_3d"),
            XGBStrategyV2(target_column="target_10d"),
            XGBStrategyV2(target_column="target_5d_sharpe"),
        ),
        weights=(2.0, 1.0, 1.0, 1.0),
    ),
    # Multi-horizon return-only (no Sharpe).  Keeps y-noise diversification
    # but stays aligned with the contest's actual eval metric (5d absolute
    # excess return).  Hypothesis: should retain the IC IR gain without
    # the mean-drop the Sharpe target caused.
    "xgb_v2_multi_horizon": lambda: EnsembleStrategy(
        name="xgb_v2_multi_horizon",
        members=(
            XGBStrategyV2(target_column="target_5d"),
            XGBStrategyV2(target_column="target_3d"),
            XGBStrategyV2(target_column="target_10d"),
        ),
        weights=(2.0, 1.0, 1.0),
    ),
    # ---- single-target hyperparam sweep on the current champion (xgb_v2) ---
    # H1: more trees + slower lr + shallower trees -> tighter shrinkage.
    "xgb_v2_h1": lambda: XGBStrategyV2(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        colsample_bytree=0.7,
    ),
    # H2: fewer trees + faster lr + deeper trees -> bias toward big-edge stocks.
    "xgb_v2_h2": lambda: XGBStrategyV2(
        n_estimators=400, max_depth=7, learning_rate=0.06,
        colsample_bytree=0.8,
    ),
    # H3: heavier regularisation, more column subsampling.
    "xgb_v2_h3": lambda: XGBStrategyV2(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        colsample_bytree=0.5, reg_lambda=3.0, min_child_weight=10,
    ),
    # H4: shallow + low lr, lots of trees (closest to "linear-ish" boosting).
    "xgb_v2_h4": lambda: XGBStrategyV2(
        n_estimators=1200, max_depth=4, learning_rate=0.025,
        colsample_bytree=0.7,
    ),
    # ---- cluster-neutralised variants ----
    # Wraps xgb_v2 / xgb_v2_h4 / multi_target with KMeans neutralisation
    # over the trailing-60d return matrix.  Removes "all picks are
    # small-cap growth" style concentration without needing an external
    # sector table.
    "xgb_v2_neutral": lambda: ClusterNeutralWrapper(
        name="xgb_v2_neutral",
        base=XGBStrategyV2(),
        n_clusters=10,
    ),
    "xgb_v2_h4_neutral": lambda: ClusterNeutralWrapper(
        name="xgb_v2_h4_neutral",
        base=XGBStrategyV2(
            n_estimators=1200, max_depth=4,
            learning_rate=0.025, colsample_bytree=0.7,
        ),
        n_clusters=10,
    ),
    "xgb_v2_neutral_8": lambda: ClusterNeutralWrapper(
        name="xgb_v2_neutral_8",
        base=XGBStrategyV2(),
        n_clusters=8,
    ),
    "xgb_v2_neutral_15": lambda: ClusterNeutralWrapper(
        name="xgb_v2_neutral_15",
        base=XGBStrategyV2(),
        n_clusters=15,
    ),
    # Best-of-IC blend: combine the three highest-IR base learners we have:
    #   - xgb_v2 default        (mean champ, IR 0.227)
    #   - xgb_v2 h4 settings    (best single IR among hp sweep, 0.294)
    #   - xgb_v2 sharpe target  (the IR-boosting member of multi_target, 0.345)
    # Hypothesis: rank-averaging three high-IR models with different biases
    # should keep the IR advantage *and* drag mean back up vs multi_target's
    # 0.48% (because we drop the mean-dragging 3d/10d horizon members).
    # 2x weight on the default xgb_v2 anchors mean toward the champion.
    "best_blend": lambda: EnsembleStrategy(
        name="best_blend",
        members=(
            XGBStrategyV2(),
            XGBStrategyV2(
                n_estimators=1200, max_depth=4,
                learning_rate=0.025, colsample_bytree=0.7,
            ),
            XGBStrategyV2(target_column="target_5d_sharpe"),
        ),
        weights=(2.0, 1.0, 1.0),
    ),
}
