"""
Feature engineering v3 — extends v2 with 5 theory-driven index-relative
factors that exploit benchmark data we already have but haven't used.

Why this round:
  - Round 3 held-out analysis showed that more *complex* models / ensembles
    over-fit the selection set.
  - But v2 takes `index_df` as input only for beta/idio-vol; the actual
    relative-momentum / outperformance signal vs the benchmark was never
    fed to the model.  Excess return is exactly what the contest scores us
    on, so feeding excess-return features to the model is logically sound
    and not a tuned trick.

New columns (5 raw + 2 extra ranks, all per-stock):
  excess_ret_5d     :  stock 5d return minus CSI500 5d return
  excess_ret_20d    :  stock 20d return minus CSI500 20d return
  outperf_pct_20d   :  fraction of last 20 trading days where the stock's
                       daily return beat the index's daily return
  accel_5_20        :  ret_5d - ret_20d / 4  (positive => recent acceleration)
  vol_price_div_5d  :  log(mean_vol_5d / mean_vol_20d) * sign(ret_5d)
                       — volume confirmation/divergence of recent move

All other columns and targets are inherited verbatim from features_v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features_v2 import (
    _per_stock_features as _v2_per_stock_features,
    FEATURE_COLUMNS as _V2_FEATURE_COLUMNS,
    RANKED_FEATURES as _V2_RANKED_FEATURES,
    TARGET_COLUMN, FORWARD_HORIZON,
    AUX_TARGETS, ALL_TARGETS,
)


# ----- Public schema --------------------------------------------------------

EXTRA_FEATURE_COLUMNS: list[str] = [
    "excess_ret_5d",
    "excess_ret_20d",
    "outperf_pct_20d",
    "accel_5_20",
    "vol_price_div_5d",
]

FEATURE_COLUMNS: list[str] = _V2_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS

# Add the two excess-return columns to the rank-feature subset; pct ranks
# of excess returns capture nonlinear "is this stock in the top decile of
# outperformers" signal that survives z-scoring.
RANKED_FEATURES: list[str] = _V2_RANKED_FEATURES + [
    "excess_ret_5d", "excess_ret_20d",
]
RANK_COLUMNS: list[str] = [f"{c}_rank" for c in RANKED_FEATURES]
ALL_FEATURES: list[str] = FEATURE_COLUMNS + RANK_COLUMNS


# ----- Per-stock features ---------------------------------------------------

def _per_stock_features(
    df: pd.DataFrame,
    index_ret: pd.Series,
    index_5d_ret: pd.Series,
    index_20d_ret: pd.Series,
) -> pd.DataFrame:
    """v2's per-stock features plus 5 index-relative additions.

    Parameters
    ----------
    index_ret      : daily CSI500 return,    Series indexed by date
    index_5d_ret   : 5d  CSI500 close-to-close return at each date
    index_20d_ret  : 20d CSI500 close-to-close return at each date
    """
    # 1. Build all v2 columns first (also gives us ret_1d/ret_5d/ret_20d).
    df = _v2_per_stock_features(df, index_ret)

    # 2. Align benchmark series to this stock's date column.
    dates = df["date"].values
    bench_1d = index_ret.reindex(dates).to_numpy()
    bench_5d = index_5d_ret.reindex(dates).to_numpy()
    bench_20d = index_20d_ret.reindex(dates).to_numpy()

    # 3. Index-relative momentum features.
    df["excess_ret_5d"] = df["ret_5d"].to_numpy() - bench_5d
    df["excess_ret_20d"] = df["ret_20d"].to_numpy() - bench_20d

    # 4. Daily outperformance hit rate: fraction of last 20 days the stock
    #    beat the index.  Captures consistency, not just net magnitude.
    outperf_flag = (df["ret_1d"].to_numpy() > bench_1d).astype(float)
    # NaN where ret_1d is NaN (first row of each stock), so propagate that.
    outperf_flag[np.isnan(df["ret_1d"].to_numpy()) | np.isnan(bench_1d)] = np.nan
    df["outperf_pct_20d"] = (
        pd.Series(outperf_flag, index=df.index).rolling(20).mean()
    )

    # 5. Price acceleration: how much of the trailing 20d return came in the
    #    last 5d.  Positive => acceleration, negative => deceleration.
    df["accel_5_20"] = df["ret_5d"] - df["ret_20d"] / 4.0

    # 6. Volume-price divergence: confirms or contradicts the 5d move.
    #    +ve = strong volume on rising days,  -ve = high volume on a sell-off
    #    or rising on weak volume (likely fade).
    vol = df["volume"].astype(float)
    vol5 = vol.rolling(5).mean()
    vol20 = vol.rolling(20).mean()
    log_ratio = np.log((vol5 + 1.0) / (vol20 + 1.0))
    df["vol_price_div_5d"] = log_ratio * np.sign(df["ret_5d"])

    return df


# ----- Cross-sectional standardisation --------------------------------------

def _cross_sectional_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily winsorize + z-score for every FEATURE_COLUMN, plus pct ranks.

    Standalone to avoid coupling to v2's hardcoded column list.
    """
    # Cross-sectional pct ranks for selected raw factors (kept un-zscored).
    for base in RANKED_FEATURES:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )

    # Per-day winsorize + zscore on every numeric feature column.
    by_date = panel.groupby("date")
    for c in FEATURE_COLUMNS:
        s = panel[c].astype(float)
        lo = by_date[c].transform(lambda x: x.quantile(0.01))
        hi = by_date[c].transform(lambda x: x.quantile(0.99))
        s = s.clip(lower=lo, upper=hi)
        mu = s.groupby(panel["date"]).transform("mean")
        sd = s.groupby(panel["date"]).transform("std")
        z = (s - mu) / sd.replace(0, np.nan)
        panel[c] = z.fillna(0.0)
    return panel


# ----- Public API -----------------------------------------------------------

def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"date", "stock_code", "open", "close", "high", "low",
              "volume", "amount"}
    missing = needed - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").set_index("date")
    index_ret = idx["close"].pct_change()
    index_5d_ret = idx["close"].pct_change(5)
    index_20d_ret = idx["close"].pct_change(20)

    panel = (
        prices.groupby("stock_code", group_keys=True)
        .apply(
            _per_stock_features,
            index_ret=index_ret,
            index_5d_ret=index_5d_ret,
            index_20d_ret=index_20d_ret,
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    missing_cols = [c for c in FEATURE_COLUMNS if c not in panel.columns]
    if missing_cols:
        raise RuntimeError(f"missing feature columns after build: {missing_cols}")

    panel = _cross_sectional_transforms(panel)
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None,
                   target: str = TARGET_COLUMN) -> pd.DataFrame:
    if target not in panel.columns:
        raise ValueError(
            f"target {target!r} not in panel; available: "
            f"{[c for c in panel.columns if c.startswith('target_')]}"
        )
    df = panel.dropna(subset=ALL_FEATURES + [target]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=ALL_FEATURES).copy()
    return df
