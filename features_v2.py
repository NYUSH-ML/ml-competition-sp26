"""
Feature engineering v2 for the CSI500 stock-selection task.

Additions over `features.py`:

1. **Richer factor families**
   - Multi-horizon momentum (1, 3, 5, 10, 20, 60) plus skip-1-week (60-5).
   - Short-term reversal (negated 5d return).
   - Volatility regimes (5d, 20d, 60d) and vol-of-vol.
   - Lottery factor (max single-day return in last 20d, Bali et al. style).
   - Daily range factor (avg (high-low)/close).
   - Amihud illiquidity (mean |ret| / amount).
   - Turnover mean and uncertainty (std).
   - Volume trend (log of 5d/20d avg volume).
   - Multi-horizon MA distances (5, 10, 20, 60).
   - RSI at 6 and 14.
   - 60d rolling beta to CSI500 + idiosyncratic vol.
   - 60d rolling skewness.

2. **Cross-sectional standardization**
   - For each trading day: winsorize at 1% / 99%, then z-score to mean 0 / std 1.
   - This removes time-varying scale (e.g. crisis-wide vol spikes) and lets
     the model learn purely cross-sectional rankings.

3. **Pandas 3.0 friendly**
   - Uses `groupby(...).apply(..., include_groups=False)` so the per-stock
     transform never re-receives the grouping column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----- Public schema --------------------------------------------------------

# Order must stay stable; the trainer reads this list.
FEATURE_COLUMNS: list[str] = [
    # Returns / momentum
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "mom_60_5",         # 60d momentum minus the most recent week
    "rev_5d",           # short-term reversal (negated 5d return)
    # Volatility family
    "vol_5d", "vol_20d", "vol_60d",
    "vol_of_vol_60d",
    "skew_60d",
    # Lottery / range
    "max_ret_20d",
    "range_20d",
    # Liquidity
    "amihud_20d",
    "turnover_ma_20d",
    "turnover_std_20d",
    "volume_z_20d",
    "vol_trend",
    # Trend / mean reversion
    "close_over_ma5", "close_over_ma10", "close_over_ma20", "close_over_ma60",
    # Oscillators
    "rsi_6", "rsi_14",
    # Beta family
    "beta_60d",
    "ivol_60d",
]

TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5

# Subset that we additionally expose as cross-sectional pct-ranks (useful
# nonlinear signal that survives the z-score).
RANKED_FEATURES: list[str] = [
    "ret_5d", "ret_20d", "vol_20d", "amihud_20d", "max_ret_20d",
]
RANK_COLUMNS: list[str] = [f"{c}_rank" for c in RANKED_FEATURES]

# Final list of model inputs (z-scored values + pct-ranks).
ALL_FEATURES: list[str] = FEATURE_COLUMNS + RANK_COLUMNS


# ----- Per-stock features ---------------------------------------------------

def _per_stock_features(df: pd.DataFrame, index_ret: pd.Series) -> pd.DataFrame:
    """All features that only depend on a single stock's time series.

    `index_ret` is a Series indexed by date with the CSI500 daily return; used
    for rolling beta / idiosyncratic vol.
    """
    df = df.sort_values("date").copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    amt = df["amount"].astype(float)

    ret_1d = close.pct_change(1)
    df["ret_1d"] = ret_1d
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)
    df["mom_60_5"] = df["ret_60d"] - df["ret_5d"]
    df["rev_5d"] = -df["ret_5d"]

    df["vol_5d"] = ret_1d.rolling(5).std()
    df["vol_20d"] = ret_1d.rolling(20).std()
    df["vol_60d"] = ret_1d.rolling(60).std()
    df["vol_of_vol_60d"] = df["vol_20d"].rolling(60).std()
    df["skew_60d"] = ret_1d.rolling(60).skew()

    df["max_ret_20d"] = ret_1d.rolling(20).max()
    df["range_20d"] = ((high - low) / close).rolling(20).mean()

    # Amihud illiquidity: mean |r| / amount; tiny constant to avoid div-zero
    illiq = ret_1d.abs() / amt.replace(0, np.nan)
    df["amihud_20d"] = illiq.rolling(20).mean()

    if "turnover" in df.columns:
        to = df["turnover"].astype(float)
        df["turnover_ma_20d"] = to.rolling(20).mean()
        df["turnover_std_20d"] = to.rolling(20).std()
    else:
        df["turnover_ma_20d"] = np.nan
        df["turnover_std_20d"] = np.nan

    vol = df["volume"].astype(float)
    vol_mean_20 = vol.rolling(20).mean()
    vol_std_20 = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean_20) / vol_std_20
    vol_mean_5 = vol.rolling(5).mean()
    df["vol_trend"] = np.log((vol_mean_5 + 1.0) / (vol_mean_20 + 1.0))

    for n in (5, 10, 20, 60):
        df[f"close_over_ma{n}"] = close / close.rolling(n).mean() - 1.0

    df["rsi_6"] = _rsi(close, 6)
    df["rsi_14"] = _rsi(close, 14)

    # Rolling beta and idiosyncratic vol vs CSI500 over 60d
    bench = index_ret.reindex(df["date"].values).to_numpy()
    beta, ivol = _rolling_beta_ivol(ret_1d.to_numpy(), bench, 60)
    df["beta_60d"] = beta
    df["ivol_60d"] = ivol

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean().replace(0, np.nan)
    rs = up / down
    return 100 - 100 / (1 + rs)


def _rolling_beta_ivol(stock: np.ndarray, bench: np.ndarray, window: int):
    """Vectorised rolling regression: stock = alpha + beta * bench + eps.

    Returns (beta, idio_vol) arrays of the same length, NaN-padded for the
    first `window-1` points and wherever inputs are NaN.
    """
    n = len(stock)
    beta = np.full(n, np.nan)
    ivol = np.full(n, np.nan)
    if n < window:
        return beta, ivol

    s = pd.Series(stock)
    b = pd.Series(bench)
    mean_s = s.rolling(window).mean()
    mean_b = b.rolling(window).mean()
    var_b = b.rolling(window).var()
    cov_sb = s.rolling(window).cov(b)
    beta_series = cov_sb / var_b
    alpha_series = mean_s - beta_series * mean_b

    beta = beta_series.to_numpy()
    # Compute idiosyncratic vol = std(residuals) over the same window.
    # residual = s - alpha - beta * b; we approximate using vectorised pieces.
    # var(s) = beta^2 * var(b) + var(eps), so var(eps) = var(s) - beta^2*var(b).
    var_s = s.rolling(window).var()
    resid_var = (var_s - beta_series ** 2 * var_b).clip(lower=0)
    ivol = np.sqrt(resid_var.to_numpy())
    # Suppress alpha (unused) to satisfy linters
    _ = alpha_series
    return beta, ivol


# ----- Cross-sectional standardisation --------------------------------------

def _cross_sectional_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily winsorize (1/99%) + z-score for every FEATURE_COLUMN, plus pct ranks.

    Implemented with `groupby(...).transform(...)` so we never lose the
    `date` column and stay fully vectorised.
    """
    # 1. Cross-sectional pct ranks for selected raw factors (kept un-zscored).
    for base in RANKED_FEATURES:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )

    # 2. Per-day winsorize + zscore on each feature column.
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
    """Return a (date, stock_code) panel with ALL_FEATURES + TARGET_COLUMN.

    `index_df` is the CSI500 daily OHLCV (used for rolling beta / idio vol).
    """
    needed = {"date", "stock_code", "open", "close", "high", "low",
              "volume", "amount"}
    missing = needed - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    index_ret = idx.set_index("date")["close"].pct_change()

    panel = (
        prices.groupby("stock_code", group_keys=True)
        .apply(_per_stock_features, index_ret=index_ret, include_groups=False)
        .reset_index(level=0)            # restore stock_code column
        .reset_index(drop=True)
    )

    # Sanity: every numeric feature column should exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in panel.columns]
    if missing_cols:
        raise RuntimeError(f"missing feature columns after build: {missing_cols}")

    panel = _cross_sectional_transforms(panel)
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=ALL_FEATURES + [TARGET_COLUMN]).copy()
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
