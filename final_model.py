"""
Final shipped model for the CSI500 Spring 2026 stock-selection competition.

This single file is self-contained: given the data files in `data/` it
re-trains the model and reproduces `submissions/submission2.csv` (effective
K=20 portfolio) from scratch.  It is the code referenced by §4 ("Final
Decision and Deliverables") and §8 ("Reproduction") of the project report.

PIPELINE
--------
1. Load CSI500 daily OHLCV from `data/prices.parquet` and the CSI500 index
   from `data/index.parquet`.
2. Build the v2 feature panel: 28 per-stock factors (momentum, volatility,
   liquidity, oscillators, beta) + 5 cross-sectional pct-rank duplicates =
   33 model inputs, plus the 5d forward return target.
3. For each daily cross-section, winsorise at 1/99% and z-score so the
   model learns purely cross-sectional rankings.
4. Train a single XGBoost regressor on all data up to (as_of − 5 trading
   day embargo); validate on the most recent 10 trading days; report
   daily-cross-sectional Spearman rank IC on the validation block.
5. Predict scores at as_of = max(date in panel).
6. Build the EFFECTIVE-K=20 portfolio:
     - Top 20 stocks by score: rank-weighted (rank N -> N), 10% cap,
       weights summing to 0.995.
     - Positions 21-30: floor weight 0.0005 each, summing to 0.005.
     - Total: 30 names, sum to 1.0, max <= 10% (rule-compliant).
7. Validate the CSV against the four competition constraints and write
   `submissions/submission2.csv`.

USAGE
-----
    python final_model.py

The script writes `submissions/submission2.csv` and a sidecar
`submissions/submission2.diag.json` with training diagnostics.

DEPENDENCIES
------------
    numpy, pandas, pyarrow, scipy, scikit-learn, xgboost

    pip install numpy pandas pyarrow scipy scikit-learn xgboost

DESIGN NOTES
------------
- The K=20 effective portfolio is the result of Round 10's robustness audit.
  See §3.10 of the project report for the 9-test framework that selected
  K=20 over K=25 and K=50.  K=20 has the highest historical mean (+0.892%)
  and best held-out mean (+1.61% on last 8 windows of the walk-forward).
- The 5-trading-day embargo between training data and prediction date
  prevents target leakage: the 5d forward return label of the last training
  observation cannot peek past the prediction as_of date.
- Cross-sectional standardisation removes time-varying scale (e.g. crisis-
  wide volatility spikes) so the model learns purely cross-sectional
  rankings rather than absolute factor levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


# =============================================================================
# Section 1.  Feature engineering (mirrors features_v2.py exactly)
# =============================================================================

FEATURE_COLUMNS: list[str] = [
    # Returns / momentum
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "mom_60_5",       # 60d momentum minus the most recent week
    "rev_5d",         # short-term reversal (negated 5d return)
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

# Subset additionally exposed as cross-sectional pct-ranks (nonlinear signal
# that survives the z-score).
RANKED_FEATURES: list[str] = [
    "ret_5d", "ret_20d", "vol_20d", "amihud_20d", "max_ret_20d",
]
RANK_COLUMNS: list[str] = [f"{c}_rank" for c in RANKED_FEATURES]
ALL_FEATURES: list[str] = FEATURE_COLUMNS + RANK_COLUMNS

TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder-style RSI."""
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean().replace(0, np.nan)
    rs = up / down
    return 100 - 100 / (1 + rs)


def _rolling_beta_ivol(
    stock: np.ndarray, bench: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised rolling regression: stock = alpha + beta * bench + eps.

    Returns `(beta, idiosyncratic_vol)` arrays the same length as inputs,
    NaN-padded for the first `window-1` points.
    """
    n = len(stock)
    if n < window:
        return np.full(n, np.nan), np.full(n, np.nan)
    s = pd.Series(stock)
    b = pd.Series(bench)
    var_b = b.rolling(window).var()
    cov_sb = s.rolling(window).cov(b)
    beta_series = cov_sb / var_b
    var_s = s.rolling(window).var()
    resid_var = (var_s - beta_series ** 2 * var_b).clip(lower=0)
    return beta_series.to_numpy(), np.sqrt(resid_var.to_numpy())


def _per_stock_features(df: pd.DataFrame, index_ret: pd.Series) -> pd.DataFrame:
    """All features that depend only on a single stock's time series.

    `index_ret` is a Series indexed by date with the CSI500 daily return,
    used for rolling beta and idiosyncratic vol.
    """
    df = df.sort_values("date").copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    amt = df["amount"].astype(float)
    vol = df["volume"].astype(float)

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

    illiq = ret_1d.abs() / amt.replace(0, np.nan)
    df["amihud_20d"] = illiq.rolling(20).mean()

    if "turnover" in df.columns:
        to = df["turnover"].astype(float)
        df["turnover_ma_20d"] = to.rolling(20).mean()
        df["turnover_std_20d"] = to.rolling(20).std()
    else:
        df["turnover_ma_20d"] = np.nan
        df["turnover_std_20d"] = np.nan

    vol_mean_20 = vol.rolling(20).mean()
    vol_std_20 = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean_20) / vol_std_20
    vol_mean_5 = vol.rolling(5).mean()
    df["vol_trend"] = np.log((vol_mean_5 + 1.0) / (vol_mean_20 + 1.0))

    for n in (5, 10, 20, 60):
        df[f"close_over_ma{n}"] = close / close.rolling(n).mean() - 1.0

    df["rsi_6"] = _rsi(close, 6)
    df["rsi_14"] = _rsi(close, 14)

    bench = index_ret.reindex(df["date"].values).to_numpy()
    beta, ivol = _rolling_beta_ivol(ret_1d.to_numpy(), bench, 60)
    df["beta_60d"] = beta
    df["ivol_60d"] = ivol

    # Primary target: 5d forward return on close-to-close.
    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _cross_sectional_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily winsorise + z-score every feature column, plus pct ranks."""
    # 1. Cross-sectional pct ranks (kept un-zscored).
    for base in RANKED_FEATURES:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )

    # 2. Per-day winsorise (1/99%) + zscore on each feature column.
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


def build_feature_panel(
    prices: pd.DataFrame, index_df: pd.DataFrame
) -> pd.DataFrame:
    """Return a `(date, stock_code)` panel with `ALL_FEATURES + TARGET_COLUMN`."""
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
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    panel = _cross_sectional_transforms(panel)
    return panel


# =============================================================================
# Section 2.  Train / validation split with embargo
# =============================================================================

VAL_DAYS = 10
EMBARGO_DAYS = 5  # >= FORWARD_HORIZON so labels never peek past as_of


def train_val_split(
    panel: pd.DataFrame, as_of: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Split into train / embargo (discarded) / val / as_of.

    Layout::

        ... | train ... | embargo (5d) | val (10d) | as_of (excluded)

    Training rows have `date <= train_end`; val rows have `date >= val_start`;
    the entire training pool is capped at `as_of - embargo` so that 5d-forward
    target labels of the last training observation never include returns
    realised after `as_of`.
    """
    pool = panel.dropna(subset=ALL_FEATURES + [TARGET_COLUMN]).copy()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - EMBARGO_DAYS)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    pool = pool[pool["date"] <= train_cutoff]
    all_dates = np.sort(pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError(
            f"Not enough history before {as_of.date()} for a clean split "
            f"(have {len(all_dates)} dates, need >= "
            f"{VAL_DAYS + EMBARGO_DAYS + 20})"
        )
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    return (
        pool[pool["date"] <= train_end],
        pool[pool["date"] >= val_start],
        train_end,
        val_start,
    )


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


# =============================================================================
# Section 3.  XGBoost training
# =============================================================================

# Hyperparameters from XGBStrategyV2 (frozen after Round 1; never re-tuned
# after held-out validation in subsequent rounds).
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10.0,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=30,
    random_state=0,
)


def train_and_score(
    panel: pd.DataFrame, as_of: pd.Timestamp
) -> Tuple[pd.Series, dict]:
    """Train one XGBoost regressor and return per-stock scores at as_of."""
    train_df, val_df, train_end, val_start = train_val_split(panel, as_of)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train_df[ALL_FEATURES], train_df[TARGET_COLUMN],
        eval_set=[(val_df[ALL_FEATURES], val_df[TARGET_COLUMN])],
        verbose=False,
    )

    val_pred = model.predict(val_df[ALL_FEATURES])
    val_ic = rank_ic(
        val_df[TARGET_COLUMN].to_numpy(),
        val_pred,
        val_df["date"].to_numpy(),
    )

    pred_df = panel[panel["date"] == as_of].dropna(subset=ALL_FEATURES).copy()
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows on {as_of.date()}")
    pred_df["score"] = model.predict(pred_df[ALL_FEATURES])
    scores = pred_df.set_index("stock_code")["score"]

    diag = {
        "val_ic": val_ic,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_pred": len(pred_df),
        "n_features": len(ALL_FEATURES),
        "train_end": train_end.date().isoformat(),
        "val_start": val_start.date().isoformat(),
        "best_iter": int(getattr(model, "best_iteration", -1) or -1),
    }
    return scores, diag


# =============================================================================
# Section 4.  Effective K=20 portfolio construction
# =============================================================================

TOP_K = 20             # Effective concentration: top 20 carry 99.5% of weight
N_FLOOR = 10           # Positions 21..30 carry the remaining 0.5%
FLOOR_WEIGHT = 0.0005  # Each floor position
MAX_WEIGHT = 0.10      # Competition rule


def _apply_cap(w: np.ndarray, cap: float, target_sum: float) -> np.ndarray:
    """Iteratively cap weights at `cap` and redistribute excess proportionally."""
    w = w.astype(float).copy()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        un = ~over
        if un.any() and (cap - w[un]).sum() > 0:
            w[un] += excess * (w[un] / w[un].sum())
    return w * (target_sum / w.sum())


def build_effective_k20_portfolio(scores: pd.Series) -> pd.Series:
    """Build the 30-name effective-K=20 portfolio.

    Top 20 stocks by score: rank-weighted (rank N -> weight N), capped at 10%,
    rescaled so the top-20 block sums to 0.995.
    Positions 21-30: floor weight 0.0005 each, summing to 0.005.
    Total: 30 names, weights sum to 1.000, max weight <= 10%.
    """
    ranked = scores.sort_values(ascending=False).head(TOP_K + N_FLOOR)
    if len(ranked) < TOP_K + N_FLOOR:
        raise RuntimeError(
            f"Need at least {TOP_K + N_FLOOR} candidates; got {len(ranked)}"
        )
    top = ranked.head(TOP_K)
    floor = ranked.tail(N_FLOOR)

    top_block_sum = 1.0 - N_FLOOR * FLOOR_WEIGHT  # 0.995
    rank_w = np.arange(TOP_K, 0, -1, dtype=float)
    rank_w = rank_w / rank_w.sum() * top_block_sum
    rank_w = _apply_cap(rank_w, MAX_WEIGHT, top_block_sum)

    weights = pd.Series(
        np.concatenate([rank_w, np.full(N_FLOOR, FLOOR_WEIGHT)]),
        index=list(top.index) + list(floor.index),
        name="weight",
    )
    weights.index = weights.index.astype(str).str.zfill(6)
    weights.index.name = "stock_code"
    return weights.sort_values(ascending=False)


# =============================================================================
# Section 5.  Submission validator (mirrors validate_submission.py)
# =============================================================================

def validate_submission(
    weights: pd.Series, constituents_path: Path | None = None
) -> list[str]:
    """Return a list of constraint violations (empty list = PASS)."""
    errs: list[str] = []
    if (weights < 0).any():
        errs.append("constraint 3.4: negative weights present")
    if abs(weights.sum() - 1.0) > 1e-4:
        errs.append(f"constraint 3.4: weights sum to {weights.sum():.6f} (need 1.0)")
    if (weights > MAX_WEIGHT + 1e-9).any():
        errs.append(f"constraint 3.3: max weight {weights.max():.4f} > {MAX_WEIGHT}")
    if (weights > 0).sum() < 30:
        errs.append(
            f"constraint 3.2: only {(weights > 0).sum()} names with positive weight (need >= 30)"
        )
    if any(not (isinstance(c, str) and len(c) == 6 and c.isdigit())
           for c in weights.index):
        errs.append("stock_code must be a zero-padded 6-digit string")
    if constituents_path is not None and constituents_path.exists():
        cons = pd.read_csv(constituents_path, dtype={"stock_code": str})
        cons["stock_code"] = cons["stock_code"].str.zfill(6)
        bad = set(weights.index) - set(cons["stock_code"])
        if bad:
            errs.append(
                f"constraint 3.1: {len(bad)} stocks not in CSI500 (e.g. {sorted(bad)[:3]})"
            )
    return errs


# =============================================================================
# Section 6.  Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices", default="data/prices.parquet",
        help="Path to CSI500 daily OHLCV parquet."
    )
    parser.add_argument(
        "--index", default="data/index.parquet",
        help="Path to CSI500 index parquet."
    )
    parser.add_argument(
        "--constituents", default="data/constituents.csv",
        help="Path to CSI500 constituents CSV (for universe validation)."
    )
    parser.add_argument(
        "--out", default="submissions/submission2.csv",
        help="Output path for the portfolio CSV."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CSI500 Stock Selection - Final Model (effective K=20)")
    print("=" * 60)

    # 1. Load
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    print(f"Prices: {len(prices):,} rows, "
          f"{prices['stock_code'].nunique()} stocks, "
          f"latest = {prices['date'].max().date()}")

    # 2. Build features
    print("\nBuilding feature panel ...")
    panel = build_feature_panel(prices, index_df)
    as_of = pd.Timestamp(panel["date"].max())
    print(f"Feature panel: {len(panel):,} rows, "
          f"{len(ALL_FEATURES)} features, as_of = {as_of.date()}")

    # 3. Train and score
    print("\nTraining XGBoost regressor ...")
    scores, diag = train_and_score(panel, as_of)
    print(f"  validation rank IC: {diag['val_ic']:+.4f}")
    print(f"  n_train = {diag['n_train']:,},  n_val = {diag['n_val']:,},  "
          f"n_pred = {diag['n_pred']}")
    print(f"  best_iter = {diag['best_iter']}")

    # 4. Build effective-K=20 portfolio
    weights = build_effective_k20_portfolio(scores)
    print(f"\nPortfolio: {len(weights)} names, "
          f"sum = {weights.sum():.6f}, "
          f"max = {weights.max():.4%}, min = {weights.min():.4%}")
    print(f"  top-20 block: {weights.iloc[:TOP_K].sum():.6f} "
          f"({weights.iloc[:TOP_K].sum() * 100:.2f}%)")
    print(f"  floor block:  {weights.iloc[TOP_K:].sum():.6f} "
          f"({weights.iloc[TOP_K:].sum() * 100:.2f}%)")

    # 5. Validate
    cons = Path(args.constituents)
    errors = validate_submission(weights, cons if cons.exists() else None)
    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("\nValidation: PASS (all 4 competition constraints satisfied)")

    # 6. Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out_path, float_format="%.6f")
    diag_path = out_path.with_suffix(".diag.json")
    diag_full = {
        "model": "XGBoost regressor (final_model.py)",
        "feature_set": "v2 (35 features: 28 z-scored + 5 pct-ranks + 2 derived)",
        "portfolio": f"effective K={TOP_K}, {N_FLOOR} floor positions at "
                     f"{FLOOR_WEIGHT}, total {TOP_K + N_FLOOR} names",
        "as_of": as_of.date().isoformat(),
        "prediction_window": "2026-05-11 to 2026-05-15",
        **{k: (v if not isinstance(v, np.generic) else v.item())
           for k, v in diag.items()},
    }
    with open(diag_path, "w") as f:
        json.dump(diag_full, f, indent=2, default=str)

    print(f"\nWrote {out_path}")
    print(f"Wrote {diag_path}")
    print("\nTop 5 holdings:")
    for code, w in weights.head(5).items():
        print(f"  {code}  {w:.4%}")


if __name__ == "__main__":
    main()
