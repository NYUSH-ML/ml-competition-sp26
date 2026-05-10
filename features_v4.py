"""
Feature engineering v4 -- extends v2 with 5 sector-relative factors using
SW (Shenwan) industry classification.

Why this round (Round 8):
  All v2/v3 features measure each stock relative to the **broad** CSI500
  index.  But A-share short-term returns are dominated by sector rotation:
  in any given week, ~70% of cross-sectional return variance comes from
  which industry the stock is in, not from idiosyncratic alpha.  By feeding
  the model BOTH "stock vs broad market" (already in v2) AND "stock vs its
  own sector" (new here) the model can finally distinguish:
    - "this stock is leading its hot sector" (real alpha)
    - "this stock is rising because its sector is hot" (mostly beta)
    - "this stock is the laggard of a hot sector" (likely mean-revert)

Industry source:
    akshare's `stock_industry_clf_hist_sw` (Shenwan classification).
    First 2 digits of the 6-digit code give 31 industries; small ones
    (<5 CSI500 stocks) are folded into a single "OT" bucket.  Mapping is
    saved once to `data/industry.parquet`.

New columns (5 raw + 2 ranks):
  sector_excess_5d        : 5d stock return minus 5d sector return
  sector_excess_20d       : 20d stock return minus 20d sector return
  sector_outperf_pct_20d  : fraction of last 20 days stock beat its sector
  sector_momentum_5d      : the sector's own 5d return (so model knows
                            whether it's a hot or cold sector)
  sector_relstrength_5d   : within-sector pct rank of 5d return (0..1).
                            Captures "leader of the pack" effect that no
                            other feature provides.

All v2 features and targets are inherited verbatim.
"""
from __future__ import annotations

from pathlib import Path

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
    "sector_excess_5d",
    "sector_excess_20d",
    "sector_outperf_pct_20d",
    "sector_momentum_5d",
    "sector_relstrength_5d",
]

FEATURE_COLUMNS: list[str] = _V2_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS

# Add the sector-excess columns to the rank-feature subset.  Pct ranks of
# excess returns capture nonlinear "is this stock in the top decile of
# sector outperformers" signal that survives z-scoring.
RANKED_FEATURES: list[str] = _V2_RANKED_FEATURES + [
    "sector_excess_5d", "sector_excess_20d",
]
RANK_COLUMNS: list[str] = [f"{c}_rank" for c in RANKED_FEATURES]
ALL_FEATURES: list[str] = FEATURE_COLUMNS + RANK_COLUMNS


# ----- Industry mapping helpers --------------------------------------------

_INDUSTRY_PARQUET = Path("data/industry.parquet")
_MIN_SECTOR_SIZE = 5  # fold smaller sectors into a single "OT" bucket


def build_industry_mapping(out_path: Path = _INDUSTRY_PARQUET) -> pd.DataFrame:
    """One-time helper: fetch SW industry classification from akshare and
    write to `data/industry.parquet`.  Re-run only if the universe changes.
    """
    import akshare as ak

    df = ak.stock_industry_clf_hist_sw()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["industry_code"] = df["industry_code"].astype(str)
    df = df.sort_values(["symbol", "start_date"])
    latest = df.groupby("symbol").tail(1).reset_index(drop=True)
    out = latest[["symbol", "industry_code"]]
    out.to_parquet(out_path)
    return out


def _load_sector_map(path: Path = _INDUSTRY_PARQUET) -> dict[str, str]:
    """stock_code -> sector_id mapping; tiny sectors collapsed to 'OT'."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: "
            f"python -c 'from features_v4 import build_industry_mapping; "
            f"build_industry_mapping()'"
        )
    df = pd.read_parquet(path)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["sector_id"] = df["industry_code"].astype(str).str.zfill(6).str[:2]
    counts = df.groupby("sector_id").size()
    tiny = set(counts[counts < _MIN_SECTOR_SIZE].index)
    df.loc[df.sector_id.isin(tiny), "sector_id"] = "OT"
    return df.set_index("symbol")["sector_id"].to_dict()


def _build_sector_daily(prices: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """Return a (sector_id, date) frame with sector_ret_1d, _5d, _20d."""
    raw = prices[["date", "stock_code", "close"]].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["stock_code"] = raw["stock_code"].astype(str).str.zfill(6)
    raw["sector_id"] = raw["stock_code"].map(sector_map).fillna("OT")
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    raw["ret_1d_raw"] = raw.groupby("stock_code")["close"].pct_change()

    # Equal-weighted daily sector return (across CSI500 constituents).
    daily = (
        raw.dropna(subset=["ret_1d_raw"])
        .groupby(["sector_id", "date"], as_index=False)["ret_1d_raw"]
        .mean()
        .rename(columns={"ret_1d_raw": "sector_ret_1d"})
        .sort_values(["sector_id", "date"])
        .reset_index(drop=True)
    )
    # Cumulative sector index, then 5d / 20d cumulative returns.
    daily["sector_idx"] = (
        daily.groupby("sector_id")["sector_ret_1d"]
        .transform(lambda s: (1.0 + s).cumprod())
    )
    daily["sector_ret_5d"] = (
        daily.groupby("sector_id")["sector_idx"].pct_change(5)
    )
    daily["sector_ret_20d"] = (
        daily.groupby("sector_id")["sector_idx"].pct_change(20)
    )
    return daily[["sector_id", "date",
                  "sector_ret_1d", "sector_ret_5d", "sector_ret_20d"]]


# ----- Cross-sectional standardisation --------------------------------------

def _cross_sectional_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily winsorize + z-score for every FEATURE_COLUMN, plus pct ranks."""
    for base in RANKED_FEATURES:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )

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
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    index_ret = idx.set_index("date")["close"].pct_change()

    # 1. Build all v2 per-stock features (raw, NOT yet z-scored).
    panel = (
        prices.groupby("stock_code", group_keys=True)
        .apply(_v2_per_stock_features, index_ret=index_ret, include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    # 2. Attach sector_id.
    sector_map = _load_sector_map()
    panel["sector_id"] = (
        panel["stock_code"].astype(str).str.zfill(6).map(sector_map).fillna("OT")
    )

    # 3. Build sector daily/5d/20d returns and merge onto panel.
    sector_daily = _build_sector_daily(prices, sector_map)
    panel = panel.merge(sector_daily, on=["sector_id", "date"], how="left")

    # 4. Sector-relative momentum features (use v2's RAW ret_1d/5d/20d).
    panel["sector_excess_5d"] = panel["ret_5d"] - panel["sector_ret_5d"]
    panel["sector_excess_20d"] = panel["ret_20d"] - panel["sector_ret_20d"]
    panel["sector_momentum_5d"] = panel["sector_ret_5d"]

    # 5. Daily outperformance hit rate vs sector over last 20d.
    panel = panel.sort_values(["stock_code", "date"]).reset_index(drop=True)
    outperf = (panel["ret_1d"] > panel["sector_ret_1d"]).astype(float)
    outperf[panel["ret_1d"].isna() | panel["sector_ret_1d"].isna()] = np.nan
    panel["sector_outperf_pct_20d"] = (
        panel.assign(_o=outperf.values)
        .groupby("stock_code")["_o"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean())
    )

    # 6. Within-sector relative strength: pct rank of 5d return inside the
    #    stock's own sector each day (0 = worst in sector, 1 = best).
    panel["sector_relstrength_5d"] = (
        panel.groupby(["date", "sector_id"])["ret_5d"]
        .rank(method="average", pct=True)
    )

    # 7. Drop the temporary merge columns we no longer need.
    panel = panel.drop(columns=[
        c for c in ["sector_id", "sector_ret_1d",
                    "sector_ret_5d", "sector_ret_20d", "sector_idx"]
        if c in panel.columns
    ])

    # 8. Sanity check every required column is present.
    missing_cols = [c for c in FEATURE_COLUMNS if c not in panel.columns]
    if missing_cols:
        raise RuntimeError(f"missing v4 feature columns: {missing_cols}")

    # 9. Cross-sectional standardization.
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
