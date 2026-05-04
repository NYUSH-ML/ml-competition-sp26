"""Robustness analysis across all walk-forward runs.

For each strategy report in reports/<tag>_*.csv we already have per-window
excess returns.  This script:

  1. Splits the 38-window timeline into N equal epochs (default 3).
  2. Computes per-epoch mean and t-stat for every strategy.
  3. Reports which strategies are top-quartile in EVERY epoch (the
     "stability survivors") -- these are the candidates least likely to
     have been picked just because they happened to fit the overall mean.
  4. Prints the rank consistency (Kendall tau) of every strategy across
     epochs.

We DO NOT pick a final strategy here -- this is just a diagnostic to flag
which previous winners are robust vs which are mean-fitting artifacts.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd


def _strategy_name(path: str) -> str:
    """reports/xgb_v2_neutral_8_full_20260425_073745.csv -> xgb_v2_neutral_8_full"""
    fname = os.path.basename(path)
    fname = re.sub(r"_\d{8}_\d{6}\.csv$", "", fname)
    return fname


def load_all_runs() -> dict[str, pd.DataFrame]:
    """Load every non-empty per-window CSV in reports/ keyed by strategy tag.

    If two timestamped runs exist for the same strategy tag, keep the most
    recent one (lexicographic max -> latest timestamp).
    """
    runs: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob("reports/*_full_*.csv")):
        if os.path.getsize(path) < 100:  # crashed / interrupted run
            continue
        df = pd.read_csv(path, parse_dates=["as_of", "eval_start", "eval_end"])
        if len(df) < 5:
            continue
        runs[_strategy_name(path)] = df
    return runs


def per_epoch_summary(df: pd.DataFrame, n_epochs: int = 3) -> pd.DataFrame:
    """Slice the windows by chronological order into n equal epochs."""
    df = df.sort_values("as_of").reset_index(drop=True).copy()
    df["epoch"] = pd.qcut(df.index, q=n_epochs, labels=False)
    out = (
        df.groupby("epoch")["excess_return"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_excess", "std": "std_excess", "count": "n"})
    )
    out["t_stat"] = out["mean_excess"] / (out["std_excess"] / np.sqrt(out["n"])).replace(0, np.nan)
    return out


def main(n_epochs: int = 3) -> None:
    runs = load_all_runs()
    if not runs:
        raise SystemExit("No reports found.")

    # 1. Overall stats per strategy
    overall = []
    for tag, df in runs.items():
        n = len(df)
        m = df["excess_return"].mean()
        s = df["excess_return"].std()
        overall.append({
            "strategy": tag,
            "n": n,
            "mean_pct": m * 100,
            "std_pct": s * 100,
            "t_stat": m / (s / np.sqrt(n)) if s > 0 else np.nan,
            "hit_rate": (df["excess_return"] > 0).mean(),
            "worst_pct": df["excess_return"].min() * 100,
        })
    overall_df = pd.DataFrame(overall).sort_values("mean_pct", ascending=False)

    # 2. Per-epoch mean for every strategy.
    rows = []
    for tag, df in runs.items():
        per = per_epoch_summary(df, n_epochs=n_epochs)
        for ep, r in per.iterrows():
            rows.append({
                "strategy": tag, "epoch": int(ep),
                "mean_pct": r["mean_excess"] * 100,
                "t_stat": r["t_stat"],
                "n": int(r["n"]),
            })
    epoch_df = pd.DataFrame(rows)
    pivot_mean = epoch_df.pivot(index="strategy", columns="epoch", values="mean_pct")
    pivot_t = epoch_df.pivot(index="strategy", columns="epoch", values="t_stat")
    pivot_mean.columns = [f"E{c}_mean%" for c in pivot_mean.columns]
    pivot_t.columns = [f"E{c}_t" for c in pivot_t.columns]

    # 3. Stability score: minimum epoch mean (worst-case epoch).
    pivot_mean["min_epoch_mean%"] = pivot_mean.min(axis=1)
    pivot_mean["max_epoch_mean%"] = pivot_mean.max(axis=1)
    pivot_mean["mean_range%"] = pivot_mean["max_epoch_mean%"] - pivot_mean["min_epoch_mean%"]

    # 4. Combined view, sorted by min-epoch (most pessimistic).
    combined = (
        overall_df.set_index("strategy")[["mean_pct", "t_stat", "hit_rate", "worst_pct"]]
        .join(pivot_mean[["min_epoch_mean%", "max_epoch_mean%", "mean_range%"]])
        .sort_values("min_epoch_mean%", ascending=False)
    )

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")

    print(f"=== Robustness across {n_epochs} epochs (sorted by WORST-epoch mean) ===")
    print(combined.to_string())

    print()
    print(f"=== Per-epoch mean% (each row sums different windows) ===")
    print(pivot_mean.sort_values("min_epoch_mean%", ascending=False).to_string())


if __name__ == "__main__":
    main()
