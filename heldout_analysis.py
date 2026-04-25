"""Honest held-out evaluation across already-completed walk-forward runs.

The earlier 20-strategy comparison used the same 38 windows for both
SELECTION and EVALUATION, which is an implicit form of overfitting --
the chosen winner could just be the strategy whose noise pattern best
matches those particular 38 windows.

This script enforces a clean train/test split:

  * SELECTION SET = first N_SEL windows (chronologically earliest)
  * HELD-OUT SET  = last N_HOLDOUT windows (chronologically latest,
                    closest to the actual contest evaluation date)

For each existing run we:
  1. Compute summary stats on the selection set ONLY.
  2. Identify the top-K strategies by selection-set mean.
  3. Look up THEIR held-out performance -- that is the unbiased estimate
     of out-of-sample alpha.

Because the held-out windows are the most recent, this directly answers
"which strategy would still have been winning in March-April 2026?"
which is the closest analogue we have for May 2026.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

N_HOLDOUT = 8     # last 8 windows form the untouched test set
TOP_K_DISPLAY = 6  # strategies to print


def _strategy_name(path: str) -> str:
    fname = os.path.basename(path)
    return re.sub(r"_\d{8}_\d{6}\.csv$", "", fname)


def load_runs() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob("reports/*_full_*.csv")):
        if os.path.getsize(path) < 100:
            continue
        df = pd.read_csv(path, parse_dates=["as_of", "eval_start", "eval_end"])
        if len(df) < 30:
            continue
        df = df.sort_values("as_of").reset_index(drop=True)
        out[_strategy_name(path)] = df
    return out


def _stats(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "mean%": np.nan, "std%": np.nan, "t_stat": np.nan,
                "hit_rate": np.nan, "worst%": np.nan}
    mean = s.mean()
    std = s.std()
    return {
        "n": n,
        "mean%": mean * 100,
        "std%": std * 100,
        "t_stat": mean / (std / np.sqrt(n)) if std > 0 else np.nan,
        "hit_rate": (s > 0).mean(),
        "worst%": s.min() * 100,
    }


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("No reports found.")

    # All runs should share the same window dates (38 windows, same as_of)
    # but defensively verify.
    sample_dates = next(iter(runs.values()))["as_of"].tolist()
    for tag, df in runs.items():
        if df["as_of"].tolist() != sample_dates:
            print(f"WARN: {tag} has different windows; skipping")
            del runs[tag]

    n_total = len(sample_dates)
    n_holdout = N_HOLDOUT
    n_select = n_total - n_holdout
    print(f"Total windows : {n_total}")
    print(f"Selection set : windows 1..{n_select}  ({sample_dates[0].date()} -> {sample_dates[n_select-1].date()})")
    print(f"Held-out set  : windows {n_select+1}..{n_total}  ({sample_dates[n_select].date()} -> {sample_dates[-1].date()})")
    print()

    rows = []
    for tag, df in runs.items():
        sel = df.iloc[:n_select]["excess_return"]
        hold = df.iloc[n_select:]["excess_return"]
        all_ = df["excess_return"]
        sel_s = _stats(sel)
        hold_s = _stats(hold)
        all_s = _stats(all_)
        rows.append({
            "strategy": tag,
            "all_mean%": all_s["mean%"],
            "all_t":     all_s["t_stat"],
            "sel_mean%": sel_s["mean%"],
            "sel_t":     sel_s["t_stat"],
            "sel_hit":   sel_s["hit_rate"],
            "hold_mean%": hold_s["mean%"],
            "hold_t":     hold_s["t_stat"],
            "hold_hit":   hold_s["hit_rate"],
            "hold_worst%": hold_s["worst%"],
            "shrink%":   hold_s["mean%"] - sel_s["mean%"],
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")

    # 1. Sort by SELECTION-SET mean, then look at HOLD-OUT.
    print("=== Ranked by SELECTION mean (would have been picked on first 30) ===")
    sel_rank = df.sort_values("sel_mean%", ascending=False).reset_index(drop=True)
    print(sel_rank[["strategy","sel_mean%","sel_t","sel_hit",
                    "hold_mean%","hold_t","hold_hit","hold_worst%","shrink%"]].to_string(index=False))
    print()

    # 2. Sort by HELD-OUT mean (the truth -- what we actually want)
    print("=== Ranked by HELD-OUT mean (the unbiased estimate) ===")
    hold_rank = df.sort_values("hold_mean%", ascending=False).reset_index(drop=True)
    print(hold_rank[["strategy","sel_mean%","sel_t",
                    "hold_mean%","hold_t","hold_hit","hold_worst%","shrink%"]].to_string(index=False))
    print()

    # 3. Selection-set top K -> their held-out performance is what we should trust.
    top_k = sel_rank.head(TOP_K_DISPLAY)
    print(f"=== Top-{TOP_K_DISPLAY} by selection mean: their held-out reality ===")
    print(top_k[["strategy","sel_mean%","hold_mean%","hold_t","hold_hit","shrink%"]].to_string(index=False))
    print()

    # 4. Did the round-2 winner (xgb_v2_neutral_8) actually keep winning out of sample?
    print("=== Spotlight: how 'winners' shrank ===")
    spotlight = ["xgb_v2_neutral_8_full", "v2_full", "xgb_v2_h4_full",
                 "xgb_v2_h4_neutral_full", "baseline_full"]
    spot_df = df[df["strategy"].isin(spotlight)].copy()
    spot_df["rank_sel"] = sel_rank.index[sel_rank["strategy"].isin(spotlight).values].tolist() if False else \
                         [sel_rank["strategy"].tolist().index(s) + 1 for s in spot_df["strategy"]]
    spot_df["rank_hold"] = [hold_rank["strategy"].tolist().index(s) + 1 for s in spot_df["strategy"]]
    print(spot_df[["strategy","rank_sel","rank_hold","sel_mean%","hold_mean%","shrink%"]]
          .sort_values("rank_sel").to_string(index=False))


if __name__ == "__main__":
    main()
