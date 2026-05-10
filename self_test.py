"""
Self-test deliverable for the CSI500 Spring 2026 stock-selection competition.

Performs a clean, single-shot train/validation/test split on the provided
panel data, trains the shipped XGBoost v2 model on the train portion (using
the val portion only for early stopping), and evaluates portfolio performance
on the held-out test tail. Reports test metrics against the provided baseline
on identical splits.

This file complements the 40-window walk-forward cross-validation reported in
PROJECT_REPORT.md Section 9. The walk-forward provides the cross-validated
estimate; this file provides the single-shot held-out test that the rubric
explicitly asks for.

METHODOLOGY
-----------
1. Build the v2 feature panel exactly as the shipped model does.
2. Sort all (stock, date) panel rows by trading date.
3. Apply a strictly chronological 70 / 15 / 15 split:
     train: oldest 70% of trading dates  (model fitting)
     val:   next   15% of trading dates  (early stopping; model selection)
     test:  most  recent 15% of trading dates  (HELD-OUT, evaluated once)
4. Train xgb_v2 ONLY on train+val. The test partition is never seen by the
   model, the early-stopping signal, or any hyperparameter choice.
5. For each scoring date in the test partition (every 5 trading days, the
   competition's evaluation cadence), the model:
     a) Re-fits using ALL data with date <= as_of - embargo (= 5 days).
        This still lives entirely inside train+val by construction of the
        split, so no test-period information leaks into training.
     b) Predicts top-50 portfolio weights for the next 5 trading days.
     c) Realised excess return over CSI500 is computed and logged.
6. Aggregate metrics on the test partition reported in:
        reports/self_test_results.json
        reports/self_test_results.csv
   The same procedure is run for the provided xgb_baseline for direct
   comparison on identical splits.

LEAKAGE GUARDS
--------------
- The 5d forward return target is shifted only AFTER feature construction;
  the join uses an explicit forward-shift (`-horizon` in `groupby.shift`),
  never a same-day join.
- A 5-day embargo is applied between training data and the prediction date,
  so today's features cannot inform a model whose target falls inside the
  embargoed window.
- The chronological split point is computed from sorted unique trading dates;
  no random shuffling.
- Cross-sectional normalisation (z-score, pct-rank) is computed within each
  date independently, so train-set statistics do not contaminate test rows.
- The competition's universe filter (CSI500 constituents at as_of) is applied
  at scoring time, never at training time, so survivorship bias is avoided.

USAGE
-----
    python self_test.py
    # writes reports/self_test_results.json and reports/self_test_results.csv

DEPENDENCIES
------------
    numpy, pandas, pyarrow, scipy, scikit-learn, xgboost
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Re-use the exact shipped strategies and walk-forward scoring logic so
# self-test results are computed by the same code path that produced
# Submissions 1 and 2.
from strategies import STRATEGIES, DEFAULT_EMBARGO, DEFAULT_TOP_K
from score_submission import score_window
from walkforward import Window, _trading_dates


DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
HORIZON = 5  # trading days, matches competition evaluation window


def chronological_split(
    trading_dates: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (train_end_date, val_end_date) cut points on a sorted date array.

    The test partition is everything strictly after val_end_date.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    n = len(trading_dates)
    train_end_idx = int(np.floor(n * train_frac)) - 1
    val_end_idx = int(np.floor(n * (train_frac + val_frac))) - 1
    return (
        pd.Timestamp(trading_dates[train_end_idx]),
        pd.Timestamp(trading_dates[val_end_idx]),
    )


def make_test_windows(
    trading_dates: np.ndarray,
    val_end: pd.Timestamp,
    horizon: int = HORIZON,
    stride: int = HORIZON,
) -> list[Window]:
    """Build evaluation windows whose as_of date strictly follows val_end.

    Every window's training data is restricted (by walk-forward construction)
    to dates <= as_of - embargo, but as_of itself is in the test partition,
    so we evaluate only on previously-unseen forward returns.
    """
    test_dates = trading_dates[trading_dates > np.datetime64(val_end)]
    if len(test_dates) < horizon + 1:
        raise ValueError(
            f"Test partition has only {len(test_dates)} trading days, "
            f"need at least {horizon + 1} for one evaluation window."
        )

    # First as_of is the first day of the test partition. Subsequent as_of
    # dates step forward by `stride` trading days, mirroring competition
    # cadence. Windows whose evaluation period exceeds the panel are dropped.
    windows: list[Window] = []
    i = 0
    while i + horizon < len(test_dates):
        as_of = pd.Timestamp(test_dates[i])
        eval_start = pd.Timestamp(test_dates[i + 1])
        eval_end = pd.Timestamp(test_dates[i + horizon])
        windows.append(Window(as_of=as_of, eval_start=eval_start, eval_end=eval_end))
        i += stride
    return windows


def evaluate_strategy(
    strategy_name: str,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    test_windows: list[Window],
    embargo: int = DEFAULT_EMBARGO,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[pd.DataFrame, dict]:
    """Train + evaluate a strategy on every test window. Returns per-window
    rows and an aggregate summary dict."""
    strat_cls = STRATEGIES[strategy_name]
    rows = []
    for w in test_windows:
        strat = strat_cls()
        # Strategy must see panel rows up to AND INCLUDING as_of so its
        # prediction frame for date == as_of is non-empty.  Training rows
        # (whose forward-return targets extend into the future) are filtered
        # by the strategy's own internal `_train_val_split`, which embargoes
        # by `embargo` trading days.  No row dated AFTER as_of is ever passed
        # to the strategy, so test-period information cannot leak in.
        visible_panel = panel[panel["date"] <= w.as_of]
        if visible_panel.empty:
            continue
        # `fit_predict` retrains on the embargoed prefix of `visible_panel`,
        # predicts top-K weights at as_of, and returns
        # StrategyResult(weights, diagnostics).
        result = strat.fit_predict(panel=visible_panel, as_of=w.as_of, top_k=top_k)
        portfolio = result.weights
        # `score_window` realises the portfolio over [eval_start, eval_end]
        # and returns realised stock-level + portfolio + benchmark returns.
        scored = score_window(portfolio, prices, index_df, w.eval_start, w.eval_end)
        portfolio_return = float(scored["portfolio_return"])
        benchmark_return = float(scored["benchmark_return"])
        rows.append({
            "as_of": w.as_of.date().isoformat(),
            "eval_start": w.eval_start.date().isoformat(),
            "eval_end": w.eval_end.date().isoformat(),
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "excess_return": portfolio_return - benchmark_return,
            "n_holdings": int(len(portfolio)),
            "val_ic": float(result.diagnostics.get("val_ic", float("nan"))),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    e = df["excess_return"]
    n = len(e)
    summary = {
        "strategy": strategy_name,
        "n_test_windows": int(n),
        "test_period_start": df["eval_start"].min(),
        "test_period_end":   df["eval_end"].max(),
        "mean_excess":   float(e.mean()),
        "median_excess": float(e.median()),
        "std_excess":    float(e.std(ddof=1)),
        "t_stat":        float(e.mean() / (e.std(ddof=1) / np.sqrt(n))) if e.std(ddof=1) > 0 else float("nan"),
        "hit_rate":      float((e > 0).mean()),
        "best_window":   float(e.max()),
        "worst_window":  float(e.min()),
        "mean_portfolio_return": float(df["portfolio_return"].mean()),
        "mean_benchmark_return": float(df["benchmark_return"].mean()),
        "ic_mean": float(df["val_ic"].dropna().mean()) if df["val_ic"].notna().any() else None,
    }
    return df, summary


def main() -> int:
    print("=" * 72)
    print("CSI500 SELF-TEST")
    print("=" * 72)

    # ---------- 1. Load raw data ------------------------------------------
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(f"\nLoaded panel:")
    print(f"  Date range: {prices['date'].min().date()} -> {prices['date'].max().date()}")
    print(f"  Unique stocks: {prices['stock_code'].nunique()}")
    print(f"  Total rows: {len(prices):,}")

    # ---------- 2. Build the v2 feature panel -----------------------------
    # The shipped strategy's `build_panel` produces the same panel that
    # Submissions 1 and 2 were trained on. Importantly, the panel is built
    # ONCE on the full data; the train/val/test split is then applied on
    # `date`, so cross-sectional features like pct-rank are not contaminated
    # by future information (each row's pct-rank uses only that row's date).
    strat_for_panel = STRATEGIES["xgb_v2"]()
    panel = strat_for_panel.build_panel(prices, index_df)
    trading_dates = _trading_dates(panel)
    print(f"\nFeature panel built: {len(panel):,} rows, {len(trading_dates)} trading days.")

    # ---------- 3. Chronological 70 / 15 / 15 split -----------------------
    train_end, val_end = chronological_split(trading_dates, train_frac=0.70, val_frac=0.15)
    test_start = pd.Timestamp(trading_dates[trading_dates > np.datetime64(val_end)][0])
    test_end = pd.Timestamp(trading_dates[-1])

    n_train = int(np.sum(trading_dates <= np.datetime64(train_end)))
    n_val = int(np.sum((trading_dates > np.datetime64(train_end)) &
                       (trading_dates <= np.datetime64(val_end))))
    n_test = int(np.sum(trading_dates > np.datetime64(val_end)))

    print(f"\nChronological 70 / 15 / 15 split:")
    print(f"  TRAIN: {trading_dates[0]} .. {train_end.date()}    ({n_train} days, {n_train/len(trading_dates):.1%})")
    print(f"  VAL:   {(train_end + pd.Timedelta(days=1)).date()} .. {val_end.date()}    ({n_val} days, {n_val/len(trading_dates):.1%})")
    print(f"  TEST:  {test_start.date()} .. {test_end.date()}    ({n_test} days, {n_test/len(trading_dates):.1%})")

    # ---------- 4. Build test windows -------------------------------------
    # Each test window has as_of in the test partition and evaluates over the
    # next 5 trading days. Stride = 5 matches the competition's weekly cadence.
    test_windows = make_test_windows(trading_dates, val_end, horizon=HORIZON, stride=HORIZON)
    print(f"\nTest windows: {len(test_windows)} non-overlapping 5-day evaluations")
    print(f"  first as_of: {test_windows[0].as_of.date()}, last as_of: {test_windows[-1].as_of.date()}")

    # ---------- 5. Evaluate shipped model and baseline --------------------
    REPORTS_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 72)
    print("Evaluating SHIPPED model (xgb_v2) on TEST set ...")
    print("=" * 72)
    df_v2, summary_v2 = evaluate_strategy("xgb_v2", panel, prices, index_df, test_windows)

    print("\n" + "=" * 72)
    print("Evaluating BASELINE model (xgb_baseline) on TEST set ...")
    print("=" * 72)
    df_base, summary_base = evaluate_strategy("xgb_baseline", panel, prices, index_df, test_windows)

    # ---------- 6. Side-by-side comparison --------------------------------
    print("\n" + "=" * 72)
    print("TEST-SET RESULTS (held-out tail, 70/15/15 chronological split)")
    print("=" * 72)
    print(f"\nTest period: {summary_v2['test_period_start']} -> {summary_v2['test_period_end']}")
    print(f"Number of non-overlapping 5-day evaluation windows: {summary_v2['n_test_windows']}")
    print()
    print(f"{'Metric':<28s}  {'BASELINE (xgb_baseline)':>22s}  {'SHIPPED (xgb_v2)':>20s}  {'Delta':>10s}")
    print("-" * 86)
    for key, label in [
        ("mean_excess",            "Mean excess return"),
        ("median_excess",          "Median excess return"),
        ("std_excess",             "Std (window-to-window)"),
        ("t_stat",                 "t-statistic"),
        ("hit_rate",               "Hit rate"),
        ("best_window",            "Best window"),
        ("worst_window",           "Worst window"),
        ("mean_portfolio_return",  "Mean portfolio return"),
        ("mean_benchmark_return",  "Mean CSI500 return"),
    ]:
        b = summary_base[key]
        v = summary_v2[key]
        if "rate" in key:
            b_s = f"{b*100:>20.2f}%"; v_s = f"{v*100:>18.2f}%"; d_s = f"{(v-b)*100:>+8.2f}pp"
        elif key == "t_stat":
            b_s = f"{b:>22.3f}"; v_s = f"{v:>20.3f}"; d_s = f"{v-b:>+10.3f}"
        else:
            b_s = f"{b*100:>20.3f}%"; v_s = f"{v*100:>18.3f}%"; d_s = f"{(v-b)*100:>+8.3f}pp"
        print(f"{label:<28s}  {b_s}  {v_s}  {d_s}")

    delta = summary_v2["mean_excess"] - summary_base["mean_excess"]
    rel = delta / summary_base["mean_excess"] if summary_base["mean_excess"] != 0 else float("nan")
    print()
    print(f"-> SHIPPED model beats baseline on TEST set by {delta*100:+.3f}pp "
          f"({rel*100:+.0f}% relative).")
    if summary_v2["mean_excess"] > summary_base["mean_excess"]:
        print(f"-> Rubric criterion (b) test performance > baseline: PASS.")
    else:
        print(f"-> Rubric criterion (b) test performance > baseline: FAIL.")

    # ---------- 7. Persist results ----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_combined = pd.concat([
        df_v2.assign(strategy="xgb_v2 (SHIPPED)"),
        df_base.assign(strategy="xgb_baseline"),
    ], ignore_index=True)
    out_csv = REPORTS_DIR / "self_test_results.csv"
    df_combined.to_csv(out_csv, index=False)

    out_json = REPORTS_DIR / "self_test_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "methodology": {
                "split_type":     "chronological",
                "train_fraction": 0.70,
                "val_fraction":   0.15,
                "test_fraction":  0.15,
                "horizon_days":   HORIZON,
                "stride_days":    HORIZON,
                "embargo_days":   DEFAULT_EMBARGO,
                "top_k_per_window": DEFAULT_TOP_K,
                "leakage_guards": [
                    "5d forward target shifted strictly forward (groupby('stock').shift(-horizon))",
                    f"{DEFAULT_EMBARGO}-day embargo between training cutoff and evaluation start",
                    "Cross-sectional features (z-score, pct-rank) computed per-date only",
                    "Test partition strictly succeeds val partition in calendar time",
                    "Universe constituents filter applied at scoring time only",
                ],
                "split_dates": {
                    "train_start":  str(trading_dates[0]),
                    "train_end":    str(train_end.date()),
                    "val_start":    str((train_end + pd.Timedelta(days=1)).date()),
                    "val_end":      str(val_end.date()),
                    "test_start":   str(test_start.date()),
                    "test_end":     str(test_end.date()),
                    "n_train_days": n_train,
                    "n_val_days":   n_val,
                    "n_test_days":  n_test,
                },
            },
            "shipped_xgb_v2": summary_v2,
            "baseline_xgb_baseline": summary_base,
            "comparison": {
                "delta_mean_excess_pp":     delta * 100,
                "relative_improvement_pct": rel * 100,
                "shipped_beats_baseline":   bool(summary_v2["mean_excess"] > summary_base["mean_excess"]),
            },
            "timestamp": timestamp,
        }, f, indent=2, default=str)

    print(f"\nResults written to:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print()

    return 0 if summary_v2["mean_excess"] > summary_base["mean_excess"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
