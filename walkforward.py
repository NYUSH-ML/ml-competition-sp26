"""
Walk-forward backtesting framework for the CSI500 stock-selection competition.

Goals
-----
- Run a strategy across many `as_of` dates with NO leakage between train and
  evaluation, exactly the way the live competition will work.
- Produce stable, comparable metrics so we can tell whether a new feature /
  model is actually better than the baseline (one-shot IC is misleading).

Design
------
For every `as_of` date in the chosen schedule we:
    1. Strategy sees the panel restricted to dates <= as_of - embargo and trains.
    2. Strategy predicts a portfolio for date == as_of.
    3. We score that portfolio over the realized window
       [as_of + 1 trading day, as_of + horizon trading days] vs CSI500.
    4. We log: validation IC, portfolio return, benchmark return, excess return,
       hit rate of beating benchmark, top 5 holdings.

Aggregate metrics
-----------------
- mean_excess          : average excess return across windows
- median_excess        : robust central tendency
- std_excess           : window-to-window noise
- hit_rate             : share of windows beating CSI500
- t_stat               : mean / (std / sqrt(n))   one-sample t against zero
- ic_mean / ic_std     : if the strategy reports val_ic
- ic_ir                : ic_mean / ic_std         (information ratio)

CLI examples
------------
    # Quick smoke test: 3 windows, last ~6 weeks
    python walkforward.py --strategy xgb_baseline --n-windows 3

    # Full sweep: monthly windows from 2025-09 onwards, write results
    python walkforward.py --strategy xgb_baseline --start 20250901 --stride 5

    # Compare two strategies side by side
    python walkforward.py --strategy xgb_baseline --strategy xgb_baseline --tag run_a
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from score_submission import score_window
from strategies import (
    DEFAULT_EMBARGO, DEFAULT_TOP_K, MAX_WEIGHT, MIN_STOCKS, STRATEGIES,
)

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"


# ----------------------------------------------------------------------------
# Window scheduling
# ----------------------------------------------------------------------------

def _trading_dates(panel: pd.DataFrame) -> np.ndarray:
    return np.sort(panel["date"].unique())


@dataclass(frozen=True)
class Window:
    as_of: pd.Timestamp        # last training date the strategy may see
    eval_start: pd.Timestamp   # first trading day in scoring window
    eval_end: pd.Timestamp     # last trading day in scoring window


def make_windows(
    trading_dates: np.ndarray,
    *,
    horizon: int = 5,
    stride: int = 5,
    n_windows: int | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    min_history: int = 120,
) -> list[Window]:
    """Generate (as_of, eval_start, eval_end) tuples that fit inside the data.

    `horizon` = trading days held in evaluation window (5 for week2-style).
    `stride` = trading days between consecutive as_of dates.  Use stride >= horizon
    to keep evaluation windows non-overlapping (cleaner statistics).
    `min_history` = required trading days before the first as_of so that the
    strategy has enough data to train.
    """
    n = len(trading_dates)
    if n < min_history + horizon + 1:
        raise RuntimeError(f"Only {n} dates available; need >= {min_history + horizon + 1}")

    earliest_idx = min_history
    latest_idx = n - horizon - 1

    windows: list[Window] = []
    for i in range(earliest_idx, latest_idx + 1, stride):
        as_of = pd.Timestamp(trading_dates[i])
        eval_start = pd.Timestamp(trading_dates[i + 1])
        eval_end = pd.Timestamp(trading_dates[i + horizon])
        if start is not None and as_of < pd.Timestamp(start):
            continue
        if end is not None and as_of > pd.Timestamp(end):
            continue
        windows.append(Window(as_of, eval_start, eval_end))

    if n_windows is not None and n_windows > 0 and len(windows) > n_windows:
        windows = windows[-n_windows:]   # most recent N windows
    return windows


# ----------------------------------------------------------------------------
# Per-window evaluation
# ----------------------------------------------------------------------------

@dataclass
class WindowResult:
    as_of: str
    eval_start: str
    eval_end: str
    val_ic: float
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    n_holdings: int
    max_weight: float
    top_5: str
    diagnostics: dict = field(default_factory=dict)


def run_window(
    strategy,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    window: Window,
    top_k: int = DEFAULT_TOP_K,
    debug: bool = False,
) -> WindowResult:
    """Train + predict at as_of, score over [eval_start, eval_end]."""
    # The strategy must only see history strictly before as_of + 1.
    # We pass the full panel; strategies use `_train_val_split` which respects
    # `as_of - embargo`.  As an extra guardrail we'd hash the prediction-frame
    # date to match window.as_of.
    sr = strategy.fit_predict(panel, window.as_of, top_k=top_k)
    weights = sr.weights

    # Sanity: portfolio passes the competition rules.
    assert abs(weights.sum() - 1.0) < 1e-4, f"weights sum to {weights.sum()}"
    assert (weights <= MAX_WEIGHT + 1e-9).all(), "10% cap violated"
    assert (weights > 0).sum() >= MIN_STOCKS, "fewer than 30 names"

    score = score_window(weights, prices, index_df, window.eval_start, window.eval_end)

    top5 = ",".join(weights.sort_values(ascending=False).head(5).index.tolist())
    return WindowResult(
        as_of=window.as_of.date().isoformat(),
        eval_start=window.eval_start.date().isoformat(),
        eval_end=window.eval_end.date().isoformat(),
        val_ic=float(sr.diagnostics.get("val_ic", float("nan"))),
        portfolio_return=score["portfolio_return"],
        benchmark_return=score["benchmark_return"],
        excess_return=score["excess_return"],
        n_holdings=int((weights > 0).sum()),
        max_weight=float(weights.max()),
        top_5=top5,
        diagnostics=sr.diagnostics if debug else {},
    )


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate(results: list[WindowResult]) -> dict:
    if not results:
        return {}
    df = pd.DataFrame([asdict(r) for r in results])

    excess = df["excess_return"].to_numpy()
    n = len(excess)
    mean_e = float(excess.mean())
    std_e = float(excess.std(ddof=1)) if n > 1 else float("nan")
    hit_rate = float((excess > 0).mean())
    t_stat = float(mean_e / (std_e / np.sqrt(n))) if std_e and not np.isnan(std_e) and std_e > 0 else float("nan")

    ic = df["val_ic"].dropna().to_numpy()
    ic_mean = float(ic.mean()) if len(ic) else float("nan")
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    ic_ir = float(ic_mean / ic_std) if ic_std and not np.isnan(ic_std) and ic_std > 0 else float("nan")

    return {
        "n_windows": n,
        "mean_excess": mean_e,
        "median_excess": float(np.median(excess)),
        "std_excess": std_e,
        "hit_rate": hit_rate,
        "t_stat": t_stat,
        "best_window": float(excess.max()),
        "worst_window": float(excess.min()),
        "mean_portfolio_return": float(df["portfolio_return"].mean()),
        "mean_benchmark_return": float(df["benchmark_return"].mean()),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
    }


# ----------------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------------

def _fmt_pct(x: float, width: int = 8) -> str:
    if pd.isna(x):
        return "nan".rjust(width)
    return f"{x*100:+.2f}%".rjust(width)


def print_report(strategy_name: str, results: list[WindowResult], summary: dict) -> None:
    print(f"\n=== walk-forward report: {strategy_name} ===")
    print(f"{'as_of':<12} {'eval_window':<24} {'val_IC':>8} "
          f"{'port':>8} {'bench':>8} {'excess':>8}  top5")
    print("-" * 110)
    for r in results:
        win = f"{r.eval_start} -> {r.eval_end}"
        ic = f"{r.val_ic:+.3f}" if not pd.isna(r.val_ic) else "  nan"
        print(f"{r.as_of:<12} {win:<24} {ic:>8} "
              f"{_fmt_pct(r.portfolio_return)} {_fmt_pct(r.benchmark_return)} "
              f"{_fmt_pct(r.excess_return)}  {r.top_5}")
    print("-" * 110)
    if summary:
        print(f"  windows         : {summary['n_windows']}")
        print(f"  excess return   : mean {_fmt_pct(summary['mean_excess']).strip()}, "
              f"median {_fmt_pct(summary['median_excess']).strip()}, "
              f"std {_fmt_pct(summary['std_excess']).strip()}")
        print(f"  best / worst    : {_fmt_pct(summary['best_window']).strip()} / "
              f"{_fmt_pct(summary['worst_window']).strip()}")
        print(f"  hit rate        : {summary['hit_rate']*100:.1f}%  "
              f"({int(summary['hit_rate'] * summary['n_windows'])}/{summary['n_windows']} beat benchmark)")
        if not pd.isna(summary['t_stat']):
            print(f"  t-stat (excess) : {summary['t_stat']:+.2f}  "
                  f"(>1.96 ~ p<0.05, but small N!)")
        if not pd.isna(summary['ic_mean']):
            print(f"  rank IC         : mean {summary['ic_mean']:+.4f}, "
                  f"std {summary['ic_std']:.4f}, IR {summary['ic_ir']:+.2f}")


def save_report(
    out_dir: Path,
    tag: str,
    strategy_name: str,
    results: list[WindowResult],
    summary: dict,
    cli_args: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{tag or strategy_name}_{ts}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    pd.DataFrame([asdict(r) for r in results]).to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(
            {"strategy": strategy_name, "args": cli_args, "summary": summary},
            f, indent=2, default=str,
        )
    return csv_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="xgb_baseline",
                   choices=sorted(STRATEGIES.keys()),
                   help="strategy name registered in strategies.STRATEGIES")
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--horizon", type=int, default=5,
                   help="trading days held in each eval window (default 5)")
    p.add_argument("--stride", type=int, default=5,
                   help="trading days between consecutive as_of dates (default 5)")
    p.add_argument("--n-windows", type=int, default=None,
                   help="if set, only use the most recent N windows")
    p.add_argument("--start", default=None,
                   help="earliest as_of (YYYYMMDD); default = let min-history decide")
    p.add_argument("--end", default=None,
                   help="latest as_of (YYYYMMDD); default = use everything")
    p.add_argument("--min-history", type=int, default=120,
                   help="minimum trading days of history before first as_of")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--tag", default="",
                   help="prefix for the saved report files")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="keep per-window strategy diagnostics in the saved report")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} -> {prices['date'].max().date()}")

    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    strategy = STRATEGIES[args.strategy]()
    print(f">> Running strategy: {strategy.name}")
    print(">> Building feature panel (one-time, strategy-specific)")
    panel = strategy.build_panel(prices, index_df)
    print(f"   panel: {len(panel):,} rows, {panel.shape[1]} columns")

    trading_dates = _trading_dates(panel)
    windows = make_windows(
        trading_dates,
        horizon=args.horizon,
        stride=args.stride,
        n_windows=args.n_windows,
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
        min_history=args.min_history,
    )
    print(f">> {len(windows)} window(s) scheduled "
          f"(horizon={args.horizon}d, stride={args.stride}d)")
    if not windows:
        raise SystemExit("No windows produced; widen --start/--end or lower --min-history.")

    results: list[WindowResult] = []
    for i, w in enumerate(windows, 1):
        print(f"   [{i}/{len(windows)}] as_of={w.as_of.date()} "
              f"eval={w.eval_start.date()}..{w.eval_end.date()}", flush=True)
        try:
            r = run_window(strategy, panel, prices, index_df, w,
                           top_k=args.top_k, debug=args.debug)
            results.append(r)
        except Exception as e:
            print(f"      FAILED: {e}")

    summary = aggregate(results)
    print_report(strategy.name, results, summary)

    if not args.no_save:
        csv_path = save_report(
            REPORTS_DIR, args.tag, strategy.name, results, summary,
            cli_args=vars(args),
        )
        print(f"\n>> Saved report to {csv_path}")


if __name__ == "__main__":
    main()
