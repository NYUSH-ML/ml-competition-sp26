"""
Generate a competition submission file from any registered strategy.

Examples
--------
    # default: best-known strategy at the latest available trading day
    python make_submission.py

    # explicit
    python make_submission.py --strategy xgb_v2 --top-k 50 \
        --out submissions/xgb_v2_k50.csv

    # backtest on a historical as_of (data after that date is unused)
    python make_submission.py --as-of 20260331 --out submissions/bt_0331.csv

After writing, the file is automatically validated against the competition
rules (>=30 names, sum=1, max weight <= 10%, all codes in CSI500 universe).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategies import STRATEGIES, DEFAULT_TOP_K
from validate_submission import validate

DEFAULT_STRATEGY = "xgb_v2"
DEFAULT_OUT = "submissions/submission.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=DEFAULT_STRATEGY,
                   choices=sorted(STRATEGIES.keys()))
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--as-of", default=None,
                   help="YYYYMMDD; default = latest day in prices.parquet")
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--index", default="data/index.parquet")
    p.add_argument("--constituents", default="data/constituents.csv")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--no-validate", action="store_true",
                   help="skip the automatic submission rules check")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"== make_submission: strategy={args.strategy}, top_k={args.top_k} ==")

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    strategy = STRATEGIES[args.strategy]()
    print(f">> Building feature panel ...")
    panel = strategy.build_panel(prices, index_df)

    if args.as_of is None:
        as_of = pd.Timestamp(panel["date"].max())
    else:
        as_of = pd.Timestamp(args.as_of)
        if as_of not in panel["date"].values:
            # Snap to last trading date <= as_of
            avail = panel["date"][panel["date"] <= as_of]
            if avail.empty:
                raise SystemExit(f"No data on or before as-of={as_of.date()}")
            as_of = pd.Timestamp(avail.max())
    print(f">> as_of trading day: {as_of.date()}")

    result = strategy.fit_predict(panel, as_of, top_k=args.top_k)
    weights = result.weights
    print(f">> Portfolio: {len(weights)} names, "
          f"sum={weights.sum():.6f}, "
          f"max={weights.max():.4%}, min={weights.min():.4%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights = weights.sort_values(ascending=False)
    weights.index = weights.index.astype(str).str.zfill(6)
    weights.index.name = "stock_code"
    weights.name = "weight"
    weights.to_csv(out_path, float_format="%.6f")
    print(f">> Wrote {out_path}")

    # Sidecar diagnostics — useful for the report write-up
    diag = {
        "strategy": args.strategy,
        "top_k": args.top_k,
        "as_of": as_of.date().isoformat(),
        "n_holdings": int(len(weights)),
        "max_weight": float(weights.max()),
        "min_weight": float(weights.min()),
        "weight_sum": float(weights.sum()),
        "diagnostics": result.diagnostics,
    }
    diag_path = out_path.with_suffix(".diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=str)
    print(f">> Wrote {diag_path}")

    if not args.no_validate:
        cons_path = Path(args.constituents)
        errors = validate(out_path, cons_path if cons_path.exists() else None)
        if errors:
            print("\nVALIDATION FAILED:")
            for e in errors:
                print(f"  - {e}")
            raise SystemExit(1)
        print(">> Validation: OK")


if __name__ == "__main__":
    main()
