"""
Step 1 -- download and cache the price panel.

Usage:
    python scripts/01_fetch_data.py [--refresh]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src import data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if the cache exists")
    args = ap.parse_args()

    prices = data.load_prices(refresh=args.refresh)

    print()
    print(f"Panel:      {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"Date range: {prices.index.min().date()} .. {prices.index.max().date()}")

    expected = set(config.all_tickers())
    got = set(prices.columns)
    missing = sorted(expected - got)
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing)}")

    # Late arrivals matter for interpreting the early windows, so surface them.
    first_valid = prices.apply(lambda s: s.first_valid_index())
    late = first_valid[first_valid > pd.Timestamp(config.START_DATE) + pd.Timedelta(days=30)]
    if len(late):
        print(f"\nTickers with history starting after {config.START_DATE}:")
        for tkr, when in late.sort_values().items():
            print(f"  {tkr:6s} {when.date()}")


if __name__ == "__main__":
    main()
