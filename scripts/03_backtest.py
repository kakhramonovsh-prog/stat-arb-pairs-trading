"""
Step 3 -- run the backtest over a chosen span.

Usage:
    python scripts/03_backtest.py --span is
    python scripts/03_backtest.py --span oos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src import backtest, data, metrics, pairs

SPANS = {
    "is": (config.START_DATE, config.IS_END_DATE),
    "oos": (config.OOS_START_DATE, config.END_DATE),
    "full": (config.START_DATE, config.END_DATE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", choices=sorted(SPANS), default="is")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--kalman", action="store_true",
                    help="pre-registered dynamic hedge ratio (PREREGISTRATION.md)")
    args = ap.parse_args()

    start, end = SPANS[args.span]

    prices = data.load_prices()
    log_px = data.log_prices(prices)

    if args.span == "oos":
        prior = log_px.index[log_px.index < pd.Timestamp(config.OOS_START_DATE)]
        start = prior[-(config.FORMATION_DAYS + 5)].strftime("%Y-%m-%d")

    print(f"Selecting pairs for span '{args.span}'...")
    cohorts = pairs.run_selection(log_px, start=start, end=end,
                                  verbose=not args.quiet)
    if args.span == "oos":
        cohorts = [c for c in cohorts
                   if c.trading_start >= pd.Timestamp(config.OOS_START_DATE)]

    print(f"\nBacktesting {len(cohorts)} cohorts...")
    daily, trades = backtest.run_backtest(cohorts, prices, log_px,
                                          verbose=not args.quiet,
                                          kalman=args.kalman)

    if daily.empty:
        print("\nNo trades generated.")
        return

    tag = f"{args.span}_kalman" if args.kalman else args.span
    label = "Kalman dynamic beta" if args.kalman else "static beta"

    bench = prices[config.BENCHMARK] if config.BENCHMARK in prices.columns else None
    print()
    print(metrics.format_report(
        f"Statistical arbitrage -- {args.span.upper()} ({label})",
        daily, trades, bench))

    daily.to_csv(config.TABLES / f"daily_{tag}.csv")
    trades.to_csv(config.TABLES / f"trades_{tag}.csv", index=False)
    print(f"\nwrote {config.TABLES / f'daily_{tag}.csv'}")
    print(f"wrote {config.TABLES / f'trades_{tag}.csv'}")


if __name__ == "__main__":
    main()
