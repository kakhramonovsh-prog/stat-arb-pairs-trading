"""
Step 6 -- pre-registered cross-sectional (Avellaneda-Lee) extension.

See PREREGISTRATION_AL.md. One run per span.

Usage:
    python scripts/06_crosssec.py --span is
    python scripts/06_crosssec.py --span oos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import crosssec, data, metrics

SPANS = {
    "is": ("2011-01-03", config.IS_END_DATE),
    "oos": (config.OOS_START_DATE, config.END_DATE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", choices=sorted(SPANS), default="is")
    args = ap.parse_args()

    start, end = SPANS[args.span]
    prices = data.load_prices()

    print(f"Cross-sectional strategy, {start} .. {end}")
    daily, trades = crosssec.run_cross_sectional(prices, start, end)

    bench = prices[config.BENCHMARK] if config.BENCHMARK in prices.columns else None
    print()
    print(metrics.format_report(
        f"Avellaneda-Lee cross-sectional -- {args.span.upper()}",
        daily, trades, bench))
    print(f"\navg open positions: {daily['n_open'].mean():.1f}   "
          f"avg eligible: {daily['n_eligible'].mean():.1f}")

    daily.to_csv(config.TABLES / f"daily_{args.span}_crosssec.csv")
    trades.to_csv(config.TABLES / f"trades_{args.span}_crosssec.csv", index=False)
    print(f"wrote {config.TABLES / f'daily_{args.span}_crosssec.csv'}")


if __name__ == "__main__":
    main()
