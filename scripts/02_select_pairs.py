"""
Step 2 -- screen the universe for cointegrated pairs, cohort by cohort.

Writes two tables to results/tables/:
    pairs_<span>.csv   one row per selected pair per cohort
    funnel_<span>.csv  the screening funnel and multiple-testing diagnostics

Usage:
    python scripts/02_select_pairs.py --span is     # in-sample  (2010-2021)
    python scripts/02_select_pairs.py --span oos    # hold-out   (2022-)
    python scripts/02_select_pairs.py --span full
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src import data, pairs

SPANS = {
    "is": (config.START_DATE, config.IS_END_DATE),
    "oos": (config.OOS_START_DATE, config.END_DATE),
    "full": (config.START_DATE, config.END_DATE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", choices=sorted(SPANS), default="is")
    args = ap.parse_args()

    start, end = SPANS[args.span]

    prices = data.load_prices()
    log_px = data.log_prices(prices)

    # The out-of-sample span needs FORMATION_DAYS of history before its first
    # trading window, and that history legitimately comes from the in-sample
    # period -- using past prices to form a book is not look-ahead. The window
    # builder is therefore given a lead-in, and the resulting cohorts are then
    # filtered to those whose TRADING period starts inside the requested span.
    if args.span == "oos":
        prior = log_px.index[log_px.index < pd.Timestamp(config.OOS_START_DATE)]
        start = prior[-(config.FORMATION_DAYS + 5)].strftime("%Y-%m-%d")

    cohorts = pairs.run_selection(log_px, start=start, end=end)
    if args.span == "oos":
        cohorts = [c for c in cohorts
                   if c.trading_start >= pd.Timestamp(config.OOS_START_DATE)]

    pair_frame = pairs.cohorts_to_frame(cohorts)
    funnel = pairs.funnel_to_frame(cohorts)

    pair_path = config.TABLES / f"pairs_{args.span}.csv"
    funnel_path = config.TABLES / f"funnel_{args.span}.csv"
    pair_frame.to_csv(pair_path, index=False)
    funnel.to_csv(funnel_path, index=False)

    print()
    print(funnel.to_string(index=False))
    print()
    print(f"cohorts:        {len(cohorts)}")
    print(f"pairs selected: {len(pair_frame)}")
    if len(pair_frame):
        print(f"median half-life: {pair_frame['half_life'].median():.1f} days")
        print(f"mean E[FDP]:      {funnel['expected_fdp'].mean():.2f}")
        top = pair_frame.assign(pair=pair_frame.y + "/" + pair_frame.x)
        print("\nmost frequently selected pairs:")
        print(top["pair"].value_counts().head(10).to_string())
    print(f"\nwrote {pair_path}")
    print(f"wrote {funnel_path}")


if __name__ == "__main__":
    main()
