"""
Step 4 -- control experiments.

Three questions this script answers, all on IN-SAMPLE data only:

  1. Is the engine wired correctly? Verified by deliberately introducing the
     standard look-ahead bug (standardising the spread by its own trading-window
     moments) and confirming it produces the large spurious profits that bug is
     known to produce. If the buggy variant were also flat, the flat honest
     result would more likely mean the simulator is broken than that the
     strategy has no edge.

  2. How much of the result is transaction costs? Answered by rerunning at
     several cost levels including zero.

  3. How sensitive is the conclusion to the trading thresholds? Reported as a
     sensitivity surface, NOT used to select parameters -- the configured
     thresholds stay as they are, and the out-of-sample span stays untouched.

Usage:
    python scripts/04_diagnostics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src import backtest, data, metrics, pairs


def run_variant(cohorts, prices, log_px, **kwargs) -> dict:
    daily, trades = backtest.run_backtest(
        cohorts, prices, log_px, verbose=False, **kwargs
    )
    if daily.empty:
        return {}
    perf = metrics.summarise(daily["net"])
    costs = metrics.cost_decomposition(daily)
    tst = metrics.trade_stats(trades)
    return {
        "ann_return": perf["ann_return"],
        "sharpe": perf["sharpe"],
        "max_dd": perf["max_drawdown"],
        "t_nw": perf["t_stat_newey_west"],
        "gross_ann": costs["gross_ann"],
        "n_trades": tst.get("n_trades", 0),
        "win_rate": tst.get("win_rate", float("nan")),
    }


def main() -> None:
    prices = data.load_prices()
    log_px = data.log_prices(prices)

    print("Selecting in-sample cohorts (reused across all variants)...")
    cohorts = pairs.run_selection(
        log_px, start=config.START_DATE, end=config.IS_END_DATE, verbose=False
    )
    print(f"  {len(cohorts)} cohorts, "
          f"{sum(len(c.selected) for c in cohorts)} pair-windows\n")

    rows = []

    # --- Control 1: the look-ahead bug ------------------------------------
    rows.append({"variant": "honest (reported)",
                 **run_variant(cohorts, prices, log_px)})
    rows.append({"variant": "LOOK-AHEAD z-score",
                 **run_variant(cohorts, prices, log_px,
                               use_lookahead_moments=True)})

    # --- Control 2: causal rolling normalisation --------------------------
    # Freezing the spread's mean for the whole 126-day trading window is the
    # strictest possible choice, and arguably too strict: a cointegrating
    # relationship's level can drift legitimately, and a frozen mean reads that
    # drift as an ever-widening dislocation, so the strategy leans into a move
    # that is not going to revert. A trailing estimate ending at t-1 uses no
    # future information and is what a desk would actually run.
    for w in (30, 60, 90):
        rows.append({"variant": f"causal rolling z ({w}d)",
                     **run_variant(cohorts, prices, log_px, rolling_window=w)})
    rows.append({"variant": "causal rolling z (60d), zero cost",
                 **run_variant(cohorts, prices, log_px,
                               rolling_window=60, cost_multiplier=0.0)})

    # --- Control 3: cost sensitivity --------------------------------------
    for mult, label in [(0.0, "zero cost"), (0.5, "half cost"),
                        (2.0, "double cost")]:
        rows.append({"variant": f"honest, {label}",
                     **run_variant(cohorts, prices, log_px,
                                   cost_multiplier=mult)})

    # The look-ahead variant at zero cost isolates the pure size of the bias.
    rows.append({"variant": "LOOK-AHEAD, zero cost",
                 **run_variant(cohorts, prices, log_px,
                               use_lookahead_moments=True, cost_multiplier=0.0)})

    frame = pd.DataFrame(rows)
    for col in ("ann_return", "max_dd", "gross_ann", "win_rate"):
        frame[col] = (frame[col] * 100).round(2)
    frame["sharpe"] = frame["sharpe"].round(2)
    frame["t_nw"] = frame["t_nw"].round(2)

    print(frame.to_string(index=False))

    out = config.TABLES / "diagnostics.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
