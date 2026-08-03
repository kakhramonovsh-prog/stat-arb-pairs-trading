"""
Step 7 -- a confidence interval for the look-ahead bias gap.

The headline claim is that the look-ahead normalisation is worth about +6.8
percentage points of annualised return. A point estimate deserves an interval:
this script runs both variants on identical cohorts, takes the DAILY difference
between the biased and honest return series, and bootstraps its annualised
mean with a moving-block bootstrap.

Blocks, not IID draws: daily strategy returns are autocorrelated (positions
persist for days), and an IID bootstrap would understate the variance of the
mean. A 21-day block length matches the Newey-West lag window used everywhere
else in the study.

In-sample only -- this is a diagnostic of the engine, not a strategy result.

Usage:
    python scripts/07_bootstrap_bias.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src import backtest, data, pairs

N_BOOT = 10_000
BLOCK = 21
SEED = 20260802


def block_bootstrap_mean(x: np.ndarray, n_boot: int, block: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap distribution of the mean of x."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    return means


def main() -> None:
    prices = data.load_prices()
    log_px = data.log_prices(prices)

    print("Selecting in-sample cohorts...")
    cohorts = pairs.run_selection(
        log_px, start=config.START_DATE, end=config.IS_END_DATE, verbose=False
    )

    print("Running honest and look-ahead variants on identical cohorts...")
    honest, _ = backtest.run_backtest(cohorts, prices, log_px, verbose=False)
    biased, _ = backtest.run_backtest(cohorts, prices, log_px, verbose=False,
                                      use_lookahead_moments=True)

    diff = (biased["net"] - honest["net"]).dropna()
    point = diff.mean() * 252

    rng = np.random.default_rng(SEED)
    boot = block_bootstrap_mean(diff.to_numpy(), N_BOOT, BLOCK, rng) * 252
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print()
    print(f"Look-ahead bias gap (annualised, net):")
    print(f"  point estimate   {point * 100:+.2f} pp")
    print(f"  95% CI           [{lo * 100:+.2f}, {hi * 100:+.2f}] pp")
    print(f"  bootstrap        {N_BOOT} draws, {BLOCK}-day moving blocks, seed {SEED}")
    print(f"  days             {len(diff)}")

    out = config.TABLES / "bootstrap_bias.csv"
    pd.DataFrame({
        "metric": ["gap_ann_point", "gap_ann_ci_lo", "gap_ann_ci_hi",
                   "n_boot", "block_days", "n_days", "seed"],
        "value": [point, lo, hi, N_BOOT, BLOCK, len(diff), SEED],
    }).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
