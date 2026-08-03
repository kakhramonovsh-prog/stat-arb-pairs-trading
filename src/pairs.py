"""
Rolling pair selection.

This module is where look-ahead bias is either avoided or introduced, so the
design is stated explicitly.

The sample is cut into a sequence of non-overlapping cohorts. Each cohort has a
FORMATION window of 252 trading days and a TRADING window of the following 126
days:

    |<-- formation: select pairs, fit beta, fit mu and sigma -->|<-- trade -->|
                                                                 |<-- next formation ...

Every quantity used to generate a trade -- which pairs are in the book, the
hedge ratio, and the mean and standard deviation that define the z-score -- is
estimated on formation data only and then frozen. The trading window never
informs its own signals.

This is the design in Gatev, Goetzmann & Rouwenhorst (2006). It is more
conservative than the common practice of estimating the cointegrating vector
once on the full sample, which is the single largest source of overstated
pairs-trading results in the retail literature: a hedge ratio fitted on data
that includes the trading period already knows where the spread is going.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src import cointegration as ct


@dataclass
class Cohort:
    """One formation/trading cycle."""

    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    selected: list[ct.PairResult] = field(default_factory=list)
    n_tested: int = 0
    n_passing_raw: int = 0      # significant before FDR correction
    n_passing_fdr: int = 0      # significant after FDR correction
    n_passing_halflife: int = 0  # and inside the half-life band
    pi0: float = float("nan")           # Storey proportion of true nulls
    expected_fdp: float = float("nan")  # expected noise fraction of the book

    @property
    def label(self) -> str:
        return f"{self.trading_start.date()}..{self.trading_end.date()}"


def build_windows(
    dates: pd.DatetimeIndex,
    formation_days: int = config.FORMATION_DAYS,
    trading_days: int = config.TRADING_DAYS,
    start: str | None = None,
    end: str | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Cut the calendar into (formation_start, formation_end, trading_start,
    trading_end) tuples.

    Boundaries are taken from the actual trading calendar rather than by adding
    fixed numbers of calendar days, so window lengths are exact in trading days
    and do not drift with holidays.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    windows = []
    i = formation_days
    while i + trading_days <= len(dates):
        windows.append((dates[i - formation_days], dates[i - 1],
                        dates[i], dates[i + trading_days - 1]))
        i += trading_days
    return windows


def select_for_window(
    log_px: pd.DataFrame,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    candidates: list[tuple[str, str, str]] | None = None,
    verbose: bool = False,
) -> tuple[list[ct.PairResult], dict[str, int]]:
    """
    Screen every candidate pair on one formation window and return the
    selected book plus a funnel of how many pairs survived each filter.

    The funnel is reported because it is diagnostic. If 1,300 pairs are tested
    and 400 pass at a nominal 5% level, that is far more than the ~65 expected
    under the null and is real evidence of widespread cointegration. If instead
    80 pass, that is indistinguishable from chance and the FDR step will
    correctly discard nearly all of them.
    """
    candidates = candidates or config.candidate_pairs()
    window = log_px.loc[(log_px.index >= formation_start) & (log_px.index <= formation_end)]
    min_obs = int(len(window) * config.MIN_OBS_FRACTION)

    results: list[ct.PairResult] = []
    for sector, a, b in candidates:
        r = ct.test_pair(window, sector, a, b, min_obs=min_obs)
        if r is not None:
            results.append(r)

    funnel = {"tested": len(results), "raw": 0, "fdr": 0, "halflife": 0,
              "pi0": float("nan"), "expected_fdp": float("nan")}
    if not results:
        return [], funnel

    # Two tests were run per pair (both regression directions) and the more
    # significant kept. Doubling each p-value is the Bonferroni correction for
    # that pair-level choice, so the minimum-taking does not sneak in an
    # uncounted look at the data.
    pvals = np.minimum(np.array([r.eg_pvalue for r in results]) * 2.0, 1.0)
    funnel["raw"] = int((pvals < config.EG_PVALUE_THRESHOLD).sum())

    # --- Multiple-testing diagnostics -------------------------------------
    # These are REPORTED, not used as a gate. Across ~1,300 candidate pairs the
    # evidence for cointegration is real in aggregate but too diffuse to
    # attribute to individual pairs: Benjamini-Hochberg at a 5% FDR typically
    # rejects nothing, because it needs the k-th smallest p-value below
    # k*alpha/m and the observed p-values sit an order of magnitude above that
    # line. Gating on BH would produce an empty book in almost every cohort,
    # which answers a question the strategy is not asking.
    #
    # The question that matters for a portfolio is not "is this specific pair
    # individually significant after correcting for 1,300 tests" but "does a
    # book formed from the strongest-evidence pairs earn its keep on data that
    # played no part in selecting it". That is settled by the out-of-sample
    # trading window, which is a stricter test than any in-sample correction:
    # a spurious pair has no reason to keep reverting after formation ends.
    #
    # So selection is rank-based, and the expected fraction of the book that is
    # noise is computed and reported for every cohort.
    reject, _ = ct.fdr_filter(pvals, config.FDR_ALPHA)
    funnel["fdr"] = int(reject.sum())
    funnel["pi0"] = ct.storey_pi0(pvals)

    survivors = [
        r for r, p in zip(results, pvals)
        if p < config.EG_PVALUE_THRESHOLD
        and config.MIN_HALF_LIFE <= r.half_life <= config.MAX_HALF_LIFE
        and np.isfinite(r.spread_std) and r.spread_std > 0
    ]
    funnel["halflife"] = len(survivors)

    # Rank by the Engle-Granger statistic: more negative is stronger evidence
    # against a unit root in the residual, i.e. a more reliably stationary
    # spread. Ranking on the statistic rather than the p-value avoids ties at
    # the floor of the tabulated p-value grid.
    survivors.sort(key=lambda r: r.eg_stat)
    selected = survivors[: config.MAX_PAIRS_PER_WINDOW]

    # A single stock can otherwise appear in many selected pairs and quietly
    # dominate the book's risk. Cap exposure by taking each pair only if
    # neither leg has already been used twice in this cohort.
    usage: dict[str, int] = {}
    capped: list[ct.PairResult] = []
    for r in selected:
        if usage.get(r.y, 0) >= 2 or usage.get(r.x, 0) >= 2:
            continue
        usage[r.y] = usage.get(r.y, 0) + 1
        usage[r.x] = usage.get(r.x, 0) + 1
        capped.append(r)

    funnel["expected_fdp"] = ct.expected_fdp(pvals, len(capped), pi0=funnel["pi0"])

    if verbose:
        print(f"    tested {funnel['tested']}, p<0.05 {funnel['raw']}, "
              f"post-FDR {funnel['fdr']}, half-life band {funnel['halflife']}, "
              f"book {len(capped)}, pi0 {funnel['pi0']:.2f}, "
              f"E[FDP] {funnel['expected_fdp']:.2f}")

    return capped, funnel


def run_selection(
    log_px: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    verbose: bool = True,
) -> list[Cohort]:
    """Build every cohort over the requested span and populate its book."""
    windows = build_windows(log_px.index, start=start, end=end)
    candidates = config.candidate_pairs()

    if verbose:
        print(f"{len(windows)} cohorts, {len(candidates)} candidate pairs per cohort")

    cohorts: list[Cohort] = []
    for fs, fe, ts, te in windows:
        cohort = Cohort(fs, fe, ts, te)
        if verbose:
            print(f"  formation {fs.date()}..{fe.date()} -> trading {ts.date()}..{te.date()}")
        selected, funnel = select_for_window(log_px, fs, fe, candidates, verbose=verbose)
        cohort.selected = selected
        cohort.n_tested = funnel["tested"]
        cohort.n_passing_raw = funnel["raw"]
        cohort.n_passing_fdr = funnel["fdr"]
        cohort.n_passing_halflife = funnel["halflife"]
        cohort.pi0 = funnel["pi0"]
        cohort.expected_fdp = funnel["expected_fdp"]
        cohorts.append(cohort)

    return cohorts


def cohorts_to_frame(cohorts: list[Cohort]) -> pd.DataFrame:
    """Flatten selected pairs across cohorts into one tidy table."""
    rows = []
    for c in cohorts:
        for r in c.selected:
            row = r.to_dict()
            row.update(
                formation_start=c.formation_start,
                formation_end=c.formation_end,
                trading_start=c.trading_start,
                trading_end=c.trading_end,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def funnel_to_frame(cohorts: list[Cohort]) -> pd.DataFrame:
    """Per-cohort screening funnel, for the paper's selection table."""
    return pd.DataFrame([
        {
            "trading_start": c.trading_start,
            "trading_end": c.trading_end,
            "tested": c.n_tested,
            "raw_p05": c.n_passing_raw,
            "post_fdr": c.n_passing_fdr,
            "in_halflife_band": c.n_passing_halflife,
            "book_size": len(c.selected),
            "expected_false_positives": round(c.n_tested * config.EG_PVALUE_THRESHOLD, 1),
            "pi0": round(c.pi0, 3),
            "expected_fdp": round(c.expected_fdp, 3),
        }
        for c in cohorts
    ])
