"""
Tests for the statistical core.

The pattern throughout: simulate a process whose true parameters are known,
then check the estimator recovers them. A test that only checks "the function
runs" verifies nothing about correctness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import cointegration as ct

RNG = np.random.default_rng(42)


def _random_walk(n: int, sigma: float = 0.01) -> pd.Series:
    return pd.Series(np.cumsum(RNG.normal(0, sigma, n)) + 4.0)


def _cointegrated_pair(n: int = 500, beta: float = 1.4, theta: float = 0.15):
    """log_x a random walk; log_y = beta*log_x + OU noise. Cointegrated by
    construction, with a known hedge ratio and known mean-reversion speed."""
    log_x = _random_walk(n)
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] * (1 - theta) + RNG.normal(0, 0.01)
    log_y = beta * log_x + 0.5 + pd.Series(ou)
    return log_y, log_x


class TestEngleGranger:
    def test_detects_constructed_cointegration(self):
        log_y, log_x = _cointegrated_pair()
        stat, pval, beta, alpha = ct.engle_granger(log_y, log_x)
        assert pval < 0.01
        assert beta == pytest.approx(1.4, abs=0.1)

    def test_does_not_reject_independent_walks(self):
        # Two unrelated random walks: the null of no cointegration is TRUE.
        # A single draw can be unlucky, so require the median p-value across
        # draws to be comfortably high.
        pvals = []
        for _ in range(10):
            a, b = _random_walk(400), _random_walk(400)
            _, p, _, _ = ct.engle_granger(a, b)
            pvals.append(p)
        assert np.median(pvals) > 0.10


class TestHalfLife:
    def test_recovers_known_speed(self):
        # AR(1) with coefficient (1-theta): half-life = -ln2 / ln(1-theta).
        theta = 0.10
        n = 4000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = x[t - 1] * (1 - theta) + RNG.normal(0, 1.0)
        expected = -np.log(2) / np.log(1 - theta)
        got = ct.half_life(pd.Series(x))
        assert got == pytest.approx(expected, rel=0.25)

    def test_infinite_for_explosive_series(self):
        # b >= 0 in the AR(1) fit means no reversion at all; the estimator
        # must return inf rather than a negative half-life. An explosive
        # series guarantees a positive b estimate.
        n = 300
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 1.01 * x[t - 1] + RNG.normal(0, 1.0)
        assert ct.half_life(pd.Series(x)) == np.inf

    def test_random_walk_bias_is_understood(self):
        # Deliberate documentation-as-test. A pure random walk has NO true
        # mean reversion, yet its estimated half-life at 500 observations has
        # a median around only ~60 days -- the same downward finite-sample
        # bias that motivates Dickey-Fuller critical values. This is exactly
        # why the study cannot rely on the half-life filter alone and must
        # report the expected false discovery proportion of the book: a
        # non-reverting spread can sneak inside the tradeable band easily.
        hls = [ct.half_life(_random_walk(500)) for _ in range(20)]
        med = float(np.median(hls))
        assert 20 < med < 500          # biased low, but not "tradeable-fast"


class TestMultipleTesting:
    def test_storey_pi0_near_one_on_uniform(self):
        p = RNG.uniform(0, 1, 2000)
        assert ct.storey_pi0(p) == pytest.approx(1.0, abs=0.06)

    def test_storey_pi0_below_one_with_signal(self):
        p = np.concatenate([RNG.uniform(0, 1, 1000),
                            RNG.uniform(0, 0.01, 300)])
        assert ct.storey_pi0(p) < 0.85

    def test_fdr_rejects_obvious_signal(self):
        p = np.concatenate([np.full(20, 1e-6), RNG.uniform(0, 1, 500)])
        reject, _ = ct.fdr_filter(p, alpha=0.05)
        assert reject[:20].all()

    def test_expected_fdp_bounds(self):
        p = RNG.uniform(0, 1, 1000)
        fdp = ct.expected_fdp(p, k=20)
        assert 0.0 <= fdp <= 1.0
