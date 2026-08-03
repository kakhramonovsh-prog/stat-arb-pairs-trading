"""
Tests for the Kalman filter: parameter recovery on simulated data with a known
drifting hedge ratio, and the structural causality property.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import kalman as kf

RNG = np.random.default_rng(7)


def _drifting_beta_pair(n: int = 700, n_form: int = 250):
    """
    log_y = beta_t * log_x + noise. Beta is CONSTANT at 1.0 through the
    formation segment -- so the OLS seed is well identified -- then ramps to
    1.5 over the remainder. The x series is given real variance: with x
    hovering near a constant, beta drift is observationally equivalent to a
    trend in y and no estimator could attribute it, which would make the test
    diagnose an unidentifiable simulation rather than the filter.
    """
    log_x = pd.Series(np.cumsum(RNG.normal(0, 0.02, n)) + 4.0)
    true_beta = np.concatenate([
        np.full(n_form, 1.0),
        np.linspace(1.0, 1.5, n - n_form),
    ])
    log_y = pd.Series(true_beta * log_x.to_numpy() + RNG.normal(0, 0.004, n))
    return log_y, log_x, true_beta


class TestFilter:
    def test_tracks_drifting_beta(self):
        # Uses a fixed, adequate delta: the property under test is that the
        # FILTER tracks a drifting state, separate from what the formation
        # MLE would choose (which is tested below on the scenario it's for).
        log_y, log_x, true_beta = _drifting_beta_pair()
        theta0, P0, rv = kf.initial_state(log_y.iloc[:250], log_x.iloc[:250])
        out = kf.filter_pair(log_y, log_x, theta0, P0, rv, delta=1e-5)
        static_error = abs(theta0[0] - true_beta[-1])
        filter_error = abs(out["beta"].iloc[-1] - true_beta[-1])
        assert filter_error < static_error / 2

    def test_innovation_is_prediction_error(self):
        # A large shock to y on day t must show up in day t's innovation --
        # the filter cannot have absorbed it into the state BEFORE observing
        # it. This is the causality property stated in the pre-registration.
        log_y, log_x, _ = _drifting_beta_pair(300)
        log_y.iloc[200] += 0.05        # 5% shock
        theta0, P0, rv = kf.initial_state(log_y.iloc[:150], log_x.iloc[:150])
        out = kf.filter_pair(log_y, log_x, theta0, P0, rv)
        assert out["spread"].iloc[200] > 0.03

    def test_mle_rejects_degenerate_delta_for_stable_pair(self):
        # The scenario that produced the documented pre-registration
        # deviation: for a pair whose true hedge ratio is CONSTANT, a large
        # delta lets the state absorb the spread day by day and destroys the
        # signal. The formation-window MLE must select a smaller delta there.
        # (When beta genuinely drifts, preferring large delta is correct --
        # that case is covered by the tracking test above.)
        n = 300
        log_x = pd.Series(np.cumsum(RNG.normal(0, 0.01, n)) + 4.0)
        log_y = pd.Series(1.3 * log_x.to_numpy() + RNG.normal(0, 0.004, n))
        theta0, P0, rv = kf.initial_state(log_y.iloc[:250], log_x.iloc[:250])
        delta, _, _ = kf.estimate_noise(log_y.iloc[:250], log_x.iloc[:250],
                                        theta0, P0, rv)
        assert delta < 1e-4
