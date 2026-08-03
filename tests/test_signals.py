"""
Tests for signal generation: the position state machine and sizing.

These rules ARE the strategy, so each exit path gets its own test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import signals


def _z(values) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestPositionStateMachine:
    def test_entry_short_on_high_z(self):
        # z above +entry means the y leg is rich: SHORT the spread.
        pos = signals.generate_positions(_z([0, 2.5, 2.5]), entry_z=2.0)
        assert pos.iloc[1] == -1.0

    def test_entry_long_on_low_z(self):
        pos = signals.generate_positions(_z([0, -2.5, -2.5]), entry_z=2.0)
        assert pos.iloc[1] == 1.0

    def test_exit_on_convergence(self):
        pos = signals.generate_positions(
            _z([0, 2.5, 1.0, 0.3]), entry_z=2.0, exit_z=0.5)
        assert pos.iloc[2] == -1.0     # still outside exit band
        assert pos.iloc[3] == 0.0      # converged

    def test_stop_loss_on_divergence(self):
        pos = signals.generate_positions(
            _z([0, 2.5, 3.0, 4.0, 1.0]), entry_z=2.0, stop_z=3.5)
        assert pos.iloc[3] == 0.0      # stopped, not held to convergence

    def test_time_stop(self):
        z = _z([0] + [2.5] * 10)
        pos = signals.generate_positions(z, entry_z=2.0, exit_z=0.5,
                                         stop_z=99, max_hold=5)
        assert pos.iloc[5] == -1.0
        assert pos.iloc[6] == 0.0      # closed by the clock

    def test_no_reentry_same_bar_after_exit(self):
        # After a stop the state machine is flat; a still-extreme z may re-open
        # only on a LATER bar, never the same one.
        z = _z([0, 2.5, 4.0, 3.0])
        pos = signals.generate_positions(z, entry_z=2.0, stop_z=3.5)
        assert pos.iloc[2] == 0.0

    def test_nan_holds_position(self):
        z = _z([0, 2.5, np.nan, 2.5])
        pos = signals.generate_positions(z, entry_z=2.0)
        assert pos.iloc[2] == -1.0     # no observable price: hold, not flatten


class TestSizing:
    def test_leg_weights_unit_gross(self):
        for beta in (0.3, 1.0, 2.7):
            wy, wx = signals.leg_weights(beta)
            assert abs(wy) + abs(wx) == pytest.approx(1.0)

    def test_leg_weights_hedge_direction(self):
        wy, wx = signals.leg_weights(1.5)
        assert wy > 0 and wx < 0       # long spread = long y, short x

    def test_zscore_degenerate_sigma_is_nan(self):
        s = pd.Series([1.0, 2.0, 3.0])
        z = signals.zscore(s, formation_mean=0.0, formation_std=0.0)
        assert z.isna().all()
