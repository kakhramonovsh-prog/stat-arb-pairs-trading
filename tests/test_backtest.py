"""
Tests for the backtest engine.

The one that matters most is causality: a position decided at the close of day
t must earn the return from t to t+1 and nothing else. Every fraudulent
backtest fails exactly this test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import backtest
from src.cointegration import PairResult


def _flat_pair(spread_path: list[float], beta: float = 1.0):
    """
    Build a two-asset price panel where asset X is constant and asset Y moves
    so that the log spread follows `spread_path` exactly. With X flat, all
    spread movement comes from Y, and the pair P&L on each day is exactly
    w_y * (Y's return) -- fully predictable by hand.
    """
    dates = pd.bdate_range("2020-01-01", periods=len(spread_path))
    log_x = np.zeros(len(spread_path))
    log_y = beta * log_x + np.array(spread_path)
    prices = pd.DataFrame({"YYY": np.exp(log_y), "XXX": np.exp(log_x)},
                          index=dates)
    log_px = np.log(prices)
    pair = PairResult(sector="Test", y="YYY", x="XXX", beta=beta, alpha=0.0,
                      eg_stat=-4.0, eg_pvalue=0.001, half_life=5.0,
                      spread_mean=0.0, spread_std=0.01, n_obs=250)
    return prices, log_px, pair, dates


class TestCausality:
    def test_entry_day_earns_nothing(self):
        # Spread jumps to +3 sigma on day 1 (entry signal at that close) and
        # reverts on day 2. The position opened at day 1's close may earn day
        # 1 -> 2 only. If the engine credited day 0 -> 1 (the entry-day move),
        # it would be trading on the signal before the signal existed.
        spread = [0.0, 0.03, 0.0, 0.0, 0.0]
        prices, log_px, pair, dates = _flat_pair(spread)
        frame = backtest.simulate_pair(pair, log_px, prices,
                                       dates[0], dates[-1])
        assert frame["gross"].iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_reversion_day_is_earned(self):
        # Short entered at +3 sigma; the day-2 reversion (-0.03 in log spread,
        # asset Y falling) is profit for the short-spread position.
        spread = [0.0, 0.03, 0.0, 0.0, 0.0]
        prices, log_px, pair, dates = _flat_pair(spread)
        frame = backtest.simulate_pair(pair, log_px, prices,
                                       dates[0], dates[-1])
        assert frame["gross"].iloc[2] > 0

    def test_no_pnl_while_flat(self):
        spread = [0.0, 0.001, -0.001, 0.0, 0.001]   # never near entry
        prices, log_px, pair, dates = _flat_pair(spread)
        frame = backtest.simulate_pair(pair, log_px, prices,
                                       dates[0], dates[-1])
        assert frame["gross"].abs().max() == pytest.approx(0.0, abs=1e-12)


class TestCosts:
    def test_costs_only_on_transitions(self):
        spread = [0.0, 0.03, 0.03, 0.03, 0.0, 0.0]
        prices, log_px, pair, dates = _flat_pair(spread)
        frame = backtest.simulate_pair(pair, log_px, prices,
                                       dates[0], dates[-1])
        charged = frame["trade_costs"] > 0
        # exactly two charge events: the open and the close
        assert charged.sum() == 2

    def test_zero_cost_multiplier(self):
        spread = [0.0, 0.03, 0.0, 0.03, 0.0, 0.0]
        prices, log_px, pair, dates = _flat_pair(spread)
        frame = backtest.simulate_pair(pair, log_px, prices,
                                       dates[0], dates[-1],
                                       cost_multiplier=0.0)
        assert frame["trade_costs"].abs().max() == 0.0
