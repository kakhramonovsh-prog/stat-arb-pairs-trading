"""
Backtest engine.

Turns cohort books and daily positions into a return stream, net of costs.

Accounting conventions, stated once so they can be checked:

  * Positions are decided at the close of day t using only information
    available at that close, and earn the return from t to t+1. This is
    enforced by shifting the position series forward one day. Nothing else in
    the file is allowed to undo that shift.

  * Each pair is given an equal share of capital, sized as
    1 / MAX_CONCURRENT_PAIRS. Capital allocated to a pair that is currently
    flat sits in cash earning nothing. This understates returns relative to the
    common practice of scaling up so the book is always fully invested, and it
    is the honest treatment: the strategy genuinely cannot deploy that capital,
    because it has no signal.

  * Transaction costs are charged on gross traded notional whenever the
    position changes, and borrow is charged daily on the short leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import signals
from src.pairs import Cohort


def simulate_pair(
    pair,
    log_px: pd.DataFrame,
    prices: pd.DataFrame,
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
    use_lookahead_moments: bool = False,
    cost_multiplier: float = 1.0,
    rolling_window: int | None = None,
    kalman: bool = False,
    formation_start: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """
    Simulate one pair over one trading window.

    Returns a daily frame with the pair's gross return, costs, and net return,
    expressed per unit of capital allocated to that pair.

    The keyword arguments drive the control experiments in
    scripts/04_diagnostics.py and default to the primary specification.
    """
    sig = signals.pair_signal_frame(
        pair, log_px, trading_start, trading_end,
        use_lookahead_moments=use_lookahead_moments,
        rolling_window=rolling_window,
        kalman=kalman,
        formation_start=formation_start,
    )
    if sig is None or sig.empty:
        return None

    px = prices.loc[sig.index, [pair.y, pair.x]]
    rets = px.pct_change()

    w_y, w_x = signals.leg_weights(pair.beta)

    # The position decided at the close of t earns t -> t+1. Shifting the
    # position rather than the returns keeps the cost calculation aligned with
    # the day the trade actually happens.
    held = sig["position"].shift(1).fillna(0.0)

    gross = held * (w_y * rets[pair.y] + w_x * rets[pair.x])

    # Turnover is the change in position, scaled by gross exposure. Opening a
    # position from flat trades 1.0 of gross notional across the two legs;
    # closing trades it back.
    turnover = sig["position"].diff().abs().fillna(sig["position"].abs())
    cost_rate = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 1e4 * cost_multiplier
    trade_costs = turnover * cost_rate

    # Borrow on whichever leg is short. Exactly one leg is short whenever the
    # pair is on, and its weight is the relevant notional.
    short_notional = held.abs() * np.where(held > 0, abs(w_x), abs(w_y))
    borrow = short_notional * (config.BORROW_COST_ANNUAL_BPS / 1e4) / 252.0

    net = gross - trade_costs - borrow

    return pd.DataFrame({
        "gross": gross.fillna(0.0),
        "trade_costs": trade_costs.fillna(0.0),
        "borrow": pd.Series(borrow, index=sig.index).fillna(0.0),
        "net": net.fillna(0.0),
        "position": sig["position"],
        "z": sig["z"],
    })


def _trade_log(pair, frame: pd.DataFrame, cohort_label: str) -> list[dict]:
    """Extract discrete round-trip trades from a daily position series."""
    trades = []
    pos = frame["position"].to_numpy()
    net = frame["net"].to_numpy()
    idx = frame.index

    open_i = None
    for t in range(len(pos)):
        prev = pos[t - 1] if t > 0 else 0.0
        if prev == 0.0 and pos[t] != 0.0:
            open_i = t
        elif prev != 0.0 and pos[t] != prev:
            if open_i is not None:
                trades.append({
                    "cohort": cohort_label,
                    "pair": f"{pair.y}/{pair.x}",
                    "direction": "long" if prev > 0 else "short",
                    "entry": idx[open_i],
                    "exit": idx[t],
                    "days": t - open_i,
                    "pnl": float(net[open_i + 1 : t + 1].sum()),
                    "entry_z": float(frame["z"].iloc[open_i]),
                    "exit_z": float(frame["z"].iloc[t]),
                })
            open_i = t if pos[t] != 0.0 else None

    # A position still open when the window ends is closed at the last price.
    if open_i is not None and pos[-1] != 0.0:
        trades.append({
            "cohort": cohort_label,
            "pair": f"{pair.y}/{pair.x}",
            "direction": "long" if pos[-1] > 0 else "short",
            "entry": idx[open_i],
            "exit": idx[-1],
            "days": len(pos) - 1 - open_i,
            "pnl": float(net[open_i + 1 :].sum()),
            "entry_z": float(frame["z"].iloc[open_i]),
            "exit_z": float(frame["z"].iloc[-1]),
            "forced_close": True,
        })

    return trades


def run_backtest(
    cohorts: list[Cohort],
    prices: pd.DataFrame,
    log_px: pd.DataFrame,
    verbose: bool = True,
    use_lookahead_moments: bool = False,
    cost_multiplier: float = 1.0,
    rolling_window: int | None = None,
    kalman: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every cohort and aggregate into a single daily portfolio return series.

    Returns (daily, trades) where `daily` is indexed by date with gross, cost
    and net portfolio returns, and `trades` is the round-trip log.
    """
    slot = 1.0 / config.MAX_CONCURRENT_PAIRS
    daily_parts: list[pd.DataFrame] = []
    all_trades: list[dict] = []

    for cohort in cohorts:
        if not cohort.selected:
            continue

        contributions = []
        for pair in cohort.selected:
            frame = simulate_pair(
                pair, log_px, prices, cohort.trading_start, cohort.trading_end,
                use_lookahead_moments=use_lookahead_moments,
                cost_multiplier=cost_multiplier,
                rolling_window=rolling_window,
                kalman=kalman,
                formation_start=cohort.formation_start,
            )
            if frame is None:
                continue
            all_trades.extend(_trade_log(pair, frame, cohort.label))
            contributions.append(
                frame[["gross", "trade_costs", "borrow", "net"]] * slot
            )

        if not contributions:
            continue

        cohort_daily = sum(contributions)
        cohort_daily["n_pairs"] = len(contributions)
        daily_parts.append(cohort_daily)

        if verbose:
            total = cohort_daily["net"].sum()
            print(f"  {cohort.label}: {len(contributions):2d} pairs, "
                  f"net {total * 100:+6.2f}%")

    if not daily_parts:
        return pd.DataFrame(), pd.DataFrame()

    daily = pd.concat(daily_parts).sort_index()
    daily = daily[~daily.index.duplicated(keep="first")]

    trades = pd.DataFrame(all_trades)
    if "forced_close" in trades.columns:
        trades["forced_close"] = trades["forced_close"].fillna(False)

    return daily, trades
