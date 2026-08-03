"""
Signal generation.

Converts a formation-period pair specification into a daily position series
over the trading window.

The one rule that governs this module: every number used to decide a trade on
day t must be observable at the close of day t, and the resulting position
earns the return from t to t+1. The shift that enforces this lives in
backtest.py, and it is the difference between a plausible backtest and a
fantasy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import cointegration as ct


def zscore(
    spread: pd.Series, formation_mean: float, formation_std: float
) -> pd.Series:
    """
    Standardise the trading-window spread using FORMATION-period moments.

    This is the subtle one. The obvious implementation standardises the spread
    by its own mean and standard deviation over the trading window, and it is
    wrong: those moments are computed from prices that have not happened yet at
    the moment of the trade. A spread that ends the window far from where it
    started will have its early deviations understated, and the backtest will
    show the strategy entering at exactly the right moments. The effect is
    large and it is the most common bug in published pairs-trading code.

    Using frozen formation moments is less flattering and is the only honest
    choice. It also means the z-score can legitimately drift far beyond the
    entry threshold if the relationship is breaking down, which is precisely
    the risk the stop-loss exists to handle.
    """
    if not np.isfinite(formation_std) or formation_std <= 0:
        return pd.Series(np.nan, index=spread.index)
    return (spread - formation_mean) / formation_std


def generate_positions(
    z: pd.Series,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_Z,
    max_hold: int = config.MAX_HOLDING_DAYS,
) -> pd.Series:
    """
    Map a z-score path to a position in the spread: +1 long, -1 short, 0 flat.

    Sign convention: the spread is log(y) - beta*log(x) - alpha. A high z means
    y is rich relative to x, so the trade is to SHORT the spread (short y, long
    x) and wait for convergence. Hence position = -1 when z > entry.

    Exits, in priority order:
      1. Convergence  -- |z| falls back inside exit_z. The trade worked.
      2. Stop loss    -- |z| exceeds stop_z. The spread is diverging rather
                         than reverting, which is the signature of a broken
                         relationship (a merger, a restructuring, a regime
                         change in one leg's business). Cointegration tests
                         cannot see these coming, so a hard stop is the only
                         defence and it must be wider than the entry threshold
                         but tight enough to cap the loss.
      3. Time stop    -- held longer than max_hold days. A spread that has not
                         reverted in roughly one half-life has probably stopped
                         being the process that was fitted.

    The loop is explicit rather than vectorised because position state is
    path-dependent: whether we are in a trade today depends on every prior day.
    """
    z = z.copy()
    values = z.to_numpy()
    positions = np.zeros(len(values), dtype=float)

    state = 0.0
    days_held = 0

    for t, zt in enumerate(values):
        if not np.isfinite(zt):
            # No observable price today: hold the existing position rather than
            # silently flattening, but let the holding clock keep running.
            if state != 0.0:
                days_held += 1
            positions[t] = state
            continue

        if state == 0.0:
            if zt > entry_z:
                state, days_held = -1.0, 0
            elif zt < -entry_z:
                state, days_held = 1.0, 0
        else:
            days_held += 1
            if abs(zt) < exit_z:
                state = 0.0
            elif abs(zt) > stop_z:
                state = 0.0
            elif days_held >= max_hold:
                state = 0.0

        positions[t] = state

    return pd.Series(positions, index=z.index, name="position")


def leg_weights(beta: float) -> tuple[float, float]:
    """
    Portfolio weights on the y and x legs for one unit of gross exposure.

    The cointegrating relationship says one unit of y is matched by beta units
    of x, so the hedge is beta-weighted, not 50/50. Weights are normalised so
    that |w_y| + |w_x| = 1, which makes every pair contribute the same gross
    exposure regardless of its hedge ratio and keeps a pair with beta = 3 from
    silently taking three times the risk of a pair with beta = 1.

    Note this is beta-neutral, not dollar-neutral: the two coincide only when
    beta = 1. Dollar-neutrality is the wrong constraint here, because it would
    leave a residual exposure to the very common factor the hedge ratio exists
    to strip out.
    """
    denom = 1.0 + abs(beta)
    return 1.0 / denom, -beta / denom


def pair_signal_frame(
    pair: ct.PairResult,
    log_px: pd.DataFrame,
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
    use_lookahead_moments: bool = False,
    rolling_window: int | None = None,
    kalman: bool = False,
    formation_start: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """
    Build the full per-day signal table for one pair over its trading window.

    Returns a frame indexed by date with columns: spread, z, position.

    Three normalisation modes:

      default            frozen formation-period mean and sigma. The strictest
                         choice and the primary specification.

      rolling_window=W   trailing mean and sigma over the W days ENDING AT
                         t-1. This is fully causal -- it uses no information
                         from day t or later -- and is closer to how the trade
                         is run in practice, because a cointegrating
                         relationship's level can legitimately drift over six
                         months and a frozen mean will read that drift as a
                         permanent, ever-widening dislocation. Reported as a
                         robustness specification.

      use_lookahead_moments
                         deliberately reproduces the standard look-ahead bug by
                         standardising with the trading window's OWN moments.
                         Used only to quantify the size of that bias as a
                         control, never for reported results.
    """
    if pair.y not in log_px.columns or pair.x not in log_px.columns:
        return None

    mask = (log_px.index >= trading_start) & (log_px.index <= trading_end)
    if not mask.any():
        return None

    if kalman:
        # Seed on the formation window, then run the filter forward through the
        # trading window. The filter is warmed on formation data so the state
        # has settled before the first trade, and every trading-window z-score
        # uses a state predicted from strictly earlier observations.
        from src import kalman as kf

        if formation_start is None:
            return None
        span = log_px.loc[
            (log_px.index >= formation_start) & (log_px.index <= trading_end),
            [pair.y, pair.x],
        ].dropna()
        form = span.loc[span.index < trading_start]
        if len(form) < 30 or len(span.loc[span.index >= trading_start]) < 5:
            return None

        theta0, P0, resid_var = kf.initial_state(form[pair.y], form[pair.x])
        delta, R, _ = kf.estimate_noise(
            form[pair.y], form[pair.x], theta0, P0, resid_var
        )
        filtered = kf.filter_pair(
            span[pair.y], span[pair.x], theta0, P0, R, delta=delta
        )

        out = filtered.loc[filtered.index >= trading_start]
        spread, z = out["spread"], out["z"]

    elif rolling_window:
        # Pull `rolling_window` extra bars of history so the first trading day
        # already has a full trailing estimate. That history predates the
        # trading window, so using it is not look-ahead.
        first = int(np.argmax(mask))
        lo = max(0, first - rolling_window)
        extended = log_px.iloc[lo : int(np.where(mask)[0][-1]) + 1][[pair.y, pair.x]]
        full_spread = ct.spread_series(
            extended[pair.y], extended[pair.x], pair.beta, pair.alpha
        )
        # shift(1) is what makes it causal: the estimate available on day t is
        # built from data through t-1 only.
        mu = full_spread.rolling(rolling_window, min_periods=20).mean().shift(1)
        sd = full_spread.rolling(rolling_window, min_periods=20).std(ddof=1).shift(1)
        z_full = (full_spread - mu) / sd.where(sd > 0)
        z = z_full.loc[z_full.index >= trading_start]
        spread = full_spread.loc[full_spread.index >= trading_start]
    else:
        window = log_px.loc[mask, [pair.y, pair.x]]
        spread = ct.spread_series(window[pair.y], window[pair.x],
                                  pair.beta, pair.alpha)
        if use_lookahead_moments:
            z = zscore(spread, float(spread.mean()), float(spread.std(ddof=1)))
        else:
            z = zscore(spread, pair.spread_mean, pair.spread_std)

    positions = generate_positions(z)
    return pd.DataFrame({"spread": spread, "z": z, "position": positions})
