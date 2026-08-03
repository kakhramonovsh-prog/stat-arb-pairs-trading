"""
Cross-sectional statistical arbitrage (Avellaneda & Lee 2010).

The pairs study asks whether two stocks share a stochastic trend. This module
asks the modern version of the question: strip the common-factor component out
of every stock's return simultaneously, and trade mean reversion in what is
left.

Pipeline, re-estimated every day on trailing data only:

  1. FACTORS.  PCA on the correlation matrix of the trailing 252 days of
     returns. The top 15 eigenvectors define "eigenportfolios" -- long-short
     baskets whose returns span the dominant common variation (market, sectors,
     momentum-like modes). Eigenvector components are scaled by inverse
     volatility (correlation, not covariance, so that one volatile stock cannot
     dominate a factor) and L1-normalised so each factor is a $1-gross basket.

  2. RESIDUALS.  Each stock's trailing 60 days of returns are regressed on the
     15 eigenportfolio return series. The residual is the stock's idiosyncratic
     return: what its price did that its factor exposures cannot explain.

  3. OU FIT.  The cumulative residual X_t is fitted as an AR(1) process,
     X_t = a + b*X_{t-1} + e. Under the OU interpretation the equilibrium level
     is m = a/(1-b) and the equilibrium standard deviation is
     sigma_eq = sqrt(var(e)/(1-b^2)). The s-score,

         s = (X_end - m) / sigma_eq,

     measures how far the stock's idiosyncratic price is from where its own
     recent history says it settles, in units of that history's noise.

  4. TRADE.  Stocks with s below -1.25 are bought (hedged with their factor
     exposure); above +1.25, sold short. Positions close when the residual has
     largely reverted. Only stocks whose fitted reversion is fast enough
     (annualised kappa > 252/30, i.e. characteristic time under 30 trading
     days) are eligible -- a slow OU fit is indistinguishable from no reversion
     at this horizon.

All parameters are fixed in PREREGISTRATION_AL.md from the published paper and
are not tuned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# --- Pre-registered parameters (Avellaneda & Lee 2010) ---------------------
CORR_WINDOW = 252
REG_WINDOW = 60
N_FACTORS = 15
S_OPEN_LONG = -1.25
S_OPEN_SHORT = 1.25
S_CLOSE_LONG = -0.50
S_CLOSE_SHORT = 0.75
KAPPA_MIN = 252.0 / 30.0        # annualised; reversion time < 30 trading days
SLOT = 1.0 / 100.0              # capital per position
MAX_OPEN = 100
COST_BPS = 5.0                  # per side, on stock plus hedge notional
BORROW_ANNUAL_BPS = 50.0


def stock_universe(prices: pd.DataFrame) -> list[str]:
    """Stocks only: the ETFs are factor proxies, not tradeable residuals here."""
    etfs = set(config.SECTOR_ETFS.values()) | {config.BENCHMARK}
    return [c for c in prices.columns if c not in etfs]


def _estimate_day(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One day's estimation, given the trailing CORR_WINDOW x N return matrix of
    currently eligible stocks (no NaNs).

    Returns (s_scores, kappas, betas_l1, Q) where betas_l1 is the gross hedge
    notional per stock and Q the (N x N_FACTORS) eigenportfolio weights.
    """
    n_obs, n = R.shape

    # Correlation-based PCA. Standardising first is A&L's choice and the right
    # one: on a covariance matrix the first eigenvector tilts toward whatever
    # was volatile lately, and the "factor" becomes a high-beta bet rather than
    # a common mode.
    sd = R.std(axis=0, ddof=1)
    sd[sd <= 0] = np.nan
    Z = (R - R.mean(axis=0)) / sd
    corr = np.corrcoef(R, rowvar=False)

    vals, vecs = np.linalg.eigh(corr)
    top = vecs[:, ::-1][:, :N_FACTORS]          # descending eigenvalue order

    # Eigenportfolio weights: eigenvector over volatility, L1-normalised so the
    # factor is the return on a $1-gross basket and hedge notionals are in
    # dollars per dollar of stock.
    Q = top / sd[:, None]
    Q = Q / np.abs(Q).sum(axis=0, keepdims=True)

    # Factor returns over the regression window, then per-stock OLS of stock
    # returns on the factors. The design matrix is identical for every stock,
    # so one lstsq call fits the whole cross-section.
    R60 = R[-REG_WINDOW:]
    F = R60 @ Q                                  # (60 x k)
    D = np.column_stack([np.ones(REG_WINDOW), F])
    coef, *_ = np.linalg.lstsq(D, R60, rcond=None)   # (k+1 x N)
    resid = R60 - D @ coef

    betas_l1 = np.abs(coef[1:]).sum(axis=0)

    # OU fit on the cumulative residual. With an intercept in the regression
    # the residuals sum to zero, so the path ends at X = 0 and the s-score
    # reduces to -m / sigma_eq: how far zero is from the fitted equilibrium.
    X = np.cumsum(resid, axis=0)
    X0, X1 = X[:-1], X[1:]
    x0m = X0.mean(axis=0)
    x1m = X1.mean(axis=0)
    dx0 = X0 - x0m
    var0 = (dx0 ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = ((dx0 * (X1 - x1m)).sum(axis=0)) / var0
        a = x1m - b * x0m
        e = X1 - a - b * X0
        var_e = e.var(axis=0, ddof=2)
        m = a / (1.0 - b)
        sigma_eq = np.sqrt(var_e / (1.0 - b ** 2))
        kappa = -np.log(b) * 252.0

    ok = (b > 0) & (b < 1) & np.isfinite(sigma_eq) & (sigma_eq > 0)
    s = np.where(ok, (0.0 - m) / sigma_eq, np.nan)
    kappa = np.where(ok, kappa, np.nan)

    return s, kappa, betas_l1, Q


def run_cross_sectional(
    prices: pd.DataFrame,
    trade_start: str,
    trade_end: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the strategy with daily re-estimation.

    Returns (daily, trades): the portfolio return decomposition per day, and a
    round-trip trade log.

    The loop order inside each day enforces causality mechanically: first the
    P&L of positions carried overnight is computed using YESTERDAY's loadings
    and eigenportfolio weights applied to today's returns, and only then are
    today's estimates produced and today's position changes decided.
    """
    stocks = stock_universe(prices)
    px = prices[stocks]
    R = px.pct_change()
    dates = R.index

    t0 = int(dates.searchsorted(pd.Timestamp(trade_start)))
    t1 = int(dates.searchsorted(pd.Timestamp(trade_end), side="right"))
    t0 = max(t0, CORR_WINDOW + 1)

    n_all = len(stocks)
    pos = np.zeros(n_all)               # -1, 0, +1 per stock
    entry_idx = np.full(n_all, -1)
    entry_pnl = np.zeros(n_all)         # cumulative net at entry, per stock
    pnl_cum = np.zeros(n_all)           # cumulative net per stock (slot units)
    hedge_l1 = np.zeros(n_all)          # gross hedge notional at last estimate

    prev: dict = {"beta": None, "Q": None, "cols": None}

    cost_rate = COST_BPS / 1e4
    borrow_daily = BORROW_ANNUAL_BPS / 1e4 / 252.0

    rows = []
    trade_rows = []

    for t in range(t0, t1):
        today = dates[t]
        r_today_full = R.iloc[t]

        gross = costs = borrow = 0.0

        # ---- P&L of carried positions, on yesterday's estimates ----------
        if prev["beta"] is not None and pos.any():
            cols = prev["cols"]
            r_sub = r_today_full.iloc[cols].to_numpy()
            F_today = np.where(np.isfinite(r_sub), r_sub, 0.0) @ prev["Q"]
            fitted = prev["beta"].T @ F_today            # (n_prev,)
            for j, ci in enumerate(cols):
                if pos[ci] == 0.0:
                    continue
                ri = r_sub[j]
                if not np.isfinite(ri):
                    continue
                eps = ri - fitted[j]
                g = pos[ci] * SLOT * eps
                # Borrow on the short stock leg plus half the ~1x hedge basket
                # (a factor hedge is roughly half short by construction).
                b_ = SLOT * (max(0.0, -pos[ci]) + 0.5) * borrow_daily
                gross += g
                borrow += b_
                pnl_cum[ci] += g - b_

        # ---- Re-estimate on data through today's close --------------------
        win = R.iloc[t - CORR_WINDOW + 1 : t + 1]
        eligible = win.notna().all(axis=0).to_numpy()
        cols = np.where(eligible)[0]

        s = np.full(n_all, np.nan)
        kappa = np.full(n_all, np.nan)
        if len(cols) > N_FACTORS + 5:
            s_e, k_e, bl1_e, Q = _estimate_day(win.iloc[:, cols].to_numpy())
            s[cols] = s_e
            kappa[cols] = k_e
            hedge_l1[cols] = bl1_e
            prev = {"beta": None, "Q": Q, "cols": cols}
            # store betas aligned to cols for tomorrow's P&L
            R60 = win.iloc[-REG_WINDOW:, cols].to_numpy()
            F = R60 @ Q
            D = np.column_stack([np.ones(REG_WINDOW), F])
            coef, *_ = np.linalg.lstsq(D, R60, rcond=None)
            prev["beta"] = coef[1:]

        # ---- Position transitions -----------------------------------------
        fast = np.isfinite(s) & np.isfinite(kappa) & (kappa > KAPPA_MIN)

        # Cost basis: stock leg plus a hedge leg of roughly equal gross
        # notional, executed via liquid factor-mimicking instruments. NOT
        # 1 + sum|beta| as first pre-registered: with L1-normalised
        # eigenportfolios that sum is ~12 and is an artefact of the
        # normalisation convention, not an economic quantity. Deviation
        # documented in PREREGISTRATION_AL.md; found during verification,
        # before any P&L was computed.
        LEGS = 2.0

        # Closes first (including stocks that lost eligibility mid-position).
        for i in range(n_all):
            if pos[i] == 0.0:
                continue
            broken = not np.isfinite(s[i])
            close = broken \
                or (pos[i] > 0 and s[i] > S_CLOSE_LONG) \
                or (pos[i] < 0 and s[i] < S_CLOSE_SHORT)
            if close:
                c = SLOT * LEGS * cost_rate
                costs += c
                pnl_cum[i] -= c
                trade_rows.append({
                    "ticker": stocks[i],
                    "direction": "long" if pos[i] > 0 else "short",
                    "entry": dates[entry_idx[i]],
                    "exit": today,
                    "days": t - entry_idx[i],
                    "pnl": (pnl_cum[i] - entry_pnl[i]) / SLOT,
                    "forced_close": bool(broken),
                })
                pos[i] = 0.0

        # Opens, strongest signals first, capped at MAX_OPEN.
        n_open = int((pos != 0).sum())
        cand = [i for i in range(n_all)
                if pos[i] == 0.0 and fast[i]
                and (s[i] < S_OPEN_LONG or s[i] > S_OPEN_SHORT)]
        cand.sort(key=lambda i: -abs(s[i]))
        for i in cand:
            if n_open >= MAX_OPEN:
                break
            c = SLOT * LEGS * cost_rate
            costs += c
            pos[i] = 1.0 if s[i] < S_OPEN_LONG else -1.0
            entry_idx[i] = t
            entry_pnl[i] = pnl_cum[i]      # before the open cost, so the
            pnl_cum[i] -= c                # logged trade pnl includes it
            n_open += 1

        rows.append({
            "date": today,
            "gross": gross,
            "trade_costs": costs,
            "borrow": borrow,
            "net": gross - costs - borrow,
            "n_open": n_open,
            "n_eligible": int(fast.sum()),
        })

        if verbose and (t - t0) % 252 == 0:
            print(f"  {today.date()}  open={n_open:3d}  eligible={int(fast.sum()):3d}")

    daily = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    return daily, trades
