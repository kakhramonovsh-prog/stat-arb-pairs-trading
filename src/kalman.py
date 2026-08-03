"""
Dynamic hedge ratio via a Kalman filter.

Specification is fixed in PREREGISTRATION.md and must not be altered after
seeing results.

The model treats the cointegrating coefficients as a latent state that follows
a random walk, rather than as constants to be estimated once per formation
window:

    observation   log(y_t) = [log(x_t), 1] . theta_t + v_t     v ~ N(0, R)
    state         theta_t  = theta_{t-1}          + w_t        w ~ N(0, Q)

with theta_t = [beta_t, alpha_t]'.

The reason this is worth doing, beyond chasing returns: the filter's innovation
is the spread, and the innovation's predicted variance is its scale. Both are
computed from the state predicted using information through t-1, so the
resulting z-score cannot see the future. The look-ahead error that the primary
study quantifies at +6.8pp of annualised return is not merely avoided here, it
is unrepresentable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Chan (2013), Algorithmic Trading, ch. 3. Published default, not tuned.
DEFAULT_DELTA = 1e-4


def filter_pair(
    log_y: pd.Series,
    log_x: pd.Series,
    theta0: np.ndarray,
    P0: np.ndarray,
    R: float,
    delta: float = DEFAULT_DELTA,
) -> pd.DataFrame:
    """
    Run the filter over aligned log-price series.

    Returns a frame indexed like the inputs with columns:
        beta, alpha   the filtered state after updating on day t
        spread        the innovation e_t (the prediction error)
        z             e_t / sqrt(S_t), the standardised innovation

    Parameters mirror the pre-registered specification: `theta0` and `P0` come
    from the formation-period OLS fit, and `R` from the formation-period
    residual variance.
    """
    y = log_y.to_numpy(dtype=float)
    x = log_x.to_numpy(dtype=float)
    n = len(y)

    # State noise. The delta parameterisation makes Q scale-free relative to
    # the observation noise, which is why delta rather than Q is the quantity
    # quoted in the literature.
    Q = (delta / (1.0 - delta)) * np.eye(2)

    theta = np.asarray(theta0, dtype=float).reshape(2)
    P = np.asarray(P0, dtype=float).reshape(2, 2)

    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    zs = np.full(n, np.nan)

    for t in range(n):
        if not (np.isfinite(y[t]) and np.isfinite(x[t])):
            betas[t], alphas[t] = theta[0], theta[1]
            continue

        H = np.array([x[t], 1.0])

        # --- Predict -------------------------------------------------------
        # The random walk means the state prediction is just the previous
        # state; only the uncertainty grows.
        theta_pred = theta
        P_pred = P + Q

        # --- Innovation ----------------------------------------------------
        # e is the spread: how far today's actual log price is from what the
        # relationship estimated through YESTERDAY predicts. S is its variance.
        y_hat = H @ theta_pred
        e = y[t] - y_hat
        S = float(H @ P_pred @ H.T + R)

        if S <= 0 or not np.isfinite(S):
            betas[t], alphas[t] = theta_pred[0], theta_pred[1]
            continue

        spreads[t] = e
        zs[t] = e / np.sqrt(S)

        # --- Update --------------------------------------------------------
        K = (P_pred @ H.T) / S
        theta = theta_pred + K * e
        P = P_pred - np.outer(K, H) @ P_pred

        betas[t], alphas[t] = theta[0], theta[1]

    return pd.DataFrame(
        {"beta": betas, "alpha": alphas, "spread": spreads, "z": zs},
        index=log_y.index,
    )


def log_likelihood(
    log_y: pd.Series,
    log_x: pd.Series,
    theta0: np.ndarray,
    P0: np.ndarray,
    R: float,
    delta: float,
) -> float:
    """
    Gaussian log-likelihood of the observations under the filter.

    The prediction-error decomposition: each observation contributes
    -0.5 * (log(2*pi*S_t) + e_t^2 / S_t), where e_t and S_t are the innovation
    and its variance. Maximising this is the standard way to estimate
    state-space noise parameters, and it uses only the data the filter has
    already seen.
    """
    out = filter_pair(log_y, log_x, theta0, P0, R, delta=delta)
    e = out["spread"].to_numpy()
    # Recover S from e and z, which the filter already computed.
    z = out["z"].to_numpy()
    ok = np.isfinite(e) & np.isfinite(z) & (np.abs(z) > 0)
    if ok.sum() < 20:
        return -np.inf
    S = (e[ok] / z[ok]) ** 2
    if not np.all(np.isfinite(S)) or np.any(S <= 0):
        return -np.inf
    return float(-0.5 * np.sum(np.log(2 * np.pi * S) + e[ok] ** 2 / S))


DELTA_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
RHO_GRID = (0.001, 0.01, 0.05, 0.2, 1.0)


def estimate_noise(
    log_y: pd.Series,
    log_x: pd.Series,
    theta0: np.ndarray,
    P0: np.ndarray,
    resid_var: float,
) -> tuple[float, float, float]:
    """
    Choose (delta, R) by maximum likelihood over a fixed grid, using FORMATION
    data only.

    Returns (delta, R, loglik).

    Why this replaced the pre-registered fixed `delta = 1e-4`: that value is
    Chan's published default for a filter run on price levels. Applied to log
    prices it makes the state noise large relative to the observation noise, so
    the filter re-fits the hedge ratio to each day's price and absorbs the very
    deviation the strategy is supposed to trade. The symptom is a
    trading-window z-score with a standard deviation near 0.15, which never
    reaches the entry threshold and produces no trades at all.

    That is a scaling error, not a result, and it was found by inspecting the
    z-score distribution before any profit-and-loss was computed. The
    replacement estimates both noise terms from the formation window by
    maximum likelihood. It is a standard estimator, it sees no trading-window
    data, and it has no access to returns -- so it cannot be tuning in the
    sense the study criticises. The deviation from PREREGISTRATION.md is
    recorded there and in README.md rather than quietly absorbed.

    `R` is searched as a fraction `rho` of the formation residual variance,
    which keeps the grid scale-free across pairs with very different spread
    volatilities.
    """
    best = (DEFAULT_DELTA, resid_var, -np.inf)
    for delta in DELTA_GRID:
        for rho in RHO_GRID:
            R = max(resid_var * rho, 1e-12)
            ll = log_likelihood(log_y, log_x, theta0, P0, R, delta)
            if ll > best[2]:
                best = (delta, R, ll)
    return best


def initial_state(
    log_y: pd.Series, log_x: pd.Series
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Seed the filter from a formation-window OLS fit.

    Returns (theta0, P0, R) where theta0 = [beta, alpha], P0 is the OLS
    coefficient covariance, and R is the residual variance. All three are
    computed on formation data only.
    """
    import statsmodels.api as sm

    frame = pd.concat([log_y, log_x], axis=1).dropna()
    if len(frame) < 30:
        return np.array([1.0, 0.0]), np.eye(2) * 1e-3, 1e-4

    yv = frame.iloc[:, 0].to_numpy()
    xv = frame.iloc[:, 1].to_numpy()

    design = np.column_stack([xv, np.ones(len(xv))])
    fit = sm.OLS(yv, design).fit()

    theta0 = np.asarray(fit.params, dtype=float)
    P0 = np.asarray(fit.cov_params(), dtype=float)
    R = float(np.var(fit.resid, ddof=2))

    if not np.isfinite(R) or R <= 0:
        R = 1e-6

    return theta0, P0, R
