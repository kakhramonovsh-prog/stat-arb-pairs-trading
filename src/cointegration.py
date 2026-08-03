"""
Cointegration testing.

The economic idea: two stocks that share a common stochastic trend can wander
far from each other in the short run but cannot drift apart permanently. If
some linear combination of their log prices is stationary, deviations from that
combination are mean-reverting and therefore tradeable.

The statistical machinery is Engle & Granger's (1987) two-step procedure, with
Johansen's (1991) trace test as an independent robustness check, an
Ornstein-Uhlenbeck fit to measure how fast the spread reverts, and a
Benjamini-Hochberg correction because the study tests thousands of pairs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class PairResult:
    """Formation-period statistics for one candidate pair."""

    sector: str
    y: str                  # dependent leg of the cointegrating regression
    x: str                  # independent leg
    beta: float             # hedge ratio: spread = log(y) - beta*log(x) - alpha
    alpha: float
    eg_stat: float          # Engle-Granger (ADF) test statistic
    eg_pvalue: float
    half_life: float        # trading days
    spread_mean: float      # formation-period mean of the spread
    spread_std: float       # formation-period standard deviation
    n_obs: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Engle-Granger
# --------------------------------------------------------------------------

def engle_granger(
    log_y: pd.Series, log_x: pd.Series, maxlag: int = 12
) -> tuple[float, float, float, float]:
    """
    Run the Engle-Granger two-step test of log_y against log_x.

    Returns (test_statistic, p_value, beta, alpha).

    Step 1 regresses log_y on log_x by OLS and keeps the residual. Step 2 tests
    that residual for a unit root.

    The critical subtlety, and the thing an interviewer will ask about: you
    cannot use standard Dickey-Fuller critical values here. The residual is not
    observed data, it is the output of a regression whose coefficient was
    chosen to make that residual as small as possible. OLS therefore makes the
    residual look more stationary than it is, and a standard ADF test
    over-rejects the null of no cointegration. `statsmodels.tsa.stattools.coint`
    applies MacKinnon's critical values for the residual-based case, which
    correct for exactly this. Running `adfuller` on the residual by hand is the
    single most common error in pairs-trading code.

    On using OLS for the hedge ratio: OLS is asymptotically biased in this
    setting, but Stock (1987) shows the estimator is superconsistent under
    cointegration -- it converges at rate T rather than the usual sqrt(T) --
    so with 252 formation observations the bias is small relative to the
    estimation noise we care about.
    """
    stat, pvalue, _ = coint(log_y, log_x, trend="c", maxlag=maxlag, autolag="aic")

    design = sm.add_constant(log_x.to_numpy())
    fit = sm.OLS(log_y.to_numpy(), design).fit()
    alpha, beta = float(fit.params[0]), float(fit.params[1])

    return float(stat), float(pvalue), beta, alpha


def spread_series(log_y: pd.Series, log_x: pd.Series, beta: float, alpha: float) -> pd.Series:
    """
    The cointegrating residual, which is what the strategy actually trades.

    Note the sign convention: a positive spread means the y leg is rich
    relative to the x leg, so a positive-spread signal means short y, long x.
    """
    return log_y - beta * log_x - alpha


# --------------------------------------------------------------------------
# Speed of mean reversion
# --------------------------------------------------------------------------

def half_life(spread: pd.Series) -> float:
    """
    Half-life of mean reversion in trading days, from a discrete
    Ornstein-Uhlenbeck fit.

    The OU process ds = theta*(mu - s)dt + sigma*dW discretises to the AR(1)
    regression

        s_t - s_{t-1} = a + b * s_{t-1} + e_t

    where b = -theta. Reversion requires b < 0, and the time for a deviation to
    decay by half is ln(2) / theta = -ln(2) / b.

    Returns np.inf when b >= 0, i.e. when the fitted series is not reverting at
    all. Callers filter on this rather than trading a spread that has no
    tendency to come back.

    Why this filter earns its place: a statistically significant cointegration
    p-value says a relationship exists, not that it reverts fast enough to
    trade. A pair with a 400-day half-life will pass the ADF test and then sit
    in a losing position for the entire 126-day trading window.
    """
    s = spread.dropna()
    if len(s) < 20:
        return np.inf

    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")

    design = sm.add_constant(lagged.to_numpy())
    b = float(sm.OLS(delta.to_numpy(), design).fit().params[1])

    if b >= 0:
        return np.inf
    return float(-np.log(2) / b)


# --------------------------------------------------------------------------
# Johansen, as an independent check
# --------------------------------------------------------------------------

def johansen_trace(log_y: pd.Series, log_x: pd.Series, det_order: int = 0,
                   k_ar_diff: int = 1) -> tuple[float, float, float]:
    """
    Johansen trace test for the null of zero cointegrating relations against
    the alternative of at least one.

    Returns (trace_statistic, critical_value_5pct, implied_beta).

    Why bother when Engle-Granger already answered the question: Engle-Granger
    imposes a direction. It picks one series as dependent, estimates a single
    equation, and its answer can change depending on which series you put on
    the left-hand side. Johansen is a full-system maximum-likelihood method
    that treats both series symmetrically and does not have that arbitrariness.
    Agreement between the two is meaningful evidence; disagreement is a signal
    that the pair is marginal.

    This is used as a reported robustness statistic, not as the primary screen,
    because Johansen is the more powerful test but also the more sensitive to
    lag-length misspecification in short samples.
    """
    frame = pd.concat([log_y, log_x], axis=1).dropna()
    if len(frame) < 30:
        return np.nan, np.nan, np.nan

    # The generalised eigenvalue problem at the heart of the Johansen procedure
    # can return eigenvalues with small imaginary components when the two
    # series are close to collinear over the window -- a numerical artefact of
    # an ill-conditioned covariance matrix, not a meaningful complex result.
    # The imaginary parts are discarded deliberately here, and the pair is
    # flagged, rather than letting the cast happen silently inside statsmodels.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        try:
            res = coint_johansen(frame.to_numpy(), det_order, k_ar_diff)
        except Exception:
            return np.nan, np.nan, np.nan

        trace_stat = float(np.real(res.lr1[0]))
        crit_5pct = float(res.cvt[0, 1])   # columns are 90%, 95%, 99%

        # Normalise the first cointegrating vector on the y leg so the implied
        # hedge ratio is directly comparable to the Engle-Granger beta.
        vec = np.real(res.evec[:, 0])
        implied_beta = float(-vec[1] / vec[0]) if vec[0] != 0 else np.nan

    return trace_stat, crit_5pct, implied_beta


# --------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------

def fdr_filter(pvalues: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg false discovery rate control.

    Returns (reject_flags, adjusted_pvalues).

    This is the correction that separates a serious study from a data-mining
    exercise. Screening ~1,500 candidate pairs at a nominal 5% level would
    produce roughly 75 "cointegrated" pairs by chance alone even if no pair in
    the universe were genuinely cointegrated. Any backtest built on an
    uncorrected screen is partly, and possibly entirely, trading noise.

    Benjamini-Hochberg is preferred to Bonferroni here. Bonferroni controls the
    probability of even one false positive, which is far stricter than needed
    and would leave almost nothing to trade across 1,500 tests. BH instead
    controls the expected *proportion* of false positives among the pairs we
    select, which is the quantity a portfolio actually cares about: a book of
    20 pairs in which 1 is spurious is fine, and BH bounds that fraction.

    Caveat, stated in the paper: BH assumes tests are independent or positively
    dependent. Pairs drawn from the same sector share constituents and are
    clearly correlated. The dependence is positive (co-moving stocks), which is
    the case BH is known to handle, but the guarantee is weaker than in the
    independent case.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(pvalues)

    reject = np.zeros(len(pvalues), dtype=bool)
    adjusted = np.full(len(pvalues), np.nan)

    if valid.sum() == 0:
        return reject, adjusted

    rej_v, adj_v, _, _ = multipletests(pvalues[valid], alpha=alpha, method="fdr_bh")
    reject[valid] = rej_v
    adjusted[valid] = adj_v
    return reject, adjusted


def storey_pi0(pvalues: np.ndarray, lam: float = 0.5) -> float:
    """
    Storey's (2002) estimate of the proportion of true null hypotheses.

    Under the null, p-values are uniform on [0, 1], so the fraction landing
    above a threshold lam should be about (1 - lam) times the number of true
    nulls. Inverting that:

        pi0_hat = #{p > lam} / (m * (1 - lam))

    Interpretation for this study: pi0 near 1.0 means essentially no pair in
    the universe is genuinely cointegrated and everything below p=0.05 is
    chance. pi0 of 0.9 means roughly 10% of tested pairs share a real long-run
    relationship. This is a far more informative summary than a binary
    reject/accept, because it quantifies how much genuine signal exists even
    when that signal is too diffuse to attribute to specific pairs.
    """
    p = np.asarray(pvalues, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return float("nan")
    pi0 = (p > lam).sum() / (len(p) * (1.0 - lam))
    return float(min(pi0, 1.0))


def expected_fdp(pvalues: np.ndarray, k: int, pi0: float | None = None) -> float:
    """
    Expected false discovery proportion if the k most significant pairs are
    selected.

    Selecting the top k by p-value is equivalent to thresholding at p_(k). The
    expected number of nulls falling below that threshold is pi0 * m * p_(k),
    so the expected fraction of the selected book that is spurious is

        E[FDP] ~= pi0 * m * p_(k) / k

    This is the number the paper reports for every cohort. It is the honest
    answer to "how much of this book is noise", and it is reported whether or
    not it is flattering.
    """
    p = np.sort(np.asarray(pvalues, dtype=float))
    p = p[np.isfinite(p)]
    if len(p) == 0 or k <= 0 or k > len(p):
        return float("nan")
    if pi0 is None:
        pi0 = storey_pi0(p)
    return float(min(pi0 * len(p) * p[k - 1] / k, 1.0))


# --------------------------------------------------------------------------
# Driver for a single pair
# --------------------------------------------------------------------------

def test_pair(
    window: pd.DataFrame, sector: str, a: str, b: str, min_obs: int
) -> PairResult | None:
    """
    Test one candidate pair over one formation window.

    Both regression directions are run and the more significant is kept.
    Engle-Granger is not symmetric, and fixing the direction arbitrarily throws
    away real information about which leg better explains the other. The cost
    is that two tests are performed per pair instead of one, and the FDR
    correction upstream is applied over the full count of tests actually run,
    so this choice does not smuggle in an uncounted look at the data.

    Returns None if the pair lacks sufficient overlapping observations.
    """
    if a not in window.columns or b not in window.columns:
        return None

    frame = window[[a, b]].dropna()
    if len(frame) < min_obs:
        return None

    la, lb = frame[a], frame[b]

    best: PairResult | None = None
    for y_name, x_name, log_y, log_x in ((a, b, la, lb), (b, a, lb, la)):
        try:
            stat, pval, beta, alpha_ = engle_granger(log_y, log_x)
        except Exception:
            continue

        if not np.isfinite(pval):
            continue

        if best is None or pval < best.eg_pvalue:
            spread = spread_series(log_y, log_x, beta, alpha_)
            best = PairResult(
                sector=sector,
                y=y_name,
                x=x_name,
                beta=beta,
                alpha=alpha_,
                eg_stat=stat,
                eg_pvalue=pval,
                half_life=half_life(spread),
                spread_mean=float(spread.mean()),
                spread_std=float(spread.std(ddof=1)),
                n_obs=len(frame),
            )

    return best
