# Pre-registration 2: cross-sectional statistical arbitrage (Avellaneda-Lee)

**Written before implementation. Every parameter below is taken from the
published specification in Avellaneda & Lee (2010), "Statistical arbitrage in
the US equities market" (*Quantitative Finance* 10(7)), not selected by search.
One run per span. Reported regardless of outcome.**

---

## Motivation

The pairs study found no bilateral cointegration edge in modern US large caps.
The natural objection is that bilateral pairs are the 1980s formulation of the
trade. The method that superseded it is cross-sectional: decompose returns into
common factors estimated by PCA, and trade mean reversion in the **residuals**
— each stock against its own factor-implied return — across the whole universe
simultaneously. This is the Avellaneda-Lee formulation and the closest public
description of how the trade is actually run.

Testing it answers the right follow-up question: not "do bilateral pairs still
work" (known: no) but "does idiosyncratic mean reversion survive costs when the
common-factor component is properly stripped out".

## Hypothesis

**H2.** Mean reversion in PCA-residual returns produces positive gross alpha
with Newey-West t > 2.0 in both the in-sample (2011–2021) and out-of-sample
(2022–2026) spans.

## Specification (all fixed in advance, from the published paper)

| Quantity | Value | Source |
|---|---|---|
| Universe | The 173 stocks (ETFs excluded) | this study |
| Correlation window | 252 trading days | A&L §3 |
| Number of PCA factors | 15 | A&L use 15 eigenportfolios |
| Factor-regression / OU window | 60 trading days | A&L §5 |
| s-score entry (long) | s < −1.25 | A&L: s_bo = 1.25 |
| s-score entry (short) | s > +1.25 | A&L: s_so = 1.25 |
| s-score exit (close long) | s > −0.50 | A&L: s_sc = 0.50 |
| s-score exit (close short) | s < +0.75 | A&L: s_bc = 0.75 |
| Speed filter | annualised κ > 252/30 | A&L §5: reversion time < 30 days |
| Position size | 1/100 of capital per name | fixed slot; max 100 concurrent |
| Transaction costs | 5 bps per side on stock plus hedge notional | consistent with the pairs study |
| Borrow | 50 bps annualised on short notional | consistent with the pairs study |

Mechanics, fixed: eigenportfolio weights are eigenvector components scaled by
inverse volatility and L1-normalised. Stock returns are regressed (with
intercept) on the 15 eigenportfolio return series over the trailing 60 days;
the cumulative residual is fitted as an AR(1)/OU process; the s-score is the
distance of the current cumulative residual from its OU equilibrium in units of
equilibrium standard deviation. The basic s-score is used (A&L report their
main results with it; the drift-adjusted variant is not implemented).

**Causality.** All estimates on day t use data through the close of day t; the
position decided at t earns the return from t to t+1. The hedge P&L for day
t+1 uses factor loadings and eigenportfolio weights estimated at t. Nothing in
the signal path sees a future price.

**Cost approximation, disclosed.** The hedge side of each position is a basket
with gross notional Σ|β| per unit of stock notional. Costs are charged at
5 bps on (1 + Σ|β|) per opening and closing transition, and borrow at 50 bps
on short stock notional plus half the hedge gross (a factor basket is roughly
half short). This is an approximation and is stated as such.

## Decision rule (fixed in advance)

Identical in form to the first pre-registration: positive **gross** annualised
alpha AND Newey-West t > 2.0, in **both** spans. Anything less is reported as a
failure of H2.

## Constraints

- One run in-sample, one run out-of-sample. No parameter adjusted after
  seeing results.
- Implementation verification (z-score/s-score distributions, factor variance
  explained, eligibility counts) is permitted before the runs; profit-and-loss
  is not computed during verification.
- If H2 fails, it is reported as a failed hypothesis, not iterated on.

---

## Deviation from this pre-registration

One deviation, found during implementation verification (s-score and factor
diagnostics), **before any profit-and-loss was computed**.

The declared cost basis of 5 bps on (1 + Σ|β|) is mis-scaled. With
eigenportfolios L1-normalised to $1 gross, each factor is a ~170-name basket
with very low volatility, so the regression betas are mechanically large:
median Σ|β| ≈ 12. That number is an artefact of the normalisation convention —
rescale the eigenportfolios and it changes arbitrarily while the strategy's
returns do not. Charging transaction costs proportional to it would be charging
costs on a bookkeeping choice.

The economically meaningful quantity is the gross notional actually traded: the
stock leg plus a hedge of roughly equal size executed via liquid
factor-mimicking instruments. Costs are therefore charged at 5 bps per side on
2× the stock notional, and borrow on the short stock leg plus half the hedge —
the same two-leg structure the pairs study prices. Σ|β| is retained as a
reported diagnostic.

---

## RESULT — H2 REJECTED, WITH THE STUDY'S MOST INFORMATIVE FINDING

Two runs, as committed. No parameter adjusted after seeing these numbers.

| | In-sample 2011–2021 | Out-of-sample 2022–2026 | Criterion |
|---|---|---|---|
| Gross annualised | **+1.31%** | **+0.82%** | positive ✓ / ✓ |
| Newey-West t (gross) | **+2.33** | +0.67 | > 2.0 ✓ / **✗** |
| Net annualised | −2.16% | −2.64% | |
| Newey-West t (net) | −3.88 | −2.15 | |
| Win rate | 56.9% | 56.5% | |
| Round trips | 17,465 | 7,209 | |
| Avg holding | 9.5 days | 9.3 days | |
| Market beta | 0.011 | 0.011 | |

**Verdict: H2 is rejected** — the out-of-sample gross t-statistic of 0.67
fails the declared threshold.

**Why this is nonetheless the study's most informative result.** The in-sample
gross alpha is statistically significant (t = 2.33) — the only specification in
the study to clear the threshold in any span. Idiosyncratic mean reversion in
PCA residuals is a real, measurable phenomenon, with a strikingly stable win
rate (56.9% / 56.5%) across 24,000+ trades in both spans. What kills it is
economics, not statistics: the average trade earns ~5 bps of gross edge over
~9.5 days, against ~20 bps of round-trip cost on two legs. The gross effect
also weakened out-of-sample (+1.31% → +0.82%), consistent with continued
crowding.

The three specifications now form a coherent progression:

| Specification | Gross IS | Gross OOS | Verdict |
|---|---|---|---|
| Bilateral pairs, static β | +0.08% | −1.66% | no signal |
| Bilateral pairs, Kalman β | +0.89% | +0.40% | weak signal, insignificant |
| Cross-sectional PCA residuals | +1.31% (t=2.33) | +0.82% | real signal, uneconomic |

Each methodological refinement recovers more genuine signal, and none of it is
enough to pay costs. The correct conclusion is not "mean reversion is dead" but
"the mean reversion that remains in daily US large-cap prices is smaller than
the cost of trading it" — which is what a mature, crowded anomaly is expected
to look like.
