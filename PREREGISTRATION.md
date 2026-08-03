# Pre-registration: dynamic hedge ratio via Kalman filter

**Written before the extension was implemented or run. Committed prior to
seeing any result. Reported below regardless of outcome.**

---

## Motivation

The primary specification estimates the hedge ratio `beta` once per formation
window by OLS and then holds it fixed for the following 126 trading days. That
result is a null: gross annualised alpha of +0.08% in-sample (2010-2021) and
-1.66% out-of-sample (2022-2026).

The most obvious objection to that null is that the hedge ratio was stale. If
the true cointegrating relationship drifts — because one firm's leverage,
business mix, or factor loading changes over six months — then a beta fitted in
January is measuring the wrong relationship by June. Under that hypothesis the
spread would look non-stationary not because the pair is uncointegrated but
because the researcher held the coefficient fixed. This is the standard
critique, and it deserves a direct test rather than a paragraph of speculation.

## Hypothesis

**H1.** Allowing the hedge ratio to evolve as a latent state will produce
positive gross alpha where a static hedge ratio did not.

## Specification (fixed in advance)

Observation equation, with `theta_t = [beta_t, alpha_t]'`:

    log(y_t) = [log(x_t), 1] · theta_t + v_t,      v_t ~ N(0, R)

State equation — the hedge ratio follows a random walk:

    theta_t = theta_{t-1} + w_t,                   w_t ~ N(0, Q)

Parameter choices, all fixed now and none to be tuned:

| Quantity | Value | Justification |
|---|---|---|
| `delta` | `1e-4` | Chan (2013), *Algorithmic Trading*, ch. 3. The published default. Not selected by search. |
| `Q` | `delta / (1 - delta) · I₂` | Standard reparameterisation of state noise in terms of `delta`. |
| `R` | Formation-period residual variance | Estimated from formation data only. |
| `theta_0` | Formation-period OLS `[beta, alpha]` | Same estimate the primary specification uses. |
| `P_0` | Formation-period OLS coefficient covariance | Principled prior rather than an arbitrary diagonal. |

Everything else is held identical to the primary specification: the same
universe, the same cohorts, the same candidate pairs and screening rule, the
same entry/exit/stop thresholds (2.0 / 0.5 / 3.5), the same 60-day time stop,
and the same cost model.

## Signal construction

The Kalman innovation *is* the spread:

    e_t = log(y_t) - [log(x_t), 1] · theta_{t|t-1}
    S_t = H_t · P_{t|t-1} · H_t' + R
    z_t = e_t / sqrt(S_t)

This has a property worth stating explicitly: `theta_{t|t-1}` is the state
predicted from information through `t-1`, so `z_t` is causal **by
construction**. The look-ahead bug quantified in the primary study — worth
+6.8pp of annualised return and 3.9 Sharpe points — is structurally impossible
in this specification. That is an argument for the method independent of
whether it turns out to be profitable.

## Decision rule (fixed in advance)

The extension is declared successful only if it produces:

1. Positive **gross** annualised alpha, and
2. A Newey-West t-statistic above 2.0,

in **both** the in-sample and out-of-sample spans. Positive gross alpha
in-sample alone is not sufficient and will be reported as a failure.

## Constraints

- **One run.** No parameter is adjusted after seeing results.
- If H1 fails, the result is reported as a failed extension in the paper. It is
  not iterated on, re-specified, or quietly dropped.
- `delta` will not be swept. A `delta` sweep would convert this from a test
  into a search, which is precisely the practice the study exists to criticise.

---

## Deviation from this pre-registration

One deviation, recorded in full.

The fixed `delta = 1e-4` above is Chan's default for a filter run on **price
levels**. Applied to **log prices** it makes the state noise large relative to
the observation noise, so the filter re-fits the hedge ratio to each day's
price and absorbs the very deviation the strategy exists to trade. The symptom:
a trading-window z-score with standard deviation ≈ 0.15, which never reaches
the ±2 entry threshold and generates **zero trades**.

This was found by inspecting the z-score distribution during implementation
verification, **before any profit-and-loss was computed**. Reporting "zero
trades" would have been reporting a scaling bug as a scientific finding.

`delta` and `R` are therefore estimated jointly by maximum likelihood over a
fixed grid (`delta` ∈ {1e-8 … 1e-4}, `R` as a fraction ρ ∈ {0.001 … 1.0} of the
formation residual variance), **using formation-window data only**. The
estimator sees no trading-window prices and no returns, so it cannot be tuning
in the sense this study criticises.

*Known weakness:* for several pairs the selected ρ sits at the lower grid
boundary (0.001), meaning the likelihood would prefer a smaller value than the
grid allows. The grid was left as declared rather than extended, since
extending it after seeing results is exactly the behaviour pre-registration
exists to prevent. This should be noted as a limitation.

---

## RESULT — H1 REJECTED

Single run, as committed. No parameters adjusted after seeing these numbers.

| | Static β (primary) | **Kalman dynamic β** | Criterion |
|---|---|---|---|
| Gross ann., in-sample | +0.08% | **+0.89%** | positive ✓ |
| Gross ann., out-of-sample | −1.66% | **+0.40%** | positive ✓ |
| Newey-West t, in-sample | −3.80 | **+0.53** | > 2.0 ✗ |
| Newey-West t, out-of-sample | −2.77 | **−0.41** | > 2.0 ✗ |
| Net ann., in-sample | −3.08% | +0.24% | |
| Net ann., out-of-sample | −4.58% | −0.27% | |

**Verdict: H1 is rejected.** Gross alpha is positive in both spans, satisfying
criterion 1, but the t-statistics of 0.53 and −0.41 are nowhere near the
required 2.0. The result is indistinguishable from zero.

**What the extension did establish.** The staleness critique was partly
correct, and the improvement is systematic rather than noise:

| | Static β | Kalman β |
|---|---|---|
| Round trips (in-sample) | 6,513 | 1,338 |
| Transaction costs, ann. | −3.06% | −0.63% |
| Win rate | 41.9% | 55.8% |
| Max drawdown | −33.2% | −7.2% |

Letting the hedge ratio evolve cuts turnover roughly fivefold, lifts the win
rate above one-half, and moves the strategy from clearly loss-making to
approximately break-even. A meaningful share of the static specification's
losses came from trading against drift in the hedge ratio rather than from
genuine spread deviations.

It is not, however, enough to produce an edge. The correct reading is that a
stale hedge ratio was *a* problem with the primary specification, but not the
reason the strategy has no alpha.
