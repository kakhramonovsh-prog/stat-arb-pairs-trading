# Methodology

Written to serve as the methods section of the accompanying working paper.
Every design choice that could plausibly have gone the other way is stated,
along with the reason it went the way it did.

---

## 1. Data

**Source.** Yahoo Finance daily bars via `yfinance`, split- and
dividend-adjusted (`auto_adjust=True`).

**Why total-return prices.** Positions are held for weeks. Raw closes would
make every dividend and split appear as a break in the spread, and the
cointegration test would read those breaks as evidence against a stable
relationship.

**Universe.** 184 securities: 11 SPDR sector ETFs and 173 large-cap US
equities, grouped by GICS sector. XLRE and XLC begin mid-sample (2015 and 2018)
and are handled as ragged rather than dropped.

**Sample.** 2010-01-04 to 2026-07-31, 4,169 trading days.

**Cleaning.** Gaps of up to 3 days are forward-filled; longer gaps are left
missing. The distinction matters: short gaps are holidays and missed prints,
whereas long gaps mean the security did not trade, and forward-filling those
would fabricate a flat series that the ADF test would happily call stationary.
Non-positive prices are masked. Tickers with under 30% coverage are dropped.

**Log prices.** All analysis is on natural logs. Log prices are closer to the
I(1) random walk the Engle-Granger procedure assumes, and a spread in logs is a
ratio in levels, so the hedge ratio is a proportional relationship rather than
one that drifts mechanically as the price level changes over sixteen years.

### 1.1 Survivorship bias

The universe is current index membership projected backwards. Firms that were
large caps in 2010 and have since been acquired or delisted are absent. Three
names were dropped mid-study for precisely this reason: **HES** (acquired by
Chevron), **IPG** (merged into Omnicom), and **MMC**.

The direction of the bias is upward — surviving firms outperformed — so a
strategy evaluated on this universe should look *better* than reality. The
study finds no edge anyway, which makes the negative result more credible
rather than less. Point-in-time constituent data (CRSP/Compustat) would remove
the bias and is not freely available.

---

## 2. Research design

### 2.1 Formation and trading windows

The sample is partitioned into non-overlapping cohorts:

```
|<-- formation: 252 days -->|<-- trading: 126 days -->|
                             |<-- next formation ...
```

Everything used to generate a trade — which pairs are in the book, the hedge
ratio β, and the mean and standard deviation defining the z-score — is
estimated on formation data and frozen before the trading window opens.

This follows Gatev, Goetzmann & Rouwenhorst (2006). It is deliberately more
conservative than estimating the cointegrating vector once over the full
sample, which is the largest single source of overstated results in applied
pairs-trading work: a hedge ratio fitted on data that includes the trading
period already knows where the spread is going.

Window boundaries are taken from the realised trading calendar rather than by
adding calendar days, so lengths are exact in trading days and do not drift
with holidays.

### 2.2 In-sample / hold-out split

| Span | Dates | Use |
|---|---|---|
| In-sample | 2010-01-01 – 2021-12-31 | all design, all exploratory work |
| Hold-out | 2022-01-01 – 2026-07-31 | run once, after design freeze |

The hold-out's first formation window draws on late-2021 prices. Using past
prices to form a book is not look-ahead; only the trading periods are reserved.

---

## 3. Cointegration testing

### 3.1 Candidate construction

Pairs are formed only within sector: ~1,300 candidates per window. A blind
all-pairs search across 184 tickers would test ~16,800 pairs and, at a 5% size,
return ~840 "significant" results under the null of no cointegration anywhere.
Restricting to same-sector pairs imposes an economic prior — firms exposed to
common demand shocks and input costs have a reason to share a stochastic trend.

### 3.2 Engle-Granger two-step

Step 1 regresses log *y* on log *x* by OLS and retains the residual. Step 2
tests that residual for a unit root.

**Critical values.** Standard Dickey-Fuller critical values are invalid here.
The residual is not observed data; it is the output of a regression whose
coefficient was chosen to minimise it. OLS therefore makes the residual look
more stationary than it is, and a naive ADF test over-rejects the null of no
cointegration. `statsmodels.tsa.stattools.coint` applies MacKinnon's
residual-based critical values, which correct for this. Calling `adfuller` on
an OLS residual by hand is the most common error in this literature.

**Direction.** Engle-Granger is not symmetric — regressing A on B yields a
different residual than B on A. Both directions are run and the more
significant retained. Each p-value is then doubled (Bonferroni over the two
directions), so the choice does not smuggle in an uncounted look at the data.

**Hedge ratio.** OLS is asymptotically biased in this setting but
*superconsistent* under cointegration: it converges at rate *T* rather than
√*T* (Stock 1987), so with 252 observations the bias is small relative to
estimation noise.

### 3.3 Johansen trace test

Reported as an independent robustness statistic, not as the primary screen.
Johansen treats both series symmetrically via full-system maximum likelihood
and lacks Engle-Granger's directional arbitrariness. It is the more powerful
test but more sensitive to lag misspecification in short samples. Agreement
between the two is meaningful; disagreement flags a marginal pair.

Complex eigenvalues arising from near-collinear windows are a numerical
artefact of an ill-conditioned covariance matrix; imaginary parts are discarded
explicitly rather than silently.

### 3.4 Half-life filter

The spread's speed of mean reversion is estimated from a discrete
Ornstein-Uhlenbeck fit. The OU process `ds = θ(μ − s)dt + σdW` discretises to

```
s_t − s_{t−1} = a + b·s_{t−1} + e_t,     half-life = −ln(2) / b
```

Reversion requires *b* < 0; otherwise the half-life is infinite and the pair is
discarded. Pairs are kept only with half-lives between 5 and 60 trading days.

This filter earns its place because a significant p-value says a relationship
exists, not that it reverts fast enough to trade. A pair with a 400-day
half-life passes the ADF test and then sits in a losing position for the entire
126-day trading window.

### 3.5 Multiple testing

Screening ~1,300 pairs at a nominal 5% level yields ~65 false positives per
window under the null. Three quantities are reported for every cohort:

- **Benjamini-Hochberg** at a 5% FDR. Preferred to Bonferroni, which controls
  the probability of *any* false positive and would leave nothing to trade. BH
  bounds the expected *proportion* of false positives, which is what a
  portfolio actually cares about. Caveat: BH assumes independence or positive
  dependence, and same-sector pairs share constituents. The dependence is
  positive, which is the case BH handles, but the guarantee is weaker.
- **Storey's π₀**, the estimated proportion of true nulls, via
  `π₀ = #{p > 0.5} / (m/2)`.
- **Expected FDP** of the selected book: `π₀ · m · p₍ₖ₎ / k`.

**Why BH is not used as a gate.** Applied as a hard filter it rejects nothing
in nearly every window: it requires the *k*-th smallest p-value below
*k*·α/*m*, and the observed p-values sit an order of magnitude above that line
(the 50th smallest is 0.021 against a required 0.0019). Gating on it produces
an empty book, which answers a question the strategy is not asking.

The portfolio question is not "is this pair individually significant after
1,300 tests" but "does a book of the strongest-evidence pairs earn its keep on
data that played no part in selecting it." That is settled by the trading
window — a stricter test, since a spurious pair has no reason to keep reverting
after formation ends. Selection is therefore rank-based on the Engle-Granger
statistic (more negative = stronger evidence against a unit root), capped at 20
pairs, with the expected noise fraction disclosed.

**Concentration cap.** No single ticker may appear in more than two selected
pairs per cohort, so one stock cannot dominate the book's risk.

---

## 4. Trading rules

**Spread.** `spread = log(y) − β·log(x) − α`. A positive spread means *y* is
rich relative to *x*.

**Standardisation.** `z = (spread − μ_formation) / σ_formation`, with both
moments frozen from the formation window.

This is the study's most consequential choice. The obvious alternative —
standardising by the trading window's own moments — uses prices that have not
happened yet. A spread ending the window far from where it started has its
early deviations understated, and the backtest shows the strategy entering at
exactly the right moments. The effect is measured in §6 and is large.

**Entry / exit.**

| Rule | Threshold | Rationale |
|---|---|---|
| Entry | \|z\| > 2.0 | position = −sign(z): short the rich leg |
| Exit | \|z\| < 0.5 | convergence |
| Stop | \|z\| > 3.5 | the relationship has broken |
| Time stop | 60 days | roughly the maximum permitted half-life |

The stop-loss exists because cointegration tests cannot see structural breaks
coming — a merger, a restructuring, a change in one leg's business mix. Frozen
formation moments mean the z-score can legitimately drift far past the entry
threshold, and the stop is the only defence.

**Position sizing.** Weights are β-neutral, not dollar-neutral:
`w_y = 1/(1+|β|)`, `w_x = −β/(1+|β|)`, normalised to unit gross exposure. The
two coincide only when β = 1. Dollar-neutrality would leave residual exposure
to the very common factor the hedge ratio exists to strip out. Normalising
gross exposure prevents a pair with β = 3 from silently taking three times the
risk of a pair with β = 1.

**Capital.** Each pair receives 1/20 of capital. Capital allocated to a flat
pair earns nothing. This understates returns relative to scaling up so the book
is always fully invested, and it is the honest treatment: the strategy cannot
deploy that capital because it has no signal.

**Timing.** Positions are decided at the close of day *t* using only
information observable at that close, and earn the return from *t* to *t*+1,
enforced by shifting the position series forward one day.

---

## 5. Costs

| Component | Level |
|---|---|
| Commission | 1 bp per leg per side |
| Slippage | 4 bps per leg per side |
| Borrow | 50 bps annualised on the short leg |

A round trip crosses the spread four times (two legs, open and close). Results
are reported net throughout; a gross-only presentation of a strategy trading a
6-day half-life spread would be meaningless.

---

## 6. Control experiments

Run on in-sample data only.

**Look-ahead control.** The trading-window-moment normalisation is deliberately
reintroduced to quantify the bias. If the buggy variant were also flat, a flat
honest result would more likely indicate a broken simulator than an absent
edge. It is not flat: Sharpe rises from −1.63 to +3.15, win rate from 41.9% to
88.8%, and maximum drawdown falls from −33.2% to −0.9%.

**Causal rolling normalisation.** Freezing μ and σ for 126 days is the
strictest choice and arguably too strict, since a cointegrating relationship's
level can drift legitimately. A trailing estimate ending at *t*−1 is fully
causal. Tested at 30, 60 and 90 days: turnover falls fivefold and win rate
rises above one-half, but gross alpha remains ≈ 0.

**Cost sensitivity.** Zero, half, and double the baseline. At zero cost the
in-sample net return is −0.02% with *t* = −0.03 — the gross edge is not small
and eaten by costs, it is absent.

---

## 7. Performance measurement

All statistics on net returns. Sharpe and Sortino annualised at 252 days over a
2% risk-free rate.

**Newey-West t-statistics** with 21 lags. Daily strategy returns are
autocorrelated because positions persist for days, so the naive standard error
is optimistic.

**Market neutrality** is tested, not asserted: strategy excess returns are
regressed on SPY excess returns with HAC errors. Realised market beta is 0.041
in-sample and 0.020 out-of-sample, confirming the hedge works — there is simply
nothing left over.

---

## 8. Pre-registered extension

See [PREREGISTRATION.md](PREREGISTRATION.md) for the full specification,
the single documented deviation from it, and the result. Summary: a Kalman
filter treating `[β_t, α_t]` as a random-walk latent state improves turnover,
win rate and drawdown substantially, turns gross alpha positive in both spans,
and still fails the declared significance threshold. H1 rejected.

The filter's innovation *is* the spread and its predicted variance *is* the
scale, both computed from a state predicted using information through *t*−1.
The z-score is therefore causal by construction, and the look-ahead error
quantified in §6 is not merely avoided but unrepresentable.
