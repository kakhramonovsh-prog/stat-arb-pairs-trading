# Statistical Arbitrage via Cointegration: A Null Result

A rigorously specified pairs-trading study on US large-cap equities,
2010–2026. The headline finding is that the strategy **does not work**, and
that the profitable results common in the applied literature are reproducible
on demand by introducing a single, specific normalisation error.

---

## Summary of findings

**1. No tradeable edge exists.** Across 184 large-cap US equities and 11 sector
ETFs over sixteen years, a cointegration pairs strategy earns gross annualised
alpha statistically indistinguishable from zero, and loses money net of
realistic costs.

| | In-sample 2011–2021 | Hold-out 2022–2026 |
|---|---|---|
| Gross annualised | +0.08% | −1.66% |
| Net annualised | **−3.08%** | **−4.58%** |
| Sharpe | −1.63 | −2.06 |
| Newey-West *t* | −3.80 | −2.77 |
| Market beta | 0.041 | 0.020 |

The hold-out span was run **once**, after the design was frozen. It is worse
than in-sample, not better.

**2. The bias that makes it look profitable is worth ~6.8 percentage points.**
Standardising the spread by the mean and standard deviation of *its own trading
window* — rather than by frozen formation-period moments — uses prices that
have not happened yet at the moment of the trade. It is a one-line change:

| Specification | Ann. return | Sharpe | Max DD | Win rate | Trades |
|---|---|---|---|---|---|
| Frozen formation moments | −3.08% | −1.63 | −33.2% | 41.9% | 6,513 |
| Causal rolling 60d | −0.78% | −0.80 | −18.5% | 54.4% | 1,325 |
| **Look-ahead z-score** | **+6.66%** | **+3.15** | **−0.9%** | **88.8%** | 348 |

A Sharpe of 3.15 with a 0.9% maximum drawdown is the signature of a backtest
that has seen the future. The bias is worth +6.8pp of annualised return and
3.9 Sharpe points.

**3. Cointegration is barely distinguishable from chance.** Screening ~1,300
within-sector pairs per formation window typically yields 45–128 pairs
significant at p < 0.05 against ~65 expected under the null. Storey's estimate
of the true-null proportion is ≈ 1.00 in almost every window, and Benjamini-
Hochberg at a 5% FDR rejects nothing. The expected false discovery proportion
of the traded book averages **0.64**.

**4. A dynamic hedge ratio helps materially, but not enough.** A pre-registered
Kalman filter extension ([PREREGISTRATION.md](PREREGISTRATION.md)) cuts turnover
fivefold, raises the win rate from 41.9% to 55.8%, and shrinks maximum drawdown
from −33% to −7%. Gross alpha turns positive in both spans. It still fails the
declared significance threshold (*t* = 0.53 in-sample, −0.41 out-of-sample) and
**H1 was rejected.**

![Look-ahead bias](results/figures/02_lookahead_bias.png)

---

## Why the null result is the interesting one

Gatev, Goetzmann & Rouwenhorst (2006) documented ~11% annualised excess returns
to distance-based pairs trading over 1962–2002. Do & Faff (2010) showed the
profitability decayed sharply after 2002. This study is consistent with the
decay having gone to completion in US large caps: by 2010–2026 there is nothing
left after costs.

That is a boring conclusion, and it is why so few published backtests reach it.
The interesting question is not whether pairs trading works — it is what
methodological choices separate the studies that find it working from those
that do not. This repository isolates one such choice and measures it.

---

## Method

**Design.** The sample is cut into non-overlapping cohorts of a 252-day
formation window followed by a 126-day trading window (Gatev et al. 2006). Pair
selection, hedge ratio estimation, and the spread's mean and standard deviation
are all estimated on formation data and then frozen. No trading-window
observation informs its own signal.

**Screening.** Candidate pairs are formed only *within* GICS sector — an
economic prior, not a computational shortcut. Engle-Granger two-step with
MacKinnon residual-based critical values (running `adfuller` on an OLS residual
by hand over-rejects, and is the most common error in pairs-trading code).
Johansen's trace test is reported as an independent check. Pairs are filtered
on an Ornstein-Uhlenbeck half-life between 5 and 60 trading days.

**Multiple testing.** ~1,300 tests per window are corrected via Benjamini-
Hochberg, with Storey's π₀ and the expected false discovery proportion of the
selected book reported for every cohort. Because BH rejects nothing at scale,
selection is rank-based and the expected noise fraction is disclosed rather
than hidden.

**Costs.** 1 bp commission plus 4 bps slippage per leg per side, and 50 bps
annualised borrow on the short leg. A round trip crosses the spread four times.
Capital allocated to a flat pair earns nothing — the strategy genuinely cannot
deploy it.

Full detail in [METHODOLOGY.md](METHODOLOGY.md).

---

## Reproducing

```bash
python -m venv venv && venv/Scripts/activate
pip install -r requirements.txt
```

```bash
python scripts/01_fetch_data.py
```

```bash
python scripts/02_select_pairs.py --span is
```

```bash
python scripts/03_backtest.py --span is
```

```bash
python scripts/04_diagnostics.py
```

```bash
python scripts/05_figures.py
```

Add `--kalman` to step 3 for the pre-registered extension, and `--span oos` for
the hold-out.

---

## Layout

```
config.py                 universe, sample split, every tunable parameter
src/data.py               download, clean, cache adjusted closes
src/cointegration.py      Engle-Granger, Johansen, OU half-life, FDR, Storey pi0
src/pairs.py              rolling formation/trading cohorts and screening
src/signals.py            z-score construction and the position state machine
src/kalman.py             dynamic hedge ratio (pre-registered extension)
src/backtest.py           P&L, costs, trade log
src/metrics.py            Sharpe, drawdown, Newey-West t, market neutrality
scripts/01..05            the pipeline, in order
PREREGISTRATION.md        extension declared in advance, result reported as-is
```

---

## Limitations

Stated plainly, because a null result is only worth as much as its caveats.

- **Survivorship bias.** The universe is today's index membership projected
  backwards. Firms acquired or delisted during the sample are absent — HES,
  IPG and MMC were dropped for exactly this reason. This biases results
  *upward*, which makes a negative finding more credible, not less.
- **Daily data only.** Intraday mean reversion is not tested. The conclusion
  applies to daily-frequency pairs trading in large caps and does not
  generalise to higher frequencies.
- **Long-only-in-spread universe.** No borrow availability constraints are
  modelled beyond a flat 50 bps; hard-to-borrow names would be worse.
- **Low test power.** Engle-Granger with 252 observations has limited power. A
  π₀ near 1.0 may partly reflect that rather than a true absence of
  cointegration.
- **Grid boundary.** In the Kalman extension the MLE for the observation-noise
  ratio frequently selects the lowest value the declared grid permits.
- **One asset class, one country, one regime.** 2010–2026 was an unusual
  monetary environment throughout.

---

## References

Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale.*
Wiley.

Do, B. & Faff, R. (2010). Does simple pairs trading still work? *Financial
Analysts Journal*, 66(4).

Engle, R. & Granger, C. (1987). Co-integration and error correction. *Econometrica*, 55(2).

Gatev, E., Goetzmann, W. & Rouwenhorst, K. (2006). Pairs trading: performance
of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3).

Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors.
*Econometrica*, 59(6).

Storey, J. (2002). A direct approach to false discovery rates. *JRSS-B*, 64(3).
