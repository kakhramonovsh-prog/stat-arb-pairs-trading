# Anatomy of a Backtest: Look-Ahead Bias and the Economics of Statistical Arbitrage in US Large Caps, 2010–2026

**Shakhrukh Kakhramonov**
Zicklin School of Business, Baruch College
kakhramonovsh@gmail.com

*Working paper — August 2026*

---

## Abstract

I evaluate three statistical arbitrage specifications on 184 US large-cap equities over 2010–2026: bilateral cointegration pairs with a static hedge ratio, the same strategy with a Kalman-filtered dynamic hedge ratio, and cross-sectional mean reversion in PCA factor residuals following Avellaneda and Lee (2010). Under a strict formation/trading discipline with a once-only hold-out, no specification is profitable net of realistic costs. Two results are of independent interest. First, a common implementation error, standardising the spread by its own trading-window moments, is worth +9.7 percentage points of annualised net return (95% CI [+8.0, +11.8], moving-block bootstrap). It converts a strategy that loses 3.1% per year into an apparent Sharpe ratio of 3.15 with an 88.8% win rate. Second, the cross-sectional specification earns statistically significant gross alpha in-sample (+1.31%, Newey-West t = 2.33) with a win rate near 57% in both spans across 24,674 round trips, but the effect breaks even only below 2.1 basis points per side. The anomaly persists because the only participants able to harvest it are already doing so. Both extensions were pre-registered with fixed parameters and declared success criteria; both hypotheses failed and are reported as failures.

*Keywords: statistical arbitrage, pairs trading, cointegration, look-ahead bias, transaction costs, multiple testing*

---

## 1. Introduction

Pairs trading has an unusual property for a quantitative strategy: its academic record and its practitioner folklore point in opposite directions. Gatev, Goetzmann and Rouwenhorst (2006) documented 11% annualised excess returns over 1962–2002, and Do and Faff (2010) showed that profitability decayed sharply after 2002. Yet profitable pairs-trading backtests remain abundant in books, blogs, and code repositories. Either the strategy revived, or something in how those backtests are built manufactures the profits.

This paper measures the second explanation. I build a pairs-trading engine under a discipline strict enough that its answer can be trusted, run it honestly, and then re-introduce the most common implementation error on purpose, to price that error alone. The error is one line of code: normalising the traded spread by moments computed over the very window being traded. It is easy to write by accident and hard to spot in someone else's backtest. On identical pair selections it moves annualised net returns by +9.7 percentage points and gives a money-losing strategy the appearance of a top-tier market-neutral fund.

The honest engine produces a well-supported null: bilateral cointegration pairs in modern US large caps earn gross alpha statistically indistinguishable from zero. Two pre-registered extensions then test whether the null reflects the method rather than the market. A Kalman-filtered dynamic hedge ratio, the standard answer to the objection that a six-month-old hedge ratio is stale, improves the strategy on every operational margin and still fails its declared significance threshold. Cross-sectional mean reversion in PCA residuals, the formulation of Avellaneda and Lee (2010) that superseded bilateral pairs in practice, does find statistically significant gross alpha in-sample. It is the study's only significant span. It survives costs in neither: the average trade earns about 5 basis points of gross edge over 9.5 days against roughly 20 basis points of round-trip cost.

The paper contributes a bootstrapped price on a specific, widespread backtest error, and a cost-decomposed account of where mean-reversion alpha in US large caps went. Measured here, that alpha still exists; it has shrunk below the cost of collecting it for anyone paying more than about 2 basis points per side.

## 2. Related literature

Engle and Granger (1987) supplied the econometric foundation: if two integrated price series share a stochastic trend, some linear combination of them is stationary, and deviations from it are transient by construction. Johansen (1991) provided the system maximum-likelihood treatment. Gatev, Goetzmann and Rouwenhorst (2006) turned the idea into the canonical trading study, and this paper adopts their rolling formation/trading design directly. Do and Faff (2010, 2012) documented the post-2002 decay and showed that transaction costs eliminate most of what remained.

Avellaneda and Lee (2010) reformulated the trade cross-sectionally: decompose returns into principal-component factors, model the residual as an Ornstein-Uhlenbeck process, and trade its standardised deviation (the s-score) across the whole universe at once. Their reported performance decays late in their own sample, a pattern this paper's out-of-sample result extends.

On method, Harvey, Liu and Zhu (2016) argue that most published cross-sectional return factors fail once multiple testing is accounted for. The screening diagnostics here, Storey (2002) estimates of the true-null proportion and Benjamini-Hochberg (1995) false discovery control, apply the same machinery inside a single strategy rather than across a literature. Chan (2013) is the source of the Kalman specification practitioners use, and of the parameter convention whose naive transfer to log prices this study documents as a failure mode.

## 3. Data

Daily split- and dividend-adjusted closes from Yahoo Finance for 184 securities: 173 large-cap US equities grouped by GICS sector and 11 SPDR sector ETFs, from 2010-01-04 to 2026-07-31 (4,169 trading days). Total-return prices matter because positions are held for weeks; on raw closes, every dividend would appear as a structural break in the spread.

Gaps of up to three days are forward-filled. Longer gaps stay missing, because forward-filling a halted or delisted security fabricates a flat price series that a stationarity test will happily call mean-reverting. All analysis uses natural log prices.

The universe is current index membership projected backwards, which induces survivorship bias: firms acquired or delisted during the sample are absent (three tickers, HES, IPG, and MMC, were dropped for this reason). The bias inflates measured returns, so it strengthens a negative finding. Point-in-time constituent data would remove it and is not freely available.

I split the sample once, before any analysis: 2010–2021 for all design and exploration, 2022–2026 as a hold-out run once per specification after the design was frozen.

## 4. The pairs engine

### 4.1 Design

The sample is cut into non-overlapping cohorts: a 252-day formation window followed by a 126-day trading window. Everything that generates a trade (the pairs in the book, the hedge ratio, the spread's mean and standard deviation) is estimated on formation data and frozen before trading begins. Positions are decided at the close of day *t* using information available at that close and earn the return from *t* to *t*+1.

### 4.2 Screening

Candidate pairs are formed only within sector, about 1,300 per cohort. The restriction is an economic prior: firms exposed to common demand and input-cost shocks have a reason to share a stochastic trend. It also disciplines the search, since a blind sweep over all 16,800 possible pairs would produce about 840 spuriously "significant" results per cohort at the 5% level.

Each candidate is tested with the Engle-Granger two-step procedure. One detail is frequently wrong in applied work: the residual of the cointegrating regression cannot be tested against standard Dickey-Fuller critical values. That residual is the output of a regression whose coefficients were chosen to minimise it, so it looks more stationary than the data justify, and a naive ADF test over-rejects. MacKinnon's residual-based critical values correct for this. I test both regression directions (the procedure is not symmetric) and double the resulting p-value, so the choice of direction is paid for rather than smuggled in.

Surviving pairs are filtered on the half-life of an Ornstein-Uhlenbeck fit to the formation spread, kept between 5 and 60 trading days. A significant p-value says a relationship exists; it says nothing about whether the relationship reverts fast enough to trade inside a 126-day window. The filter has limited power of its own. In simulation, a pure random walk of 500 observations shows a median estimated half-life near 60 days, the same finite-sample bias that motivates Dickey-Fuller critical values. This is one more reason the study leans on disclosed error rates rather than any single screen.

### 4.3 Multiple testing, and why the book is selected by rank

Across the 1,300 tests per cohort, between 45 and 128 pairs pass at p < 0.05, against about 65 expected under the null. The excess is real but diffuse. Storey's estimate of the true-null proportion sits near 1.00 in almost every cohort, and Benjamini-Hochberg control at a 5% false discovery rate rejects nothing: it requires the k-th smallest p-value to fall below k·α/m, and the observed 50th-smallest p-value is 0.021 against a required 0.0019.

Rather than trade an empty book, I select the 20 strongest pairs by Engle-Granger statistic and disclose the contamination: the expected false discovery proportion of the selected book, computed from Storey's estimator, averages 0.64 across cohorts. The justification is that the portfolio question, whether a book of the strongest-evidence pairs earns its keep on data that played no role in selecting it, is answered by the out-of-sample trading window. That window is a harsher test than any in-sample correction, because a spurious pair has no reason to keep reverting after its formation window ends.

### 4.4 Trading rules and costs

The spread is standardised by its frozen formation moments. Positions open at |z| > 2, close at |z| < 0.5, and stop out at |z| > 3.5 or after 60 days. Position weights are beta-neutral with unit gross exposure per pair; each pair receives 1/20 of capital, and capital in flat pairs earns nothing. Costs are 1 bp commission and 4 bps slippage per leg per side, with a round trip crossing the spread four times, plus 50 bps annualised borrow on the short leg.

## 5. Results I: the honest engine and the price of look-ahead

The honest engine loses money in both spans (Table 1): net −3.08% annualised in-sample (Sharpe −1.63, Newey-West t = −3.80) and −4.58% on the hold-out. The instructive number is gross: +0.08% in-sample. At zero cost the strategy returns −0.02% (t = −0.03). The edge is not small and cost-consumed; it is absent. Market beta is 0.04 in-sample and 0.02 out-of-sample. The hedging works as designed, and nothing is left after the hedge.

**Table 1 — Pairs strategy, primary specification**

| | In-sample 2011–2021 | Hold-out 2022–2026 |
|---|---|---|
| Gross annualised | +0.08% | −1.66% |
| Net annualised | −3.08% | −4.58% |
| Sharpe (net) | −1.63 | −2.06 |
| Newey-West t (net) | −3.80 | −2.77 |
| Max drawdown | −33.2% | −19.8% |
| Round trips | 6,513 | 2,309 |
| Win rate | 41.9% | 43.0% |
| Market beta | 0.041 | 0.020 |

A flat honest result is also consistent with a broken simulator, so I validate the engine by breaking it in the direction the literature's errors break. Standardising each spread by the mean and standard deviation of its own trading window, moments not knowable until the window ends, yields on identical cohorts and identical selected pairs: +6.66% net annualised, Sharpe 3.15, maximum drawdown −0.9%, win rate 88.8%.

The gap between the biased and honest daily return series is +9.74 percentage points annualised. A moving-block bootstrap (10,000 draws, 21-day blocks, matching the Newey-West lag window) puts the 95% confidence interval at [+7.97, +11.82]. Decomposed, +6.7 points is gross-signal inflation: the biased z-score enters near what hindsight identifies as extremes. The remainder is the extra transaction cost the honest signal incurs by trading nearly nineteen times as often (6,513 round trips against 348) without hindsight's timing.

Two further controls bound the honest specification's fragility. A causal trailing normalisation (rolling 30/60/90-day moments lagged one day), arguably closer to how a desk runs the trade than frozen moments, raises the win rate above 50% and cuts turnover fivefold, but gross alpha stays within rounding error of zero. Cost sensitivity confirms the null is not a cost artefact, since the strategy fails at zero cost.

## 6. Results II: two pre-registered extensions

Both extensions were specified in advance. Parameters were fixed from published sources, success criteria declared, and one run permitted per span, with the specification documents committed to the repository before results existed. Each records one documented deviation, found during implementation verification before any P&L was computed. Both hypotheses failed their declared criteria and are reported as failures.

### 6.1 Kalman dynamic hedge ratio

The obvious objection to Section 5 is staleness: a hedge ratio estimated in January measures the wrong relationship by June. The extension treats the cointegrating coefficients as a random-walk latent state, updated daily by a Kalman filter seeded from formation-window OLS. The filter's innovation is the spread and its predicted variance is the spread's scale, both computed from the state predicted using information through the previous day. The look-ahead error of Section 5 therefore cannot be expressed in this specification at all.

The pre-registered fixed state-noise parameter (δ = 10⁻⁴, from Chan 2013) turned out to be calibrated for price levels. On log prices it lets the state absorb the spread itself, producing z-scores that never reach the entry threshold and zero trades. I caught this at the verification stage from the z-score distribution alone. The documented fix estimates the noise parameters by maximum likelihood on formation data only, over a fixed grid.

The result (Table 2) is a large operational improvement that fails its significance criterion. Turnover falls fivefold, the win rate rises from 41.9% to 55.8%, maximum drawdown shrinks from −33% to −7%, and gross alpha turns positive in both spans, but at t = 0.53 in-sample and −0.41 out-of-sample against a declared threshold of 2.0. The staleness critique was partly correct: a meaningful share of the static specification's losses came from trading against hedge-ratio drift. It does not explain why the strategy has no alpha.

**Table 2 — Extensions, gross annualised alpha**

| Specification | In-sample | t (NW) | Hold-out | t (NW) | Verdict |
|---|---|---|---|---|---|
| Pairs, static β | +0.08% | — | −1.66% | — | no signal |
| Pairs, Kalman β | +0.89% | 0.53 | +0.40% | −0.41 | H1 rejected |
| Cross-sectional PCA residuals | +1.31% | **2.33** | +0.82% | 0.67 | H2 rejected |

### 6.2 Cross-sectional PCA residual reversion

The second extension implements Avellaneda and Lee (2010) with their published parameters: fifteen eigenportfolio factors from a trailing 252-day correlation matrix, per-stock factor regressions over trailing 60 days, an OU fit to the cumulative residual yielding an s-score, entry at |s| > 1.25, exits at −0.50 and +0.75, and a speed filter requiring annualised κ > 252/30. Estimation repeats daily, and every quantity used on day *t* comes from data through the close of *t*. The fifteen factors explain 55–62% of return variance across the sample, matching the original paper.

The pre-registered cost basis, proportional to the sum of absolute factor loadings, turned out to depend on an arbitrary normalisation convention of the eigenportfolios. The documented deviation charges the same two-leg cost structure as the pairs study. I caught this, too, before any P&L existed.

This specification produces the study's only statistically significant result: gross alpha of +1.31% in-sample with a Newey-West t of 2.33, on a win rate of 56.9% over 17,465 round trips. Out-of-sample the gross alpha remains positive (+0.82%) with a nearly unchanged win rate (56.5%, 7,209 trips) but a t of 0.67, short of the declared criterion, so H2 fails. Net of costs, both spans lose 2–3% per year: the average trade's 5 bps of gross edge over 9.5 days cannot pay 20 bps of round-trip cost.

Because costs enter the accounting linearly, the saved return decomposition implies a breakeven cost of 2.1 bps per side in-sample and 1.3 bps out-of-sample, against the 5 bps this study assumes. Idiosyncratic mean reversion in US large caps is real, stable in frequency, and measurable, and only executors whose all-in costs sit near the institutional floor can harvest it. An anomaly priced below the marginal participant's cost of entry needs no further explanation for its persistence.

## 7. The progression, read as one result

The three specifications line up. Moving from static bilateral pairs to a dynamic hedge ratio to cross-sectional residual reversion, gross in-sample alpha rises from +0.08% to +0.89% to +1.31%, win rates rise from 42% to 56% to 57%, and the last step reaches statistical significance. Each methodological refinement recovers more genuine signal. None of it clears the cost hurdle, and the recovered signal shrinks out-of-sample in every case, most sharply for the significant one (+1.31% to +0.82%), consistent with continued crowding.

Two readings fit this pattern. In the optimistic one, further refinements (better factor models, higher frequencies, smarter execution) continue the progression and eventually clear costs. In the disciplined one, the progression is what a mature anomaly looks like from outside: each improvement recovers a larger fraction of a pie that is itself shrinking toward the institutional cost floor. The out-of-sample decay favours the second reading. Either way, any future profitable backtest in this family inherits a quantified burden: demonstrate an edge exceeding +9.7pp of potential look-ahead contamination, survive a once-only hold-out, and break even below 2 bps per side.

## 8. Limitations

Survivorship bias inflates all reported returns by an unknown amount; it cannot rescue a negative result but does affect the gross-alpha point estimates. Daily bars leave intraday reversion untested, so the conclusions apply to daily-frequency trading only. Engle-Granger with 252 observations has modest power, and the near-unity Storey estimates partly reflect the test rather than the world. The Kalman noise-parameter grid frequently binds at its declared lower boundary; the pre-registration ruled out extending it after the fact. Cost assumptions are stylised, and the breakeven statements inherit that. The sample covers one asset class in one country through one unusual monetary regime.

## 9. Conclusion

An honest pairs-trading backtest on modern US large caps returns approximately nothing gross and loses money net. The profitable versions of this strategy that circulate publicly are largely reproducible by a one-line normalisation error worth +9.7 percentage points a year. The modern cross-sectional formulation still contains statistically demonstrable mean reversion, found here at t = 2.33 in-sample with win rates within half a point of each other across spans, but its economic value now lives below 2.1 basis points per side of execution cost. Much of what appears to beat this market measures the researcher's information set rather than the market's inefficiency, and this paper puts a number on the difference.

## References

Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, 57(1), 289–300.

Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.

Do, B., & Faff, R. (2010). Does simple pairs trading still work? *Financial Analysts Journal*, 66(4), 83–95.

Do, B., & Faff, R. (2012). Are pairs trading profits robust to trading costs? *Journal of Financial Research*, 35(2), 261–287.

Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251–276.

Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.

Storey, J. D. (2002). A direct approach to false discovery rates. *Journal of the Royal Statistical Society, Series B*, 64(3), 479–498.

---

*Code, data pipeline, pre-registration documents, and all results tables are public at the accompanying repository. Both pre-registered hypotheses in this paper failed; the pre-registration files record one documented specification deviation each, both identified before any profit-and-loss computation.*
