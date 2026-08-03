"""
Central configuration for the statistical arbitrage study.

Everything that could be tuned lives here, in one file, so that the set of
researcher degrees of freedom is auditable. A reader of the paper can see the
complete parameter surface at a glance and judge how much room there was to
overfit.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Sample period
# --------------------------------------------------------------------------
# The full sample is split once, before any analysis is run:
#
#   IN-SAMPLE  (2010-01-01 .. 2021-12-31)  design, parameter selection, all
#                                          exploratory work
#   OUT-OF-SAMPLE (2022-01-01 .. present)  touched exactly once, after the
#                                          design is frozen
#
# This is the single most important control in the study. Pairs trading is
# notoriously easy to overfit: Gatev, Goetzmann & Rouwenhorst (2006) reported
# ~11% annualised excess return over 1962-2002, but Do & Faff (2010) showed the
# strategy's profitability decayed sharply after 2002 as the trade became
# crowded. Any result that is only visible in-sample should be assumed dead.
START_DATE = "2010-01-01"
IS_END_DATE = "2021-12-31"
OOS_START_DATE = "2022-01-01"
END_DATE = "2026-08-01"

# --------------------------------------------------------------------------
# Formation / trading window design (Gatev et al. 2006)
# --------------------------------------------------------------------------
# FORMATION_DAYS of history is used to (a) test each candidate pair for
# cointegration, (b) estimate the hedge ratio, and (c) estimate the spread's
# mean and standard deviation. The next TRADING_DAYS are then traded using
# ONLY those formation-period estimates. Windows roll forward by TRADING_DAYS,
# so every trading day in the study is genuinely out-of-sample relative to the
# statistics that generated its signals.
FORMATION_DAYS = 252        # ~12 months of trading days
TRADING_DAYS = 126          # ~6 months
MIN_OBS_FRACTION = 0.95     # a ticker must have >=95% of the window's bars

# --------------------------------------------------------------------------
# Cointegration screening
# --------------------------------------------------------------------------
# Candidate pairs are formed only WITHIN a sector. This is an economic prior,
# not a computational shortcut: two firms exposed to the same demand shocks and
# input costs have a plausible reason to share a stochastic trend. A blind
# all-pairs search over 190 tickers would test ~18,000 pairs, and at a 5% size
# would produce ~900 "significant" results under the null of no cointegration.
EG_PVALUE_THRESHOLD = 0.05  # Engle-Granger p-value, BEFORE FDR correction
FDR_ALPHA = 0.05            # Benjamini-Hochberg false discovery rate

# Half-life of mean reversion, in trading days, estimated from an
# Ornstein-Uhlenbeck fit to the formation-period spread. Pairs outside this
# band are discarded: a half-life under the floor is usually microstructure
# noise that transaction costs will eat, and above the ceiling the spread will
# not round-trip inside a 126-day trading window.
MIN_HALF_LIFE = 5
MAX_HALF_LIFE = 60

MAX_PAIRS_PER_WINDOW = 20   # top N by Engle-Granger test statistic

# --------------------------------------------------------------------------
# Trading rules
# --------------------------------------------------------------------------
ENTRY_Z = 2.0               # open when |z| exceeds this
EXIT_Z = 0.5                # close when |z| falls back inside this
STOP_Z = 3.5                # abandon the position: the relationship has broken
MAX_HOLDING_DAYS = 60       # time stop, roughly MAX_HALF_LIFE

# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------
# A pairs trade is two legs, and both open and close. One "round trip" is
# therefore four crossings of the spread. Reporting results gross of costs is
# the most common way pairs-trading backtests mislead.
COMMISSION_BPS = 1.0        # per leg, per side
SLIPPAGE_BPS = 4.0          # per leg, per side; half-spread + impact
BORROW_COST_ANNUAL_BPS = 50.0   # cost of carrying the short leg

CAPITAL = 1_000_000.0       # notional, for reporting in dollar terms
MAX_CONCURRENT_PAIRS = 20
RISK_FREE_ANNUAL = 0.02     # for Sharpe; see METHODOLOGY.md on this choice

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# SPDR sector ETFs. XLRE (real estate) began trading in 2015 and XLC
# (communication services) in 2018, so both are absent from the early part of
# the sample. The data layer handles ragged start dates rather than silently
# dropping them.
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "HealthCare": "XLV",
    "Energy": "XLE",
    "Staples": "XLP",
    "Discretionary": "XLY",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Materials": "XLB",
    "RealEstate": "XLRE",
    "Communication": "XLC",
}

# Large-cap constituents grouped by GICS sector. Pairs are searched within
# each list.
#
# KNOWN LIMITATION -- survivorship bias. This universe is today's index
# membership projected backwards. Firms that were large caps in 2010 and were
# subsequently acquired or delisted are absent, which biases results upward to
# an unknown degree. Point-in-time constituent data (CRSP, Compustat) would fix
# this and is not freely available. This is disclosed in METHODOLOGY.md rather
# than quietly ignored.
UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC",
        "QCOM", "TXN", "CSCO", "IBM", "ACN", "INTU", "AMAT", "MU", "ADI",
        "LRCX", "KLAC",
    ],
    # BK was retired as a ticker when Bank of New York Mellon rebranded to
    # "BNY"; Yahoo serves the full 2010-present history under the new symbol.
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "AXP", "USB",
        "PNC", "TFC", "BNY", "CB", "AIG", "MET", "PRU", "TRV", "ALL",
    ],
    "HealthCare": [
        "JNJ", "UNH", "LLY", "PFE", "MRK", "ABT", "TMO", "DHR", "BMY", "AMGN",
        "GILD", "CVS", "CI", "MDT", "SYK", "BSX", "ZTS", "REGN", "VRTX",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "WMB",
        "KMI", "OKE", "HAL", "DVN",
    ],
    "Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "SYY", "KR", "STZ", "HSY",
    ],
    "Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
        "TGT", "GM", "F", "ROST", "YUM", "MAR",
    ],
    "Industrials": [
        "CAT", "HON", "UNP", "BA", "GE", "RTX", "LMT", "UPS", "DE", "MMM",
        "ADP", "CSX", "NSC", "EMR", "ETN", "ITW", "GD", "NOC", "WM", "FDX",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG",
        "WEC", "ES", "AEE", "DTE", "PPL",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM", "PPG",
        "IFF", "ALB",
    ],
    "RealEstate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR",
    ],
    "Communication": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        "EA", "OMC",
    ],
}

# Names excluded because Yahoo no longer serves them. Each was acquired during
# the sample, and each exclusion is itself an instance of the survivorship bias
# documented above -- these are exactly the firms whose disappearance the
# universe fails to capture:
#
#   HES  Hess, acquired by Chevron (2024)
#   IPG  Interpublic, merged into Omnicom
#   MMC  Marsh & McLennan, no longer resolvable on Yahoo
#
# A point-in-time database would retain their price history up to the delisting
# date and let the strategy hold them until the deal closed. The direction of
# the resulting bias is discussed in METHODOLOGY.md.
DELISTED_EXCLUSIONS = ["HES", "IPG", "MMC"]

BENCHMARK = "SPY"  # used only to measure market neutrality, never traded


def all_tickers() -> list[str]:
    """Every symbol the study needs to download, de-duplicated and sorted."""
    tickers = {BENCHMARK}
    tickers.update(SECTOR_ETFS.values())
    for names in UNIVERSE.values():
        tickers.update(names)
    return sorted(tickers)


def candidate_pairs() -> list[tuple[str, str, str]]:
    """
    All within-sector ordered pairs as (sector, ticker_a, ticker_b).

    Each unordered pair appears once. The Engle-Granger test is not symmetric
    in its two arguments -- regressing A on B gives a different residual series
    than B on A -- so the direction is fixed here and handled explicitly in
    src/cointegration.py rather than left to chance.
    """
    pairs = []
    for sector, names in UNIVERSE.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((sector, names[i], names[j]))
    return pairs
