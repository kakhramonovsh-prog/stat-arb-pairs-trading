"""
Data acquisition and cleaning.

Downloads split- and dividend-adjusted daily closes from Yahoo Finance and
caches them to parquet, so the rest of the study runs offline and reproducibly.

Why adjusted closes: the strategy holds a long and a short leg for weeks at a
time. Using raw closes would make every dividend and every split look like a
sudden break in the spread, and the cointegration test would read those breaks
as evidence against a stable relationship. Total-return prices are the correct
input for any holding-period study.
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

import config


CACHE = config.DATA_RAW / "prices.parquet"


def download_prices(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    batch_size: int = 40,
    pause: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch adjusted daily closes for `tickers` and return a wide DataFrame
    indexed by date with one column per ticker.

    Downloads are batched because Yahoo throttles large single requests and
    silently returns partial data when it does -- a failure mode that is easy
    to miss and would quietly corrupt every downstream result.
    """
    tickers = tickers or config.all_tickers()
    frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  downloading {i + 1}-{i + len(batch)} of {len(tickers)}...")
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,   # 'Close' becomes the total-return price
            progress=False,
            threads=True,
        )
        frames.append(_extract_close(raw, batch))
        if i + batch_size < len(tickers):
            time.sleep(pause)

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index().sort_index(axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices


def _extract_close(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    """
    Pull the Close block out of whatever shape yfinance returned.

    yfinance gives a MultiIndex column frame for multi-ticker requests and a
    flat frame for a single ticker. Both shapes are handled explicitly instead
    of assuming one, because the shape silently changes with the length of the
    request list.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw[["Close"]].copy()
        close.columns = batch[:1]
    return close


def clean_prices(prices: pd.DataFrame, max_ffill_days: int = 3) -> pd.DataFrame:
    """
    Forward-fill short gaps, then drop tickers that are too sparse to use.

    Short gaps are exchange holidays and single missing prints; filling them
    avoids spurious jumps in the spread. Long gaps mean the security did not
    trade -- a late IPO, a halt, a delisting -- and forward-filling those would
    fabricate a flat price series that the cointegration test would happily
    call a stable relationship. So the fill is capped at `max_ffill_days` and
    anything still missing stays NaN, to be excluded window by window.
    """
    filled = prices.ffill(limit=max_ffill_days)

    coverage = filled.notna().mean()
    too_sparse = coverage[coverage < 0.30].index.tolist()
    if too_sparse:
        print(f"  dropping {len(too_sparse)} tickers with <30% coverage: "
              f"{', '.join(sorted(too_sparse))}")
        filled = filled.drop(columns=too_sparse)

    # A non-positive price is always bad data, and log prices are undefined
    # there. Mark rather than fill.
    filled = filled.where(filled > 0)
    return filled


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Return the cached price panel, downloading it first if needed."""
    if CACHE.exists() and not refresh:
        prices = pd.read_parquet(CACHE)
        prices.index = pd.to_datetime(prices.index)
        return prices

    print("Downloading price history from Yahoo Finance...")
    prices = clean_prices(download_prices())
    prices.to_parquet(CACHE)
    print(f"Cached {prices.shape[1]} tickers x {prices.shape[0]} days -> {CACHE}")
    return prices


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Natural log of the price panel.

    The cointegration framework is applied to log prices throughout. Two
    reasons. First, log prices are much closer to an I(1) random walk than
    raw prices, which is the maintained assumption of the Engle-Granger
    procedure. Second, a spread in logs is a ratio in levels, so the hedge
    ratio is a stable proportional relationship rather than one that drifts
    mechanically as the price level changes over sixteen years.
    """
    import numpy as np

    return pd.DataFrame(
        np.log(prices.to_numpy()), index=prices.index, columns=prices.columns
    )


def window_slice(
    log_px: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, min_frac: float
) -> pd.DataFrame:
    """
    Return the sub-panel over [start, end), keeping only tickers that have at
    least `min_frac` of the window's observations, with remaining rows dropped
    pairwise by the caller.
    """
    win = log_px.loc[(log_px.index >= start) & (log_px.index < end)]
    if win.empty:
        return win
    keep = win.notna().mean() >= min_frac
    return win.loc[:, keep[keep].index]
