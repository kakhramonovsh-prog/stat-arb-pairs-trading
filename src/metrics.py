"""
Performance measurement.

Every statistic here is computed on NET returns. Reporting gross performance
for a strategy that trades a 6-day half-life spread is meaningless: the whole
question is whether the edge survives the cost of harvesting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config

TRADING_DAYS_PER_YEAR = 252


def summarise(returns: pd.Series, rf_annual: float = config.RISK_FREE_ANNUAL) -> dict:
    """Headline risk and return statistics for a daily return series."""
    r = returns.dropna()
    if len(r) < 2:
        return {}

    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = r - rf_daily

    ann_return = float(r.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if excess.std(ddof=1) > 0 else np.nan

    downside = r[r < 0]
    sortino = float(r.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    # A t-statistic on the mean daily return, which is the quantity a referee
    # will ask for. Daily strategy returns are mildly autocorrelated because
    # positions persist for days, so the naive standard error is optimistic;
    # the Newey-West version below is the one to quote.
    t_naive = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) \
        if r.std(ddof=1) > 0 else np.nan

    ols = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 21}
    )
    t_nw = float(ols.tvalues[0])

    return {
        "n_days": int(len(r)),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "hit_rate_daily": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "t_stat_naive": t_naive,
        "t_stat_newey_west": t_nw,
        "total_return": float(equity.iloc[-1] - 1.0),
    }


def market_neutrality(
    returns: pd.Series, benchmark_prices: pd.Series,
    rf_annual: float = config.RISK_FREE_ANNUAL
) -> dict:
    """
    Regress strategy returns on the market to check the neutrality claim.

    A pairs strategy is supposed to be market-neutral by construction: every
    position is long one stock and short another, so broad market moves should
    largely cancel. "Largely" is doing work in that sentence, and this
    regression is how the claim gets tested rather than asserted.

    A beta near zero with a positive, significant alpha is the result the
    strategy is meant to produce. A materially non-zero beta means the hedge
    ratios are not neutralising market exposure and part of any measured return
    is just equity risk premium in disguise.
    """
    r = returns.dropna()
    mkt = benchmark_prices.pct_change().reindex(r.index).dropna()
    r = r.reindex(mkt.index)

    if len(r) < 30:
        return {}

    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    y = (r - rf_daily).to_numpy()
    X = sm.add_constant((mkt - rf_daily).to_numpy())

    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 21})

    return {
        "alpha_daily": float(fit.params[0]),
        "alpha_annual": float(fit.params[0] * TRADING_DAYS_PER_YEAR),
        "alpha_t": float(fit.tvalues[0]),
        "beta_market": float(fit.params[1]),
        "beta_t": float(fit.tvalues[1]),
        "r_squared": float(fit.rsquared),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    """Round-trip level statistics."""
    if trades.empty:
        return {}

    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    return {
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl_bps": float(pnl.mean() * 1e4),
        "avg_win_bps": float(wins.mean() * 1e4) if len(wins) else np.nan,
        "avg_loss_bps": float(losses.mean() * 1e4) if len(losses) else np.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan,
        "avg_holding_days": float(trades["days"].mean()),
        "median_holding_days": float(trades["days"].median()),
    }


def cost_decomposition(daily: pd.DataFrame) -> dict:
    """
    How much of the gross edge the costs consume.

    Reported because it is the single most useful diagnostic for a
    short-half-life spread strategy: if gross is positive and net is negative,
    the signal is real but uneconomic, which is a different conclusion from
    "there is no signal" and points at different fixes.
    """
    return {
        "gross_ann": float(daily["gross"].mean() * TRADING_DAYS_PER_YEAR),
        "trade_costs_ann": float(daily["trade_costs"].mean() * TRADING_DAYS_PER_YEAR),
        "borrow_ann": float(daily["borrow"].mean() * TRADING_DAYS_PER_YEAR),
        "net_ann": float(daily["net"].mean() * TRADING_DAYS_PER_YEAR),
    }


def format_report(name: str, daily: pd.DataFrame, trades: pd.DataFrame,
                  benchmark: pd.Series | None = None) -> str:
    """Human-readable summary block."""
    lines = [f"{'=' * 62}", f" {name}", f"{'=' * 62}"]

    perf = summarise(daily["net"])
    costs = cost_decomposition(daily)
    tstats = trade_stats(trades)

    lines.append("\nReturns (net of costs)")
    lines.append(f"  annualised return   {perf['ann_return'] * 100:>8.2f}%")
    lines.append(f"  annualised vol      {perf['ann_vol'] * 100:>8.2f}%")
    lines.append(f"  Sharpe ratio        {perf['sharpe']:>8.2f}")
    lines.append(f"  Sortino ratio       {perf['sortino']:>8.2f}")
    lines.append(f"  max drawdown        {perf['max_drawdown'] * 100:>8.2f}%")
    lines.append(f"  total return        {perf['total_return'] * 100:>8.2f}%")
    lines.append(f"  t-stat (Newey-West) {perf['t_stat_newey_west']:>8.2f}")

    lines.append("\nCost decomposition (annualised)")
    lines.append(f"  gross               {costs['gross_ann'] * 100:>8.2f}%")
    lines.append(f"  transaction costs   {-costs['trade_costs_ann'] * 100:>8.2f}%")
    lines.append(f"  borrow              {-costs['borrow_ann'] * 100:>8.2f}%")
    lines.append(f"  net                 {costs['net_ann'] * 100:>8.2f}%")

    if tstats:
        lines.append("\nTrades")
        lines.append(f"  round trips         {tstats['n_trades']:>8d}")
        lines.append(f"  win rate            {tstats['win_rate'] * 100:>8.1f}%")
        lines.append(f"  avg P&L per trade   {tstats['avg_pnl_bps']:>8.1f} bps")
        lines.append(f"  profit factor       {tstats['profit_factor']:>8.2f}")
        lines.append(f"  avg holding days    {tstats['avg_holding_days']:>8.1f}")

    if benchmark is not None:
        neutral = market_neutrality(daily["net"], benchmark)
        if neutral:
            lines.append("\nMarket neutrality (vs SPY)")
            lines.append(f"  annualised alpha    {neutral['alpha_annual'] * 100:>8.2f}%")
            lines.append(f"  alpha t-stat        {neutral['alpha_t']:>8.2f}")
            lines.append(f"  market beta         {neutral['beta_market']:>8.3f}")
            lines.append(f"  R-squared           {neutral['r_squared']:>8.3f}")

    return "\n".join(lines)
