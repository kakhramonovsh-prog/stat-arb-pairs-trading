"""
Step 5 -- build the figures used in the write-up.

Reads the CSVs written by steps 2-4, so it is cheap to rerun and does not
re-estimate anything.

Usage:
    python scripts/05_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import data, kalman as kf

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _equity(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return (1.0 + frame["net"]).cumprod()


def fig_equity_curves() -> None:
    """Cumulative net return for each specification, in-sample and hold-out."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, span, title in zip(axes, ("is", "oos"),
                               ("In-sample 2011-2021", "Hold-out 2022-2026")):
        static = _equity(config.TABLES / f"daily_{span}.csv")
        kalman = _equity(config.TABLES / f"daily_{span}_kalman.csv")
        crosssec = _equity(config.TABLES / f"daily_{span}_crosssec.csv")

        if static is not None:
            ax.plot(static.index, (static - 1) * 100, lw=1.3,
                    color="#c0392b", label="Pairs, static beta")
        if kalman is not None:
            ax.plot(kalman.index, (kalman - 1) * 100, lw=1.3,
                    color="#2c3e50", label="Pairs, Kalman beta")
        if crosssec is not None:
            ax.plot(crosssec.index, (crosssec - 1) * 100, lw=1.3,
                    color="#8e44ad", label="Cross-sectional (A&L)")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title)
        ax.set_ylabel("cumulative net return (%)")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Cointegration pairs strategy: neither specification earns its costs",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "01_equity_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_lookahead_bias() -> None:
    """The headline exhibit: what the look-ahead bug is worth."""
    path = config.TABLES / "diagnostics.csv"
    if not path.exists():
        return
    diag = pd.read_csv(path)

    keep = ["honest (reported)", "causal rolling z (60d)", "LOOK-AHEAD z-score"]
    sub = diag[diag["variant"].isin(keep)].set_index("variant").reindex(keep)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    colours = ["#c0392b", "#e67e22", "#27ae60"]
    labels = ["Frozen\nformation", "Causal\nrolling 60d", "LOOK-AHEAD\n(biased)"]

    for ax, col, title, unit in zip(
        axes,
        ["ann_return", "sharpe", "win_rate"],
        ["Annualised net return", "Sharpe ratio", "Win rate"],
        ["%", "", "%"],
    ):
        ax.bar(labels, sub[col].to_numpy(), color=colours, width=0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit)
        for i, v in enumerate(sub[col].to_numpy()):
            ax.text(i, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle(
        "One line of code: standardising the spread by its own trading-window moments",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(config.FIGURES / "02_lookahead_bias.png", bbox_inches="tight")
    plt.close(fig)


def fig_screening_funnel() -> None:
    """Cointegration evidence over time against the chance benchmark."""
    path = config.TABLES / "funnel_is.csv"
    if not path.exists():
        return
    funnel = pd.read_csv(path, parse_dates=["trading_start"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    ax.plot(funnel["trading_start"], funnel["raw_p05"], "o-", lw=1.2,
            color="#2c3e50", ms=4, label="pairs with p < 0.05")
    ax.plot(funnel["trading_start"], funnel["expected_false_positives"], "--",
            lw=1.2, color="#c0392b", label="expected by chance alone")
    ax.set_title("Cointegration is barely distinguishable from chance", fontsize=9)
    ax.set_ylabel("pairs")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(funnel["trading_start"], funnel["expected_fdp"], "o-", lw=1.2,
            color="#8e44ad", ms=4)
    ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.text(funnel["trading_start"].iloc[0], 0.52, "half the book is noise",
            fontsize=7, color="black", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Expected false discovery proportion of the selected book",
                 fontsize=9)
    ax.set_ylabel("E[FDP]")

    fig.tight_layout()
    fig.savefig(config.FIGURES / "03_screening_funnel.png", bbox_inches="tight")
    plt.close(fig)


def fig_hedge_ratio_drift() -> None:
    """Static versus filtered hedge ratio for an illustrative pair."""
    prices = data.load_prices()
    log_px = data.log_prices(prices)

    a, b = "MET", "PRU"
    if a not in log_px.columns or b not in log_px.columns:
        return

    form = log_px.loc["2016-01-01":"2016-12-31", [a, b]].dropna()
    span = log_px.loc["2016-01-01":"2018-12-31", [a, b]].dropna()

    theta0, P0, rv = kf.initial_state(form[a], form[b])
    delta, R, _ = kf.estimate_noise(form[a], form[b], theta0, P0, rv)
    out = kf.filter_pair(span[a], span[b], theta0, P0, R, delta=delta)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                             height_ratios=[1.2, 1])

    ax = axes[0]
    ax.plot(out.index, out["beta"], lw=1.3, color="#2c3e50",
            label="Kalman filtered beta")
    ax.axhline(theta0[0], color="#c0392b", ls="--", lw=1.2,
               label=f"static OLS beta = {theta0[0]:.3f}")
    ax.set_ylabel("hedge ratio")
    ax.set_title(f"{a}/{b}: the hedge ratio is not a constant", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(out.index, out["z"], lw=0.9, color="#34495e")
    for lvl, c in [(config.ENTRY_Z, "#c0392b"), (-config.ENTRY_Z, "#c0392b")]:
        ax.axhline(lvl, color=c, ls="--", lw=0.9, alpha=0.8)
    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax.set_ylabel("z-score")
    ax.set_title("Standardised innovation, causal by construction", fontsize=9)

    fig.tight_layout()
    fig.savefig(config.FIGURES / "04_hedge_ratio_drift.png", bbox_inches="tight")
    plt.close(fig)


def fig_cost_frontier() -> None:
    """
    Net annualised return as a function of the assumed per-side cost, for each
    specification. Costs enter linearly, so the frontier needs no reruns: each
    line is gross - (c / c_assumed) * costs - borrow from the saved dailies.
    The x-intercept of each line IS that specification's breakeven cost.
    """
    specs = [
        ("daily_is.csv", "Pairs, static beta", "#c0392b"),
        ("daily_is_kalman.csv", "Pairs, Kalman beta", "#2c3e50"),
        ("daily_is_crosssec.csv", "Cross-sectional (A&L)", "#8e44ad"),
    ]
    grid = np.linspace(0, 10, 101)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for fname, label, colour in specs:
        path = config.TABLES / fname
        if not path.exists():
            continue
        d = pd.read_csv(path, index_col=0, parse_dates=True)
        gross = d["gross"].mean() * 252
        costs5 = d["trade_costs"].mean() * 252     # at the assumed 5 bps/side
        borrow = d["borrow"].mean() * 252
        net = (gross - (grid / 5.0) * costs5 - borrow) * 100
        ax.plot(grid, net, lw=1.5, color=colour, label=label)
        if costs5 > 0:
            be = 5.0 * (gross - borrow) / costs5
            if 0 < be < 10:
                ax.plot([be], [0], "o", color=colour, ms=5)
                ax.annotate(f"{be:.1f} bps", (be, 0), textcoords="offset points",
                            xytext=(4, 6), fontsize=8, color=colour)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(5.0, color="gray", lw=0.8, ls=":", alpha=0.8)
    ax.text(5.05, ax.get_ylim()[0] * 0.9, "assumed in study", fontsize=7,
            color="gray", rotation=90, va="bottom")
    ax.set_xlabel("assumed cost per leg per side (bps)")
    ax.set_ylabel("net annualised return (%)")
    ax.set_title("Who can trade this? Net return vs execution cost, in-sample",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "05_cost_frontier.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_equity_curves()
    fig_lookahead_bias()
    fig_screening_funnel()
    fig_hedge_ratio_drift()
    fig_cost_frontier()
    made = sorted(p.name for p in config.FIGURES.glob("*.png"))
    print(f"wrote {len(made)} figures to {config.FIGURES}")
    for name in made:
        print(f"  {name}")


if __name__ == "__main__":
    main()
