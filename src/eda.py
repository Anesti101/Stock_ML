from __future__ import annotations

from typing import Iterable, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn is optional but provides nicer defaults; we fall back gracefully if missing
try:
    import seaborn as sns
    _HAS_SEABORN = True
    sns.set_context("talk")
    sns.set_style("whitegrid")
except Exception:
    _HAS_SEABORN = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Core summaries
# ---------------------------------------------------------------------

def describe_frame(df: pd.DataFrame, percentiles: Iterable[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> pd.DataFrame:
    """
    What this does:
        Returns a rich descriptive statistics table for the given DataFrame.

    How it works:
        - Uses pandas describe with custom percentiles.
        - Adds count of non-nulls and number of unique values (if low-cardinality).

    Why we need it:
        Quick sanity check of central tendency, spread, and tails; useful for README tables.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    desc = df.describe(percentiles=list(percentiles)).T  # columns → rows for readability
    desc["non_null"] = df.notna().sum(axis=0)
    # Only compute nunique when it’s not huge (guard performance)
    with pd.option_context("mode.use_inf_as_na", True):
        nun = df.apply(lambda c: c.nunique(dropna=True) if c.nunique(dropna=True) <= 5000 else np.nan)
    desc["n_unique"] = nun
    return desc.sort_index()


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    What this does:
        Summarises missing values per column: count, percent, first/last NA dates for time series.

    How it works:
        - Counts NaNs per column and divides by row count for percentages.
        - For time series, finds first and last index where column is NA.

    Why we need it:
        Makes data quality issues obvious and quantifiable before modelling.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    total = len(df)
    na_count = df.isna().sum()
    na_pct = (na_count / total) * 100.0

    def _first_last_na(col: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        mask = col.isna()
        if not mask.any():
            return (None, None)
        idx = col.index[mask]
        return (idx.min(), idx.max())

    first_last = df.apply(_first_last_na, axis=0, result_type="expand")
    first_last.columns = ["first_na", "last_na"]

    out = pd.concat([na_count.rename("na_count"), na_pct.rename("na_pct"), first_last], axis=1)
    return out.sort_values("na_pct", ascending=False)


# ---------------------------------------------------------------------
# Plotting helpers (each returns the matplotlib Axes)
# ---------------------------------------------------------------------

def _ensure_axes(ax: Optional[plt.Axes]) -> plt.Axes:
    """Create a new Axes if None was provided."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    return ax


def plot_price_trends(
    prices: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    normalise: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "Price trends (normalised to 1.0 at start)",
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    What this does:
        Plots price lines for selected tickers across time. Optionally normalises to 1.0 at start.

    How it works:
        - Subsets selected columns.
        - If normalise=True, divides each series by its first non-null value.
        - Draws a multi-line time series plot.

    Why we need it:
        Quick visual of relative performance across assets; great first chart in a notebook/README.
    """
    if prices.empty:
        raise ValueError("prices is empty.")

    ax = _ensure_axes(ax)
    cols = list(tickers) if tickers is not None else list(prices.columns)

    data = prices[cols].copy()
    if normalise:
        data = data / data.ffill().bfill().iloc[0]

    if _HAS_SEABORN:
        data.plot(ax=ax, linewidth=1.6)
    else:
        ax.plot(data.index, data.values)
        ax.legend(cols)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised price" if normalise else "Price")
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_returns_hist(
    returns: pd.DataFrame,
    ticker: str,
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    What this does:
        Plots a histogram (with KDE if seaborn is available) of a single asset's returns.

    How it works:
        - Drops NaNs and infs.
        - Uses seaborn.histplot with kde=True when available; matplotlib otherwise.

    Why we need it:
        Visualises return distribution shape (skew/kurtosis), helpful for risk discussions.
    """
    if ticker not in returns.columns:
        raise KeyError(f"{ticker} not found in returns.")

    ax = _ensure_axes(ax)
    series = returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()
    t = title or f"Return distribution: {ticker}"

    if _HAS_SEABORN:
        sns.histplot(series, bins=bins, kde=True, ax=ax)
    else:
        ax.hist(series, bins=bins, alpha=0.85)

    ax.set_title(t)
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_boxplot_by_ticker(
    returns: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Return distribution by ticker (box plots)",
    rotate_labels: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    What this does:
        Box plots of return distributions across selected tickers.

    How it works:
        - Subsets tickers.
        - Uses seaborn.boxplot or pandas boxplot as fallback.

    Why we need it:
        Side-by-side comparison of dispersion/outliers across assets; great for spotting anomalies.
    """
    if returns.empty:
        raise ValueError("returns is empty.")

    ax = _ensure_axes(ax)
    cols = list(tickers) if tickers is not None else list(returns.columns)
    sub = returns[cols]

    if _HAS_SEABORN:
        sns.boxplot(data=sub, ax=ax)
    else:
        sub.plot(kind="box", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Return")

    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    method: str = "pearson",
    title: Optional[str] = None,
    annot: bool = False,
    vmin: float = -1.0,
    vmax: float = 1.0,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    What this does:
        Heatmap of the correlation matrix between columns of `frame`.

    How it works:
        - Computes frame.corr(method=method).
        - Uses seaborn.heatmap when available; matplotlib pcolormesh otherwise.

    Why we need it:
        Quickly visualises relationships/clusters; a staple EDA chart for finance feature sets.
    """
    if frame.empty:
        raise ValueError("frame is empty.")

    ax = _ensure_axes(ax)
    corr = frame.corr(method=method)
    t = title or f"{method.capitalize()} correlation heatmap"

    if _HAS_SEABORN:
        hm = sns.heatmap(
            corr, ax=ax, cmap="vlag", center=0.0, vmin=vmin, vmax=vmax,
            annot=annot, fmt=".2f", linewidths=0.3, linecolor="white", cbar=True
        )
    else:
        c = ax.pcolormesh(corr.values, vmin=vmin, vmax=vmax)
        plt.colorbar(c, ax=ax)
        ax.set_xticks(np.arange(0.5, len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(0.5, len(corr.index)), labels=corr.index)

    ax.set_title(t)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Ticker")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def rolling_stats_plot(
    series: pd.Series,
    windows: Tuple[int, int] = (20, 60),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    What this does:
        Plots a single series with rolling mean and rolling standard deviation overlays.

    How it works:
        - Computes .rolling(w).mean() and .rolling(w).std() for provided windows.
        - Draws the original series + overlays for each window.

    Why we need it:
        A quick way to visualise regime changes in level and volatility.
    """
    if series.empty:
        raise ValueError("series is empty.")

    ax = _ensure_axes(ax)
    s = series.copy()
    t = title or f"Rolling stats for {getattr(series, 'name', 'series')}"

    ax.plot(s.index, s.values, label="value", linewidth=1.5)

    for w in windows:
        m = s.rolling(w, min_periods=max(1, w // 2)).mean()
        sd = s.rolling(w, min_periods=max(1, w // 2)).std()
        ax.plot(m.index, m.values, label=f"mean_{w}", linestyle="--", linewidth=1.2)
        ax.plot(sd.index, sd.values, label=f"std_{w}", linestyle=":", linewidth=1.2)

    ax.set_title(t)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


# ---------------------------------------------------------------------
# Convenience wrappers for common tasks
# ---------------------------------------------------------------------

def quick_eda_summary(
    prices: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None,
    max_cols: int = 15,
) -> dict[str, pd.DataFrame]:
    """
    What this does:
        Produces a compact set of EDA tables you can print or export.

    How it works:
        - describe_frame on prices or returns (first non-None).
        - missingness_report on the same.
        - correlation matrix on returns if provided (limited to max_cols to keep readable).

    Why we need it:
        One call to generate the core tables for a README/report; saves time and ensures consistency.
    """
    out: dict[str, pd.DataFrame] = {}

    base = returns if returns is not None else prices
    if base is None:
        raise ValueError("Provide at least one of `prices` or `returns`.")

    # clip number of columns to keep printed output manageable
    use = base.iloc[:, :max_cols].copy()

    out["describe"] = describe_frame(use)
    out["missingness"] = missingness_report(use)

    if returns is not None:
        out["corr"] = returns.iloc[:, :max_cols].corr()

    return out
