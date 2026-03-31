"""
signal_pipeline.py — Post-processing pipeline for gravity model signals.

Steps:
  1. Volatility scaling      — raw_signal / rolling_vol
  2. Cross-sectional ranking — percentile ranks per date (0 to 1)
  3. Market regime filter    — zero out signals on bearish days (MA50 < MA200)
  4. Momentum combination    — blend gravity rank signal with momentum rank
  5. Portfolio construction  — long top 20%, short bottom 20%
  6. Evaluation              — IC, ICIR, Sharpe ratio
  7. Visualisations          — rolling IC, cumulative returns, IC histogram
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Volatility Scaling
# ---------------------------------------------------------------------------

def volatility_scale(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 20,
    min_periods: int = 5,
    floor: float = 1e-8,
) -> pd.DataFrame:
    """
    Scale each asset's signal by its rolling return volatility.

    scaled_signal[t, i] = signal[t, i] / rolling_std(returns[t-window:t, i])

    Dividing by vol normalises signal magnitude across high- and low-vol regimes,
    so the same signal strength means the same expected Sharpe regardless of the
    asset's recent turbulence.

    Parameters
    ----------
    signals  : DataFrame (date x asset) of raw signals.
    returns  : DataFrame (date x asset) of log returns — used to compute vol.
    window   : Rolling window for std computation (default 20 = ~1 trading month).
    min_periods: Minimum observations before std is non-NaN (default 5).
    floor    : Minimum vol to avoid dividing by near-zero (default 1e-8).

    Returns
    -------
    scaled : DataFrame same shape as signals, NaN where vol is unavailable.
    """
    # Rolling std of returns — shift(1) so today's return is not in today's vol
    rolling_vol = (
        returns.shift(1)
               .rolling(window, min_periods=min_periods)
               .std()
               .reindex_like(signals)
    )
    # Floor vol to avoid extreme magnification
    rolling_vol = rolling_vol.clip(lower=floor)
    scaled = signals / rolling_vol
    return scaled


# ---------------------------------------------------------------------------
# Step 2: Cross-Sectional Ranking
# ---------------------------------------------------------------------------

def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each row (date) to percentile ranks across assets.

    rank_signal[t, i] = rank of signal[t, i] among all assets on day t,
                        normalised to [0, 1].

    Ranking neutralises outliers and turns any signal into a comparable
    ordinal score regardless of its original units or scale.

    Parameters
    ----------
    df : DataFrame (date x asset).

    Returns
    -------
    ranked : DataFrame same shape, values in [0, 1], NaN preserved as NaN.
    """
    # pct=True gives rank / count, i.e. a value in (0, 1]
    ranked = df.rank(axis=1, method="average", pct=True, na_option="keep")
    return ranked


# ---------------------------------------------------------------------------
# Step 3: Market Regime Filter
# ---------------------------------------------------------------------------

def regime_filter(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    ma_short: int = 50,
    ma_long: int = 200,
    benchmark_ticker: Optional[str] = None,
) -> pd.DataFrame:
    """
    Zero out signals on days when the market is in a bearish regime.

    Regime is bullish  when MA(ma_short) > MA(ma_long) — trend is up.
    Regime is bearish  when MA(ma_short) <= MA(ma_long) — trend is down.

    On bearish days, all signals are set to 0 (flat portfolio).

    Parameters
    ----------
    signal           : DataFrame (date x asset) of signals to filter.
    prices           : DataFrame (date x asset) of close prices.
    ma_short         : Short moving-average window (default 50).
    ma_long          : Long  moving-average window (default 200).
    benchmark_ticker : Single ticker used to define the regime.  When None,
                       the equal-weighted average of all columns is used.

    Returns
    -------
    filtered : DataFrame same shape as signal.  Bearish-regime rows are 0.
    """
    # Build the benchmark price series
    if benchmark_ticker is not None and benchmark_ticker in prices.columns:
        bench = prices[benchmark_ticker]
    else:
        bench = prices.mean(axis=1)

    # Compute moving averages (use min_periods = window so early rows are NaN)
    ma_s = bench.rolling(ma_short, min_periods=ma_short).mean()
    ma_l = bench.rolling(ma_long,  min_periods=ma_long ).mean()

    # Bullish flag: True when short MA is above long MA; NaN where either MA
    # is undefined — treat as bearish (no signal) to be conservative.
    bullish = (ma_s > ma_l).reindex(signal.index)
    bullish = bullish.fillna(False)

    # Broadcast: multiply each row by 1 (bullish) or 0 (bearish)
    filtered = signal.mul(bullish.astype(float), axis=0)
    return filtered


# ---------------------------------------------------------------------------
# Step 4: Momentum Combination
# ---------------------------------------------------------------------------

def combine_with_momentum(
    rank_signal: pd.DataFrame,
    returns: pd.DataFrame,
    momentum_weight: float = 0.3,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Blend gravity rank signal with a cross-sectional momentum rank signal.

    final_signal = (1 - momentum_weight) * rank_signal
                 + momentum_weight       * momentum_rank

    Combining two independently-positive signals (gravity and momentum) can
    raise IC and reduce drawdowns via diversification.

    Parameters
    ----------
    rank_signal      : DataFrame (date x asset), values in [0, 1] from step 2.
    returns          : DataFrame (date x asset) of log returns.
    momentum_weight  : Weight on the momentum component (default 0.3).
    lookback         : Cumulative-return window for momentum (default 60 days).

    Returns
    -------
    combined : DataFrame same shape as rank_signal, values approximately in
               [0, 1].
    """
    if not 0.0 <= momentum_weight <= 1.0:
        raise ValueError("momentum_weight must be in [0, 1]")

    # Cumulative log-return over 'lookback' days, shifted 1 day to avoid
    # look-ahead (today's price is known after today's close).
    mom_raw = returns.shift(1).rolling(lookback, min_periods=lookback // 2).sum()

    # Cross-sectional percentile rank of momentum
    mom_rank = cross_sectional_rank(mom_raw).reindex_like(rank_signal)

    gravity_weight = 1.0 - momentum_weight
    combined = gravity_weight * rank_signal + momentum_weight * mom_rank
    return combined


# ---------------------------------------------------------------------------
# Step 5: Portfolio Construction
# ---------------------------------------------------------------------------

def construct_portfolio(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    top_pct: float = 0.2,
    cost_bps: float = 0.0,
) -> pd.Series:
    """
    Construct a daily long-short portfolio from signal ranks.

    - Long  the top    `top_pct` fraction of assets by signal.
    - Short the bottom `top_pct` fraction.
    - Each leg is equally weighted; portfolio sums to zero (dollar-neutral).
    - Returns are shifted by 1 day so the signal on day t is traded on day t+1.

    Parameters
    ----------
    signal   : DataFrame (date x asset) of final signals.
    returns  : DataFrame (date x asset) of log returns.
    top_pct  : Fraction of assets in each leg (default 0.2 = top/bottom 20%).
    cost_bps : One-way transaction cost in basis points (default 0 = no cost).
                Applied each time the portfolio turns over (daily rebalancing).

    Returns
    -------
    port_returns : Series (date,) of daily portfolio log-return.
    """
    if not 0 < top_pct < 0.5:
        raise ValueError("top_pct must be in (0, 0.5)")

    # Align returns to signal's date/asset grid
    rets = returns.reindex_like(signal)

    port_daily = []
    dates = []

    for dt in signal.index:
        sig_row = signal.loc[dt].dropna()
        if len(sig_row) < 4:
            continue

        n = len(sig_row)
        k = max(1, int(np.floor(n * top_pct)))

        sorted_sig = sig_row.sort_values()
        short_tickers = sorted_sig.iloc[:k].index        # bottom k
        long_tickers  = sorted_sig.iloc[-k:].index       # top k

        # Next-day returns (shift already applied in the loop via tomorrow's row)
        # We look up the *next* date's return to avoid look-ahead.
        idx_pos = signal.index.get_loc(dt)
        if idx_pos + 1 >= len(signal.index):
            continue
        next_dt = signal.index[idx_pos + 1]

        if next_dt not in rets.index:
            continue

        r_next = rets.loc[next_dt]

        long_ret  = r_next[long_tickers].mean()
        short_ret = r_next[short_tickers].mean()

        # Long-short return (both legs equally weighted, dollar-neutral)
        port_ret = 0.5 * long_ret - 0.5 * short_ret

        # Subtract transaction cost (in log-return units, bps / 10000)
        port_ret -= cost_bps / 10_000

        port_daily.append(port_ret)
        dates.append(next_dt)

    return pd.Series(port_daily, index=dates, name="portfolio_return")


# ---------------------------------------------------------------------------
# Step 6: Evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    method: str = "spearman",
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute IC, ICIR, and annualised Sharpe ratio for the pipeline.

    Parameters
    ----------
    signal            : DataFrame (date x asset) of final signals.
    fwd_returns       : DataFrame (date x asset) of forward returns.
    portfolio_returns : Series of daily portfolio returns from construct_portfolio.
    method            : 'spearman' or 'pearson' for IC (default 'spearman').
    trading_days_per_year : Used to annualise Sharpe (default 252).

    Returns
    -------
    metrics : dict with keys 'ic_series', 'mean_ic', 'icir', 'sharpe',
              'annual_return', 'annual_vol', 'cumulative_return'.
    """
    # --- IC ---
    s, r = signal.align(fwd_returns, join="inner", axis=0)
    r = r.reindex_like(s)

    ic_values = []
    ic_dates  = []
    for dt in s.index:
        x = s.loc[dt]
        y = r.loc[dt]
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        if len(x) < 3:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            val = x.rank().corr(y.rank()) if method == "spearman" else x.corr(y)
        if pd.notna(val):
            ic_values.append(val)
            ic_dates.append(dt)

    ic_series = pd.Series(ic_values, index=ic_dates, name=f"IC_{method}")
    mean_ic   = ic_series.mean()
    icir      = mean_ic / (ic_series.std() + 1e-12)

    # --- Sharpe ---
    pr = portfolio_returns.dropna()
    annual_return = pr.mean() * trading_days_per_year
    annual_vol    = pr.std()  * np.sqrt(trading_days_per_year)
    sharpe        = annual_return / (annual_vol + 1e-12)
    cum_ret       = pr.cumsum().iloc[-1] if len(pr) else np.nan

    return {
        "ic_series"         : ic_series,
        "mean_ic"           : mean_ic,
        "icir"              : icir,
        "sharpe"            : sharpe,
        "annual_return"     : annual_return,
        "annual_vol"        : annual_vol,
        "cumulative_return" : cum_ret,
    }


# ---------------------------------------------------------------------------
# Step 7: Visualisations
# ---------------------------------------------------------------------------

def plot_pipeline_results(
    metrics: dict,
    portfolio_returns: pd.Series,
    title_prefix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    """
    Produce three diagnostic plots:
      1. 60-day rolling mean IC over time.
      2. Cumulative portfolio returns.
      3. IC distribution histogram.

    Parameters
    ----------
    metrics           : dict returned by evaluate_pipeline.
    portfolio_returns : Series of daily portfolio returns.
    title_prefix      : Optional string prepended to each plot title.
    save_dir          : If provided, saves PNGs to this directory.
    """
    ic_series = metrics["ic_series"]
    mean_ic   = metrics["mean_ic"]
    sharpe    = metrics["sharpe"]

    prefix = f"{title_prefix} — " if title_prefix else ""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{prefix}Pipeline Diagnostics", fontsize=13)

    # --- Plot 1: Rolling IC ---
    ax1 = axes[0]
    rolling_ic = ic_series.rolling(60, min_periods=20).mean()
    ax1.plot(ic_series.index, ic_series.values, alpha=0.25,
             color="steelblue", linewidth=0.8, label="Daily IC")
    ax1.plot(rolling_ic.index, rolling_ic.values, color="steelblue",
             linewidth=2, label="60d rolling mean IC")
    ax1.axhline(0,       color="black", linestyle="--", linewidth=0.8)
    ax1.axhline(mean_ic, color="red",   linestyle="--", linewidth=1,
                label=f"Mean IC = {mean_ic:.4f}")
    ax1.set_title("Rolling IC (60-day)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("IC")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Cumulative Returns ---
    ax2 = axes[1]
    pr = portfolio_returns.dropna()
    cum = pr.cumsum()
    ax2.plot(cum.index, cum.values, color="seagreen", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title(f"Cumulative Portfolio Returns\n(Sharpe = {sharpe:.2f})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative log-return")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: IC Distribution ---
    ax3 = axes[2]
    ax3.hist(ic_series.values, bins=40, color="salmon", edgecolor="white",
             alpha=0.85)
    ax3.axvline(0,       color="black", linestyle="--", linewidth=0.8)
    ax3.axvline(mean_ic, color="red",   linestyle="--", linewidth=1.2,
                label=f"Mean = {mean_ic:.4f}")
    ax3.set_title("IC Distribution")
    ax3.set_xlabel("IC Value")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "pipeline_diagnostics.png",
                    bbox_inches="tight", dpi=150)

    return fig, axes
