
from __future__ import annotations
import logging
import warnings
from typing import Optional, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _zscore_by_row(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    What this does:
        Row-wise z-score normalisation: (x - row_mean) / row_std.

    How it works:
        - Computes mean and std per date (row).
        - Normalises each element by its row stats.
        - Adds a small epsilon to avoid divide-by-zero.

    Why we need it:
        Turns day-specific signals into comparable scales across time, improving stability.
    """
    # compute row means
    mu = df.mean(axis=1)
    # compute row stds
    sigma = df.std(axis=1)
    # broadcast-normalise; eps floors the denominator so zero-std rows yield 0 rather than NaN
    return (df.sub(mu, axis=0)).div(sigma + eps, axis=0)


# ---------------------------------------------------------------------
# Mass, Distance, Momentum
# ---------------------------------------------------------------------

def compute_masses(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    market_caps: Optional[pd.DataFrame] = None,
    method: str = "dollar_volume",
    roll_window: int = 20,
    min_mass: float = 1e-8,
) -> pd.DataFrame:
    """
    What this does:
        Estimates a "mass" for each asset and day. Options:
        - 'market_cap'   : directly use provided market caps.
        - 'dollar_volume': rolling mean of price * volume (liquidity proxy).

    How it works:
        - If market caps are given and method='market_cap', use those.
        - Else compute rolling mean of (prices * volume) over roll_window.
        - Floors masses by min_mass to avoid zeros.

    Why we need it:
        Heavier assets should pull more. Market cap or liquidity are sensible, explainable proxies.
    """
    # defensive coercion
    if method not in {"market_cap", "dollar_volume", "equal"}:
        raise ValueError("method must be 'market_cap', 'dollar_volume', or 'equal'")

    if method == "market_cap":
        # require market_caps to be provided
        if market_caps is None:
            raise ValueError("market_caps must be provided when method='market_cap'")
        masses = market_caps.copy()  # copy to avoid mutating caller data
    elif method == "dollar_volume":
        # dollar volume requires volume
        if volume is None:
            raise ValueError("volume must be provided when method='dollar_volume'")
        # compute instantaneous dollar volume
        dollar_vol = prices * volume
        # average over a rolling window to smooth noise
        masses = dollar_vol.rolling(roll_window, min_periods=max(2, roll_window // 3)).mean()
    else:
        # equal mass: every asset has weight 1.0 at every date
        masses = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

    # replace non-positive/NaN masses with a tiny floor to keep formulas stable
    masses = masses.fillna(min_mass).clip(lower=min_mass)
    return masses


def correlation_distance(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: int = 30,
    eps: float = 1e-6,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    What this does:
        Builds a daily "distance" matrix between assets using correlation distance:
            d_ij = sqrt(2 * (1 - corr_ij))

    How it works:
        - For each date, compute rolling correlation over the past 'window' days.
        - Convert correlation to a distance metric (>=0).
        - Returns a dict: {date -> DataFrame (assets x assets)}.

    Why we need it:
        Assets that co-move (high corr) are 'close', so their pull is stronger in the analogy.
    """
    # Ensure columns align across time
    tickers = returns.columns.tolist()
    # Container for per-date distance matrices
    out: Dict[pd.Timestamp, pd.DataFrame] = {}

    # Compute ALL pairwise rolling correlations in one vectorised call using
    # pandas' C backend. This avoids the O(n * p^2) Python loop that called
    # window_slice.corr() on every single date.
    rolling_corr = returns.rolling(window=window, min_periods=min_periods).corr()

    # rolling_corr has a 2-level MultiIndex: (date, ticker). Group by date and
    # build one distance matrix per date — only a cheap Python groupby, not a
    # full correlation computation.
    for dt, block in rolling_corr.groupby(level=0):
        corr = block.droplevel(0).reindex(index=tickers, columns=tickers)
        if corr.isna().all().all():
            continue  # min_periods not met for this date
        # transform correlation to distance
        dist = np.sqrt(2.0 * (1.0 - corr.clip(-1 + eps, 1 - eps)))
        # small numerical floor
        out[dt] = dist.fillna(0.0).clip(lower=0.0)

    return out


def compute_momentum(
    prices: pd.DataFrame,
    lookback: int = 20,
    kind: str = "log",  # 'log' or 'simple'
) -> pd.Series:
    """
    What this does:
        Produces a 1D momentum vector (per date) used as "direction" in the gravity model.

    How it works:
        - Computes log or simple returns over 'lookback'.
        - At each date, outputs the cross-sectional z-scored momentum for assets available.

    Why we need it:
        Gravity gives magnitude of interaction; momentum provides a direction (pull towards
        recent winners, or repulsion if you invert the sign).
    """
    # compute return over the lookback window
    if kind not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")

    if kind == "simple":
        mom = prices.pct_change(lookback)  # (P_t / P_{t-LB}) - 1
    else:
        mom = np.log(prices).diff(lookback)  # ln(P_t) - ln(P_{t-LB})

    # z-score momentum cross-sectionally each day (row-wise)
    mom_z = _zscore_by_row(mom)
    # For the gravity direction we often want a single vector per date; however, we use
    # the per-asset values as direction multipliers in the force aggregation step.
    # We return the full DataFrame (assets per date) so it can align with forces.
    return mom_z


# ---------------------------------------------------------------------
# Forces and Signals
# ---------------------------------------------------------------------

def gravity_force_matrix(
    masses: pd.Series | pd.DataFrame,
    distances: pd.DataFrame,
    G: float = 1.0,
    eps: float = 1e-6,
    cap: Optional[float] = None,
) -> pd.DataFrame:
    """
    What this does:
        Computes pairwise gravity force magnitudes:
            F_ij = G * (m_i * m_j) / (d_ij^2 + eps), with F_ii = 0.

    How it works:
        - Broadcasts masses to outer product m_i * m_j.
        - Divides by squared distance (plus epsilon for stability).
        - Zeroes the diagonal; optionally caps extremes.

    Why we need it:
        Encodes the "pull" between assets: heavier + closer = stronger interaction.
    """
    # if masses is a DataFrame, squeeze single-row or single-column to Series
    if isinstance(masses, pd.DataFrame):
        if masses.shape[0] == 1:
            masses = masses.iloc[0]
        elif masses.shape[1] == 1:
            masses = masses.iloc[:, 0]

    # ensure alignment of indices/columns
    m = masses.reindex(distances.index)
    m = m.fillna(np.nanmedian(m.values))
    # compute outer product of masses
    m_outer = np.outer(m.values, m.values)
    # squared distances with small epsilon to avoid infinities
    denom = np.square(distances.values) + eps
    # gravity formula element-wise
    F = G * (m_outer / denom)
    # zero the diagonal (no self-force)
    np.fill_diagonal(F, 0.0)

    # cap forces if requested to reduce sensitivity to very small distances
    if cap is not None:
        F = np.clip(F, 0.0, cap)

    # wrap back into a DataFrame with proper labels
    F_df = pd.DataFrame(F, index=distances.index, columns=distances.columns)
    return F_df


def net_force_signal(
    F: pd.DataFrame,
    momentum: pd.Series | pd.DataFrame,
    normalise: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    What this does:
        Aggregates pairwise forces into a per-asset "net force" signal, tilted by momentum.

    How it works:
        - For each i, compute sum_j F_ij * momentum_j  (direction and magnitude from momentum).
        - Optionally row-normalise cross-sectionally to get comparable signals each day.

    Why we need it:
        Converts the force matrix into actionable signals (rank/tilt/weights).
    """
    # Ensure momentum is in DataFrame form with same index/columns as F
    if isinstance(momentum, pd.Series):
        # Broadcast momentum across rows so each row uses the same vector
        mom_vec = momentum.reindex(F.columns).fillna(0.0).values
        # Multiply each column j by the actual momentum value (not just its sign),
        # so stronger-momentum assets exert proportionally more directional pull.
        directed = F.values * mom_vec[None, :]
        # Sum across columns to get net force for each row i
        net = directed.sum(axis=1)
        # Wrap as Series
        s = pd.Series(net, index=F.index, name="net_force")
        return s if not normalise else (s - s.mean()) / (s.std() + 1e-12)
    else:
        # DataFrame momentum: align and use date-wise direction
        momentum = momentum.reindex(columns=F.columns)
        # Use actual momentum values as weights (not just sign)
        directed = F.values * momentum.values
        net = directed.sum(axis=1)
        s = pd.Series(net, index=F.index, name="net_force")
        return s if not normalise else (s - s.mean()) / (s.std() + 1e-12)


# ---------------------------------------------------------------------
# Orchestrator (daily panel)
# ---------------------------------------------------------------------

def gravity_signals_pipeline(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    market_caps: Optional[pd.DataFrame] = None,
    *,
    mass_method: str = "dollar_volume",
    mass_window: int = 20,
    dist_window: int = 60,
    dist_min_periods: int = 30,
    momentum_lookback: int = 20,
    momentum_kind: str = "log",
    reversal_lookback: Optional[int] = None,
    reversal_weight: float = 0.3,
    G: float = 1.0,
    eps: float = 1e-6,
    force_cap: Optional[float] = None,
    zscore_signals: bool = True,
) -> pd.DataFrame:
    """
    What this does:
        Full, time-series pipeline:
          1) Compute masses (dollar volume or market cap).
          2) Compute rolling correlation distances.
          3) For each date with a distance matrix, compute force matrix and net-force signal
             tilted by a combined direction = long-term momentum - reversal_weight * short-term momentum.
          4) Optionally z-score signals cross-sectionally per day.

        Returns a DataFrame of daily signals (index=date, columns=assets).

    How it works:
        - Uses compute_masses, correlation_distance, gravity_force_matrix, compute_momentum,
          and net_force_signal in a loop over dates where distance is defined.
        - If reversal_lookback is set, a short-term reversal component is subtracted from
          the momentum direction, combining two documented anomalies in one signal.

    Why we need it:
        Provides a single, clean entry point to generate signals you can backtest or chart.
    """
    # Compute daily returns if not provided (log returns, safe default)
    if returns is None:
        returns = np.log(prices).diff()
        
    chosen_method = mass_method
    if mass_method == "dollar_volume" and volume is None:
        warnings.warn(
            "mass_method='dollar_volume' requires 'volume', but none was provided. "
            "Falling back to mass_method='equal'."
        )
        chosen_method = "equal"

    if mass_method == "market_cap" and market_caps is None:
        warnings.warn(
            "mass_method='market_cap' requires 'market_caps', but none provided. "
            "Falling back to mass_method='equal'."
        )
        chosen_method = "equal"

    # Masses: either from market_caps or rolling dollar volume
    masses = compute_masses(
        prices=prices,
        volume=volume,
        market_caps=market_caps,
        method=chosen_method,
        roll_window=mass_window,
    )
    
    masses = masses.ffill().bfill() # to ensure masses have minimal gaps
    
    # Long-term momentum used as primary direction signal (z-scored cross-sectionally)
    mom = compute_momentum(
        prices=prices,
        lookback=momentum_lookback,
        kind=momentum_kind,
    )

    # Optional: subtract a short-term reversal component.
    # Stocks that surged recently tend to mean-revert; reducing their directional
    # pull improves signal quality at medium-to-long horizons.
    if reversal_lookback is not None:
        rev = compute_momentum(
            prices=prices,
            lookback=reversal_lookback,
            kind=momentum_kind,
        )
        mom = _zscore_by_row(mom - reversal_weight * rev)

    # Rolling correlation distance matrices per date
    dist_mats = correlation_distance(
        returns=returns,
        window=dist_window,
        min_periods=dist_min_periods,
        eps=eps,
    )

    # Prepare an output container for signals per date
    signals: Dict[pd.Timestamp, pd.Series] = {}

    # Iterate only over dates for which we have a valid distance matrix
    for dt, dist in dist_mats.items():
        # Extract masses for this date directly — masses is fully indexed to
        # prices.index and has no NaN after ffill().bfill(), so loc[dt] is safe.
        m_t = masses.loc[dt].reindex(dist.index, fill_value=0.0)
        # Compute pairwise forces at this date
        F_t = gravity_force_matrix(
            masses=m_t,
            distances=dist,
            G=G,
            eps=eps,
            cap=force_cap,
        )
        # Use momentum direction at 'dt' (per-asset)
        mom_t = mom.loc[dt].reindex(F_t.columns).fillna(0.0)
        # Net force per asset at 'dt'
        s_t = net_force_signal(F_t, momentum=mom_t, normalise=False)
        # Store
        signals[dt] = s_t

    # Concatenate daily Series into a DataFrame (index=date, columns=assets)
    signal_df = pd.DataFrame(signals).T.reindex(index=prices.index)  # align to full date index

    # Optionally z-score per day to stabilise scale
    if zscore_signals:
        signal_df = _zscore_by_row(signal_df)

    return signal_df


# ---------------------------------------------------------------------
# Simple evaluation helper
# ---------------------------------------------------------------------

def information_coefficient(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    What this does:
        Computes a daily cross-sectional IC between your signals and forward returns.

    How it works:
        - Aligns signals and fwd_returns by date and assets.
        - For each date, computes Spearman (rank) or Pearson correlation across assets.
        - Returns a time series of IC values.

    Why we need it:
        Quick, interview-friendly performance diagnostic for cross-sectional signals.
    """
    # align indices/columns
    s = signals.align(fwd_returns, join="inner", axis=0)[0].align(fwd_returns, join="inner", axis=1)[0]
    r = fwd_returns.reindex_like(s)

    # choose correlation function
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

    ic_values = []
    dates = []
    for dt in s.index:
        # get same-date vectors and drop any paired NaNs
        x = s.loc[dt]
        y = r.loc[dt]
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        if len(x) < 3:
            continue
        if method == "spearman":
            # rank transform then Pearson on ranks
            x_rank = x.rank()
            y_rank = y.rank()
            val = x_rank.corr(y_rank)
        else:
            val = x.corr(y)
        if pd.notna(val):
            ic_values.append(val)
            dates.append(dt)

    return pd.Series(ic_values, index=dates, name=f"IC_{method}")
