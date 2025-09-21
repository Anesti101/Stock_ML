from __future__ import annotations  # allows forward type refs in type hints
from typing import Iterable, Tuple, Optional, Dict
import logging                     # structured progress + debug info
from pathlib import Path           # robust filesystem paths
import numpy as np                 # fast numeric operations
import pandas as pd                # tabular data operations

# yfinance is optional at import-time to avoid hard dependency outside fetch
try:
    import yfinance as yf          # Yahoo Finance API wrapper
except Exception:                   # pragma: no cover (optional dependency)
    yf = None

# ---------------------------------------------------------------------------
# Logging setup (module-level): keep it simple; users can override in notebooks
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------


def fetch_prices_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    What this does:
        Downloads historical prices for the given tickers and time window from Yahoo Finance
        using yfinance, returning a *wide* DataFrame of Adjusted Close prices indexed by date.

    How it works:
        - Validates that yfinance is installed.
        - Uses yf.download to fetch an (Open, High, Low, Close, Adj Close, Volume) Panel.
        - Selects the Adjusted Close (or Close if not auto-adjusting), pivots to wide format.
        - Sorts the index, drops duplicate dates, enforces DatetimeIndex.
        - Returns a clean price frame with columns per ticker.

    Why we need it:
        This provides a reproducible, scriptable data source—no manual CSVs—showing you can
        acquire real-world data programmatically.
    """
    # Ensure tickers is a list (some APIs need list-like)
    tickers = list(tickers)  # convert in case a generator/tuple was passed
    
    # Defensive check: yfinance must be available to fetch remote data
    if yf is None:  # avoid import-time hard failure
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    
    # Informative log so users know what's being fetched
    logger.info("Fetching prices for %d tickers from %s to %s at %s interval",
                len(tickers), start, end, interval)
    
    # Call Yahoo Finance bulk downloader; returns MultiIndex columns by field→ticker
    df = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=auto_adjust, progress=progress, group_by="column"
    )
    
    # If nothing returned, raise helpful error
    if df is None or df.empty:
        raise ValueError("No data returned. Check tickers, dates, or interval.")
    
    # Pick the right field to use: Adjusted Close if auto_adjust else Close
    field = "Adj Close" if auto_adjust else "Close"
    
    # When multiple tickers, yfinance returns a column level → select field level
    if isinstance(df.columns, pd.MultiIndex):
        df = df[field]  # select the desired price field across tickers
    else:
        # Single-ticker case: we rename the single column to the ticker for consistency
        only_ticker = tickers[0]
        df = df.rename(columns={field: only_ticker})[[only_ticker]]
        
    # Enforce DateTimeIndex, sorted, unique
    df.index = pd.to_datetime(df.index)         # ensure datetime index
    df = df.sort_index()                        # chronological order
    df = df[~df.index.duplicated(keep="first")] # drop duplicate timestamps
    
    # Final shape log for transparency 
    logger.info("Fetched price frame shape: %s", df.shape)
    
    # Return wide DataFrame with columns named by ticker eg. 'AAPL', 'MSFT' 
    return df 


def validate_price_frame(df: pd.DataFrame) -> None:
    """
    What this does:
        Validates structural assumptions about the price DataFrame (index type/order,
        duplicates, numeric dtypes, and non-empty).

    How it works:
        - Checks DataFrame non-emptiness.
        - Ensures DatetimeIndex, sorted ascending, no duplicates.
        - Ensures all columns are numeric.

    Why we need it:
        Early validation prevents subtle bugs later (mis-sorted returns, hidden duplicates).
        This also shows interviewers you practice defensive data engineering.
    """
    # DataFrame should not be empty; early stop if it is
    if df.empty:
        raise ValueError("Price DataFrame is empty.")
    # Index must be datetime for time series ops
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas.DatetimeIndex.")
    # Index should be strictly increasing (no back-in-time rows)
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending (monotonic increasing).")
    # No duplicate timestamps allowed
    if df.index.has_duplicates:
        raise ValueError("Index contains duplicate timestamps.")
    # Columns must be numeric (floats/ints) for return calculations
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise TypeError("All columns must be numeric types.")


def align_and_fill(
    df: pd.DataFrame,
    freq: str = "B",
    ffill_limit: int = 5,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """
    What this does:
        Aligns all tickers to a common calendar (default business days), forward-fills
        short gaps, and drops columns with poor data coverage.

    How it works:
        - Reindexes to a complete calendar (freq).
        - Forward-fills up to ffill_limit to fill small market/holiday gaps.
        - Computes non-NA coverage per column and drops those below min_coverage.

    Why we need it:
        Comparable time series across assets requires a unified timeline. Brief gaps are
        common in real data; controlled filling stabilizes downstream stats and models.
    """
    # Build full date range from first to last timestamp at the desired frequency
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    # Reindex to full calendar; introduces NaNs where data is missing
    aligned = df.reindex(full_index)
    # Forward-fill with a cap so long gaps don't propagate too far
    aligned = aligned.ffill(limit=ffill_limit).bfill(limit=1) # backfill 1 to handle leading NaNs
    # Calculate coverage ratio per column after filling
    coverage = aligned.notna().mean(axis=0)
    # Keep only columns meeting the minimum coverage requirement
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    # Subset to the retained columns
    trimmed = aligned[keep_cols]
    # Log how many columns were dropped (if any)
    dropped = set(df.columns) - set(keep_cols)
    if dropped:
        logger.info("Dropped %d columns for low coverage: %s", len(dropped), sorted(dropped))
    # Return the aligned, gap-filled, column-filtered frame
    return trimmed


def winsorize_outliers(
    df: pd.DataFrame,
    z_thresh: float = 6.0,
) -> pd.DataFrame:
    """
    What this does:
        Caps extreme values (per column) based on a z-score threshold to reduce the impact
        of outliers without removing rows.

    How it works:
        - Computes column-wise mean and std.
        - Derives lower/upper caps at mean ± z_thresh*std.
        - Clips values outside those caps.

    Why we need it:
        Spurious ticks/splits can create extreme values that distort stats and charts.
        Winsorization is a simple, explainable fix that preserves dataset size.
    """
    # Compute per-column mean (μ) and std (σ)
    mu = df.mean(axis=0) # per column maan
    sigma = df.std(axis=0) # per column std
    # Compute lower and upper caps per column
    lower = mu - z_thresh * sigma
    upper = mu + z_thresh * sigma
    # Clip values to the [lower, upper] band
    capped = df.clip(lower=lower, upper=upper, axis="columns")
    # Return winsorized DataFrame
    return capped


def compute_returns(
    prices: pd.DataFrame,
    kind: str = "log",
) -> pd.DataFrame:
    """
    What this does:
        Converts price levels to returns (either 'log' or 'simple').

    How it works:
        - For 'simple': uses pct_change.
        - For 'log': uses diff of natural log prices.
        - Leaves the first row as NaN (no previous price), which callers can drop if needed.

    Why we need it:
        Returns are stationary(er) and comparable across assets/time; they’re the basis for
        risk/alpha metrics, factor models, and most real analyses.
    """
    # Ensure valid choice of return kind
    if kind not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")
    # Compute returns depending on selected method
    if kind == "simple":
        rets = prices.pct_change()             # (P_t / P_{t-1}) - 1
    else:
        rets = np.log(prices).diff()           # ln(P_t) - ln(P_{t-1})
    # Return returns DataFrame with first row NaN by construction
    return rets


def add_technical_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    windows: Tuple[int, int, int] = (5, 20, 60),
) -> pd.DataFrame:
    """
    What this does:
        Builds simple, interview-ready technical features from prices & returns:
        - Rolling means (SMA)
        - Momentum (price change over window)
        - Rolling volatility (std of returns)

    How it works:
        - For each window in (short, mid, long), compute SMA on prices.
        - Compute momentum as price / price.shift(window) - 1.
        - Compute rolling volatility as returns.rolling(window).std().
        - Returns a DataFrame with MultiIndex columns: (feature_name, ticker).

    Why we need it:
        Shows feature engineering skills clearly and reproducibly, and creates inputs that
        are commonly used in EDA or simple predictive baselines.
    """
    # Prepare list to collect (feature_name, DataFrame) pairs
    feats: Dict[Tuple[str, int], pd.DataFrame] = {}
    # Loop over given rolling windows
    for w in windows:
        # Simple moving average of prices
        sma = prices.rolling(window=w, min_periods=max(1, w // 2)).mean()
        feats[("sma", w)] = sma
        # Momentum: percent change over w periods
        mom = prices.pct_change(periods=w)
        feats[("momentum", w)] = mom
        # Rolling volatility on returns
        vol = returns.rolling(window=w, min_periods=max(1, w // 2)).std()
        feats[("volatility", w)] = vol
    # Concatenate features along columns with MultiIndex (feature, window, ticker)
    feat_df = pd.concat(
        {fname: frame for (fname, _w), frame in feats.items()},
        axis=1
    )
    # Reorder columns to (feature, ticker) with window embedded in feature name for clarity
    # e.g., ('sma_20', 'AAPL')
    new_cols = []
    for top in feat_df.columns.levels[0]:
        # figure out windows we computed for this feature
        # we derive them by scanning feats keys
        for (fname, w), frame in feats.items():
            if fname == top:
                # build target name like "sma_20"
                new_cols.append((f"{fname}_{w}",))
    # The above builds names but we need to rebuild columns systematically:
    # Easiest is to rebuild by iterating feats again and concatenating.
    pieces = []
    for (fname, w), frame in feats.items():
        frame = frame.copy()
        frame.columns = pd.MultiIndex.from_product([[f"{fname}_{w}"], frame.columns])
        pieces.append(frame)
    feat_df = pd.concat(pieces, axis=1).sort_index(axis=1)
    # Return MultiIndex (feature, ticker) columns
    return feat_df


def save_processed(
    df: pd.DataFrame,
    out_path: Path | str,
    save_parquet: bool = True,
    save_csv: bool = True,
) -> None:
    """
    What this does:
        Saves a processed DataFrame to disk in Parquet and/or CSV, creating folders if needed.

    How it works:
        - Ensures parent directory exists.
        - Writes df to .parquet (fast/binary) and/or .csv (human-readable).

    Why we need it:
        Reproducible outputs and easy handoff; reviewers can inspect your data or load it
        elsewhere (Power BI/Tableau/Excel) without running your entire pipeline.
    """
    # Normalize to Path object for OS-agnostic behavior
    out_path = Path(out_path)
    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save Parquet if requested (best for Python ecosystems)
    if save_parquet:
        df.to_parquet(out_path.with_suffix(".parquet"))
    # Save CSV if requested (friendly for spreadsheets/BI tools)
    if save_csv:
        df.to_csv(out_path.with_suffix(".csv"), index=True)


def prepare_price_data(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    return_kind: str = "log",
    calendar_freq: str = "B",
    ffill_limit: int = 5,
    min_coverage: float = 0.8,
    winsorize_z: Optional[float] = 6.0,
    feature_windows: Tuple[int, int, int] = (5, 20, 60),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    What this does:
        High-level pipeline to fetch → validate → align/fill → (optional) winsorize → compute returns
        → build technical features. Returns (prices, returns, features).

    How it works:
        - Uses fetch_prices_yfinance to get adjusted close prices.
        - validate_price_frame to ensure structure is sound.
        - align_and_fill to standardize calendar and handle short gaps.
        - Optional winsorization to cap extremes.
        - compute_returns for 'log' or 'simple' returns.
        - add_technical_features to build SMA/momentum/volatility.

    Why we need it:
        A single orchestrator function makes your code easy to run, test, and reuse in
        notebooks or scripts. It demonstrates end-to-end data preparation skills.
    """
    # Step 1: fetch raw (adjusted) close prices
    prices = fetch_prices_yfinance(
        tickers=tickers, start=start, end=end, interval=interval, auto_adjust=True
    )
    # Step 2: validate the fetched price frame
    validate_price_frame(prices)
    # Step 3: align to a common calendar and fill small gaps
    prices = align_and_fill(
        prices, freq=calendar_freq, ffill_limit=ffill_limit, min_coverage=min_coverage
    )
    # Optional: cap outliers to reduce extreme effects before returns/features
    if winsorize_z is not None:
        prices = winsorize_outliers(prices, z_thresh=winsorize_z)
    # Step 4: compute returns
    returns = compute_returns(prices, kind=return_kind)
    # Step 5: create technical features
    features = add_technical_features(prices=prices, returns=returns, windows=feature_windows)
    # Log final shapes for transparency
    logger.info("Final shapes → prices:%s returns:%s features:%s",
                prices.shape, returns.shape, features.shape)
    # Return prepared artifacts
    return prices, returns, features
