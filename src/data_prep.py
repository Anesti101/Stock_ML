<<<<<<< HEAD
"""Data preparation and loading utilities for stock market data.

This module provides functions to fetch historical stock data from Yahoo Finance
and prepare it for analysis and machine learning tasks.
"""
from __future__ import annotations  # allows forward type refs in type hints
from typing import Iterable, Tuple, Optional, Dict, Union
import logging                     # structured progress + debug info
from pathlib import Path           # robust filesystem paths
import numpy as np                 # fast numeric operations
import pandas as pd                # tabular data operations


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import yfinance as yf          # Yahoo Finance API wrapper
except ImportError:                 # pragma: no cover (optional dependency)
    yf = None
    logger.warning("yfinance not installed. Install with: pip install yfinance")


def fetch_data_yfinance(
    tickers: Union[str, Iterable[str]],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False) -> pd.DataFrame:
    """Fetch historical market data from Yahoo Finance.
    
    Args:
        tickers: Single ticker string or iterable of ticker symbols (e.g., 'AAPL' or ['AAPL', 'GOOGL'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        interval: Data interval - valid values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        auto_adjust: Adjust all OHLC prices automatically
        progress: Show download progress bar
    
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume) indexed by date
        
    Raises:
        ImportError: If yfinance is not installed
        ValueError: If no data is returned or invalid parameters
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    
    # Convert single ticker to list for consistent handling
    if isinstance(tickers, str):
        tickers = [tickers]
    else:
        tickers = list(tickers)
    
    if not tickers:
        raise ValueError("At least one ticker symbol must be provided")
    
    logger.info(f"Fetching data for {len(tickers)} ticker(s): {tickers}")
    logger.info(f"Period: {start} to {end}, Interval: {interval}")
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=progress,
            group_by='ticker' if len(tickers) > 1 else 'column'
        )
        
        if data.empty:
            raise ValueError(f"No data returned for tickers: {tickers}. Check ticker symbols and date range.")
        
        logger.info(f"Successfully fetched {len(data)} rows of data")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise


def validate_stock_data(data: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """Validate that stock data has expected structure.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names. Defaults to standard OHLCV columns.
    
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for sufficient data
    if len(data) < 2:
        raise ValueError(f"Insufficient data: only {len(data)} rows")
    
    # Check for any data in Close column
    if data['Close'].isna().all():
        raise ValueError("All Close prices are NaN")
    
    logger.info("Stock data validation passed")
    return True


def save_stock_data(data: pd.DataFrame, filepath: Union[str, Path], format: str = 'csv') -> None:
    """Save stock data to file.
    
    Args:
        data: DataFrame to save
        filepath: Path where data should be saved
        format: File format - 'csv', 'parquet', or 'pickle'
    
    Raises:
        ValueError: If unsupported format is specified
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        data.to_csv(filepath)
    elif format == 'parquet':
        data.to_parquet(filepath)
    elif format == 'pickle':
        data.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'parquet', or 'pickle'")
    
    logger.info(f"Data saved to {filepath}")


def load_stock_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load stock data from file.
    
    Args:
        filepath: Path to data file
    
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If unsupported file format
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif suffix == '.parquet':
        data = pd.read_parquet(filepath)
    elif suffix in ['.pkl', '.pickle']:
        data = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.info(f"Data loaded from {filepath}: {len(data)} rows")
    return data
=======
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


def _configure_yfinance_cache() -> None:
    """Use a repo-local yfinance SQLite cache when the package supports it."""
    if yf is None or not hasattr(yf, "set_tz_cache_location"):
        return
    cache_dir = Path(__file__).resolve().parent.parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


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
    _configure_yfinance_cache()
    
    # Informative log so users know what's being fetched
    logger.info("Fetching prices for %d tickers from %s to %s at %s interval",
                len(tickers), start, end, interval)
    
    # Call Yahoo Finance bulk downloader; returns MultiIndex columns by field→ticker
    df = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=False, progress=progress, group_by="column" 
    )
    
    # If nothing returned, raise helpful error
    if df is None or df.empty:
        raise ValueError("No data returned. Check tickers, dates, or interval.")
    
    
    
    preferred = "Adj Close"
    fallback = "Close"

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if preferred in level0:
            df = df[preferred]
        elif fallback in level0:
            df = df[fallback]
        else:
            raise KeyError(f"Neither '{preferred}' nor '{fallback}' present in downloaded columns: {sorted(set(level0))}")
    else:
        only_ticker = tickers[0]
        if preferred in df.columns:
            df = df.rename(columns={preferred: only_ticker})[[only_ticker]]
        elif fallback in df.columns:
            df = df.rename(columns={fallback: only_ticker})[[only_ticker]]
        else:
            raise KeyError(f"Single-ticker download missing '{preferred}' and '{fallback}'. Columns: {df.columns.tolist()}")
        
    # Enforce DateTimeIndex, sorted, unique
    df.index = pd.to_datetime(df.index)         # ensure datetime index
    df = df.sort_index()                        # chronological order
    df = df[~df.index.duplicated(keep="first")] # drop duplicate timestamps
    
    # Final shape log for transparency 
    logger.info("Fetched price frame shape: %s", df.shape)
    
    # Return wide DataFrame with columns named by ticker eg. 'AAPL', 'MSFT' 
    return df 

def fetch_volumes_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    progress: bool = False,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    _configure_yfinance_cache()
    logger.info("Fetching volumes for %d tickers from %s to %s at %s interval",
                len(tickers), start, end, interval)
    df = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=False, progress=progress, group_by="column"
    )
    if df is None or df.empty:
        raise ValueError("No volume data returned. Check tickers, dates, or interval.")
    if isinstance(df.columns, pd.MultiIndex):
        if "Volume" in df.columns.get_level_values(0):
            df = df["Volume"]
        else:
            raise KeyError("'Volume' not present in downloaded columns")
    else:
        only = list(tickers)[0]
        if "Volume" in df.columns:
            df = df.rename(columns={"Volume": only})[[only]]
        else:
            raise KeyError(f"Single-ticker download missing 'Volume'. Columns: {df.columns.tolist()}")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info("Fetched volume frame shape: %s", df.shape)
    return df


def _fetch_prices_and_volumes(
    tickers: list,
    start: str,
    end: str,
    interval: str = "1d",
    progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    What this does:
        Issues a single yf.download() call and splits the result into a prices
        DataFrame (Adj Close / Close) and a volumes DataFrame (Volume).

    How it works:
        - Calls yf.download once with all tickers and fields.
        - Extracts the Adj Close (or Close) slice for prices.
        - Extracts the Volume slice for volumes.
        - Enforces DatetimeIndex, sorted, unique on both frames.

    Why we need it:
        yfinance returns all fields (OHLCV) in one request. Calling download
        twice (once for prices, once for volumes) wastes a network round-trip
        and risks getting mismatched data if Yahoo's response differs between
        calls. This helper halves the API surface.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    _configure_yfinance_cache()

    logger.info(
        "Fetching prices+volumes for %d tickers from %s to %s at %s interval (single download)",
        len(tickers), start, end, interval,
    )
    df = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=False, progress=progress, group_by="column"
    )
    if df is None or df.empty:
        raise ValueError("No data returned. Check tickers, dates, or interval.")

    # --- Extract prices (Adj Close preferred, Close fallback) ---
    preferred, fallback = "Adj Close", "Close"
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if preferred in level0:
            prices = df[preferred].copy()
        elif fallback in level0:
            prices = df[fallback].copy()
        else:
            raise KeyError(
                f"Neither '{preferred}' nor '{fallback}' present in columns: "
                f"{sorted(set(level0))}"
            )
        # --- Extract volumes ---
        if "Volume" in level0:
            volumes = df["Volume"].copy()
        else:
            raise KeyError("'Volume' not present in downloaded columns")
    else:
        only = tickers[0]
        if preferred in df.columns:
            prices = df.rename(columns={preferred: only})[[only]]
        elif fallback in df.columns:
            prices = df.rename(columns={fallback: only})[[only]]
        else:
            raise KeyError(
                f"Single-ticker download missing '{preferred}' and '{fallback}'. "
                f"Columns: {df.columns.tolist()}"
            )
        if "Volume" in df.columns:
            volumes = df[["Volume"]].rename(columns={"Volume": only})
        else:
            raise KeyError(
                f"Single-ticker download missing 'Volume'. Columns: {df.columns.tolist()}"
            )

    for frame in (prices, volumes):
        frame.index = pd.to_datetime(frame.index)
        frame.sort_index(inplace=True)
    prices = prices[~prices.index.duplicated(keep="first")]
    volumes = volumes[~volumes.index.duplicated(keep="first")]

    logger.info(
        "Single-download: prices shape %s, volumes shape %s",
        prices.shape, volumes.shape,
    )
    return prices, volumes



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
    # Columns must be numeric (floats/ints) for return calculations.
    # Pandas extension dtypes such as StringDtype can raise inside
    # np.issubdtype, so use pandas' dtype predicate for a clean error.
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
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
        Winsorisation is a simple, explainable fix that preserves dataset size.
    """
    # Compute per-column mean (μ) and std (σ)
    mu = df.mean(axis=0) # per column mean
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
    # Build MultiIndex (feature_window, ticker) columns, e.g. ('sma_20', 'AAPL')
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
) -> Tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    What this does:
        High-level pipeline to fetch → validate → align/fill → compute returns
        → (optional) winsorize returns → build technical features.
        Returns (prices, returns, features, volumes).

    How it works:
        - Uses _fetch_prices_and_volumes to get prices and volumes in one API call.
        - validate_price_frame to ensure structure is sound.
        - align_and_fill to standardize calendar and handle short gaps.
        - compute_returns for 'log' or 'simple' returns on unmodified prices.
        - Optional winsorization of returns (stationary) rather than price levels,
          avoiding artificial discontinuities from clipping non-stationary prices.
        - add_technical_features to build SMA/momentum/volatility.

    Why we need it:
        A single orchestrator function makes your code easy to run, test, and reuse in
        notebooks or scripts. It demonstrates end-to-end data preparation skills.
    """
    # Step 1: fetch raw (adjusted) close prices AND volumes in a single API call
    tickers = list(tickers)
    try:
        prices, volumes = _fetch_prices_and_volumes(
            tickers=tickers, start=start, end=end, interval=interval
        )
    except Exception as e:
        # If the combined fetch fails, fall back to the separately exposed
        # fetch helpers. Tests and notebooks can mock these helpers directly,
        # and real runs still get volume data when the second request works.
        logger.warning(
            "Combined price+volume fetch failed (%s). Retrying separate price and volume fetches.", e
        )
        prices = fetch_prices_yfinance(
            tickers=tickers, start=start, end=end, interval=interval, auto_adjust=True
        )
        try:
            volumes = fetch_volumes_yfinance(
                tickers=tickers, start=start, end=end, interval=interval
            )
        except Exception as volume_error:
            logger.warning(
                "Separate volume fetch failed (%s). Continuing with empty volumes.",
                volume_error,
            )
            volumes = pd.DataFrame(
                index=prices.index, columns=prices.columns, dtype="float64"
            )

    # Step 2: validate the fetched price frame
    validate_price_frame(prices)
    # Step 3: align to a common calendar and fill small gaps
    prices = align_and_fill(
        prices, freq=calendar_freq, ffill_limit=ffill_limit, min_coverage=min_coverage
    )
    volumes = align_and_fill(
        volumes, freq=calendar_freq, ffill_limit=ffill_limit, min_coverage=min_coverage
    )
    # Step 4: compute returns on unwinsorized prices (returns are stationary)
    returns = compute_returns(prices, kind=return_kind)
    # Optional: winsorize returns (stationary series) rather than price levels.
    # Winsorizing price levels then differencing creates artificial discontinuities
    # at clipped points; winsorizing the already-differenced returns avoids this.
    if winsorize_z is not None:
        returns = winsorize_outliers(returns, z_thresh=winsorize_z)
    # Step 5: create technical features
    features = add_technical_features(prices=prices, returns=returns, windows=feature_windows)
    # Log final shapes for transparency
    logger.info("Final shapes - prices:%s returns:%s features:%s",
                prices.shape, returns.shape, features.shape)
    # Return prepared artifacts
    return prices, returns, features, volumes
>>>>>>> 42feef065a94f074c01fe79e91a56ee1602a6361
