"""Data preparation and loading utilities for stock market data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd


try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _normalise_tickers(tickers: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("At least one ticker symbol must be provided")
    return ticker_list


def _configure_yfinance_cache() -> None:
    """Use a repo-local yfinance cache when the package supports it."""
    if yf is None or not hasattr(yf, "set_tz_cache_location"):
        return
    cache_dir = Path(__file__).resolve().parent.parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def _clean_downloaded_wide(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame[~frame.index.duplicated(keep="first")]


def _restore_inferable_index_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.freq is not None:
        return frame
    if len(frame.index) < 3:
        return frame
    inferred = pd.infer_freq(frame.index)
    if inferred is not None:
        frame = frame.copy()
        frame.index.freq = inferred
    return frame


def _downcast_integer_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.select_dtypes(include=["integer"]).columns:
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    return frame


def fetch_data_yfinance(
    tickers: Union[str, Iterable[str]],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch raw OHLCV data from Yahoo Finance.

    This legacy helper is kept for dashboard/notebook compatibility. The main
    modelling pipeline uses the optimized price+volume fetcher below.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    ticker_list = _normalise_tickers(tickers)
    _configure_yfinance_cache()

    logger.info("Fetching raw data for %d ticker(s): %s", len(ticker_list), ticker_list)
    data = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="ticker" if len(ticker_list) > 1 else "column",
    )
    if data is None or data.empty:
        raise ValueError(f"No data returned for tickers: {ticker_list}")
    return data


def validate_stock_data(
    data: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
) -> bool:
    """Validate a simple OHLCV stock-data frame."""
    if required_columns is None:
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if data.empty:
        raise ValueError("DataFrame is empty")

    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(data) < 2:
        raise ValueError(f"Insufficient data: only {len(data)} rows")

    if "Close" in data.columns and data["Close"].isna().all():
        raise ValueError("All Close prices are NaN")

    return True


def save_stock_data(
    data: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = "csv",
) -> None:
    """Save stock data to CSV, Parquet, or pickle."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        data.to_csv(filepath)
    elif format == "parquet":
        data.to_parquet(filepath)
    elif format == "pickle":
        data.to_pickle(filepath)
    else:
        raise ValueError("Unsupported format: use 'csv', 'parquet', or 'pickle'")


def load_stock_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load stock data from CSV, Parquet, or pickle."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        return _downcast_integer_columns(
            _restore_inferable_index_frequency(
                pd.read_csv(filepath, index_col=0, parse_dates=True)
            )
        )
    if suffix == ".parquet":
        return pd.read_parquet(filepath)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(filepath)
    raise ValueError(f"Unsupported file format: {suffix}")


def fetch_prices_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Download a wide adjusted-close price frame from Yahoo Finance."""
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    ticker_list = _normalise_tickers(tickers)
    _configure_yfinance_cache()

    logger.info(
        "Fetching prices for %d tickers from %s to %s at %s interval",
        len(ticker_list),
        start,
        end,
        interval,
    )
    df = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="column",
    )
    if df is None or df.empty:
        raise ValueError("No data returned. Check tickers, dates, or interval.")

    preferred = "Adj Close"
    fallback = "Close"
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if preferred in level0:
            prices = df[preferred].copy()
        elif fallback in level0:
            prices = df[fallback].copy()
        else:
            raise KeyError(
                f"Neither '{preferred}' nor '{fallback}' present in downloaded columns: "
                f"{sorted(set(level0))}"
            )
    else:
        only = ticker_list[0]
        if preferred in df.columns:
            prices = df.rename(columns={preferred: only})[[only]]
        elif fallback in df.columns:
            prices = df.rename(columns={fallback: only})[[only]]
        else:
            raise KeyError(
                f"Single-ticker download missing '{preferred}' and '{fallback}'. "
                f"Columns: {df.columns.tolist()}"
            )

    prices = _clean_downloaded_wide(prices)
    logger.info("Fetched price frame shape: %s", prices.shape)
    return prices


def fetch_volumes_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    progress: bool = False,
) -> pd.DataFrame:
    """Download a wide volume frame from Yahoo Finance."""
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    ticker_list = _normalise_tickers(tickers)
    _configure_yfinance_cache()

    logger.info(
        "Fetching volumes for %d tickers from %s to %s at %s interval",
        len(ticker_list),
        start,
        end,
        interval,
    )
    df = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=progress,
        group_by="column",
    )
    if df is None or df.empty:
        raise ValueError("No volume data returned. Check tickers, dates, or interval.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Volume" not in df.columns.get_level_values(0):
            raise KeyError("'Volume' not present in downloaded columns")
        volumes = df["Volume"].copy()
    else:
        only = ticker_list[0]
        if "Volume" not in df.columns:
            raise KeyError(
                f"Single-ticker download missing 'Volume'. Columns: {df.columns.tolist()}"
            )
        volumes = df[["Volume"]].rename(columns={"Volume": only})

    volumes = _clean_downloaded_wide(volumes)
    logger.info("Fetched volume frame shape: %s", volumes.shape)
    return volumes


def _fetch_prices_and_volumes(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch prices and volumes with one optimized yfinance request."""
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    ticker_list = _normalise_tickers(tickers)
    _configure_yfinance_cache()

    logger.info(
        "Fetching prices+volumes for %d tickers from %s to %s at %s interval",
        len(ticker_list),
        start,
        end,
        interval,
    )
    df = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=progress,
        group_by="column",
    )
    if df is None or df.empty:
        raise ValueError("No data returned. Check tickers, dates, or interval.")

    preferred = "Adj Close"
    fallback = "Close"
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
        if "Volume" not in level0:
            raise KeyError("'Volume' not present in downloaded columns")
        volumes = df["Volume"].copy()
    else:
        only = ticker_list[0]
        if preferred in df.columns:
            prices = df.rename(columns={preferred: only})[[only]]
        elif fallback in df.columns:
            prices = df.rename(columns={fallback: only})[[only]]
        else:
            raise KeyError(
                f"Single-ticker download missing '{preferred}' and '{fallback}'. "
                f"Columns: {df.columns.tolist()}"
            )
        if "Volume" not in df.columns:
            raise KeyError(
                f"Single-ticker download missing 'Volume'. Columns: {df.columns.tolist()}"
            )
        volumes = df[["Volume"]].rename(columns={"Volume": only})

    prices = _clean_downloaded_wide(prices)
    volumes = _clean_downloaded_wide(volumes)
    logger.info("Single-download: prices shape %s, volumes shape %s", prices.shape, volumes.shape)
    return prices, volumes


def validate_price_frame(df: pd.DataFrame) -> None:
    """Validate structural assumptions about a wide price DataFrame."""
    if df.empty:
        raise ValueError("Price DataFrame is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas.DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending (monotonic increasing).")
    if df.index.has_duplicates:
        raise ValueError("Index contains duplicate timestamps.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise TypeError("All columns must be numeric types.")


def align_and_fill(
    df: pd.DataFrame,
    freq: str = "B",
    ffill_limit: int = 5,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """Align tickers to a common calendar, fill short gaps, and drop sparse columns."""
    if df.empty:
        return df.copy()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    aligned = df.reindex(full_index)
    aligned = aligned.ffill(limit=ffill_limit).bfill(limit=1)

    coverage = aligned.notna().mean(axis=0)
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    trimmed = aligned[keep_cols]

    dropped = set(df.columns) - set(keep_cols)
    if dropped:
        logger.info("Dropped %d columns for low coverage: %s", len(dropped), sorted(dropped))
    return trimmed


def winsorize_outliers(
    df: pd.DataFrame,
    z_thresh: float = 6.0,
    min_periods: int = 2,
) -> pd.DataFrame:
    """Causally cap extreme values using only each column's prior observations.

    The cap for date t is fitted from dates strictly before t, so observations
    in the holdout period cannot influence transformed training-period values.
    """
    if z_thresh <= 0:
        raise ValueError("z_thresh must be positive")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")

    values = df.astype(float)
    history = values.shift(1)
    mu = history.expanding(min_periods=min_periods).mean()
    sigma = history.expanding(min_periods=min_periods).std()
    usable_caps = sigma.gt(0) & mu.notna()

    lower = mu - z_thresh * sigma
    upper = mu + z_thresh * sigma
    capped = values.clip(lower=lower, upper=upper)
    return values.where(~usable_caps, capped)


def compute_returns(
    prices: pd.DataFrame,
    kind: str = "log",
) -> pd.DataFrame:
    """Convert prices to log or simple returns."""
    if kind not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")
    if kind == "simple":
        return prices.pct_change()
    return np.log(prices).diff()


def add_technical_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    windows: Tuple[int, int, int] = (5, 20, 60),
) -> pd.DataFrame:
    """Build moving-average, momentum, and rolling-volatility features."""
    feats: Dict[Tuple[str, int], pd.DataFrame] = {}
    for window in windows:
        feats[("sma", window)] = prices.rolling(
            window=window, min_periods=max(1, window // 2)
        ).mean()
        feats[("momentum", window)] = prices.pct_change(periods=window)
        feats[("volatility", window)] = returns.rolling(
            window=window, min_periods=max(1, window // 2)
        ).std()

    pieces = []
    for (feature_name, window), frame in feats.items():
        feature_frame = frame.copy()
        feature_frame.columns = pd.MultiIndex.from_product(
            [[f"{feature_name}_{window}"], feature_frame.columns]
        )
        pieces.append(feature_frame)
    return pd.concat(pieces, axis=1).sort_index(axis=1)


def save_processed(
    df: pd.DataFrame,
    out_path: Path | str,
    save_parquet: bool = True,
    save_csv: bool = True,
) -> None:
    """Save a processed DataFrame to Parquet and/or CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if save_parquet:
        df.to_parquet(out_path.with_suffix(".parquet"))
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch, validate, align, transform, and featurize prices and volumes."""
    ticker_list = _normalise_tickers(tickers)

    try:
        prices, volumes = _fetch_prices_and_volumes(
            tickers=ticker_list,
            start=start,
            end=end,
            interval=interval,
        )
    except Exception as exc:
        logger.warning(
            "Combined price+volume fetch failed (%s). Retrying separate helpers.",
            exc,
        )
        prices = fetch_prices_yfinance(
            tickers=ticker_list,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
        )
        try:
            volumes = fetch_volumes_yfinance(
                tickers=ticker_list,
                start=start,
                end=end,
                interval=interval,
            )
        except Exception as volume_exc:
            logger.warning(
                "Separate volume fetch failed (%s). Continuing with NaN volumes.",
                volume_exc,
            )
            volumes = pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")

    validate_price_frame(prices)
    prices = align_and_fill(
        prices,
        freq=calendar_freq,
        ffill_limit=ffill_limit,
        min_coverage=min_coverage,
    )
    volumes = align_and_fill(
        volumes,
        freq=calendar_freq,
        ffill_limit=ffill_limit,
        min_coverage=min_coverage,
    )
    volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    returns = compute_returns(prices, kind=return_kind)
    if winsorize_z is not None:
        returns = winsorize_outliers(returns, z_thresh=winsorize_z)

    features = add_technical_features(
        prices=prices,
        returns=returns,
        windows=feature_windows,
    )
    logger.info(
        "Final shapes - prices:%s returns:%s features:%s volumes:%s",
        prices.shape,
        returns.shape,
        features.shape,
        volumes.shape,
    )
    return prices, returns, features, volumes
