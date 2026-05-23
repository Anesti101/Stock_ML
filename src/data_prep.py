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
