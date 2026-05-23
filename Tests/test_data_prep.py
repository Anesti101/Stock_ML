"""Unit tests for data_prep module.

Tests cover data fetching, validation, saving, and loading functionality.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_prep import (
    fetch_data_yfinance,
    validate_stock_data,
    save_stock_data,
    load_stock_data
)


# Fixtures
@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    return data


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# Tests for fetch_data_yfinance
class TestFetchDataYfinance:
    """Tests for fetch_data_yfinance function."""
    
    @patch('data_prep.yf')
    def test_fetch_single_ticker_success(self, mock_yf, sample_stock_data):
        """Test successful data fetch for single ticker."""
        mock_yf.download.return_value = sample_stock_data
        
        result = fetch_data_yfinance('AAPL', '2023-01-01', '2023-01-10')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_yf.download.assert_called_once()
        assert mock_yf.download.call_args[1]['tickers'] == ['AAPL']
    
    @patch('data_prep.yf')
    def test_fetch_multiple_tickers_success(self, mock_yf, sample_stock_data):
        """Test successful data fetch for multiple tickers."""
        mock_yf.download.return_value = sample_stock_data
        
        result = fetch_data_yfinance(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-10')
        
        assert isinstance(result, pd.DataFrame)
        mock_yf.download.assert_called_once()
        assert mock_yf.download.call_args[1]['tickers'] == ['AAPL', 'GOOGL']
        assert mock_yf.download.call_args[1]['group_by'] == 'ticker'
    
    @patch('data_prep.yf', None)
    def test_fetch_yfinance_not_installed(self):
        """Test error when yfinance is not installed."""
        with pytest.raises(ImportError, match="yfinance is not installed"):
            fetch_data_yfinance('AAPL', '2023-01-01', '2023-01-10')
    
    @patch('data_prep.yf')
    def test_fetch_empty_ticker_list(self, mock_yf):
        """Test error with empty ticker list."""
        with pytest.raises(ValueError, match="At least one ticker symbol must be provided"):
            fetch_data_yfinance([], '2023-01-01', '2023-01-10')
    
    @patch('data_prep.yf')
    def test_fetch_no_data_returned(self, mock_yf):
        """Test error when no data is returned."""
        mock_yf.download.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No data returned"):
            fetch_data_yfinance('INVALID', '2023-01-01', '2023-01-10')
    
    @patch('data_prep.yf')
    def test_fetch_with_custom_interval(self, mock_yf, sample_stock_data):
        """Test data fetch with custom interval."""
        mock_yf.download.return_value = sample_stock_data
        
        result = fetch_data_yfinance('AAPL', '2023-01-01', '2023-01-10', interval='1h')
        
        assert mock_yf.download.call_args[1]['interval'] == '1h'
    
    @patch('data_prep.yf')
    def test_fetch_with_progress(self, mock_yf, sample_stock_data):
        """Test data fetch with progress bar enabled."""
        mock_yf.download.return_value = sample_stock_data
        
        result = fetch_data_yfinance('AAPL', '2023-01-01', '2023-01-10', progress=True)
        
        assert mock_yf.download.call_args[1]['progress'] is True
    
    @patch('data_prep.yf')
    def test_fetch_exception_handling(self, mock_yf):
        """Test exception handling during data fetch."""
        mock_yf.download.side_effect = Exception("Network error")
        
        with pytest.raises(Exception, match="Network error"):
            fetch_data_yfinance('AAPL', '2023-01-01', '2023-01-10')


# Tests for validate_stock_data
class TestValidateStockData:
    """Tests for validate_stock_data function."""
    
    def test_validate_success(self, sample_stock_data):
        """Test successful validation of stock data."""
        result = validate_stock_data(sample_stock_data)
        assert result is True
    
    def test_validate_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_stock_data(pd.DataFrame())
    
    def test_validate_missing_columns(self, sample_stock_data):
        """Test validation fails for missing required columns."""
        incomplete_data = sample_stock_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_stock_data(incomplete_data)
    
    def test_validate_custom_columns(self, sample_stock_data):
        """Test validation with custom required columns."""
        result = validate_stock_data(sample_stock_data, required_columns=['Open', 'Close'])
        assert result is True
    
    def test_validate_insufficient_data(self):
        """Test validation fails for insufficient data rows."""
        data = pd.DataFrame({
            'Open': [100],
            'High': [110],
            'Low': [90],
            'Close': [105],
            'Volume': [1000000]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            validate_stock_data(data)
    
    def test_validate_all_nan_close(self, sample_stock_data):
        """Test validation fails when all Close prices are NaN."""
        sample_stock_data['Close'] = np.nan
        
        with pytest.raises(ValueError, match="All Close prices are NaN"):
            validate_stock_data(sample_stock_data)


# Tests for save_stock_data
class TestSaveStockData:
    """Tests for save_stock_data function."""
    
    def test_save_csv(self, sample_stock_data, temp_data_dir):
        """Test saving data to CSV format."""
        filepath = temp_data_dir / "test_data.csv"
        save_stock_data(sample_stock_data, filepath, format='csv')
        
        assert filepath.exists()
        loaded_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(loaded_data) == len(sample_stock_data)
    
    def test_save_pickle(self, sample_stock_data, temp_data_dir):
        """Test saving data to pickle format."""
        filepath = temp_data_dir / "test_data.pkl"
        save_stock_data(sample_stock_data, filepath, format='pickle')
        
        assert filepath.exists()
    
    def test_save_parquet(self, sample_stock_data, temp_data_dir):
        """Test saving data to parquet format."""
        filepath = temp_data_dir / "test_data.parquet"
        save_stock_data(sample_stock_data, filepath, format='parquet')
        
        assert filepath.exists()
    
    def test_save_invalid_format(self, sample_stock_data, temp_data_dir):
        """Test error with invalid format."""
        filepath = temp_data_dir / "test_data.xyz"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            save_stock_data(sample_stock_data, filepath, format='invalid')
    
    def test_save_creates_directory(self, sample_stock_data, temp_data_dir):
        """Test that save creates parent directories if they don't exist."""
        filepath = temp_data_dir / "subdir" / "test_data.csv"
        save_stock_data(sample_stock_data, filepath, format='csv')
        
        assert filepath.exists()
        assert filepath.parent.exists()


# Tests for load_stock_data
class TestLoadStockData:
    """Tests for load_stock_data function."""
    
    def test_load_csv(self, sample_stock_data, temp_data_dir):
        """Test loading data from CSV format."""
        filepath = temp_data_dir / "test_data.csv"
        sample_stock_data.to_csv(filepath)
        
        loaded_data = load_stock_data(filepath)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_stock_data)
        assert list(loaded_data.columns) == list(sample_stock_data.columns)
    
    def test_load_pickle(self, sample_stock_data, temp_data_dir):
        """Test loading data from pickle format."""
        filepath = temp_data_dir / "test_data.pkl"
        sample_stock_data.to_pickle(filepath)
        
        loaded_data = load_stock_data(filepath)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_stock_data)
    
    def test_load_parquet(self, sample_stock_data, temp_data_dir):
        """Test loading data from parquet format."""
        filepath = temp_data_dir / "test_data.parquet"
        sample_stock_data.to_parquet(filepath)
        
        loaded_data = load_stock_data(filepath)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_stock_data)
    
    def test_load_file_not_found(self, temp_data_dir):
        """Test error when file doesn't exist."""
        filepath = temp_data_dir / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_stock_data(filepath)
    
    def test_load_unsupported_format(self, temp_data_dir):
        """Test error with unsupported file format."""
        filepath = temp_data_dir / "test_data.xyz"
        filepath.touch()  # Create empty file
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_stock_data(filepath)


# Integration tests
class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_save_and_load_roundtrip(self, sample_stock_data, temp_data_dir):
        """Test saving and loading data maintains integrity."""
        filepath = temp_data_dir / "roundtrip.csv"
        
        save_stock_data(sample_stock_data, filepath)
        loaded_data = load_stock_data(filepath)
        
        pd.testing.assert_frame_equal(
            sample_stock_data.round(6),  # Round to avoid floating point precision issues
            loaded_data.round(6)
        )
    
    def test_validate_after_load(self, sample_stock_data, temp_data_dir):
        """Test validation of loaded data."""
        filepath = temp_data_dir / "validate_test.csv"
        
        save_stock_data(sample_stock_data, filepath)
        loaded_data = load_stock_data(filepath)
        
        assert validate_stock_data(loaded_data) is True
    
    @patch('data_prep.yf')
    def test_fetch_and_validate(self, mock_yf, sample_stock_data):
        """Test fetching and validating data."""
        mock_yf.download.return_value = sample_stock_data
        
        data = fetch_data_yfinance('AAPL', '2023-01-01', '2023-01-10')
        assert validate_stock_data(data) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
