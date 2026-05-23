"""Unit tests for dashboard module.

Tests cover technical indicator calculations, chart creation, and callback functions.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add Dashboard to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Dashboard'))

from dashboard import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_summary_stats,
    create_candlestick_chart,
    create_technical_indicators_chart,
    create_rsi_chart,
    create_volume_chart
)


# Fixtures
@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create realistic price movement
    base_price = 100
    price_changes = np.random.normal(0, 2, len(dates))
    close_prices = base_price + np.cumsum(price_changes)
    
    data = pd.DataFrame({
        'Open': close_prices + np.random.uniform(-1, 1, len(dates)),
        'High': close_prices + np.random.uniform(1, 3, len(dates)),
        'Low': close_prices + np.random.uniform(-3, -1, len(dates)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data


# Tests for calculate_moving_averages
class TestCalculateMovingAverages:
    """Tests for calculate_moving_averages function."""
    
    def test_default_windows(self, sample_stock_data):
        """Test MA calculation with default windows."""
        result = calculate_moving_averages(sample_stock_data)
        
        assert 'SMA_5' in result.columns
        assert 'SMA_10' in result.columns
        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        
        # Original columns should still be present
        assert all(col in result.columns for col in sample_stock_data.columns)
    
    def test_custom_windows(self, sample_stock_data):
        """Test MA calculation with custom windows."""
        result = calculate_moving_averages(sample_stock_data, windows=[7, 14])
        
        assert 'SMA_7' in result.columns
        assert 'SMA_14' in result.columns
        assert 'SMA_5' not in result.columns
    
    def test_ma_values(self, sample_stock_data):
        """Test that MA values are calculated correctly."""
        result = calculate_moving_averages(sample_stock_data, windows=[5])
        
        # Calculate expected MA manually for verification
        expected_ma5 = sample_stock_data['Close'].rolling(window=5).mean()
        
        pd.testing.assert_series_equal(result['SMA_5'], expected_ma5)
    
    def test_ma_nan_values(self, sample_stock_data):
        """Test that MA has NaN values for initial period."""
        result = calculate_moving_averages(sample_stock_data, windows=[5])
        
        # First 4 values should be NaN
        assert result['SMA_5'].iloc[:4].isna().all()
        # 5th value onwards should have data
        assert not result['SMA_5'].iloc[4:].isna().all()
    
    def test_original_data_not_modified(self, sample_stock_data):
        """Test that original dataframe is not modified."""
        original_cols = set(sample_stock_data.columns)
        calculate_moving_averages(sample_stock_data)
        
        assert set(sample_stock_data.columns) == original_cols


# Tests for calculate_rsi
class TestCalculateRSI:
    """Tests for calculate_rsi function."""
    
    def test_rsi_output_type(self, sample_stock_data):
        """Test RSI returns a Series."""
        result = calculate_rsi(sample_stock_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_rsi_range(self, sample_stock_data):
        """Test RSI values are within valid range (0-100)."""
        result = calculate_rsi(sample_stock_data)
        
        # Drop NaN values
        valid_values = result.dropna()
        
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_rsi_custom_period(self, sample_stock_data):
        """Test RSI with custom period."""
        result = calculate_rsi(sample_stock_data, period=7)
        
        assert isinstance(result, pd.Series)
        # First (period) values should be NaN
        assert result.iloc[:7].isna().any()
    
    def test_rsi_increasing_prices(self):
        """Test RSI for continuously increasing prices."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'Close': np.arange(100, 150)  # Continuously increasing
        }, index=dates)
        
        result = calculate_rsi(data, period=14)
        
        # RSI should be high (>50) for increasing prices
        assert result.iloc[-1] > 50


# Tests for calculate_bollinger_bands
class TestCalculateBollingerBands:
    """Tests for calculate_bollinger_bands function."""
    
    def test_bollinger_bands_output(self, sample_stock_data):
        """Test Bollinger Bands returns three series."""
        middle, upper, lower = calculate_bollinger_bands(sample_stock_data)
        
        assert isinstance(middle, pd.Series)
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        
        assert len(middle) == len(sample_stock_data)
        assert len(upper) == len(sample_stock_data)
        assert len(lower) == len(sample_stock_data)
    
    def test_bollinger_bands_relationship(self, sample_stock_data):
        """Test that upper > middle > lower."""
        middle, upper, lower = calculate_bollinger_bands(sample_stock_data)
        
        # Remove NaN values
        valid_idx = middle.notna()
        
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
    
    def test_bollinger_bands_custom_params(self, sample_stock_data):
        """Test Bollinger Bands with custom parameters."""
        middle, upper, lower = calculate_bollinger_bands(sample_stock_data, window=10, num_std=3)
        
        assert isinstance(middle, pd.Series)
        # First 9 values should be NaN for window=10
        assert middle.iloc[:9].isna().all()
    
    def test_middle_band_equals_sma(self, sample_stock_data):
        """Test that middle band equals SMA."""
        middle, _, _ = calculate_bollinger_bands(sample_stock_data, window=20)
        expected_sma = sample_stock_data['Close'].rolling(window=20).mean()
        
        pd.testing.assert_series_equal(middle, expected_sma)


# Tests for calculate_summary_stats
class TestCalculateSummaryStats:
    """Tests for calculate_summary_stats function."""
    
    def test_summary_stats_keys(self, sample_stock_data):
        """Test that all expected keys are present."""
        result = calculate_summary_stats(sample_stock_data)
        
        expected_keys = [
            'current_price', 'price_change', 'price_change_pct',
            'high', 'low', 'avg_volume', 'volatility'
        ]
        
        assert all(key in result for key in expected_keys)
    
    def test_summary_stats_format(self, sample_stock_data):
        """Test that stats are formatted as strings."""
        result = calculate_summary_stats(sample_stock_data)
        
        assert isinstance(result['current_price'], str)
        assert result['current_price'].startswith('$')
        assert result['price_change_pct'].endswith('%')
    
    def test_price_change_calculation(self, sample_stock_data):
        """Test that price change is calculated correctly."""
        result = calculate_summary_stats(sample_stock_data)
        
        current = sample_stock_data['Close'].iloc[-1]
        first = sample_stock_data['Close'].iloc[0]
        expected_change = current - first
        
        # Extract numeric value from formatted string
        actual_change = float(result['price_change'].replace('$', ''))
        
        assert abs(actual_change - expected_change) < 0.01
    
    def test_high_low_values(self, sample_stock_data):
        """Test high and low are correct."""
        result = calculate_summary_stats(sample_stock_data)
        
        expected_high = sample_stock_data['High'].max()
        expected_low = sample_stock_data['Low'].min()
        
        actual_high = float(result['high'].replace('$', ''))
        actual_low = float(result['low'].replace('$', ''))
        
        assert abs(actual_high - expected_high) < 0.01
        assert abs(actual_low - expected_low) < 0.01


# Tests for chart creation functions
class TestChartCreation:
    """Tests for chart creation functions."""
    
    def test_candlestick_chart_creation(self, sample_stock_data):
        """Test candlestick chart is created."""
        fig = create_candlestick_chart(sample_stock_data, 'AAPL')
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.data[0].type == 'candlestick'
        assert 'AAPL' in fig.layout.title.text
    
    def test_technical_indicators_chart_creation(self, sample_stock_data):
        """Test technical indicators chart is created."""
        fig = create_technical_indicators_chart(sample_stock_data, 'AAPL')
        
        assert fig is not None
        assert len(fig.data) > 0  # Should have multiple traces
        
        # Check for close price and MA traces
        trace_names = [trace.name for trace in fig.data]
        assert 'Close Price' in trace_names
        assert any('SMA' in name for name in trace_names)
    
    def test_rsi_chart_creation(self, sample_stock_data):
        """Test RSI chart is created."""
        fig = create_rsi_chart(sample_stock_data, 'AAPL')
        
        assert fig is not None
        assert len(fig.data) > 0
        assert 'RSI' in fig.data[0].name
        assert 'AAPL' in fig.layout.title.text
        
        # Check for overbought/oversold lines
        assert len(fig.layout.shapes) >= 2  # Horizontal lines
    
    def test_volume_chart_creation(self, sample_stock_data):
        """Test volume chart is created."""
        fig = create_volume_chart(sample_stock_data, 'AAPL')
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.data[0].type == 'bar'
        assert 'AAPL' in fig.layout.title.text
    
    def test_chart_with_minimal_data(self):
        """Test charts work with minimal data."""
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 60),
            'High': np.random.uniform(110, 120, 60),
            'Low': np.random.uniform(90, 100, 60),
            'Close': np.random.uniform(100, 110, 60),
            'Volume': np.random.randint(1000000, 10000000, 60)
        }, index=dates)
        
        # Should not raise errors
        fig1 = create_candlestick_chart(data, 'TEST')
        fig2 = create_technical_indicators_chart(data, 'TEST')
        fig3 = create_rsi_chart(data, 'TEST')
        fig4 = create_volume_chart(data, 'TEST')
        
        assert all(fig is not None for fig in [fig1, fig2, fig3, fig4])


# Integration tests
class TestIntegration:
    """Integration tests for dashboard components."""
    
    def test_full_analysis_pipeline(self, sample_stock_data):
        """Test complete analysis pipeline."""
        # Calculate all indicators
        df_with_ma = calculate_moving_averages(sample_stock_data)
        rsi = calculate_rsi(sample_stock_data)
        middle, upper, lower = calculate_bollinger_bands(sample_stock_data)
        stats = calculate_summary_stats(sample_stock_data)
        
        # Create all charts
        candlestick = create_candlestick_chart(sample_stock_data, 'AAPL')
        technical = create_technical_indicators_chart(sample_stock_data, 'AAPL')
        rsi_chart = create_rsi_chart(sample_stock_data, 'AAPL')
        volume = create_volume_chart(sample_stock_data, 'AAPL')
        
        # Verify all components created successfully
        assert df_with_ma is not None
        assert rsi is not None
        assert stats is not None
        assert all(fig is not None for fig in [candlestick, technical, rsi_chart, volume])
    
    def test_data_consistency(self, sample_stock_data):
        """Test that calculated indicators maintain data consistency."""
        df_with_ma = calculate_moving_averages(sample_stock_data)
        
        # Index should remain the same
        pd.testing.assert_index_equal(df_with_ma.index, sample_stock_data.index)
        
        # Original data should be unchanged
        for col in sample_stock_data.columns:
            pd.testing.assert_series_equal(
                df_with_ma[col],
                sample_stock_data[col]
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
