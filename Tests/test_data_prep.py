"""
Unit tests for data_prep module (refactored for new API).

Covers:
- Price & volume fetching via yfinance
- Validation of price frames
- Alignment / filling
- Winsorisation of outliers
- Return calculation
- Technical feature generation
- Saving processed data
- End-to-end prepare_price_data pipeline
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------
# Ensure we can import the src package
# ---------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_prep import (  # noqa: E402
    fetch_prices_yfinance,
    fetch_volumes_yfinance,
    validate_price_frame,
    align_and_fill,
    winsorize_outliers,
    compute_returns,
    add_technical_features,
    save_processed,
    prepare_price_data,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def sample_price_frame():
    """Simple single-ticker price frame with clean DatetimeIndex."""
    dates = pd.date_range("2023-01-02", "2023-01-10", freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 110, len(dates))}, index=dates)
    return prices


@pytest.fixture
def sample_multi_price_frame():
    """Multi-ticker price frame."""
    dates = pd.date_range("2023-01-02", "2023-01-20", freq="B")
    data = {
        "AAPL": np.linspace(100, 120, len(dates)),
        "MSFT": np.linspace(200, 220, len(dates)),
        "GOOGL": np.linspace(50, 60, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_volume_frame(sample_multi_price_frame):
    """Volume frame aligned with sample_multi_price_frame."""
    vols = sample_multi_price_frame.copy()
    vols[:] = 1_000_000  # constant volume
    return vols


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for saving processed outputs."""
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    return out_dir


# ---------------------------------------------------------------------
# Tests for fetch_prices_yfinance
# ---------------------------------------------------------------------
class TestFetchPricesYfinance:
    """Tests for fetch_prices_yfinance."""

    @patch("data_prep.yf")
    def test_fetch_single_ticker_success(self, mock_yf, sample_price_frame):
        """Successful data fetch for a single ticker."""
        # Simulate yfinance MultiIndex columns: (field, ticker)
        dates = sample_price_frame.index
        cols = pd.MultiIndex.from_product([["Adj Close"], ["AAPL"]])
        mock_df = pd.DataFrame(sample_price_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        result = fetch_prices_yfinance(["AAPL"], "2023-01-01", "2023-01-10")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert list(result.columns) == ["AAPL"]
        mock_yf.download.assert_called_once()
        kwargs = mock_yf.download.call_args.kwargs
        assert kwargs["tickers"] == ["AAPL"]
        assert kwargs["group_by"] == "column"

    @patch("data_prep.yf")
    def test_fetch_multi_ticker_success(self, mock_yf, sample_multi_price_frame):
        """Successful data fetch for multiple tickers."""
        dates = sample_multi_price_frame.index
        cols = pd.MultiIndex.from_product(
            [["Adj Close"], list(sample_multi_price_frame.columns)]
        )
        mock_df = pd.DataFrame(
            sample_multi_price_frame.values, index=dates, columns=cols
        )
        mock_yf.download.return_value = mock_df

        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = fetch_prices_yfinance(tickers, "2023-01-01", "2023-01-20")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert set(result.columns) == set(tickers)
        mock_yf.download.assert_called_once()
        assert mock_yf.download.call_args.kwargs["tickers"] == tickers

    @patch("data_prep.yf", None)
    def test_fetch_yfinance_not_installed(self):
        """Error when yfinance is not installed."""
        with pytest.raises(ImportError, match="yfinance is not installed"):
            fetch_prices_yfinance(["AAPL"], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_no_data_returned(self, mock_yf):
        """Error when no data is returned."""
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data returned"):
            fetch_prices_yfinance(["INVALID"], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_uses_interval_and_progress(self, mock_yf, sample_price_frame):
        """Check interval & progress parameters are passed correctly."""
        dates = sample_price_frame.index
        cols = pd.MultiIndex.from_product([["Adj Close"], ["AAPL"]])
        mock_df = pd.DataFrame(sample_price_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        _ = fetch_prices_yfinance(
            ["AAPL"],
            "2023-01-01",
            "2023-01-10",
            interval="1h",
            progress=True,
        )
        kwargs = mock_yf.download.call_args.kwargs
        assert kwargs["interval"] == "1h"
        assert kwargs["progress"] is True

    @patch("data_prep.yf")
    def test_fetch_raises_if_adj_close_and_close_missing(self, mock_yf, sample_price_frame):
        """If the required fields are missing, we should get a KeyError."""
        # Build MultiIndex without Adj Close / Close
        dates = sample_price_frame.index
        cols = pd.MultiIndex.from_product([["High"], ["AAPL"]])
        mock_df = pd.DataFrame(sample_price_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        with pytest.raises(KeyError):
            fetch_prices_yfinance(["AAPL"], "2023-01-01", "2023-01-10")


# ---------------------------------------------------------------------
# Tests for fetch_volumes_yfinance
# ---------------------------------------------------------------------
class TestFetchVolumesYfinance:
    """Tests for fetch_volumes_yfinance."""

    @patch("data_prep.yf")
    def test_fetch_volumes_success(self, mock_yf, sample_volume_frame):
        """Successful volume fetch with MultiIndex Volume field."""
        dates = sample_volume_frame.index
        tickers = list(sample_volume_frame.columns)
        cols = pd.MultiIndex.from_product([["Volume"], tickers])
        mock_df = pd.DataFrame(sample_volume_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        result = fetch_volumes_yfinance(tickers, "2023-01-01", "2023-01-20")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert set(result.columns) == set(tickers)

    @patch("data_prep.yf")
    def test_fetch_volumes_no_data(self, mock_yf):
        """No volume data returned should raise ValueError."""
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No volume data returned"):
            fetch_volumes_yfinance(["AAPL"], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_volumes_missing_volume_field(self, mock_yf, sample_volume_frame):
        """Missing Volume field should raise KeyError."""
        dates = sample_volume_frame.index
        tickers = list(sample_volume_frame.columns)
        cols = pd.MultiIndex.from_product([["Adj Close"], tickers])
        mock_df = pd.DataFrame(sample_volume_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        with pytest.raises(KeyError, match="Volume"):
            fetch_volumes_yfinance(tickers, "2023-01-01", "2023-01-10")


# ---------------------------------------------------------------------
# Tests for validate_price_frame
# ---------------------------------------------------------------------
class TestValidatePriceFrame:
    """Tests for validate_price_frame."""

    def test_validate_success(self, sample_price_frame):
        validate_price_frame(sample_price_frame)  # should not raise

    def test_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            validate_price_frame(pd.DataFrame())

    def test_non_datetime_index(self, sample_price_frame):
        df = sample_price_frame.reset_index(drop=True)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_price_frame(df)

    def test_unsorted_index(self, sample_price_frame):
        df = sample_price_frame.sort_index(ascending=False)
        with pytest.raises(ValueError, match="sorted ascending"):
            validate_price_frame(df)

    def test_duplicate_index(self, sample_price_frame):
        df = sample_price_frame.copy()
        dup_index = df.index.tolist() + [df.index[-1]]
        df = df.reindex(dup_index)
        with pytest.raises(ValueError, match="duplicate"):
            validate_price_frame(df)

    def test_non_numeric_columns(self, sample_price_frame):
        df = sample_price_frame.copy()
        df["TEXT"] = "abc"
        with pytest.raises(TypeError, match="numeric"):
            validate_price_frame(df)


# ---------------------------------------------------------------------
# Tests for align_and_fill
# ---------------------------------------------------------------------
class TestAlignAndFill:
    """Tests for align_and_fill."""

    def test_aligns_to_business_calendar(self, sample_price_frame):
        aligned = align_and_fill(sample_price_frame, freq="B")
        # Index should start/end at same bounds but be continuous business days
        expected_index = pd.date_range(
            sample_price_frame.index.min(),
            sample_price_frame.index.max(),
            freq="B",
        )
        assert aligned.index.equals(expected_index)

    def test_drops_low_coverage_columns(self, sample_multi_price_frame):
        df = sample_multi_price_frame.copy()
        # Make one column mostly NaN to force low coverage
        df["GOOGL"] = np.nan
        aligned = align_and_fill(df, freq="B", min_coverage=0.9)
        assert "GOOGL" not in aligned.columns
        assert "AAPL" in aligned.columns
        assert "MSFT" in aligned.columns


# ---------------------------------------------------------------------
# Tests for winsorize_outliers
# ---------------------------------------------------------------------
class TestWinsorizeOutliers:
    """Tests for winsorize_outliers."""

    def test_caps_extreme_values(self, sample_price_frame):
        df = sample_price_frame.copy()
        df.iloc[-1, 0] = 10_000  # big outlier
        capped = winsorize_outliers(df, z_thresh=2.5)
        # Outlier should be reduced relative to original
        assert capped.iloc[-1, 0] <= df.iloc[-1, 0]


# ---------------------------------------------------------------------
# Tests for compute_returns
# ---------------------------------------------------------------------
class TestComputeReturns:
    """Tests for compute_returns."""

    def test_simple_returns(self, sample_price_frame):
        rets = compute_returns(sample_price_frame, kind="simple")
        assert isinstance(rets, pd.DataFrame)
        assert rets.iloc[0].isna().all()  # first row NaN
        # simple return of last/penultimate
        expected = sample_price_frame["AAPL"].pct_change().iloc[-1]
        assert np.isclose(rets["AAPL"].iloc[-1], expected)

    def test_log_returns(self, sample_price_frame):
        rets = compute_returns(sample_price_frame, kind="log")
        expected = np.log(sample_price_frame["AAPL"]).diff().iloc[-1]
        assert np.isclose(rets["AAPL"].iloc[-1], expected)

    def test_invalid_kind(self, sample_price_frame):
        with pytest.raises(ValueError, match="kind must be 'log' or 'simple'"):
            compute_returns(sample_price_frame, kind="foo")


# ---------------------------------------------------------------------
# Tests for add_technical_features
# ---------------------------------------------------------------------
class TestAddTechnicalFeatures:
    """Tests for add_technical_features."""

    def test_features_structure(self, sample_price_frame):
        returns = compute_returns(sample_price_frame)
        feat_df = add_technical_features(
            prices=sample_price_frame,
            returns=returns,
            windows=(5, 10),
        )
        assert isinstance(feat_df, pd.DataFrame)
        # Columns should be MultiIndex (feature_name, ticker)
        assert isinstance(feat_df.columns, pd.MultiIndex)
        feature_levels = set(feat_df.columns.get_level_values(0))
        assert "sma_5" in feature_levels
        assert "momentum_5" in feature_levels
        assert "volatility_5" in feature_levels


# ---------------------------------------------------------------------
# Tests for save_processed
# ---------------------------------------------------------------------
class TestSaveProcessed:
    """Tests for save_processed."""

    def test_save_creates_files(self, sample_price_frame, temp_data_dir):
        out_base = temp_data_dir / "prices"
        save_processed(sample_price_frame, out_base)

        parquet_path = out_base.with_suffix(".parquet")
        csv_path = out_base.with_suffix(".csv")

        assert parquet_path.exists()
        assert csv_path.exists()

        loaded_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        assert len(loaded_csv) == len(sample_price_frame)


# ---------------------------------------------------------------------
# Integration / pipeline tests for prepare_price_data
# ---------------------------------------------------------------------
class TestPreparePriceData:
    """Integration tests for prepare_price_data."""

    @patch("data_prep.fetch_volumes_yfinance")
    @patch("data_prep.fetch_prices_yfinance")
    def test_pipeline_shapes(
        self,
        mock_fetch_prices,
        mock_fetch_volumes,
        sample_multi_price_frame,
        sample_volume_frame,
    ):
        """End-to-end pipeline produces non-empty prices, returns, features, volumes."""
        mock_fetch_prices.return_value = sample_multi_price_frame
        mock_fetch_volumes.return_value = sample_volume_frame

        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices, returns, features, volumes = prepare_price_data(
            tickers=tickers,
            start="2023-01-01",
            end="2023-01-31",
            interval="1d",
        )

        # Basic shape sanity checks
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(returns, pd.DataFrame)
        assert isinstance(features, pd.DataFrame)
        assert isinstance(volumes, pd.DataFrame)

        assert not prices.empty
        assert not returns.empty
        assert not features.empty
        assert not volumes.empty

        # Index alignment
        assert prices.index.equals(returns.index)
        assert prices.index.equals(volumes.index)

    @patch("data_prep.fetch_volumes_yfinance")
    @patch("data_prep.fetch_prices_yfinance")
    def test_pipeline_winsorization_toggle(
        self,
        mock_fetch_prices,
        mock_fetch_volumes,
        sample_multi_price_frame,
        sample_volume_frame,
    ):
        """Pipeline works with winsorisation disabled."""
        mock_fetch_prices.return_value = sample_multi_price_frame
        mock_fetch_volumes.return_value = sample_volume_frame

        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices, returns, features, volumes = prepare_price_data(
            tickers=tickers,
            start="2023-01-01",
            end="2023-01-31",
            interval="1d",
            winsorize_z=None,
        )

        # Just ensure nothing exploded
        assert not prices.empty
        assert not returns.empty
        assert not features.empty
        assert not volumes.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
