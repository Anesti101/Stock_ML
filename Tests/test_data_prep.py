"""
Unit tests for data_prep.

These tests cover the legacy raw-data helpers, the current price/volume feature
pipeline, and the no-look-ahead guarantees in preprocessing.
"""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_prep import (  # noqa: E402
    align_and_fill,
    add_technical_features,
    compute_returns,
    fetch_data_yfinance,
    fetch_prices_yfinance,
    fetch_volumes_yfinance,
    load_stock_data,
    prepare_price_data,
    save_processed,
    save_stock_data,
    validate_price_frame,
    validate_stock_data,
    winsorize_outliers,
)


@pytest.fixture
def sample_stock_data():
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    return pd.DataFrame(
        {
            "Open": np.random.uniform(100, 110, len(dates)),
            "High": np.random.uniform(110, 120, len(dates)),
            "Low": np.random.uniform(90, 100, len(dates)),
            "Close": np.random.uniform(100, 110, len(dates)),
            "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def sample_price_frame():
    dates = pd.date_range("2023-01-02", "2023-01-10", freq="B")
    return pd.DataFrame({"AAPL": np.linspace(100, 110, len(dates))}, index=dates)


@pytest.fixture
def sample_multi_price_frame():
    dates = pd.date_range("2023-01-02", "2023-01-20", freq="B")
    return pd.DataFrame(
        {
            "AAPL": np.linspace(100, 120, len(dates)),
            "MSFT": np.linspace(200, 220, len(dates)),
            "GOOGL": np.linspace(50, 60, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def sample_volume_frame(sample_multi_price_frame):
    volumes = sample_multi_price_frame.copy()
    volumes[:] = 1_000_000
    return volumes


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    return data_dir


class TestFetchDataYfinance:
    @patch("data_prep.yf")
    def test_fetch_single_ticker_success(self, mock_yf, sample_stock_data):
        mock_yf.download.return_value = sample_stock_data

        result = fetch_data_yfinance("AAPL", "2023-01-01", "2023-01-10")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_yf.download.assert_called_once()
        assert mock_yf.download.call_args.kwargs["tickers"] == ["AAPL"]

    @patch("data_prep.yf")
    def test_fetch_multiple_tickers_success(self, mock_yf, sample_stock_data):
        mock_yf.download.return_value = sample_stock_data

        result = fetch_data_yfinance(["AAPL", "GOOGL"], "2023-01-01", "2023-01-10")

        assert isinstance(result, pd.DataFrame)
        mock_yf.download.assert_called_once()
        assert mock_yf.download.call_args.kwargs["tickers"] == ["AAPL", "GOOGL"]
        assert mock_yf.download.call_args.kwargs["group_by"] == "ticker"

    @patch("data_prep.yf", None)
    def test_fetch_yfinance_not_installed(self):
        with pytest.raises(ImportError, match="yfinance is not installed"):
            fetch_data_yfinance("AAPL", "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_empty_ticker_list(self, mock_yf):
        with pytest.raises(ValueError, match="At least one ticker symbol"):
            fetch_data_yfinance([], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_no_data_returned(self, mock_yf):
        mock_yf.download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data returned"):
            fetch_data_yfinance("INVALID", "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_with_custom_interval(self, mock_yf, sample_stock_data):
        mock_yf.download.return_value = sample_stock_data

        fetch_data_yfinance("AAPL", "2023-01-01", "2023-01-10", interval="1h")

        assert mock_yf.download.call_args.kwargs["interval"] == "1h"

    @patch("data_prep.yf")
    def test_fetch_with_progress(self, mock_yf, sample_stock_data):
        mock_yf.download.return_value = sample_stock_data

        fetch_data_yfinance("AAPL", "2023-01-01", "2023-01-10", progress=True)

        assert mock_yf.download.call_args.kwargs["progress"] is True

    @patch("data_prep.yf")
    def test_fetch_exception_handling(self, mock_yf):
        mock_yf.download.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            fetch_data_yfinance("AAPL", "2023-01-01", "2023-01-10")


class TestValidateStockData:
    def test_validate_success(self, sample_stock_data):
        assert validate_stock_data(sample_stock_data) is True

    def test_validate_empty_dataframe(self):
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_stock_data(pd.DataFrame())

    def test_validate_missing_columns(self, sample_stock_data):
        incomplete_data = sample_stock_data.drop(columns=["Close"])

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_stock_data(incomplete_data)

    def test_validate_custom_columns(self, sample_stock_data):
        assert validate_stock_data(sample_stock_data, required_columns=["Open", "Close"]) is True

    def test_validate_insufficient_data(self):
        data = pd.DataFrame(
            {
                "Open": [100],
                "High": [110],
                "Low": [90],
                "Close": [105],
                "Volume": [1_000_000],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            validate_stock_data(data)

    def test_validate_all_nan_close(self, sample_stock_data):
        sample_stock_data["Close"] = np.nan

        with pytest.raises(ValueError, match="All Close prices are NaN"):
            validate_stock_data(sample_stock_data)


class TestSaveAndLoadStockData:
    def test_save_csv(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.csv"
        save_stock_data(sample_stock_data, filepath, format="csv")

        assert filepath.exists()
        loaded_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(loaded_data) == len(sample_stock_data)

    def test_save_pickle(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.pkl"
        save_stock_data(sample_stock_data, filepath, format="pickle")

        assert filepath.exists()

    def test_save_parquet(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.parquet"
        save_stock_data(sample_stock_data, filepath, format="parquet")

        assert filepath.exists()

    def test_save_invalid_format(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.xyz"

        with pytest.raises(ValueError, match="Unsupported format"):
            save_stock_data(sample_stock_data, filepath, format="invalid")

    def test_save_creates_directory(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "subdir" / "test_data.csv"
        save_stock_data(sample_stock_data, filepath, format="csv")

        assert filepath.exists()
        assert filepath.parent.exists()

    def test_load_csv(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.csv"
        sample_stock_data.to_csv(filepath)

        loaded_data = load_stock_data(filepath)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_stock_data)
        assert list(loaded_data.columns) == list(sample_stock_data.columns)

    def test_load_pickle(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.pkl"
        sample_stock_data.to_pickle(filepath)

        loaded_data = load_stock_data(filepath)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_stock_data)

    def test_load_parquet(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "test_data.parquet"
        sample_stock_data.to_parquet(filepath)

        loaded_data = load_stock_data(filepath)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_stock_data)

    def test_load_file_not_found(self, temp_data_dir):
        filepath = temp_data_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="File not found"):
            load_stock_data(filepath)

    def test_load_unsupported_format(self, temp_data_dir):
        filepath = temp_data_dir / "test_data.xyz"
        filepath.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_stock_data(filepath)

    def test_save_and_load_roundtrip(self, sample_stock_data, temp_data_dir):
        filepath = temp_data_dir / "roundtrip.csv"

        save_stock_data(sample_stock_data, filepath)
        loaded_data = load_stock_data(filepath)

        pd.testing.assert_frame_equal(sample_stock_data.round(6), loaded_data.round(6))


class TestFetchPricesYfinance:
    @patch("data_prep.yf")
    def test_fetch_single_ticker_success(self, mock_yf, sample_price_frame):
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
        dates = sample_multi_price_frame.index
        cols = pd.MultiIndex.from_product([["Adj Close"], list(sample_multi_price_frame.columns)])
        mock_df = pd.DataFrame(sample_multi_price_frame.values, index=dates, columns=cols)
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
        with pytest.raises(ImportError, match="yfinance is not installed"):
            fetch_prices_yfinance(["AAPL"], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_no_data_returned(self, mock_yf):
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data returned"):
            fetch_prices_yfinance(["INVALID"], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_uses_interval_and_progress(self, mock_yf, sample_price_frame):
        dates = sample_price_frame.index
        cols = pd.MultiIndex.from_product([["Adj Close"], ["AAPL"]])
        mock_df = pd.DataFrame(sample_price_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        fetch_prices_yfinance(
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
        dates = sample_price_frame.index
        cols = pd.MultiIndex.from_product([["High"], ["AAPL"]])
        mock_df = pd.DataFrame(sample_price_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        with pytest.raises(KeyError):
            fetch_prices_yfinance(["AAPL"], "2023-01-01", "2023-01-10")


class TestFetchVolumesYfinance:
    @patch("data_prep.yf")
    def test_fetch_volumes_success(self, mock_yf, sample_volume_frame):
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
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No volume data returned"):
            fetch_volumes_yfinance(["AAPL"], "2023-01-01", "2023-01-10")

    @patch("data_prep.yf")
    def test_fetch_volumes_missing_volume_field(self, mock_yf, sample_volume_frame):
        dates = sample_volume_frame.index
        tickers = list(sample_volume_frame.columns)
        cols = pd.MultiIndex.from_product([["Adj Close"], tickers])
        mock_df = pd.DataFrame(sample_volume_frame.values, index=dates, columns=cols)
        mock_yf.download.return_value = mock_df

        with pytest.raises(KeyError, match="Volume"):
            fetch_volumes_yfinance(tickers, "2023-01-01", "2023-01-10")


class TestValidatePriceFrame:
    def test_validate_success(self, sample_price_frame):
        validate_price_frame(sample_price_frame)

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
        duplicate_index = df.index.tolist() + [df.index[-1]]
        df = df.reindex(duplicate_index)
        with pytest.raises(ValueError, match="duplicate"):
            validate_price_frame(df)

    def test_non_numeric_columns(self, sample_price_frame):
        df = sample_price_frame.copy()
        df["TEXT"] = "abc"
        with pytest.raises(TypeError, match="numeric"):
            validate_price_frame(df)


class TestAlignAndFill:
    def test_aligns_to_business_calendar(self, sample_price_frame):
        aligned = align_and_fill(sample_price_frame, freq="B")
        expected_index = pd.date_range(
            sample_price_frame.index.min(),
            sample_price_frame.index.max(),
            freq="B",
        )
        assert aligned.index.equals(expected_index)

    def test_drops_low_coverage_columns(self, sample_multi_price_frame):
        df = sample_multi_price_frame.copy()
        df["GOOGL"] = np.nan

        aligned = align_and_fill(df, freq="B", min_coverage=0.9)

        assert "GOOGL" not in aligned.columns
        assert "AAPL" in aligned.columns
        assert "MSFT" in aligned.columns


class TestWinsorizeOutliers:
    def test_caps_extreme_values(self, sample_price_frame):
        df = sample_price_frame.copy()
        df.iloc[-1, 0] = 10_000

        capped = winsorize_outliers(df, z_thresh=2.5)

        assert capped.iloc[-1, 0] < df.iloc[-1, 0]

    def test_causal_transform_is_unchanged_when_future_changes(self):
        dates = pd.bdate_range("2022-12-01", periods=50)
        base = pd.DataFrame(
            {
                "AAPL": np.linspace(-0.01, 0.01, len(dates)),
                "MSFT": np.linspace(0.02, -0.02, len(dates)),
            },
            index=dates,
        )
        shocked = base.copy()
        test_start = pd.Timestamp("2023-01-02")
        shocked.loc[shocked.index >= test_start, "AAPL"] = 1_000.0
        shocked.loc[shocked.index >= test_start, "MSFT"] = -1_000.0

        base_capped = winsorize_outliers(base, z_thresh=2.0)
        shocked_capped = winsorize_outliers(shocked, z_thresh=2.0)

        pd.testing.assert_frame_equal(
            base_capped.loc[base_capped.index < test_start],
            shocked_capped.loc[shocked_capped.index < test_start],
        )


class TestComputeReturns:
    def test_simple_returns(self, sample_price_frame):
        rets = compute_returns(sample_price_frame, kind="simple")
        expected = sample_price_frame["AAPL"].pct_change().iloc[-1]

        assert isinstance(rets, pd.DataFrame)
        assert rets.iloc[0].isna().all()
        assert np.isclose(rets["AAPL"].iloc[-1], expected)

    def test_log_returns(self, sample_price_frame):
        rets = compute_returns(sample_price_frame, kind="log")
        expected = np.log(sample_price_frame["AAPL"]).diff().iloc[-1]

        assert np.isclose(rets["AAPL"].iloc[-1], expected)

    def test_invalid_kind(self, sample_price_frame):
        with pytest.raises(ValueError, match="kind must be 'log' or 'simple'"):
            compute_returns(sample_price_frame, kind="foo")


class TestAddTechnicalFeatures:
    def test_features_structure(self, sample_price_frame):
        returns = compute_returns(sample_price_frame)
        feat_df = add_technical_features(
            prices=sample_price_frame,
            returns=returns,
            windows=(5, 10),
        )

        assert isinstance(feat_df, pd.DataFrame)
        assert isinstance(feat_df.columns, pd.MultiIndex)
        feature_levels = set(feat_df.columns.get_level_values(0))
        assert "sma_5" in feature_levels
        assert "momentum_5" in feature_levels
        assert "volatility_5" in feature_levels


class TestSaveProcessed:
    def test_save_creates_files(self, sample_price_frame, temp_data_dir):
        out_base = temp_data_dir / "prices"
        save_processed(sample_price_frame, out_base)

        parquet_path = out_base.with_suffix(".parquet")
        csv_path = out_base.with_suffix(".csv")

        assert parquet_path.exists()
        assert csv_path.exists()

        loaded_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        assert len(loaded_csv) == len(sample_price_frame)


class TestPreparePriceData:
    @patch("data_prep._fetch_prices_and_volumes")
    def test_pipeline_shapes(
        self,
        mock_fetch_combined,
        sample_multi_price_frame,
        sample_volume_frame,
    ):
        mock_fetch_combined.return_value = (sample_multi_price_frame, sample_volume_frame)

        prices, returns, features, volumes = prepare_price_data(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start="2023-01-01",
            end="2023-01-31",
            interval="1d",
        )

        assert isinstance(prices, pd.DataFrame)
        assert isinstance(returns, pd.DataFrame)
        assert isinstance(features, pd.DataFrame)
        assert isinstance(volumes, pd.DataFrame)
        assert not prices.empty
        assert not returns.empty
        assert not features.empty
        assert not volumes.empty
        assert prices.index.equals(returns.index)
        assert prices.index.equals(volumes.index)
        mock_fetch_combined.assert_called_once()

    @patch("data_prep._fetch_prices_and_volumes")
    def test_pipeline_winsorization_toggle(
        self,
        mock_fetch_combined,
        sample_multi_price_frame,
        sample_volume_frame,
    ):
        mock_fetch_combined.return_value = (sample_multi_price_frame, sample_volume_frame)

        prices, returns, features, volumes = prepare_price_data(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start="2023-01-01",
            end="2023-01-31",
            interval="1d",
            winsorize_z=None,
        )

        assert not prices.empty
        assert not returns.empty
        assert not features.empty
        assert not volumes.empty

    @patch("data_prep.fetch_volumes_yfinance")
    @patch("data_prep.fetch_prices_yfinance")
    @patch("data_prep._fetch_prices_and_volumes")
    def test_pipeline_uses_combined_fetch_without_separate_network_calls(
        self,
        mock_fetch_combined,
        mock_fetch_prices,
        mock_fetch_volumes,
        sample_multi_price_frame,
        sample_volume_frame,
    ):
        mock_fetch_combined.return_value = (sample_multi_price_frame, sample_volume_frame)

        prepare_price_data(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start="2023-01-01",
            end="2023-01-31",
            interval="1d",
        )

        mock_fetch_combined.assert_called_once()
        mock_fetch_prices.assert_not_called()
        mock_fetch_volumes.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
