import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_prep import add_technical_features, compute_returns  # noqa: E402
from supervised_regression import (  # noqa: E402
    TARGET_COLUMN,
    TARGET_END_COLUMN,
    build_supervised_dataset,
    build_supervised_feature_panel,
    evaluate_predictions,
    make_forward_log_returns,
    purged_expanding_time_series_splits,
)


def _sample_prices(periods: int = 330) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    t = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "AAA": 100.0 * np.exp(0.0010 * t + 0.010 * np.sin(t / 11.0)),
            "BBB": 80.0 * np.exp(0.0006 * t + 0.012 * np.cos(t / 13.0)),
            "CCC": 120.0 * np.exp(0.0002 * t + 0.008 * np.sin(t / 7.0)),
            "DDD": 90.0 * np.exp(0.0008 * t + 0.009 * np.cos(t / 17.0)),
        },
        index=dates,
    )


def _sample_inputs(periods: int = 330):
    prices = _sample_prices(periods)
    returns = compute_returns(prices, kind="log")
    features = add_technical_features(prices, returns, windows=(5, 20, 60))
    gravity = returns.rolling(20, min_periods=5).sum()
    return prices, returns, features, gravity


def test_make_forward_log_returns_uses_exact_horizon():
    dates = pd.bdate_range("2023-01-02", periods=12)
    prices = pd.DataFrame({"AAA": np.exp(np.arange(12, dtype=float))}, index=dates)

    target = make_forward_log_returns(prices, horizon=5)

    assert np.isclose(target.loc[dates[0], "AAA"], 5.0)
    assert np.isclose(target.loc[dates[3], "AAA"], 5.0)
    assert target.tail(5)["AAA"].isna().all()


def test_supervised_split_purges_training_labels_crossing_test_start():
    prices, returns, features, gravity = _sample_inputs()

    dataset = build_supervised_dataset(
        prices=prices,
        returns=returns,
        technical_features=features,
        gravity_signals=gravity,
        horizon=20,
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-03-31",
    )

    assert TARGET_COLUMN not in dataset.feature_columns
    assert TARGET_END_COLUMN not in dataset.feature_columns
    assert dataset.train[TARGET_END_COLUMN].max() <= pd.Timestamp("2020-12-31")
    assert dataset.test.index.get_level_values("date").min() >= pd.Timestamp("2021-01-01")
    assert dataset.test[TARGET_END_COLUMN].max() <= pd.Timestamp("2021-03-31")


def test_feature_panel_is_asof_safe_when_future_prices_change():
    prices, returns, features, gravity = _sample_inputs(periods=260)
    asof_date = prices.index[150]

    base_panel = build_supervised_feature_panel(
        prices=prices,
        returns=returns,
        technical_features=features,
        gravity_signals=gravity,
        horizon=20,
    )

    shocked_prices = prices.copy()
    shocked_prices.loc[shocked_prices.index > asof_date] *= 3.0
    shocked_returns = compute_returns(shocked_prices, kind="log")
    shocked_features = add_technical_features(
        shocked_prices, shocked_returns, windows=(5, 20, 60)
    )
    shocked_gravity = shocked_returns.rolling(20, min_periods=5).sum()

    shocked_panel = build_supervised_feature_panel(
        prices=shocked_prices,
        returns=shocked_returns,
        technical_features=shocked_features,
        gravity_signals=shocked_gravity,
        horizon=20,
    )

    feature_columns = [
        col for col in base_panel.columns if col not in {TARGET_COLUMN, TARGET_END_COLUMN}
    ]
    base_row = base_panel.loc[(asof_date, slice(None)), feature_columns]
    shocked_row = shocked_panel.loc[(asof_date, slice(None)), feature_columns]

    pd.testing.assert_frame_equal(base_row, shocked_row)


def test_purged_expanding_splits_do_not_overlap_validation_labels():
    prices, returns, features, gravity = _sample_inputs()
    dataset = build_supervised_dataset(
        prices=prices,
        returns=returns,
        technical_features=features,
        gravity_signals=gravity,
        horizon=20,
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-03-31",
    )

    splits = purged_expanding_time_series_splits(
        dataset.train.index,
        dataset.train[TARGET_END_COLUMN],
        n_splits=3,
    )

    assert splits
    for train_idx, val_idx in splits:
        train_fold = dataset.train.iloc[train_idx]
        val_fold = dataset.train.iloc[val_idx]
        val_start = val_fold.index.get_level_values("date").min()

        assert train_fold.index.get_level_values("date").max() < val_start
        assert train_fold[TARGET_END_COLUMN].max() < val_start


def test_evaluate_predictions_uses_next_day_returns_for_portfolio_sharpe():
    dates = pd.bdate_range("2023-01-02", periods=4)
    tickers = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    y_pred = pd.Series(
        np.tile([0.1, 0.2, 0.3, 0.4], len(dates)),
        index=index,
        name="prediction",
    )
    y_true = pd.Series(
        np.tile([1.0, -1.0, 1.0, -1.0], len(dates)),
        index=index,
        name=TARGET_COLUMN,
    )
    next_day_returns = pd.Series(
        [
            0.00,
            0.01,
            0.02,
            0.04,
            0.01,
            0.00,
            0.01,
            0.02,
            0.02,
            0.01,
            0.00,
            -0.01,
            -0.01,
            0.00,
            0.01,
            0.03,
        ],
        index=index,
        name="next_day_return",
    )

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        next_day_returns=next_day_returns,
        top_pct=0.25,
    )
    expected_daily_returns = pd.Series([0.02, 0.005, -0.015, 0.02])
    expected_sharpe = (
        expected_daily_returns.mean() / expected_daily_returns.std() * np.sqrt(252)
    )

    assert np.isclose(
        metrics["rank_portfolio_mean_daily_return"],
        expected_daily_returns.mean(),
    )
    assert np.isclose(metrics["rank_portfolio_sharpe"], expected_sharpe)
