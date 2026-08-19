from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signal_pipeline import combine_with_momentum, construct_portfolio  # noqa: E402


def test_combine_with_momentum_preserves_flat_bearish_rows():
    dates = pd.bdate_range("2023-01-02", periods=8)
    tickers = ["A", "B", "C", "D"]
    rank_signal = pd.DataFrame(
        np.tile([0.1, 0.4, 0.7, 1.0], (len(dates), 1)),
        index=dates,
        columns=tickers,
    )
    bearish_date = dates[4]
    rank_signal.loc[bearish_date] = 0.0
    returns = pd.DataFrame(
        np.arange(len(dates) * len(tickers), dtype=float).reshape(len(dates), len(tickers))
        / 1_000.0,
        index=dates,
        columns=tickers,
    )

    combined = combine_with_momentum(
        rank_signal,
        returns,
        momentum_weight=0.5,
        lookback=2,
    )

    assert combined.loc[bearish_date].eq(0.0).all()


def test_construct_portfolio_returns_flat_when_signal_has_no_dispersion():
    dates = pd.bdate_range("2023-01-02", periods=3)
    tickers = ["A", "B", "C", "D"]
    signal = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.4, 0.7, 1.0],
            [0.2, 0.3, 0.8, 0.9],
        ],
        index=dates,
        columns=tickers,
    )
    returns = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.20, -0.10, 0.15, -0.25],
            [0.01, 0.02, -0.01, -0.02],
        ],
        index=dates,
        columns=tickers,
    )

    portfolio_returns = construct_portfolio(signal, returns, top_pct=0.25)

    assert portfolio_returns.loc[dates[1]] == 0.0
