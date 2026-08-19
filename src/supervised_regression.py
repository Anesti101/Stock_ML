from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TARGET_COLUMN = "future_return_20d"
TARGET_END_COLUMN = "target_end_date"
NEXT_DAY_RETURN_COLUMN = "next_day_return"


@dataclass
class SupervisedDataset:
    """Long stock-date panel split into purged train and untouched test sets."""

    panel: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_column: str = TARGET_COLUMN
    target_end_column: str = TARGET_END_COLUMN
    train_next_day_returns: pd.Series | None = None
    test_next_day_returns: pd.Series | None = None

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train[self.feature_columns]

    @property
    def y_train(self) -> pd.Series:
        return self.train[self.target_column]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test[self.feature_columns]

    @property
    def y_test(self) -> pd.Series:
        return self.test[self.target_column]

    @property
    def next_day_train(self) -> pd.Series | None:
        return self.train_next_day_returns

    @property
    def next_day_test(self) -> pd.Series | None:
        return self.test_next_day_returns


@dataclass
class PredictionResult:
    name: str
    train_predictions: pd.Series
    test_predictions: pd.Series
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    estimator: object | None = None
    selected_params: dict[str, object] | None = None
    cv_results: pd.DataFrame | None = None


def make_forward_log_returns(prices: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Return log(price[t+horizon]) - log(price[t]) for each stock/date.

    The final `horizon` rows are NaN because the realised future return is not
    yet observable inside the supplied price history.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    log_prices = np.log(prices.astype(float))
    return log_prices.shift(-horizon) - log_prices


def make_target_end_dates(index: Iterable[pd.Timestamp], horizon: int = 20) -> pd.Series:
    """Map each prediction date to the date whose close finishes its target."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    dates = pd.DatetimeIndex(index)
    end_dates = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    if len(dates) > horizon:
        end_dates.iloc[:-horizon] = dates[horizon:]
    end_dates.name = TARGET_END_COLUMN
    return end_dates


def _stack_wide(frame: pd.DataFrame, name: str) -> pd.Series:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)

    index = pd.MultiIndex.from_product(
        [frame.index, frame.columns],
        names=["date", "ticker"],
    )
    values = frame.to_numpy().reshape(-1)
    return pd.Series(values, index=index, name=name)


def _row_zscore(frame: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    mu = frame.mean(axis=1)
    sigma = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mu, axis=0).div(sigma + eps, axis=0)


def _technical_features_to_panel(features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame()

    if not isinstance(features.columns, pd.MultiIndex):
        return _stack_wide(features, "technical_feature").to_frame()

    pieces: list[pd.Series] = []
    for feature_name in features.columns.get_level_values(0).unique():
        feature_frame = features.xs(feature_name, axis=1, level=0)
        pieces.append(_stack_wide(feature_frame, str(feature_name)))

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1)


def _rolling_corr_to_market(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = max(10, window // 2)
    market_return = returns.mean(axis=1)
    return returns.apply(
        lambda col: col.rolling(window, min_periods=min_periods).corr(market_return)
    )


def _regime_feature(
    prices: pd.DataFrame,
    ma_short: int = 50,
    ma_long: int = 200,
) -> pd.DataFrame:
    market_price = prices.mean(axis=1)
    short_ma = market_price.rolling(ma_short, min_periods=ma_short).mean()
    long_ma = market_price.rolling(ma_long, min_periods=ma_long).mean()
    bullish = (short_ma > long_ma).fillna(False).astype(float)
    return pd.DataFrame(
        np.repeat(bullish.to_numpy()[:, None], len(prices.columns), axis=1),
        index=prices.index,
        columns=prices.columns,
    )


def build_supervised_feature_panel(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    technical_features: pd.DataFrame | None,
    gravity_signals: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    *,
    horizon: int = 20,
    market_corr_window: int = 60,
) -> pd.DataFrame:
    """
    Build a stock-date supervised learning panel from existing project outputs.

    All feature transforms are rolling, contemporaneous, or lagged-only. The only
    negative shift is in `future_return_20d`, which is explicitly excluded from
    the feature list during training.
    """
    prices = prices.sort_index()
    returns = returns.reindex_like(prices)
    gravity_signals = gravity_signals.reindex_like(prices)

    feature_pieces: list[pd.Series | pd.DataFrame] = [
        _technical_features_to_panel(technical_features),
        _stack_wide(gravity_signals, "gravity_signal"),
        _stack_wide(gravity_signals.rank(axis=1, pct=True), "gravity_rank"),
    ]

    log_prices = np.log(prices.astype(float))
    for window in (20, 60):
        feature_pieces.append(
            _stack_wide(log_prices.diff(window), f"log_momentum_{window}")
        )
        feature_pieces.append(
            _stack_wide(
                returns.rolling(window, min_periods=max(5, window // 2)).std(),
                f"rolling_volatility_{window}",
            )
        )

    corr_to_market = _rolling_corr_to_market(
        returns, window=market_corr_window, min_periods=max(10, market_corr_window // 2)
    )
    corr_distance = np.sqrt(2.0 * (1.0 - corr_to_market.clip(-0.999999, 0.999999)))
    feature_pieces.extend(
        [
            _stack_wide(corr_to_market, f"corr_to_market_{market_corr_window}"),
            _stack_wide(corr_distance, f"corr_distance_to_market_{market_corr_window}"),
            _stack_wide(_regime_feature(prices), "bullish_regime"),
        ]
    )

    if volumes is not None and not volumes.empty and volumes.notna().any().any():
        aligned_volumes = volumes.reindex_like(prices)
        dollar_volume = prices * aligned_volumes
        rolling_liquidity = dollar_volume.rolling(20, min_periods=5).mean()
        log_liquidity = np.log(rolling_liquidity.replace(0.0, np.nan))
        feature_pieces.append(_stack_wide(_row_zscore(log_liquidity), "liquidity_z_20"))

    target = make_forward_log_returns(prices, horizon=horizon)
    target_long = _stack_wide(target, TARGET_COLUMN)
    target_end_by_date = make_target_end_dates(prices.index, horizon=horizon).to_dict()
    target_end = pd.Series(
        [target_end_by_date.get(dt, pd.NaT) for dt in target_long.index.get_level_values("date")],
        index=target_long.index,
        name=TARGET_END_COLUMN,
        dtype="datetime64[ns]",
    )

    panel = pd.concat(feature_pieces + [target_long, target_end], axis=1)
    numeric_columns = [col for col in panel.columns if col != TARGET_END_COLUMN]
    panel[numeric_columns] = panel[numeric_columns].replace([np.inf, -np.inf], np.nan)

    feature_columns = [
        col
        for col in numeric_columns
        if col != TARGET_COLUMN and panel[col].notna().any()
    ]
    return panel[feature_columns + [TARGET_COLUMN, TARGET_END_COLUMN]].sort_index()


def infer_feature_columns(
    panel: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    target_end_column: str = TARGET_END_COLUMN,
    next_day_return_column: str = NEXT_DAY_RETURN_COLUMN,
) -> list[str]:
    return [
        col
        for col in panel.columns
        if col not in {target_column, target_end_column, next_day_return_column}
        and panel[col].notna().any()
    ]


def chronological_train_test_split(
    panel: pd.DataFrame,
    *,
    train_start: str = "2020-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2024-12-31",
    feature_columns: Sequence[str] | None = None,
    target_column: str = TARGET_COLUMN,
    target_end_column: str = TARGET_END_COLUMN,
    purge_label_overlap: bool = True,
) -> SupervisedDataset:
    """
    Split by prediction date and purge labels that cross split boundaries.

    Purging means a train row dated 2022-12-20 is dropped for a 20-day horizon,
    because its realised target would use January 2023 test-period prices.
    """
    if not isinstance(panel.index, pd.MultiIndex) or "date" not in panel.index.names:
        raise ValueError("panel must use a MultiIndex with a 'date' level")

    if feature_columns is None:
        feature_columns = infer_feature_columns(panel, target_column, target_end_column)
    feature_columns = list(feature_columns)

    required_columns = feature_columns + [target_column, target_end_column]
    missing = [col for col in required_columns if col not in panel.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean = panel.copy()
    numeric_required_columns = [
        col for col in required_columns if col != target_end_column
    ]
    clean[numeric_required_columns] = clean[numeric_required_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    clean = clean.dropna(subset=required_columns)

    prediction_dates = pd.DatetimeIndex(clean.index.get_level_values("date"))
    target_end_dates = pd.to_datetime(clean[target_end_column]).to_numpy()

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    train_mask = (prediction_dates >= train_start_ts) & (prediction_dates <= train_end_ts)
    test_mask = (prediction_dates >= test_start_ts) & (prediction_dates <= test_end_ts)

    if purge_label_overlap:
        train_mask &= target_end_dates <= np.datetime64(train_end_ts)
        test_mask &= target_end_dates <= np.datetime64(test_end_ts)

    train = clean.loc[train_mask]
    test = clean.loc[test_mask]

    if train.empty:
        raise ValueError("Training split is empty after feature/target cleaning")
    if test.empty:
        raise ValueError("Test split is empty after feature/target cleaning")

    return SupervisedDataset(
        panel=clean,
        train=train,
        test=test,
        feature_columns=feature_columns,
        target_column=target_column,
        target_end_column=target_end_column,
    )


def build_supervised_dataset(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    technical_features: pd.DataFrame | None,
    gravity_signals: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    *,
    horizon: int = 20,
    train_start: str = "2020-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2024-12-31",
) -> SupervisedDataset:
    panel = build_supervised_feature_panel(
        prices=prices,
        returns=returns,
        technical_features=technical_features,
        gravity_signals=gravity_signals,
        volumes=volumes,
        horizon=horizon,
    )
    feature_columns = infer_feature_columns(panel)
    dataset = chronological_train_test_split(
        panel,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        feature_columns=feature_columns,
        purge_label_overlap=True,
    )
    next_day_returns = _stack_wide(
        returns.reindex_like(prices).shift(-1),
        NEXT_DAY_RETURN_COLUMN,
    )
    dataset.train_next_day_returns = next_day_returns.reindex(dataset.train.index)
    dataset.test_next_day_returns = next_day_returns.reindex(dataset.test.index)
    return dataset


def purged_expanding_time_series_splits(
    index: pd.MultiIndex,
    target_end_dates: pd.Series,
    *,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Date-grouped expanding splits with label-overlap purging."""
    if "date" not in index.names:
        raise ValueError("index must contain a 'date' level")
    dates = pd.DatetimeIndex(index.get_level_values("date"))
    unique_dates = pd.DatetimeIndex(pd.unique(dates)).sort_values()
    if len(unique_dates) < n_splits + 2:
        return []

    fold_size = len(unique_dates) // (n_splits + 1)
    if fold_size == 0:
        return []

    target_end = pd.to_datetime(target_end_dates).to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        val_start_pos = fold_size * (fold + 1)
        val_end_pos = min(fold_size * (fold + 2) - 1, len(unique_dates) - 1)
        if val_start_pos >= len(unique_dates):
            break

        train_end_date = unique_dates[val_start_pos - 1]
        val_start_date = unique_dates[val_start_pos]
        val_end_date = unique_dates[val_end_pos]

        train_mask = (dates <= train_end_date) & (target_end < np.datetime64(val_start_date))
        val_mask = (
            (dates >= val_start_date)
            & (dates <= val_end_date)
            & (target_end <= np.datetime64(val_end_date))
        )

        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))

    return splits


def _valid_pair_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    paired = pd.concat([y_true.rename("actual"), y_pred.rename("predicted")], axis=1)
    return paired.replace([np.inf, -np.inf], np.nan).dropna()


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    paired = _valid_pair_frame(y_true, y_pred)
    if paired.empty:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    actual = paired["actual"].to_numpy(dtype=float)
    predicted = paired["predicted"].to_numpy(dtype=float)
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(actual - actual.mean())))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def daily_information_coefficient(
    scores: pd.Series,
    realised_returns: pd.Series,
    *,
    method: str = "spearman",
) -> pd.Series:
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

    paired = _valid_pair_frame(realised_returns, scores)
    if paired.empty:
        return pd.Series(dtype=float, name=f"IC_{method}")

    ic_values: list[float] = []
    ic_dates: list[pd.Timestamp] = []
    for dt, group in paired.groupby(level="date"):
        if len(group) < 3:
            continue
        actual = group["actual"]
        predicted = group["predicted"]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (
                predicted.rank().corr(actual.rank())
                if method == "spearman"
                else predicted.corr(actual)
            )
        if pd.notna(corr):
            ic_values.append(float(corr))
            ic_dates.append(dt)

    return pd.Series(ic_values, index=ic_dates, name=f"IC_{method}")


def ranked_long_short_forward_returns(
    scores: pd.Series,
    realised_returns: pd.Series,
    *,
    top_pct: float = 0.2,
) -> pd.Series:
    if not 0.0 < top_pct < 0.5:
        raise ValueError("top_pct must be in (0, 0.5)")

    paired = _valid_pair_frame(realised_returns, scores)
    returns: list[float] = []
    dates: list[pd.Timestamp] = []

    for dt, group in paired.groupby(level="date"):
        if len(group) < 4:
            continue
        k = max(1, int(np.floor(len(group) * top_pct)))
        ranked = group.sort_values("predicted")
        short_leg = ranked.iloc[:k]["actual"].mean()
        long_leg = ranked.iloc[-k:]["actual"].mean()
        returns.append(float(0.5 * long_leg - 0.5 * short_leg))
        dates.append(dt)

    return pd.Series(returns, index=dates, name="ranked_long_short_forward_return")


def ranked_long_short_next_day_returns(
    scores: pd.Series,
    next_day_returns: pd.Series,
    *,
    top_pct: float = 0.2,
) -> pd.Series:
    """Build daily long-short returns from each date's predicted ranking."""
    if not 0.0 < top_pct < 0.5:
        raise ValueError("top_pct must be in (0, 0.5)")

    paired = _valid_pair_frame(next_day_returns, scores)
    returns: list[float] = []
    dates: list[pd.Timestamp] = []

    for dt, group in paired.groupby(level="date"):
        if len(group) < 4:
            continue
        if group["predicted"].nunique(dropna=True) < 2:
            returns.append(0.0)
            dates.append(dt)
            continue

        k = max(1, int(np.floor(len(group) * top_pct)))
        ranked = group.sort_values("predicted")
        short_leg = ranked.iloc[:k]["actual"].mean()
        long_leg = ranked.iloc[-k:]["actual"].mean()
        returns.append(float(0.5 * long_leg - 0.5 * short_leg))
        dates.append(dt)

    return pd.Series(returns, index=dates, name="ranked_long_short_next_day_return")


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    next_day_returns: pd.Series | None = None,
    horizon: int = 20,
    top_pct: float = 0.2,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    ic = daily_information_coefficient(y_pred, y_true, method="spearman")
    if next_day_returns is None:
        portfolio_returns = pd.Series(dtype=float, name="ranked_long_short_next_day_return")
    else:
        portfolio_returns = ranked_long_short_next_day_returns(
            y_pred,
            next_day_returns,
            top_pct=top_pct,
        )

    mean_ic = float(ic.mean()) if len(ic) else np.nan
    icir = float(mean_ic / (ic.std() + 1e-12)) if len(ic) else np.nan
    mean_port = float(portfolio_returns.mean()) if len(portfolio_returns) else np.nan
    vol_port = float(portfolio_returns.std()) if len(portfolio_returns) else np.nan
    sharpe = (
        float(mean_port / (vol_port + 1e-12) * np.sqrt(trading_days_per_year))
        if len(portfolio_returns)
        else np.nan
    )

    metrics.update(
        {
            "mean_ic": mean_ic,
            "icir": icir,
            "rank_portfolio_sharpe": sharpe,
            "rank_portfolio_mean_daily_return": mean_port,
            "n_portfolio_days": float(len(portfolio_returns)),
            "n_obs": float(len(_valid_pair_frame(y_true, y_pred))),
            "n_ic_dates": float(len(ic)),
        }
    )
    return metrics


def fit_gravity_baseline(
    dataset: SupervisedDataset,
    *,
    signal_column: str = "gravity_signal",
    horizon: int = 20,
) -> PredictionResult:
    """
    Calibrate the raw gravity signal to return units using train data only.

    Ranking metrics still use the same calibrated monotonic score, so this is a
    fair baseline for MAE/RMSE/R2 without peeking at the test target.
    """
    if signal_column not in dataset.feature_columns:
        raise KeyError(f"{signal_column!r} is not available in the feature panel")

    train_pair = _valid_pair_frame(dataset.y_train, dataset.train[signal_column])
    x = train_pair["predicted"].to_numpy(dtype=float)
    y = train_pair["actual"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]

    train_pred = pd.Series(
        intercept + slope * dataset.train[signal_column],
        index=dataset.train.index,
        name="gravity_baseline",
    )
    test_pred = pd.Series(
        intercept + slope * dataset.test[signal_column],
        index=dataset.test.index,
        name="gravity_baseline",
    )

    return PredictionResult(
        name="gravity_baseline",
        train_predictions=train_pred,
        test_predictions=test_pred,
        train_metrics=evaluate_predictions(
            dataset.y_train,
            train_pred,
            next_day_returns=dataset.next_day_train,
            horizon=horizon,
        ),
        test_metrics=evaluate_predictions(
            dataset.y_test,
            test_pred,
            next_day_returns=dataset.next_day_test,
            horizon=horizon,
        ),
        estimator={"intercept": float(intercept), "slope": float(slope)},
        selected_params={"signal_column": signal_column},
    )


def _make_ridge_pipeline(alpha: float):
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "scikit-learn is required for supervised Ridge/RandomForest training. "
            "Install scikit-learn, and optionally xgboost for the tree model."
        ) from exc

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _make_tree_pipeline(random_state: int = 42):
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "scikit-learn is required for supervised Ridge/RandomForest training. "
            "Install scikit-learn, and optionally xgboost for the tree model."
        ) from exc

    try:  # pragma: no cover - xgboost availability is environment-specific
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        name = "xgboost_regressor"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
        name = "random_forest_regressor"

    return name, Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def select_ridge_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    target_end_dates: pd.Series,
    *,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    n_splits: int = 3,
) -> tuple[float, pd.DataFrame]:
    """Choose Ridge alpha using purged expanding validation inside train data."""
    splits = purged_expanding_time_series_splits(
        X.index, target_end_dates, n_splits=n_splits
    )
    if not splits:
        return float(alphas[0]), pd.DataFrame()

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        fold_rmses: list[float] = []
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            model = _make_ridge_pipeline(alpha=float(alpha))
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pd.Series(model.predict(X.iloc[val_idx]), index=X.iloc[val_idx].index)
            rmse = regression_metrics(y.iloc[val_idx], pred)["rmse"]
            fold_rmses.append(rmse)
            rows.append({"alpha": float(alpha), "fold": float(fold_id), "rmse": rmse})

        rows.append(
            {
                "alpha": float(alpha),
                "fold": 0.0,
                "rmse": float(np.nanmean(fold_rmses)),
            }
        )

    cv_results = pd.DataFrame(rows)
    mean_rows = cv_results[cv_results["fold"] == 0.0].sort_values(["rmse", "alpha"])
    selected_alpha = float(mean_rows.iloc[0]["alpha"])
    return selected_alpha, cv_results


def fit_supervised_models(
    dataset: SupervisedDataset,
    *,
    horizon: int = 20,
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    n_splits: int = 3,
    random_state: int = 42,
) -> list[PredictionResult]:
    """
    Fit Ridge plus one tree model. Test data is used only for final evaluation.
    """
    selected_alpha, cv_results = select_ridge_alpha(
        dataset.X_train,
        dataset.y_train,
        dataset.train[dataset.target_end_column],
        alphas=ridge_alphas,
        n_splits=n_splits,
    )

    ridge = _make_ridge_pipeline(alpha=selected_alpha)
    ridge.fit(dataset.X_train, dataset.y_train)

    tree_name, tree = _make_tree_pipeline(random_state=random_state)
    tree.fit(dataset.X_train, dataset.y_train)

    fitted = [
        ("ridge_regression", ridge, {"alpha": selected_alpha}, cv_results),
        (tree_name, tree, {}, None),
    ]

    results: list[PredictionResult] = []
    for name, estimator, selected_params, cv_frame in fitted:
        train_pred = pd.Series(
            estimator.predict(dataset.X_train), index=dataset.train.index, name=name
        )
        test_pred = pd.Series(
            estimator.predict(dataset.X_test), index=dataset.test.index, name=name
        )
        results.append(
            PredictionResult(
                name=name,
                train_predictions=train_pred,
                test_predictions=test_pred,
                train_metrics=evaluate_predictions(
                    dataset.y_train,
                    train_pred,
                    next_day_returns=dataset.next_day_train,
                    horizon=horizon,
                ),
                test_metrics=evaluate_predictions(
                    dataset.y_test,
                    test_pred,
                    next_day_returns=dataset.next_day_test,
                    horizon=horizon,
                ),
                estimator=estimator,
                selected_params=selected_params,
                cv_results=cv_frame,
            )
        )

    return results


def performance_table(results: Sequence[PredictionResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_order = [
        "mae",
        "rmse",
        "r2",
        "mean_ic",
        "icir",
        "rank_portfolio_sharpe",
        "rank_portfolio_mean_daily_return",
        "n_portfolio_days",
        "n_obs",
        "n_ic_dates",
    ]
    for result in results:
        for split, metrics in (
            ("train", result.train_metrics),
            ("test", result.test_metrics),
        ):
            row: dict[str, object] = {"model": result.name, "split": split}
            row.update({metric: metrics.get(metric, np.nan) for metric in metric_order})
            rows.append(row)
    return pd.DataFrame(rows)
