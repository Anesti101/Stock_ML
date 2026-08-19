# Stock_ML

Stock_ML is a quantitative research project for cross-sectional equity return prediction. The core idea is a gravity-inspired signal: liquid, highly correlated stocks exert stronger "pull", and momentum gives that pull a direction. The project now also uses that signal system as a supervised regression feature generator.

This project predicts **20-trading-day forward log returns**, not exact future stock prices.

## What The Project Does

- Downloads adjusted close prices and volumes from Yahoo Finance.
- Cleans and aligns multi-stock price panels on a business-day calendar.
- Builds technical features such as SMA, momentum, and rolling volatility.
- Generates gravity-inspired cross-sectional stock signals.
- Evaluates the gravity signal with Information Coefficient and long-short portfolio metrics.
- Builds a supervised stock-date dataset for 20-day forward return regression.
- Trains Ridge Regression plus a tree-based regressor: XGBoost when installed, otherwise RandomForestRegressor from scikit-learn.
- Compares ML models against a train-calibrated gravity-signal baseline on an untouched 2023-2024 test set.

## Project Structure

```text
Stock_ML/
|-- src/
|   |-- data_prep.py              # yfinance download, cleaning, returns, technical features
|   |-- gravity_model.py          # gravity-inspired masses, distances, forces, signals, IC
|   |-- signal_pipeline.py        # volatility scaling, ranking, regime filter, portfolios
|   |-- supervised_regression.py  # supervised dataset, leakage controls, ML training/evaluation
|   |-- eda.py                    # EDA summaries and plots
|   `-- main.py                   # end-to-end research script
|-- Dashboard/
|   `-- dashboard.py              # Dash/Plotly stock dashboard
|-- Tests/
|   |-- test_data_prep.py
|   |-- test_dashboard.py
|   `-- test_supervised_regression.py
`-- README.md
```

## Supervised Regression Component

Each supervised row represents one `(date, ticker)` prediction point. The target is:

```python
future_return_20d = log(price[t + 20]) - log(price[t])
```

Feature columns are built from data available at or before the prediction date:

- Raw gravity signal and cross-sectional gravity rank.
- Existing technical features from `add_technical_features`, including SMA, momentum, and rolling volatility.
- Additional log momentum and rolling volatility features.
- Correlation-to-market and correlation-distance-to-market features.
- Bullish/bearish market regime feature.
- Liquidity z-score from rolling dollar volume when volume data is available.

The target column and target end date metadata are never included as model features.

## Leakage Prevention

The supervised split is chronological and purged:

- Train prediction dates: 2020-01-01 through 2022-12-31.
- Test prediction dates: 2023-01-01 through 2024-12-31.
- Any train row whose 20-day target would finish after 2022-12-31 is dropped, so training labels do not use test-period returns.
- Test rows are also dropped when the 20-day realised target is unavailable inside the supplied data.
- Ridge alpha selection uses purged expanding time-series validation inside the training set only.
- The final 2023-2024 test set is used only once for final reporting.

## Models And Metrics

Baselines and models:

- `gravity_baseline`: raw gravity signal calibrated to return units using train data only.
- `ridge_regression`: Ridge with median imputation, standardisation, and train-only alpha selection.
- `xgboost_regressor` if XGBoost is installed.
- `random_forest_regressor` if XGBoost is not installed.

Final metrics:

- MAE
- RMSE
- R2
- Spearman rank correlation / Information Coefficient
- ICIR
- Ranked long-short portfolio Sharpe using predicted 20-day return rankings

## Running The Project

Install the core dependencies, then run:

```powershell
pip install -r requirements.txt
```

```powershell
python src/main.py
```

The script prints:

- Gravity configuration selected using train data only.
- Selected supervised features.
- Train/test matrix dimensions.
- Gravity baseline performance.
- Ridge and tree-model performance when `scikit-learn` is installed.

XGBoost is optional. If it is not installed, the code uses `RandomForestRegressor`.
`pyarrow` is included in `requirements.txt` so processed data can be saved as Parquet.

## Testing

Run the tests from the repository root:

```powershell
pytest Tests -q
```

The supervised tests check:

- Exact 20-day forward log-return construction.
- The target and metadata columns are excluded from model features.
- Train labels are purged when their 20-day target crosses into the test period.
- Expanding validation folds purge label overlap.
- Feature values at an as-of date do not change when only future prices are modified.

## Interview Framing

A concise way to describe this project:

> I built a cross-sectional equity research pipeline that starts with a physics-inspired gravity signal, then turns it into a supervised learning problem. Each stock-date row uses only information available at that date, and the model predicts 20-day forward log returns rather than prices. I compare Ridge and a tree ensemble against the original gravity signal on a purged chronological train/test split, using return-error metrics, rank IC, and portfolio Sharpe to evaluate whether ML adds value beyond the handcrafted quant signal.

## Disclaimer

This repository is for educational and research purposes only. It is not financial advice, and model outputs should not be used as the sole basis for investment decisions.
