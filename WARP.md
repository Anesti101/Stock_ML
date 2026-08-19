# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Stock_ML is a machine learning project for stock market trend prediction using technical indicators, classification models, and time-series forecasting. The project focuses on:
- Classifying trend direction (up/down) using Logistic Regression, Random Forest, and XGBoost
- Forecasting stock prices using ARIMA, Prophet, and LSTM
- Analyzing market volatility and risk characteristics
- Providing interactive dashboards for exploration

**Important**: This is research/educational code, not financial advice.

## Project Architecture

### Current Implementation Status
The project has core functionality implemented:
- **src/data_prep.py**: Complete data fetching, validation, and I/O utilities:
  - `fetch_data_yfinance()` - Download historical stock data from Yahoo Finance
  - `validate_stock_data()` - Validate DataFrame structure and content
  - `save_stock_data()` / `load_stock_data()` - Save/load data in CSV, Parquet, or Pickle formats
- **Dashboard/dashboard.py**: Full-featured interactive dashboard with:
  - Candlestick charts, technical indicators (SMA, RSI, Bollinger Bands)
  - Volume analysis and summary statistics
  - Dash web application on port 8050
- **Tests/**: Comprehensive unit tests:
  - `test_data_prep.py` - 20+ tests for data operations (95%+ coverage)
  - `test_dashboard.py` - 25+ tests for dashboard functions and indicators
- **Notebooks**: Two working notebooks exist (`Linear regression.ipynb`, `Untitled.ipynb`) for exploratory analysis
- **Data directories**: Structured with `raw/`, `processed/`, and `external/` subdirectories (currently empty)
- **Reports/**: Directory exists but contains no implementation yet

### Planned Architecture (from README)
```
src/
├── data/          # Data loading and preprocessing modules
├── features/      # Technical indicator calculations (SMA, EMA, RSI, MACD, Bollinger Bands)
├── models/        # ML model implementations (classification & forecasting)
├── visualisation/ # Plotting and charting utilities
└── utils/         # Helper functions and configuration
```

**Current gap**: Most planned modules are not yet implemented. Only `data_prep.py` exists.

## Development Commands

### Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt
```

Core dependencies include:
- `yfinance` (Yahoo Finance API)
- `pandas`, `numpy` (data manipulation)
- `scikit-learn` (ML models)
- `xgboost` (gradient boosting)
- `statsmodels` (ARIMA)
- `prophet` (time-series forecasting)
- `plotly`/`dash` (dashboard)
- `pytest`, `pytest-cov`, `pytest-mock` (testing)

### Running Tests
```powershell
# Run all tests
pytest Tests/ -v

# Run specific test file
pytest Tests/test_data_prep.py -v
pytest Tests/test_dashboard.py -v

# Run with coverage report
pytest Tests/ --cov=src --cov=Dashboard --cov-report=html

# Run tests by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

### Interactive Dashboard
```powershell
# Launch dashboard
python Dashboard/dashboard.py
# Navigate to http://localhost:8050
```

The dashboard provides:
- Real-time stock data fetching for any ticker
- Candlestick charts with OHLCV data
- Technical indicators: SMA (5, 10, 20, 50), Bollinger Bands, RSI
- Volume analysis with color-coded bars
- Summary statistics: current price, change %, high/low, volatility

### Jupyter Notebooks
```powershell
# Launch Jupyter
jupyter notebook

# Working notebooks in .ipynb_checkpoints/:
# - Linear regression.ipynb
# - Untitled.ipynb
```

## Key Technical Considerations

### Time-Series Data Handling
- **Walk-forward validation**: Use temporal splits to prevent data leakage (future data must not influence past predictions)
- **Avoid lookahead bias**: Technical indicators should only use historical data available at prediction time
- When splitting data, use chronological order: train on earlier periods, validate/test on later periods

### Feature Engineering Pipeline
Technical indicators to implement:
- **Moving Averages**: SMA (5, 10, 20, 50-day), EMA (12, 26-day)
- **Momentum**: RSI (14-day), MACD (with signal line and histogram)
- **Volatility**: Bollinger Bands, rolling standard deviation
- **Returns**: Daily, weekly, monthly percentage changes

### Model Evaluation Strategy
- **Classification metrics**: Accuracy, Balanced Accuracy, F1-score, Precision, Recall
- **Forecasting metrics**: RMSE, MAE (compare against naïve baseline)
- **Financial metrics**: Sharpe ratio, maximum drawdown
- Always include baseline comparison (e.g., "predict yesterday's price" for forecasting)

### Data Sources
- Primary: Yahoo Finance via `yfinance` library
- Data format: OHLCV (Open, High, Low, Close, Volume) + Adjusted Close
- Typical period: 2+ years for training, recent months for validation

## Code Style & Conventions

- **Language**: Python 3.x
- **Style**: PEP 8 with British spelling in comments and documentation
- **Type hints**: Use where appropriate (see `data_prep.py` example with `from __future__ import annotations`)
- **Docstrings**: Include for all public functions and classes
- **Logging**: Use `logging` module for structured output (already imported in `data_prep.py`)

## Development Workflow

1. **Data Collection**: Use `fetch_data_yfinance()` or develop enhanced data loaders
2. **Feature Engineering**: Build technical indicator calculation modules in `src/features/`
3. **Model Development**: Implement classifiers and forecasting models in `src/models/`
4. **Validation**: Create proper time-series cross-validation in notebooks
5. **Visualization**: Build dashboard components once models are trained
6. **Testing**: Add unit tests for data validation and model functionality

## Testing Philosophy

- **Comprehensive coverage**: 95%+ coverage for core modules
- **Mock external dependencies**: Use `@patch` for yfinance API calls to avoid network dependencies
- **Fixtures for test data**: Reusable test data via pytest fixtures
- **Test isolation**: Each test is independent and can run in any order
- **Edge case testing**: Empty data, invalid inputs, boundary conditions
- **Integration tests**: Test complete workflows (fetch → validate → save → load)

## Important Notes

- The actual notebook files are stored in `.ipynb_checkpoints/` directory (unusual structure)
- British spelling convention: "visualisation" not "visualization", "analyse" not "analyze"
- Dashboard uses mock data fallback if yfinance is unavailable
- All tests use mocked API calls to avoid rate limits and network dependencies
