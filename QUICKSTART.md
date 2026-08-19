# Stock_ML Quick Start Guide

Get up and running with Stock_ML in 5 minutes!

## 1️⃣ Installation

```powershell
# Navigate to project directory
cd C:\Users\anest\OneDrive\Documents\ML\Stock_ML\Stock_ML

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 2️⃣ Verify Installation

```powershell
# Run tests to verify everything works
pytest Tests/ -v
```

Expected output:
```
Tests/test_data_prep.py::TestFetchDataYfinance::test_fetch_single_ticker_success PASSED
Tests/test_data_prep.py::TestFetchDataYfinance::test_fetch_multiple_tickers_success PASSED
...
========================= 45 passed in 2.34s =========================
```

## 3️⃣ Fetch Stock Data

Create a file `fetch_example.py`:

```python
from src.data_prep import fetch_data_yfinance, validate_stock_data, save_stock_data

# Fetch Apple stock data
data = fetch_data_yfinance('AAPL', '2023-01-01', '2024-01-01')

# Validate it
validate_stock_data(data)

# Save to CSV
save_stock_data(data, 'Data/raw/AAPL_2023.csv')

print(f"✅ Fetched {len(data)} rows of data")
print(data.head())
```

Run it:
```powershell
python fetch_example.py
```

## 4️⃣ Launch Dashboard

```powershell
python Dashboard/dashboard.py
```

Then open your browser to: **http://localhost:8050**

### Using the Dashboard:
1. Enter a ticker symbol (e.g., AAPL, GOOGL, MSFT)
2. Select a time period (1 month to 5 years)
3. Click "Load Data"
4. Explore the interactive charts!

**Available Charts:**
- 📈 Candlestick chart with OHLCV data
- 📊 Technical indicators (SMA 5/10/20/50, Bollinger Bands)
- 📉 RSI indicator with overbought/oversold levels
- 📊 Volume analysis with color-coded bars

## 5️⃣ Calculate Technical Indicators

Create `indicators_example.py`:

```python
from src.data_prep import fetch_data_yfinance
from Dashboard.dashboard import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_summary_stats
)

# Fetch data
data = fetch_data_yfinance('MSFT', '2023-01-01', '2024-01-01')

# Calculate indicators
data_with_ma = calculate_moving_averages(data)
rsi = calculate_rsi(data)
middle, upper, lower = calculate_bollinger_bands(data)

# Get summary stats
stats = calculate_summary_stats(data)

print("📊 Summary Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

print(f"\n📈 Latest RSI: {rsi.iloc[-1]:.2f}")
print(f"📈 Latest Close: ${data['Close'].iloc[-1]:.2f}")
print(f"📈 SMA 50: ${data_with_ma['SMA_50'].iloc[-1]:.2f}")
```

Run it:
```powershell
python indicators_example.py
```

## 📚 Next Steps

### Explore the Codebase
- `src/data_prep.py` - Data fetching and I/O utilities
- `Dashboard/dashboard.py` - Interactive visualization dashboard
- `Tests/` - Comprehensive unit tests

### Read Documentation
- `README.md` - Project overview and methodology
- `WARP.md` - Development guidelines for AI assistants
- `IMPLEMENTATION_SUMMARY.md` - What's been built
- `Tests/README.md` - Testing guide

### Run Tests with Coverage
```powershell
pytest Tests/ --cov=src --cov=Dashboard --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Customize Technical Indicators
```python
# Custom moving average windows
data_with_ma = calculate_moving_averages(data, windows=[7, 14, 30, 100])

# Custom RSI period
rsi_7 = calculate_rsi(data, period=7)

# Custom Bollinger Bands
middle, upper, lower = calculate_bollinger_bands(data, window=10, num_std=3)
```

## 🐛 Troubleshooting

### Issue: "yfinance not found"
```powershell
pip install yfinance
```

### Issue: "Dash not found"
```powershell
pip install dash plotly
```

### Issue: Tests fail with import errors
Make sure you're in the project root:
```powershell
cd C:\Users\anest\OneDrive\Documents\ML\Stock_ML\Stock_ML
pytest Tests/
```

### Issue: Dashboard won't start
Check if port 8050 is already in use:
```powershell
netstat -ano | findstr :8050
```

## 💡 Tips

1. **Use virtual environments** to avoid dependency conflicts
2. **Run tests regularly** to catch issues early: `pytest Tests/ -v`
3. **Check logs** for debugging - all functions use Python logging
4. **Start with small date ranges** when fetching data to avoid rate limits
5. **Use the dashboard** for quick exploratory analysis

## 📞 Support

- Check `Tests/README.md` for testing help
- Read `IMPLEMENTATION_SUMMARY.md` for technical details
- Review `WARP.md` for development guidelines

## 🎉 You're Ready!

Start exploring stock data and building ML models!

```powershell
# Fetch some data
python -c "from src.data_prep import fetch_data_yfinance; print(fetch_data_yfinance('AAPL', '2024-01-01', '2024-01-31'))"

# Launch the dashboard
python Dashboard/dashboard.py
```

Happy coding! 📈🚀
