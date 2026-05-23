# Tests Directory

This directory contains comprehensive unit and integration tests for the Stock_ML project.

## Test Structure

```
Tests/
├── test_data_prep.py      # Tests for data fetching, validation, and file I/O
├── test_dashboard.py       # Tests for dashboard functions and technical indicators
└── README.md               # This file
```

## Running Tests

### Run All Tests
```powershell
# From project root (Stock_ML/)
pytest Tests/ -v
```

### Run Specific Test File
```powershell
pytest Tests/test_data_prep.py -v
pytest Tests/test_dashboard.py -v
```

### Run with Coverage Report
```powershell
pytest Tests/ --cov=src --cov=Dashboard --cov-report=html
# View coverage report at htmlcov/index.html
```

### Run Tests by Marker
```powershell
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only dashboard tests
pytest -m dashboard
```

## Test Coverage

### test_data_prep.py
Tests for `src/data_prep.py`:
- ✅ `fetch_data_yfinance()` - data fetching with mocked API calls
- ✅ `validate_stock_data()` - data validation logic
- ✅ `save_stock_data()` - saving to CSV, Parquet, Pickle formats
- ✅ `load_stock_data()` - loading from various formats
- ✅ Integration tests for save/load roundtrips

**Key features:**
- Mock yfinance API calls to avoid network dependencies
- Test error handling and edge cases
- Validate data integrity across save/load cycles

### test_dashboard.py
Tests for `Dashboard/dashboard.py`:
- ✅ `calculate_moving_averages()` - SMA calculations
- ✅ `calculate_rsi()` - RSI indicator (14-period)
- ✅ `calculate_bollinger_bands()` - Bollinger Bands
- ✅ `calculate_summary_stats()` - Summary statistics
- ✅ Chart creation functions (candlestick, technical, RSI, volume)
- ✅ Integration tests for full analysis pipeline

**Key features:**
- Reproducible tests with seeded random data
- Verification of technical indicator formulas
- Chart creation validation

## Writing New Tests

### Test Structure
```python
import pytest
from pathlib import Path
import sys

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from your_module import your_function

@pytest.fixture
def sample_data():
    """Create test data."""
    return create_test_data()

class TestYourFunction:
    """Tests for your_function."""
    
    def test_basic_functionality(self, sample_data):
        """Test basic case."""
        result = your_function(sample_data)
        assert result is not None
```

### Best Practices
1. **Use fixtures** for common test data
2. **Mock external dependencies** (APIs, file system when appropriate)
3. **Test edge cases** (empty data, invalid inputs, etc.)
4. **Use descriptive test names** that explain what's being tested
5. **Keep tests isolated** - each test should be independent
6. **Test both success and failure paths**

## Test Requirements

All tests require:
- `pytest>=7.4.0`
- `pytest-cov>=4.1.0` (for coverage reports)
- `pytest-mock>=3.11.0` (for mocking)

Install with:
```powershell
pip install -r requirements.txt
```

## CI/CD Integration

To integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest Tests/ --cov=src --cov=Dashboard --cov-report=xml
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running pytest from the project root:
```powershell
cd C:\Users\anest\OneDrive\Documents\ML\Stock_ML\Stock_ML
pytest Tests/
```

### Missing Dependencies
Install all requirements:
```powershell
pip install -r requirements.txt
```

### Slow Tests
Skip slow tests with:
```powershell
pytest -m "not slow"
```
