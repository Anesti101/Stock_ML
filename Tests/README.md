# Tests

Run all tests from the project root:

```powershell
pytest Tests -q
```

## Test Files

```text
Tests/
|-- test_data_prep.py              # Data fetching, validation, returns, features, saves
|-- test_dashboard.py               # Dashboard indicator and chart helpers
|-- test_supervised_regression.py   # 20-day target and leakage controls
`-- README.md
```

## Supervised Regression Checks

`test_supervised_regression.py` verifies:

- `future_return_20d = log(price[t+20]) - log(price[t])`.
- The target and target end date metadata are not model features.
- Training rows are purged when their target horizon crosses into the test period.
- Expanding validation folds purge label overlap.
- Feature values at an as-of date are unchanged when only future prices are modified.

## Notes

The tests mock external data where needed and should be run from the repository root so `src/` and `Dashboard/` imports resolve consistently.
