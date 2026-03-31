from data_prep import prepare_price_data
from eda import quick_eda_summary, plot_price_trends
from gravity_model import gravity_signals_pipeline, information_coefficient
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: fetch and prep data ---
# Expanded from 6 to 30 tickers across 7 sectors for meaningful cross-sectional IC.
# With 6 stocks the daily IC was computed from only 6 data points — far too noisy.
tickers = [
    # Technology / Communication
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL", "AMD", "INTC",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # Consumer
    "WMT", "HD", "MCD", "COST", "NKE",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "BA",
    # Media / Entertainment
    "DIS", "NFLX",
]

prices, rets, feats, vols = prepare_price_data(
    tickers=tickers,
    start="2020-01-01",
    end="2024-12-31",
    return_kind="log",
)

print("Prices shape:", prices.shape)
print("Returns shape:", rets.shape)
print("Features shape:", feats.shape)

# --- Step 2: EDA summaries ---
eda_out = quick_eda_summary(prices=prices, returns=rets)
print("\nDescribe:\n", eda_out["describe"].head())
print("\nMissingness:\n", eda_out["missingness"].head())
print("\nCorrelation:\n", eda_out["corr"].head())

# --- Step 3: Plot a price trend ---
plot_price_trends(prices, tickers=["AAPL", "MSFT"])
#plt.show()

# --- Step 4: Train/test split (defined before any model evaluation) ---
# 2020-2022 = train (in-sample, for parameter search only)
# 2023-2024 = test  (out-of-sample, the honest IC)
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

# Horizon pre-specified based on model design: momentum_lookback=240 (1-year signal),
# so we evaluate at a 20-day forward horizon (standard medium-term check).
HORIZON = 20

print(f"\nTrain period: {prices.index[0].date()} to {TRAIN_END}")
print(f"Test period:  {TEST_START} to {prices.index[-1].date()}")
print(f"Horizon:      {HORIZON} days (pre-specified, not searched)")

# Forward log-returns at the pre-specified horizon (log convention matches returns used in model)
fwd = np.log(prices).diff(HORIZON).shift(-HORIZON)

# --- Step 5: Parameter search on TRAIN data only ---
print("\n" + "="*60)
print("PARAMETER SEARCH: Train data only (2020-2022)")
print("="*60)

# reversal_lookback=5 included in all configs: short-term (5-day) reversal
# subtracts from the long-term momentum direction to reduce noise from
# recent price spikes that are likely to mean-revert.
param_configs = [
    {"mass_window": 20, "dist_window": 30,  "momentum_lookback": 120, "reversal_lookback": 5},
    {"mass_window": 30, "dist_window": 60,  "momentum_lookback": 20,  "reversal_lookback": 5},
    {"mass_window": 60, "dist_window": 120, "momentum_lookback": 240, "reversal_lookback": 5},
    {"mass_window": 10, "dist_window": 20,  "momentum_lookback": 60,  "reversal_lookback": 5},
    # Baseline without reversal, for comparison
    {"mass_window": 60, "dist_window": 120, "momentum_lookback": 240, "reversal_lookback": None},
]

best_train_ic = None
best_config = None

for config in param_configs:
    print(f"\nTesting config: {config}")
    signals_cfg = gravity_signals_pipeline(
        prices=prices,
        returns=rets,
        volume=vols,
        mass_method="dollar_volume",
        mass_window=config["mass_window"],
        dist_window=config["dist_window"],
        momentum_lookback=config["momentum_lookback"],
        reversal_lookback=config["reversal_lookback"],
    )
    # Evaluate IC only on train period
    ic_cfg = information_coefficient(
        signals_cfg.loc[:TRAIN_END], fwd.loc[:TRAIN_END], method="spearman"
    )
    mean_ic = ic_cfg.mean()
    print(f"  → Train IC: {mean_ic:.4f}")

    if best_train_ic is None or abs(mean_ic) > abs(best_train_ic):
        best_train_ic = mean_ic
        best_config = config

print(f"\nBest config (by train IC): {best_config}")
print(f"Train IC: {best_train_ic:.4f}")

# --- Step 6: Run final model with best config, evaluate on TEST data ---
print("\n" + "="*60)
print("FINAL EVALUATION: Out-of-sample test period (2023-2024)")
print("="*60)

signals = gravity_signals_pipeline(
    prices=prices,
    returns=rets,
    volume=vols,
    mass_method="dollar_volume",
    mass_window=best_config["mass_window"],
    dist_window=best_config["dist_window"],
    momentum_lookback=best_config["momentum_lookback"],
    reversal_lookback=best_config["reversal_lookback"],
)

print("\nSignal statistics (full period):")
print("  Mean:", signals.mean().mean())
print("  Std:", signals.std().mean())
print("  Min:", signals.min().min())
print("  Max:", signals.max().max())
print("  Skewness:", signals.skew().mean())

# Out-of-sample IC — this is the honest number
ic_test = information_coefficient(
    signals.loc[TEST_START:], fwd.loc[TEST_START:], method="spearman"
)
ic_train = information_coefficient(
    signals.loc[:TRAIN_END], fwd.loc[:TRAIN_END], method="spearman"
)

print(f"\nTrain IC (in-sample,     reference only): {ic_train.mean():.4f}")
print(f"Test  IC (out-of-sample, honest result):  {ic_test.mean():.4f}")
print(f"Train ICIR: {ic_train.mean() / (ic_train.std() + 1e-12):.4f}")
print(f"Test  ICIR: {ic_test.mean() / (ic_test.std() + 1e-12):.4f}")

# --- Step 7: Plot out-of-sample IC only ---
ic_test_rolling = ic_test.rolling(window=30).mean()

plt.figure(figsize=(10, 5))
plt.plot(ic_test, alpha=0.3, label="Daily IC (test)", linewidth=0.8)
plt.plot(ic_test_rolling, color="red", label="30-day rolling mean", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axhline(
    ic_test.mean(), color="blue", linestyle="--", linewidth=1,
    label=f"Mean IC = {ic_test.mean():.4f}"
)
plt.title(
    f"Out-of-Sample IC (Spearman) — {HORIZON}-day horizon "
    f"[{TEST_START} to {prices.index[-1].date()}]"
)
plt.ylabel("IC Value")
plt.xlabel("Date")
plt.legend()
#plt.show()
