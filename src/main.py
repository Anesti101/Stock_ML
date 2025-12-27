from data_prep import prepare_price_data
from eda import quick_eda_summary, plot_price_trends
from gravity_model import gravity_signals_pipeline, information_coefficient
import matplotlib.pyplot as plt

# --- Step 1: fetch and prep data ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]

prices, rets, feats, vols = prepare_price_data( # fetches price data and computes returns/features
    tickers=tickers,
    start="2020-01-01",
    end="2024-12-31",
    return_kind="log",
)

print("Prices shape:", prices.shape) # shape: (dates, tickers)
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

# --- Step 4: Run gravity model signals ---
signals = gravity_signals_pipeline(
    prices=prices, # price data 
    returns=rets, # return data 
    volume=vols,  # trading volume data 
    mass_method="dollar_volume", # mass based on dollar volume 
    mass_window=60, # lookback window for mass calc 
    dist_window=60, # lookback window for distance calc
    momentum_lookback=240, # momentum lookback for distance calc
)

print("\nSignals shape:", signals.shape)
print("Signal statistics:")
print("  Mean:", signals.mean().mean())
print("  Std:", signals.std().mean())
print("  Min:", signals.min().min())
print("  Max:", signals.max().max())
print("  Skewness:", signals.skew().mean())

# INVERT SIGNALS: multiply by -1 to flip direction (original IC was negative)
signals = signals * -1
print("\nSignals INVERTED (multiplied by -1)")
print("Signal statistics (after inversion):")
print("  Mean:", signals.mean().mean())
print("  Std:", signals.std().mean())
print("  Min:", signals.min().min())
print("  Max:", signals.max().max())

results = {}  # store mean ICs per horizon

# test multiple forward-return horizons
for k in [1, 5, 10, 20, 40, 60, 120, 240, 360, 480, 720]:
    fwd = prices.pct_change(k).shift(-k)
    ic_k = information_coefficient(signals, fwd, method="spearman")
    mean_ic = ic_k.mean()
    results[k] = mean_ic
    print(f"{k}-day horizon: {mean_ic:.4f}")

# choose the best (highest absolute or positive IC)
best_k = max(results, key=lambda x: abs(results[x]))
print(f"\nBest horizon (by absolute IC): {best_k} days -> Mean IC = {results[best_k]:.4f}")

# recompute full IC series for that horizon for plotting
fwd_best = prices.pct_change(best_k).shift(-best_k)
ic = information_coefficient(signals, fwd_best, method="spearman") # IC over time 
print("\nMean IC (Spearman):", ic.mean()) # average IC across time 
# Compute a 30-day rolling mean of IC
ic_rolling = ic.rolling(window=30).mean()


plt.figure(figsize=(10,5))
plt.plot(ic, alpha=0.3, label="Daily IC", linewidth=0.8)
plt.plot(ic_rolling, color="red", label="30-day rolling mean", linewidth=2)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Daily vs Rolling Information Coefficient (Spearman)")
plt.ylabel("IC Value")
plt.xlabel("Date")
plt.legend()
#plt.show()

plt.figure(figsize=(10,5))
ic.plot(title="Daily Information Coefficient (Spearman)", linewidth=1)
plt.axhline(0, color="black", linestyle="--")  # reference line
plt.ylabel("IC Value")
plt.xlabel("Date")
#plt.show()

# --- PARAMETER TUNING SECTION ---
print("\n" + "="*60)
print("PARAMETER TUNING: Testing different configurations")
print("="*60)

param_configs = [
    {"mass_window": 20, "dist_window": 30, "momentum_lookback": 120},
    {"mass_window": 30, "dist_window": 60, "momentum_lookback": 20},
    {"mass_window": 60, "dist_window": 120, "momentum_lookback": 240},
    {"mass_window": 10, "dist_window": 20, "momentum_lookback": 60},
]

best_overall_ic = None
best_overall_config = None

for config in param_configs:
    print(f"\nTesting config: {config}")
    
    signals_test = gravity_signals_pipeline(
        prices=prices,
        returns=rets,
        volume=vols,
        mass_method="dollar_volume",
        mass_window=config["mass_window"],
        dist_window=config["dist_window"],
        momentum_lookback=config["momentum_lookback"],
    )
    
    # Test IC at different horizons
    test_results = {}
    for k in [1, 5, 10, 20, 40, 60]:
        fwd_test = prices.pct_change(k).shift(-k)
        ic_test = information_coefficient(signals_test, fwd_test, method="spearman")
        test_results[k] = ic_test.mean()
    
    best_k_test = max(test_results, key=lambda x: abs(test_results[x]))
    mean_ic_test = test_results[best_k_test]
    
    print(f"  → Best horizon: {best_k_test} days, Mean IC: {mean_ic_test:.4f}")
    
    if best_overall_ic is None or abs(mean_ic_test) > abs(best_overall_ic):
        best_overall_ic = mean_ic_test
        best_overall_config = config

print(f"\n{'='*60}")
print(f"BEST CONFIG FOUND:")
print(f"  Parameters: {best_overall_config}")
print(f"  Mean IC: {best_overall_ic:.4f}")
print(f"{'='*60}")
