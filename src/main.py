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
    momentum_lookback=20, # momentum lookback for distance calc
)

print("\nSignals shape:", signals.shape) 

# --- Step 5: Evaluate with IC (Information Coefficient) ---
fwd5 = prices.pct_change(5).shift(-5)  # forward 5-day returns
ic = information_coefficient(signals, fwd5, method="spearman") # IC over time 
print("\nMean IC (Spearman):", ic.mean()) # average IC across time 

plt.figure(figsize=(10,5))
ic.plot(title="Daily Information Coefficient (Spearman)", linewidth=1)
plt.axhline(0, color="black", linestyle="--")  # reference line
plt.ylabel("IC Value")
plt.xlabel("Date")
plt.show()
