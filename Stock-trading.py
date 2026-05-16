#Set up required applications
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data_large_cap = yf.download("SAP.DE", start="2021-01-01", end="2026-01-01")
data_large_cap.index
data_large_cap.shape
data_large_cap.info()
data_large_cap.isnull().sum()
print(data_large_cap.index)
print(data_large_cap.isnull().sum())
print(data_large_cap.shape)
print(data_large_cap.head())
data_mid_cap= yf.download("SIE.DE", start="2021-01-01", end="2026-01-01")
print(data_mid_cap.head())
print(data_mid_cap.shape)
data_mid_cap.index
data_mid_cap.shape
data_mid_cap.info()
data_mid_cap.isnull().sum()
data_small_cap= yf.download("RHM.DE", start="2021-01-01", end="2026-01-01")
data_small_cap.index
data_small_cap.shape
data_small_cap.info()
data_small_cap.isnull().sum()
print(data_small_cap.head())
print(data_small_cap.shape)
data_large_cap["Stock"] = "SAP.DE"
data_mid_cap["Stock"] = "SIE.DE"
data_small_cap["Stock"] = "RHM.DE"


# ============================================
# STEP 4 — RESET INDEX
# ============================================

# Convert Date index into normal column

data_large_cap = data_large_cap.reset_index()
data_mid_cap = data_mid_cap.reset_index()
data_small_cap = data_small_cap.reset_index()


# ============================================
# STEP 5 — COMBINE DATASETS
# ============================================

combined_data = pd.concat([
    data_large_cap,
    data_mid_cap,
    data_small_cap
])


# ============================================
# STEP 6 — SORT BY DATE
# ============================================

combined_data = combined_data.sort_values(
    by=["Stock", "Date"]
)


# ============================================
# STEP 7 — COMPUTE RETURNS
# ============================================

# Calculate percentage change in closing price
# separately for each stock

combined_data["Return"] = (
    combined_data
    .groupby("Stock")["Close"]
    .pct_change()
)


# ============================================
# STEP 8 — REMOVE NaN VALUES
# ============================================

combined_data = combined_data.dropna()


# ============================================
# STEP 9 — CHECK FINAL DATASET
# ============================================

print(combined_data.head())

print("\nDataset Shape:")
print(combined_data.shape)

print("\nMissing Values:")
print(combined_data.isnull().sum())


# ============================================
# STEP 10 — PLOT RETURNS
# ============================================

plt.figure(figsize=(12, 6))

for stock in combined_data["Stock"].unique():

    stock_data = combined_data[
        combined_data["Stock"] == stock
    ]

    plt.plot(
        stock_data["Date"],
        stock_data["Return"],
        label=stock
    )

plt.title("Stock Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()

plt.show()


# ============================================
# STEP 11 — SAVE CLEAN DATASET
# ============================================

combined_data.to_csv(
    "combined_stock_data.csv",
    index=False
)

print("\nDataset saved successfully.")