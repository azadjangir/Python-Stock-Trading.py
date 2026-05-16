
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



data_large_cap = data_large_cap.reset_index()
data_mid_cap = data_mid_cap.reset_index()
data_small_cap = data_small_cap.reset_index()

combined_data = pd.concat([
    data_large_cap,
    data_mid_cap,
    data_small_cap
])



combined_data = combined_data.sort_values(
    by=["Stock", "Date"]
)

# Calculate percentage change in closing price
# separately for each stock

combined_data["Return"] = (
    combined_data
    .groupby("Stock")["Close"]
    .pct_change()
)


combined_data = combined_data.dropna()


print(combined_data.head())

print("\nDataset Shape:")
print(combined_data.shape)

print("\nMissing Values:")
print(combined_data.isnull().sum())

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

combined_data.to_csv(
    "combined_stock_data.csv",
    index=False
)

print("\nDataset saved successfully.")
