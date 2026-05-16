import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

tickers = {
    "SAP.DE": "large_cap",
    "SIE.DE": "mid_cap",
    "RHM.DE": "volatile"
}

all_data = {}

for ticker in tickers:
    df = yf.download(ticker, start="2021-01-01", end="2026-01-01", auto_adjust=True)

    df.columns = [col[0] for col in df.columns]

    df["Stock"] = ticker
    df = df.reset_index()
    all_data[ticker] = df

combined_data = pd.concat(all_data.values()).sort_values(["Stock", "Date"]).reset_index(drop=True)

combined_data["simple_return"] = (
    combined_data.groupby("Stock")["Close"].pct_change()
)

combined_data["log_return"] = (
    combined_data.groupby("Stock")["Close"]
    .transform(lambda x: np.log(x / x.shift(1)))
)

combined_data["forward_return_5d"] = (
    combined_data.groupby("Stock")["Close"]
    .transform(lambda x: (x.shift(-5) - x) / x)
)

combined_data["y"] = np.where(
    combined_data["forward_return_5d"] > 0.005, 1,
    np.where(
        combined_data["forward_return_5d"] < -0.005, 0,
        np.nan
    )
)

rows_before = len(combined_data)
combined_data = combined_data.dropna(subset=["y", "simple_return", "log_return", "forward_return_5d"])
rows_after = len(combined_data)

print(f"Rows before dropping: {rows_before}")
print(f"Rows after dropping:  {rows_after}")
print(f"Rows lost:            {rows_before - rows_after}")

print("\nClass distribution per ticker:")
print(
    combined_data.groupby("Stock")["y"]
    .value_counts(normalize=True)
    .mul(100).round(2)
)

plt.figure(figsize=(12, 6))
for stock in combined_data["Stock"].unique():
    stock_data = combined_data[combined_data["Stock"] == stock]
    plt.plot(stock_data["Date"], stock_data["simple_return"], label=stock)
plt.title("Daily Simple Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.show()

os.makedirs("data/raw", exist_ok=True)

for ticker in tickers:
    ticker_df = combined_data[combined_data["Stock"] == ticker]
    ticker_df.to_csv(f"data/raw/{ticker.replace('.', '_')}.csv", index=False)
    print(f"Saved: data/raw/{ticker.replace('.', '_')}.csv")

print("\nFinal dataset shape:")
print(combined_data.shape)
print("\nSample rows:")
print(combined_data[["Date", "Stock", "Close", "simple_return", "log_return", "forward_return_5d", "y"]].head(10))
