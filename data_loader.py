
import pandas as pd
import numpy as np
import yfinance as yf

from config import (
    ETF_UNIVERSE,
    DATA_START,
    DOWNLOAD_END,
    DATA_DIR,
    TEST_START,
    TEST_END
)

# ==========================================
# 1. Download Historical ETF Data
# ==========================================

print("Downloading historical ETF data...")

raw = yf.download(
    tickers=ETF_UNIVERSE,
    start=DATA_START,
    end=DOWNLOAD_END,
    auto_adjust=False,
    group_by="column",
    threads=True,
    progress=True
)

if raw.empty:
    raise ValueError("No market data downloaded.")

# ==========================================
# 2. Extract Price Fields
# ==========================================

def extract_field(field):
    if field not in raw.columns.get_level_values(0):
        raise ValueError(f"Missing field: {field}")

    return raw[field].reindex(
        columns=ETF_UNIVERSE
    ).sort_index()


open_prices = extract_field("Open")
close_prices = extract_field("Close")
adj_close_prices = extract_field("Adj Close")
volume = extract_field("Volume")

# ==========================================
# 3. Data Quality Checks
# ==========================================

print("\n=== DATA QUALITY CHECK ===")
print("Trading days:", len(close_prices))
print("ETF count:", len(close_prices.columns))
print("Start date:", close_prices.index.min().date())
print("End date:", close_prices.index.max().date())

print("\nMissing Values — Close:")
print(close_prices.isna().sum())

print("\nMissing Values — Adjusted Close:")
print(adj_close_prices.isna().sum())

print("\nMissing Values — Open:")
print(open_prices.isna().sum())

print("\nMissing Values — Volume:")
print(volume.isna().sum())

# Prices must be positive
for name, df in {
    "Open": open_prices,
    "Close": close_prices,
    "Adjusted Close": adj_close_prices
}.items():
    if (df.dropna() <= 0).any().any():
        raise ValueError(
            f"Non-positive prices detected in {name}."
        )

# Check that all ETFs have valid dates
if close_prices.isna().all().any():
    raise ValueError(
        "At least one ETF has no valid closing prices."
    )

# Check test period availability
test_prices = close_prices.loc[
    TEST_START:TEST_END
]

if test_prices.empty:
    raise ValueError(
        "Test period price data is missing."
    )

print("\nTest Period Trading Days:", len(test_prices))

# ==========================================
# 4. Save Raw Data
# ==========================================

open_prices.to_csv(DATA_DIR / "etf_daily_open.csv")
close_prices.to_csv(DATA_DIR / "etf_daily_close.csv")
adj_close_prices.to_csv(
    DATA_DIR / "etf_daily_adjusted_close.csv"
)
volume.to_csv(DATA_DIR / "etf_daily_volume.csv")

# ==========================================
# 5. Save Adjusted Open Approximation
# ==========================================

# Use the same adjustment factor as adjusted close
adjustment_factor = (
    adj_close_prices / close_prices
)

adjusted_open = (
    open_prices * adjustment_factor
)

adjusted_open.to_csv(
    DATA_DIR / "etf_daily_adjusted_open.csv"
)

# ==========================================
# 6. Display Sample Data
# ==========================================

print("\n=== FIRST FIVE ROWS ===")
print(close_prices.head().to_string())

print("\n=== LAST FIVE ROWS ===")
print(close_prices.tail().to_string())

print("\n=== DATA COLLECTION COMPLETED ===")
print("Files saved in:", DATA_DIR)
