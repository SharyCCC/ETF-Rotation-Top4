
import pandas as pd
import numpy as np

from config import (
    DATA_DIR,
    ETF_UNIVERSE,
    RISK_UNIVERSE,
    MOMENTUM_WINDOWS,
    MOMENTUM_WEIGHTS,
    TOP_N,
    REBALANCE_FREQUENCY,
    TEST_START,
    TEST_END
)

# ==========================================
# 1. Load Adjusted Close Prices
# ==========================================

prices = pd.read_csv(
    DATA_DIR / "etf_daily_adjusted_close.csv",
    index_col=0,
    parse_dates=True
).sort_index()

prices = prices[ETF_UNIVERSE]

# ==========================================
# 2. Calculate Momentum Components
# ==========================================

momentum_components = {}

for window in MOMENTUM_WINDOWS:

    returns = prices.pct_change(
        periods=window,
        fill_method=None
    )

    momentum_components[window] = returns

# ==========================================
# 3. Composite Momentum Score
# ==========================================

momentum_scores = sum(
    weight * momentum_components[window]
    for window, weight in zip(
        MOMENTUM_WINDOWS,
        MOMENTUM_WEIGHTS
    )
)

# Exclude defensive BIL
risk_scores = momentum_scores[RISK_UNIVERSE]

# ==========================================
# 4. ETF Momentum Ranking
# ==========================================

momentum_rank = risk_scores.rank(
    axis=1,
    ascending=False,
    method="first"
)

# ==========================================
# 5. Identify Weekly Signal Dates
# ==========================================

# Select the last actual trading day
# of each Friday-ending week
weekly_signal_dates = (
    prices.groupby(
        prices.index.to_period(
            REBALANCE_FREQUENCY
        )
    )
    .tail(1)
    .index
)

# ==========================================
# 6. Select Dynamic Top 4
# ==========================================

records = []

for signal_date in weekly_signal_dates:

    row = risk_scores.loc[signal_date].dropna()

    # Ensure sufficient momentum history
    if len(row) < TOP_N:
        continue

    selected = row.nlargest(TOP_N)

    # Determine next available trading day
    idx = prices.index.get_loc(signal_date)

    if idx + 1 < len(prices.index):
        execution_date = prices.index[idx + 1]
    else:
        execution_date = pd.NaT

    record = {
        "Signal_Date": signal_date,
        "Execution_Date": execution_date
    }

    for i, (ticker, score) in enumerate(
        selected.items(), start=1
    ):
        record[f"Top_{i}"] = ticker
        record[f"Score_{i}"] = score

    records.append(record)

weekly_top4 = pd.DataFrame(records)

# ==========================================
# 7. Validation Checks
# ==========================================

top_columns = [
    f"Top_{i}" for i in range(1, TOP_N + 1)
]

assert weekly_top4[top_columns].notna().all().all()

assert weekly_top4[top_columns].apply(
    lambda row: len(set(row)) == TOP_N,
    axis=1
).all()

assert not weekly_top4[top_columns].isin(
    ["BIL"]
).any().any()

assert weekly_top4["Signal_Date"].is_monotonic_increasing

# ==========================================
# 8. Save Results
# ==========================================

momentum_scores.to_csv(
    DATA_DIR / "momentum_scores.csv"
)

momentum_rank.to_csv(
    DATA_DIR / "momentum_rank.csv"
)

weekly_top4.to_csv(
    DATA_DIR / "weekly_top4_signals.csv",
    index=False
)

# ==========================================
# 9. Display Results
# ==========================================

print("\n=== LATEST MOMENTUM SCORES ===")
print(
    momentum_scores.tail().round(4).to_string()
)

print("\n=== LATEST WEEKLY TOP 4 ===")
print(
    weekly_top4.tail(8).to_string(index=False)
)

test_signals = weekly_top4[
    (weekly_top4["Signal_Date"] >= TEST_START)
    & (weekly_top4["Signal_Date"] <= TEST_END)
]

print("\n=== TEST PERIOD TOP 4 SIGNALS ===")
print(test_signals.to_string(index=False))

print("\nTotal weekly signals:", len(weekly_top4))

print("\nMomentum signal construction completed.")
