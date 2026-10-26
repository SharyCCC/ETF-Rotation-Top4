
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    ETF_UNIVERSE,
    DEFENSIVE_ETF,
    TIMING_ETF,
    MA_WINDOW,
    ALLOCATION_METHODS,
    TOP_N,
    TEST_START,
    TEST_END,
)

# ==========================================
# 1. Load Historical Prices
# ==========================================

prices = pd.read_csv(
    DATA_DIR / "etf_daily_adjusted_close.csv",
    index_col=0,
    parse_dates=True
).sort_index()

prices = prices[ETF_UNIVERSE]

# ==========================================
# 2. Calculate Market Timing Indicator
# ==========================================

spy_close = prices[TIMING_ETF]

spy_ma = spy_close.rolling(
    window=MA_WINDOW,
    min_periods=MA_WINDOW
).mean()

market_regime = pd.Series(
    np.where(
        spy_close > spy_ma,
        "Risk-On",
        "Risk-Off"
    ),
    index=prices.index,
    name="Market_Regime"
)

# No regime before the MA is available
market_regime.loc[spy_ma.isna()] = pd.NA

# ==========================================
# 3. Load Weekly Top 4 Allocations
# ==========================================

allocations = pd.read_csv(
    DATA_DIR / "weekly_allocations.csv",
    parse_dates=["Signal_Date", "Execution_Date"]
)

top_columns = [
    f"ETF_{i}" for i in range(1, TOP_N + 1)
]

weight_columns = [
    f"Weight_{i}" for i in range(1, TOP_N + 1)
]

# ==========================================
# 4. Combine Allocation & Market Timing
# ==========================================

records = []
weight_records = []

for _, row in allocations.iterrows():

    signal_date = row["Signal_Date"]
    execution_date = row["Execution_Date"]
    method = row["Method"]

    if method not in ALLOCATION_METHODS:
        raise ValueError(f"Unknown method: {method}")

    if signal_date not in market_regime.index:
        raise ValueError(
            f"Signal date not found: {signal_date}"
        )

    regime = market_regime.loc[signal_date]

    if pd.isna(regime):
        continue

    target = pd.Series(
        0.0,
        index=ETF_UNIVERSE,
        dtype=float
    )

    if regime == "Risk-On":
        for etf_col, weight_col in zip(
            top_columns, weight_columns
        ):
            ticker = row[etf_col]
            weight = float(row[weight_col])

            target.loc[ticker] = weight

    else:
        target.loc[DEFENSIVE_ETF] = 1.0

    # Verify target portfolio
    assert np.isclose(target.sum(), 1.0, atol=1e-8)
    assert (target >= -1e-8).all()

    record = {
        "Signal_Date": signal_date,
        "Execution_Date": execution_date,
        "Method": method,
        "Market_Regime": regime,
    }

    for i in range(1, TOP_N + 1):
        record[f"Top_{i}"] = row[f"ETF_{i}"]

    records.append(record)

    weight_record = record.copy()
    weight_record.update(target.to_dict())
    weight_records.append(weight_record)

# ==========================================
# 5. Create Output Tables
# ==========================================

weekly_signals = pd.DataFrame(records)
target_weights = pd.DataFrame(weight_records)

# Check three methods per signal date
method_counts = (
    target_weights.groupby("Signal_Date")["Method"]
    .nunique()
)

assert (method_counts == len(ALLOCATION_METHODS)).all()

# Check each target weight vector sums to one
assert np.allclose(
    target_weights[ETF_UNIVERSE].sum(axis=1),
    1.0,
    atol=1e-8
)

# Check Risk-Off portfolio
risk_off = target_weights[
    target_weights["Market_Regime"] == "Risk-Off"
]

assert np.allclose(risk_off[DEFENSIVE_ETF], 1.0)

assert np.allclose(
    risk_off[
        [x for x in ETF_UNIVERSE if x != DEFENSIVE_ETF]
    ].to_numpy(),
    0.0
)

# ==========================================
# 6. Save Results
# ==========================================

weekly_signals.to_csv(
    DATA_DIR / "weekly_strategy_signals.csv",
    index=False
)

target_weights.to_csv(
    DATA_DIR / "weekly_target_weights.csv",
    index=False
)

# ==========================================
# 7. Display Latest Signals
# ==========================================

print("\n=== LATEST WEEKLY STRATEGY SIGNALS ===")

print(
    weekly_signals.tail(12).to_string(index=False)
)

# ==========================================
# 8. Test Period Target Weights
# ==========================================

test_weights = target_weights[
    (target_weights["Signal_Date"] >= TEST_START)
    & (target_weights["Signal_Date"] <= TEST_END)
]

columns_to_show = [
    "Signal_Date",
    "Execution_Date",
    "Method",
    "Market_Regime",
] + ETF_UNIVERSE

print("\n=== TEST PERIOD TARGET WEIGHTS ===")

print(
    test_weights[columns_to_show].round({
        ticker: 4 for ticker in ETF_UNIVERSE
    })
    .to_string(index=False)
)

print("\n=== MARKET REGIME DISTRIBUTION ===")

regime_distribution = (
    weekly_signals[
        weekly_signals["Method"] == ALLOCATION_METHODS[0]
    ]["Market_Regime"].value_counts()
)

print(regime_distribution)

print("\nTotal strategy signal rows:", len(weekly_signals))
print("Total target weight rows:", len(target_weights))
print("\nStrategy signal construction completed.")
