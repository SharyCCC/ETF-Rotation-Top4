
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    ETF_UNIVERSE,
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
)

# ==========================================
# 1. Load adjusted prices
# ==========================================

def load_prices(filename):
    return pd.read_csv(
        DATA_DIR / filename,
        index_col=0,
        parse_dates=True
    ).sort_index()[ETF_UNIVERSE]


open_prices = load_prices(
    "etf_daily_adjusted_open.csv"
)

close_prices = load_prices(
    "etf_daily_adjusted_close.csv"
)

# Match the main strategy's first trading date
strategy = pd.read_csv(
    OUTPUT_DIR / "equal_weight_equity.csv",
    index_col=0,
    parse_dates=True
)

start_date = strategy.index.min()
dates = strategy.index

assert dates.isin(close_prices.index).all()

print("Benchmark Start:", start_date.date())


# ==========================================
# 2. Buy-and-hold simulator
# ==========================================

def buy_and_hold(target_weights):

    target = pd.Series(
        target_weights,
        index=ETF_UNIVERSE,
        dtype=float
    )

    assert np.isclose(target.sum(), 1.0)
    assert (target >= 0).all()

    initial_open = open_prices.loc[start_date]

    invested = (
        INITIAL_CAPITAL / (1 + TRANSACTION_COST)
    )

    cost = invested * TRANSACTION_COST

    shares = target * invested / initial_open

    cash = INITIAL_CAPITAL - invested - cost

    equity = (
        close_prices.loc[dates]
        .mul(shares, axis=1)
        .sum(axis=1)
        + cash
    )

    returns = equity.pct_change()

    # Include the first trading day's return
    returns.iloc[0] = (
        equity.iloc[0] / INITIAL_CAPITAL - 1
    )

    return equity, returns, cost


# ==========================================
# 3. SPY Buy & Hold
# ==========================================

spy_weights = pd.Series(
    0.0, index=ETF_UNIVERSE
)

spy_weights["SPY"] = 1.0

spy_equity, spy_returns, spy_cost = (
    buy_and_hold(spy_weights)
)


# ==========================================
# 4. Equal Weight 12 ETFs Buy & Hold
# ==========================================

equal_weights = pd.Series(
    1.0 / len(ETF_UNIVERSE),
    index=ETF_UNIVERSE
)

equal_equity, equal_returns, equal_cost = (
    buy_and_hold(equal_weights)
)


# ==========================================
# 5. Combine and export
# ==========================================

benchmark_equity = pd.DataFrame({
    "SPY Buy & Hold": spy_equity,
    "Equal Weight 12 Buy & Hold": equal_equity
})

benchmark_returns = pd.DataFrame({
    "SPY Buy & Hold": spy_returns,
    "Equal Weight 12 Buy & Hold": equal_returns
})

benchmark_equity.to_csv(
    OUTPUT_DIR / "benchmark_equity.csv"
)

benchmark_returns.to_csv(
    OUTPUT_DIR / "benchmark_daily_returns.csv"
)


# ==========================================
# 6. Validation checks
# ==========================================

for column in benchmark_equity.columns:

    equity = benchmark_equity[column]
    r = benchmark_returns[column]

    assert np.isclose(
        (1 + r).prod(),
        equity.iloc[-1] / INITIAL_CAPITAL,
        atol=1e-10
    )

    assert np.isfinite(equity).all()
    assert (equity > 0).all()


# ==========================================
# 7. Output
# ==========================================

print("\n=== BENCHMARK RESULTS ===")

for name in benchmark_equity.columns:

    final_equity = benchmark_equity[name].iloc[-1]

    cumulative = (
        final_equity / INITIAL_CAPITAL - 1
    )

    print(f"\n{name}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Cumulative Return: {cumulative:.2%}")

print("\nInitial transaction cost:", f"${spy_cost:,.2f}")
print("\nBenchmark construction completed.")
