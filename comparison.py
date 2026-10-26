
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    TRADING_DAYS,
    RISK_FREE_RATE,
    INITIAL_CAPITAL,
)

# ==========================================
# 1. Load strategy and benchmark returns
# ==========================================

strategy_returns = pd.read_csv(
    OUTPUT_DIR / "three_strategy_daily_returns.csv",
    index_col=0,
    parse_dates=True
).sort_index()

benchmark_returns = pd.read_csv(
    OUTPUT_DIR / "benchmark_daily_returns.csv",
    index_col=0,
    parse_dates=True
).sort_index()

# Combine all five strategies
all_returns = pd.concat(
    [strategy_returns, benchmark_returns],
    axis=1,
    join="inner"
)

assert len(all_returns.columns) == 5
assert all_returns.notna().all().all()
assert all_returns.index.is_unique
assert all_returns.index.is_monotonic_increasing

print("Five strategies:")
print(all_returns.columns.tolist())

print("\nTrading days:", len(all_returns))
print("Start:", all_returns.index.min().date())
print("End:", all_returns.index.max().date())

# ==========================================
# 2. Performance calculation
# ==========================================

def calculate_metrics(returns):

    r = returns.dropna().astype(float)

    n = len(r)

    if n < 2:
        raise ValueError("Insufficient observations.")

    cumulative = (1 + r).prod() - 1

    annual_return = (
        (1 + cumulative) ** (TRADING_DAYS / n) - 1
    )

    annual_vol = (
        r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    daily_rf = (
        (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS)
        - 1
    )

    daily_std = r.std(ddof=1)

    sharpe = (
        (r.mean() - daily_rf)
        / daily_std
        * np.sqrt(TRADING_DAYS)
        if daily_std > 0
        else np.nan
    )

    wealth = np.r_[
        1.0,
        (1 + r).cumprod().to_numpy()
    ]

    peak = np.maximum.accumulate(wealth)

    max_drawdown = (wealth / peak - 1).min()

    return {
        "Cumulative Return": cumulative,
        "Annual Return": annual_return,
        "Annual Sharpe Ratio": sharpe,
        "Annual Volatility": annual_vol,
        "Max Drawdown": max_drawdown
    }


# ==========================================
# 3. Define comparison periods
# ==========================================

periods = {
    "Full History": (
        None,
        TEST_END
    ),
    "Validation": (
        VALIDATION_START,
        VALIDATION_END
    ),
    "Test Period": (
        TEST_START,
        TEST_END
    )
}


# ==========================================
# 4. Calculate all five strategy metrics
# ==========================================

def format_metrics(df):

    out = df.copy()

    percent_cols = [
        "Cumulative Return",
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown"
    ]

    for col in percent_cols:
        out[col] = out[col].map(
            lambda x: f"{x:.2%}"
        )

    out["Annual Sharpe Ratio"] = (
        out["Annual Sharpe Ratio"].map(
            lambda x: f"{x:.3f}"
        )
    )

    return out


for period_name, (start, end) in periods.items():

    period_data = all_returns.loc[start:end]

    if period_name == "Test Period":
        assert len(period_data) == 17

    results = {
        strategy: calculate_metrics(
            period_data[strategy]
        )
        for strategy in period_data.columns
    }

    comparison = pd.DataFrame(results).T

    filename = (
        period_name.lower()
        .replace(" ", "_")
        + "_five_strategy_comparison.csv"
    )

    comparison.to_csv(OUTPUT_DIR / filename)

    print("\n" + "=" * 90)
    print(period_name.upper())
    print("=" * 90)

    print(format_metrics(comparison).to_string())


# ==========================================
# 5. Normalized cumulative wealth
# ==========================================

normalized_wealth = (
    1 + all_returns
).cumprod()

normalized_wealth.to_csv(
    OUTPUT_DIR / "five_strategy_wealth.csv"
)

# Drawdown for all five strategies
running_peak = normalized_wealth.cummax()

# Include initial wealth of 1.0 in peak
running_peak = running_peak.clip(lower=1.0)

drawdown = (
    normalized_wealth / running_peak - 1
)

drawdown.to_csv(
    OUTPUT_DIR / "five_strategy_drawdown.csv"
)

# ==========================================
# 6. Test-period cumulative returns
# ==========================================

test_data = all_returns.loc[
    TEST_START:TEST_END
]

test_wealth = (1 + test_data).cumprod()

test_cumulative = test_wealth - 1

test_cumulative.to_csv(
    OUTPUT_DIR / "five_strategy_test_cumulative.csv"
)

# ==========================================
# 7. Consistency checks
# ==========================================

assert np.allclose(
    (1 + all_returns).prod().to_numpy(),
    normalized_wealth.iloc[-1].to_numpy()
)

assert np.isfinite(normalized_wealth.to_numpy()).all()

print("\nPASS: All five strategy returns are valid.")
print("Five-strategy comparison completed.")
print("Results saved in:", OUTPUT_DIR)
