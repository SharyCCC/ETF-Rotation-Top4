
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    ALLOCATION_METHODS,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    TRADING_DAYS,
    RISK_FREE_RATE,
    INITIAL_CAPITAL,
)


# ==========================================
# 1. Load Backtest Results
# ==========================================

def load_equity(method):
    filename = method.lower().replace(" ", "_")

    df = pd.read_csv(
        OUTPUT_DIR / f"{filename}_equity.csv",
        index_col=0,
        parse_dates=True
    ).sort_index()

    return df


# ==========================================
# 2. Calculate Performance Metrics
# ==========================================

def calculate_metrics(returns):

    returns = returns.dropna().astype(float)

    n = len(returns)

    if n < 2:
        raise ValueError("Insufficient observations.")

    # Cumulative Return
    cumulative = (1 + returns).prod() - 1

    # Annualized Return
    annual_return = (
        (1 + cumulative) ** (TRADING_DAYS / n) - 1
    )

    # Annualized Volatility
    daily_vol = returns.std(ddof=1)

    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)

    # Annualized Sharpe Ratio
    daily_rf = (
        (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    )

    sharpe = (
        (returns.mean() - daily_rf)
        / daily_vol
        * np.sqrt(TRADING_DAYS)
        if daily_vol > 0
        else np.nan
    )

    # Drawdown within selected evaluation period
    wealth = np.r_[
        1.0,
        (1 + returns).cumprod().to_numpy()
    ]

    running_peak = np.maximum.accumulate(wealth)

    drawdown = wealth / running_peak - 1

    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative,
        "Annual Return": annual_return,
        "Annual Sharpe Ratio": sharpe,
        "Annual Volatility": annual_vol,
        "Max Drawdown": max_drawdown
    }


# ==========================================
# 3. Define Evaluation Periods
# ==========================================

PERIODS = {
    "Full History": (None, TEST_END),
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
# 4. Run Performance Comparison
# ==========================================

results = {}
period_returns = {}

for period_name, (start, end) in PERIODS.items():

    metrics_records = {}
    returns_records = {}

    for method in ALLOCATION_METHODS:

        equity = load_equity(method)

        # Returns calculated from continuous
        # equity history, not rebased prices
        returns = equity["Daily_Return"].loc[
            start:end
        ].dropna()

        if period_name == "Test Period":
            assert len(returns) == 17

        metrics_records[method] = calculate_metrics(
            returns
        )

        returns_records[method] = returns

    comparison = pd.DataFrame(
        metrics_records
    ).T

    results[period_name] = comparison

    period_returns[period_name] = pd.DataFrame(
        returns_records
    )

    safe_name = (
        period_name.lower().replace(" ", "_")
    )

    comparison.to_csv(
        OUTPUT_DIR / f"{safe_name}_performance.csv"
    )


# ==========================================
# 5. Format and Print Results
# ==========================================

def format_metrics(df):

    display = df.copy()

    percentage_columns = [
        "Cumulative Return",
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown"
    ]

    for col in percentage_columns:
        display[col] = display[col].map(
            lambda x: f"{x:.2%}"
        )

    display["Annual Sharpe Ratio"] = (
        display["Annual Sharpe Ratio"].map(
            lambda x: f"{x:.3f}"
        )
    )

    return display


for period_name, df in results.items():

    print("\n" + "=" * 75)
    print(period_name.upper(), "PERFORMANCE")
    print("=" * 75)

    print(format_metrics(df).to_string())


# ==========================================
# 6. Test Period Equity Verification
# ==========================================

print("\n" + "=" * 75)
print("TEST PERIOD EQUITY VERIFICATION")
print("=" * 75)

for method in ALLOCATION_METHODS:

    equity = load_equity(method)

    previous = equity.loc[
        equity.index < pd.Timestamp(TEST_START),
        "Equity"
    ].iloc[-1]

    final = equity.loc[
        :TEST_END, "Equity"
    ].iloc[-1]

    cumulative = final / previous - 1

    reported = results["Test Period"].loc[
        method, "Cumulative Return"
    ]

    assert np.isclose(
        cumulative, reported, atol=1e-10
    )

    print(f"\n{method}")
    print(f"Starting equity: ${previous:,.2f}")
    print(f"Ending equity:   ${final:,.2f}")
    print(f"Test return:     {cumulative:.4%}")
    print("PASS")


# ==========================================
# 7. Save Test Cumulative Return Curves
# ==========================================

test_returns = period_returns["Test Period"]

test_cumulative = (
    (1 + test_returns).cumprod() - 1
)

test_cumulative.to_csv(
    OUTPUT_DIR / "test_cumulative_returns.csv"
)


print("\nPerformance evaluation completed.")
print("Results saved in:", OUTPUT_DIR)
