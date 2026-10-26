
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    ETF_UNIVERSE,
    ALLOCATION_METHODS,
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    TEST_START,
    TEST_END,
)


import os
from pathlib import Path

# Optional robustness settings
cost_bps = os.getenv("ROBUSTNESS_COST_BPS")
custom_output = os.getenv("ROBUSTNESS_OUTPUT_DIR")

if cost_bps is not None:
    TRANSACTION_COST = float(cost_bps) / 10000

if custom_output is not None:
    OUTPUT_DIR = Path(custom_output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not (0 <= TRANSACTION_COST < 1):
    raise ValueError("Invalid transaction cost.")


# ==========================================
# 1. Load data
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

assert open_prices.index.equals(close_prices.index)
assert np.isfinite(open_prices.to_numpy()).all()
assert np.isfinite(close_prices.to_numpy()).all()
assert (open_prices.to_numpy() > 0).all()
assert (close_prices.to_numpy() > 0).all()

signals = pd.read_csv(
    DATA_DIR / "weekly_target_weights.csv",
    parse_dates=["Signal_Date", "Execution_Date"]
)

# Only signals that can be executed
signals = signals.dropna(
    subset=["Execution_Date"]
).copy()

assert not signals.duplicated(
    ["Execution_Date", "Method"]
).any()

assert signals["Execution_Date"].isin(
    open_prices.index
).all()

assert (
    signals["Execution_Date"] > signals["Signal_Date"]
).all()

# Trading begins at the first executable signal
BACKTEST_START = signals["Execution_Date"].min()

dates = close_prices.loc[BACKTEST_START:].index


# ==========================================
# 2. Trade execution
# ==========================================

def execute_target(
    cash,
    shares,
    open_p,
    target,
    cost_rate
):
    """
    Execute target weights at adjusted open.

    Transaction costs are calculated on
    absolute traded notional.

    Portfolio target weights are based on
    equity after transaction costs.
    """

    current_value = shares * open_p
    equity_before = cash + current_value.sum()

    # Solve post-trade equity accounting for costs
    post_equity = equity_before

    for _ in range(1000):

        desired_value = target * post_equity

        trade_value = desired_value - current_value

        cost = (
            np.abs(trade_value).sum() * cost_rate
        )

        new_equity = equity_before - cost

        if abs(new_equity - post_equity) < 1e-10:
            post_equity = new_equity
            break

        post_equity = new_equity

    desired_value = target * post_equity
    trade_value = desired_value - current_value

    costs = np.abs(trade_value) * cost_rate

    new_shares = desired_value / open_p

    new_cash = (
        cash
        - trade_value.sum()
        - costs.sum()
    )

    # Accounting checks
    assert np.isclose(
        new_cash + (new_shares * open_p).sum(),
        equity_before - costs.sum(),
        atol=1e-6
    )

    assert new_cash >= -1e-6
    assert np.isfinite(new_shares).all()

    return (
        new_cash,
        new_shares,
        trade_value,
        costs
    )


# ==========================================
# 3. Backtest one allocation method
# ==========================================

def run_backtest(method):

    method_signals = signals[
        signals["Method"] == method
    ].copy()

    orders = {
        row["Execution_Date"]: row
        for _, row in method_signals.iterrows()
    }

    cash = float(INITIAL_CAPITAL)

    shares = np.zeros(
        len(ETF_UNIVERSE),
        dtype=float
    )

    equity_records = []
    weight_records = []
    trade_records = []

    for date in dates:

        open_p = open_prices.loc[
            date
        ].to_numpy(dtype=float)

        close_p = close_prices.loc[
            date
        ].to_numpy(dtype=float)

        # ----------------------------------
        # Execute scheduled weekly order
        # ----------------------------------

        if date in orders:

            signal = orders[date]

            target = signal[
                ETF_UNIVERSE
            ].to_numpy(dtype=float)

            assert np.isfinite(target).all()
            assert np.isclose(target.sum(), 1.0)
            assert (target >= -1e-8).all()

            (
                cash,
                new_shares,
                traded,
                costs
            ) = execute_target(
                cash,
                shares,
                open_p,
                target,
                TRANSACTION_COST
            )

            change = new_shares - shares

            for i, ticker in enumerate(ETF_UNIVERSE):

                if abs(traded[i]) <= 1e-7:
                    continue

                trade_records.append({
                    "Signal_Date": signal["Signal_Date"],
                    "Execution_Date": date,
                    "Method": method,
                    "Market_Regime": signal["Market_Regime"],
                    "Ticker": ticker,
                    "Action": (
                        "BUY"
                        if traded[i] > 0
                        else "SELL"
                    ),
                    "Trade_Value": abs(traded[i]),
                    "Price": open_p[i],
                    "Shares_Changed": change[i],
                    "Transaction_Cost": costs[i]
                })

            shares = new_shares

        # ----------------------------------
        # Mark portfolio at daily close
        # ----------------------------------

        position_values = shares * close_p

        equity = cash + position_values.sum()

        assert equity > 0
        assert np.isfinite(equity)

        equity_records.append({
            "Date": date,
            "Equity": equity,
            "Cash": cash
        })

        weights = position_values / equity

        weight_record = {
            "Date": date,
            "Cash_Weight": cash / equity
        }

        for i, ticker in enumerate(ETF_UNIVERSE):
            weight_record[ticker] = weights[i]

        weight_records.append(weight_record)

    # ----------------------------------
    # 4. Create output DataFrames
    # ----------------------------------

    equity_df = pd.DataFrame(
        equity_records
    ).set_index("Date")

    equity_df["Daily_Return"] = (
        equity_df["Equity"].pct_change()
    )

    # First day return is relative to starting capital
    equity_df.iloc[
        0,
        equity_df.columns.get_loc("Daily_Return")
    ] = (
        equity_df["Equity"].iloc[0]
        / INITIAL_CAPITAL - 1
    )

    equity_df["Drawdown"] = (
        equity_df["Equity"]
        / equity_df["Equity"].cummax()
        - 1
    )

    weights_df = pd.DataFrame(
        weight_records
    ).set_index("Date")

    trades_df = pd.DataFrame(trade_records)

    return equity_df, weights_df, trades_df


# ==========================================
# 5. Run three independent strategies
# ==========================================

all_equity = {}
all_weights = {}
all_trades = []

for method in ALLOCATION_METHODS:

    print(f"\nRunning strategy: {method}")

    equity, weights, trades = run_backtest(method)

    all_equity[method] = equity
    all_weights[method] = weights

    all_trades.append(trades)

    safe_name = (
        method.lower().replace(" ", "_")
    )

    equity.to_csv(
        OUTPUT_DIR / f"{safe_name}_equity.csv"
    )

    weights.to_csv(
        OUTPUT_DIR / f"{safe_name}_actual_weights.csv"
    )

    trades.to_csv(
        OUTPUT_DIR / f"{safe_name}_trades.csv",
        index=False
    )

    print(
        f"Final equity: ${equity['Equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Transaction costs: "
        f"${trades['Transaction_Cost'].sum():,.2f}"
    )

    print("Trade records:", len(trades))


# ==========================================
# 6. Combined Portfolio Equity
# ==========================================

combined_equity = pd.DataFrame({
    method: df["Equity"]
    for method, df in all_equity.items()
})

combined_returns = pd.DataFrame({
    method: df["Daily_Return"]
    for method, df in all_equity.items()
})

combined_equity.to_csv(
    OUTPUT_DIR / "three_strategy_equity.csv"
)

combined_returns.to_csv(
    OUTPUT_DIR / "three_strategy_daily_returns.csv"
)


# ==========================================
# 7. Final test-period results
# ==========================================

print("\n=== TEST PERIOD EQUITY ===")

test_equity = combined_equity.loc[
    TEST_START:TEST_END
]

print(test_equity.round(2).to_string())

print("\n=== TEST PERIOD FINAL EQUITY ===")

print(
    test_equity.iloc[-1].round(2).to_string()
)

print("\n=== BACKTEST COMPLETED ===")
print("Backtest starts:", BACKTEST_START.date())
print("Backtest ends:", dates[-1].date())
print("Results saved in:", OUTPUT_DIR)
