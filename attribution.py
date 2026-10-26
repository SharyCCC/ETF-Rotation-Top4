
import os
import sys
import shutil
import subprocess

import numpy as np
import pandas as pd

from config import (
    BASE_DIR, DATA_DIR, OUTPUT_DIR,
    ETF_UNIVERSE, ALLOCATION_METHODS,
    TOP_N, TRADING_DAYS,
    VALIDATION_START, VALIDATION_END,
    TEST_START, TEST_END,
)

# ==========================================
# 1. Set up independent scenario
# ==========================================

ROOT = OUTPUT_DIR / "attribution_no_timing"
SCENARIO_DATA = ROOT / "data"
SCENARIO_OUTPUT = ROOT / "outputs"

SCENARIO_DATA.mkdir(parents=True, exist_ok=True)
SCENARIO_OUTPUT.mkdir(parents=True, exist_ok=True)

for filename in [
    "etf_daily_adjusted_close.csv",
    "etf_daily_adjusted_open.csv",
]:
    shutil.copy2(
        DATA_DIR / filename,
        SCENARIO_DATA / filename
    )

# ==========================================
# 2. Build Momentum-Only Target Weights
# ==========================================

allocations = pd.read_csv(
    DATA_DIR / "weekly_allocations.csv",
    parse_dates=["Signal_Date", "Execution_Date"]
)

records = []

for _, row in allocations.iterrows():

    record = {
        "Signal_Date": row["Signal_Date"],
        "Execution_Date": row["Execution_Date"],
        "Method": row["Method"],
        "Market_Regime": "Risk-On",
    }

    for i in range(1, TOP_N + 1):
        record[f"Top_{i}"] = row[f"ETF_{i}"]

    for ticker in ETF_UNIVERSE:
        record[ticker] = 0.0

    for i in range(1, TOP_N + 1):
        ticker = row[f"ETF_{i}"]
        record[ticker] = float(row[f"Weight_{i}"])

    records.append(record)

target_weights = pd.DataFrame(records)

assert np.allclose(
    target_weights[ETF_UNIVERSE].sum(axis=1),
    1.0
)

assert np.allclose(target_weights["BIL"], 0.0)

target_weights.to_csv(
    SCENARIO_DATA / "weekly_target_weights.csv",
    index=False
)

print("Momentum-Only signals generated.")
print("Weekly signals:", len(target_weights) // 3)

# ==========================================
# 3. Run existing Backtesting Engine
# ==========================================

PIPELINE = """
import sys
import runpy
import config

config.DATA_DIR = config.Path(sys.argv[1])
config.OUTPUT_DIR = config.Path(sys.argv[2])

runpy.run_path("backtest.py", run_name="__main__")
"""

env = os.environ.copy()
env.pop("ROBUSTNESS_COST_BPS", None)
env.pop("ROBUSTNESS_OUTPUT_DIR", None)

subprocess.run(
    [
        sys.executable,
        "-c",
        PIPELINE,
        str(SCENARIO_DATA.resolve()),
        str(SCENARIO_OUTPUT.resolve()),
    ],
    cwd=BASE_DIR,
    env=env,
    check=True
)

# ==========================================
# 4. Performance Metrics
# ==========================================

def calculate_metrics(returns):

    r = returns.dropna().astype(float)
    n = len(r)

    cumulative = (1 + r).prod() - 1

    annual_return = (
        (1 + cumulative) ** (TRADING_DAYS / n) - 1
    )

    std = r.std(ddof=1)
    annual_vol = std * np.sqrt(TRADING_DAYS)

    sharpe = (
        r.mean() / std * np.sqrt(TRADING_DAYS)
        if std > 0 else np.nan
    )

    wealth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    peak = np.maximum.accumulate(wealth)

    return {
        "Cumulative Return": cumulative,
        "Annual Return": annual_return,
        "Sharpe Ratio": sharpe,
        "Annual Volatility": annual_vol,
        "Max Drawdown": (wealth / peak - 1).min(),
    }


PERIODS = {
    "Full History": (None, TEST_END),
    "Validation": (VALIDATION_START, VALIDATION_END),
    "Test Period": (TEST_START, TEST_END),
}

results = []

# ==========================================
# 5. Compare Full vs Momentum-Only
# ==========================================

for method in ALLOCATION_METHODS:

    safe = method.lower().replace(" ", "_")

    for strategy_name, directory in [
        ("Full Strategy", OUTPUT_DIR),
        ("Momentum-Only", SCENARIO_OUTPUT),
    ]:

        equity = pd.read_csv(
            directory / f"{safe}_equity.csv",
            index_col=0,
            parse_dates=True
        )

        trades = pd.read_csv(
            directory / f"{safe}_trades.csv"
        )

        for period_name, (start, end) in PERIODS.items():

            returns = equity["Daily_Return"].loc[start:end]

            metrics = calculate_metrics(returns)

            results.append({
                "Method": method,
                "Strategy": strategy_name,
                "Period": period_name,
                "Total Historical Cost":
                    trades["Transaction_Cost"].sum(),
                **metrics,
            })


comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT_DIR / "attribution_market_timing.csv",
    index=False
)

# ==========================================
# 6. Display Results
# ==========================================

for period in PERIODS:

    subset = comparison[
        comparison["Period"] == period
    ]

    print("\n" + "=" * 80)
    print(f"{period.upper()} — MARKET TIMING ATTRIBUTION")
    print("=" * 80)

    for metric in [
        "Cumulative Return",
        "Sharpe Ratio",
        "Max Drawdown",
    ]:

        print(f"\n{metric}:")

        table = subset.pivot(
            index="Method",
            columns="Strategy",
            values=metric
        )

        if metric == "Sharpe Ratio":
            print(table.round(3).to_string())
        else:
            print(
                table.map(
                    lambda x: f"{x:.2%}"
                ).to_string()
            )

print("\n=== TRANSACTION COST COMPARISON ===")

costs = comparison[
    comparison["Period"] == "Full History"
].pivot(
    index="Method",
    columns="Strategy",
    values="Total Historical Cost"
)

print(costs.round(2).to_string())

print("\nPASS: Market Timing Attribution Completed.")
