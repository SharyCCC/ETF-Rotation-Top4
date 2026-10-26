
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    OUTPUT_DIR,
    ALLOCATION_METHODS,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    TRADING_DAYS,
)

# ==========================================
# 1. Robustness Parameters
# ==========================================

COST_SCENARIOS_BPS = [0, 5, 10, 20]

ROBUSTNESS_DIR = OUTPUT_DIR / "robustness_cost"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = {
    "Full History": (None, TEST_END),
    "Validation": (
        VALIDATION_START, VALIDATION_END
    ),
    "Test Period": (
        TEST_START, TEST_END
    ),
}


# ==========================================
# 2. Performance Metrics
# ==========================================

def calculate_metrics(returns):

    r = returns.dropna().astype(float)

    n = len(r)

    if n < 2:
        raise ValueError("Insufficient return observations.")

    cumulative = (1 + r).prod() - 1

    annual_return = (
        (1 + cumulative) ** (TRADING_DAYS / n) - 1
    )

    daily_vol = r.std(ddof=1)

    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)

    sharpe = (
        r.mean() / daily_vol * np.sqrt(TRADING_DAYS)
        if daily_vol > 0
        else np.nan
    )

    wealth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    peak = np.maximum.accumulate(wealth)
    max_dd = (wealth / peak - 1).min()

    return {
        "Cumulative Return": cumulative,
        "Annual Return": annual_return,
        "Sharpe Ratio": sharpe,
        "Annual Volatility": annual_vol,
        "Max Drawdown": max_dd,
    }


# ==========================================
# 3. Run Each Cost Scenario
# ==========================================

records = []
cost_records = []

for cost_bps in COST_SCENARIOS_BPS:

    scenario_dir = (
        ROBUSTNESS_DIR / f"cost_{cost_bps}bps"
    )

    scenario_dir.mkdir(
        parents=True, exist_ok=True
    )

    env = os.environ.copy()

    env["ROBUSTNESS_COST_BPS"] = str(cost_bps)
    env["ROBUSTNESS_OUTPUT_DIR"] = str(
        scenario_dir.resolve()
    )

    print("\n" + "=" * 70, flush=True)
    print(
        f"RUNNING TRANSACTION COST: {cost_bps} bps",
        flush=True
    )
    print("=" * 70, flush=True)

    subprocess.run(
        [sys.executable, "backtest.py"],
        cwd=BASE_DIR,
        env=env,
        check=True
    )

    # ======================================
    # 4. Read Each Allocation Method
    # ======================================

    for method in ALLOCATION_METHODS:

        safe_name = method.lower().replace(" ", "_")

        equity = pd.read_csv(
            scenario_dir / f"{safe_name}_equity.csv",
            index_col=0,
            parse_dates=True
        )

        trades = pd.read_csv(
            scenario_dir / f"{safe_name}_trades.csv"
        )

        total_cost = trades[
            "Transaction_Cost"
        ].sum()

        total_notional = trades[
            "Trade_Value"
        ].sum()

        cost_records.append({
            "Cost (bps)": cost_bps,
            "Method": method,
            "Total Transaction Cost": total_cost,
            "Total Traded Notional": total_notional,
            "Final Equity": equity["Equity"].iloc[-1],
            "Trade Records": len(trades),
        })

        for period_name, (start, end) in PERIODS.items():

            period_returns = equity[
                "Daily_Return"
            ].loc[start:end]

            metrics = calculate_metrics(
                period_returns
            )

            records.append({
                "Cost (bps)": cost_bps,
                "Method": method,
                "Period": period_name,
                **metrics
            })


# ==========================================
# 5. Assemble Results
# ==========================================

results = pd.DataFrame(records)
cost_summary = pd.DataFrame(cost_records)

results.to_csv(
    OUTPUT_DIR / "robustness_cost_sensitivity.csv",
    index=False
)

cost_summary.to_csv(
    OUTPUT_DIR / "robustness_cost_summary.csv",
    index=False
)


# ==========================================
# 6. Validate 10 bps Baseline
# ==========================================

print("\n=== BASELINE CONSISTENCY CHECK ===")

for method in ALLOCATION_METHODS:

    safe_name = method.lower().replace(" ", "_")

    baseline = pd.read_csv(
        OUTPUT_DIR / f"{safe_name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    scenario = pd.read_csv(
        ROBUSTNESS_DIR
        / "cost_10bps"
        / f"{safe_name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    assert baseline.index.equals(scenario.index)

    difference = (
        baseline["Equity"] - scenario["Equity"]
    ).abs().max()

    assert difference < 1e-6

    print(f"{method}: PASS, max error = {difference:.10f}")


# ==========================================
# 7. Display Results by Period
# ==========================================

for period_name in PERIODS:

    print("\n" + "=" * 85)
    print(f"{period_name.upper()} — COST SENSITIVITY")
    print("=" * 85)

    subset = results[
        results["Period"] == period_name
    ]

    table = subset.pivot(
        index="Cost (bps)",
        columns="Method",
        values="Cumulative Return"
    )

    formatted = table.map(
        lambda x: f"{x:.2%}"
    )

    print(formatted.to_string())


# ==========================================
# 8. Transaction Cost Summary
# ==========================================

print("\n=== TOTAL TRANSACTION COSTS ===")

cost_table = cost_summary.pivot(
    index="Cost (bps)",
    columns="Method",
    values="Total Transaction Cost"
)

print(cost_table.round(2).to_string())

print("\nPASS: Transaction Cost Robustness Completed.")
print("Results saved in:", OUTPUT_DIR)
