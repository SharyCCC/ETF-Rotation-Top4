
import os
import sys
import shutil
import subprocess

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    ALLOCATION_METHODS,
    TRADING_DAYS,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
)

# ==========================================
# 1. Market Timing Scenarios
# ==========================================

SCENARIOS = {
    "MA10": 10,
    "MA20": 20,
    "MA30": 30,
}

ROOT = OUTPUT_DIR / "robustness_timing"
ROOT.mkdir(parents=True, exist_ok=True)

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
# 2. Scenario Pipeline
# ==========================================

PIPELINE = """
import sys
import runpy
import config

config.DATA_DIR = config.Path(sys.argv[1])
config.OUTPUT_DIR = config.Path(sys.argv[2])
config.MA_WINDOW = int(sys.argv[3])

for filename in ["strategy.py", "backtest.py"]:
    print("Running:", filename, flush=True)
    runpy.run_path(filename, run_name="__main__")
"""


# ==========================================
# 3. Performance Metrics
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

    daily_std = r.std(ddof=1)

    annual_vol = daily_std * np.sqrt(TRADING_DAYS)

    sharpe = (
        r.mean() / daily_std * np.sqrt(TRADING_DAYS)
        if daily_std > 0
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
# 4. Run Scenarios
# ==========================================

records = []
regime_records = []

for scenario, ma_window in SCENARIOS.items():

    print("\n" + "=" * 65, flush=True)
    print(
        f"RUNNING MARKET TIMING: {scenario}",
        flush=True
    )
    print("=" * 65, flush=True)

    folder = ROOT / scenario.lower()

    scenario_data = folder / "data"
    scenario_output = folder / "outputs"

    scenario_data.mkdir(parents=True, exist_ok=True)
    scenario_output.mkdir(parents=True, exist_ok=True)

    # Only the data needed by strategy and backtest
    for filename in [
        "etf_daily_adjusted_close.csv",
        "etf_daily_adjusted_open.csv",
        "weekly_allocations.csv",
    ]:
        shutil.copy2(
            DATA_DIR / filename,
            scenario_data / filename
        )

    env = os.environ.copy()
    env.pop("ROBUSTNESS_COST_BPS", None)
    env.pop("ROBUSTNESS_OUTPUT_DIR", None)

    subprocess.run(
        [
            sys.executable,
            "-c",
            PIPELINE,
            str(scenario_data.resolve()),
            str(scenario_output.resolve()),
            str(ma_window),
        ],
        cwd=BASE_DIR,
        env=env,
        check=True,
    )

    # --------------------------------------
    # Market Regime Distribution
    # --------------------------------------

    signals = pd.read_csv(
        scenario_data / "weekly_strategy_signals.csv",
        parse_dates=["Signal_Date", "Execution_Date"]
    )

    unique_signals = signals[
        signals["Method"] == ALLOCATION_METHODS[0]
    ]

    # Exclude final non-executable signal
    executed = unique_signals.dropna(
        subset=["Execution_Date"]
    )

    counts = executed[
        "Market_Regime"
    ].value_counts()

    regime_records.append({
        "Scenario": scenario,
        "Risk-On Signals": counts.get("Risk-On", 0),
        "Risk-Off Signals": counts.get("Risk-Off", 0),
        "Executed Signals": len(executed),
    })

    # --------------------------------------
    # Performance of Three Allocations
    # --------------------------------------

    for method in ALLOCATION_METHODS:

        safe_name = method.lower().replace(" ", "_")

        equity = pd.read_csv(
            scenario_output / f"{safe_name}_equity.csv",
            index_col=0,
            parse_dates=True
        )

        trades = pd.read_csv(
            scenario_output / f"{safe_name}_trades.csv"
        )

        for period_name, (start, end) in PERIODS.items():

            r = equity["Daily_Return"].loc[start:end]

            if period_name == "Test Period":
                assert len(r) == 17

            records.append({
                "Scenario": scenario,
                "MA Window": ma_window,
                "Method": method,
                "Period": period_name,
                "Total Historical Cost":
                    trades["Transaction_Cost"].sum(),
                **calculate_metrics(r),
            })


# ==========================================
# 5. Save Results
# ==========================================

results = pd.DataFrame(records)
regimes = pd.DataFrame(regime_records)

results.to_csv(
    OUTPUT_DIR / "robustness_market_timing.csv",
    index=False
)

regimes.to_csv(
    OUTPUT_DIR / "robustness_market_regimes.csv",
    index=False
)


# ==========================================
# 6. MA20 Baseline Consistency
# ==========================================

print("\n=== MA20 BASELINE CHECK ===")

for method in ALLOCATION_METHODS:

    name = method.lower().replace(" ", "_")

    baseline = pd.read_csv(
        OUTPUT_DIR / f"{name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    rerun = pd.read_csv(
        ROOT / "ma20" / "outputs"
        / f"{name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    assert baseline.index.equals(rerun.index)

    error = (
        baseline["Equity"] - rerun["Equity"]
    ).abs().max()

    assert error < 1e-6

    print(
        f"{method}: PASS | max error = {error:.10f}"
    )


# ==========================================
# 7. Display Market Regime Statistics
# ==========================================

print("\n=== MARKET REGIME DISTRIBUTION ===")

print(
    regimes.set_index("Scenario").to_string()
)


# ==========================================
# 8. Display Performance
# ==========================================

for period in PERIODS:

    subset = results[
        results["Period"] == period
    ]

    print("\n" + "=" * 80)
    print(f"{period.upper()} — MARKET TIMING ROBUSTNESS")
    print("=" * 80)

    cumulative = subset.pivot(
        index="Scenario",
        columns="Method",
        values="Cumulative Return"
    )

    print("\nCumulative Return:")
    print(
        cumulative.loc[
            ["MA10", "MA20", "MA30"]
        ].map(lambda x: f"{x:.2%}").to_string()
    )

    sharpe = subset.pivot(
        index="Scenario",
        columns="Method",
        values="Sharpe Ratio"
    )

    print("\nSharpe Ratio:")
    print(
        sharpe.loc[
            ["MA10", "MA20", "MA30"]
        ].round(3).to_string()
    )

    drawdown = subset.pivot(
        index="Scenario",
        columns="Method",
        values="Max Drawdown"
    )

    print("\nMax Drawdown:")
    print(
        drawdown.loc[
            ["MA10", "MA20", "MA30"]
        ].map(lambda x: f"{x:.2%}").to_string()
    )


print("\nPASS: Market Timing Robustness Completed.")
