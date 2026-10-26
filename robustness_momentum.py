
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

SCENARIOS = {
    "Short": [15, 40, 100],
    "Baseline": [20, 60, 120],
    "Long": [25, 70, 130],
}

RESULT_DIR = OUTPUT_DIR / "robustness_momentum"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = {
    "Full History": (None, TEST_END),
    "Validation": (VALIDATION_START, VALIDATION_END),
    "Test Period": (TEST_START, TEST_END),
}

# Full calculation pipeline for each scenario.
# Config is overridden only in the child process.
RUN_PIPELINE = """
import sys
import runpy
import config

config.DATA_DIR = config.Path(sys.argv[1])
config.OUTPUT_DIR = config.Path(sys.argv[2])
config.MOMENTUM_WINDOWS = [
    int(x) for x in sys.argv[3].split(",")
]

for filename in [
    "signals.py",
    "allocation.py",
    "strategy.py",
    "backtest.py",
]:
    print("Running:", filename, flush=True)
    runpy.run_path(filename, run_name="__main__")
"""


def calculate_metrics(returns):
    r = returns.dropna().astype(float)
    n = len(r)

    if n < 2:
        raise ValueError("Not enough observations")

    cumulative = (1 + r).prod() - 1
    annual_return = (
        (1 + cumulative) ** (TRADING_DAYS / n) - 1
    )

    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)

    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if r.std(ddof=1) > 0 else np.nan
    )

    wealth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    peak = np.maximum.accumulate(wealth)

    return {
        "Cumulative Return": cumulative,
        "Annual Return": annual_return,
        "Sharpe Ratio": sharpe,
        "Annual Volatility": vol,
        "Max Drawdown": (wealth / peak - 1).min(),
    }


records = []
selection_records = []

for name, windows in SCENARIOS.items():

    print(f"\n{'=' * 65}", flush=True)
    print(
        f"SCENARIO: {name} | Windows: {windows}",
        flush=True
    )
    print(f"{'=' * 65}", flush=True)

    scenario_dir = RESULT_DIR / name.lower()
    scenario_data = scenario_dir / "data"
    scenario_output = scenario_dir / "outputs"

    scenario_data.mkdir(parents=True, exist_ok=True)
    scenario_output.mkdir(parents=True, exist_ok=True)

    # Copy original data; don't overwrite baseline outputs
    for filename in [
        "etf_daily_adjusted_close.csv",
        "etf_daily_adjusted_open.csv",
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
            RUN_PIPELINE,
            str(scenario_data.resolve()),
            str(scenario_output.resolve()),
            ",".join(map(str, windows)),
        ],
        cwd=BASE_DIR,
        env=env,
        check=True,
    )

    # Read weekly Top 4 to compare selection stability
    weekly = pd.read_csv(
        scenario_data / "weekly_top4_signals.csv",
        parse_dates=["Signal_Date"]
    )

    for _, row in weekly.iterrows():
        selection_records.append({
            "Scenario": name,
            "Signal_Date": row["Signal_Date"],
            "Selected": "|".join(sorted([
                row["Top_1"], row["Top_2"],
                row["Top_3"], row["Top_4"]
            ]))
        })

    for method in ALLOCATION_METHODS:
        safe_name = method.lower().replace(" ", "_")

        equity = pd.read_csv(
            scenario_output / f"{safe_name}_equity.csv",
            index_col=0,
            parse_dates=True
        )

        for period_name, (start, end) in PERIODS.items():

            metrics = calculate_metrics(
                equity["Daily_Return"].loc[start:end]
            )

            records.append({
                "Scenario": name,
                "Windows": "/".join(map(str, windows)),
                "Method": method,
                "Period": period_name,
                **metrics,
            })


# ==========================================
# Save performance
# ==========================================

results = pd.DataFrame(records)

results.to_csv(
    OUTPUT_DIR / "robustness_momentum_windows.csv",
    index=False
)

selection = pd.DataFrame(selection_records)

selection.to_csv(
    OUTPUT_DIR / "robustness_momentum_selections.csv",
    index=False
)


# ==========================================
# Baseline consistency
# ==========================================

print("\n=== BASELINE CONSISTENCY CHECK ===")

for method in ALLOCATION_METHODS:

    safe_name = method.lower().replace(" ", "_")

    original = pd.read_csv(
        OUTPUT_DIR / f"{safe_name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    rerun = pd.read_csv(
        RESULT_DIR / "baseline" / "outputs"
        / f"{safe_name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    assert original.index.equals(rerun.index)

    error = (
        original["Equity"] - rerun["Equity"]
    ).abs().max()

    assert error < 1e-6

    print(f"{method}: PASS | max error = {error:.10f}")


# ==========================================
# Compare Top 4 selections
# ==========================================

selection_wide = selection.pivot(
    index="Signal_Date",
    columns="Scenario",
    values="Selected"
)

print("\n=== TOP 4 SELECTION STABILITY ===")

for scenario in ["Short", "Long"]:

    common = selection_wide[
        ["Baseline", scenario]
    ].dropna()

    identical = (
        common["Baseline"] == common[scenario]
    ).mean()

    overlaps = []

    for _, row in common.iterrows():
        baseline_set = set(row["Baseline"].split("|"))
        other_set = set(row[scenario].split("|"))
        overlaps.append(
            len(baseline_set & other_set) / 4
        )

    print(
        f"{scenario}: "
        f"identical sets = {identical:.2%}, "
        f"average overlap = {np.mean(overlaps):.2%}"
    )


# ==========================================
# Print performance comparisons
# ==========================================

for period in PERIODS:

    subset = results[results["Period"] == period]

    print("\n" + "=" * 85)
    print(f"{period.upper()} — MOMENTUM ROBUSTNESS")
    print("=" * 85)

    table = subset.pivot(
        index="Scenario",
        columns="Method",
        values="Cumulative Return"
    )

    print(
        table.loc[
            ["Short", "Baseline", "Long"]
        ].map(lambda x: f"{x:.2%}").to_string()
    )


print("\nPASS: Momentum Window Robustness Completed.")
