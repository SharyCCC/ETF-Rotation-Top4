
import os
import sys
import shutil
import subprocess

import numpy as np
import pandas as pd

from config import (
    BASE_DIR, DATA_DIR, OUTPUT_DIR,
    ALLOCATION_METHODS,
    TRADING_DAYS,
    VALIDATION_START, VALIDATION_END,
    TEST_START, TEST_END,
)

SCENARIOS = {
    2: 0.75,
    3: 0.50,
    4: 0.40,
}

ROOT = OUTPUT_DIR / "robustness_topn"
ROOT.mkdir(parents=True, exist_ok=True)

PERIODS = {
    "Validation": (VALIDATION_START, VALIDATION_END),
    "Test Period": (TEST_START, TEST_END),
}

PIPELINE = """
import sys
import runpy
import config

config.DATA_DIR = config.Path(sys.argv[1])
config.OUTPUT_DIR = config.Path(sys.argv[2])
config.TOP_N = int(sys.argv[3])
config.MAX_WEIGHT = float(sys.argv[4])

for script in [
    "signals.py",
    "allocation.py",
    "strategy.py",
    "backtest.py",
]:
    print("Running:", script, flush=True)
    runpy.run_path(script, run_name="__main__")
"""


def performance(r):
    r = r.dropna().astype(float)
    n = len(r)

    cumulative = (1 + r).prod() - 1
    annual = (1 + cumulative) ** (TRADING_DAYS / n) - 1

    daily_vol = r.std(ddof=1)
    volatility = daily_vol * np.sqrt(TRADING_DAYS)

    sharpe = (
        r.mean() / daily_vol * np.sqrt(TRADING_DAYS)
        if daily_vol > 0 else np.nan
    )

    wealth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    peak = np.maximum.accumulate(wealth)

    return {
        "Cumulative Return": cumulative,
        "Annual Return": annual,
        "Sharpe Ratio": sharpe,
        "Annual Volatility": volatility,
        "Max Drawdown": (wealth / peak - 1).min(),
    }


records = []

for top_n, cap in SCENARIOS.items():

    print(f"\n{'=' * 65}", flush=True)
    print(f"RUNNING TOP {top_n} | CAP {cap:.0%}", flush=True)
    print(f"{'=' * 65}", flush=True)

    folder = ROOT / f"top_{top_n}"
    data_dir = folder / "data"
    output_dir = folder / "outputs"

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "etf_daily_adjusted_close.csv",
        "etf_daily_adjusted_open.csv",
    ]:
        shutil.copy2(
            DATA_DIR / filename,
            data_dir / filename
        )

    env = os.environ.copy()
    env.pop("ROBUSTNESS_COST_BPS", None)
    env.pop("ROBUSTNESS_OUTPUT_DIR", None)

    subprocess.run(
        [
            sys.executable,
            "-c",
            PIPELINE,
            str(data_dir.resolve()),
            str(output_dir.resolve()),
            str(top_n),
            str(cap),
        ],
        cwd=BASE_DIR,
        env=env,
        check=True,
    )

    for method in ALLOCATION_METHODS:

        name = method.lower().replace(" ", "_")

        equity = pd.read_csv(
            output_dir / f"{name}_equity.csv",
            index_col=0,
            parse_dates=True
        )

        trades = pd.read_csv(
            output_dir / f"{name}_trades.csv"
        )

        for period_name, (start, end) in PERIODS.items():

            r = equity["Daily_Return"].loc[start:end]

            if period_name == "Test Period":
                assert len(r) == 17

            records.append({
                "Top N": top_n,
                "Max Weight": cap,
                "Method": method,
                "Period": period_name,
                "Total Historical Cost":
                    trades["Transaction_Cost"].sum(),
                **performance(r)
            })


# ==========================================
# Save Results
# ==========================================

results = pd.DataFrame(records)

results.to_csv(
    OUTPUT_DIR / "robustness_topn.csv",
    index=False
)


# ==========================================
# Verify Top 4 Baseline
# ==========================================

print("\n=== TOP 4 BASELINE CHECK ===")

for method in ALLOCATION_METHODS:

    name = method.lower().replace(" ", "_")

    original = pd.read_csv(
        OUTPUT_DIR / f"{name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    rerun = pd.read_csv(
        ROOT / "top_4" / "outputs"
        / f"{name}_equity.csv",
        index_col=0,
        parse_dates=True
    )

    assert original.index.equals(rerun.index)

    error = (
        original["Equity"] - rerun["Equity"]
    ).abs().max()

    assert error < 1e-6

    print(f"{method}: PASS | error = {error:.10f}")


# ==========================================
# Display Results
# ==========================================

for period in PERIODS:

    subset = results[
        results["Period"] == period
    ]

    print("\n" + "=" * 75)
    print(f"{period.upper()} — TOP N ROBUSTNESS")
    print("=" * 75)

    table = subset.pivot(
        index="Top N",
        columns="Method",
        values="Cumulative Return"
    )

    print(
        table.map(lambda x: f"{x:.2%}").to_string()
    )

    sharpe_table = subset.pivot(
        index="Top N",
        columns="Method",
        values="Sharpe Ratio"
    )

    print("\nSharpe Ratio:")
    print(sharpe_table.round(3).to_string())


print("\nPASS: Top N Robustness Completed.")
