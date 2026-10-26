
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    DATA_DIR,
    ETF_UNIVERSE,
    TOP_N,
    VOL_WINDOW,
    MAX_WEIGHT,
    MIN_WEIGHT,
)

# ==========================================
# 1. Load historical data
# ==========================================

prices = pd.read_csv(
    DATA_DIR / "etf_daily_adjusted_close.csv",
    index_col=0,
    parse_dates=True
).sort_index()[ETF_UNIVERSE]

daily_returns = prices.pct_change(fill_method=None)

weekly_top4 = pd.read_csv(
    DATA_DIR / "weekly_top4_signals.csv",
    parse_dates=["Signal_Date", "Execution_Date"]
)

METHODS = [
    "Equal Weight",
    "Inverse Volatility",
    "Risk Parity"
]


# ==========================================
# 2. Validate weights
# ==========================================

def validate_weights(weights):

    w = np.asarray(weights, dtype=float)

    assert len(w) == TOP_N
    assert np.isfinite(w).all()
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= MIN_WEIGHT - 1e-6).all()
    assert (w <= MAX_WEIGHT + 1e-6).all()

    return w


# ==========================================
# 3. Capped allocation
# ==========================================

def capped_allocation(raw_weights):
    """
    Normalize positive scores while enforcing
    the maximum ETF weight using bisection.
    """

    raw = np.asarray(raw_weights, dtype=float)

    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("Allocation scores must be positive.")

    if len(raw) * MAX_WEIGHT < 1:
        raise ValueError("Maximum weight is infeasible.")

    raw = raw / raw.sum()

    low, high = 0.0, 1.0

    while np.minimum(high * raw, MAX_WEIGHT).sum() < 1:
        high *= 2

    for _ in range(100):
        mid = (low + high) / 2
        total = np.minimum(mid * raw, MAX_WEIGHT).sum()

        if total < 1:
            low = mid
        else:
            high = mid

    weights = np.minimum(high * raw, MAX_WEIGHT)

    return validate_weights(weights)


# ==========================================
# 4. Equal Weight
# ==========================================

def equal_weight():

    return validate_weights(
        np.repeat(1.0 / TOP_N, TOP_N)
    )


# ==========================================
# 5. Inverse Volatility
# ==========================================

def inverse_volatility(returns):

    vol = returns.std(ddof=1).to_numpy()

    if not np.isfinite(vol).all() or (vol <= 0).any():
        raise ValueError("Invalid historical volatility.")

    inverse_vol = 1.0 / vol

    return capped_allocation(inverse_vol)


# ==========================================
# 6. Risk Parity
# ==========================================

def risk_parity(returns):

    
    covariance = returns.cov().to_numpy().copy()

    if not np.isfinite(covariance).all():
        raise ValueError("Invalid covariance matrix.")

    # Small numerical regularization
    covariance = covariance + np.eye(TOP_N) * 1e-12


    initial_weights = equal_weight()

    def objective(w):

        portfolio_variance = w @ covariance @ w

        if portfolio_variance <= 0:
            return 1e6

        # Fractional contribution to portfolio variance
        contribution = (
            w * (covariance @ w)
            / portfolio_variance
        )

        target = np.repeat(1.0 / TOP_N, TOP_N)

        return np.sum((contribution - target) ** 2)

    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    }]

    bounds = [
        (MIN_WEIGHT, MAX_WEIGHT)
        for _ in range(TOP_N)
    ]

    result = minimize(
        objective,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12
        }
    )

    if not result.success:
        raise RuntimeError(
            f"Risk Parity optimization failed: {result.message}"
        )

    weights = validate_weights(result.x)

    return weights


# ==========================================
# 7. Generate weekly allocations
# ==========================================

records = []

for _, signal in weekly_top4.iterrows():

    signal_date = signal["Signal_Date"]

    selected = [
        signal[f"Top_{i}"]
        for i in range(1, TOP_N + 1)
    ]

    # Last 60 observed daily returns up to signal date
    historical = daily_returns.loc[
        :signal_date, selected
    ].tail(VOL_WINDOW)

    if (
        len(historical) < VOL_WINDOW
        or historical.isna().any().any()
    ):
        continue

    allocations = {
        "Equal Weight": equal_weight(),
        "Inverse Volatility": inverse_volatility(historical),
        "Risk Parity": risk_parity(historical)
    }

    for method, weights in allocations.items():

        record = {
            "Signal_Date": signal_date,
            "Execution_Date": signal["Execution_Date"],
            "Method": method
        }

        for i, ticker in enumerate(selected):
            record[f"ETF_{i+1}"] = ticker
            record[f"Weight_{i+1}"] = weights[i]

        records.append(record)


# ==========================================
# 8. Export and validate
# ==========================================

allocation_df = pd.DataFrame(records)

weight_columns = [
    f"Weight_{i}"
    for i in range(1, TOP_N + 1)
]

assert not allocation_df.empty
assert allocation_df[weight_columns].notna().all().all()

assert np.allclose(
    allocation_df[weight_columns].sum(axis=1),
    1.0
)

allocation_df.to_csv(
    DATA_DIR / "weekly_allocations.csv",
    index=False
)


# ==========================================
# 9. Display latest allocations
# ==========================================

print("\n=== LATEST WEEKLY ALLOCATIONS ===")

latest_date = allocation_df["Signal_Date"].max()

latest = allocation_df[
    allocation_df["Signal_Date"] == latest_date
].copy()

for col in weight_columns:
    latest[col] = latest[col].map(
        lambda x: f"{x:.2%}"
    )

print(latest.to_string(index=False))

print("\nTotal allocation records:", len(allocation_df))

print("\nPortfolio allocation completed.")
