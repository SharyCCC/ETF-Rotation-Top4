
from pathlib import Path

# ==========================================
# 1. Project Directories
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. ETF Universe
# ==========================================

ETF_UNIVERSE = [
    "EWS", "SPY", "AGG", "GLD", "BIL",
    "QQQ", "IWM", "EFA", "EEM",
    "XLK", "XLE", "DBC"
]

DEFENSIVE_ETF = "BIL"

RISK_UNIVERSE = [
    etf for etf in ETF_UNIVERSE
    if etf != DEFENSIVE_ETF
]


# ==========================================
# 3. Momentum Selection
# ==========================================

MOMENTUM_WINDOWS = [20, 60, 120]
MOMENTUM_WEIGHTS = [0.5, 0.3, 0.2]

TOP_N = 4


# ==========================================
# 4. Portfolio Allocation
# ==========================================

ALLOCATION_METHODS = [
    "Equal Weight",
    "Inverse Volatility",
    "Risk Parity"
]

VOL_WINDOW = 60

# Long-only position constraints
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.40


# ==========================================
# 5. Market Timing
# ==========================================

TIMING_ETF = "SPY"
MA_WINDOW = 20

RISK_OFF_ASSET = "BIL"


# ==========================================
# 6. Trading Rules
# ==========================================

REBALANCE_FREQUENCY = "W-FRI"

TRANSACTION_COST = 0.001

INITIAL_CAPITAL = 100000


# ==========================================
# 7. Historical Periods
# ==========================================

DATA_START = "2018-01-01"

DEVELOPMENT_END = "2023-12-31"

VALIDATION_START = "2024-01-01"
VALIDATION_END = "2026-09-30"

TEST_START = "2026-10-01"
TEST_END = "2026-10-23"

# yfinance end date is exclusive
DOWNLOAD_END = "2026-10-24"


# ==========================================
# 8. Performance Settings
# ==========================================

TRADING_DAYS = 252
RISK_FREE_RATE = 0.0


# ==========================================
# 9. Configuration Validation
# ==========================================

assert len(MOMENTUM_WINDOWS) == len(MOMENTUM_WEIGHTS)

assert abs(sum(MOMENTUM_WEIGHTS) - 1.0) < 1e-10

assert TOP_N <= len(RISK_UNIVERSE)

assert TOP_N * MAX_WEIGHT >= 1.0

assert DEFENSIVE_ETF not in RISK_UNIVERSE


if __name__ == "__main__":
    print("Configuration loaded successfully.")
    print("ETF Universe:", len(ETF_UNIVERSE), "ETFs")
    print("Dynamic Selection: Top", TOP_N)
    print("Allocation Methods:", ALLOCATION_METHODS)
    print("Volatility Window:", VOL_WINDOW)
    print("Test Period:", TEST_START, "to", TEST_END)
