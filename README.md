# Dynamic Top 4 ETF Momentum Rotation & Portfolio Allocation

A quantitative investment research project comparing three portfolio allocation methods within a dynamic ETF momentum rotation strategy.

**动态 Top 4 ETF 动量轮动与投资组合配置研究**

本项目基于历史 ETF 数据，研究动态动量选股、资金配置和市场择时机制对投资组合收益与风险的影响。

## 1. Project Overview / 项目介绍

The strategy dynamically selects the four highest-momentum ETFs each week and compares three portfolio allocation approaches:

- Equal Weight
- Inverse Volatility
- Risk Parity

A market timing filter based on the SPY 20-day Moving Average determines whether the portfolio holds the selected ETFs or switches to the defensive asset BIL.

本项目每周重新选择动量最高的四只 ETF，并比较三种资金分配方法。通过 SPY 20 日移动平均线判断市场状态，在 Risk-Off 时配置 100% BIL。

## 2. Strategy Configuration

| Component | Configuration |
|---|---|
| ETF Universe | EWS, SPY, AGG, GLD, BIL, QQQ, IWM, EFA, EEM, XLK, XLE, DBC |
| Risk Asset Candidates | 11 ETFs, excluding BIL |
| Momentum Windows | 20 / 60 / 120 trading days |
| Momentum Weights | 50% / 30% / 20% |
| Portfolio Selection | Dynamic Top 4 |
| Rebalancing | Weekly |
| Portfolio Allocations | Equal Weight, Inverse Volatility, Risk Parity |
| Volatility Estimation | 60 trading days |
| Maximum ETF Weight | 40% during Risk-On |
| Market Timing | SPY 20-day Moving Average |
| Risk-Off Allocation | 100% BIL |
| Execution | Next trading day's adjusted open |
| Transaction Cost | 10 bps per traded notional |
| Initial Capital | $100,000 |
| Test Period | October 1–23, 2026 |

## 3. Research Workflow / 研究流程

1. **Data Collection:** Download and validate historical ETF prices.
2. **Momentum Calculation:** Compute the composite momentum score using 20/60/120-day returns.
3. **Dynamic Top 4 Selection:** Rank eligible ETFs weekly and select the top four.
4. **Portfolio Allocation:** Calculate Equal Weight, Inverse Volatility, and Risk Parity allocations.
5. **Market Timing:** Apply the SPY MA20 Risk-On/Risk-Off rule.
6. **Backtesting:** Simulate next-open trades, transaction costs, cash and holdings.
7. **Performance Evaluation:** Calculate returns, Sharpe Ratio, volatility and maximum drawdown.
8. **Benchmark Comparison:** Compare against SPY Buy & Hold and Equal Weight 12 ETF Buy & Hold.
9. **Robustness Analysis:** Test transaction costs, momentum windows, Top N and MA windows.
10. **Strategy Attribution:** Compare the full strategy against Momentum-Only.
11. **Dashboard:** Visualize and explore the research results using Streamlit.

## 4. Main Results / 核心结果

### Full History (2018–2026)

| Strategy | Cumulative Return | Sharpe Ratio | Max Drawdown |
|---|---:|---:|---:|
| Top 4 Equal Weight | 73.24% | 0.615 | -16.24% |
| Top 4 Inverse Volatility | 63.76% | 0.583 | -18.01% |
| Top 4 Risk Parity | 59.89% | 0.558 | -19.30% |
| SPY Buy & Hold | 218.96% | 0.828 | -33.72% |
| Equal Weight 12 Buy & Hold | 159.61% | 0.852 | -25.75% |

### Test Period (October 1–23, 2026)

| Strategy | Cumulative Return |
|---|---:|
| Top 4 Equal Weight | +0.82% |
| Top 4 Inverse Volatility | +0.75% |
| Top 4 Risk Parity | +0.77% |
| SPY Buy & Hold | -0.35% |
| Equal Weight 12 Buy & Hold | -1.91% |

**Key findings / 主要发现：**

- Equal Weight produced the highest long-term return among the three active allocation methods.
- Risk Parity achieved the highest active-strategy validation Sharpe Ratio of 1.215.
- All three active strategies outperformed both passive benchmarks during the 17-day test period.
- Market timing reduced drawdowns in the validation period, but introduced substantial long-term opportunity costs.
- Trading costs materially affected long-term strategy performance.

## 5. How to Run the Dashboard / 如何运行

**Requirements:** Git and Python 3.12.

### macOS / Linux

```bash
git clone https://github.com/SharyCCC/ETF-Rotation-Top4.git
cd ETF-Rotation-Top4

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Windows PowerShell

```powershell
git clone https://github.com/SharyCCC/ETF-Rotation-Top4.git
cd ETF-Rotation-Top4

py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

**http://localhost:8501**

The repository includes precomputed CSV files in `data/` and `outputs/`. You do not need to rerun the backtests simply to view the Dashboard.

项目已经包含计算好的回测数据。组员只需要安装依赖并运行 Streamlit，即可查看已有研究结果。

## 6. Dashboard Features

The Streamlit Dashboard contains six interactive sections:

1. Performance Comparison
2. Dynamic Top 4 Selection
3. Portfolio Allocation
4. Trades & Transaction Costs
5. Robustness Analysis
6. Market Timing Attribution

## 7. Project Structure

```text
ETF-Rotation-Top4/
├── app.py
├── config.py
├── data_loader.py
├── signals.py
├── allocation.py
├── strategy.py
├── backtest.py
├── metrics.py
├── benchmarks.py
├── comparison.py
├── robustness.py
├── robustness_momentum.py
├── robustness_topn.py
├── robustness_timing.py
├── attribution.py
├── requirements.txt
├── .python-version
├── data/
└── outputs/
```

## 8. Research Limitations / 研究限制

- The October test period contains only 17 trading days.
- Different momentum lookback settings have different initial backtest dates, which affects full-history comparability.
- Some strategy design choices were evaluated after earlier historical results had already been examined.
- The simulation uses adjusted opening prices as execution proxies, allows fractional shares and assumes a fixed proportional transaction cost.
- The Sharpe Ratio calculations use a zero risk-free rate.
- Historical performance does not guarantee future returns.

本研究仅用于学术分析。短期测试结果不能代表长期绩效，回测也无法完全复制真实市场中的交易成本与成交条件。

## 9. Collaboration

Team members can clone the repository, review the Python scripts, inspect the precomputed results, and contribute improvements through GitHub.

For collaborative updates, create a separate Git branch and submit a Pull Request rather than directly replacing validated baseline results.

---

**Academic quantitative finance research project. For educational purposes only.**