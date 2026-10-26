
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DATA_DIR, OUTPUT_DIR, ETF_UNIVERSE,
    TEST_START, TEST_END,
    VALIDATION_START, VALIDATION_END
)

st.set_page_config(
    page_title="Dynamic Top 4 ETF Rotation",
    page_icon="📈",
    layout="wide"
)

STRATEGIES = [
    "Equal Weight",
    "Inverse Volatility",
    "Risk Parity",
    "SPY Buy & Hold",
    "Equal Weight 12 Buy & Hold"
]

ACTIVE = STRATEGIES[:3]

PERIODS = {
    "Full History": (
        None, TEST_END,
        "full_history_five_strategy_comparison.csv"
    ),
    "Validation": (
        VALIDATION_START, VALIDATION_END,
        "validation_five_strategy_comparison.csv"
    ),
    "Test Period": (
        TEST_START, TEST_END,
        "test_period_five_strategy_comparison.csv"
    )
}


@st.cache_data
def load_data():

    returns_active = pd.read_csv(
        OUTPUT_DIR / "three_strategy_daily_returns.csv",
        index_col=0, parse_dates=True
    )

    returns_bench = pd.read_csv(
        OUTPUT_DIR / "benchmark_daily_returns.csv",
        index_col=0, parse_dates=True
    )

    returns = pd.concat(
        [returns_active, returns_bench],
        axis=1, join="inner"
    )[STRATEGIES]

    allocations = pd.read_csv(
        DATA_DIR / "weekly_allocations.csv",
        parse_dates=["Signal_Date", "Execution_Date"]
    )

    signals = pd.read_csv(
        DATA_DIR / "weekly_target_weights.csv",
        parse_dates=["Signal_Date", "Execution_Date"]
    )

    equity = pd.read_csv(
        OUTPUT_DIR / "three_strategy_equity.csv",
        index_col=0, parse_dates=True
    )

    trades = {}

    for method in ACTIVE:
        name = method.lower().replace(" ", "_")
        trades[method] = pd.read_csv(
            OUTPUT_DIR / f"{name}_trades.csv",
            parse_dates=["Signal_Date", "Execution_Date"]
        )

    return returns, allocations, signals, equity, trades


returns, allocations, signals, equity, trades = load_data()


# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("Strategy Controls")

period_name = st.sidebar.selectbox(
    "Evaluation Period",
    list(PERIODS.keys()),
    index=2
)

selected_strategies = st.sidebar.multiselect(
    "Strategies to Display",
    STRATEGIES,
    default=STRATEGIES
)

allocation_method = st.sidebar.selectbox(
    "Allocation Detail",
    ACTIVE
)

start, end, performance_file = PERIODS[period_name]

period_returns = returns.loc[start:end]

performance = pd.read_csv(
    OUTPUT_DIR / performance_file,
    index_col=0
)


# ==========================================
# Dashboard Header
# ==========================================

st.title("Dynamic Top 4 ETF Momentum Rotation")

st.caption(
    "Weekly Momentum Selection | "
    "Equal Weight vs Inverse Volatility vs Risk Parity | "
    "SPY Market Timing | 10 bps Transaction Cost"
)

st.subheader(f"Performance Overview — {period_name}")

if not selected_strategies:
    st.warning("Please select at least one strategy.")
    st.stop()


# ==========================================
# Performance KPI
# ==========================================

highlight = allocation_method

kpi = performance.loc[highlight]

kpi_values = [
    ("Cumulative Return", f"{kpi['Cumulative Return']:.2%}"),
    ("Annual Return", f"{kpi['Annual Return']:.2%}"),
    ("Sharpe Ratio", f"{kpi['Annual Sharpe Ratio']:.3f}"),
    ("Annual Volatility", f"{kpi['Annual Volatility']:.2%}"),
    ("Max Drawdown", f"{kpi['Max Drawdown']:.2%}")
]

cols = st.columns(5)

for col, (label, value) in zip(cols, kpi_values):
    col.metric(label, value)

st.caption(f"Highlighted allocation: {highlight}")


# ==========================================
# Performance Comparison Table
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Performance",
    "Dynamic Top 4",
    "Portfolio Allocation",
    "Trades & Costs",
    "Robustness Analysis",
    "Market Timing Attribution"
])

with tab1:

    st.subheader("Five-Strategy Performance Comparison")

    formatted = performance.loc[selected_strategies].copy()

    for col in [
        "Cumulative Return",
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown"
    ]:
        formatted[col] = formatted[col].map(
            lambda x: f"{x:.2%}"
        )

    formatted["Annual Sharpe Ratio"] = (
        formatted["Annual Sharpe Ratio"].map(
            lambda x: f"{x:.3f}"
        )
    )

    st.dataframe(formatted, use_container_width=True)

    # All curves rebased at the start of the selected period
    period_wealth = (
        1 + period_returns[selected_strategies]
    ).cumprod()

    period_wealth = pd.concat([
        pd.DataFrame(
            1.0,
            index=[period_wealth.index[0] - pd.Timedelta(days=1)],
            columns=period_wealth.columns
        ),
        period_wealth
    ])

    st.subheader("Normalized Portfolio Wealth")

    fig = px.line(
        period_wealth,
        labels={"value": "Portfolio Wealth", "index": "Date"},
    )

    fig.update_layout(
        legend_title_text="Strategy",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown Comparison")

    peak = period_wealth.cummax()
    drawdown = period_wealth / peak - 1

    fig_dd = px.line(
        drawdown,
        labels={"value": "Drawdown", "index": "Date"}
    )

    fig_dd.update_layout(
        legend_title_text="Strategy",
        hovermode="x unified",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig_dd, use_container_width=True)


# ==========================================
# Dynamic Top 4
# ==========================================

with tab2:

    st.subheader("Weekly Top 4 ETF Selection")

    display_signals = signals[
        signals["Method"] == "Equal Weight"
    ].copy()

    display_signals = display_signals[
        display_signals["Signal_Date"].between(
            start or returns.index[0],
            end
        )
    ]

    columns = [
        "Signal_Date", "Execution_Date",
        "Market_Regime",
        "Top_1", "Top_2", "Top_3", "Top_4"
    ]

    st.dataframe(
        display_signals[columns].sort_values(
            "Signal_Date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Market Regime Distribution")

    regime_counts = display_signals[
        "Market_Regime"
    ].value_counts()

    fig_regime = px.pie(
        values=regime_counts.values,
        names=regime_counts.index,
        hole=0.45
    )

    st.plotly_chart(fig_regime, use_container_width=True)


# ==========================================
# Allocation Comparison
# ==========================================

with tab3:

    st.subheader("Weekly Allocation Comparison")

    dates_available = sorted(
        allocations["Signal_Date"].unique(),
        reverse=True
    )

    chosen_date = st.selectbox(
        "Choose a Signal Date",
        dates_available,
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
    )

    weekly = allocations[
        allocations["Signal_Date"] == chosen_date
    ]

    rows = []

    for _, row in weekly.iterrows():

        for i in range(1, 5):
            rows.append({
                "Method": row["Method"],
                "ETF": row[f"ETF_{i}"],
                "Weight": row[f"Weight_{i}"]
            })

    allocation_chart = pd.DataFrame(rows)

    fig_alloc = px.bar(
        allocation_chart,
        x="Method",
        y="Weight",
        color="ETF",
        barmode="stack",
        text_auto=".1%"
    )

    fig_alloc.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Portfolio Weight"
    )

    st.plotly_chart(fig_alloc, use_container_width=True)

    selected_signals = signals[
        signals["Signal_Date"] == chosen_date
    ]

    if not selected_signals.empty:
        regime = selected_signals[
            "Market_Regime"
        ].iloc[0]

        st.info(
            f"Market Regime: {regime}. "
            + (
                "Actual target is 100% BIL."
                if regime == "Risk-Off"
                else "Top 4 allocations are active."
            )
        )

    st.dataframe(
        allocation_chart.pivot(
            index="ETF",
            columns="Method",
            values="Weight"
        ).style.format("{:.2%}"),
        use_container_width=True
    )


# ==========================================
# Trades and Transaction Costs
# ==========================================

with tab4:

    st.subheader("Transaction Cost Analysis")

    cost_records = []

    for method in ACTIVE:

        df = trades[method]

        cost_records.append({
            "Method": method,
            "Total Cost": df["Transaction_Cost"].sum(),
            "Trade Records": len(df),
            "Total Traded Notional": df["Trade_Value"].sum()
        })

    cost_table = pd.DataFrame(cost_records)

    st.dataframe(
        cost_table,
        use_container_width=True,
        hide_index=True
    )

    fig_cost = px.bar(
        cost_table,
        x="Method",
        y="Total Cost",
        text_auto=",.2f"
    )

    st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader(f"Trade History — {allocation_method}")

    chosen_trades = trades[allocation_method].copy()

    chosen_trades = chosen_trades[
        chosen_trades["Execution_Date"].between(
            start or returns.index[0],
            end
        )
    ]

    chosen_trades = chosen_trades.sort_values(
        "Execution_Date", ascending=False
    )

    st.dataframe(
        chosen_trades,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Trade History CSV",
        data=chosen_trades.to_csv(index=False),
        file_name="trade_history.csv",
        mime="text/csv"
    )


# ==========================================
# Robustness Analysis
# ==========================================

with tab5:

    st.subheader("Strategy Robustness Analysis")

    st.caption(
        "One parameter is varied at a time. "
        "Other strategy settings remain fixed."
    )

    test_type = st.selectbox(
        "Robustness Test",
        [
            "Transaction Cost",
            "Momentum Windows",
            "Top N Selection",
            "Market Timing"
        ]
    )

    robustness_period = st.selectbox(
        "Evaluation Period",
        ["Full History", "Validation", "Test Period"],
        index=1
    )

    metric_options = {
        "Cumulative Return": "Cumulative Return",
        "Annual Return": "Annual Return",
        "Sharpe Ratio": "Sharpe Ratio",
        "Annual Volatility": "Annual Volatility",
        "Max Drawdown": "Max Drawdown"
    }

    selected_metric = st.selectbox(
        "Performance Metric",
        list(metric_options.keys())
    )

    # --------------------------------------
    # Choose robustness dataset
    # --------------------------------------

    if test_type == "Transaction Cost":

        file = OUTPUT_DIR / "robustness_cost_sensitivity.csv"
        scenario_col = "Cost (bps)"
        scenario_order = [0, 5, 10, 20]

    elif test_type == "Momentum Windows":

        file = OUTPUT_DIR / "robustness_momentum_windows.csv"
        scenario_col = "Scenario"
        scenario_order = ["Short", "Baseline", "Long"]

    elif test_type == "Top N Selection":

        file = OUTPUT_DIR / "robustness_topn.csv"
        scenario_col = "Top N"
        scenario_order = [2, 3, 4]

    else:

        file = OUTPUT_DIR / "robustness_market_timing.csv"
        scenario_col = "Scenario"
        scenario_order = ["MA10", "MA20", "MA30"]

    # --------------------------------------
    # Load and filter data
    # --------------------------------------

    if not file.exists():

        st.warning(
            f"Robustness results not found: {file.name}"
        )

    else:

        robustness = pd.read_csv(file)

        if (
            test_type == "Top N Selection"
            and robustness_period == "Full History"
        ):
            st.info(
                "Top N analysis currently uses Validation "
                "and Test periods. Please select one of them."
            )

        else:

            filtered = robustness[
                robustness["Period"] == robustness_period
            ].copy()

            metric_col = metric_options[selected_metric]

            table = filtered.pivot(
                index=scenario_col,
                columns="Method",
                values=metric_col
            )

            table = table.reindex(
                index=scenario_order,
                columns=ACTIVE
            )

            st.subheader(
                f"{test_type} — {selected_metric}"
            )

            # ----------------------------------
            # Plot
            # ----------------------------------

            chart_data = table.reset_index().melt(
                id_vars=scenario_col,
                var_name="Allocation Method",
                value_name=selected_metric
            )

            fig = px.bar(
                chart_data,
                x=scenario_col,
                y=selected_metric,
                color="Allocation Method",
                barmode="group"
            )

            if selected_metric != "Sharpe Ratio":
                fig.update_yaxes(tickformat=".1%")

            fig.update_layout(
                hovermode="x unified",
                legend_title_text="Allocation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ----------------------------------
            # Comparison table
            # ----------------------------------

            st.subheader("Detailed Results")

            formatted = table.copy()

            if selected_metric == "Sharpe Ratio":
                formatted = formatted.map(
                    lambda x: f"{x:.3f}"
                )
            else:
                formatted = formatted.map(
                    lambda x: f"{x:.2%}"
                )

            st.dataframe(
                formatted,
                use_container_width=True
            )

            # ----------------------------------
            # Download
            # ----------------------------------

            st.download_button(
                "Download Robustness Results",
                data=filtered.to_csv(index=False),
                file_name=(
                    "robustness_"
                    + test_type.lower().replace(" ", "_")
                    + ".csv"
                ),
                mime="text/csv"
            )

    # --------------------------------------
    # Methodological Notes
    # --------------------------------------

    st.divider()

    st.markdown("**Methodological Notes**")

    if test_type == "Transaction Cost":
        st.write(
            "Transaction costs are applied to absolute "
            "traded notional. Each scenario is independently "
            "backtested, accounting for compounding effects."
        )

    elif test_type == "Momentum Windows":
        st.write(
            "Momentum scenarios use different 20/60/120-type "
            "lookback lengths but identical score weights. "
            "Full-history starting dates may differ."
        )

    elif test_type == "Top N Selection":
        st.write(
            "Top 2, Top 3, and Top 4 use maximum position "
            "weights of 75%, 50%, and 40%, respectively. "
            "The concentration constraint changes together "
            "with the number of selected ETFs."
        )

    else:
        st.write(
            "Market timing uses SPY adjusted closing prices "
            "and MA10, MA20, or MA30. The moving-average "
            "window is the only parameter varied."
        )

    st.caption(
        "Robustness comparisons are descriptive. "
        "The short October test period does not establish "
        "statistical significance or future performance."
    )


# ==========================================
# Market Timing Attribution
# ==========================================

with tab6:

    st.subheader("Market Timing Attribution")

    st.caption(
        "Full Strategy: Dynamic Top 4 with SPY MA20. "
        "Momentum-Only: Same Top 4 and allocation, "
        "without market timing."
    )

    attribution = pd.read_csv(
        OUTPUT_DIR / "attribution_market_timing.csv"
    )

    attr_period = st.selectbox(
        "Evaluation Period",
        ["Full History", "Validation", "Test Period"],
        index=1,
        key="attr_period"
    )

    attr_method = st.selectbox(
        "Allocation Method",
        ACTIVE,
        key="attr_method"
    )

    subset = attribution[
        (attribution["Period"] == attr_period)
        & (attribution["Method"] == attr_method)
    ].copy()

    # --------------------------------------
    # Performance Comparison
    # --------------------------------------

    st.markdown("#### Performance Comparison")

    metric_names = [
        "Cumulative Return",
        "Annual Return",
        "Sharpe Ratio",
        "Annual Volatility",
        "Max Drawdown"
    ]

    table = subset.set_index("Strategy")[metric_names].T

    
    # Convert to object dtype before formatting numbers as strings
    display_table = table.astype(object).copy()

    for metric in metric_names:

        if metric == "Sharpe Ratio":
            display_table.loc[metric] = (
                table.loc[metric].map(
                    lambda x: f"{x:.3f}"
                )
            )

        else:
            display_table.loc[metric] = (
                table.loc[metric].map(
                    lambda x: f"{x:.2%}"
                )
            )


    st.dataframe(
        display_table,
        use_container_width=True
    )

    # --------------------------------------
    # Cumulative Wealth
    # --------------------------------------

    def read_equity(folder, method):

        safe = method.lower().replace(" ", "_")

        df = pd.read_csv(
            folder / f"{safe}_equity.csv",
            index_col=0,
            parse_dates=True
        )

        return df["Daily_Return"]

    full_returns = read_equity(
        OUTPUT_DIR,
        attr_method
    )

    momentum_returns = read_equity(
        OUTPUT_DIR
        / "attribution_no_timing"
        / "outputs",
        attr_method
    )

    period_bounds = {
        "Full History": (None, TEST_END),
        "Validation": (
            VALIDATION_START, VALIDATION_END
        ),
        "Test Period": (
            TEST_START, TEST_END
        )
    }

    start_date, end_date = period_bounds[attr_period]

    comparison_returns = pd.DataFrame({
        "Full Strategy": full_returns,
        "Momentum-Only": momentum_returns
    }).loc[start_date:end_date]

    wealth = (1 + comparison_returns).cumprod()

    fig_wealth = px.line(
        wealth,
        title="Normalized Portfolio Wealth",
        labels={
            "index": "Date",
            "value": "Portfolio Wealth"
        }
    )

    fig_wealth.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_wealth,
        use_container_width=True
    )

    # --------------------------------------
    # Drawdown Comparison
    # --------------------------------------

    peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / peak - 1

    fig_dd = px.line(
        drawdown,
        title="Drawdown Comparison",
        labels={
            "index": "Date",
            "value": "Drawdown"
        }
    )

    fig_dd.update_yaxes(tickformat=".1%")

    st.plotly_chart(
        fig_dd,
        use_container_width=True
    )

    # --------------------------------------
    # Cost Comparison
    # --------------------------------------

    st.markdown("#### Full-History Transaction Costs")

    cost_table = subset[
        ["Strategy", "Total Historical Cost"]
    ].copy()

    st.dataframe(
        cost_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Transaction costs above cover the entire "
        "backtest, not only the selected period."
    )

    st.info(
        "Market timing can reduce drawdown but may "
        "also reduce returns through defensive "
        "positioning and additional trading costs."
    )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Backtest assumptions: adjusted prices, next-day-open "
    "execution, fractional shares, 10 bps per traded notional, "
    "zero risk-free rate. Historical results are not guarantees "
    "of future performance."
)
