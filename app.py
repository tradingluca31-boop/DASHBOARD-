"""
MQL5 TRADING ANALYTICS DASHBOARD
Professional Wall Street Grade Analysis Platform

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MQL5DataLoader, DataValidator, create_sample_data
from metrics import QuantMetricsCalculator, MonteCarloSimulator, TimeAnalyzer, SymbolAnalyzer
from visualizations import (
    create_equity_curve, create_drawdown_chart, create_monthly_returns_heatmap,
    create_hourly_performance, create_profit_distribution, create_cumulative_pnl_by_symbol,
    create_win_loss_streak, create_trade_scatter, create_rolling_metrics,
    create_monte_carlo_chart, create_symbol_performance_bar, create_risk_reward_scatter,
    create_trade_duration_analysis, COLORS
)

# Page configuration
st.set_page_config(
    page_title="MQL5 Trading Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Wall Street theme
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0f0f18 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121a 0%, #0a0a0f 100%);
        border-right: 1px solid #1e1e2e;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif !important;
    }

    h1 {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #12121a 0%, #1a1a24 100%);
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stMetricLabel"] {
        color: #8892a0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] svg {
        display: none;
    }

    /* Positive delta */
    [data-testid="stMetricDelta"]:has(div[data-testid="stMetricDeltaText-positive"]) {
        color: #00ff88 !important;
    }

    /* Negative delta */
    [data-testid="stMetricDelta"]:has(div[data-testid="stMetricDeltaText-negative"]) {
        color: #ff4757 !important;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #12121a 0%, #1a1a24 100%);
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .metric-card-title {
        color: #8892a0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .metric-card-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-card-value.positive {
        color: #00ff88;
    }

    .metric-card-value.negative {
        color: #ff4757;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #12121a;
        border: 2px dashed #1e1e2e;
        border-radius: 12px;
        padding: 20px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #00d4ff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: #12121a;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        color: #8892a0;
        padding: 10px 20px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: #1a1a24;
        color: #ffffff;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff20, #00ff8820) !important;
        border-color: #00d4ff !important;
        color: #ffffff !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #12121a !important;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        color: #ffffff !important;
    }

    /* Tables */
    .stDataFrame {
        background: #12121a;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        color: #0a0a0f;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }

    /* Selectbox */
    [data-testid="stSelectbox"] {
        background: #12121a;
        border-radius: 8px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
    }

    /* Divider */
    hr {
        border-color: #1e1e2e;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #00d4ff10, #00d4ff05);
        border-left: 4px solid #00d4ff;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #ffd70010, #ffd70005);
        border-left: 4px solid #ffd700;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }

    /* Logo styling */
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 20px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }

    ::-webkit-scrollbar-thumb {
        background: #1e1e2e;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #2e2e3e;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", is_positive: bool = None):
    """Render a custom metric card"""
    value_class = ""
    if is_positive is not None:
        value_class = "positive" if is_positive else "negative"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value {value_class}">{value}</div>
        <div style="color: #6b7280; font-size: 0.8rem; margin-top: 4px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format value as currency"""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:,.2f}%"


def get_metric_color(value: float, threshold: float = 0) -> str:
    """Get color based on value"""
    return COLORS['accent_green'] if value > threshold else COLORS['accent_red']


def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="logo-text">MQL5 ANALYTICS</p>', unsafe_allow_html=True)
        st.markdown("**Professional Trading Dashboard**")
        st.markdown("---")

        # File upload
        st.markdown("### 📁 Data Source")
        uploaded_file = st.file_uploader(
            "Upload your trades CSV",
            type=['csv', 'txt'],
            help="Export your trades from MT4/MT5 as CSV"
        )

        use_sample = st.checkbox("Use sample data for demo", value=uploaded_file is None)

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")
        initial_balance = st.number_input(
            "Initial Balance ($)",
            min_value=100,
            max_value=10000000,
            value=10000,
            step=1000
        )

        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        ) / 100

        monte_carlo_sims = st.select_slider(
            "Monte Carlo Simulations",
            options=[100, 500, 1000, 5000, 10000],
            value=1000
        )

        st.markdown("---")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.75rem;">
        <strong>Pro Tips:</strong><br>
        • Export trades from MT4/MT5 history<br>
        • Include all columns for best analysis<br>
        • Minimum 30 trades recommended
        </div>
        """, unsafe_allow_html=True)

    # Main content
    st.markdown("# 📊 Trading Performance Analytics")
    st.markdown("*Institutional-grade analysis for your MQL5 Expert Advisor*")

    # Load data
    if use_sample or uploaded_file is None:
        df = create_sample_data(150)
        st.info("📌 Using sample data. Upload your CSV for real analysis.")
    else:
        try:
            loader = MQL5DataLoader()
            df = loader.load_csv(uploaded_file)
            df, warnings = DataValidator.validate_trades(df)
            for w in warnings:
                st.warning(w)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

    # Calculate metrics
    calculator = QuantMetricsCalculator(df, initial_balance)
    calculator.RISK_FREE_RATE = risk_free_rate
    metrics = calculator.get_all_metrics()

    # Time analyzer
    time_analyzer = TimeAnalyzer(df)

    # Symbol analyzer
    symbol_analyzer = SymbolAnalyzer(df)

    # ===== OVERVIEW SECTION =====
    st.markdown("## 💎 Performance Overview")

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        net_pnl = metrics['net_profit']
        st.metric(
            "Net P/L",
            format_currency(net_pnl),
            f"{metrics['total_return_pct']:+.2f}%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1f}%",
            f"{metrics['winning_trades']}W / {metrics['losing_trades']}L"
        )

    with col3:
        st.metric(
            "Profit Factor",
            f"{metrics['profit_factor']:.2f}",
            "Good" if metrics['profit_factor'] > 1.5 else "Needs work"
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            "Excellent" if metrics['sharpe_ratio'] > 2 else "Good" if metrics['sharpe_ratio'] > 1 else "Low"
        )

    with col5:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown_pct']:.2f}%",
            f"{metrics['max_drawdown_duration']} trades"
        )

    # Secondary metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
    with col2:
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
    with col3:
        st.metric("Expectancy", format_currency(metrics['expectancy']))
    with col4:
        st.metric("Avg R:R", f"{metrics['avg_rr_ratio']:.2f}")
    with col5:
        st.metric("Recovery Factor", f"{metrics['recovery_factor']:.2f}")
    with col6:
        st.metric("Kelly %", f"{metrics['kelly_criterion']*100:.1f}%")

    st.markdown("---")

    # ===== CHARTS SECTION =====
    tabs = st.tabs([
        "📈 Equity & Drawdown",
        "📊 Performance Analysis",
        "⏰ Time Analysis",
        "💱 Symbol Analysis",
        "🎲 Monte Carlo",
        "📋 Trade Log"
    ])

    # Tab 1: Equity & Drawdown
    with tabs[0]:
        st.plotly_chart(create_equity_curve(df, initial_balance), use_container_width=True)
        st.plotly_chart(create_drawdown_chart(df, initial_balance), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_trade_scatter(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_win_loss_streak(df), use_container_width=True)

    # Tab 2: Performance Analysis
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_profit_distribution(df), use_container_width=True)

        with col2:
            st.plotly_chart(create_monthly_returns_heatmap(df), use_container_width=True)

        st.plotly_chart(create_rolling_metrics(df), use_container_width=True)

        # Risk metrics detail
        st.markdown("### 📊 Detailed Risk Metrics")
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

        with risk_col1:
            st.metric("VaR (95%)", f"{metrics['var_95']:.2f}%")
            st.metric("Skewness", f"{metrics['skewness']:.2f}")

        with risk_col2:
            st.metric("CVaR (95%)", f"{metrics['cvar_95']:.2f}%")
            st.metric("Kurtosis", f"{metrics['kurtosis']:.2f}")

        with risk_col3:
            st.metric("Ulcer Index", f"{metrics['ulcer_index']:.2f}")
            st.metric("Tail Ratio", f"{metrics['tail_ratio']:.2f}")

        with risk_col4:
            st.metric("Omega Ratio", f"{metrics['omega_ratio']:.2f}")
            st.metric("Gain to Pain", f"{metrics['gain_to_pain']:.2f}")

    # Tab 3: Time Analysis
    with tabs[2]:
        st.plotly_chart(create_hourly_performance(df), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📅 Daily Performance")
            daily_perf = time_analyzer.get_daily_performance()
            if not daily_perf.empty:
                st.dataframe(
                    daily_perf.style.format({
                        'total_profit': '${:,.2f}',
                        'avg_profit': '${:,.2f}',
                        'trade_count': '{:.0f}',
                        'win_rate': '{:.1f}%'
                    }).background_gradient(cmap='RdYlGn', subset=['total_profit']),
                    use_container_width=True
                )

        with col2:
            st.markdown("### 📊 Hourly Performance")
            hourly_perf = time_analyzer.get_hourly_performance()
            if not hourly_perf.empty:
                st.dataframe(
                    hourly_perf.style.format({
                        'total_profit': '${:,.2f}',
                        'avg_profit': '${:,.2f}',
                        'trade_count': '{:.0f}',
                        'win_rate': '{:.1f}%'
                    }).background_gradient(cmap='RdYlGn', subset=['total_profit']),
                    use_container_width=True,
                    height=400
                )

        if 'duration_minutes' in df.columns or ('open_time' in df.columns and 'close_time' in df.columns):
            st.plotly_chart(create_trade_duration_analysis(df), use_container_width=True)

    # Tab 4: Symbol Analysis
    with tabs[3]:
        if 'symbol' in df.columns:
            st.plotly_chart(create_symbol_performance_bar(df), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(create_cumulative_pnl_by_symbol(df), use_container_width=True)

            with col2:
                st.plotly_chart(create_risk_reward_scatter(df), use_container_width=True)

            st.markdown("### 💰 Symbol Statistics")
            symbol_stats = symbol_analyzer.get_symbol_performance()
            if not symbol_stats.empty:
                st.dataframe(
                    symbol_stats.style.format({
                        'total_profit': '${:,.2f}',
                        'avg_profit': '${:,.2f}',
                        'trade_count': '{:.0f}',
                        'win_rate': '{:.1f}%',
                        'avg_win': '${:,.2f}',
                        'avg_loss': '${:,.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['total_profit']),
                    use_container_width=True
                )
        else:
            st.warning("No symbol data available in your CSV. Add a 'symbol' column for symbol analysis.")

    # Tab 5: Monte Carlo
    with tabs[4]:
        st.markdown("### 🎲 Monte Carlo Simulation")
        st.markdown("""
        Monte Carlo simulation tests the robustness of your strategy by randomly reordering trades
        to see the range of possible outcomes. This helps understand the role of luck vs skill.
        """)

        if st.button("Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Running {monte_carlo_sims:,} simulations..."):
                mc_simulator = MonteCarloSimulator(df['profit'], initial_balance)
                mc_results = mc_simulator.run_simulation(n_simulations=monte_carlo_sims)

                st.plotly_chart(create_monte_carlo_chart(mc_results), use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Median Final Equity",
                        format_currency(mc_results['median_final_equity'])
                    )
                with col2:
                    st.metric(
                        "5th Percentile",
                        format_currency(mc_results['equity_5th_percentile']),
                        "Worst case scenario"
                    )
                with col3:
                    st.metric(
                        "95th Percentile",
                        format_currency(mc_results['equity_95th_percentile']),
                        "Best case scenario"
                    )
                with col4:
                    st.metric(
                        "Probability of Profit",
                        f"{mc_results['probability_of_profit']:.1f}%"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Median Max Drawdown",
                        f"{mc_results['median_max_dd']:.2f}%"
                    )
                with col2:
                    st.metric(
                        "95th %ile Max DD",
                        f"{mc_results['max_dd_95th_percentile']:.2f}%",
                        "Worst expected drawdown"
                    )

                if mc_results['probability_of_ruin'] > 0:
                    st.error(f"⚠️ Probability of Ruin (losing 50%+): {mc_results['probability_of_ruin']:.1f}%")

    # Tab 6: Trade Log
    with tabs[5]:
        st.markdown("### 📋 Trade History")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'symbol' in df.columns:
                symbols = ['All'] + list(df['symbol'].unique())
                selected_symbol = st.selectbox("Filter by Symbol", symbols)

        with col2:
            if 'type' in df.columns:
                types = ['All'] + list(df['type'].unique())
                selected_type = st.selectbox("Filter by Type", types)

        with col3:
            profit_filter = st.selectbox("Filter by Result", ['All', 'Winners', 'Losers'])

        # Apply filters
        filtered_df = df.copy()

        if 'symbol' in df.columns and selected_symbol != 'All':
            filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]

        if 'type' in df.columns and selected_type != 'All':
            filtered_df = filtered_df[filtered_df['type'] == selected_type]

        if profit_filter == 'Winners':
            filtered_df = filtered_df[filtered_df['profit'] > 0]
        elif profit_filter == 'Losers':
            filtered_df = filtered_df[filtered_df['profit'] < 0]

        # Display columns
        display_cols = ['close_time', 'symbol', 'type', 'volume', 'profit']
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].sort_values('close_time', ascending=False).head(100).style.format({
                'profit': '${:,.2f}',
                'volume': '{:.2f}'
            }).applymap(
                lambda x: f'color: {COLORS["accent_green"]}' if isinstance(x, (int, float)) and x > 0
                else f'color: {COLORS["accent_red"]}' if isinstance(x, (int, float)) and x < 0
                else '',
                subset=['profit']
            ),
            use_container_width=True,
            height=500
        )

        st.markdown(f"*Showing {min(100, len(filtered_df))} of {len(filtered_df)} trades*")

    # ===== RECOMMENDATIONS SECTION =====
    st.markdown("---")
    st.markdown("## 🎯 EA Optimization Recommendations")

    recommendations = []

    # Win rate analysis
    if metrics['win_rate'] < 40:
        recommendations.append({
            'type': 'warning',
            'title': 'Low Win Rate',
            'text': f"Your win rate ({metrics['win_rate']:.1f}%) is below 40%. Consider tightening entry criteria or reviewing your signal logic."
        })
    elif metrics['win_rate'] > 70:
        recommendations.append({
            'type': 'info',
            'title': 'High Win Rate',
            'text': f"Excellent win rate ({metrics['win_rate']:.1f}%)! Ensure your R:R ratio ({metrics['avg_rr_ratio']:.2f}) supports long-term profitability."
        })

    # Profit factor analysis
    if metrics['profit_factor'] < 1.2:
        recommendations.append({
            'type': 'error',
            'title': 'Low Profit Factor',
            'text': f"Profit factor ({metrics['profit_factor']:.2f}) is concerning. Target minimum 1.5 for robust strategies."
        })

    # Drawdown analysis
    if abs(metrics['max_drawdown_pct']) > 20:
        recommendations.append({
            'type': 'warning',
            'title': 'High Drawdown',
            'text': f"Max drawdown ({metrics['max_drawdown_pct']:.1f}%) exceeds 20%. Consider reducing position size or adding drawdown limits."
        })

    # Sharpe analysis
    if metrics['sharpe_ratio'] < 1:
        recommendations.append({
            'type': 'warning',
            'title': 'Low Risk-Adjusted Returns',
            'text': f"Sharpe ratio ({metrics['sharpe_ratio']:.2f}) below 1 indicates suboptimal risk-adjusted returns."
        })

    # Kelly criterion
    if metrics['kelly_criterion'] > 0.25:
        recommendations.append({
            'type': 'info',
            'title': 'Position Sizing',
            'text': f"Kelly criterion suggests {metrics['kelly_criterion']*100:.1f}% risk per trade. Use half-Kelly ({metrics['kelly_criterion']*50:.1f}%) for safety."
        })

    # Consecutive losses
    if metrics['consecutive_losses'] > 5:
        recommendations.append({
            'type': 'warning',
            'title': 'Losing Streaks',
            'text': f"Maximum {metrics['consecutive_losses']} consecutive losses detected. Ensure your position sizing can handle extended drawdowns."
        })

    # Display recommendations
    for rec in recommendations:
        if rec['type'] == 'error':
            st.error(f"**{rec['title']}**: {rec['text']}")
        elif rec['type'] == 'warning':
            st.warning(f"**{rec['title']}**: {rec['text']}")
        else:
            st.info(f"**{rec['title']}**: {rec['text']}")

    if not recommendations:
        st.success("✅ Your EA shows solid performance metrics! Continue monitoring and consider forward testing.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 20px;">
        <p>MQL5 Trading Analytics Dashboard | Professional Wall Street Grade Analysis</p>
        <p style="font-size: 0.75rem;">Built for quantitative traders and EA developers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
