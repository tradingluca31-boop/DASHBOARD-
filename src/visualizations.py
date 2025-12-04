"""
MQL5 Trading Analytics - Professional Visualization Module
Wall Street grade charts and visualizations using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


# Professional Wall Street color palette
COLORS = {
    'bg_dark': '#0a0a0f',
    'bg_card': '#12121a',
    'bg_secondary': '#1a1a24',
    'text_primary': '#ffffff',
    'text_secondary': '#8892a0',
    'accent_blue': '#00d4ff',
    'accent_green': '#00ff88',
    'accent_red': '#ff4757',
    'accent_gold': '#ffd700',
    'accent_purple': '#a855f7',
    'grid': '#1e1e2e',
    'profit': '#00ff88',
    'loss': '#ff4757',
    'neutral': '#6b7280',
}

# Chart layout template
LAYOUT_TEMPLATE = {
    'paper_bgcolor': COLORS['bg_card'],
    'plot_bgcolor': COLORS['bg_card'],
    'font': {'family': 'Inter, SF Pro Display, -apple-system, sans-serif', 'color': COLORS['text_primary']},
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid'],
        'tickfont': {'color': COLORS['text_secondary']},
        'titlefont': {'color': COLORS['text_secondary']}
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid'],
        'tickfont': {'color': COLORS['text_secondary']},
        'titlefont': {'color': COLORS['text_secondary']}
    },
    'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
}


def apply_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply professional layout to figure"""
    fig.update_layout(
        paper_bgcolor=COLORS['bg_card'],
        plot_bgcolor=COLORS['bg_card'],
        font=dict(family='Inter, SF Pro Display, -apple-system, sans-serif', color=COLORS['text_primary']),
        title=title,
        height=height,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_secondary'], size=10),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hoverlabel=dict(
            bgcolor=COLORS['bg_secondary'],
            font_size=12,
            font_family='Inter, sans-serif'
        )
    )
    return fig


def create_equity_curve(df: pd.DataFrame, initial_balance: float = 10000) -> go.Figure:
    """
    Create professional equity curve with benchmark comparison
    """
    df = df.copy().sort_values('close_time').reset_index(drop=True)
    df['cumulative_profit'] = df['profit'].cumsum()
    df['equity'] = initial_balance + df['cumulative_profit']
    df['running_max'] = df['equity'].cummax()

    fig = go.Figure()

    # Add equity curve
    fig.add_trace(go.Scatter(
        x=df['close_time'],
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color=COLORS['accent_blue'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)',
        hovertemplate='<b>Equity:</b> $%{y:,.2f}<br><b>Date:</b> %{x}<extra></extra>'
    ))

    # Add high water mark
    fig.add_trace(go.Scatter(
        x=df['close_time'],
        y=df['running_max'],
        mode='lines',
        name='High Water Mark',
        line=dict(color=COLORS['accent_gold'], width=1, dash='dot'),
        hovertemplate='<b>Peak:</b> $%{y:,.2f}<extra></extra>'
    ))

    # Add initial balance line
    fig.add_hline(
        y=initial_balance,
        line_dash="dash",
        line_color=COLORS['text_secondary'],
        annotation_text="Initial Balance",
        annotation_position="right"
    )

    fig = apply_layout(fig, "📈 Equity Curve", 450)
    fig.update_yaxes(tickformat='$,.0f')

    return fig


def create_drawdown_chart(df: pd.DataFrame, initial_balance: float = 10000) -> go.Figure:
    """
    Create underwater equity / drawdown chart
    """
    df = df.copy().sort_values('close_time').reset_index(drop=True)
    df['cumulative_profit'] = df['profit'].cumsum()
    df['equity'] = initial_balance + df['cumulative_profit']
    df['running_max'] = df['equity'].cummax()
    df['drawdown_pct'] = ((df['equity'] - df['running_max']) / df['running_max']) * 100

    fig = go.Figure()

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=df['close_time'],
        y=df['drawdown_pct'],
        mode='lines',
        name='Drawdown',
        line=dict(color=COLORS['accent_red'], width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 71, 87, 0.3)',
        hovertemplate='<b>Drawdown:</b> %{y:.2f}%<br><b>Date:</b> %{x}<extra></extra>'
    ))

    # Add threshold lines
    for threshold, alpha in [(-5, 0.3), (-10, 0.2), (-20, 0.1)]:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=f'rgba(255, 71, 87, {alpha})',
            annotation_text=f"{threshold}%",
            annotation_position="right"
        )

    fig = apply_layout(fig, "📉 Drawdown Analysis", 350)
    fig.update_yaxes(ticksuffix='%')

    return fig


def create_monthly_returns_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create monthly returns heatmap like professional trading platforms
    """
    df = df.copy()
    df['close_time'] = pd.to_datetime(df['close_time'])
    df['year'] = df['close_time'].dt.year
    df['month'] = df['close_time'].dt.month

    # Aggregate monthly returns
    monthly = df.groupby(['year', 'month'])['profit'].sum().reset_index()
    monthly_pivot = monthly.pivot(index='year', columns='month', values='profit')

    # Fill missing months with 0
    all_months = list(range(1, 13))
    for m in all_months:
        if m not in monthly_pivot.columns:
            monthly_pivot[m] = np.nan
    monthly_pivot = monthly_pivot[sorted(monthly_pivot.columns)]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Custom colorscale (red for negative, green for positive)
    colorscale = [
        [0.0, COLORS['accent_red']],
        [0.5, COLORS['bg_secondary']],
        [1.0, COLORS['accent_green']]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values,
        x=month_names,
        y=monthly_pivot.index.astype(str),
        colorscale=colorscale,
        zmid=0,
        text=monthly_pivot.values,
        texttemplate='$%{text:.0f}',
        textfont={'size': 10, 'color': 'white'},
        hovertemplate='<b>%{y} %{x}</b><br>P/L: $%{z:,.2f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text='P/L ($)', font=dict(color=COLORS['text_secondary'])),
            tickfont=dict(color=COLORS['text_secondary'])
        )
    ))

    fig = apply_layout(fig, "📅 Monthly Returns Heatmap", 300)

    return fig


def create_hourly_performance(df: pd.DataFrame) -> go.Figure:
    """
    Create hourly performance heatmap
    """
    df = df.copy()
    df['close_time'] = pd.to_datetime(df['close_time'])
    df['hour'] = df['close_time'].dt.hour
    df['day_of_week'] = df['close_time'].dt.dayofweek

    hourly = df.groupby(['day_of_week', 'hour']).agg({
        'profit': 'sum'
    }).reset_index()

    hourly_pivot = hourly.pivot(index='day_of_week', columns='hour', values='profit')

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    colorscale = [
        [0.0, COLORS['accent_red']],
        [0.5, COLORS['bg_secondary']],
        [1.0, COLORS['accent_green']]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=hourly_pivot.values,
        x=[f'{h:02d}:00' for h in range(24)],
        y=[days[i] for i in hourly_pivot.index],
        colorscale=colorscale,
        zmid=0,
        hovertemplate='<b>%{y} %{x}</b><br>P/L: $%{z:,.2f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text='P/L ($)', font=dict(color=COLORS['text_secondary'])),
            tickfont=dict(color=COLORS['text_secondary'])
        )
    ))

    fig = apply_layout(fig, "⏰ Performance by Day & Hour", 350)

    return fig


def create_profit_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create profit/loss distribution histogram with statistics
    """
    profits = df['profit'].dropna()

    fig = go.Figure()

    # Create histogram
    fig.add_trace(go.Histogram(
        x=profits,
        nbinsx=50,
        name='Trade P/L',
        marker=dict(
            color=[COLORS['accent_green'] if x > 0 else COLORS['accent_red'] for x in profits],
            line=dict(color=COLORS['bg_dark'], width=0.5)
        ),
        hovertemplate='<b>P/L Range:</b> $%{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))

    # Add mean line
    mean_profit = profits.mean()
    fig.add_vline(
        x=mean_profit,
        line_dash="dash",
        line_color=COLORS['accent_blue'],
        annotation_text=f"Mean: ${mean_profit:.2f}",
        annotation_position="top"
    )

    # Add zero line
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=COLORS['text_secondary'],
        annotation_text="Breakeven"
    )

    fig = apply_layout(fig, "📊 Trade P/L Distribution", 350)
    fig.update_xaxes(title='Profit/Loss ($)')
    fig.update_yaxes(title='Number of Trades')

    return fig


def create_cumulative_pnl_by_symbol(df: pd.DataFrame) -> go.Figure:
    """
    Create cumulative P/L chart broken down by symbol
    """
    if 'symbol' not in df.columns:
        return go.Figure()

    df = df.copy().sort_values('close_time')

    fig = go.Figure()

    symbols = df['symbol'].unique()
    colors = px.colors.qualitative.Set2

    for i, symbol in enumerate(symbols):
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df['cumulative'] = symbol_df['profit'].cumsum()

        fig.add_trace(go.Scatter(
            x=symbol_df['close_time'],
            y=symbol_df['cumulative'],
            mode='lines',
            name=symbol,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'<b>{symbol}</b><br>Cumulative: $%{{y:,.2f}}<extra></extra>'
        ))

    fig = apply_layout(fig, "💱 Cumulative P/L by Symbol", 400)
    fig.update_yaxes(tickformat='$,.0f')

    return fig


def create_win_loss_streak(df: pd.DataFrame) -> go.Figure:
    """
    Create win/loss streak visualization
    """
    df = df.copy().sort_values('close_time').reset_index(drop=True)
    df['is_win'] = (df['profit'] > 0).astype(int)

    # Calculate streaks
    streaks = []
    current_streak = 0
    streak_type = None

    for _, row in df.iterrows():
        is_win = row['is_win']

        if streak_type is None:
            streak_type = is_win
            current_streak = 1
        elif is_win == streak_type:
            current_streak += 1
        else:
            streaks.append({'type': 'win' if streak_type else 'loss', 'length': current_streak})
            streak_type = is_win
            current_streak = 1

    if current_streak > 0:
        streaks.append({'type': 'win' if streak_type else 'loss', 'length': current_streak})

    streak_df = pd.DataFrame(streaks)

    fig = go.Figure()

    colors = [COLORS['accent_green'] if t == 'win' else COLORS['accent_red'] for t in streak_df['type']]

    fig.add_trace(go.Bar(
        x=list(range(len(streak_df))),
        y=streak_df['length'],
        marker_color=colors,
        hovertemplate='<b>Streak %{x}</b><br>Type: %{customdata}<br>Length: %{y}<extra></extra>',
        customdata=streak_df['type']
    ))

    fig = apply_layout(fig, "🔥 Win/Loss Streaks", 300)
    fig.update_xaxes(title='Streak Number')
    fig.update_yaxes(title='Streak Length')

    return fig


def create_trade_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Create scatter plot of trades with size based on profit magnitude
    """
    df = df.copy().sort_values('close_time').reset_index(drop=True)

    colors = [COLORS['accent_green'] if p > 0 else COLORS['accent_red'] for p in df['profit']]
    sizes = np.abs(df['profit']) / np.abs(df['profit']).max() * 30 + 5

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['close_time'],
        y=df['profit'],
        mode='markers',
        marker=dict(
            color=colors,
            size=sizes,
            opacity=0.7,
            line=dict(color=COLORS['bg_dark'], width=1)
        ),
        hovertemplate='<b>Trade</b><br>P/L: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="solid", line_color=COLORS['text_secondary'])

    fig = apply_layout(fig, "🎯 Individual Trade Performance", 400)
    fig.update_yaxes(tickformat='$,.0f', title='Profit/Loss')

    return fig


def create_rolling_metrics(df: pd.DataFrame, window: int = 20) -> go.Figure:
    """
    Create rolling performance metrics chart
    """
    df = df.copy().sort_values('close_time').reset_index(drop=True)

    # Calculate rolling metrics
    df['rolling_win_rate'] = df['profit'].apply(lambda x: 1 if x > 0 else 0).rolling(window).mean() * 100
    df['rolling_avg_profit'] = df['profit'].rolling(window).mean()
    df['rolling_profit_factor'] = df.apply(
        lambda x: None, axis=1
    )  # Placeholder - complex calculation

    # Calculate rolling Sharpe
    df['rolling_sharpe'] = df['profit'].rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'Rolling Win Rate ({window} trades)', f'Rolling Sharpe Ratio ({window} trades)')
    )

    # Win rate
    fig.add_trace(go.Scatter(
        x=df['close_time'],
        y=df['rolling_win_rate'],
        mode='lines',
        name='Win Rate',
        line=dict(color=COLORS['accent_blue'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ), row=1, col=1)

    fig.add_hline(y=50, line_dash="dash", line_color=COLORS['text_secondary'], row=1, col=1)

    # Sharpe ratio
    fig.add_trace(go.Scatter(
        x=df['close_time'],
        y=df['rolling_sharpe'],
        mode='lines',
        name='Sharpe Ratio',
        line=dict(color=COLORS['accent_gold'], width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 215, 0, 0.1)'
    ), row=2, col=1)

    fig.add_hline(y=1, line_dash="dash", line_color=COLORS['accent_green'], row=2, col=1,
                  annotation_text="Good (>1)")
    fig.add_hline(y=2, line_dash="dash", line_color=COLORS['accent_gold'], row=2, col=1,
                  annotation_text="Excellent (>2)")

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        height=500,
        showlegend=True
    )

    return fig


def create_monte_carlo_chart(simulation_results: Dict) -> go.Figure:
    """
    Create Monte Carlo simulation visualization
    """
    if 'all_results' not in simulation_results:
        return go.Figure()

    final_equities = simulation_results['all_results']['final_equity']

    fig = go.Figure()

    # Histogram of final equities
    fig.add_trace(go.Histogram(
        x=final_equities,
        nbinsx=50,
        name='Final Equity Distribution',
        marker=dict(
            color=COLORS['accent_blue'],
            line=dict(color=COLORS['bg_dark'], width=0.5)
        ),
        opacity=0.7
    ))

    # Add percentile lines
    p5 = simulation_results['equity_5th_percentile']
    p50 = simulation_results['median_final_equity']
    p95 = simulation_results['equity_95th_percentile']

    for val, label, color in [(p5, '5th %ile', COLORS['accent_red']),
                               (p50, 'Median', COLORS['accent_gold']),
                               (p95, '95th %ile', COLORS['accent_green'])]:
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                      annotation_text=f"{label}: ${val:,.0f}")

    fig = apply_layout(fig, "🎲 Monte Carlo Simulation - Final Equity Distribution", 400)
    fig.update_xaxes(tickformat='$,.0f', title='Final Equity')
    fig.update_yaxes(title='Frequency')

    return fig


def create_symbol_performance_bar(df: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart of performance by symbol
    """
    if 'symbol' not in df.columns:
        return go.Figure()

    symbol_stats = df.groupby('symbol').agg({
        'profit': ['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    symbol_stats.columns = ['total_profit', 'trade_count', 'win_rate']
    symbol_stats = symbol_stats.sort_values('total_profit')

    colors = [COLORS['accent_green'] if x > 0 else COLORS['accent_red']
              for x in symbol_stats['total_profit']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=symbol_stats['total_profit'],
        y=symbol_stats.index,
        orientation='h',
        marker_color=colors,
        text=[f"${x:,.0f} | {n} trades | {w:.1f}% WR"
              for x, n, w in zip(symbol_stats['total_profit'],
                                symbol_stats['trade_count'],
                                symbol_stats['win_rate'])],
        textposition='auto',
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{y}</b><br>Total P/L: $%{x:,.2f}<extra></extra>'
    ))

    fig.add_vline(x=0, line_color=COLORS['text_secondary'])

    fig = apply_layout(fig, "💰 Performance by Symbol", max(300, len(symbol_stats) * 40))
    fig.update_xaxes(tickformat='$,.0f', title='Total P/L')

    return fig


def create_risk_reward_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Create risk/reward analysis scatter plot
    """
    if 'symbol' not in df.columns:
        return go.Figure()

    # Calculate per-symbol stats
    symbol_stats = []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        wins = symbol_df[symbol_df['profit'] > 0]['profit']
        losses = symbol_df[symbol_df['profit'] < 0]['profit']

        if len(wins) > 0 and len(losses) > 0:
            symbol_stats.append({
                'symbol': symbol,
                'avg_win': wins.mean(),
                'avg_loss': abs(losses.mean()),
                'win_rate': len(wins) / len(symbol_df) * 100,
                'trade_count': len(symbol_df),
                'total_profit': symbol_df['profit'].sum()
            })

    if not symbol_stats:
        return go.Figure()

    stats_df = pd.DataFrame(symbol_stats)

    fig = go.Figure()

    colors = [COLORS['accent_green'] if x > 0 else COLORS['accent_red']
              for x in stats_df['total_profit']]

    fig.add_trace(go.Scatter(
        x=stats_df['avg_loss'],
        y=stats_df['avg_win'],
        mode='markers+text',
        marker=dict(
            size=stats_df['trade_count'] / stats_df['trade_count'].max() * 50 + 10,
            color=colors,
            opacity=0.7,
            line=dict(color='white', width=1)
        ),
        text=stats_df['symbol'],
        textposition='top center',
        textfont=dict(color=COLORS['text_primary'], size=10),
        hovertemplate='<b>%{text}</b><br>Avg Win: $%{y:.2f}<br>Avg Loss: $%{x:.2f}<br>Win Rate: %{customdata:.1f}%<extra></extra>',
        customdata=stats_df['win_rate']
    ))

    # Add R:R ratio lines
    max_val = max(stats_df['avg_win'].max(), stats_df['avg_loss'].max())
    for ratio, color, dash in [(1, COLORS['text_secondary'], 'solid'),
                                (2, COLORS['accent_gold'], 'dash'),
                                (3, COLORS['accent_green'], 'dot')]:
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val * ratio],
            mode='lines',
            name=f'{ratio}:1 R:R',
            line=dict(color=color, dash=dash, width=1),
            showlegend=True
        ))

    fig = apply_layout(fig, "⚖️ Risk/Reward Analysis by Symbol", 450)
    fig.update_xaxes(title='Average Loss ($)', tickformat='$,.0f')
    fig.update_yaxes(title='Average Win ($)', tickformat='$,.0f')

    return fig


def create_trade_duration_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Analyze trade duration vs profitability
    """
    if 'duration_minutes' not in df.columns:
        if 'open_time' in df.columns and 'close_time' in df.columns:
            df = df.copy()
            df['duration_minutes'] = (pd.to_datetime(df['close_time']) -
                                      pd.to_datetime(df['open_time'])).dt.total_seconds() / 60
        else:
            return go.Figure()

    df = df.copy()
    df['duration_hours'] = df['duration_minutes'] / 60

    # Bin durations
    bins = [0, 1, 4, 24, 72, 168, float('inf')]
    labels = ['<1h', '1-4h', '4-24h', '1-3d', '3-7d', '>7d']
    df['duration_bin'] = pd.cut(df['duration_hours'], bins=bins, labels=labels)

    duration_stats = df.groupby('duration_bin').agg({
        'profit': ['sum', 'mean', 'count', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    duration_stats.columns = ['total_profit', 'avg_profit', 'count', 'win_rate']

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=('Total Profit by Duration', 'Win Rate by Duration')
    )

    colors = [COLORS['accent_green'] if x > 0 else COLORS['accent_red']
              for x in duration_stats['total_profit']]

    fig.add_trace(go.Bar(
        x=duration_stats.index.astype(str),
        y=duration_stats['total_profit'],
        marker_color=colors,
        name='Total Profit',
        text=[f"${x:,.0f}" for x in duration_stats['total_profit']],
        textposition='outside'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=duration_stats.index.astype(str),
        y=duration_stats['win_rate'],
        marker_color=COLORS['accent_blue'],
        name='Win Rate',
        text=[f"{x:.1f}%" for x in duration_stats['win_rate']],
        textposition='outside'
    ), row=1, col=2)

    fig.add_hline(y=50, line_dash="dash", line_color=COLORS['text_secondary'],
                  row=1, col=2, annotation_text="50%")

    fig.update_layout(
        **LAYOUT_TEMPLATE,
        height=400,
        showlegend=False
    )

    return fig
