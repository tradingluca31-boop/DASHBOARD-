"""
MQL5 Trading Analytics Package
Professional Wall Street Grade Analysis Tools
"""

from .data_loader import MQL5DataLoader, DataValidator, create_sample_data
from .metrics import QuantMetricsCalculator, MonteCarloSimulator, TimeAnalyzer, SymbolAnalyzer
from .visualizations import (
    create_equity_curve,
    create_drawdown_chart,
    create_monthly_returns_heatmap,
    create_hourly_performance,
    create_profit_distribution,
    create_cumulative_pnl_by_symbol,
    create_win_loss_streak,
    create_trade_scatter,
    create_rolling_metrics,
    create_monte_carlo_chart,
    create_symbol_performance_bar,
    create_risk_reward_scatter,
    create_trade_duration_analysis
)
from .advanced_analytics import AdvancedAnalytics, RiskManager

__all__ = [
    'MQL5DataLoader',
    'DataValidator',
    'create_sample_data',
    'QuantMetricsCalculator',
    'MonteCarloSimulator',
    'TimeAnalyzer',
    'SymbolAnalyzer',
    'AdvancedAnalytics',
    'RiskManager',
    'create_equity_curve',
    'create_drawdown_chart',
    'create_monthly_returns_heatmap',
    'create_hourly_performance',
    'create_profit_distribution',
    'create_cumulative_pnl_by_symbol',
    'create_win_loss_streak',
    'create_trade_scatter',
    'create_rolling_metrics',
    'create_monte_carlo_chart',
    'create_symbol_performance_bar',
    'create_risk_reward_scatter',
    'create_trade_duration_analysis'
]

__version__ = '1.0.0'
