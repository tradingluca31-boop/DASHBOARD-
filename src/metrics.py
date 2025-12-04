"""
MQL5 Trading Analytics - Quantitative Metrics Module
Professional-grade trading metrics used by institutional traders
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Container for risk-adjusted performance metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    treynor_ratio: float


@dataclass
class DrawdownMetrics:
    """Drawdown analysis metrics"""
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: float
    current_drawdown: float
    recovery_factor: float
    ulcer_index: float


@dataclass
class TradeMetrics:
    """Trade-level statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_rr_ratio: float
    expectancy: float
    kelly_criterion: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration: float
    avg_bars_in_trade: int


class QuantMetricsCalculator:
    """
    Professional quantitative metrics calculator
    Implements institutional-grade performance analytics
    """

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate

    def __init__(self, trades_df: pd.DataFrame, initial_balance: float = 10000):
        self.trades = trades_df
        self.initial_balance = initial_balance
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and validate trade data"""
        required_cols = ['profit', 'close_time']
        for col in required_cols:
            if col not in self.trades.columns:
                # Try alternative column names
                alt_names = {
                    'profit': ['Profit', 'PnL', 'pnl', 'net_profit', 'NetProfit'],
                    'close_time': ['CloseTime', 'close_date', 'CloseDate', 'exit_time', 'ExitTime']
                }
                for alt in alt_names.get(col, []):
                    if alt in self.trades.columns:
                        self.trades[col] = self.trades[alt]
                        break

        # Calculate cumulative equity
        self.trades = self.trades.sort_values('close_time').reset_index(drop=True)
        self.trades['cumulative_profit'] = self.trades['profit'].cumsum()
        self.trades['equity'] = self.initial_balance + self.trades['cumulative_profit']
        self.trades['returns'] = self.trades['profit'] / self.trades['equity'].shift(1).fillna(self.initial_balance)

        # Calculate running max for drawdown
        self.trades['running_max'] = self.trades['equity'].cummax()
        self.trades['drawdown'] = (self.trades['equity'] - self.trades['running_max']) / self.trades['running_max']
        self.trades['drawdown_abs'] = self.trades['equity'] - self.trades['running_max']

    def calculate_sharpe_ratio(self, annualize: bool = True) -> float:
        """
        Calculate Sharpe Ratio - risk-adjusted return metric
        Sharpe = (Return - Risk Free Rate) / Volatility
        """
        returns = self.trades['returns'].dropna()
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR)

        if returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / returns.std()

        if annualize:
            sharpe *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return round(sharpe, 4)

    def calculate_sortino_ratio(self, annualize: bool = True) -> float:
        """
        Calculate Sortino Ratio - penalizes only downside volatility
        Sortino = (Return - Risk Free Rate) / Downside Deviation
        """
        returns = self.trades['returns'].dropna()
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR)

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No losing trades

        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_std

        if annualize:
            sortino *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return round(sortino, 4)

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio - return vs max drawdown
        Calmar = Annualized Return / Max Drawdown
        """
        total_return = (self.trades['equity'].iloc[-1] - self.initial_balance) / self.initial_balance
        max_dd = abs(self.trades['drawdown'].min())

        if max_dd == 0:
            return float('inf')

        # Annualize return
        n_days = (self.trades['close_time'].max() - self.trades['close_time'].min()).days
        if n_days <= 0:
            n_days = len(self.trades)

        annualized_return = (1 + total_return) ** (365 / n_days) - 1

        return round(annualized_return / max_dd, 4)

    def calculate_omega_ratio(self, threshold: float = 0) -> float:
        """
        Calculate Omega Ratio - probability weighted ratio of gains vs losses
        """
        returns = self.trades['returns'].dropna()

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if losses.sum() == 0:
            return float('inf')

        return round(gains.sum() / losses.sum(), 4)

    def calculate_information_ratio(self, benchmark_returns: Optional[pd.Series] = None) -> float:
        """
        Calculate Information Ratio - active return vs tracking error
        """
        returns = self.trades['returns'].dropna()

        if benchmark_returns is None:
            # Use zero as benchmark (absolute return)
            benchmark_returns = pd.Series([0] * len(returns))

        active_returns = returns.values - benchmark_returns.values[:len(returns)]
        tracking_error = np.std(active_returns)

        if tracking_error == 0:
            return 0.0

        return round(np.mean(active_returns) / tracking_error * np.sqrt(self.TRADING_DAYS_PER_YEAR), 4)

    def calculate_max_drawdown(self) -> Tuple[float, int]:
        """
        Calculate Maximum Drawdown and its duration
        Returns: (max_dd_percentage, duration_in_trades)
        """
        equity = self.trades['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        max_dd = drawdowns.min()

        # Calculate duration
        max_dd_idx = np.argmin(drawdowns)

        # Find peak before max drawdown
        peak_idx = np.argmax(equity[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

        # Find recovery point after max drawdown
        recovery_idx = max_dd_idx
        for i in range(max_dd_idx, len(equity)):
            if equity[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break
        else:
            recovery_idx = len(equity) - 1

        duration = recovery_idx - peak_idx

        return round(max_dd * 100, 2), duration

    def calculate_ulcer_index(self) -> float:
        """
        Calculate Ulcer Index - measures downside volatility
        Focuses on depth and duration of drawdowns
        """
        drawdowns = self.trades['drawdown'].values * 100
        ulcer = np.sqrt(np.mean(drawdowns ** 2))
        return round(ulcer, 4)

    def calculate_profit_factor(self) -> float:
        """
        Calculate Profit Factor - gross profits / gross losses
        """
        profits = self.trades[self.trades['profit'] > 0]['profit'].sum()
        losses = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())

        if losses == 0:
            return float('inf')

        return round(profits / losses, 4)

    def calculate_expectancy(self) -> float:
        """
        Calculate Mathematical Expectancy per trade
        E = (Win% × AvgWin) - (Loss% × AvgLoss)
        """
        wins = self.trades[self.trades['profit'] > 0]['profit']
        losses = self.trades[self.trades['profit'] < 0]['profit']

        win_rate = len(wins) / len(self.trades) if len(self.trades) > 0 else 0
        loss_rate = 1 - win_rate

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return round(expectancy, 2)

    def calculate_kelly_criterion(self) -> float:
        """
        Calculate Kelly Criterion - optimal position sizing
        Kelly% = W - [(1-W) / R]
        where W = win rate, R = win/loss ratio
        """
        wins = self.trades[self.trades['profit'] > 0]['profit']
        losses = self.trades[self.trades['profit'] < 0]['profit']

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        win_rate = len(wins) / len(self.trades)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return 1.0

        rr_ratio = avg_win / avg_loss

        kelly = win_rate - ((1 - win_rate) / rr_ratio)

        # Cap at reasonable levels
        kelly = max(0, min(kelly, 0.5))

        return round(kelly, 4)

    def calculate_consecutive_stats(self) -> Tuple[int, int]:
        """
        Calculate maximum consecutive wins and losses
        """
        is_win = (self.trades['profit'] > 0).astype(int)

        # Calculate consecutive wins
        win_groups = (is_win != is_win.shift()).cumsum()
        win_counts = is_win.groupby(win_groups).cumsum()
        max_consecutive_wins = win_counts[is_win == 1].max() if any(is_win == 1) else 0

        # Calculate consecutive losses
        is_loss = (self.trades['profit'] < 0).astype(int)
        loss_groups = (is_loss != is_loss.shift()).cumsum()
        loss_counts = is_loss.groupby(loss_groups).cumsum()
        max_consecutive_losses = loss_counts[is_loss == 1].max() if any(is_loss == 1) else 0

        return int(max_consecutive_wins), int(max_consecutive_losses)

    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk at given confidence level
        """
        returns = self.trades['returns'].dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        return round(var * 100, 4)

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        """
        returns = self.trades['returns'].dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        return round(cvar * 100, 4)

    def calculate_skewness_kurtosis(self) -> Tuple[float, float]:
        """
        Calculate distribution shape metrics
        """
        returns = self.trades['returns'].dropna()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        return round(skew, 4), round(kurt, 4)

    def calculate_recovery_factor(self) -> float:
        """
        Calculate Recovery Factor - Net Profit / Max Drawdown
        """
        net_profit = self.trades['equity'].iloc[-1] - self.initial_balance
        max_dd_abs = abs(self.trades['drawdown_abs'].min())

        if max_dd_abs == 0:
            return float('inf')

        return round(net_profit / max_dd_abs, 4)

    def calculate_tail_ratio(self) -> float:
        """
        Calculate Tail Ratio - 95th percentile / 5th percentile
        Measures asymmetry of return distribution
        """
        returns = self.trades['returns'].dropna()
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)

        if p5 == 0:
            return float('inf') if p95 > 0 else 0

        return round(abs(p95 / p5), 4)

    def calculate_gain_to_pain_ratio(self) -> float:
        """
        Calculate Gain to Pain Ratio
        Sum of all returns / abs(sum of negative returns)
        """
        returns = self.trades['returns'].dropna()
        total_returns = returns.sum()
        pain = abs(returns[returns < 0].sum())

        if pain == 0:
            return float('inf')

        return round(total_returns / pain, 4)

    def get_all_metrics(self) -> Dict:
        """
        Calculate and return all metrics as a dictionary
        """
        max_dd, max_dd_duration = self.calculate_max_drawdown()
        consecutive_wins, consecutive_losses = self.calculate_consecutive_stats()
        skew, kurt = self.calculate_skewness_kurtosis()

        wins = self.trades[self.trades['profit'] > 0]
        losses = self.trades[self.trades['profit'] < 0]

        return {
            # Performance Metrics
            'total_trades': len(self.trades),
            'net_profit': round(self.trades['equity'].iloc[-1] - self.initial_balance, 2),
            'total_return_pct': round((self.trades['equity'].iloc[-1] - self.initial_balance) / self.initial_balance * 100, 2),
            'win_rate': round(len(wins) / len(self.trades) * 100, 2) if len(self.trades) > 0 else 0,
            'profit_factor': self.calculate_profit_factor(),
            'expectancy': self.calculate_expectancy(),

            # Risk Metrics
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'omega_ratio': self.calculate_omega_ratio(),
            'information_ratio': self.calculate_information_ratio(),

            # Drawdown Metrics
            'max_drawdown_pct': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'ulcer_index': self.calculate_ulcer_index(),
            'recovery_factor': self.calculate_recovery_factor(),

            # Trade Statistics
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'avg_win': round(wins['profit'].mean(), 2) if len(wins) > 0 else 0,
            'avg_loss': round(losses['profit'].mean(), 2) if len(losses) > 0 else 0,
            'largest_win': round(wins['profit'].max(), 2) if len(wins) > 0 else 0,
            'largest_loss': round(losses['profit'].min(), 2) if len(losses) > 0 else 0,
            'avg_rr_ratio': round(abs(wins['profit'].mean() / losses['profit'].mean()), 2) if len(losses) > 0 and len(wins) > 0 else 0,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,

            # Risk Measures
            'var_95': self.calculate_var(0.95),
            'cvar_95': self.calculate_cvar(0.95),
            'kelly_criterion': self.calculate_kelly_criterion(),

            # Distribution Stats
            'skewness': skew,
            'kurtosis': kurt,
            'tail_ratio': self.calculate_tail_ratio(),
            'gain_to_pain': self.calculate_gain_to_pain_ratio(),
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing
    """

    def __init__(self, trades: pd.Series, initial_balance: float = 10000):
        self.trades = trades.values
        self.initial_balance = initial_balance

    def run_simulation(self, n_simulations: int = 1000, n_trades: int = None) -> Dict:
        """
        Run Monte Carlo simulation with trade reshuffling
        """
        if n_trades is None:
            n_trades = len(self.trades)

        results = {
            'final_equity': [],
            'max_drawdown': [],
            'sharpe': []
        }

        for _ in range(n_simulations):
            # Randomly reshuffle trades
            shuffled = np.random.choice(self.trades, size=n_trades, replace=True)

            # Calculate equity curve
            equity = self.initial_balance + np.cumsum(shuffled)

            # Calculate metrics
            results['final_equity'].append(equity[-1])

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            results['max_drawdown'].append(abs(drawdowns.min()) * 100)

            # Simple Sharpe approximation
            returns = shuffled / np.roll(equity, 1)
            returns[0] = shuffled[0] / self.initial_balance
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            results['sharpe'].append(sharpe)

        return {
            'median_final_equity': np.median(results['final_equity']),
            'equity_5th_percentile': np.percentile(results['final_equity'], 5),
            'equity_95th_percentile': np.percentile(results['final_equity'], 95),
            'median_max_dd': np.median(results['max_drawdown']),
            'max_dd_95th_percentile': np.percentile(results['max_drawdown'], 95),
            'median_sharpe': np.median(results['sharpe']),
            'probability_of_profit': sum(1 for e in results['final_equity'] if e > self.initial_balance) / n_simulations * 100,
            'probability_of_ruin': sum(1 for e in results['final_equity'] if e < self.initial_balance * 0.5) / n_simulations * 100,
            'all_results': results
        }


class TimeAnalyzer:
    """
    Time-based analysis of trading performance
    """

    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df.copy()
        self._prepare_time_features()

    def _prepare_time_features(self):
        """Extract time features from trade data"""
        if 'close_time' in self.trades.columns:
            self.trades['close_time'] = pd.to_datetime(self.trades['close_time'])
            self.trades['hour'] = self.trades['close_time'].dt.hour
            self.trades['day_of_week'] = self.trades['close_time'].dt.dayofweek
            self.trades['month'] = self.trades['close_time'].dt.month
            self.trades['week'] = self.trades['close_time'].dt.isocalendar().week
            self.trades['year'] = self.trades['close_time'].dt.year

    def get_hourly_performance(self) -> pd.DataFrame:
        """Analyze performance by hour of day"""
        if 'hour' not in self.trades.columns:
            return pd.DataFrame()

        hourly = self.trades.groupby('hour').agg({
            'profit': ['sum', 'mean', 'count', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        hourly.columns = ['total_profit', 'avg_profit', 'trade_count', 'win_rate']
        return hourly

    def get_daily_performance(self) -> pd.DataFrame:
        """Analyze performance by day of week"""
        if 'day_of_week' not in self.trades.columns:
            return pd.DataFrame()

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = self.trades.groupby('day_of_week').agg({
            'profit': ['sum', 'mean', 'count', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        daily.columns = ['total_profit', 'avg_profit', 'trade_count', 'win_rate']
        daily.index = [days[i] for i in daily.index]
        return daily

    def get_monthly_performance(self) -> pd.DataFrame:
        """Analyze performance by month"""
        if 'month' not in self.trades.columns:
            return pd.DataFrame()

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly = self.trades.groupby('month').agg({
            'profit': ['sum', 'mean', 'count', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        monthly.columns = ['total_profit', 'avg_profit', 'trade_count', 'win_rate']
        monthly.index = [months[i-1] for i in monthly.index]
        return monthly

    def get_weekly_returns(self) -> pd.DataFrame:
        """Get weekly aggregated returns"""
        if 'year' not in self.trades.columns:
            return pd.DataFrame()

        weekly = self.trades.groupby(['year', 'week']).agg({
            'profit': 'sum'
        }).reset_index()
        weekly.columns = ['year', 'week', 'profit']
        return weekly


class SymbolAnalyzer:
    """
    Symbol/Instrument-based analysis
    """

    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df.copy()

    def get_symbol_performance(self) -> pd.DataFrame:
        """Analyze performance by trading symbol"""
        symbol_col = None
        for col in ['symbol', 'Symbol', 'instrument', 'Instrument', 'pair', 'Pair']:
            if col in self.trades.columns:
                symbol_col = col
                break

        if symbol_col is None:
            return pd.DataFrame()

        symbol_stats = self.trades.groupby(symbol_col).agg({
            'profit': ['sum', 'mean', 'count',
                      lambda x: (x > 0).sum() / len(x) * 100,
                      lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0,
                      lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0]
        }).round(2)
        symbol_stats.columns = ['total_profit', 'avg_profit', 'trade_count', 'win_rate', 'avg_win', 'avg_loss']
        symbol_stats = symbol_stats.sort_values('total_profit', ascending=False)
        return symbol_stats
