"""
MQL5 Trading Analytics - Advanced Analytics Module
Institutional-grade quantitative analysis tools
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class StrategyBenchmark:
    """Benchmark comparison results"""
    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    information_ratio: float


class AdvancedAnalytics:
    """
    Advanced quantitative analytics for trading strategies
    """

    def __init__(self, trades_df: pd.DataFrame, initial_balance: float = 10000):
        self.trades = trades_df.copy()
        self.initial_balance = initial_balance
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for analysis"""
        self.trades = self.trades.sort_values('close_time').reset_index(drop=True)
        self.trades['cumulative_profit'] = self.trades['profit'].cumsum()
        self.trades['equity'] = self.initial_balance + self.trades['cumulative_profit']
        self.trades['returns'] = self.trades['profit'] / self.trades['equity'].shift(1).fillna(self.initial_balance)
        self.trades['log_returns'] = np.log(self.trades['equity'] / self.trades['equity'].shift(1).fillna(self.initial_balance))

    def regime_detection(self, n_regimes: int = 2) -> pd.DataFrame:
        """
        Detect market regimes based on volatility clustering
        Uses rolling volatility to identify different market states
        """
        returns = self.trades['returns'].dropna()

        if len(returns) < 20:
            return pd.DataFrame()

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=10).std()

        # Simple regime classification based on volatility percentiles
        vol_percentiles = rolling_vol.quantile([0.33, 0.67])

        def classify_regime(vol):
            if pd.isna(vol):
                return 'Unknown'
            elif vol <= vol_percentiles.iloc[0]:
                return 'Low Volatility'
            elif vol <= vol_percentiles.iloc[1]:
                return 'Medium Volatility'
            else:
                return 'High Volatility'

        self.trades['regime'] = rolling_vol.apply(classify_regime)

        # Analyze performance per regime
        regime_performance = self.trades.groupby('regime').agg({
            'profit': ['sum', 'mean', 'count', lambda x: (x > 0).sum() / len(x) * 100],
            'returns': ['mean', 'std']
        }).round(4)

        regime_performance.columns = ['total_pnl', 'avg_pnl', 'trade_count', 'win_rate', 'avg_return', 'volatility']

        return regime_performance

    def calculate_optimal_position_size(self, target_vol: float = 0.02) -> Dict:
        """
        Calculate optimal position sizing using volatility targeting
        """
        returns = self.trades['returns'].dropna()

        if len(returns) < 20:
            return {}

        # Current volatility
        current_vol = returns.rolling(20).std().iloc[-1]

        # Vol-targeted position size
        vol_target_multiplier = target_vol / current_vol if current_vol > 0 else 1

        # Kelly criterion
        wins = self.trades[self.trades['profit'] > 0]['profit']
        losses = self.trades[self.trades['profit'] < 0]['profit']

        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(self.trades)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss)) if avg_loss > 0 else 0
            kelly = max(0, min(kelly, 1))
        else:
            kelly = 0

        # Optimal f (Ralph Vince)
        returns_array = returns.values
        if len(returns_array) > 0:
            def negative_twr(f):
                if f <= 0 or f >= 1:
                    return float('inf')
                holding_period_returns = 1 + f * returns_array / abs(returns_array.min())
                if any(holding_period_returns <= 0):
                    return float('inf')
                return -np.prod(holding_period_returns)

            try:
                result = minimize(negative_twr, 0.1, bounds=[(0.01, 0.99)])
                optimal_f = result.x[0] if result.success else kelly
            except:
                optimal_f = kelly
        else:
            optimal_f = kelly

        return {
            'current_volatility': round(current_vol * 100, 4),
            'vol_target_multiplier': round(vol_target_multiplier, 4),
            'kelly_criterion': round(kelly * 100, 2),
            'half_kelly': round(kelly * 50, 2),
            'optimal_f': round(optimal_f * 100, 2),
            'recommended_risk_pct': round(min(kelly * 50, optimal_f * 50, 2), 2)  # Conservative estimate
        }

    def analyze_trade_dependency(self) -> Dict:
        """
        Analyze if trades are independent or show autocorrelation
        """
        returns = self.trades['returns'].dropna()
        is_win = (self.trades['profit'] > 0).astype(int)

        if len(returns) < 30:
            return {}

        # Autocorrelation of returns
        autocorr_returns = []
        for lag in range(1, min(11, len(returns) // 3)):
            autocorr_returns.append({
                'lag': lag,
                'autocorr': returns.autocorr(lag=lag)
            })

        # Autocorrelation of wins/losses
        autocorr_wins = []
        for lag in range(1, min(11, len(is_win) // 3)):
            autocorr_wins.append({
                'lag': lag,
                'autocorr': is_win.autocorr(lag=lag)
            })

        # Runs test for randomness
        runs, n1, n2 = 0, 0, 0
        for i in range(len(is_win)):
            if is_win.iloc[i] == 1:
                n1 += 1
            else:
                n2 += 1
            if i > 0 and is_win.iloc[i] != is_win.iloc[i-1]:
                runs += 1
        runs += 1  # Count the last run

        # Expected runs under randomness
        n = n1 + n2
        expected_runs = (2 * n1 * n2) / n + 1 if n > 0 else 0
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))) if n > 1 else 1

        z_score = (runs - expected_runs) / std_runs if std_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'returns_autocorr': autocorr_returns,
            'wins_autocorr': autocorr_wins,
            'runs_test': {
                'observed_runs': runs,
                'expected_runs': round(expected_runs, 2),
                'z_score': round(z_score, 4),
                'p_value': round(p_value, 4),
                'is_random': p_value > 0.05
            },
            'conclusion': 'Trades appear independent' if p_value > 0.05 else 'Trades show dependency - consider this in position sizing'
        }

    def calculate_drawdown_analytics(self) -> Dict:
        """
        Comprehensive drawdown analysis
        """
        equity = self.trades['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        # Find all drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_starts = []
        drawdown_ends = []

        for i in range(1, len(in_drawdown)):
            if in_drawdown[i] and not in_drawdown[i-1]:
                drawdown_starts.append(i)
            elif not in_drawdown[i] and in_drawdown[i-1]:
                drawdown_ends.append(i)

        # If still in drawdown, mark current position as end
        if len(drawdown_starts) > len(drawdown_ends):
            drawdown_ends.append(len(equity) - 1)

        # Analyze each drawdown
        drawdown_analysis = []
        for start, end in zip(drawdown_starts, drawdown_ends):
            dd_period = drawdowns[start:end+1]
            max_dd = dd_period.min()
            max_dd_idx = np.argmin(dd_period) + start

            drawdown_analysis.append({
                'start_idx': start,
                'end_idx': end,
                'trough_idx': max_dd_idx,
                'max_drawdown': round(max_dd * 100, 2),
                'duration': end - start,
                'recovery_time': end - max_dd_idx
            })

        # Sort by severity
        drawdown_analysis = sorted(drawdown_analysis, key=lambda x: x['max_drawdown'])

        # Statistics
        if drawdown_analysis:
            dd_depths = [d['max_drawdown'] for d in drawdown_analysis]
            dd_durations = [d['duration'] for d in drawdown_analysis]

            return {
                'total_drawdowns': len(drawdown_analysis),
                'worst_drawdowns': drawdown_analysis[:5],  # Top 5 worst
                'avg_drawdown': round(np.mean(dd_depths), 2),
                'avg_duration': round(np.mean(dd_durations), 1),
                'max_drawdown': round(min(dd_depths), 2),
                'max_duration': max(dd_durations),
                'current_drawdown': round(drawdowns[-1] * 100, 2),
                'time_in_drawdown_pct': round(sum(in_drawdown) / len(in_drawdown) * 100, 2)
            }

        return {}

    def calculate_trade_efficiency(self) -> Dict:
        """
        Analyze trade efficiency metrics
        """
        df = self.trades.copy()

        # MAE/MFE analysis if available
        efficiency_metrics = {}

        # Basic efficiency
        wins = df[df['profit'] > 0]
        losses = df[df['profit'] < 0]

        if len(wins) > 0 and len(losses) > 0:
            # Edge ratio
            avg_win = wins['profit'].mean()
            avg_loss = abs(losses['profit'].mean())
            edge_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

            # Payoff ratio distribution
            win_percentiles = wins['profit'].quantile([0.25, 0.5, 0.75]).values
            loss_percentiles = losses['profit'].quantile([0.25, 0.5, 0.75]).values

            # Trade quality score (custom metric)
            win_rate = len(wins) / len(df)
            quality_score = (win_rate * edge_ratio) / (1 + abs(df['profit'].skew()))

            efficiency_metrics = {
                'edge_ratio': round(edge_ratio, 4),
                'win_percentiles': {
                    '25th': round(win_percentiles[0], 2),
                    '50th': round(win_percentiles[1], 2),
                    '75th': round(win_percentiles[2], 2)
                },
                'loss_percentiles': {
                    '25th': round(loss_percentiles[0], 2),
                    '50th': round(loss_percentiles[1], 2),
                    '75th': round(loss_percentiles[2], 2)
                },
                'trade_quality_score': round(quality_score, 4),
                'profit_concentration': round(
                    wins.nlargest(int(len(wins) * 0.2), 'profit')['profit'].sum() / wins['profit'].sum() * 100, 2
                ) if len(wins) > 5 else 0,  # % of profits from top 20% of trades
                'loss_concentration': round(
                    abs(losses.nsmallest(int(len(losses) * 0.2), 'profit')['profit'].sum()) / abs(losses['profit'].sum()) * 100, 2
                ) if len(losses) > 5 else 0  # % of losses from worst 20% of trades
            }

        return efficiency_metrics

    def calculate_stability_metrics(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling stability metrics to detect strategy degradation
        """
        df = self.trades.copy()

        if len(df) < window * 2:
            return pd.DataFrame()

        # Rolling metrics
        df['rolling_win_rate'] = df['profit'].apply(lambda x: 1 if x > 0 else 0).rolling(window).mean() * 100
        df['rolling_avg_profit'] = df['profit'].rolling(window).mean()
        df['rolling_volatility'] = df['returns'].rolling(window).std()

        # Rolling Sharpe
        df['rolling_sharpe'] = df['returns'].rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )

        # Profit factor rolling
        def rolling_pf(x):
            wins = x[x > 0].sum()
            losses = abs(x[x < 0].sum())
            return wins / losses if losses > 0 else float('inf')

        df['rolling_profit_factor'] = df['profit'].rolling(window).apply(rolling_pf)

        # Detect degradation (compare recent vs historical)
        recent_period = df.tail(window)
        historical = df.iloc[:-window]

        degradation_signals = []

        if len(historical) > 0:
            # Win rate degradation
            if recent_period['rolling_win_rate'].mean() < historical['rolling_win_rate'].mean() * 0.8:
                degradation_signals.append('Win rate declining')

            # Profit factor degradation
            recent_pf = recent_period['rolling_profit_factor'].replace([np.inf, -np.inf], np.nan).mean()
            hist_pf = historical['rolling_profit_factor'].replace([np.inf, -np.inf], np.nan).mean()
            if recent_pf < hist_pf * 0.7:
                degradation_signals.append('Profit factor declining')

            # Volatility increase
            if recent_period['rolling_volatility'].mean() > historical['rolling_volatility'].mean() * 1.5:
                degradation_signals.append('Volatility increasing')

        return {
            'rolling_metrics': df[['close_time', 'rolling_win_rate', 'rolling_avg_profit',
                                   'rolling_volatility', 'rolling_sharpe', 'rolling_profit_factor']].tail(50),
            'degradation_signals': degradation_signals,
            'is_degrading': len(degradation_signals) > 1
        }

    def calculate_market_correlation(self) -> Dict:
        """
        Analyze correlation with market sessions and volatility
        """
        df = self.trades.copy()

        if 'close_time' not in df.columns:
            return {}

        df['close_time'] = pd.to_datetime(df['close_time'])
        df['hour'] = df['close_time'].dt.hour

        # Define trading sessions (UTC)
        def get_session(hour):
            if 0 <= hour < 8:
                return 'Asian'
            elif 8 <= hour < 16:
                return 'European'
            else:
                return 'American'

        df['session'] = df['hour'].apply(get_session)

        session_analysis = df.groupby('session').agg({
            'profit': ['sum', 'mean', 'count', lambda x: (x > 0).sum() / len(x) * 100],
            'returns': ['std']
        }).round(4)

        session_analysis.columns = ['total_pnl', 'avg_pnl', 'trade_count', 'win_rate', 'volatility']

        # Best/worst sessions
        best_session = session_analysis['total_pnl'].idxmax()
        worst_session = session_analysis['total_pnl'].idxmin()

        return {
            'session_performance': session_analysis.to_dict(),
            'best_session': best_session,
            'worst_session': worst_session,
            'recommendation': f"Focus on {best_session} session, consider reducing {worst_session} exposure"
        }


class RiskManager:
    """
    Risk management analysis and recommendations
    """

    def __init__(self, trades_df: pd.DataFrame, initial_balance: float = 10000):
        self.trades = trades_df.copy()
        self.initial_balance = initial_balance

    def calculate_risk_of_ruin(self, risk_per_trade: float = 0.02, ruin_threshold: float = 0.5) -> Dict:
        """
        Calculate probability of ruin using analytical formula
        """
        wins = self.trades[self.trades['profit'] > 0]
        losses = self.trades[self.trades['profit'] < 0]

        if len(wins) == 0 or len(losses) == 0:
            return {}

        win_rate = len(wins) / len(self.trades)
        avg_win = wins['profit'].mean()
        avg_loss = abs(losses['profit'].mean())

        # Edge per trade
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Risk of ruin formula (simplified)
        if edge <= 0:
            risk_of_ruin = 1.0
        else:
            # Using exponential approximation
            a = avg_win / (avg_win + avg_loss)  # Win size ratio
            q = 1 - win_rate

            if win_rate > 0.5 and a > 0:
                risk_of_ruin = ((q / win_rate) ** (1 / risk_per_trade * (1 - ruin_threshold)))
            else:
                risk_of_ruin = min(1.0, q / win_rate if win_rate > 0 else 1.0)

        risk_of_ruin = min(max(risk_of_ruin, 0), 1)

        return {
            'risk_of_ruin': round(risk_of_ruin * 100, 2),
            'edge_per_trade': round(edge, 2),
            'risk_level': 'HIGH' if risk_of_ruin > 0.1 else 'MEDIUM' if risk_of_ruin > 0.01 else 'LOW',
            'recommendation': f"With {risk_per_trade*100}% risk per trade, your ruin probability is {risk_of_ruin*100:.2f}%"
        }

    def calculate_optimal_stop_loss(self) -> Dict:
        """
        Analyze optimal stop loss placement based on historical data
        """
        losses = self.trades[self.trades['profit'] < 0]['profit']

        if len(losses) < 10:
            return {}

        loss_percentiles = {
            '50th': abs(losses.quantile(0.5)),
            '75th': abs(losses.quantile(0.75)),
            '90th': abs(losses.quantile(0.9)),
            '95th': abs(losses.quantile(0.95))
        }

        # Recommendation based on distribution
        recommended_sl = abs(losses.quantile(0.75))

        return {
            'loss_distribution': loss_percentiles,
            'average_loss': round(abs(losses.mean()), 2),
            'max_loss': round(abs(losses.min()), 2),
            'recommended_stop_loss': round(recommended_sl, 2),
            'explanation': f"Setting SL at ${recommended_sl:.2f} would capture 75% of your historical losses"
        }

    def stress_test(self, scenarios: Dict[str, float] = None) -> Dict:
        """
        Perform stress testing on the strategy
        """
        if scenarios is None:
            scenarios = {
                'Flash Crash (-30% worst trade)': 0.3,
                'Extended Drawdown (2x max DD)': 2.0,
                'Win Rate Drop (-20%)': 0.2,
                'Volatility Spike (2x)': 2.0
            }

        results = {}
        current_equity = self.initial_balance + self.trades['profit'].sum()

        # Current metrics
        wins = self.trades[self.trades['profit'] > 0]
        losses = self.trades[self.trades['profit'] < 0]
        max_dd = abs((self.trades['profit'].cumsum() + self.initial_balance).min() - self.initial_balance)

        # Flash crash scenario
        worst_trade = abs(losses['profit'].min()) if len(losses) > 0 else 0
        flash_crash_impact = worst_trade * (1 + scenarios['Flash Crash (-30% worst trade)'])
        results['Flash Crash'] = {
            'impact': round(flash_crash_impact, 2),
            'equity_after': round(current_equity - flash_crash_impact, 2),
            'survival': current_equity - flash_crash_impact > self.initial_balance * 0.5
        }

        # Extended drawdown
        extended_dd = max_dd * scenarios['Extended Drawdown (2x max DD)']
        results['Extended Drawdown'] = {
            'impact': round(extended_dd, 2),
            'equity_after': round(current_equity - extended_dd, 2),
            'survival': current_equity - extended_dd > self.initial_balance * 0.5
        }

        # Win rate drop
        if len(wins) > 0 and len(losses) > 0:
            current_wr = len(wins) / len(self.trades)
            new_wr = current_wr * (1 - scenarios['Win Rate Drop (-20%)'])
            avg_win = wins['profit'].mean()
            avg_loss = abs(losses['profit'].mean())
            new_expectancy = new_wr * avg_win - (1 - new_wr) * avg_loss

            results['Win Rate Drop'] = {
                'new_win_rate': round(new_wr * 100, 2),
                'new_expectancy': round(new_expectancy, 2),
                'still_profitable': new_expectancy > 0
            }

        return results
