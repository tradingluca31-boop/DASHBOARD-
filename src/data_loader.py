"""
MQL5 Trading Analytics - Data Loading and Processing Module
Handles various CSV export formats from MT4/MT5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import re


class MQL5DataLoader:
    """
    Intelligent data loader for MQL5/MT4 trade exports
    Automatically detects and parses various CSV formats
    """

    # Common column name mappings for different export formats
    COLUMN_MAPPINGS = {
        'ticket': ['ticket', 'Ticket', 'Order', 'order', 'OrderTicket', 'Deal', 'deal', '#'],
        'open_time': ['open_time', 'OpenTime', 'Open Time', 'Entry Time', 'entry_time', 'EntryTime', 'Time'],
        'close_time': ['close_time', 'CloseTime', 'Close Time', 'Exit Time', 'exit_time', 'ExitTime'],
        'type': ['type', 'Type', 'Direction', 'direction', 'Side', 'side', 'Action'],
        'volume': ['volume', 'Volume', 'Lots', 'lots', 'Size', 'size', 'Quantity', 'qty'],
        'symbol': ['symbol', 'Symbol', 'Instrument', 'instrument', 'Pair', 'pair', 'Asset'],
        'open_price': ['open_price', 'OpenPrice', 'Open Price', 'Entry Price', 'entry_price', 'EntryPrice', 'Price'],
        'close_price': ['close_price', 'ClosePrice', 'Close Price', 'Exit Price', 'exit_price', 'ExitPrice'],
        'sl': ['sl', 'SL', 'Stop Loss', 'stop_loss', 'StopLoss', 'S/L'],
        'tp': ['tp', 'TP', 'Take Profit', 'take_profit', 'TakeProfit', 'T/P'],
        'profit': ['profit', 'Profit', 'PnL', 'pnl', 'Net Profit', 'net_profit', 'NetProfit', 'P/L', 'Result'],
        'commission': ['commission', 'Commission', 'Comm', 'comm'],
        'swap': ['swap', 'Swap', 'Rollover', 'rollover'],
        'comment': ['comment', 'Comment', 'Note', 'note', 'Magic', 'magic'],
        'magic': ['magic', 'Magic', 'MagicNumber', 'magic_number', 'EA'],
        'duration': ['duration', 'Duration', 'Holding Time', 'holding_time', 'HoldingTime'],
        'pips': ['pips', 'Pips', 'Points', 'points'],
    }

    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.detected_format: str = "unknown"
        self.column_mapping: Dict[str, str] = {}

    def load_csv(self, file_path: str, encoding: str = 'utf-8',
                 delimiter: str = None, skip_rows: int = 0) -> pd.DataFrame:
        """
        Load CSV file with automatic format detection
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try different delimiters if not specified
        delimiters = [delimiter] if delimiter else [',', ';', '\t', '|']

        for delim in delimiters:
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter=delim,
                    encoding=encoding,
                    skiprows=skip_rows,
                    on_bad_lines='skip'
                )

                # Check if parsing was successful (more than 1 column usually)
                if len(df.columns) > 1:
                    self.raw_data = df
                    self._detect_and_map_columns()
                    self._process_data()
                    return self.processed_data

            except Exception as e:
                continue

        # Try alternative encodings
        for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
                if len(df.columns) > 1:
                    self.raw_data = df
                    self._detect_and_map_columns()
                    self._process_data()
                    return self.processed_data
            except:
                continue

        raise ValueError("Could not parse CSV file. Please check the format.")

    def _detect_and_map_columns(self):
        """
        Automatically detect and map column names to standardized names
        """
        if self.raw_data is None:
            return

        columns = self.raw_data.columns.tolist()
        self.column_mapping = {}

        for standard_name, possible_names in self.COLUMN_MAPPINGS.items():
            for col in columns:
                # Clean column name for comparison
                col_clean = col.strip().lower().replace('_', ' ').replace('-', ' ')

                for possible in possible_names:
                    possible_clean = possible.lower().replace('_', ' ').replace('-', ' ')

                    if col_clean == possible_clean or col.strip() == possible:
                        self.column_mapping[standard_name] = col
                        break

                if standard_name in self.column_mapping:
                    break

        # Detect format based on available columns
        if 'ticket' in self.column_mapping and 'magic' in self.column_mapping:
            self.detected_format = "MT5_Full"
        elif 'ticket' in self.column_mapping:
            self.detected_format = "MT4_Standard"
        elif 'profit' in self.column_mapping:
            self.detected_format = "Simple_PnL"
        else:
            self.detected_format = "Custom"

    def _process_data(self):
        """
        Process and standardize the loaded data
        """
        if self.raw_data is None:
            return

        df = self.raw_data.copy()

        # Rename columns to standard names
        rename_map = {v: k for k, v in self.column_mapping.items()}
        df = df.rename(columns=rename_map)

        # Process datetime columns
        for col in ['open_time', 'close_time']:
            if col in df.columns:
                df[col] = self._parse_datetime(df[col])

        # Process numeric columns
        numeric_cols = ['profit', 'volume', 'open_price', 'close_price', 'sl', 'tp',
                       'commission', 'swap', 'pips']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Process trade type
        if 'type' in df.columns:
            df['type'] = df['type'].apply(self._standardize_trade_type)

        # Calculate additional fields if possible
        df = self._calculate_derived_fields(df)

        # Remove rows with no profit data
        if 'profit' in df.columns:
            df = df.dropna(subset=['profit'])

        # Sort by close time if available
        if 'close_time' in df.columns:
            df = df.sort_values('close_time').reset_index(drop=True)

        self.processed_data = df

    def _parse_datetime(self, series: pd.Series) -> pd.Series:
        """
        Parse datetime from various formats
        """
        # Common datetime formats in MT4/MT5 exports
        formats = [
            '%Y.%m.%d %H:%M:%S',
            '%Y.%m.%d %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d.%m.%Y %H:%M:%S',
            '%d.%m.%Y %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%Y%m%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(series, format=fmt)
            except:
                continue

        # Fallback to pandas auto-detection
        return pd.to_datetime(series, errors='coerce')

    def _standardize_trade_type(self, trade_type) -> str:
        """
        Standardize trade type to 'BUY' or 'SELL'
        """
        if pd.isna(trade_type):
            return 'UNKNOWN'

        trade_str = str(trade_type).upper().strip()

        buy_types = ['BUY', 'LONG', 'B', '0', 'OP_BUY']
        sell_types = ['SELL', 'SHORT', 'S', '1', 'OP_SELL']

        if any(t in trade_str for t in buy_types):
            return 'BUY'
        elif any(t in trade_str for t in sell_types):
            return 'SELL'
        else:
            return trade_str

    def _calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional useful fields from existing data
        """
        # Calculate duration if we have open and close times
        if 'open_time' in df.columns and 'close_time' in df.columns:
            df['duration_minutes'] = (df['close_time'] - df['open_time']).dt.total_seconds() / 60

        # Calculate pips if we have prices
        if all(col in df.columns for col in ['open_price', 'close_price', 'type']):
            df['price_diff'] = df.apply(
                lambda row: (row['close_price'] - row['open_price'])
                if row.get('type') == 'BUY'
                else (row['open_price'] - row['close_price']),
                axis=1
            )

            # Estimate pip value based on symbol (simplified)
            if 'symbol' in df.columns:
                df['pips_estimated'] = df.apply(
                    lambda row: self._calculate_pips(row),
                    axis=1
                )

        # Calculate gross profit (before commission/swap)
        if 'profit' in df.columns:
            df['gross_profit'] = df['profit']
            if 'commission' in df.columns:
                df['gross_profit'] = df['gross_profit'] + df['commission'].fillna(0).abs()
            if 'swap' in df.columns:
                df['gross_profit'] = df['gross_profit'] + df['swap'].fillna(0).abs()

        # Win/Loss flag
        if 'profit' in df.columns:
            df['is_winner'] = (df['profit'] > 0).astype(int)

        return df

    def _calculate_pips(self, row) -> float:
        """
        Calculate pips based on symbol type
        """
        if 'price_diff' not in row or pd.isna(row.get('price_diff')):
            return np.nan

        symbol = str(row.get('symbol', '')).upper()
        price_diff = row['price_diff']

        # JPY pairs (2 decimal places)
        if 'JPY' in symbol:
            return price_diff * 100
        # Gold (XAU)
        elif 'XAU' in symbol or 'GOLD' in symbol:
            return price_diff * 10
        # Most forex pairs (4-5 decimal places)
        else:
            return price_diff * 10000

    def get_summary(self) -> Dict:
        """
        Get a summary of the loaded data
        """
        if self.processed_data is None:
            return {"error": "No data loaded"}

        df = self.processed_data

        summary = {
            "total_trades": len(df),
            "detected_format": self.detected_format,
            "columns_detected": list(self.column_mapping.keys()),
            "date_range": None,
            "symbols": None,
            "trade_types": None
        }

        if 'close_time' in df.columns:
            summary["date_range"] = {
                "start": df['close_time'].min().strftime('%Y-%m-%d') if pd.notna(df['close_time'].min()) else None,
                "end": df['close_time'].max().strftime('%Y-%m-%d') if pd.notna(df['close_time'].max()) else None
            }

        if 'symbol' in df.columns:
            summary["symbols"] = df['symbol'].unique().tolist()

        if 'type' in df.columns:
            summary["trade_types"] = df['type'].value_counts().to_dict()

        return summary


class DataValidator:
    """
    Validate and clean trade data
    """

    @staticmethod
    def validate_trades(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate trade data and return cleaned data with warnings
        """
        warnings = []
        cleaned_df = df.copy()

        # Check for required columns
        if 'profit' not in cleaned_df.columns:
            warnings.append("WARNING: No profit column found. Some metrics cannot be calculated.")

        # Check for missing values in critical columns
        critical_cols = ['profit', 'close_time']
        for col in critical_cols:
            if col in cleaned_df.columns:
                missing = cleaned_df[col].isna().sum()
                if missing > 0:
                    warnings.append(f"WARNING: {missing} missing values in '{col}' column")
                    cleaned_df = cleaned_df.dropna(subset=[col])

        # Check for zero/negative volume
        if 'volume' in cleaned_df.columns:
            invalid_vol = (cleaned_df['volume'] <= 0).sum()
            if invalid_vol > 0:
                warnings.append(f"WARNING: {invalid_vol} trades with invalid volume")

        # Check for suspiciously large profits/losses
        if 'profit' in cleaned_df.columns:
            profit_std = cleaned_df['profit'].std()
            profit_mean = cleaned_df['profit'].mean()
            outliers = ((cleaned_df['profit'] > profit_mean + 5 * profit_std) |
                       (cleaned_df['profit'] < profit_mean - 5 * profit_std)).sum()
            if outliers > 0:
                warnings.append(f"INFO: {outliers} potential outlier trades detected (>5 std from mean)")

        # Check date ordering
        if 'close_time' in cleaned_df.columns:
            if not cleaned_df['close_time'].is_monotonic_increasing:
                warnings.append("INFO: Trades were re-ordered by close time")
                cleaned_df = cleaned_df.sort_values('close_time')

        return cleaned_df, warnings


def create_sample_data(n_trades: int = 100) -> pd.DataFrame:
    """
    Create sample trade data for testing
    """
    np.random.seed(42)

    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_trades, freq='4H')

    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    types = ['BUY', 'SELL']

    # Generate realistic PnL distribution (slightly positive skew)
    profits = np.random.normal(10, 50, n_trades)  # Mean $10, std $50
    # Add some bigger wins
    big_wins_idx = np.random.choice(n_trades, size=int(n_trades * 0.1), replace=False)
    profits[big_wins_idx] = np.random.uniform(100, 300, len(big_wins_idx))
    # Add some bigger losses
    big_loss_idx = np.random.choice([i for i in range(n_trades) if i not in big_wins_idx],
                                    size=int(n_trades * 0.05), replace=False)
    profits[big_loss_idx] = np.random.uniform(-200, -100, len(big_loss_idx))

    data = {
        'ticket': range(1000, 1000 + n_trades),
        'open_time': dates - pd.Timedelta(hours=np.random.randint(1, 48)),
        'close_time': dates,
        'type': np.random.choice(types, n_trades),
        'symbol': np.random.choice(symbols, n_trades),
        'volume': np.random.choice([0.01, 0.02, 0.05, 0.1, 0.2], n_trades),
        'open_price': np.random.uniform(1.0, 2.0, n_trades),
        'profit': profits.round(2),
        'commission': np.random.uniform(-2, -0.5, n_trades).round(2),
        'swap': np.random.uniform(-1, 1, n_trades).round(2),
        'magic': np.random.choice([12345, 12346, 12347], n_trades)
    }

    data['close_price'] = data['open_price'] + np.random.uniform(-0.01, 0.01, n_trades)

    return pd.DataFrame(data)
