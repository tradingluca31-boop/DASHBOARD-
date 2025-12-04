# MQL5 Trading Analytics Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Professional Wall Street Grade Analytics for MQL5 Expert Advisors**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Metrics](#metrics) • [Screenshots](#screenshots)

</div>

---

## Overview

A comprehensive trading analytics dashboard designed for MQL5/MT4/MT5 Expert Advisor developers. Analyze your EA performance with institutional-grade metrics and visualizations.

## Features

### 📊 Quantitative Metrics
- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar, Omega Ratios
- **Performance**: Profit Factor, Expectancy, Win Rate, R:R Ratio
- **Risk Management**: VaR, CVaR, Kelly Criterion, Ulcer Index
- **Distribution Analysis**: Skewness, Kurtosis, Tail Ratio

### 📈 Professional Visualizations
- Interactive Equity Curve with High Water Mark
- Underwater Drawdown Analysis
- Monthly Returns Heatmap
- Hourly Performance Heatmap (Day × Hour)
- P/L Distribution Histogram
- Rolling Performance Metrics
- Symbol Performance Breakdown
- Risk/Reward Scatter Analysis

### 🎲 Advanced Analytics
- **Monte Carlo Simulation**: Test strategy robustness (up to 10,000 simulations)
- **Regime Detection**: Identify market volatility states
- **Trade Dependency Analysis**: Autocorrelation and runs tests
- **Stress Testing**: Flash crash, extended drawdown scenarios
- **Optimization Recommendations**: AI-powered suggestions

### 🎨 Wall Street Dark Theme
- Professional dark UI inspired by Bloomberg Terminal
- Gradient accents and modern typography
- Responsive design for all screen sizes

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start (Windows)

```bash
git clone https://github.com/tradingluca31-boop/DASHBOARD-.git
cd DASHBOARD-
run_dashboard.bat
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/tradingluca31-boop/DASHBOARD-.git
cd DASHBOARD-

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Usage

### CSV Format

Export your trades from MT4/MT5 with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `profit` | ✅ | Trade P/L |
| `close_time` | ✅ | Trade close timestamp |
| `symbol` | ⬚ | Trading instrument |
| `type` | ⬚ | BUY/SELL |
| `volume` | ⬚ | Lot size |
| `open_time` | ⬚ | Trade open timestamp |
| `open_price` | ⬚ | Entry price |
| `close_price` | ⬚ | Exit price |
| `commission` | ⬚ | Trading commission |
| `swap` | ⬚ | Swap/rollover |
| `magic` | ⬚ | EA magic number |

### Example CSV

```csv
ticket,open_time,close_time,type,symbol,volume,profit,commission
1001,2024.01.02 08:30:00,2024.01.02 14:45:00,BUY,EURUSD,0.10,22.20,-0.70
1002,2024.01.02 15:00:00,2024.01.02 18:30:00,SELL,GBPUSD,0.05,6.80,-0.35
```

## Metrics

### Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | (Return - Rf) / σ | >1 Good, >2 Excellent |
| **Sortino Ratio** | (Return - Rf) / σ_down | Higher = better downside risk |
| **Calmar Ratio** | Annual Return / Max DD | >3 Excellent |
| **Profit Factor** | Gross Profit / Gross Loss | >1.5 Good, >2 Excellent |
| **Expectancy** | (W% × AvgWin) - (L% × AvgLoss) | Positive = profitable edge |

### Risk Metrics

| Metric | Description |
|--------|-------------|
| **VaR (95%)** | Maximum expected loss at 95% confidence |
| **CVaR (95%)** | Expected loss beyond VaR |
| **Kelly Criterion** | Optimal position sizing |
| **Ulcer Index** | Depth and duration of drawdowns |
| **Recovery Factor** | Net Profit / Max Drawdown |

## Project Structure

```
mql5-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── run_dashboard.bat         # Windows launcher
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   └── sample_trades.csv    # Example CSV format
└── src/
    ├── __init__.py
    ├── data_loader.py       # CSV parsing & validation
    ├── metrics.py           # Quantitative metrics
    ├── visualizations.py    # Plotly charts
    └── advanced_analytics.py # Monte Carlo, stress tests
```

## Dependencies

- `streamlit>=1.28.0` - Web framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `plotly>=5.18.0` - Interactive charts
- `scipy>=1.11.0` - Statistical functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**Built for Quantitative Traders & EA Developers**

</div>
