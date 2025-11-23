# OrderFlow Absorption Backtest

A comprehensive backtesting system for order flow trading strategies, specifically designed for absorption and shift detection using footprint data.

## Strategy Overview

This system detects **absorption patterns** in order flow data:

### Long Setup (Sell Absorption)
1. **Absorption Phase**: Large bid imbalances (3+ consecutive rows at 3x ratio), negative delta, price holds above previous low
2. **Shift Phase**: Ask imbalance appears, buy aggression visible, delta may flip positive
3. **Entry**: After shift confirmation, stop loss below absorption low

### Short Setup (Buy Absorption)
Vice versa of the long setup.

## Features

- **Real Footprint Data**: Builds actual footprint candles from Binance AggTrades (tick data)
- **Diagonal Imbalance Detection**: ATAS-style diagonal imbalance calculation
- **Configurable Parameters**: All strategy rules are configurable via YAML
- **Multiple TP Modes**: Fixed R:R, multiple targets, trailing stop after TP1
- **Comprehensive Metrics**: Win rate, profit factor, Sharpe ratio, MFE/MAE, and more
- **HTML/CSV Reports**: Detailed trade reports with equity curves

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd OrderFlow-absorption

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

```bash
# Download BTCUSDT data for January 2024
python scripts/download_data.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31

# Or download all symbols from config
python scripts/download_data.py --all-symbols
```

### 2. Configure Strategy

Edit `config.yaml` to adjust:
- Symbols and timeframe
- Imbalance thresholds
- Absorption rules
- Entry/exit parameters
- Risk management

### 3. Run Backtest

```bash
# Run backtest on BTCUSDT
python scripts/run_backtest.py --symbol BTCUSDT

# Run on all symbols
python scripts/run_backtest.py --all-symbols

# Long only
python scripts/run_backtest.py --symbol BTCUSDT --direction LONG
```

### 4. Analyze Results

```bash
# Analyze a specific report
python scripts/analyze_results.py --report data/results/report_BTCUSDT_20240101_120000
```

## Project Structure

```
OrderFlow-absorption/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
│
├── src/
│   ├── config.py           # Configuration management
│   ├── data/               # Data layer
│   │   ├── binance_client.py   # Binance API client
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── models.py           # Data models
│   │
│   ├── footprint/          # Footprint engine
│   │   ├── builder.py          # Build footprint from ticks
│   │   ├── delta.py            # Delta calculations
│   │   ├── imbalance.py        # Imbalance detection
│   │   └── structures.py       # Data structures
│   │
│   ├── strategy/           # Strategy logic
│   │   ├── absorption.py       # Absorption detection
│   │   ├── shift.py            # Shift detection
│   │   ├── signals.py          # Signal generation
│   │   └── rules.py            # Strategy rules
│   │
│   ├── backtest/           # Backtesting engine
│   │   ├── engine.py           # Main backtest loop
│   │   ├── position.py         # Position management
│   │   ├── risk.py             # Risk management
│   │   └── metrics.py          # Performance metrics
│   │
│   └── visualization/      # Reporting
│       └── report.py           # Report generation
│
├── scripts/
│   ├── download_data.py    # Data download script
│   ├── run_backtest.py     # Backtest runner
│   └── analyze_results.py  # Results analysis
│
└── data/                   # Data storage (gitignored)
    ├── raw/                # Raw tick data
    ├── processed/          # Processed data
    └── results/            # Backtest results
```

## Configuration Guide

### Key Parameters

```yaml
# Imbalance detection
imbalance:
  ratio_threshold: 3.0      # 300% = 3x diagonal ratio
  consecutive_rows: 3       # Required consecutive imbalances

# Absorption rules
absorption:
  delta_lookback: 2         # Compare with N previous candles
  min_imbalance_rows: 3     # Minimum consecutive rows

# Take profit
take_profit:
  mode: "rr_then_trail"     # R:R first, then trailing
  rr_ratio: 2.0             # 1:2 risk:reward
  trailing:
    activation_rr: 1.0      # Activate after 1R profit
    trail_percent: 0.005    # Trail by 0.5%
```

## Output Example

```
═══════════════════════════════════════════════════════════
         ORDERFLOW ABSORPTION BACKTEST REPORT
═══════════════════════════════════════════════════════════

PERFORMANCE METRICS
────────────────────────────────────────
Total Trades:        156
Win Rate:            62.3%
Profit Factor:       2.14
Avg R Multiple:      1.43R
Max Drawdown:        -8.2%
Sharpe Ratio:        1.87

TRADE BREAKDOWN
────────────────────────────────────────
Long Trades:         98 (64.3% win)
Short Trades:        58 (58.6% win)
Avg Winner:          +2.1R
Avg Loser:           -0.92R

EXIT ANALYSIS
────────────────────────────────────────
SL:                  37 (23.7%)
TP:                  71 (45.5%)
TRAIL:               48 (30.8%)
```

## How It Works

### Footprint Building
1. Fetch AggTrades from Binance (tick-level data)
2. Group trades by candle period (e.g., 10 minutes)
3. Quantize prices to tick size (e.g., $100 levels)
4. Classify each trade as bid (sell aggressor) or ask (buy aggressor)
5. Calculate delta, POC, and other metrics

### Imbalance Detection
```
BID Imbalance at price P:
  bid_volume[P] > ratio × ask_volume[P + tick_size]

ASK Imbalance at price P:
  ask_volume[P] > ratio × bid_volume[P - tick_size]
```

### Absorption Detection
- Strong imbalances (3+ consecutive rows)
- Large delta compared to previous candles
- Price holds support/resistance
- Wick rejection

### Shift Detection
- Opposite imbalance appears
- Delta may flip direction
- Aggression shifts to opposite side

## Requirements

- Python 3.10+
- pandas, numpy
- httpx (async HTTP)
- pydantic (configuration)
- matplotlib, plotly (visualization)

## License

MIT License
