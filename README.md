# Trading Agent Performance Dashboard

Interactive Streamlit dashboard for analyzing cryptocurrency trading agent performance.

## Features

- **Multi-file support**: Load and merge multiple trade history CSV files
- **Comprehensive metrics**: Net PnL, Gross PnL, Fees breakdown
- **Per-coin analysis**: Individual coin performance tracking
- **Visual trade analysis**: Color-coded buy/sell markers with PnL-based sizing
- **Price correlation**: Trade execution prices vs cumulative PnL

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run dashboard.py
```

The dashboard will automatically detect all `trade_history_*.csv` files in the current directory.

## Data Format

CSV files should contain the following columns:
- `time`: Timestamp
- `coin`: Cryptocurrency symbol
- `dir`: Trade direction (Open Long, Close Short, etc.)
- `px`: Execution price
- `sz`: Trade size
- `closedPnl`: Realized PnL (net, includes fees)
- `fee`: Transaction fee

## Key Insights

The dashboard helps identify:
- Overall profitability (Gross vs Net)
- Fee impact on performance
- Best/worst performing coins
- Trade timing and price correlation
- High-impact trades (via marker sizing)
