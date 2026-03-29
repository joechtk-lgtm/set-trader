# SET AI Trader

AI-powered stock screener and paper trading system for the Stock Exchange of Thailand (SET).

Combines quantitative multi-factor scoring with Claude AI conviction analysis to identify buy/sell opportunities in Thai equities — then executes paper trades automatically with full risk management.

## How It Works

1. **Screen** — Scores SET50 stocks across DCF valuation, momentum, quality factors, and dividend yield. Banks get a separate P/B + ROE model.
2. **Analyze** — Top candidates go to Claude AI, which reads fundamentals and assigns a conviction score (1–10).
3. **Trade** — Auto-buys on conviction 7+; auto-sells on stop-loss (−8%), take-profit (+20%), or conviction drop.
4. **Monitor** — Live dashboard tracks equity curve, drawdown, allocation, trade history, and activity log.

## Features

**Screening & Signals**
- DCF intrinsic value with margin-of-safety filter
- Multi-factor composite: value, quality, momentum, sentiment
- Claude AI conviction analysis (Haiku for production runs)
- Configurable thresholds, weights, and universe

**Risk Management**
- Hard stop-loss at −8% per position
- Take-profit target at +20%
- Daily loss limit (−5% halt)
- Max 5 concurrent positions, 20% max per position

**Dashboard** (`dashboard/index.html`)
- Equity curve chart with percentage returns
- Drawdown chart
- Portfolio allocation donut
- Stop-loss proximity indicators
- Trade history table with expandable detail rows
- Activity log with event-type filtering and pagination
- Performance metrics: total return, win rate, Sharpe, max drawdown
- Auto-refresh with manual refresh button

**CLI** (`run.py`)
- Interactive menu: run screener, view portfolio, manual buy/sell, check stop-losses
- Open/close dashboard server from the menu
- Trade history viewer

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Python, yfinance, pandas, numpy, scipy |
| AI | Claude Haiku API (Anthropic) |
| Logging | JSONL structured event log |
| Dashboard | Single-file HTML/CSS/JS, Chart.js |
| Server | Python `http.server` (dev) |

## Setup

```bash
# Clone and enter project
git clone https://github.com/<your-username>/set_trader.git
cd set_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key (or run with --no-ai)
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Interactive menu
```bash
python run.py
```

### Run screener directly
```bash
# Test universe (6 stocks), with AI
python screener.py --universe test

# SET50 full universe, no AI (faster, free)
python screener.py --universe set50 --no-ai

# Dry run — score and rank without executing trades
python screener.py --dry-run
```

### Dashboard
```bash
# From the run.py menu: press 'd' to open, 'c' to close
# Or manually:
python -m http.server 8080
# Then open http://localhost:8080/dashboard/
```

### Shell aliases (optional)
```bash
alias setai="cd ~/Projects/set_trader && source venv/bin/activate && python run.py"
alias setdash="cd ~/Projects/set_trader && python -m http.server 8080"
```

## Project Structure

```
set_trader/
├── screener.py              # Main screening pipeline
├── run.py                   # Interactive CLI entry point
├── config.py                # All configurable parameters
├── signals/
│   ├── ai_signal.py         # Claude API integration
│   ├── dcf.py               # DCF valuation model
│   └── factors.py           # Multi-factor scoring
├── portfolio/
│   └── paper_trader.py      # Paper trading engine
├── data/
│   ├── fetcher.py           # yfinance data wrapper (24hr cache)
│   └── validator.py         # Data validation
├── logs/
│   └── logger.py            # JSONL structured logger
├── dashboard/
│   └── index.html           # Single-file live dashboard
├── backtest/
│   ├── engine.py            # Backtesting engine
│   ├── strategies.py        # Strategy definitions
│   └── run_backtest.py      # Backtest runner
└── examples/                # Sample data (2-month sim)
```

## Example Data

The `examples/` directory contains a simulated 2-month trading history (Jan–Mar 2026) using real historical SET prices — useful for testing the dashboard without running the screener.

## Status

Paper trading since March 2026.

---

