# SET AI Paper Trader

An AI-powered paper trading system for the Stock Exchange of Thailand (SET).
Combines DCF valuation, multi-factor scoring, and Claude AI reasoning to
identify opportunities and simulate trades with virtual 25,000 THB.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Claude API key
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```
Or edit `config.py` directly.

### 3. Run the interactive menu
```bash
python run.py
```

### 4. Run the screener directly
```bash
# Fast screen (no AI, free)
python screener.py --no-ai

# Full AI analysis on test universe (6 stocks)
python screener.py

# Full SET50 scan with AI
python screener.py --universe set50

# Show signals but do not execute any trades
python screener.py --dry-run
```

### 5. View the dashboard
```bash
cd dashboard
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

---

## How the System Works

```
Every run:
  1. Fetch fundamentals for universe (yfinance .BK data)
  2. Compute DCF intrinsic value for each stock
  3. Score each stock on 4 factors (Value, Quality, Momentum, Yield)
  4. Filter top candidates (composite score > 40)
  5. Send top 5 candidates to Claude API for AI reasoning
  6. Claude returns: action, conviction, position size, thesis, risks
  7. Paper trader executes BUY if conviction >= 6 and no red flags
  8. Stop-losses and take-profits auto-checked on each run
```

---

## Signal Layers

### Layer 1: DCF Valuation
- Projects 5-year Free Cash Flow
- Computes WACC using CAPM (Thai risk-free rate + equity risk premium)
- Calculates margin of safety vs current market price
- Only buys stocks with >15% margin of safety

### Layer 2: Multi-Factor Scoring
| Factor  | Weight | Signals Used |
|---------|--------|-------------|
| Value   | 30%    | P/E, P/B, EV/EBITDA |
| Quality | 30%    | ROE, Margins, Debt/Equity |
| Momentum| 25%    | 12-1M returns, RSI, vs MA20 |
| Yield   | 15%    | Dividend yield, payout ratio |

### Layer 3: Claude AI
- Reviews all quantitative signals together
- Writes investment thesis in plain language
- Flags accounting red flags (negative FCF, high debt, etc.)
- Sets position size and stop-loss recommendations
- Returns structured JSON decision

---

## Risk Controls

| Control | Default | Setting |
|---------|---------|---------|
| Max positions | 5 | `MAX_POSITIONS` |
| Max per stock | 20% | `MAX_POSITION_PCT` |
| Stop-loss | 8% | `STOP_LOSS_PCT` |
| Take-profit | 20% | `TAKE_PROFIT_PCT` |
| Daily loss limit | 5% | `DAILY_LOSS_LIMIT` |
| Min AI conviction | 6/10 | Hard-coded in `auto_trade()` |

---

## Project Structure

```
set_trader/
├── run.py              Interactive menu (start here)
├── screener.py         Main signal pipeline + auto-trader
├── config.py           All settings (edit this to customize)
├── requirements.txt
├── data/
│   ├── fetcher.py      yfinance data fetching + caching
│   ├── cache/          Cached fundamental data (auto-created)
│   ├── paper_trades.json
│   └── portfolio.json
├── signals/
│   ├── dcf.py          DCF valuation engine
│   ├── factors.py      Multi-factor scoring (Value/Quality/Momentum/Yield)
│   └── ai_signal.py    Claude API integration
├── portfolio/
│   └── paper_trader.py Paper trading engine (buy/sell/P&L tracking)
└── dashboard/
    └── index.html      Web dashboard (open with python -m http.server)
```

---

## Customizing

### Change the stock universe
Edit `config.py`:
```python
TEST_UNIVERSE = ["PTT", "ADVANC", "KBANK"]  # your watchlist
```

### Adjust signal weights
```python
SIGNAL_WEIGHTS = {
    "value":    0.35,  # increase value focus
    "quality":  0.35,
    "momentum": 0.20,
    "sentiment":0.10,
}
```

### Change capital allocation
```python
PAPER_CAPITAL_THB = 25_000.0  # your starting virtual capital
MAX_POSITIONS     = 5          # max concurrent positions
MAX_POSITION_PCT  = 0.20       # max 20% per stock
```

---

## Paper Trading Rules

1. Run the screener on weekday mornings before 10:00 BKK time
2. Let the system run for at least 2 months before evaluating
3. Track your win rate and average P&L per trade
4. Only move to live trading if paper results are consistently positive
5. Keep a trading journal: note when your thesis was right or wrong

---

## Upgrading to Live Trading (Later)

When paper results are good, the swap to live trading is small:
1. Replace `PaperTrader.buy/sell()` with Settrade Open API calls
2. All signal logic stays exactly the same
3. Add real-time price streaming (Settrade WebSocket)

---

## Disclaimer

This is an educational paper trading system. It is not financial advice.
Trading real money involves significant risk of loss. Always do your own
research. The authors are not responsible for any financial losses.
