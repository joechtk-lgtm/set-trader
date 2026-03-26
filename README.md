# SET AI Trader

AI-powered paper trading system for the Stock Exchange of Thailand (SET).

## Overview
Automated weekly screener that combines quantitative factor scoring with
Claude AI conviction analysis to identify buy/sell signals in Thai equities.

## How It Works
1. **Weekly screener**: Scores SET50 stocks on DCF valuation, momentum,
   quality factors, and dividend yield
2. **AI layer**: Claude analyzes top candidates and assigns conviction 1-10
3. **Automated execution**: Auto-buys on conviction 7+, auto-sells on
   stop-loss (-8%), take-profit (+20%), or conviction drop
4. **Bank model**: Separate P/B + ROE valuation for Thai banks (KBANK, SCB, etc.)

## Tech Stack
- Python + yfinance for data fetching
- Claude Haiku (production) / Sonnet (development) for AI analysis
- JSONL event logging for full audit trail
- HTML dashboard served via Python HTTP server

## Running Locally
```bash
cd set_trader
source venv/bin/activate
python run.py
```

## Example Data
See `examples/` for a simulated 2-month trading history (Jan-Mar 2026)
using real historical price data.

## Project Status
Paper trading (live since March 2026). Built as part of USC Marshall MBA
portfolio demonstrating AI product development and quantitative finance skills.

---
*Chayut (Joe) Teeradakorn | USC Marshall MBA 2026*
