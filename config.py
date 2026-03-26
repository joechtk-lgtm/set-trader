# ============================================================
#  SET AI Trader - Configuration
#  Edit this file to customize your trading parameters
# ============================================================

import os

# ── API Keys ─────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_KEY_HERE")

# ── Paper Trading Settings ───────────────────────────────────
PAPER_CAPITAL_THB   = 25_000.0   # Starting virtual capital
MAX_POSITIONS       = 5          # Max open positions at once
MAX_POSITION_PCT    = 0.20       # Max 20% per single stock
MIN_POSITION_PCT    = 0.05       # Min 5% per position
STOP_LOSS_PCT       = 0.08       # 8% hard stop-loss
TAKE_PROFIT_PCT     = 0.20       # 20% take-profit target
DAILY_LOSS_LIMIT    = 0.05       # Halt trading if down 5% today
BROKERAGE_FEE_PCT   = 0.0025     # 0.25% per side (typical Thai broker)

# ── Signal Weights ───────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "value":    0.30,   # DCF + valuation ratios
    "quality":  0.30,   # ROE, margins, debt
    "momentum": 0.25,   # Price momentum signals
    "sentiment":0.15,   # AI news/filing analysis
}

# ── DCF Assumptions ─────────────────────────────────────────
RISK_FREE_RATE      = 0.025   # Thai 10-yr govt bond yield
EQUITY_RISK_PREMIUM = 0.065   # SET historical ERP
TERMINAL_GROWTH     = 0.025   # Long-term perpetual growth
DCF_PROJECTION_YRS  = 5       # Years to project FCF
MARGIN_OF_SAFETY    = 0.15    # Min 15% discount to intrinsic value

# ── Stock Universe ───────────────────────────────────────────
# SET50 blue chips - reliable data via yfinance (.BK suffix)
SET50_UNIVERSE = [
    "PTT", "ADVANC", "KBANK", "SCB", "BBL", "PTTEP",
    "CPALL", "SCC", "TOP", "MINT", "BH", "BDMS",
    "AOT", "IVL", "KTB", "DELTA", "HMPRO", "LH",
    "GULF", "BGRIM", "CPN", "GPSC", "IRPC", "MAKRO",
    "MTC", "OSP", "RATCH", "SAWAD", "SCGP", "TTB",
    "TRUE", "WHA", "BANPU", "CBG", "CENTEL", "COM7",
    "DTAC", "EA", "ESSO", "GLOBAL",
]

# Smaller watchlist for faster testing
TEST_UNIVERSE = ["PTT", "ADVANC", "KBANK", "CPALL", "BDMS", "AOT"]

# ── Screening Thresholds ─────────────────────────────────────
MIN_COMPOSITE_SCORE  = 40     # Minimum to qualify for AI analysis
MIN_MARGIN_OF_SAFETY = 10     # Minimum DCF discount % to consider buying
MAX_PE_RATIO         = 40     # Filter out extremely expensive stocks
MAX_DEBT_EQUITY      = 2.0    # Filter out over-leveraged companies

# ── Scheduling ───────────────────────────────────────────────
SCREEN_TIME = "09:00"         # Run screener at market open (BKK time)
REBALANCE_DAY = "Saturday"    # Weekly rebalance day

# ── Data Paths ───────────────────────────────────────────────
DATA_DIR        = "data/cache"
TRADES_FILE     = "data/paper_trades.json"
PORTFOLIO_FILE  = "data/portfolio.json"
LOG_FILE        = "data/trading.log"
