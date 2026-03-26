"""
SET Trader - Configuration
All settings live here. Change these to tune your system.
"""

# API Keys
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# Trading Mode
TRADING_MODE = "paper"
PAPER_STARTING_CAPITAL = 25000  # THB

# Universe - SET50 tickers (Yahoo Finance .BK suffix)
SET50_TICKERS = [
    "PTT", "ADVANC", "KBANK", "SCB", "BBL", "PTTEP",
    "CPALL", "SCC", "TOP", "MINT", "BH", "BDMS",
    "AOT", "IVL", "KTB", "BAM", "BGRIM", "DELTA",
    "HMPRO", "CPF", "TRUE", "TU", "GPSC", "PTTGC",
    "BTS", "CENTEL", "CPN", "EGCO", "IRPC", "JMT",
    "KCE", "MTC", "RATCH", "SAWAD", "TISCO", "WHA",
]

# Risk Management
MAX_POSITION_SIZE_PCT   = 0.10
MIN_POSITION_SIZE_PCT   = 0.03
STOP_LOSS_PCT           = 0.08
TAKE_PROFIT_PCT         = 0.20
MAX_OPEN_POSITIONS      = 5
DAILY_LOSS_LIMIT_PCT    = 0.05
MAX_DRAWDOWN_PCT        = 0.15

# Signal Weights (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "value":     0.30,
    "quality":   0.30,
    "momentum":  0.25,
    "yield":     0.10,
    "sentiment": 0.05,
}

# DCF Assumptions
DCF_YEARS                = 5
DCF_TERMINAL_GROWTH      = 0.025
DCF_RISK_FREE_RATE       = 0.025
DCF_MARKET_PREMIUM       = 0.065
DCF_MIN_MARGIN_OF_SAFETY = 0.15

# Screening Thresholds
MIN_COMPOSITE_SCORE      = 50
MIN_QUALITY_SCORE        = 40
MAX_PE_RATIO             = 30
MAX_DEBT_EQUITY          = 2.0
MIN_ROE                  = 0.08

# Schedule
SCREEN_HOUR              = 18
REBALANCE_DAY            = "saturday"

# Logging
LOG_FILE                 = "logs/trader.log"
TRADE_LOG_FILE           = "logs/trades.json"
