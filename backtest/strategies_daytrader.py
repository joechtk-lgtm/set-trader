"""
backtest/strategies_daytrader.py

Genuinely high-frequency strategies: 30-80 trades per 3 months.
Target hold: 1-3 days max (forced exit). Daily signals.

Design philosophy:
  - Every stock scanned EVERY day
  - Loose entry conditions (fire often)
  - Tight stops + max hold = fast capital recycling
  - Multiple re-entries on the same stock allowed
  - Focus on edge, not perfection

Strategies:
  1. EMA Micro-Trend         - 3/8 EMA cross, fires almost daily
  2. RSI Daily Rotation      - RSI < 50 with price above MA20
  3. Overnight Reversal      - Price < yesterday's low, vol spike
  4. Mean Reversion Z-Score  - Z-score of price vs 10-day mean
  5. Range Breakout Daily    - Today's high breaks 5-day range
  6. Momentum Continuation   - 2-day green + vol surge
  7. TURBO COMBO             - Any 1 signal = trade (maximum trades)
"""

import pandas as pd
import numpy as np
from typing import Optional

MAX_HOLD_DAYS = 3   # Force exit after this many days (handled in engine)


# ── STRATEGY 1: EMA Micro-Trend (3/8 EMA Cross) ────────────────────────────
# Fires almost every trending day

def strategy_ema_micro(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Fast EMA(3) vs EMA(8) crossover.
    Much faster than 20/50 — generates a signal almost every day a trend exists.
    """
    if len(df) < 10:
        return None

    c = df["Close"]
    ema3 = c.ewm(span=3, adjust=False).mean()
    ema8 = c.ewm(span=8, adjust=False).mean()

    e3_now  = ema3.iloc[-1]
    e8_now  = ema8.iloc[-1]
    e3_prev = ema3.iloc[-2]
    e8_prev = ema8.iloc[-2]

    if any(pd.isna([e3_now, e8_now, e3_prev, e8_prev])):
        return None

    # EMA3 crossing above EMA8 (fresh signal)
    cross_up = e3_prev <= e8_prev and e3_now > e8_now
    # EMA3 above EMA8 and widening (continuation)
    widening = e3_now > e8_now and (e3_now - e8_now) > (e3_prev - e8_prev)

    vol_ratio = df["VolRatio"].iloc[-1]
    vol_ok    = pd.isna(vol_ratio) or vol_ratio > 0.8  # very loose vol filter

    rsi = df["RSI"].iloc[-1]
    not_overbought = pd.isna(rsi) or rsi < 72

    if cross_up and vol_ok and not_overbought:    return 90
    if widening and vol_ok and not_overbought:    return 68
    if e3_now > e8_now and not_overbought:        return 52
    return None


# ── STRATEGY 2: RSI Daily Rotation ────────────────────────────────────────
# RSI 35-55 zone + uptrend = buy. Fires on most dips.

def strategy_rsi_rotation(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Loosened RSI: buy when RSI is in 30-55 zone AND price is above MA20.
    This fires 2-3x per week per stock vs 1x per month with RSI < 30.
    """
    if len(df) < 22:
        return None

    rsi   = df["RSI"].iloc[-1]
    close = df["Close"].iloc[-1]
    ma20  = df["MA20"].iloc[-1]

    if pd.isna(rsi) or pd.isna(ma20):
        return None

    above_ma20 = close > ma20
    rsi_rising = df["RSI"].iloc[-1] > df["RSI"].iloc[-2] if len(df) >= 2 else True

    if rsi < 30 and above_ma20:                      return 95
    if rsi < 38 and above_ma20:                      return 82
    if rsi < 45 and above_ma20 and rsi_rising:       return 68
    if rsi < 52 and above_ma20 and rsi_rising:       return 55
    return None


# ── STRATEGY 3: Overnight Reversal ────────────────────────────────────────
# Today's price below yesterday's low = overreaction, buy the dip

def strategy_overnight_reversal(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    When price opens or closes below yesterday's low, it often snaps back.
    This exploits overnight overreaction and morning capitulation.
    Target: 1-2 day reversal.
    """
    if len(df) < 3:
        return None

    today_close  = df["Close"].iloc[-1]
    today_low    = df["Low"].iloc[-1]
    yest_low     = df["Low"].iloc[-2]
    yest_close   = df["Close"].iloc[-2]
    yest_open    = df["Open"].iloc[-2]

    if any(pd.isna([today_close, yest_low, yest_close])):
        return None

    # Price closed below yesterday's low (bearish overextension)
    below_yest_low = today_close < yest_low

    # Size of the overextension
    overext_pct = (yest_low - today_close) / yest_low * 100

    vol_ratio = df["VolRatio"].iloc[-1]

    if below_yest_low:
        if overext_pct > 2.0 and vol_ratio > 1.5:   return 92
        if overext_pct > 1.5:                         return 78
        if overext_pct > 0.8 and vol_ratio > 1.3:    return 65
        return 52
    return None


# ── STRATEGY 4: Z-Score Mean Reversion ────────────────────────────────────
# Price too far below its 10-day mean = buy

def strategy_zscore(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Z-score of today's close vs 10-day rolling mean.
    A z-score of -1.5 fires frequently (roughly every 1-2 weeks per stock)
    vs the 5-year RSI < 30 which barely fires.
    """
    if len(df) < 12:
        return None

    window = 10
    close  = df["Close"].iloc[-window:]
    mean   = close.mean()
    std    = close.std()

    if std == 0 or pd.isna(std):
        return None

    z = (df["Close"].iloc[-1] - mean) / std

    # Negative z = price below mean = mean reversion buy
    if z < -2.0:   return 95
    if z < -1.5:   return 82
    if z < -1.0:   return 68
    if z < -0.5:   return 54
    return None


# ── STRATEGY 5: 5-Day High Breakout ────────────────────────────────────────
# Much tighter than 52-week breakout — fires constantly

def strategy_5day_breakout(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Today's high breaks above the 5-day trading range high.
    Fires almost every week when a stock makes a new 5-day high on volume.
    """
    if len(df) < 7:
        return None

    today_close = df["Close"].iloc[-1]
    high_5d     = df["High"].iloc[-6:-1].max()  # last 5 days, not today

    if pd.isna(high_5d):
        return None

    breakout_pct = (today_close - high_5d) / high_5d * 100

    vol_ratio = df["VolRatio"].iloc[-1]
    rsi       = df["RSI"].iloc[-1]
    not_overbought = pd.isna(rsi) or rsi < 70

    if today_close > high_5d:
        if breakout_pct > 1.5 and vol_ratio > 1.8 and not_overbought: return 92
        if breakout_pct > 1.0 and vol_ratio > 1.4 and not_overbought: return 78
        if breakout_pct > 0.5 and vol_ratio > 1.2 and not_overbought: return 63
        if vol_ratio > 1.5 and not_overbought:                         return 55
    return None


# ── STRATEGY 6: 2-Day Green Momentum ──────────────────────────────────────
# Two consecutive green days + volume = buy day 3 continuation

def strategy_2day_green(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    When a stock closes up 2 consecutive days on rising volume,
    buy for continuation on day 3. Simple, fires often.
    """
    if len(df) < 4:
        return None

    c = df["Close"]
    v = df["Volume"]

    day0_green = c.iloc[-1] > c.iloc[-2]
    day1_green = c.iloc[-2] > c.iloc[-3]
    vol_rising = v.iloc[-1] > v.iloc[-2] > v.iloc[-3]

    ret_2d = (c.iloc[-1] / c.iloc[-3] - 1) * 100

    rsi = df["RSI"].iloc[-1]
    not_overbought = pd.isna(rsi) or rsi < 68

    if day0_green and day1_green and not_overbought:
        if vol_rising and ret_2d > 3:   return 88
        if vol_rising and ret_2d > 1.5: return 75
        if ret_2d > 2:                  return 62
        return 52
    return None


# ── STRATEGY 7: TURBO COMBO ────────────────────────────────────────────────
# Fires on ANY signal. Maximum trade frequency.

def strategy_turbo(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    TURBO: any one of the above signals = buy. Designed for max frequency.
    Works best with tight stops (3-4%) and short max hold (2 days).
    Pros: lots of trades, good P&L statistics fast.
    Cons: more losing trades, higher fees.
    """
    if len(df) < 14:
        return None

    signals = [
        strategy_ema_micro(symbol, date, df, portfolio_state),
        strategy_rsi_rotation(symbol, date, df, portfolio_state),
        strategy_overnight_reversal(symbol, date, df, portfolio_state),
        strategy_zscore(symbol, date, df, portfolio_state),
        strategy_5day_breakout(symbol, date, df, portfolio_state),
        strategy_2day_green(symbol, date, df, portfolio_state),
    ]

    active = [s for s in signals if s is not None]
    if not active:
        return None

    # Average score, bonus for multiple signals
    avg = sum(active) / len(active)
    bonus = (len(active) - 1) * 3   # small bonus per extra signal
    return min(100, avg + bonus)


# ── REGISTRY ────────────────────────────────────────────────────────────────

DAYTRADER_STRATEGIES = {
    "ema_micro": {
        "func": strategy_ema_micro,
        "name": "EMA 3/8 Micro-Trend",
        "desc": "Fast EMA cross, fires every trend day",
        "stop_loss": 0.04, "take_profit": 0.06, "max_hold": 3,
    },
    "rsi_rotation": {
        "func": strategy_rsi_rotation,
        "name": "RSI Daily Rotation",
        "desc": "RSI 30-55 dips above MA20. Very frequent.",
        "stop_loss": 0.04, "take_profit": 0.07, "max_hold": 3,
    },
    "overnight_reversal": {
        "func": strategy_overnight_reversal,
        "name": "Overnight Reversal",
        "desc": "Price below yesterday's low = snap-back",
        "stop_loss": 0.03, "take_profit": 0.05, "max_hold": 2,
    },
    "zscore": {
        "func": strategy_zscore,
        "name": "Z-Score Mean Reversion",
        "desc": "Price too far below 10-day mean",
        "stop_loss": 0.04, "take_profit": 0.07, "max_hold": 3,
    },
    "5day_breakout": {
        "func": strategy_5day_breakout,
        "name": "5-Day High Breakout",
        "desc": "New 5-day high on volume",
        "stop_loss": 0.04, "take_profit": 0.07, "max_hold": 3,
    },
    "2day_green": {
        "func": strategy_2day_green,
        "name": "2-Day Green Continuation",
        "desc": "2 green days + volume = buy day 3",
        "stop_loss": 0.04, "take_profit": 0.06, "max_hold": 2,
    },
    "turbo": {
        "func": strategy_turbo,
        "name": "TURBO (Any Signal)",
        "desc": "Maximum trades. Any signal = enter.",
        "stop_loss": 0.03, "take_profit": 0.05, "max_hold": 2,
    },
}
