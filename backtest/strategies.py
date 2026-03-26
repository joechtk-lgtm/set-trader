"""
backtest/strategies.py

Pre-built trading strategies ready to backtest.
Each strategy is a function with the signature:

    (symbol, date, df_to_date, portfolio_state) -> float or None

Returns a score 0-100 (higher = stronger signal), or None to skip.
The backtester buys the highest-scoring stocks each rebalance period.

Strategies included:
  1. RSI Mean Reversion         - Buy oversold, sell overbought
  2. Dual Moving Average        - EMA crossover trend following
  3. Momentum 12-1              - Academic price momentum factor
  4. Multi-Factor Value+Momentum - Combined quant approach
  5. Breakout + Volume          - Price breakout with volume confirmation
  6. Quality Trend              - Strong trend + low volatility
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── STRATEGY 1: RSI Mean Reversion ──────────────────────────────────────────

def strategy_rsi_reversion(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Buy when RSI is oversold (< 35), sell at 8% stop / 20% target.
    Simple but effective on SET blue chips.
    Best in: sideways / mildly trending markets.
    """
    if len(df) < 20:
        return None

    rsi = df["RSI"].iloc[-1]
    if pd.isna(rsi):
        return None

    # Only buy on deep oversold readings
    if rsi < 25:   return 95
    if rsi < 30:   return 80
    if rsi < 35:   return 65
    return None   # no signal


# ── STRATEGY 2: Dual Moving Average Crossover ────────────────────────────────

def strategy_ma_crossover(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Classic trend-following: buy when EMA20 crosses above EMA50.
    Adds volume confirmation to reduce false signals.
    Best in: trending markets.
    """
    if len(df) < 55:
        return None

    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    prev_ma20 = df["MA20"].iloc[-2]
    prev_ma50 = df["MA50"].iloc[-2]

    if any(pd.isna([ma20, ma50, prev_ma20, prev_ma50])):
        return None

    # Golden cross: MA20 just crossed above MA50
    just_crossed_up = (prev_ma20 <= prev_ma50) and (ma20 > ma50)
    # Still bullish: MA20 above MA50 and trending up
    bullish_trend   = ma20 > ma50 and ma20 > prev_ma20

    vol_ratio = df["VolRatio"].iloc[-1]
    vol_confirm = not pd.isna(vol_ratio) and vol_ratio > 1.2  # above-average volume

    if just_crossed_up and vol_confirm:
        return 90
    elif just_crossed_up:
        return 75
    elif bullish_trend and vol_confirm:
        return 60
    return None


# ── STRATEGY 3: Momentum 12-1 ────────────────────────────────────────────────

def strategy_momentum_12_1(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Fama-French momentum factor: buy top performers over past 12 months,
    excluding the most recent 1 month (avoids short-term reversal).
    Academic evidence is strong for this factor globally.
    Best in: trending markets with clear winners and losers.
    """
    if len(df) < 252:
        return None

    # 12-month return, skip most recent month
    ret_12m = df["Ret12M"].iloc[-1]
    ret_1m  = df["Ret1M"].iloc[-1]

    if pd.isna(ret_12m) or pd.isna(ret_1m):
        return None

    # True momentum: 12m return minus 1m return
    momentum = (ret_12m - ret_1m) * 100

    if momentum > 40:    return 95
    if momentum > 25:    return 80
    if momentum > 15:    return 65
    if momentum > 8:     return 50
    return None


# ── STRATEGY 4: Multi-Factor (Value + Momentum + Quality Proxy) ──────────────

def strategy_multi_factor(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Combines momentum, mean-reversion, and trend quality signals.
    Uses only price/volume data (no fundamental data in backtest).
    This is the most sophisticated price-only strategy.
    Best in: all market conditions (diversified signal mix).
    """
    if len(df) < 60:
        return None

    score = 0.0
    weights = 0.0

    # --- Momentum component (30%) ---
    ret_3m = df["Ret3M"].iloc[-1]
    if not pd.isna(ret_3m):
        mom_score = 0
        if ret_3m > 0.20:   mom_score = 100
        elif ret_3m > 0.10: mom_score = 80
        elif ret_3m > 0.05: mom_score = 60
        elif ret_3m > 0:    mom_score = 45
        elif ret_3m > -0.10:mom_score = 30
        else:               mom_score = 10
        score   += mom_score * 0.30
        weights += 0.30

    # --- Mean reversion component (25%) ---
    rsi = df["RSI"].iloc[-1]
    if not pd.isna(rsi):
        rsi_score = 0
        if rsi < 30:        rsi_score = 100
        elif rsi < 40:      rsi_score = 75
        elif rsi < 50:      rsi_score = 55
        elif rsi < 60:      rsi_score = 45
        elif rsi < 70:      rsi_score = 25
        else:               rsi_score = 10
        score   += rsi_score * 0.25
        weights += 0.25

    # --- Trend quality component (25%) ---
    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    close = df["Close"].iloc[-1]
    if not pd.isna(ma20) and not pd.isna(ma50):
        trend_score = 0
        if close > ma20 > ma50:     trend_score = 90   # strong uptrend
        elif close > ma20:           trend_score = 65   # above short MA
        elif close > ma50:           trend_score = 50   # above medium MA
        elif close < ma20 < ma50:    trend_score = 10   # downtrend
        else:                        trend_score = 35
        score   += trend_score * 0.25
        weights += 0.25

    # --- Volatility quality (20%): prefer low-vol momentum ---
    if len(df) >= 20:
        recent_vol = df["Close"].pct_change().iloc[-20:].std() * np.sqrt(252)
        if not pd.isna(recent_vol):
            # Lower volatility = higher score
            vol_score = max(0, 100 - (recent_vol * 200))
            score   += vol_score * 0.20
            weights += 0.20

    if weights == 0:
        return None

    final = score / weights
    return final if final >= 50 else None   # only return positive signals


# ── STRATEGY 5: Breakout + Volume ────────────────────────────────────────────

def strategy_breakout(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    52-week high breakout strategy with volume surge confirmation.
    When a stock breaks to new highs on high volume, it often continues.
    Best in: bull markets with clear sector rotation.
    """
    if len(df) < 252:
        return None

    close    = df["Close"].iloc[-1]
    high_52w = df["High"].iloc[-252:-1].max()  # exclude today
    vol_ratio = df["VolRatio"].iloc[-1]

    if pd.isna(high_52w) or pd.isna(vol_ratio):
        return None

    # Price breaking to new 52-week high
    if close > high_52w:
        # Strong volume surge: 2x+ average
        if vol_ratio > 2.5:   return 95
        if vol_ratio > 2.0:   return 85
        if vol_ratio > 1.5:   return 72
        if vol_ratio > 1.2:   return 60
        return 45  # breakout but weak volume
    return None


# ── STRATEGY 6: Low Volatility Trend ────────────────────────────────────────

def strategy_low_vol_trend(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Low Volatility anomaly: low-risk stocks outperform on risk-adjusted basis.
    Buy stocks in an uptrend with decreasing volatility (institutional accumulation).
    Best in: late bull / early bear markets (defensive).
    """
    if len(df) < 60:
        return None

    close = df["Close"].iloc[-1]
    ma50  = df["MA50"].iloc[-1]
    ma200 = df["MA200"].iloc[-1]

    if pd.isna(ma50) or pd.isna(ma200):
        return None

    # Must be in uptrend
    if close < ma50:
        return None

    # Compute recent volatility (21 days) vs longer-term (63 days)
    vol_21d  = df["Close"].pct_change().iloc[-21:].std()
    vol_63d  = df["Close"].pct_change().iloc[-63:].std()

    if pd.isna(vol_21d) or pd.isna(vol_63d) or vol_63d == 0:
        return None

    # Volatility contracting = institutional accumulation signal
    vol_ratio_regime = vol_21d / vol_63d

    score = 0
    if close > ma200:    score += 30  # long-term uptrend
    if close > ma50:     score += 20  # short-term uptrend

    # Reward vol contraction
    if vol_ratio_regime < 0.7:   score += 50
    elif vol_ratio_regime < 0.85: score += 35
    elif vol_ratio_regime < 1.0:  score += 20
    else:                         score -= 10

    return score if score >= 50 else None


# ── STRATEGY REGISTRY ────────────────────────────────────────────────────────

STRATEGIES = {
    "rsi_reversion":   {
        "func": strategy_rsi_reversion,
        "name": "RSI Mean Reversion",
        "desc": "Buy oversold (RSI < 35), cut at stop-loss",
        "freq": "W",
    },
    "ma_crossover": {
        "func": strategy_ma_crossover,
        "name": "MA Crossover Trend",
        "desc": "EMA20 x EMA50 golden cross with volume confirmation",
        "freq": "W",
    },
    "momentum_12_1": {
        "func": strategy_momentum_12_1,
        "name": "Momentum 12-1 Month",
        "desc": "Fama-French academic momentum factor",
        "freq": "M",
    },
    "multi_factor": {
        "func": strategy_multi_factor,
        "name": "Multi-Factor (Momentum + RSI + Trend)",
        "desc": "Composite of 4 price signals, most robust",
        "freq": "W",
    },
    "breakout": {
        "func": strategy_breakout,
        "name": "52-Week High Breakout",
        "desc": "New highs on volume surge",
        "freq": "W",
    },
    "low_vol_trend": {
        "func": strategy_low_vol_trend,
        "name": "Low Volatility Trend",
        "desc": "Defensive: uptrend + vol contraction",
        "freq": "W",
    },
}
