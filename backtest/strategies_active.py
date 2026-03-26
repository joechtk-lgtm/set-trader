"""
backtest/strategies_active.py

Short-term, high-frequency trading strategies designed for daily rebalancing.
These target 1-10 day holding periods, not weeks/months.

Strategies:
  1. Daily RSI Scalp          - Tight RSI bands, fast entries/exits
  2. Opening Gap Fade         - Fade overnight gaps (contrarian)
  3. 3-Day Momentum           - Very short momentum burst
  4. Bollinger Band Squeeze   - Buy breakouts from compression
  5. Volume Spike Reversal    - High-volume capitulation signal
  6. Dual Thrust             - Classic intraday range breakout
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── STRATEGY 1: Daily RSI Scalp ────────────────────────────────────────────
# Hold 1-5 days, enter on RSI extremes, exit fast

def strategy_rsi_scalp(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Ultra-short RSI: buy when RSI drops below 30, exit at RSI > 50 or stop.
    Designed for 1-5 day holds. High trade frequency.
    """
    if len(df) < 14:
        return None

    rsi = df["RSI"].iloc[-1]
    rsi_prev = df["RSI"].iloc[-2] if len(df) >= 2 else rsi

    if pd.isna(rsi):
        return None

    # Must be dropping into oversold, not already bouncing
    rsi_falling = rsi < rsi_prev

    if rsi < 20:   return 100   # extreme oversold
    if rsi < 25:   return 88
    if rsi < 30:   return 72
    if rsi < 33 and rsi_falling: return 55
    return None


# ── STRATEGY 2: Opening Gap Fade ───────────────────────────────────────────
# Gap down opens often fill — buy the gap, exit in 1-3 days

def strategy_gap_fade(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Overnight gap fade: if today's open is significantly lower than
    yesterday's close, buy the dip expecting a gap fill.
    Strong on SET blue chips that have institutional support levels.
    """
    if len(df) < 5:
        return None

    today_open  = df["Open"].iloc[-1]
    today_close = df["Close"].iloc[-1]
    prev_close  = df["Close"].iloc[-2]

    if pd.isna(today_open) or pd.isna(prev_close) or prev_close == 0:
        return None

    gap_pct = (today_open - prev_close) / prev_close * 100

    # Check volume confirmation (high volume on gap = more reliable)
    vol_ratio = df["VolRatio"].iloc[-1]
    vol_surge = not pd.isna(vol_ratio) and vol_ratio > 1.5

    # Gap down: buy expecting fill
    if gap_pct < -3.0 and vol_surge:  return 95
    if gap_pct < -2.0 and vol_surge:  return 82
    if gap_pct < -2.5:                return 72
    if gap_pct < -1.5 and vol_surge:  return 60
    return None


# ── STRATEGY 3: 3-Day Momentum Burst ───────────────────────────────────────

def strategy_3day_momentum(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Capture short-term momentum bursts. Buy stocks that surged in
    the last 3 days on increasing volume, ride the next 3-5 days.
    Based on post-earnings drift and institutional chasing behavior.
    """
    if len(df) < 10:
        return None

    # 3-day return
    ret_3d = (df["Close"].iloc[-1] / df["Close"].iloc[-4] - 1) * 100
    # 1-day return
    ret_1d = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100

    vol_ratio = df["VolRatio"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    if pd.isna(ret_3d) or pd.isna(vol_ratio):
        return None

    # Not overbought (RSI < 70) and momentum + volume confirmation
    not_overbought = pd.isna(rsi) or rsi < 68

    if ret_3d > 6 and vol_ratio > 2.0 and not_overbought:  return 95
    if ret_3d > 4 and vol_ratio > 1.8 and not_overbought:  return 82
    if ret_3d > 3 and vol_ratio > 1.5 and not_overbought:  return 68
    if ret_3d > 2 and vol_ratio > 2.0 and not_overbought:  return 58
    return None


# ── STRATEGY 4: Bollinger Band Squeeze Breakout ─────────────────────────────

def strategy_bb_squeeze(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Bollinger Band squeeze: when bands are tight (low volatility),
    a breakout is coming. Buy when price breaks above upper band.
    Works well for SET stocks that trend after consolidation.
    """
    if len(df) < 25:
        return None

    close = df["Close"]
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    # Bandwidth = relative width of bands
    bw = ((upper - lower) / ma20 * 100).iloc[-1]
    bw_prev_20 = ((upper - lower) / ma20 * 100).iloc[-20:].mean()

    current_close = close.iloc[-1]
    current_upper = upper.iloc[-1]
    current_ma    = ma20.iloc[-1]

    if pd.isna(bw) or pd.isna(bw_prev_20) or pd.isna(current_upper):
        return None

    # Squeeze condition: bands much tighter than recent average
    is_squeezed = bw < bw_prev_20 * 0.75

    # Breakout condition: price punching above upper band
    breaking_up = current_close > current_upper

    # Trending up inside band
    near_upper  = current_close > current_ma and (current_close / current_upper) > 0.97

    vol_ratio = df["VolRatio"].iloc[-1]
    vol_ok    = pd.isna(vol_ratio) or vol_ratio > 1.2

    if is_squeezed and breaking_up and vol_ok:  return 95
    if breaking_up and vol_ok:                   return 75
    if is_squeezed and near_upper and vol_ok:    return 60
    return None


# ── STRATEGY 5: Volume Spike Reversal ────────────────────────────────────────

def strategy_volume_spike(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Capitulation signal: extreme volume + price down = buyers stepping in.
    When retail panic-sells into institutional buying, price recovers fast.
    Typical hold: 2-7 days.
    """
    if len(df) < 22:
        return None

    vol_ratio   = df["VolRatio"].iloc[-1]
    ret_1d      = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
    rsi         = df["RSI"].iloc[-1]

    if pd.isna(vol_ratio) or pd.isna(ret_1d):
        return None

    # Capitulation: big volume + price down + oversold
    is_oversold = not pd.isna(rsi) and rsi < 40
    big_volume  = vol_ratio > 2.5
    price_down  = ret_1d < -1.5

    if big_volume and price_down and is_oversold and vol_ratio > 4.0: return 98
    if big_volume and price_down and is_oversold:                      return 85
    if big_volume and price_down and vol_ratio > 3.0:                  return 70
    if vol_ratio > 3.0 and price_down:                                 return 55
    return None


# ── STRATEGY 6: Dual Thrust (Classic Intraday Range) ──────────────────────

def strategy_dual_thrust(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Dual Thrust: classic prop-desk strategy adapted for daily bars.
    Computes a range from recent high/low, buys breakout of upper range.
    Originally designed for intraday, works on daily bars with wider stops.
    """
    if len(df) < 6:
        return None

    lookback = 4  # use last 4 days to compute range
    k = 0.5       # thrust multiplier (0.5 is standard)

    recent = df.iloc[-lookback-1:-1]  # exclude today
    HH = recent["High"].max()         # highest high
    HC = recent["Close"].max()        # highest close
    LC = recent["Close"].min()        # lowest close
    LL = recent["Low"].min()          # lowest low

    # Dual thrust range
    range_val = max(HH - LC, HC - LL)
    open_today = df["Open"].iloc[-1]
    close_today = df["Close"].iloc[-1]

    upper_trigger = open_today + k * range_val

    vol_ratio = df["VolRatio"].iloc[-1]

    if pd.isna(range_val) or range_val <= 0:
        return None

    # Price broke above upper trigger on good volume
    if close_today > upper_trigger:
        breakout_strength = (close_today - upper_trigger) / range_val * 100
        vol_boost = (vol_ratio or 1.0)

        score = min(95, 50 + breakout_strength * 5 + (vol_boost - 1) * 10)
        return score if score > 55 else None

    return None


# ── COMBO: All Active Signals Combined ─────────────────────────────────────

def strategy_active_combo(symbol, date, df, portfolio_state) -> Optional[float]:
    """
    Master active strategy: combines all short-term signals.
    A stock needs agreement from multiple signals to be bought.
    This reduces false signals and improves hit rate.
    Rebalances daily, holds 2-10 days.
    """
    if len(df) < 25:
        return None

    votes = []
    weights = []

    s1 = strategy_rsi_scalp(symbol, date, df, portfolio_state)
    if s1: votes.append(s1); weights.append(0.25)

    s2 = strategy_gap_fade(symbol, date, df, portfolio_state)
    if s2: votes.append(s2); weights.append(0.20)

    s3 = strategy_3day_momentum(symbol, date, df, portfolio_state)
    if s3: votes.append(s3); weights.append(0.20)

    s4 = strategy_bb_squeeze(symbol, date, df, portfolio_state)
    if s4: votes.append(s4); weights.append(0.20)

    s5 = strategy_volume_spike(symbol, date, df, portfolio_state)
    if s5: votes.append(s5); weights.append(0.15)

    if len(votes) == 0:
        return None

    # Require at least 2 signals agreeing for a trade
    if len(votes) < 2:
        return None

    # Weighted average of agreeing signals, bonus for consensus
    total_weight = sum(weights[:len(votes)])
    weighted_avg = sum(v * w for v, w in zip(votes, weights)) / total_weight
    consensus_bonus = len(votes) * 5  # +5 per extra confirming signal

    return min(100, weighted_avg + consensus_bonus)


# ── STRATEGY REGISTRY ────────────────────────────────────────────────────────

ACTIVE_STRATEGIES = {
    "rsi_scalp": {
        "func": strategy_rsi_scalp,
        "name": "RSI Scalp (Daily)",
        "desc": "Buy RSI < 30, exit fast. 1-5 day holds.",
        "freq": "D",
        "stop_loss": 0.04,      # tight 4% stop
        "take_profit": 0.08,    # 8% target
    },
    "gap_fade": {
        "func": strategy_gap_fade,
        "name": "Gap Fade",
        "desc": "Buy gap-down opens, ride the fill. 1-3 days.",
        "freq": "D",
        "stop_loss": 0.03,
        "take_profit": 0.06,
    },
    "3day_momentum": {
        "func": strategy_3day_momentum,
        "name": "3-Day Momentum Burst",
        "desc": "Chase 3-day surges on volume. 3-7 days.",
        "freq": "D",
        "stop_loss": 0.05,
        "take_profit": 0.10,
    },
    "bb_squeeze": {
        "func": strategy_bb_squeeze,
        "name": "Bollinger Squeeze Breakout",
        "desc": "Buy compression breakouts. 5-10 days.",
        "freq": "D",
        "stop_loss": 0.05,
        "take_profit": 0.12,
    },
    "volume_spike": {
        "func": strategy_volume_spike,
        "name": "Volume Spike Reversal",
        "desc": "Buy capitulation spikes. 2-7 days.",
        "freq": "D",
        "stop_loss": 0.04,
        "take_profit": 0.08,
    },
    "dual_thrust": {
        "func": strategy_dual_thrust,
        "name": "Dual Thrust Breakout",
        "desc": "Range breakout. 2-5 days.",
        "freq": "D",
        "stop_loss": 0.04,
        "take_profit": 0.08,
    },
    "active_combo": {
        "func": strategy_active_combo,
        "name": "Active Combo (Multi-Signal)",
        "desc": "Needs 2+ signals. Best risk-adjusted.",
        "freq": "D",
        "stop_loss": 0.05,
        "take_profit": 0.10,
    },
}
