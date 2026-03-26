"""
signals/factors.py
Multi-factor scoring engine.
Scores each stock across Value, Quality, Momentum, and Yield factors.
Based on Fama-French 5-factor model adapted for the SET market.
"""

import numpy as np
import pandas as pd
from typing import Optional
import sys
sys.path.append("..")
import config


def _safe(val, default=None):
    """Return val if not None/NaN, else default."""
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except TypeError:
        pass
    return val


def score_value(fund: dict) -> dict:
    """
    Value Factor: Reward cheap stocks, penalize expensive ones.
    Signals: P/E, P/B, EV/EBITDA, FCF Yield
    Score range: 0-100 (higher = better value)
    """
    score = 50.0  # baseline
    signals = {}

    pe = _safe(fund.get("pe_ratio"))
    if pe:
        if pe < 10:   signals["pe"] = 25
        elif pe < 15: signals["pe"] = 20
        elif pe < 20: signals["pe"] = 12
        elif pe < 30: signals["pe"] = 0
        else:         signals["pe"] = -15
        score += signals["pe"]

    pb = _safe(fund.get("pb_ratio"))
    if pb:
        if pb < 0.8:  signals["pb"] = 20
        elif pb < 1.2:signals["pb"] = 15
        elif pb < 2.0:signals["pb"] = 8
        elif pb < 3.0:signals["pb"] = 0
        else:         signals["pb"] = -10
        score += signals["pb"]

    ev_ebitda = _safe(fund.get("ev_ebitda"))
    if ev_ebitda:
        if ev_ebitda < 6:    signals["ev_ebitda"] = 15
        elif ev_ebitda < 10: signals["ev_ebitda"] = 8
        elif ev_ebitda < 15: signals["ev_ebitda"] = 0
        else:                signals["ev_ebitda"] = -10
        score += signals["ev_ebitda"]

    return {"score": round(np.clip(score, 0, 100), 1), "signals": signals}


def score_quality(fund: dict) -> dict:
    """
    Quality Factor: Reward profitable, low-debt, efficient companies.
    Signals: ROE, Profit Margin, Debt/Equity, Current Ratio
    Score range: 0-100
    """
    score = 50.0
    signals = {}

    roe = _safe(fund.get("roe"))
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:   signals["roe"] = 25
        elif roe_pct > 15: signals["roe"] = 18
        elif roe_pct > 10: signals["roe"] = 10
        elif roe_pct > 5:  signals["roe"] = 0
        else:              signals["roe"] = -15
        score += signals["roe"]

    margin = _safe(fund.get("profit_margin"))
    if margin:
        margin_pct = margin * 100
        if margin_pct > 20:   signals["margin"] = 20
        elif margin_pct > 10: signals["margin"] = 12
        elif margin_pct > 5:  signals["margin"] = 5
        elif margin_pct > 0:  signals["margin"] = 0
        else:                  signals["margin"] = -20
        score += signals["margin"]

    de = _safe(fund.get("debt_equity"))
    if de:
        if de < 30:    signals["debt_equity"] = 15
        elif de < 80:  signals["debt_equity"] = 8
        elif de < 150: signals["debt_equity"] = 0
        elif de < 250: signals["debt_equity"] = -10
        else:          signals["debt_equity"] = -20
        score += signals["debt_equity"]

    cr = _safe(fund.get("current_ratio"))
    if cr:
        if cr > 2.0:   signals["current_ratio"] = 10
        elif cr > 1.5: signals["current_ratio"] = 5
        elif cr > 1.0: signals["current_ratio"] = 0
        else:          signals["current_ratio"] = -10
        score += signals["current_ratio"]

    return {"score": round(np.clip(score, 0, 100), 1), "signals": signals}


def score_momentum(price_history: Optional[pd.DataFrame]) -> dict:
    """
    Momentum Factor: Reward recent price strength, skip 1-month reversal.
    Classic Fama-French: 12-month return minus 1-month return.
    Score range: 0-100
    """
    if price_history is None or len(price_history) < 60:
        return {"score": 50.0, "signals": {}, "note": "insufficient data"}

    closes = price_history["Close"]
    current = float(closes.iloc[-1])

    signals = {}
    score = 50.0

    # 12-1 month momentum (skip last month to avoid reversal)
    if len(closes) >= 252:
        ret_12m = (current / float(closes.iloc[-252]) - 1) * 100
    elif len(closes) >= 120:
        ret_12m = (current / float(closes.iloc[0]) - 1) * 100
    else:
        ret_12m = None

    ret_1m = (current / float(closes.iloc[-21]) - 1) * 100 if len(closes) >= 21 else None

    if ret_12m is not None:
        signals["ret_12m"] = round(ret_12m, 2)
        if ret_12m > 30:    score += 30
        elif ret_12m > 15:  score += 20
        elif ret_12m > 5:   score += 10
        elif ret_12m > -5:  score += 0
        elif ret_12m > -15: score -= 10
        else:               score -= 20

    # RSI-like signal from last 14 days
    if len(closes) >= 14:
        daily_changes = closes.diff().dropna().iloc[-14:]
        gains  = daily_changes[daily_changes > 0].sum()
        losses = -daily_changes[daily_changes < 0].sum()
        if losses > 0:
            rs  = gains / losses
            rsi = 100 - (100 / (1 + rs))
            signals["rsi_14"] = round(float(rsi), 1)
            if rsi < 30:      score += 15   # oversold = buy opportunity
            elif rsi < 45:    score += 5
            elif rsi > 70:    score -= 15   # overbought
            elif rsi > 55:    score -= 5

    # 20-day trend direction (price vs MA20)
    if len(closes) >= 20:
        ma20 = float(closes.iloc[-20:].mean())
        pct_vs_ma20 = (current / ma20 - 1) * 100
        signals["pct_vs_ma20"] = round(pct_vs_ma20, 2)
        if pct_vs_ma20 > 5:    score += 10
        elif pct_vs_ma20 > 0:  score += 5
        elif pct_vs_ma20 < -5: score -= 10
        else:                   score -= 2

    return {"score": round(np.clip(score, 0, 100), 1), "signals": signals}


def score_yield(fund: dict) -> dict:
    """
    Yield Factor: Reward high, sustainable dividend payers.
    Score range: 0-100
    """
    score = 40.0  # lower baseline (many SET stocks pay dividends)
    signals = {}

    div_yield = _safe(fund.get("dividend_yield"))
    if div_yield:
        div_pct = div_yield * 100
        signals["div_yield"] = round(div_pct, 2)
        if div_pct > 5:    score += 40
        elif div_pct > 3:  score += 28
        elif div_pct > 1:  score += 15
        else:              score += 5

    payout = _safe(fund.get("payout_ratio"))
    if payout:
        if 0.3 < payout < 0.7:  signals["payout"] = "sustainable"; score += 10
        elif payout > 1.0:       signals["payout"] = "unsustainable"; score -= 20
        else:                    signals["payout"] = "ok"

    return {"score": round(np.clip(score, 0, 100), 1), "signals": signals}


def compute_composite_score(fund: dict, price_history: Optional[pd.DataFrame] = None) -> dict:
    """
    Compute the weighted composite factor score for a stock.
    Returns all individual factor scores plus the final composite.
    """
    value    = score_value(fund)
    quality  = score_quality(fund)
    momentum = score_momentum(price_history)
    yield_f  = score_yield(fund)

    w = config.SIGNAL_WEIGHTS
    composite = (
        value["score"]    * w["value"] +
        quality["score"]  * w["quality"] +
        momentum["score"] * w["momentum"] +
        yield_f["score"]  * w["sentiment"]   # using sentiment weight for yield here
    )

    # Determine signal strength
    if composite >= 70:   signal = "STRONG BUY"
    elif composite >= 60: signal = "BUY"
    elif composite >= 50: signal = "HOLD"
    elif composite >= 40: signal = "WEAK"
    else:                 signal = "AVOID"

    return {
        "symbol":    fund.get("symbol", "?"),
        "composite": round(composite, 1),
        "signal":    signal,
        "value":     value,
        "quality":   quality,
        "momentum":  momentum,
        "yield":     yield_f,
    }
