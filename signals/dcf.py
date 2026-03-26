"""
signals/dcf.py
Automated DCF (Discounted Cash Flow) valuation engine for SET stocks.
Computes intrinsic value and margin of safety for each stock.
Banks are valued using P/B + ROE instead of DCF (deposits are raw material, not debt).
"""

import numpy as np
from typing import Optional
import sys
sys.path.append("..")
import config

# Thai bank tickers — use P/B + ROE model instead of DCF
BANK_TICKERS = ["KBANK", "SCB", "KTB", "BBL", "BAY", "TISCO", "KKP", "TCAP"]


def compute_wacc(fundamentals: dict) -> float:
    """
    Weighted Average Cost of Capital for a SET-listed company.
    Uses CAPM for cost of equity, Thai corporate bond rate for cost of debt.
    """
    beta = fundamentals.get("beta") or 1.0
    beta = max(0.3, min(beta, 3.0))  # clamp to reasonable range

    # CAPM: Cost of Equity = Risk-Free + Beta * Equity Risk Premium
    cost_of_equity = config.RISK_FREE_RATE + beta * config.EQUITY_RISK_PREMIUM

    # Capital structure
    total_debt   = fundamentals.get("total_debt", 0) or 0
    market_cap   = fundamentals.get("market_cap", 1) or 1
    total_capital = total_debt + market_cap

    debt_ratio   = total_debt / total_capital
    equity_ratio = 1 - debt_ratio

    # Cost of debt: Thai corporate lending rate, net of tax shield
    cost_of_debt = 0.055  # ~5.5% for investment-grade SET companies
    tax_rate     = 0.20   # Thai corporate tax rate

    wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))

    # Sanity bounds: 5% to 20%
    return max(0.05, min(wacc, 0.20))


def estimate_fcf_growth(fundamentals: dict) -> float:
    """
    Estimate near-term FCF growth rate from available signals.
    Uses revenue growth, earnings growth, and sector defaults.
    """
    rev_growth      = fundamentals.get("revenue_growth") or 0
    earnings_growth = fundamentals.get("earnings_growth") or 0

    # Blend the two signals
    if rev_growth and earnings_growth:
        raw = (rev_growth * 0.4) + (earnings_growth * 0.6)
    elif earnings_growth:
        raw = earnings_growth
    elif rev_growth:
        raw = rev_growth * 0.8  # slightly haircut revenue growth
    else:
        raw = 0.05  # default 5% if no data

    # Clamp to realistic bounds: -10% to +25%
    return max(-0.10, min(raw, 0.25))


def calculate_bank_score(symbol: str, data: dict) -> dict:
    """
    Bank-specific valuation using P/B ratio and ROE.
    Standard DCF is inappropriate for banks because deposits are their
    raw material, not debt — DCF produces garbage values for them.

    Returns dict with method, pb_ratio, roe_pct, pb_score, roe_score,
    composite_score, and verdict.
    """
    pb  = data.get("pb_ratio")
    roe = data.get("roe")   # yfinance returns as fraction: 0.12 = 12%

    if pb is None or roe is None:
        return {
            "method":          "BANK_PB_ROE",
            "pb_ratio":        None,
            "roe_pct":         None,
            "pb_score":        50,
            "roe_score":       50,
            "composite_score": 50,
            "verdict":         "DATA_UNAVAILABLE",
        }

    roe_pct = roe * 100

    # P/B scoring — lower is better for banks
    if pb < 0.8:
        pb_score = 90
    elif pb < 1.0:
        pb_score = 75
    elif pb < 1.3:
        pb_score = 60
    elif pb < 1.6:
        pb_score = 45
    else:
        pb_score = 25

    # ROE scoring — higher is better
    if roe_pct > 15:
        roe_score = 90
    elif roe_pct > 12:
        roe_score = 75
    elif roe_pct > 9:
        roe_score = 60
    elif roe_pct > 6:
        roe_score = 40
    else:
        roe_score = 20

    composite = int(pb_score * 0.5 + roe_score * 0.5)

    if composite >= 70:
        verdict = "UNDERVALUED"
    elif composite < 40:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIR"

    return {
        "method":          "BANK_PB_ROE",
        "pb_ratio":        round(pb, 3),
        "roe_pct":         round(roe_pct, 2),
        "pb_score":        pb_score,
        "roe_score":       roe_score,
        "composite_score": composite,
        "verdict":         verdict,
    }


def run_dcf(fundamentals: dict) -> Optional[dict]:
    """
    Run a 5-year DCF analysis on a stock.
    Banks are automatically routed to calculate_bank_score() instead.

    Returns:
        dict with intrinsic_price, margin_of_safety, wacc, verdict
        or None if insufficient data
    """
    symbol        = fundamentals.get("symbol", "UNKNOWN")

    # Banks cannot be valued with DCF — route to the dedicated model
    if symbol in BANK_TICKERS:
        print(f"  Using BANK MODEL (P/B + ROE) for {symbol}")
        return calculate_bank_score(symbol, fundamentals)

    current_price = fundamentals.get("current_price", 0)
    shares        = fundamentals.get("shares_outstanding", 0)

    if not current_price or not shares or shares == 0:
        return None

    # Get Free Cash Flow
    ocf   = fundamentals.get("operating_cash_flow", None)
    capex = fundamentals.get("capital_expenditure", None)

    if ocf is None:
        return None  # Cannot run DCF without cash flow data

    fcf = ocf + (capex or 0)  # capex is already negative in yfinance

    if fcf <= 0:
        # Negative FCF: stock is burning cash, skip DCF
        return {
            "symbol":            symbol,
            "intrinsic_price":   None,
            "current_price":     current_price,
            "margin_of_safety":  None,
            "wacc_pct":          None,
            "fcf_growth_pct":    None,
            "verdict":           "NEGATIVE FCF - DCF N/A",
            "dcf_valid":         False,
        }

    wacc        = compute_wacc(fundamentals)
    fcf_growth  = estimate_fcf_growth(fundamentals)
    terminal_g  = config.TERMINAL_GROWTH
    years       = config.DCF_PROJECTION_YRS

    # Project FCF and discount back to present value
    projected_fcfs = [fcf * (1 + fcf_growth) ** yr for yr in range(1, years + 1)]
    pv_fcfs        = sum(cf / (1 + wacc) ** t for t, cf in enumerate(projected_fcfs, 1))

    # Terminal value: Gordon Growth Model
    terminal_fcf   = projected_fcfs[-1] * (1 + terminal_g)
    terminal_value = terminal_fcf / (wacc - terminal_g) if wacc > terminal_g else 0
    pv_terminal    = terminal_value / (1 + wacc) ** years

    # Enterprise Value -> Equity Value -> Per Share
    enterprise_value = pv_fcfs + pv_terminal
    net_debt         = (fundamentals.get("total_debt", 0) or 0) - (fundamentals.get("total_cash", 0) or 0)
    equity_value     = enterprise_value - net_debt
    intrinsic_price  = equity_value / shares

    if intrinsic_price <= 0:
        return {
            "symbol":           symbol,
            "intrinsic_price":  0,
            "current_price":    current_price,
            "margin_of_safety": -100,
            "wacc_pct":         round(wacc * 100, 2),
            "fcf_growth_pct":   round(fcf_growth * 100, 2),
            "verdict":          "OVERVALUED - Negative equity value",
            "dcf_valid":        False,
        }

    mos = (intrinsic_price - current_price) / intrinsic_price * 100

    if mos >= config.MARGIN_OF_SAFETY * 100:
        verdict = "UNDERVALUED"
    elif mos >= 0:
        verdict = "FAIRLY VALUED"
    elif mos >= -15:
        verdict = "SLIGHTLY OVERVALUED"
    else:
        verdict = "OVERVALUED"

    return {
        "symbol":           symbol,
        "intrinsic_price":  round(intrinsic_price, 2),
        "current_price":    round(current_price, 2),
        "margin_of_safety": round(mos, 1),
        "wacc_pct":         round(wacc * 100, 2),
        "fcf_growth_pct":   round(fcf_growth * 100, 2),
        "pv_fcf":           round(pv_fcfs, 0),
        "pv_terminal":      round(pv_terminal, 0),
        "verdict":          verdict,
        "dcf_valid":        True,
    }


def get_valuation(symbol: str) -> Optional[dict]:
    """
    Standalone valuation entry point — fetches data then routes:
      - Banks  → calculate_bank_score() (P/B + ROE)
      - Others → run_dcf()
    """
    from data.fetcher import fetch_fundamentals
    data = fetch_fundamentals(symbol)
    if data is None:
        return None
    return run_dcf(data)  # run_dcf handles bank routing internally
