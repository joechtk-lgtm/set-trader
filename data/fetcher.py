"""
data/fetcher.py
Fetches price and fundamental data for SET-listed stocks.
Uses yfinance with .BK suffix for Bangkok Stock Exchange.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, time
from datetime import datetime, timedelta
from typing import Optional
import sys
sys.path.append("..")
import config

def _download_set(symbol: str, **kwargs) -> "pd.DataFrame":
    """Wrapper around yf.download that handles multi-level columns from newer yfinance."""
    import yfinance as yf
    df = _download_set(symbol, **kwargs)
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    return df



os.makedirs(config.DATA_DIR, exist_ok=True)


def set_ticker(symbol: str) -> str:
    """Convert bare symbol to yfinance SET format."""
    if symbol.endswith(".BK"):
        return symbol
    return f"{symbol}.BK"


def fetch_price_history(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV price history for a SET stock.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        ticker = yf.Ticker(set_ticker(symbol))
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"  [WARN] Price fetch failed for {symbol}: {e}")
        return None


def fetch_fundamentals(symbol: str) -> Optional[dict]:
    """
    Fetch fundamental data for a SET stock.
    Caches results for 24 hours to avoid repeated API calls.
    """
    cache_file = os.path.join(config.DATA_DIR, f"{symbol}_fundamentals.json")

    # Return cached data if fresh (< 24 hours old)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400:
            with open(cache_file) as f:
                return json.load(f)

    try:
        ticker = yf.Ticker(set_ticker(symbol))
        info = ticker.info

        # Extract key fundamentals
        fundamentals = {
            "symbol":           symbol,
            "name":             info.get("longName", symbol),
            "sector":           info.get("sector", "Unknown"),
            "industry":         info.get("industry", "Unknown"),
            "current_price":    info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap":       info.get("marketCap", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),

            # Valuation multiples
            "pe_ratio":         info.get("trailingPE", None),
            "forward_pe":       info.get("forwardPE", None),
            "pb_ratio":         info.get("priceToBook", None),
            "ev_ebitda":        info.get("enterpriseToEbitda", None),
            "ps_ratio":         info.get("priceToSalesTrailing12Months", None),

            # Quality metrics
            "roe":              info.get("returnOnEquity", None),
            "roa":              info.get("returnOnAssets", None),
            "profit_margin":    info.get("profitMargins", None),
            "operating_margin": info.get("operatingMargins", None),
            "gross_margin":     info.get("grossMargins", None),

            # Balance sheet
            "total_debt":       info.get("totalDebt", 0),
            "total_cash":       info.get("totalCash", 0),
            "debt_equity":      info.get("debtToEquity", None),
            "current_ratio":    info.get("currentRatio", None),

            # Growth
            "revenue_growth":   info.get("revenueGrowth", None),
            "earnings_growth":  info.get("earningsGrowth", None),

            # Dividend
            "dividend_yield":   info.get("dividendYield", None),
            "payout_ratio":     info.get("payoutRatio", None),

            # Risk
            "beta":             info.get("beta", 1.0),

            "fetched_at":       datetime.now().isoformat(),
        }

        # Fetch cash flow for DCF
        try:
            cf = ticker.cashflow
            if not cf.empty and len(cf.columns) > 0:
                ocf_row = [r for r in cf.index if "Operating" in str(r) or "operating" in str(r)]
                capex_row = [r for r in cf.index if "Capital" in str(r) or "capital" in str(r)]
                if ocf_row:
                    fundamentals["operating_cash_flow"] = float(cf.loc[ocf_row[0]].iloc[0])
                if capex_row:
                    fundamentals["capital_expenditure"] = float(cf.loc[capex_row[0]].iloc[0])
        except Exception:
            pass

        # Cache the result
        with open(cache_file, "w") as f:
            json.dump(fundamentals, f, indent=2, default=str)

        return fundamentals

    except Exception as e:
        print(f"  [WARN] Fundamentals fetch failed for {symbol}: {e}")
        return None


def fetch_batch(symbols: list, delay: float = 0.5) -> dict:
    """
    Fetch fundamentals for a list of symbols with rate limiting.
    Returns dict: {symbol: fundamentals_dict}
    """
    results = {}
    total = len(symbols)

    print(f"\nFetching data for {total} stocks...")
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{total}] {sym}...", end=" ")
        data = fetch_fundamentals(sym)
        if data:
            results[sym] = data
            print(f"ok (price: {data.get('current_price', '?')} THB)")
        else:
            print("failed")
        time.sleep(delay)

    print(f"\nFetched {len(results)}/{total} stocks successfully.\n")
    return results


def get_current_price(symbol: str) -> Optional[float]:
    """Get the latest price for a symbol."""
    try:
        ticker = yf.Ticker(set_ticker(symbol))
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None
    except Exception:
        return None
