"""
examples/generate_examples.py
Generates 9 weeks of simulated SET AI Trader history (Jan 12 - Mar 8, 2026)
using real historical price data and the actual scoring logic.

AI conviction is simulated deterministically (no Claude API calls).
"""

import sys
import os
import json
from datetime import datetime, timedelta, timezone

# Set up paths so we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np

import config
from signals.dcf import run_dcf, BANK_TICKERS, calculate_bank_score
from signals.factors import compute_composite_score

# ── Configuration ────────────────────────────────────────────────────────────

SYMBOLS = ["PTT", "ADVANC", "KBANK", "CPALL", "BDMS", "AOT"]
START_DATE = datetime(2026, 1, 12)  # First Sunday
NUM_WEEKS = 9
STARTING_CAPITAL = 25_000.0
MAX_POSITIONS = 3
MAX_POSITION_PCT = 0.20
STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.20
BROKERAGE_FEE_PCT = 0.0025

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(OUTPUT_DIR, "trading_example.jsonl")
PORTFOLIO_PATH = os.path.join(OUTPUT_DIR, "portfolio_example.json")
TRADES_PATH = os.path.join(OUTPUT_DIR, "paper_trades_example.json")


# ── Helpers ──────────────────────────────────────────────────────────────────

def write_event(f, event_type: str, ts: str, data: dict = None):
    entry = {"timestamp": ts, "event_type": event_type}
    if data:
        entry.update(data)
    f.write(json.dumps(entry, default=str) + "\n")


def fetch_weekly_prices(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV for the full simulation period + lookback for momentum."""
    ticker_str = f"{symbol}.BK"
    lookback_start = START_DATE - timedelta(days=400)  # ~1yr for momentum calc
    end_date = START_DATE + timedelta(weeks=NUM_WEEKS + 1)

    print(f"  Fetching {ticker_str}...", end=" ")
    try:
        df = yf.download(ticker_str, start=lookback_start.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"), progress=False)
        if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        print(f"{len(df)} rows")
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return pd.DataFrame()


def fetch_fundamentals_snapshot(symbol: str) -> dict:
    """Fetch current fundamentals (used as a static snapshot for all weeks)."""
    ticker_str = f"{symbol}.BK"
    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info
        fund = {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "total_debt": info.get("totalDebt", 0),
            "total_cash": info.get("totalCash", 0),
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "beta": info.get("beta", 1.0),
        }
        # Cash flow for DCF
        try:
            cf = tk.cashflow
            if not cf.empty and len(cf.columns) > 0:
                ocf_row = [r for r in cf.index if "Operating" in str(r) or "operating" in str(r)]
                capex_row = [r for r in cf.index if "Capital" in str(r) or "capital" in str(r)]
                if ocf_row:
                    fund["operating_cash_flow"] = float(cf.loc[ocf_row[0]].iloc[0])
                if capex_row:
                    fund["capital_expenditure"] = float(cf.loc[capex_row[0]].iloc[0])
        except Exception:
            pass
        return fund
    except Exception as e:
        print(f"  [WARN] Fundamentals failed for {symbol}: {e}")
        return None


def get_friday_close(price_df: pd.DataFrame, sunday: datetime) -> float:
    """Get the Friday closing price for the week ending before this Sunday."""
    friday = sunday - timedelta(days=2)
    # Search backward up to 5 days to find the nearest trading day
    for offset in range(6):
        day = friday - timedelta(days=offset)
        day_str = day.strftime("%Y-%m-%d")
        matches = price_df.loc[price_df.index.strftime("%Y-%m-%d") == day_str]
        if not matches.empty:
            return float(matches["Close"].iloc[0])
    return None


def get_price_history_up_to(price_df: pd.DataFrame, sunday: datetime) -> pd.DataFrame:
    """Get price history up to the Friday before this Sunday (for momentum calc)."""
    friday = sunday - timedelta(days=2)
    mask = price_df.index <= pd.Timestamp(friday)
    return price_df[mask]


def simulate_conviction(composite: float, dcf_result: dict, prev_price: float,
                        current_price: float, week_idx: int) -> tuple:
    """
    Deterministic AI conviction simulation.
    Returns (conviction, action).
    """
    # Price momentum adjustment
    momentum_adj = 0
    if prev_price and current_price and prev_price > 0:
        pct_change = (current_price - prev_price) / prev_price * 100
        if pct_change > 2:
            momentum_adj = 1
        elif pct_change < -2:
            momentum_adj = -1

    dcf_verdict = dcf_result.get("verdict", "") if dcf_result else ""
    is_bank = dcf_result.get("method") == "BANK_PB_ROE" if dcf_result else False

    if composite >= 75 and dcf_verdict == "UNDERVALUED":
        conviction = 8 + momentum_adj
        action = "BUY"
    elif composite >= 70 and is_bank and dcf_verdict == "FAIR":
        conviction = 7 + momentum_adj
        action = "BUY"
    elif composite >= 65:
        conviction = 6 + momentum_adj
        action = "BUY"
    elif composite >= 55:
        conviction = 5 + momentum_adj
        action = "HOLD"
    else:
        conviction = 3 + momentum_adj
        action = "AVOID"

    conviction = max(1, min(10, conviction))
    return conviction, action


# ── Main simulation ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SET AI Trader — Example Data Generator")
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to "
          f"{(START_DATE + timedelta(weeks=NUM_WEEKS-1)).strftime('%Y-%m-%d')}")
    print("=" * 60)

    # Phase 1: Fetch all data
    print("\nPhase 1: Fetching historical price data...\n")
    price_data = {}
    for sym in SYMBOLS:
        df = fetch_weekly_prices(sym)
        if not df.empty:
            price_data[sym] = df

    print(f"\nPhase 2: Fetching fundamentals...\n")
    fundamentals = {}
    for sym in SYMBOLS:
        print(f"  {sym}...", end=" ")
        fund = fetch_fundamentals_snapshot(sym)
        if fund:
            fundamentals[sym] = fund
            print("ok")
        else:
            print("failed")

    if not price_data or not fundamentals:
        print("\nERROR: Could not fetch sufficient data. Aborting.")
        return

    # Phase 2: Simulate weekly trading
    print(f"\nPhase 3: Simulating {NUM_WEEKS} weeks of trading...\n")

    portfolio = {
        "cash": STARTING_CAPITAL,
        "starting_capital": STARTING_CAPITAL,
        "positions": {},
        "created_at": START_DATE.isoformat(),
        "last_updated": START_DATE.isoformat(),
    }
    trades = []
    prev_prices = {}
    conviction_history = {}  # symbol -> last conviction at entry

    log_file = open(LOG_PATH, "w")

    for week_idx in range(NUM_WEEKS):
        sunday = START_DATE + timedelta(weeks=week_idx)
        ts = sunday.replace(hour=9, minute=0, tzinfo=timezone.utc).isoformat()
        print(f"\n--- Week {week_idx + 1}: {sunday.strftime('%Y-%m-%d')} ---")

        # Screen all stocks
        week_results = []
        for sym in SYMBOLS:
            if sym not in price_data or sym not in fundamentals:
                continue

            price = get_friday_close(price_data[sym], sunday)
            if price is None:
                continue

            fund = fundamentals[sym].copy()
            fund["current_price"] = price

            price_hist = get_price_history_up_to(price_data[sym], sunday)
            dcf_result = run_dcf(fund)
            factors = compute_composite_score(fund, price_hist)

            prev_price = prev_prices.get(sym)
            conviction, action = simulate_conviction(
                factors["composite"], dcf_result, prev_price, price, week_idx
            )

            week_results.append({
                "symbol": sym,
                "price": price,
                "composite": factors["composite"],
                "signal": factors["signal"],
                "dcf_verdict": dcf_result.get("verdict", "N/A") if dcf_result else "N/A",
                "conviction": conviction,
                "action": action,
                "dcf_result": dcf_result,
            })

            prev_prices[sym] = price

        # Write WEEKLY_SCREEN event
        screen_data = [
            {
                "symbol": r["symbol"],
                "composite_score": round(r["composite"], 2),
                "quant_signal": r["signal"],
                "dcf_verdict": r["dcf_verdict"],
                "price": round(r["price"], 2),
                "conviction": r["conviction"],
                "action": r["action"],
            }
            for r in week_results
        ]
        write_event(log_file, "WEEKLY_SCREEN", ts, {"stocks": screen_data})

        for r in week_results:
            print(f"  {r['symbol']:<8} price={r['price']:>8.2f}  composite={r['composite']:>5.1f}  "
                  f"conviction={r['conviction']}  {r['action']}")

        # Check existing positions for stop-loss, take-profit, conviction drop
        for sym in list(portfolio["positions"].keys()):
            pos = portfolio["positions"][sym]
            current_price_sym = None
            for r in week_results:
                if r["symbol"] == sym:
                    current_price_sym = r["price"]
                    break
            if current_price_sym is None:
                continue

            entry_price = pos["avg_cost"]
            pct_change = (current_price_sym - entry_price) / entry_price

            # Stop-loss
            if pct_change <= -STOP_LOSS_PCT:
                shares = pos["shares"]
                gross = shares * current_price_sym
                fee = gross * BROKERAGE_FEE_PCT
                net = gross - fee
                pnl = net - pos["total_cost"]
                pnl_pct = (pnl / pos["total_cost"]) * 100

                portfolio["cash"] += net
                del portfolio["positions"][sym]

                trade = {
                    "id": len(trades) + 1, "type": "SELL", "symbol": sym,
                    "shares": shares, "price": round(current_price_sym, 2),
                    "net_proceeds": round(net, 2), "fee": round(fee, 2),
                    "realized_pl": round(pnl, 2), "realized_pct": round(pnl_pct, 2),
                    "reason": "STOP_LOSS", "timestamp": ts,
                }
                trades.append(trade)
                write_event(log_file, "SELL", ts, trade)
                print(f"  >> STOP-LOSS: Sold {sym} @ {current_price_sym:.2f} "
                      f"(P&L: {pnl:+.0f} THB, {pnl_pct:+.1f}%)")
                continue

            # Take-profit
            if pct_change >= TAKE_PROFIT_PCT:
                shares = pos["shares"]
                gross = shares * current_price_sym
                fee = gross * BROKERAGE_FEE_PCT
                net = gross - fee
                pnl = net - pos["total_cost"]
                pnl_pct = (pnl / pos["total_cost"]) * 100

                portfolio["cash"] += net
                del portfolio["positions"][sym]

                trade = {
                    "id": len(trades) + 1, "type": "SELL", "symbol": sym,
                    "shares": shares, "price": round(current_price_sym, 2),
                    "net_proceeds": round(net, 2), "fee": round(fee, 2),
                    "realized_pl": round(pnl, 2), "realized_pct": round(pnl_pct, 2),
                    "reason": "TAKE_PROFIT", "timestamp": ts,
                }
                trades.append(trade)
                write_event(log_file, "SELL", ts, trade)
                print(f"  >> TAKE-PROFIT: Sold {sym} @ {current_price_sym:.2f} "
                      f"(P&L: {pnl:+.0f} THB, {pnl_pct:+.1f}%)")
                continue

            # Conviction drop: fell 3+ from entry AND current < 4
            entry_conviction = conviction_history.get(sym, 5)
            current_conviction = None
            for r in week_results:
                if r["symbol"] == sym:
                    current_conviction = r["conviction"]
                    break
            if current_conviction is not None:
                drop = entry_conviction - current_conviction
                if drop >= 3 and current_conviction < 4:
                    shares = pos["shares"]
                    gross = shares * current_price_sym
                    fee = gross * BROKERAGE_FEE_PCT
                    net = gross - fee
                    pnl = net - pos["total_cost"]
                    pnl_pct = (pnl / pos["total_cost"]) * 100

                    portfolio["cash"] += net
                    del portfolio["positions"][sym]

                    trade = {
                        "id": len(trades) + 1, "type": "SELL", "symbol": sym,
                        "shares": shares, "price": round(current_price_sym, 2),
                        "net_proceeds": round(net, 2), "fee": round(fee, 2),
                        "realized_pl": round(pnl, 2), "realized_pct": round(pnl_pct, 2),
                        "reason": "CONVICTION_DROP",
                        "old_conviction": entry_conviction,
                        "new_conviction": current_conviction,
                        "timestamp": ts,
                    }
                    trades.append(trade)
                    write_event(log_file, "SELL", ts, trade)
                    print(f"  >> CONVICTION DROP: Sold {sym} (conviction {entry_conviction}->{current_conviction})")
                    continue

        # Buy signals: conviction >= 7, not already held, max 3 positions
        for r in sorted(week_results, key=lambda x: x["conviction"], reverse=True):
            sym = r["symbol"]
            if sym in portfolio["positions"]:
                continue
            if len(portfolio["positions"]) >= MAX_POSITIONS:
                break
            if r["conviction"] < 7:
                continue

            # Calculate position size
            portfolio_val = portfolio["cash"] + sum(
                p["shares"] * p["avg_cost"] for p in portfolio["positions"].values()
            )
            capital = min(portfolio_val * MAX_POSITION_PCT, portfolio["cash"] * 0.90)
            if capital < 1000:
                continue

            price = r["price"]
            fee = capital * BROKERAGE_FEE_PCT
            shares = int((capital - fee) / price)
            if shares < 1:
                continue

            total_cost = shares * price + fee
            portfolio["cash"] -= total_cost
            portfolio["positions"][sym] = {
                "shares": shares,
                "avg_cost": round(price, 2),
                "total_cost": round(total_cost, 2),
                "stop_loss_pct": STOP_LOSS_PCT,
                "ai_conviction": r["conviction"],
                "entry_date": ts,
            }
            conviction_history[sym] = r["conviction"]

            trade = {
                "id": len(trades) + 1, "type": "BUY", "symbol": sym,
                "shares": shares, "price": round(price, 2),
                "total_cost": round(total_cost, 2), "fee": round(fee, 2),
                "conviction": r["conviction"], "reason": r["action"],
                "timestamp": ts,
            }
            trades.append(trade)
            write_event(log_file, "BUY", ts, trade)
            print(f"  >> BUY: {sym} {shares} shares @ {price:.2f} THB "
                  f"(conviction={r['conviction']})")

        # Portfolio snapshot
        positions_value = sum(
            pos["shares"] * (next((r["price"] for r in week_results if r["symbol"] == s), pos["avg_cost"]))
            for s, pos in portfolio["positions"].items()
        )
        total_value = portfolio["cash"] + positions_value
        return_pct = (total_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100

        write_event(log_file, "PORTFOLIO_SNAPSHOT", ts, {
            "cash": round(portfolio["cash"], 2),
            "positions_value": round(positions_value, 2),
            "total_value": round(total_value, 2),
            "total_return_pct": round(return_pct, 2),
            "open_positions": [
                {"symbol": s, "shares": p["shares"], "avg_cost": p["avg_cost"]}
                for s, p in portfolio["positions"].items()
            ],
        })

        print(f"  Portfolio: {total_value:,.0f} THB ({return_pct:+.2f}%) | "
              f"Cash: {portfolio['cash']:,.0f} | Positions: {len(portfolio['positions'])}")

    log_file.close()

    # Save final state
    portfolio["last_updated"] = (START_DATE + timedelta(weeks=NUM_WEEKS - 1)).isoformat()
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2)

    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Weeks simulated : {NUM_WEEKS}")
    print(f"  Total trades    : {len(trades)}")
    buys = [t for t in trades if t["type"] == "BUY"]
    sells = [t for t in trades if t["type"] == "SELL"]
    print(f"  Buys            : {len(buys)}")
    print(f"  Sells           : {len(sells)}")
    if sells:
        realized = sum(t.get("realized_pl", 0) for t in sells)
        wins = [t for t in sells if t.get("realized_pl", 0) > 0]
        print(f"  Realized P&L    : {realized:+,.0f} THB")
        print(f"  Win rate        : {len(wins)}/{len(sells)}")
    print(f"  Final cash      : {portfolio['cash']:,.0f} THB")
    print(f"  Open positions  : {len(portfolio['positions'])}")
    for sym, pos in portfolio["positions"].items():
        print(f"    {sym}: {pos['shares']} shares @ {pos['avg_cost']:.2f}")
    total_val = portfolio["cash"] + sum(
        p["shares"] * p["avg_cost"] for p in portfolio["positions"].values()
    )
    ret = (total_val - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    print(f"  Portfolio value : {total_val:,.0f} THB ({ret:+.2f}%)")
    print(f"\n  Output files:")
    print(f"    {LOG_PATH}")
    print(f"    {PORTFOLIO_PATH}")
    print(f"    {TRADES_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
