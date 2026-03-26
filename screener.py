"""
screener.py
The main signal pipeline: screen SET stocks, rank by composite signal,
send top candidates to Claude AI, and generate trade recommendations.
Run this daily (or on-demand) to find opportunities.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

import config
from data.fetcher import fetch_fundamentals, fetch_price_history, fetch_batch, get_current_price
from data.validator import is_safe_to_trade
from signals.dcf import run_dcf
from signals.factors import compute_composite_score
from signals.ai_signal import analyze_stock
from portfolio.paper_trader import PaperTrader
from logs.logger import Logger

_logger = Logger()

_JSONL_PATH = "logs/trading.jsonl"


def _get_last_ai_conviction(symbol: str) -> Optional[int]:
    """
    Scan trading.jsonl and return the conviction score from the most recent
    AI_ANALYSIS event for `symbol`. Returns None if no record found.
    """
    if not os.path.exists(_JSONL_PATH):
        return None
    last = None
    try:
        with open(_JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("event_type") == "AI_ANALYSIS" and ev.get("symbol") == symbol:
                        last = ev.get("conviction")
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return last


def check_conviction_drops(
    trader: "PaperTrader",
    results: list,
    dry_run: bool = False,
) -> int:
    """
    For each held position that received a new AI analysis this screener run,
    sell automatically if the new conviction score is < 4.

    Args:
        trader:  PaperTrader with current open positions
        results: Screener output (must include AI results)
        dry_run: If True, log the decision but don't actually sell

    Returns:
        Number of positions closed due to conviction drop
    """
    # Build map: symbol -> new AI conviction (only for AI-analyzed stocks this run)
    ai_map = {
        r["symbol"]: r["ai"]["conviction"]
        for r in results
        if r.get("ai") and r["ai"].get("conviction") is not None
    }

    sold = 0
    for symbol in list(trader.portfolio["positions"]):
        if symbol not in ai_map:
            continue  # not screened this run — leave it alone

        new_conviction = ai_map[symbol]
        pos = trader.portfolio["positions"][symbol]

        # Use conviction recorded at BUY time, not the log (log already has today's value)
        entry_conviction = pos.get("ai_conviction")
        if entry_conviction is None:
            continue  # no entry conviction stored — can't evaluate a drop

        drop = entry_conviction - new_conviction
        # Real drop: fell 2+ points from entry AND current conviction is below 4
        if not (drop >= 2 and new_conviction < 4):
            print(f"  {symbol}: conviction {new_conviction}/10 (entry {entry_conviction}/10, "
                  f"drop={drop}) — not a real drop, holding.")
            continue

        price = get_current_price(symbol) or pos["avg_cost"]
        print(f"  ⚠ CONVICTION DROP: {symbol} sold - AI conviction fell to "
              f"{new_conviction}/10 (was {entry_conviction}/10 at entry, drop={drop})")

        if not dry_run:
            sell_result = trader.sell(
                symbol, price,
                f"CONVICTION DROP: AI conviction {new_conviction}/10 (was {entry_conviction}/10 at entry)"
            )
            if "error" not in sell_result:
                sold += 1
                _logger.conviction_drop_sell(
                    symbol=symbol,
                    old_conviction=entry_conviction,
                    new_conviction=new_conviction,
                    exit_price=price,
                    pnl_thb=sell_result.get("realized_pl", 0),
                    pnl_pct=sell_result.get("realized_pct", 0),
                )

    return sold


def screen_universe(symbols: list = None, use_ai: bool = True, top_n: int = 5) -> list:
    """
    Full screening pipeline:
    1. Fetch fundamentals for all symbols
    2. Compute DCF + factor scores for each
    3. Filter to top candidates
    4. Run Claude AI analysis on top candidates only (cost control)
    5. Return ranked list of trade opportunities

    Args:
        symbols:  List of SET tickers. Defaults to config.TEST_UNIVERSE
        use_ai:   Whether to call Claude API (costs money per call)
        top_n:    Number of top stocks to send to Claude

    Returns:
        List of opportunity dicts, sorted by conviction score
    """
    if symbols is None:
        symbols = config.TEST_UNIVERSE

    print(f"\n{'='*60}")
    print(f"  SET AI SCREENER  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Universe: {len(symbols)} stocks | AI analysis: {use_ai}")
    print(f"{'='*60}")

    _logger.screen_start(universe=symbols, use_ai=use_ai)

    results = []

    # Phase 1: Data + Quantitative Scoring
    print("\nPhase 1: Fetching data and computing signals...\n")
    for symbol in symbols:
        print(f"  Analyzing {symbol}...", end=" ")

        fund = fetch_fundamentals(symbol)
        if not fund:
            print("no data")
            continue

        price_hist = fetch_price_history(symbol, period="1y")

        # Data quality gate: skip this stock if data is suspect
        if price_hist is not None and not price_hist.empty:
            if not is_safe_to_trade(symbol, price_hist):
                print(f"skipped (data quality check failed)")
                continue
        
        dcf        = run_dcf(fund)
        factors    = compute_composite_score(fund, price_hist)

        price = fund.get("current_price", 0)
        if not price:
            print("no price")
            continue

        # Quick pre-filter: skip obviously bad stocks
        pe = fund.get("pe_ratio")
        if pe and pe > config.MAX_PE_RATIO:
            print(f"filtered (P/E={pe:.0f} too high)")
            continue

        de = fund.get("debt_equity")
        if de and de > config.MAX_DEBT_EQUITY * 100:
            print(f"filtered (D/E={de:.0f} too high)")
            continue

        result = {
            "symbol":         symbol,
            "name":           fund.get("name", symbol),
            "sector":         fund.get("sector", "Unknown"),
            "price":          round(price, 2),
            "fundamentals":   fund,
            "dcf":            dcf,
            "factors":        factors,
            "composite":      factors["composite"],
            "quant_signal":   factors["signal"],
            "ai":             None,
            "screened_at":    datetime.now().isoformat(),
        }
        results.append(result)
        _logger.screen_result(
            symbol=symbol,
            composite_score=factors["composite"],
            quant_signal=factors["signal"],
            dcf_verdict=dcf.get("verdict", "N/A") if dcf else "N/A",
            price=price,
        )
        print(f"composite={factors['composite']:.0f} | {factors['signal']} | "
              f"DCF: {dcf.get('verdict', 'N/A') if dcf else 'N/A'}")

    # Sort by composite score
    results.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\nPhase 1 complete: {len(results)} stocks passed initial screen.")

    # Phase 2: AI Analysis on top candidates only
    if use_ai and results:
        candidates = [r for r in results if r["composite"] >= config.MIN_COMPOSITE_SCORE][:top_n]
        print(f"\nPhase 2: Running Claude AI analysis on top {len(candidates)} candidates...\n")

        for r in candidates:
            symbol = r["symbol"]
            print(f"  Claude analyzing {symbol} ({r['name'][:30]})...", end=" ")

            dcf = r["dcf"]
            if dcf is None:
                print("skipped (no valuation data)")
                r["ai"] = None
                continue

            # Bank model uses "BANK_PB_ROE" method — it has no dcf_valid flag; allow through
            is_bank_model = dcf.get("method") == "BANK_PB_ROE"
            if not is_bank_model and not dcf.get("dcf_valid", False):
                print("skipped (no valid DCF)")
                r["ai"] = None
                continue

            if is_bank_model:
            ai_result = analyze_stock(
                symbol       = symbol,
                fundamentals = r["fundamentals"],
                dcf_result   = r["dcf"],
                factor_scores= r["factors"],
                news_headlines=[]  # extend this with real news scraping
            )
            r["ai"] = ai_result
            action     = ai_result.get("action", "HOLD")
            conviction = ai_result.get("conviction", 0)
            _logger.ai_analysis(
                symbol=symbol,
                action=action,
                conviction=conviction,
                position_size_pct=ai_result.get("position_size_pct", 0),
                thesis=ai_result.get("thesis", ""),
                bull_case=ai_result.get("bull_case", ""),
                bear_case=ai_result.get("bear_case", ""),
                key_risks=ai_result.get("key_risks", []),
                red_flags=ai_result.get("red_flags", []),
            )
            print(f"{action} | conviction={conviction}/10")

    # Final sort: AI conviction (if available) then composite score
    def sort_key(r):
        ai_conv = r["ai"]["conviction"] if r["ai"] else 0
        return (ai_conv * 10 + r["composite"])

    results.sort(key=sort_key, reverse=True)

    # Print summary
    print(f"\n{'='*60}")
    print("  SCREENING RESULTS - RANKED BY OPPORTUNITY")
    print(f"{'='*60}")
    print(f"  {'SYMBOL':<8} {'SCORE':>6} {'QUANT':>12} {'AI':>10} {'CONV':>6} {'PRICE':>8}")
    print(f"  {'-'*56}")
    for r in results:
        ai_action = r["ai"]["action"] if r["ai"] else "N/A"
        ai_conv   = r["ai"]["conviction"] if r["ai"] else "-"
        mos = r["dcf"].get("margin_of_safety") if r["dcf"] else None
        mos_str = f"{mos:+.0f}% MoS" if mos is not None else "DCF N/A"
        print(f"  {r['symbol']:<8} {r['composite']:>6.0f} {r['quant_signal']:>12} "
              f"{ai_action:>10} {str(ai_conv):>6} {r['price']:>8.2f}")
    print(f"{'='*60}\n")

    return results


def auto_trade(
    trader: PaperTrader,
    opportunities: list,
    dry_run: bool = False
) -> list:
    """
    Execute paper trades based on screener output.

    Args:
        trader:        PaperTrader instance
        opportunities: Screener results (sorted by conviction)
        dry_run:       If True, show what would be traded but don't execute

    Returns:
        List of executed (or simulated) trade records
    """
    executed = []

    # First: check stop-losses on existing positions
    print("\nChecking stop-losses on open positions...")
    triggers = trader.check_stop_losses()
    _logger.stop_loss_check(
        positions_checked=len(trader.portfolio["positions"]),
        triggered=[t["symbol"] for t in triggers],
    )
    for trigger in triggers:
        print(f"  ! {trigger['symbol']}: {trigger['reason']}")
        if not dry_run:
            pos = trader.portfolio["positions"].get(trigger["symbol"], {})
            sell_result = trader.sell(trigger["symbol"], trigger["price"], trigger["reason"])
            if "error" not in sell_result:
                _logger.stop_loss_triggered(
                    symbol=trigger["symbol"],
                    entry_price=pos.get("avg_cost", 0),
                    exit_price=trigger["price"],
                    pnl_thb=sell_result.get("realized_pl", 0),
                    pnl_pct=sell_result.get("realized_pct", 0),
                    holding_days=sell_result.get("holding_days", 0),
                )

    # Conviction-drop sells (only when this run included AI analysis)
    has_ai = any(r.get("ai") for r in opportunities)
    if has_ai:
        print("\nChecking AI conviction drops on held positions...")
        check_conviction_drops(trader, opportunities, dry_run=dry_run)

    # Check daily loss limit
    if trader.check_daily_loss_limit():
        print("\n  DAILY LOSS LIMIT HIT. No new trades today.")
        return executed

    # Execute new BUY signals
    print("\nEvaluating new trade signals...\n")
    for opp in opportunities:
        symbol = opp["symbol"]
        ai     = opp.get("ai")
        price  = opp["price"]

        # Already holding this stock?
        if symbol in trader.portfolio["positions"]:
            print(f"  {symbol}: Already in portfolio, skipping.")
            continue

        # Need AI analysis and BUY signal to act
        if not ai or ai["action"] != "BUY":
            continue

        conviction = ai.get("conviction", 0)
        pos_size   = ai.get("position_size_pct", 5)
        stop_loss  = ai.get("stop_loss_pct", config.STOP_LOSS_PCT * 100) / 100
        thesis     = ai.get("thesis", "Quantitative signal")
        red_flags  = ai.get("red_flags", [])

        # Skip low-conviction picks
        if conviction < 6:
            print(f"  {symbol}: Conviction {conviction}/10 too low, skipping.")
            continue

        # Skip stocks with red flags
        if len(red_flags) >= 2:
            print(f"  {symbol}: {len(red_flags)} red flags, skipping.")
            for flag in red_flags:
                print(f"    - {flag}")
            continue

        print(f"  SIGNAL: {symbol} | Conviction {conviction}/10 | Size {pos_size}%")
        print(f"  Thesis: {thesis[:80]}...")
        if red_flags:
            print(f"  Flags: {', '.join(red_flags)}")

        if not dry_run:
            result = trader.buy(
                symbol           = symbol,
                price            = price,
                position_size_pct= pos_size,
                reason           = thesis,
                stop_loss_pct    = stop_loss,
                ai_conviction    = conviction,
            )
            if "error" not in result:
                executed.append(result)
        else:
            print(f"  [DRY RUN] Would BUY {symbol} @ {price:.2f} THB ({pos_size}% position)")
            executed.append({"symbol": symbol, "action": "BUY (simulated)", "price": price})

    return executed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SET AI Stock Screener")
    parser.add_argument("--universe", choices=["test", "set50"], default="test",
                       help="Stock universe to scan")
    parser.add_argument("--no-ai", action="store_true",
                       help="Skip Claude AI analysis (faster, free)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show signals without executing trades")
    parser.add_argument("--no-trade", action="store_true",
                       help="Screen only, do not attempt to trade")
    args = parser.parse_args()

    universe = config.SET50_UNIVERSE if args.universe == "set50" else config.TEST_UNIVERSE
    use_ai   = not args.no_ai

    # Run screener
    opportunities = screen_universe(
        symbols = universe,
        use_ai  = use_ai,
        top_n   = 5,
    )

    # Execute trades (or show what would be executed)
    if not args.no_trade:
        trader = PaperTrader()
        print("\n" + "="*60)
        print("  PAPER TRADING ENGINE")
        print("="*60)
        trader.print_portfolio()

        auto_trade(trader, opportunities, dry_run=args.dry_run)

        print("\nUpdated Portfolio:")
        trader.print_portfolio()
        trader.log_snapshot()
