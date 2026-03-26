"""
run.py
Quick-start entry point with interactive menu.
Run this first to get started.

Usage:
    python run.py
"""

import os
import sys
import json

def check_setup():
    """Check that dependencies and API key are configured."""
    try:
        import yfinance, anthropic, pandas, numpy
        print("  Dependencies: OK")
    except ImportError as e:
        print(f"\n  Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
        return False

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "YOUR_KEY_HERE":
        import config
        if config.ANTHROPIC_API_KEY == "YOUR_KEY_HERE":
            print("\n  ANTHROPIC_API_KEY not set.")
            print("  Set it in config.py or as environment variable:")
            print("    export ANTHROPIC_API_KEY=sk-ant-...")
            print("\n  You can still run in --no-ai mode without it.")
    else:
        print("  API Key: OK")

    return True


def menu():
    print("""
╔══════════════════════════════════════════════════════════╗
║          SET AI PAPER TRADER  -  MAIN MENU               ║
╠══════════════════════════════════════════════════════════╣
║  1. Run screener (test universe, with AI)                ║
║  2. Run screener (test universe, no AI - faster/free)    ║
║  3. Run screener (SET50 full universe)                   ║
║  4. View portfolio summary                               ║
║  5. Manual BUY order                                     ║
║  6. Manual SELL order                                    ║
║  7. Check stop-losses + take-profits                     ║
║  8. View trade history                                   ║
║  9. Reset portfolio (start fresh)                        ║
║  0. Exit                                                 ║
╚══════════════════════════════════════════════════════════╝""")
    return input("\n  Choose option: ").strip()


def run():
    print("\n  SET AI Paper Trader - Starting up...")
    if not check_setup():
        return

    from portfolio.paper_trader import PaperTrader
    import config

    while True:
        choice = menu()

        if choice == "1":
            from screener import screen_universe, auto_trade
            opps = screen_universe(config.TEST_UNIVERSE, use_ai=True)
            trader = PaperTrader()
            auto_trade(trader, opps, dry_run=False)
            trader.print_portfolio()

        elif choice == "2":
            from screener import screen_universe, auto_trade
            opps = screen_universe(config.TEST_UNIVERSE, use_ai=False)
            trader = PaperTrader()
            auto_trade(trader, opps, dry_run=True)  # dry run since no AI conviction
            trader.print_portfolio()

        elif choice == "3":
            from screener import screen_universe, auto_trade
            print("\n  Scanning SET50 (this will take ~2 minutes)...")
            opps = screen_universe(config.SET50_UNIVERSE, use_ai=True, top_n=8)
            trader = PaperTrader()
            auto_trade(trader, opps, dry_run=False)
            trader.print_portfolio()

        elif choice == "4":
            trader = PaperTrader()
            trader.print_portfolio()

        elif choice == "5":
            # Manual BUY
            trader = PaperTrader()
            symbol = input("\n  Ticker symbol (e.g. PTT): ").upper().strip()
            try:
                from data.fetcher import get_current_price
                price = get_current_price(symbol)
                if price:
                    print(f"  Current price: {price:.2f} THB")
                    use_price = input(f"  Use this price? (y/n): ")
                    if use_price.lower() != 'y':
                        price = float(input("  Enter price: "))
                else:
                    price = float(input("  Enter price manually: "))

                pct   = float(input("  Position size % of portfolio (e.g. 8): "))
                reason = input("  Reason for buying: ")
                result = trader.buy(symbol, price, pct, reason)
                if "error" in result:
                    print(f"\n  Error: {result['error']}")
                else:
                    trader.print_portfolio()
            except (ValueError, KeyboardInterrupt):
                print("  Cancelled.")

        elif choice == "6":
            # Manual SELL
            trader = PaperTrader()
            if not trader.portfolio["positions"]:
                print("\n  No open positions.")
                continue
            print("\n  Open positions:", list(trader.portfolio["positions"].keys()))
            symbol = input("  Ticker to sell: ").upper().strip()
            try:
                from data.fetcher import get_current_price
                price = get_current_price(symbol)
                if price:
                    print(f"  Current price: {price:.2f} THB")
                    use_price = input(f"  Use this price? (y/n): ")
                    if use_price.lower() != 'y':
                        price = float(input("  Enter price: "))
                else:
                    price = float(input("  Enter price manually: "))

                reason = input("  Reason for selling: ")
                result = trader.sell(symbol, price, reason)
                if "error" in result:
                    print(f"\n  Error: {result['error']}")
                else:
                    trader.print_portfolio()
            except (ValueError, KeyboardInterrupt):
                print("  Cancelled.")

        elif choice == "7":
            trader = PaperTrader()
            print("\n  Checking stop-losses and take-profits...")
            triggers = trader.check_stop_losses()
            if not triggers:
                print("  All positions are within limits.")
            else:
                for t in triggers:
                    print(f"\n  TRIGGERED: {t['symbol']} - {t['reason']}")
                    auto_sell = input(f"  Auto-sell {t['symbol']}? (y/n): ")
                    if auto_sell.lower() == 'y':
                        trader.sell(t['symbol'], t['price'], t['reason'])

        elif choice == "8":
            trader = PaperTrader()
            trades = trader.trades
            if not trades:
                print("\n  No trades yet.")
                continue
            print(f"\n  {'ID':>4} {'TYPE':<6} {'SYMBOL':<8} {'SHARES':>6} {'PRICE':>10} {'P&L':>12} {'DATE':<20}")
            print("  " + "-"*72)
            for t in trades[-20:]:  # show last 20
                pl_str = f"{t.get('realized_pl', 0):+,.0f}" if t["type"] == "SELL" else ""
                print(f"  {t['id']:>4} {t['type']:<6} {t['symbol']:<8} {t['shares']:>6} "
                      f"{t['price']:>10.2f} {pl_str:>12} {t['timestamp'][:19]}")

        elif choice == "9":
            confirm = input("\n  Reset portfolio? All trades will be cleared. (yes/no): ")
            if confirm.lower() == "yes":
                import config
                for f in [config.PORTFOLIO_FILE, config.TRADES_FILE]:
                    if os.path.exists(f):
                        os.remove(f)
                # Clear cache
                import shutil
                if os.path.exists(config.DATA_DIR):
                    shutil.rmtree(config.DATA_DIR)
                    os.makedirs(config.DATA_DIR)
                print("  Portfolio reset. Starting with 25,000 THB.")

        elif choice == "0":
            print("\n  Goodbye!\n")
            break

        else:
            print("  Invalid option.")


if __name__ == "__main__":
    run()
