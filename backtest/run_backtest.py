"""
backtest/run_backtest.py

Main entry point for running historical simulations.

Usage examples:

    # Backtest best strategy on test universe (2020-2024)
    python backtest/run_backtest.py

    # Try a specific strategy
    python backtest/run_backtest.py --strategy momentum_12_1

    # Compare all strategies
    python backtest/run_backtest.py --compare

    # Custom date range and universe
    python backtest/run_backtest.py --start 2019-01-01 --end 2023-12-31 --universe set50

    # Walk-forward validation (most realistic)
    python backtest/run_backtest.py --walk-forward
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from backtest.engine import Backtester
from backtest.strategies import STRATEGIES
from backtest.report import plot_results


def run_single(strategy_key, symbols, start, end, capital=25000, output_dir="backtest_results"):
    strat = STRATEGIES[strategy_key]
    print(f"\n{'='*62}")
    print(f"  BACKTESTING: {strat['name']}")
    print(f"  {strat['desc']}")
    print(f"{'='*62}")

    bt = Backtester(
        capital        = capital,
        max_positions  = config.MAX_POSITIONS,
        max_pos_pct    = config.MAX_POSITION_PCT,
        stop_loss_pct  = config.STOP_LOSS_PCT,
        take_profit_pct= config.TAKE_PROFIT_PCT,
        fee_pct        = config.BROKERAGE_FEE_PCT,
        rebalance_freq = strat["freq"],
    )
    bt.load_data(symbols, start=start, end=end)
    results = bt.run(strat["func"], benchmark_symbol=symbols[0])
    bt.print_report(results)

    print(f"\nGenerating visual report...")
    os.makedirs(output_dir, exist_ok=True)
    chart_path, html_path = plot_results(results, strat["name"], output_dir)
    print(f"\nOpen in browser: {html_path}")

    return results


def compare_strategies(symbols, start, end, capital=25000):
    """Run all strategies and compare side by side."""
    print(f"\n{'='*62}")
    print(f"  STRATEGY COMPARISON")
    print(f"  Universe: {symbols}")
    print(f"  Period:   {start} to {end}")
    print(f"  Capital:  ฿{capital:,}")
    print(f"{'='*62}")

    summary = []

    for key, strat in STRATEGIES.items():
        print(f"\n  Running: {strat['name']}...")
        try:
            bt = Backtester(
                capital        = capital,
                max_positions  = config.MAX_POSITIONS,
                max_pos_pct    = config.MAX_POSITION_PCT,
                stop_loss_pct  = config.STOP_LOSS_PCT,
                take_profit_pct= config.TAKE_PROFIT_PCT,
                fee_pct        = config.BROKERAGE_FEE_PCT,
                rebalance_freq = strat["freq"],
            )
            bt.load_data(symbols, start=start, end=end)
            results = bt.run(strat["func"], benchmark_symbol=symbols[0])
            m = results["metrics"]
            summary.append({
                "Strategy":     strat["name"],
                "CAGR %":       m["cagr_pct"],
                "Sharpe":       m["sharpe_ratio"],
                "Max DD %":     m["max_drawdown_pct"],
                "Win Rate %":   m["win_rate_pct"],
                "Trades":       m["total_trades"],
                "PF":           m["profit_factor"],
                "End ฿":        m["ending_capital"],
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            summary.append({"Strategy": strat["name"], "CAGR %": "ERROR"})

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"  {'STRATEGY':<35} {'CAGR':>7} {'SHARPE':>7} {'MAX DD':>8} {'WIN%':>7} {'TRADES':>7} {'END ฿':>12}")
    print(f"  {'-'*85}")
    for row in sorted(summary, key=lambda x: x.get("CAGR %", -999), reverse=True):
        if row.get("CAGR %") == "ERROR":
            print(f"  {row['Strategy']:<35}  ERROR")
            continue
        print(f"  {row['Strategy']:<35} "
              f"{row['CAGR %']:>+7.1f}% "
              f"{row['Sharpe']:>7.2f} "
              f"{row['Max DD %']:>7.1f}% "
              f"{row['Win Rate %']:>6.1f}% "
              f"{row['Trades']:>7} "
              f"฿{row['End ฿']:>10,.0f}")
    print(f"{'='*90}")


def walk_forward_validation(strategy_key, symbols, start, end, n_splits=4, capital=25000):
    """
    Walk-forward analysis: train on expanding window, test out-of-sample.
    Most realistic performance estimate.
    """
    import pandas as pd
    from datetime import datetime, timedelta

    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Strategy: {STRATEGIES[strategy_key]['name']}")
    print(f"  {n_splits} time splits")
    print(f"{'='*62}")

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    total_days = (end_dt - start_dt).days
    split_size = total_days // n_splits

    wf_results = []

    for i in range(n_splits):
        split_start = start_dt + pd.Timedelta(days=i * split_size)
        split_end   = split_start + pd.Timedelta(days=split_size)
        if split_end > end_dt:
            split_end = end_dt

        period_label = f"{split_start.date()} to {split_end.date()}"
        print(f"\n  Period {i+1}/{n_splits}: {period_label}")

        try:
            strat = STRATEGIES[strategy_key]
            bt = Backtester(
                capital        = capital,
                max_positions  = config.MAX_POSITIONS,
                max_pos_pct    = config.MAX_POSITION_PCT,
                stop_loss_pct  = config.STOP_LOSS_PCT,
                take_profit_pct= config.TAKE_PROFIT_PCT,
                fee_pct        = config.BROKERAGE_FEE_PCT,
                rebalance_freq = strat["freq"],
            )
            bt.load_data(symbols, start=str(split_start.date()), end=str(split_end.date()))
            results = bt.run(strat["func"])
            m = results["metrics"]
            wf_results.append({
                "Period":    period_label,
                "CAGR":      m["cagr_pct"],
                "Sharpe":    m["sharpe_ratio"],
                "Max DD":    m["max_drawdown_pct"],
                "Win Rate":  m["win_rate_pct"],
                "Trades":    m["total_trades"],
            })
            print(f"    CAGR={m['cagr_pct']:+.1f}% | Sharpe={m['sharpe_ratio']:.2f} | DD={m['max_drawdown_pct']:.1f}%")
        except Exception as e:
            print(f"    ERROR: {e}")

    if wf_results:
        import numpy as np
        cagrs   = [r["CAGR"] for r in wf_results]
        sharpes = [r["Sharpe"] for r in wf_results]
        print(f"\n  WALK-FORWARD SUMMARY")
        print(f"  Avg CAGR:      {np.mean(cagrs):+.1f}% (std: {np.std(cagrs):.1f}%)")
        print(f"  Avg Sharpe:    {np.mean(sharpes):.2f}")
        print(f"  % Positive:    {sum(1 for c in cagrs if c > 0)/len(cagrs)*100:.0f}%")
        print(f"\n  Note: High variance across periods = strategy may be overfit.")
        print(f"  Consistent results across periods = robust strategy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SET AI Backtester")
    parser.add_argument("--strategy",   default="multi_factor",
                       choices=list(STRATEGIES.keys()),
                       help="Strategy to backtest")
    parser.add_argument("--compare",    action="store_true",
                       help="Compare all strategies side by side")
    parser.add_argument("--walk-forward", action="store_true",
                       help="Run walk-forward validation")
    parser.add_argument("--start",      default="2020-01-01",
                       help="Start date YYYY-MM-DD")
    parser.add_argument("--end",        default="2024-12-31",
                       help="End date YYYY-MM-DD")
    parser.add_argument("--capital",    type=float, default=25000,
                       help="Starting capital in THB")
    parser.add_argument("--universe",  choices=["test", "set50"], default="test",
                       help="Stock universe")
    parser.add_argument("--output",    default="backtest_results",
                       help="Output directory for reports")
    args = parser.parse_args()

    symbols = config.SET50_UNIVERSE if args.universe == "set50" else config.TEST_UNIVERSE

    if args.compare:
        compare_strategies(symbols, args.start, args.end, args.capital)
    elif args.walk_forward:
        walk_forward_validation(args.strategy, symbols, args.start, args.end, capital=args.capital)
    else:
        run_single(args.strategy, symbols, args.start, args.end, args.capital, args.output)
