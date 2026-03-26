"""
backtest/engine.py

Historical simulation engine for the SET AI Trading System.
Walks through time bar-by-bar to avoid lookahead bias.

Key features:
  - Point-in-time data: signals only use data available on that day
  - Realistic costs: brokerage fees on every trade
  - Multiple strategies: can backtest any signal combination
  - Full trade log + daily equity curve
  - No future data leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, List, Dict, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class Trade:
    """Represents a single completed trade."""
    def __init__(self, symbol, entry_date, entry_price, shares, reason):
        self.symbol      = symbol
        self.entry_date  = entry_date
        self.entry_price = entry_price
        self.shares      = shares
        self.entry_fee   = shares * entry_price * config.BROKERAGE_FEE_PCT
        self.reason      = reason

        self.exit_date   = None
        self.exit_price  = None
        self.exit_reason = None
        self.exit_fee    = 0

    @property
    def is_open(self):
        return self.exit_date is None

    @property
    def cost_basis(self):
        return self.shares * self.entry_price + self.entry_fee

    def close(self, date, price, reason):
        self.exit_date   = date
        self.exit_price  = price
        self.exit_reason = reason
        self.exit_fee    = self.shares * price * config.BROKERAGE_FEE_PCT

    @property
    def realized_pnl(self):
        if not self.exit_price:
            return 0
        gross = self.shares * (self.exit_price - self.entry_price)
        return gross - self.entry_fee - self.exit_fee

    @property
    def realized_pct(self):
        return (self.realized_pnl / self.cost_basis) * 100 if self.cost_basis else 0

    @property
    def holding_days(self):
        if self.exit_date and self.entry_date:
            return (self.exit_date - self.entry_date).days
        return None

    def to_dict(self):
        return {
            "symbol":       self.symbol,
            "entry_date":   self.entry_date.strftime("%Y-%m-%d"),
            "entry_price":  round(self.entry_price, 2),
            "exit_date":    self.exit_date.strftime("%Y-%m-%d") if self.exit_date else None,
            "exit_price":   round(self.exit_price, 2) if self.exit_price else None,
            "shares":       self.shares,
            "cost_basis":   round(self.cost_basis, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "realized_pct": round(self.realized_pct, 2),
            "holding_days": self.holding_days,
            "reason":       self.reason,
            "exit_reason":  self.exit_reason,
        }


class Position:
    """Tracks an open position in the portfolio."""
    def __init__(self, trade: Trade, stop_loss_pct: float, take_profit_pct: float):
        self.trade          = trade
        self.stop_price     = trade.entry_price * (1 - stop_loss_pct)
        self.take_profit_price = trade.entry_price * (1 + take_profit_pct)

    @property
    def symbol(self): return self.trade.symbol

    def current_value(self, price):
        return self.trade.shares * price

    def unrealized_pnl(self, price):
        return self.trade.shares * (price - self.trade.entry_price)

    def check_exit(self, high, low, close):
        """Check if stop-loss or take-profit was hit. Returns (exit_price, reason) or None."""
        # Stop-loss hit
        if low <= self.stop_price:
            return (self.stop_price, "STOP_LOSS")
        # Take-profit hit
        if high >= self.take_profit_price:
            return (self.take_profit_price, "TAKE_PROFIT")
        return None


class Backtester:
    """
    Walk-forward backtester for SET trading strategies.

    Usage:
        bt = Backtester(capital=25000)
        bt.load_data(["PTT", "ADVANC", "KBANK"], start="2020-01-01", end="2024-12-31")
        results = bt.run(strategy_func)
        bt.print_report(results)
    """

    def __init__(
        self,
        capital:         float = 25_000.0,
        max_positions:   int   = 5,
        max_pos_pct:     float = 0.20,
        stop_loss_pct:   float = 0.08,
        take_profit_pct: float = 0.20,
        fee_pct:         float = 0.0025,
        rebalance_freq:  str   = "W",     # W=weekly, M=monthly, D=daily
        max_hold_days:   int   = None,    # Force exit after N days (None = disabled)
    ):
        self.max_hold_days    = max_hold_days
        self.initial_capital  = capital
        self.max_positions    = max_positions
        self.max_pos_pct      = max_pos_pct
        self.stop_loss_pct    = stop_loss_pct
        self.take_profit_pct  = take_profit_pct
        self.fee_pct          = fee_pct
        self.rebalance_freq   = rebalance_freq

        self.price_data: Dict[str, pd.DataFrame] = {}
        self.all_dates:  List[datetime]           = []

    def load_data(self, symbols: List[str], start: str = "2020-01-01", end: str = None):
        """
        Fetch and cache historical OHLCV data for all symbols.
        Uses yfinance with .BK suffix for SET.
        """
        import yfinance as yf

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        print(f"\nLoading historical data: {start} to {end}")
        print(f"Universe: {len(symbols)} stocks\n")

        all_date_set = set()

        for sym in symbols:
            ticker = f"{sym}.BK"
            try:
                df = yf.download(ticker, start=start, end=end,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    print(f"  {sym}: no data")
                    continue

                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.index = pd.to_datetime(df.index)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

                # Compute rolling technical indicators (point-in-time safe)
                df["MA20"]  = df["Close"].rolling(20).mean()
                df["MA50"]  = df["Close"].rolling(50).mean()
                df["MA200"] = df["Close"].rolling(200).mean()
                df["RSI"]   = self._compute_rsi(df["Close"], 14)
                df["ATR"]   = self._compute_atr(df, 14)

                # Momentum signals
                df["Ret1M"]  = df["Close"].pct_change(21)   # 1-month return
                df["Ret3M"]  = df["Close"].pct_change(63)   # 3-month return
                df["Ret12M"] = df["Close"].pct_change(252)  # 12-month return

                # Volume signal
                df["VolMA20"] = df["Volume"].rolling(20).mean()
                df["VolRatio"] = df["Volume"] / df["VolMA20"]

                self.price_data[sym] = df
                all_date_set.update(df.index.tolist())
                print(f"  {sym}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

            except Exception as e:
                print(f"  {sym}: error - {e}")

        self.all_dates = sorted(all_date_set)
        print(f"\nData loaded: {len(self.price_data)} stocks, {len(self.all_dates)} trading days")
        return self

    def _compute_rsi(self, closes: pd.Series, period: int = 14) -> pd.Series:
        delta  = closes.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["High"] - df["Low"]
        hpc = (df["High"] - df["Close"].shift()).abs()
        lpc = (df["Low"]  - df["Close"].shift()).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _get_row(self, symbol: str, date) -> Optional[pd.Series]:
        """Get OHLCV row for a symbol on a specific date. Returns None if no data."""
        df = self.price_data.get(symbol)
        if df is None or date not in df.index:
            return None
        return df.loc[date]

    def run(self, strategy_func: Callable, benchmark_symbol: str = "PTT") -> dict:
        """
        Run the backtest.

        strategy_func signature:
            (symbol, date, price_data_up_to_date, portfolio_state) -> float or None
            Returns: signal score 0-100, or None to skip this stock

        Returns dict with equity_curve, trades, metrics.
        """
        cash      = self.initial_capital
        positions: Dict[str, Position] = {}
        trades:    List[Trade]         = []
        equity_curve = []

        # Determine rebalance dates
        all_dates_pd = pd.DatetimeIndex(self.all_dates)
        if self.rebalance_freq == "W":
            rebalance_dates = set(all_dates_pd[all_dates_pd.day_of_week == 0])  # Mondays
        elif self.rebalance_freq == "M":
            rebalance_dates = set(all_dates_pd[all_dates_pd.is_month_start])
        else:  # Daily
            rebalance_dates = set(all_dates_pd)

        print(f"\nRunning backtest ({self.rebalance_freq} rebalance)...")
        print(f"Period: {self.all_dates[0].date()} to {self.all_dates[-1].date()}")
        print(f"Starting capital: {self.initial_capital:,.0f} THB\n")

        for date in self.all_dates:
            # --- 1. Mark-to-market open positions ---
            pos_value = 0
            for sym, pos in list(positions.items()):
                row = self._get_row(sym, date)
                if row is None:
                    continue

                # Check stop-loss / take-profit / max hold
                exit_signal = pos.check_exit(row["High"], row["Low"], row["Close"])
                max_hold_hit = (
                    self.max_hold_days is not None and
                    (date - pos.trade.entry_date).days >= self.max_hold_days
                )
                if exit_signal:
                    exit_price, exit_reason = exit_signal
                    pos.trade.close(date, exit_price, exit_reason)
                    proceeds = pos.trade.shares * exit_price * (1 - self.fee_pct)
                    cash    += proceeds
                    trades.append(pos.trade)
                    del positions[sym]
                elif max_hold_hit:
                    pos.trade.close(date, row["Close"], "MAX_HOLD")
                    proceeds = pos.trade.shares * row["Close"] * (1 - self.fee_pct)
                    cash    += proceeds
                    trades.append(pos.trade)
                    del positions[sym]
                else:
                    pos_value += pos.current_value(row["Close"])

            portfolio_value = cash + pos_value
            equity_curve.append({
                "date":       date,
                "cash":       cash,
                "pos_value":  pos_value,
                "total":      portfolio_value,
                "n_positions": len(positions),
            })

            # --- 2. Rebalance: scan for new signals ---
            if date not in rebalance_dates:
                continue

            # Build portfolio state snapshot
            port_state = {
                "cash":           cash,
                "positions":      list(positions.keys()),
                "portfolio_value": portfolio_value,
                "n_positions":    len(positions),
            }

            # Score all stocks using only data up to this date
            signals = []
            for sym in self.price_data:
                if sym in positions:
                    continue  # already holding

                df = self.price_data[sym]
                df_to_date = df[df.index <= date]

                if len(df_to_date) < 50:
                    continue  # not enough history

                try:
                    score = strategy_func(sym, date, df_to_date, port_state)
                    if score is not None and score > 0:
                        last_row = df_to_date.iloc[-1]
                        signals.append({
                            "symbol": sym,
                            "score":  score,
                            "price":  last_row["Close"],
                            "row":    last_row,
                        })
                except Exception:
                    continue

            # Sort by signal strength, take top N
            signals.sort(key=lambda x: x["score"], reverse=True)
            slots_available = self.max_positions - len(positions)

            for sig in signals[:slots_available]:
                sym   = sig["symbol"]
                price = sig["price"]

                if price <= 0 or sym in positions:
                    continue

                # Position sizing
                alloc   = portfolio_value * min(self.max_pos_pct, 1 / self.max_positions)
                alloc   = min(alloc, cash * 0.95)
                shares  = int(alloc / (price * (1 + self.fee_pct)))

                if shares < 1 or alloc < 500:
                    continue

                cost  = shares * price * (1 + self.fee_pct)
                cash -= cost

                trade = Trade(sym, date, price, shares, f"Signal={sig['score']:.1f}")
                pos   = Position(trade, self.stop_loss_pct, self.take_profit_pct)
                positions[sym] = pos

        # --- Close all remaining positions at end ---
        last_date = self.all_dates[-1]
        for sym, pos in list(positions.items()):
            row = self._get_row(sym, last_date)
            if row is not None:
                exit_price = row["Close"]
                pos.trade.close(last_date, exit_price, "END_OF_BACKTEST")
                proceeds = pos.trade.shares * exit_price * (1 - self.fee_pct)
                cash    += proceeds
                trades.append(pos.trade)

        equity_df = pd.DataFrame(equity_curve).set_index("date")

        # Compute benchmark
        benchmark = self._compute_benchmark(benchmark_symbol)

        # Compute performance metrics
        metrics = self._compute_metrics(equity_df, trades, benchmark)

        print(f"Backtest complete: {len(trades)} trades executed")
        return {
            "equity_curve": equity_df,
            "trades":       trades,
            "metrics":      metrics,
            "benchmark":    benchmark,
        }

    def _compute_benchmark(self, symbol: str) -> Optional[pd.Series]:
        """Buy-and-hold benchmark for comparison."""
        if symbol not in self.price_data:
            return None
        prices = self.price_data[symbol]["Close"]
        return (prices / prices.iloc[0]) * self.initial_capital

    def _compute_metrics(self, equity_df: pd.DataFrame, trades: List[Trade], benchmark) -> dict:
        """Compute comprehensive performance statistics."""
        total     = equity_df["total"]
        returns   = total.pct_change().dropna()

        # Basic returns
        total_return_pct = (total.iloc[-1] / total.iloc[0] - 1) * 100
        n_years = (total.index[-1] - total.index[0]).days / 365.25
        cagr    = ((total.iloc[-1] / total.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Risk metrics
        vol         = returns.std() * np.sqrt(252) * 100
        daily_rf    = config.RISK_FREE_RATE / 252
        excess_ret  = returns - daily_rf
        sharpe      = (excess_ret.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Sortino ratio (downside deviation only)
        downside    = returns[returns < 0].std() * np.sqrt(252)
        sortino     = (returns.mean() * 252 - config.RISK_FREE_RATE) / downside if downside > 0 else 0

        # Max drawdown
        rolling_max = total.cummax()
        drawdown    = (total - rolling_max) / rolling_max * 100
        max_dd      = drawdown.min()

        # Calmar ratio
        calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0

        # Trade statistics
        closed = [t for t in trades if not t.is_open]
        wins   = [t for t in closed if t.realized_pnl > 0]
        losses = [t for t in closed if t.realized_pnl <= 0]

        win_rate      = len(wins) / len(closed) * 100 if closed else 0
        avg_win_pct   = np.mean([t.realized_pct for t in wins]) if wins else 0
        avg_loss_pct  = np.mean([t.realized_pct for t in losses]) if losses else 0
        profit_factor = (sum(t.realized_pnl for t in wins) /
                        abs(sum(t.realized_pnl for t in losses))) if losses else float('inf')
        avg_hold_days = np.mean([t.holding_days for t in closed if t.holding_days]) if closed else 0

        total_pnl = sum(t.realized_pnl for t in closed)

        # Benchmark comparison
        bench_return = None
        if benchmark is not None:
            bench_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100

        return {
            "total_return_pct":  round(total_return_pct, 2),
            "cagr_pct":          round(cagr, 2),
            "volatility_pct":    round(vol, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "sortino_ratio":     round(sortino, 3),
            "max_drawdown_pct":  round(max_dd, 2),
            "calmar_ratio":      round(calmar, 3),
            "total_trades":      len(closed),
            "win_rate_pct":      round(win_rate, 1),
            "avg_win_pct":       round(avg_win_pct, 2),
            "avg_loss_pct":      round(avg_loss_pct, 2),
            "profit_factor":     round(profit_factor, 2),
            "avg_holding_days":  round(avg_hold_days, 1),
            "total_pnl_thb":     round(total_pnl, 0),
            "starting_capital":  self.initial_capital,
            "ending_capital":    round(equity_df["total"].iloc[-1], 0),
            "benchmark_return":  round(bench_return, 2) if bench_return else None,
            "alpha":             round(total_return_pct - bench_return, 2) if bench_return else None,
        }

    def print_report(self, results: dict):
        """Print a formatted performance report."""
        m = results["metrics"]
        trades = results["trades"]

        print("\n" + "=" * 62)
        print("  BACKTEST PERFORMANCE REPORT")
        print("=" * 62)

        print(f"\n  RETURNS")
        print(f"  {'Starting Capital':<28}: {m['starting_capital']:>10,.0f} THB")
        print(f"  {'Ending Capital':<28}: {m['ending_capital']:>10,.0f} THB")
        print(f"  {'Total Return':<28}: {m['total_return_pct']:>+10.2f}%")
        print(f"  {'CAGR':<28}: {m['cagr_pct']:>+10.2f}%")
        if m.get("benchmark_return"):
            print(f"  {'Benchmark (Buy & Hold)':<28}: {m['benchmark_return']:>+10.2f}%")
            print(f"  {'Alpha vs Benchmark':<28}: {m['alpha']:>+10.2f}%")

        print(f"\n  RISK METRICS")
        print(f"  {'Annual Volatility':<28}: {m['volatility_pct']:>10.2f}%")
        print(f"  {'Sharpe Ratio':<28}: {m['sharpe_ratio']:>10.3f}")
        print(f"  {'Sortino Ratio':<28}: {m['sortino_ratio']:>10.3f}")
        print(f"  {'Max Drawdown':<28}: {m['max_drawdown_pct']:>10.2f}%")
        print(f"  {'Calmar Ratio':<28}: {m['calmar_ratio']:>10.3f}")

        print(f"\n  TRADE STATISTICS")
        print(f"  {'Total Trades':<28}: {m['total_trades']:>10}")
        print(f"  {'Win Rate':<28}: {m['win_rate_pct']:>10.1f}%")
        print(f"  {'Avg Win':<28}: {m['avg_win_pct']:>+10.2f}%")
        print(f"  {'Avg Loss':<28}: {m['avg_loss_pct']:>+10.2f}%")
        print(f"  {'Profit Factor':<28}: {m['profit_factor']:>10.2f}")
        print(f"  {'Avg Holding Period':<28}: {m['avg_holding_days']:>10.1f} days")
        print(f"  {'Total Realized P&L':<28}: {m['total_pnl_thb']:>+10,.0f} THB")

        # Grade the strategy
        grade, comment = self._grade_strategy(m)
        print(f"\n  STRATEGY GRADE: {grade}")
        print(f"  {comment}")

        # Show top/worst trades
        closed = [t for t in trades if not t.is_open and t.realized_pct is not None]
        if closed:
            print(f"\n  TOP 5 WINNING TRADES")
            for t in sorted(closed, key=lambda x: x.realized_pct, reverse=True)[:5]:
                print(f"    {t.symbol:<8} {t.entry_date.date()} -> {t.exit_date.date()} "
                      f"  {t.realized_pct:>+7.1f}%  ({t.holding_days}d)")
            print(f"\n  WORST 5 LOSING TRADES")
            for t in sorted(closed, key=lambda x: x.realized_pct)[:5]:
                print(f"    {t.symbol:<8} {t.entry_date.date()} -> {t.exit_date.date()} "
                      f"  {t.realized_pct:>+7.1f}%  ({t.holding_days}d)")

        print("=" * 62)

    def _grade_strategy(self, m: dict):
        sharpe = m["sharpe_ratio"]
        win    = m["win_rate_pct"]
        dd     = abs(m["max_drawdown_pct"])
        cagr   = m["cagr_pct"]

        if sharpe >= 1.5 and cagr >= 20 and dd <= 20:
            return "A+", "Exceptional. Institutional-grade performance."
        elif sharpe >= 1.0 and cagr >= 15 and dd <= 25:
            return "A",  "Strong. Consistent alpha with controlled risk."
        elif sharpe >= 0.7 and cagr >= 10:
            return "B",  "Good. Solid returns, acceptable risk."
        elif sharpe >= 0.5 and cagr >= 5:
            return "C",  "Mediocre. Needs improvement before live trading."
        elif cagr > 0:
            return "D",  "Weak. Barely positive. Significant work needed."
        else:
            return "F",  "Strategy loses money. Do not trade live."
