"""
portfolio/paper_trader.py
Paper Trading Engine for SET stocks.

Simulates real trading with:
- Virtual 25,000 THB capital
- Realistic brokerage fees
- Stop-loss and take-profit automation
- Full trade history and P&L tracking
- Portfolio state persistence (survives restarts)
"""

import json
import os
from datetime import datetime, date, timedelta
from typing import Optional
import sys
import yfinance as yf
sys.path.append("..")
import config
from data.fetcher import get_current_price
from logs.logger import Logger

os.makedirs("data", exist_ok=True)


def _get_last_dividend_date(symbol: str):
    """Return the most recent ex-dividend date from yfinance, or None."""
    try:
        tk = yf.Ticker(f"{symbol}.BK" if not symbol.endswith(".BK") else symbol)
        divs = tk.dividends
        if divs is not None and not divs.empty:
            return divs.index[-1].date()
    except Exception:
        pass
    return None


def is_near_xd_date(symbol: str, days_window: int = 5) -> bool:
    """
    Return True if the most recent or upcoming ex-dividend date is within
    days_window calendar days of today.
    Fail-safe: returns False when no dividend data is available so that
    stop-losses are NOT suppressed spuriously.
    """
    try:
        tk = yf.Ticker(f"{symbol}.BK" if not symbol.endswith(".BK") else symbol)
        today        = date.today()
        window_start = today - timedelta(days=days_window)
        window_end   = today + timedelta(days=days_window)

        # Check historical dividends for a recent XD date
        divs = tk.dividends
        if divs is not None and not divs.empty:
            last_xd = divs.index[-1].date()
            if window_start <= last_xd <= window_end:
                print(f"  [XD GUARD] {symbol}: last XD date {last_xd} is within {days_window} days")
                return True

        # Check forward calendar for an upcoming XD date
        try:
            cal = tk.calendar
            if cal is not None and isinstance(cal, dict):
                ex_date = cal.get("Ex-Dividend Date") or cal.get("ex_dividend_date")
                if ex_date:
                    if hasattr(ex_date, "date"):
                        ex_date = ex_date.date()
                    elif isinstance(ex_date, str):
                        ex_date = datetime.strptime(ex_date[:10], "%Y-%m-%d").date()
                    if window_start <= ex_date <= window_end:
                        print(f"  [XD GUARD] {symbol}: upcoming XD date {ex_date} within {days_window} days")
                        return True
        except Exception:
            pass

        return False

    except Exception as e:
        print(f"  [XD GUARD] Could not check dividend data for {symbol}: {e}")
        return False  # fail-safe: don't suppress stops on data errors


class PaperTrader:
    """
    Simulates a live brokerage account with paper money.
    Tracks positions, executes virtual trades, and logs everything.
    """

    def __init__(self):
        self.portfolio = self._load_portfolio()
        self.trades    = self._load_trades()
        self.logger    = Logger()

    # ── Persistence ──────────────────────────────────────────

    def _load_portfolio(self) -> dict:
        if os.path.exists(config.PORTFOLIO_FILE):
            with open(config.PORTFOLIO_FILE) as f:
                return json.load(f)
        # Initialize fresh portfolio
        return {
            "cash":         config.PAPER_CAPITAL_THB,
            "starting_capital": config.PAPER_CAPITAL_THB,
            "positions":    {},   # symbol -> {shares, avg_cost, ...}
            "created_at":   datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load_trades(self) -> list:
        if os.path.exists(config.TRADES_FILE):
            with open(config.TRADES_FILE) as f:
                return json.load(f)
        return []

    def _save(self):
        self.portfolio["last_updated"] = datetime.now().isoformat()
        with open(config.PORTFOLIO_FILE, "w") as f:
            json.dump(self.portfolio, f, indent=2)
        with open(config.TRADES_FILE, "w") as f:
            json.dump(self.trades, f, indent=2)

    # ── Portfolio Metrics ────────────────────────────────────

    def get_portfolio_value(self, live_prices: dict = None) -> dict:
        """
        Calculate current portfolio value.
        Pass live_prices dict {symbol: price} or it fetches live.
        """
        positions_value = 0.0
        positions_detail = {}

        for symbol, pos in self.portfolio["positions"].items():
            price = None
            if live_prices and symbol in live_prices:
                price = live_prices[symbol]
            else:
                price = get_current_price(symbol) or pos["avg_cost"]

            current_value = pos["shares"] * price
            cost_basis    = pos["shares"] * pos["avg_cost"]
            unrealized_pl = current_value - cost_basis
            unrealized_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0

            positions_detail[symbol] = {
                **pos,
                "current_price":   round(price, 2),
                "current_value":   round(current_value, 2),
                "cost_basis":      round(cost_basis, 2),
                "unrealized_pl":   round(unrealized_pl, 2),
                "unrealized_pct":  round(unrealized_pct, 2),
                "stop_loss_price": round(pos["avg_cost"] * (1 - pos.get("stop_loss_pct", config.STOP_LOSS_PCT)), 2),
                "target_price":    round(pos["avg_cost"] * (1 + config.TAKE_PROFIT_PCT), 2),
            }
            positions_value += current_value

        total_value     = self.portfolio["cash"] + positions_value
        starting        = self.portfolio["starting_capital"]
        total_return    = total_value - starting
        total_return_pct = (total_return / starting * 100)

        realized_pl = sum(t["realized_pl"] for t in self.trades if t["type"] == "SELL")

        return {
            "cash":              round(self.portfolio["cash"], 2),
            "positions_value":   round(positions_value, 2),
            "total_value":       round(total_value, 2),
            "starting_capital":  starting,
            "total_return_thb":  round(total_return, 2),
            "total_return_pct":  round(total_return_pct, 2),
            "realized_pl":       round(realized_pl, 2),
            "unrealized_pl":     round(positions_value - sum(
                                      p["shares"] * p["avg_cost"]
                                      for p in self.portfolio["positions"].values()
                                  ), 2),
            "num_positions":     len(self.portfolio["positions"]),
            "positions":         positions_detail,
        }

    # ── Order Execution ──────────────────────────────────────

    def buy(self,
            symbol:          str,
            price:           float,
            position_size_pct: float,
            reason:          str,
            stop_loss_pct:   float = None,
            ai_conviction:   int = 5) -> dict:
        """
        Execute a virtual BUY order.

        Args:
            symbol:           SET ticker (without .BK)
            price:            Current market price in THB
            position_size_pct: % of portfolio to allocate (e.g., 8 = 8%)
            reason:           Why we're buying (for the log)
            stop_loss_pct:    Stop-loss distance (default from config)
            ai_conviction:    Claude's conviction score 1-10

        Returns:
            Trade record dict, or error dict
        """
        # Validations
        if symbol in self.portfolio["positions"]:
            return {"error": f"Already holding {symbol}. Use add_to_position() to add."}

        if price <= 0:
            return {"error": f"Invalid price {price} for {symbol}. Refusing to buy."}

        if price < 0.50:
            return {"error": f"Price {price} THB is below minimum. Stock may be delisted."}

        if len(self.portfolio["positions"]) >= config.MAX_POSITIONS:
            return {"error": f"Max positions ({config.MAX_POSITIONS}) reached."}

        portfolio_val = self.portfolio["cash"] + sum(
            p["shares"] * p["avg_cost"] for p in self.portfolio["positions"].values()
        )

        # Calculate position size
        pct = min(position_size_pct / 100, config.MAX_POSITION_PCT)
        pct = max(pct, config.MIN_POSITION_PCT)
        capital_to_deploy = portfolio_val * pct

        if capital_to_deploy > self.portfolio["cash"]:
            capital_to_deploy = self.portfolio["cash"] * 0.90  # use up to 90% of remaining cash

        if capital_to_deploy < 1000:  # minimum order ~1000 THB
            return {"error": f"Insufficient cash. Have {self.portfolio['cash']:.0f} THB, need at least 1,000 THB."}

        # Brokerage fee
        fee = capital_to_deploy * config.BROKERAGE_FEE_PCT
        shares = int((capital_to_deploy - fee) / price)

        if shares < 1:
            return {"error": "Position too small to buy even 1 share."}

        total_cost = (shares * price) + fee

        # Execute the trade
        self.portfolio["cash"] -= total_cost
        self.portfolio["positions"][symbol] = {
            "shares":          shares,
            "avg_cost":        round(price, 2),
            "total_cost":      round(total_cost, 2),
            "position_pct":    round(pct * 100, 1),
            "stop_loss_pct":   stop_loss_pct or config.STOP_LOSS_PCT,
            "ai_conviction":   ai_conviction,
            "entry_reason":    reason,
            "entry_date":      datetime.now().isoformat(),
        }

        trade = {
            "id":           len(self.trades) + 1,
            "type":         "BUY",
            "symbol":       symbol,
            "shares":       shares,
            "price":        round(price, 2),
            "value":        round(shares * price, 2),
            "fee":          round(fee, 2),
            "total_cost":   round(total_cost, 2),
            "position_pct": round(pct * 100, 1),
            "reason":       reason,
            "conviction":   ai_conviction,
            "timestamp":    datetime.now().isoformat(),
            "realized_pl":  0,
        }
        self.trades.append(trade)
        self._save()

        _approx_value = self.portfolio["cash"] + sum(
            p["shares"] * p["avg_cost"] for p in self.portfolio["positions"].values()
        )
        self.logger.trade_executed(
            symbol=symbol,
            action="BUY",
            shares=shares,
            price=price,
            cost=total_cost,
            fee=fee,
            stop_loss=price * (1 - (stop_loss_pct or config.STOP_LOSS_PCT)),
            reason=reason,
            portfolio_value_after=_approx_value,
        )
        self._log_snapshot()

        print(f"\n  BUY  {symbol}: {shares} shares @ {price:.2f} THB")
        print(f"       Cost: {total_cost:.0f} THB (fee: {fee:.0f} THB)")
        print(f"       Stop-loss: {price * (1 - (stop_loss_pct or config.STOP_LOSS_PCT)):.2f} THB")
        print(f"       Reason: {reason[:80]}")

        return trade

    def sell(self,
             symbol: str,
             price:  float,
             reason: str) -> dict:
        """
        Execute a virtual SELL order and realize P&L.
        """
        if symbol not in self.portfolio["positions"]:
            return {"error": f"No position in {symbol}."}

        pos   = self.portfolio["positions"][symbol]
        shares = pos["shares"]

        # Brokerage fee on sale
        gross_proceeds = shares * price
        fee            = gross_proceeds * config.BROKERAGE_FEE_PCT
        net_proceeds   = gross_proceeds - fee

        realized_pl    = net_proceeds - pos["total_cost"]
        realized_pct   = (realized_pl / pos["total_cost"] * 100)

        # Update portfolio
        self.portfolio["cash"] += net_proceeds
        del self.portfolio["positions"][symbol]

        trade = {
            "id":             len(self.trades) + 1,
            "type":           "SELL",
            "symbol":         symbol,
            "shares":         shares,
            "price":          round(price, 2),
            "gross_proceeds": round(gross_proceeds, 2),
            "fee":            round(fee, 2),
            "net_proceeds":   round(net_proceeds, 2),
            "avg_cost":       pos["avg_cost"],
            "realized_pl":    round(realized_pl, 2),
            "realized_pct":   round(realized_pct, 2),
            "holding_days":   (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days,
            "reason":         reason,
            "timestamp":      datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._save()

        _approx_value = self.portfolio["cash"] + sum(
            p["shares"] * p["avg_cost"] for p in self.portfolio["positions"].values()
        )
        self.logger.trade_executed(
            symbol=symbol,
            action="SELL",
            shares=shares,
            price=price,
            cost=pos["total_cost"],
            fee=fee,
            stop_loss=0.0,
            reason=reason,
            portfolio_value_after=_approx_value,
        )
        self._log_snapshot()

        emoji = "" if realized_pl >= 0 else ""
        print(f"\n  SELL {symbol}: {shares} shares @ {price:.2f} THB")
        print(f"       P&L: {realized_pl:+.0f} THB ({realized_pct:+.1f}%) {emoji}")
        print(f"       Reason: {reason[:80]}")

        return trade

    # ── Risk Checks ──────────────────────────────────────────

    def check_stop_losses(self, live_prices: dict = None) -> list:
        """
        Check all positions against their stop-loss levels.
        Returns list of stop-loss triggers (caller handles the sells).
        Take-profits are handled automatically inside check_take_profits().
        Stop-losses near ex-dividend dates are suppressed (XD guard).
        """
        triggered     = []
        xd_suppressed = 0
        # Build a price cache so check_take_profits() can reuse fetched prices
        price_cache   = dict(live_prices) if live_prices else {}

        for symbol, pos in self.portfolio["positions"].items():
            if symbol in price_cache:
                price = price_cache[symbol]
            else:
                price = get_current_price(symbol)
                if price is not None:
                    price_cache[symbol] = price

            if price is None:
                continue

            stop_price   = pos["avg_cost"] * (1 - pos.get("stop_loss_pct", config.STOP_LOSS_PCT))
            distance_pct = (price / pos["avg_cost"] - 1) * 100

            if price <= stop_price:
                if is_near_xd_date(symbol):
                    xd_suppressed += 1
                    xd_date = _get_last_dividend_date(symbol)
                    self.logger.xd_guard_triggered(
                        symbol=symbol,
                        current_price=price,
                        stop_price=stop_price,
                        distance_pct=distance_pct,
                        xd_date=str(xd_date) if xd_date else "unknown",
                    )
                    print(f"  ⚠ XD GUARD: {symbol} stop suppressed - "
                          f"possible ex-dividend drop. Check manually.")
                    continue

                triggered.append({
                    "symbol": symbol, "price": price,
                    "reason": f"STOP-LOSS triggered ({distance_pct:.1f}%)",
                    "type":   "stop_loss",
                })

        # Run take-profit check (auto-sells, reuses cached prices)
        tp_count = self.check_take_profits(price_cache)

        print(f"  Stops triggered: {len(triggered)} | XD suppressed: {xd_suppressed} | "
              f"Take profits: {tp_count}")
        return triggered

    def check_take_profits(self, live_prices: dict = None) -> int:
        """
        Check all positions against the take-profit target and auto-close any that hit.
        Called automatically at the end of check_stop_losses(); can also be called standalone.
        Returns count of take-profits executed.
        """
        tp_count = 0

        for symbol in list(self.portfolio["positions"]):
            pos = self.portfolio["positions"].get(symbol)
            if pos is None:
                continue  # already sold earlier in this loop

            if live_prices and symbol in live_prices:
                price = live_prices[symbol]
            else:
                price = get_current_price(symbol)

            if price is None:
                continue

            take_price = pos["avg_cost"] * (1 + config.TAKE_PROFIT_PCT)
            if price < take_price:
                continue

            # Take-profit hit — auto-close the position
            pnl_pct_est  = (price / pos["avg_cost"] - 1) * 100
            holding_days = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
            sell_result  = self.sell(symbol, price,
                                     f"TAKE-PROFIT triggered (+{pnl_pct_est:.1f}%)")

            if "error" in sell_result:
                continue

            tp_count += 1
            pnl_thb  = sell_result.get("realized_pl", 0)
            pnl_pct  = sell_result.get("realized_pct", pnl_pct_est)
            self.logger.take_profit_triggered(
                symbol=symbol,
                entry_price=pos["avg_cost"],
                exit_price=price,
                pnl_thb=pnl_thb,
                pnl_pct=pnl_pct,
                holding_days=sell_result.get("holding_days", holding_days),
            )
            print(f"  🎯 TAKE PROFIT: {symbol} closed at +{pnl_pct:.1f}% gain. "
                  f"Profit: ฿{pnl_thb:.0f}")

        return tp_count

    def check_daily_loss_limit(self) -> bool:
        """Returns True if daily loss limit has been hit (trading should halt)."""
        today = datetime.now().date().isoformat()
        today_trades = [t for t in self.trades
                       if t["timestamp"][:10] == today and t["type"] == "SELL"]
        daily_loss = sum(t.get("realized_pl", 0) for t in today_trades)
        limit = self.portfolio["starting_capital"] * config.DAILY_LOSS_LIMIT
        return daily_loss < -limit

    # ── Snapshot logging ─────────────────────────────────────

    def _log_snapshot(self):
        """Write a PORTFOLIO_SNAPSHOT event using cost-basis prices (no live lookup)."""
        cash = self.portfolio["cash"]
        positions_value = sum(
            p["shares"] * p["avg_cost"]
            for p in self.portfolio["positions"].values()
        )
        total_value  = cash + positions_value
        starting     = self.portfolio["starting_capital"]
        return_pct   = (total_value - starting) / starting * 100 if starting else 0
        open_pos = [
            {"symbol": sym, "shares": pos["shares"], "avg_cost": pos["avg_cost"]}
            for sym, pos in self.portfolio["positions"].items()
        ]
        self.logger.portfolio_snapshot(
            cash=cash,
            positions_value=positions_value,
            total_value=total_value,
            total_return_pct=return_pct,
            open_positions=open_pos,
        )

    def log_snapshot(self):
        """Public wrapper — call this at the end of a screener run."""
        self._log_snapshot()

    # ── Reporting ────────────────────────────────────────────

    def print_portfolio(self):
        """Print a formatted portfolio summary to the console."""
        pv = self.get_portfolio_value()

        print("\n" + "=" * 62)
        print("  SET AI PAPER TRADER - PORTFOLIO SUMMARY")
        print("=" * 62)
        print(f"  Starting Capital : {pv['starting_capital']:>10,.0f} THB")
        print(f"  Cash Available   : {pv['cash']:>10,.0f} THB")
        print(f"  Positions Value  : {pv['positions_value']:>10,.0f} THB")
        print(f"  Total Value      : {pv['total_value']:>10,.0f} THB")
        print(f"  Total Return     : {pv['total_return_thb']:>+10,.0f} THB  ({pv['total_return_pct']:+.2f}%)")
        print(f"  Realized P&L     : {pv['realized_pl']:>+10,.0f} THB")
        print(f"  Unrealized P&L   : {pv['unrealized_pl']:>+10,.0f} THB")

        if pv["positions"]:
            print(f"\n  OPEN POSITIONS ({pv['num_positions']})")
            print("  " + "-" * 58)
            print(f"  {'SYMBOL':<8} {'SHARES':>6} {'AVG COST':>10} {'PRICE':>10} {'P&L':>10} {'%':>7}")
            print("  " + "-" * 58)
            for sym, pos in pv["positions"].items():
                pl_str = f"{pos['unrealized_pl']:+,.0f}"
                pct_str = f"{pos['unrealized_pct']:+.1f}%"
                flag = "" if pos["unrealized_pl"] >= 0 else ""
                print(f"  {sym:<8} {pos['shares']:>6} {pos['avg_cost']:>10.2f} "
                      f"{pos['current_price']:>10.2f} {pl_str:>10} {pct_str:>7} {flag}")

        print(f"\n  Total Trades Executed: {len(self.trades)}")
        sell_trades = [t for t in self.trades if t["type"] == "SELL"]
        if sell_trades:
            wins = [t for t in sell_trades if t["realized_pl"] > 0]
            print(f"  Win Rate: {len(wins)}/{len(sell_trades)} ({len(wins)/len(sell_trades)*100:.0f}%)")
        print("=" * 62)
