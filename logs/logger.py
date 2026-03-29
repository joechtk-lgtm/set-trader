"""
logs/logger.py
Structured JSONL logger for the SET AI Trader.

Every event is written as a single JSON object on its own line.
Load the entire log with: pd.read_json("logs/trading.jsonl", lines=True)
"""

import json
import os
import traceback as _tb
from datetime import datetime, timezone


class Logger:
    """
    Append-only JSONL event logger.

    Usage:
        logger = Logger()
        logger.screen_start(symbols=["PTT", "ADVANC"], use_ai=True)
    """

    def __init__(self, path: str = "logs/trading.jsonl", source: str = "auto"):
        self.path = path
        self.source = source          # "manual" or "auto"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ── Core writer ───────────────────────────────────────────────────────────

    def _write(self, event_type: str, **fields):
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "source":     fields.pop("source", self.source),
            **fields,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ── Screener events ───────────────────────────────────────────────────────

    def screen_start(self, universe: list, use_ai: bool):
        """Logged when the screener begins a new run."""
        self._write(
            "SCREEN_START",
            universe=universe,
            universe_size=len(universe),
            use_ai=use_ai,
        )

    def screen_result(
        self,
        symbol:          str,
        composite_score: float,
        quant_signal:    str,
        dcf_verdict:     str,
        price:           float,
    ):
        """Logged once per stock that passes the quantitative screen."""
        self._write(
            "SCREEN_RESULT",
            symbol=symbol,
            composite_score=round(composite_score, 2),
            quant_signal=quant_signal,
            dcf_verdict=dcf_verdict,
            price=price,
        )

    def ai_analysis(
        self,
        symbol:           str,
        action:           str,
        conviction:       int,
        position_size_pct: float,
        thesis:           str,
        bull_case:        str  = "",
        bear_case:        str  = "",
        key_risks:        list = None,
        red_flags:        list = None,
    ):
        """Logged when Claude returns a signal for a stock."""
        self._write(
            "AI_ANALYSIS",
            symbol=symbol,
            action=action,
            conviction=conviction,
            position_size_pct=position_size_pct,
            thesis=thesis,
            bull_case=bull_case,
            bear_case=bear_case,
            key_risks=key_risks or [],
            red_flags=red_flags or [],
        )

    # ── Trade events ──────────────────────────────────────────────────────────

    def trade_executed(
        self,
        symbol:                str,
        action:                str,
        shares:                int,
        price:                 float,
        cost:                  float,
        fee:                   float,
        stop_loss:             float,
        reason:                str,
        portfolio_value_after: float,
    ):
        """Logged on every completed buy or sell order."""
        self._write(
            "TRADE_EXECUTED",
            symbol=symbol,
            action=action,
            shares=shares,
            price=round(price, 2),
            cost=round(cost, 2),
            fee=round(fee, 2),
            stop_loss=round(stop_loss, 2),
            reason=reason,
            portfolio_value_after=round(portfolio_value_after, 2),
        )

    def take_profit_triggered(
        self,
        symbol:       str,
        entry_price:  float,
        exit_price:   float,
        pnl_thb:      float,
        pnl_pct:      float,
        holding_days: int,
    ):
        """Logged when a take-profit target fires and the position is auto-closed."""
        self._write(
            "TAKE_PROFIT_TRIGGERED",
            symbol=symbol,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            pnl_thb=round(pnl_thb, 2),
            pnl_pct=round(pnl_pct, 2),
            holding_days=holding_days,
        )

    def conviction_drop_sell(
        self,
        symbol:          str,
        old_conviction:  int,
        new_conviction:  int,
        exit_price:      float,
        pnl_thb:         float,
        pnl_pct:         float,
    ):
        """Logged when AI conviction drops below threshold and position is closed."""
        self._write(
            "CONVICTION_DROP_SELL",
            symbol=symbol,
            old_conviction=old_conviction,
            new_conviction=new_conviction,
            exit_price=round(exit_price, 2),
            pnl_thb=round(pnl_thb, 2),
            pnl_pct=round(pnl_pct, 2),
        )

    def xd_guard_triggered(
        self,
        symbol:       str,
        current_price: float,
        stop_price:   float,
        distance_pct: float,
        xd_date:      str,
    ):
        """Logged when an XD guard suppresses a stop-loss."""
        self._write(
            "XD_GUARD_TRIGGERED",
            symbol=symbol,
            current_price=round(current_price, 2),
            stop_price=round(stop_price, 2),
            distance_pct=round(distance_pct, 2),
            xd_date=xd_date,
        )

    def stop_loss_check(self, positions_checked: int, triggered: list):
        """Logged after scanning all open positions for stop-loss breaches."""
        self._write(
            "STOP_LOSS_CHECK",
            positions_checked=positions_checked,
            triggered=triggered,
        )

    def stop_loss_triggered(
        self,
        symbol:       str,
        entry_price:  float,
        exit_price:   float,
        pnl_thb:      float,
        pnl_pct:      float,
        holding_days: int,
    ):
        """Logged when a stop-loss or take-profit level fires."""
        self._write(
            "STOP_LOSS_TRIGGERED",
            symbol=symbol,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            pnl_thb=round(pnl_thb, 2),
            pnl_pct=round(pnl_pct, 2),
            holding_days=holding_days,
        )

    # ── Portfolio events ──────────────────────────────────────────────────────

    def portfolio_snapshot(
        self,
        cash:             float,
        positions_value:  float,
        total_value:      float,
        total_return_pct: float,
        open_positions:   list,
    ):
        """
        Point-in-time snapshot used to build the equity curve.
        open_positions: list of {symbol, shares, avg_cost} dicts.
        """
        self._write(
            "PORTFOLIO_SNAPSHOT",
            cash=round(cash, 2),
            positions_value=round(positions_value, 2),
            total_value=round(total_value, 2),
            total_return_pct=round(total_return_pct, 2),
            open_positions=open_positions,
        )

    # ── Error events ──────────────────────────────────────────────────────────

    def error(self, error_source: str, message: str, exc: Exception = None):
        """Logged whenever an exception is caught."""
        trace = _tb.format_exc() if exc else None
        self._write(
            "ERROR",
            error_source=error_source,
            message=str(message),
            traceback=trace,
        )
