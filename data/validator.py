"""
data/validator.py

Data quality validator for SET price data fetched via yfinance.
Catches bad prices before they reach the signal engine or trading engine.

Checks:
  1. Price sanity (zero, negative, or impossibly large)
  2. Single-day gap spikes (>25% move that is likely a data error)
  3. Corporate action detection (stock split / large dividend)
  4. Volume anomalies (zero volume or impossibly large)
  5. Stale data (last price is too old)
  6. Missing OHLC consistency (High < Low, Close outside High/Low)

Usage:
  from data.validator import validate_dataframe, ValidationResult
  result = validate_dataframe(df, symbol="PTT")
  if not result.ok:
      print(result.report())
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import numpy as np


# ── Thresholds ─────────────────────────────────────────────────────────────

# Single-day return beyond this is flagged as suspicious
SPIKE_THRESHOLD_PCT      = 25.0

# Single-day return beyond this is almost certainly a data error
HARD_SPIKE_THRESHOLD_PCT = 50.0

# If the last data point is older than this, data is stale
STALE_DAYS               = 5

# Volume: flag if a bar has zero volume on a supposedly trading day
MIN_VOLUME               = 0

# Price: anything below this is probably delisted or bad data
MIN_PRICE_THB            = 0.50

# Price: anything above this is suspicious for most SET stocks
MAX_PRICE_THB            = 50_000.0

# OHLC tolerance: allow tiny floating point differences
OHLC_TOLERANCE           = 0.01


@dataclass
class ValidationIssue:
    severity: str        # "ERROR" | "WARNING" | "INFO"
    code: str            # machine-readable code
    message: str         # human-readable description
    date: Optional[str] = None
    value: Optional[float] = None


@dataclass
class ValidationResult:
    symbol: str
    ok: bool                          # False if any ERROR exists
    issues: List[ValidationIssue] = field(default_factory=list)
    rows_checked: int = 0
    clean_pct: float = 100.0

    def errors(self):
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self):
        return [i for i in self.issues if i.severity == "WARNING"]

    def report(self) -> str:
        lines = [
            f"Validation: {self.symbol}  |  {'PASS' if self.ok else 'FAIL'}  |  "
            f"{self.rows_checked} bars  |  {self.clean_pct:.1f}% clean"
        ]
        for issue in self.issues:
            prefix = {"ERROR": "  ERROR  ", "WARNING": "  WARN   ", "INFO": "  INFO   "}[issue.severity]
            date_str = f" [{issue.date}]" if issue.date else ""
            val_str  = f" (value={issue.value:.4f})" if issue.value is not None else ""
            lines.append(f"{prefix}{issue.code}{date_str}{val_str}: {issue.message}")
        return "\n".join(lines)


def validate_dataframe(df: pd.DataFrame, symbol: str) -> ValidationResult:
    """
    Run all quality checks on a price DataFrame.
    DataFrame must have columns: Open, High, Low, Close, Volume.
    Index must be DatetimeIndex.
    """
    result = ValidationResult(symbol=symbol, ok=True, rows_checked=len(df))
    issues = result.issues

    if df.empty:
        issues.append(ValidationIssue("ERROR", "EMPTY_DATA", "DataFrame is empty"))
        result.ok = False
        return result

    # ── 1. Required columns ────────────────────────────────────────────────
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        issues.append(ValidationIssue("ERROR", "MISSING_COLUMNS",
            f"Missing columns: {missing}"))
        result.ok = False
        return result

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    open_ = df["Open"]
    vol   = df["Volume"]

    # ── 2. Stale data ──────────────────────────────────────────────────────
    last_date = pd.Timestamp(df.index[-1])
    now = pd.Timestamp.now(tz=last_date.tz)
    days_old = (now - last_date).days if last_date.tz else (pd.Timestamp.now() - last_date).days

    # Skip weekends in staleness check
    if days_old > STALE_DAYS:
        issues.append(ValidationIssue("WARNING", "STALE_DATA",
            f"Last data point is {days_old} days old. yfinance may be delayed.",
            date=str(last_date.date())))
    elif days_old > STALE_DAYS * 2:
        issues.append(ValidationIssue("ERROR", "VERY_STALE_DATA",
            f"Last data point is {days_old} days old. Data feed may be broken.",
            date=str(last_date.date())))
        result.ok = False

    # ── 3. Price range sanity ──────────────────────────────────────────────
    bad_price_low  = (close < MIN_PRICE_THB) & (close > 0)
    bad_price_high = close > MAX_PRICE_THB
    zero_price     = close <= 0

    for date in df.index[zero_price]:
        issues.append(ValidationIssue("ERROR", "ZERO_PRICE",
            f"Close price is zero or negative. Data error or delisted.",
            date=str(date.date()), value=float(close[date])))
        result.ok = False

    for date in df.index[bad_price_low]:
        issues.append(ValidationIssue("WARNING", "PRICE_TOO_LOW",
            f"Close price below {MIN_PRICE_THB} THB. Possible delisting or penny stock.",
            date=str(date.date()), value=float(close[date])))

    for date in df.index[bad_price_high]:
        issues.append(ValidationIssue("WARNING", "PRICE_VERY_HIGH",
            f"Close price above {MAX_PRICE_THB:,.0f} THB. Verify this is correct.",
            date=str(date.date()), value=float(close[date])))

    # ── 4. Single-day spike detection ─────────────────────────────────────
    daily_ret = close.pct_change().fillna(0) * 100

    hard_spikes = daily_ret.abs() > HARD_SPIKE_THRESHOLD_PCT
    soft_spikes = (daily_ret.abs() > SPIKE_THRESHOLD_PCT) & ~hard_spikes

    for date in df.index[hard_spikes]:
        ret = float(daily_ret[date])
        issues.append(ValidationIssue("ERROR", "PRICE_SPIKE_HARD",
            f"Single-day move of {ret:+.1f}% is almost certainly a data error "
            f"(split/dividend unadjusted or bad tick). Do NOT trade on this data.",
            date=str(date.date()), value=ret))
        result.ok = False

    for date in df.index[soft_spikes]:
        ret = float(daily_ret[date])
        issues.append(ValidationIssue("WARNING", "PRICE_SPIKE_SOFT",
            f"Single-day move of {ret:+.1f}% is suspicious. "
            f"Could be corporate action or genuine gap. Verify before trading.",
            date=str(date.date()), value=ret))

    # ── 5. Corporate action heuristic ─────────────────────────────────────
    # A price that drops exactly 50% or 33% is likely a 2:1 or 3:1 split
    for date in df.index[1:]:
        ret = float(daily_ret.get(date, 0))
        if -52 < ret < -48:  # close to -50%
            issues.append(ValidationIssue("WARNING", "POSSIBLE_SPLIT_2FOR1",
                f"Price dropped ~50%. Possible 2-for-1 stock split. "
                f"Check if yfinance adjusted for this.",
                date=str(date.date()), value=ret))
        if -35 < ret < -31:  # close to -33%
            issues.append(ValidationIssue("WARNING", "POSSIBLE_SPLIT_3FOR2",
                f"Price dropped ~33%. Possible 3-for-2 stock split. "
                f"Check if yfinance adjusted for this.",
                date=str(date.date()), value=ret))

    # ── 6. OHLC consistency ────────────────────────────────────────────────
    bad_hl = high < low - OHLC_TOLERANCE
    bad_ch = close > high + OHLC_TOLERANCE
    bad_cl = close < low - OHLC_TOLERANCE
    bad_oh = open_ > high + OHLC_TOLERANCE
    bad_ol = open_ < low - OHLC_TOLERANCE

    for mask, code, msg in [
        (bad_hl, "OHLC_HIGH_LT_LOW",  "High < Low: impossible OHLC bar"),
        (bad_ch, "OHLC_CLOSE_GT_HIGH","Close > High: impossible OHLC bar"),
        (bad_cl, "OHLC_CLOSE_LT_LOW", "Close < Low: impossible OHLC bar"),
        (bad_oh, "OHLC_OPEN_GT_HIGH", "Open > High: impossible OHLC bar"),
        (bad_ol, "OHLC_OPEN_LT_LOW",  "Open < Low: impossible OHLC bar"),
    ]:
        for date in df.index[mask]:
            issues.append(ValidationIssue("ERROR", code, msg,
                date=str(date.date())))
            result.ok = False

    # ── 7. Volume checks ───────────────────────────────────────────────────
    zero_vol = vol == 0
    n_zero_vol = zero_vol.sum()
    if n_zero_vol > 0:
        zero_vol_pct = n_zero_vol / len(df) * 100
        if zero_vol_pct > 20:
            issues.append(ValidationIssue("ERROR", "EXCESSIVE_ZERO_VOLUME",
                f"{n_zero_vol} bars ({zero_vol_pct:.0f}%) have zero volume. "
                f"This stock may be illiquid or data is missing."))
            result.ok = False
        elif n_zero_vol > 2:
            issues.append(ValidationIssue("WARNING", "SOME_ZERO_VOLUME",
                f"{n_zero_vol} bars ({zero_vol_pct:.0f}%) have zero volume. "
                f"Normal for Thai holidays, suspicious if not."))

    # ── 8. NaN checks ─────────────────────────────────────────────────────
    for col in ["Open", "High", "Low", "Close"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            nan_pct = n_nan / len(df) * 100
            severity = "ERROR" if nan_pct > 10 else "WARNING"
            issues.append(ValidationIssue(severity, f"NAN_IN_{col.upper()}",
                f"{n_nan} NaN values in {col} ({nan_pct:.1f}%). "
                f"Gaps in data will distort signals."))
            if severity == "ERROR":
                result.ok = False

    # ── Summary ────────────────────────────────────────────────────────────
    error_rows = set()
    for issue in issues:
        if issue.severity == "ERROR" and issue.date:
            error_rows.add(issue.date)

    result.clean_pct = max(0.0, (len(df) - len(error_rows)) / len(df) * 100)
    return result


def validate_universe(price_data: dict, min_clean_pct: float = 90.0) -> dict:
    """
    Validate all stocks in a universe dict {symbol: DataFrame}.
    Returns dict with 'passed', 'failed', and 'results'.

    Args:
        price_data: dict of {symbol: DataFrame}
        min_clean_pct: minimum % of clean bars required to pass

    Returns:
        {
          'passed': [list of clean symbols],
          'failed': [list of dirty symbols],
          'results': {symbol: ValidationResult},
        }
    """
    passed = []
    failed = []
    results = {}

    print(f"\nData Quality Check ({len(price_data)} stocks):")
    print("  " + "-" * 55)

    for symbol, df in price_data.items():
        result = validate_dataframe(df, symbol)

        # Also fail if clean_pct is below threshold
        if result.clean_pct < min_clean_pct:
            result.ok = False

        results[symbol] = result
        status = "PASS" if result.ok else "FAIL"
        n_errors   = len(result.errors())
        n_warnings = len(result.warnings())

        print(f"  {symbol:<8} {status:<5}  "
              f"{result.clean_pct:>5.1f}% clean  "
              f"{n_errors} errors  {n_warnings} warnings")

        if result.ok:
            passed.append(symbol)
        else:
            failed.append(symbol)

    print("  " + "-" * 55)
    print(f"  Passed: {len(passed)}  Failed: {len(failed)}\n")

    if failed:
        print("  Failed stocks will be excluded from trading today.")
        for sym in failed:
            print(f"\n  Detail: {sym}")
            print("  " + results[sym].report())

    return {"passed": passed, "failed": failed, "results": results}


def is_safe_to_trade(symbol: str, df: pd.DataFrame,
                     verbose: bool = False) -> bool:
    """
    Quick single-stock check. Returns True if safe to trade.
    Use this inside screener.py before placing any order.

    Example:
        if not is_safe_to_trade(symbol, df):
            print(f"Skipping {symbol}: data quality issue")
            continue
    """
    result = validate_dataframe(df, symbol)
    if verbose or not result.ok:
        print(result.report())
    return result.ok


# ── CLI usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.fetcher import DataFetcher

    symbols = sys.argv[1:] or ["PTT", "ADVANC", "KBANK", "CPALL", "BDMS"]
    print(f"Fetching data for: {symbols}")

    fetcher = DataFetcher()
    price_data = {}
    for sym in symbols:
        df = fetcher.get_historical(sym)
        if df is not None and not df.empty:
            price_data[sym] = df

    summary = validate_universe(price_data)

    print("\nSafe to trade:")
    for sym in summary["passed"]:
        print(f"  {sym}")

    if summary["failed"]:
        print("\nExcluded (data quality):")
        for sym in summary["failed"]:
            print(f"  {sym}")
