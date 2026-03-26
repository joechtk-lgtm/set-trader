# ============================================================
# SET AUTO TRADER - Technical Indicators
# ============================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Indicators:
    """Snapshot of all technical indicators for one symbol."""
    symbol: str
    price: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ma_short: float
    ma_long: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    volume: float
    avg_volume: float
    atr: float

    @property
    def ma_trend(self) -> str:
        """Bullish if short MA above long MA."""
        if self.ma_short > self.ma_long:
            return "bullish"
        elif self.ma_short < self.ma_long:
            return "bearish"
        return "neutral"

    @property
    def bb_position(self) -> float:
        """Where is price within Bollinger Bands? 0=lower, 1=upper."""
        band_width = self.bb_upper - self.bb_lower
        if band_width == 0:
            return 0.5
        return (self.price - self.bb_lower) / band_width

    @property
    def volume_ratio(self) -> float:
        """Current volume vs average. >1.5 = high volume."""
        if self.avg_volume == 0:
            return 1.0
        return self.volume / self.avg_volume

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": round(self.price, 2),
            "rsi": round(self.rsi, 1),
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_hist": round(self.macd_hist, 4),
            "ma_short": round(self.ma_short, 2),
            "ma_long": round(self.ma_long, 2),
            "ma_trend": self.ma_trend,
            "bb_upper": round(self.bb_upper, 2),
            "bb_lower": round(self.bb_lower, 2),
            "bb_position": round(self.bb_position, 2),
            "volume": int(self.volume),
            "volume_ratio": round(self.volume_ratio, 2),
            "atr": round(self.atr, 4),
        }


class IndicatorEngine:
    """Calculates all technical indicators from OHLCV data."""

    def __init__(
        self,
        rsi_period: int = 14,
        ma_short: int = 10,
        ma_long: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period

    def calculate(self, symbol: str, df: pd.DataFrame) -> Optional[Indicators]:
        """
        Calculate all indicators from a price DataFrame.

        Args:
            symbol: Stock ticker
            df: DataFrame with Open, High, Low, Close, Volume columns

        Returns:
            Indicators object with latest values, or None if insufficient data
        """
        min_required = max(self.ma_long, self.macd_slow + self.macd_signal_period) + 5
        if len(df) < min_required:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        try:
            rsi = self._rsi(close, self.rsi_period).iloc[-1]
            macd_line, signal_line, histogram = self._macd(
                close, self.macd_fast, self.macd_slow, self.macd_signal_period
            )
            ma_s = close.rolling(self.ma_short).mean().iloc[-1]
            ma_l = close.rolling(self.ma_long).mean().iloc[-1]
            bb_upper, bb_mid, bb_lower = self._bollinger(close, self.bb_period, self.bb_std)
            atr = self._atr(high, low, close, self.atr_period).iloc[-1]
            avg_vol = volume.rolling(20).mean().iloc[-1]

            return Indicators(
                symbol=symbol,
                price=float(close.iloc[-1]),
                rsi=float(rsi),
                macd=float(macd_line.iloc[-1]),
                macd_signal=float(signal_line.iloc[-1]),
                macd_hist=float(histogram.iloc[-1]),
                ma_short=float(ma_s),
                ma_long=float(ma_l),
                bb_upper=float(bb_upper.iloc[-1]),
                bb_mid=float(bb_mid.iloc[-1]),
                bb_lower=float(bb_lower.iloc[-1]),
                volume=float(volume.iloc[-1]),
                avg_volume=float(avg_vol),
                atr=float(atr),
            )

        except Exception as e:
            return None

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, float("inf"))
        return 100 - (100 / (1 + rs))

    def _macd(
        self, close: pd.Series, fast: int, slow: int, signal: int
    ):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _bollinger(self, close: pd.Series, period: int, std_dev: float):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    def _atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
