# ============================================================
# SET AUTO TRADER - Data Pipeline
# Fetches real-time and historical data for SET stocks
# ============================================================

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class SETDataFeed:
    """
    Fetches market data for SET-listed stocks.

    Primary source: yfinance (ticker format: SYMBOL.BK)
    e.g. PTT.BK, ADVANC.BK, KBANK.BK

    For real-time production use, replace with:
    - Settrade Streaming API (websocket)
    - Finansia Syrus API
    - SETSMART data subscription
    """

    SUFFIX = ".BK"

    def __init__(self, cache_minutes: int = 15):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self.cache_minutes = cache_minutes

    def _to_yf_ticker(self, symbol: str) -> str:
        """Convert SET symbol to yfinance format."""
        if not symbol.endswith(self.SUFFIX):
            return symbol + self.SUFFIX
        return symbol

    def get_historical(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: SET ticker e.g. 'PTT'
            period: '1mo', '3mo', '6mo', '1y'
            interval: '1d', '1h', '30m', '15m'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{period}_{interval}"

        # Return cached data if fresh
        if self._is_cached(cache_key):
            logger.debug(f"Cache hit for {symbol}")
            return self._cache[cache_key]

        try:
            ticker = self._to_yf_ticker(symbol)
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df.index = pd.to_datetime(df.index)
            df = df.dropna()

            self._cache[cache_key] = df
            self._cache_time[cache_key] = datetime.now()

            logger.info(f"Fetched {len(df)} bars for {symbol} ({period}/{interval})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent closing price."""
        df = self.get_historical(symbol, period="5d", interval="1d")
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])

    def get_batch(self, symbols: List[str], period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        for symbol in symbols:
            df = self.get_historical(symbol, period=period)
            if df is not None:
                results[symbol] = df
            time.sleep(0.3)  # Be respectful to the API
        return results

    def get_quote_summary(self, symbol: str) -> Dict:
        """Get current quote info: price, volume, 52-week range."""
        try:
            ticker = yf.Ticker(self._to_yf_ticker(symbol))
            info = ticker.info
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache:
            return False
        age = datetime.now() - self._cache_time[key]
        return age.total_seconds() < self.cache_minutes * 60

    def clear_cache(self):
        self._cache.clear()
        self._cache_time.clear()


class SETNewsFeed:
    """
    Fetches recent news for SET-listed stocks.
    Used for Claude AI sentiment analysis.
    """

    def __init__(self):
        pass

    def get_news(self, symbol: str, max_items: int = 5) -> List[Dict]:
        """Fetch recent news headlines for a symbol."""
        try:
            ticker = yf.Ticker(f"{symbol}.BK")
            news = ticker.news or []
            results = []
            for item in news[:max_items]:
                content = item.get("content", {})
                results.append({
                    "title": content.get("title", ""),
                    "summary": content.get("summary", ""),
                    "published": content.get("pubDate", ""),
                    "url": content.get("canonicalUrl", {}).get("url", ""),
                })
            return results
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []

    def get_market_news(self, max_items: int = 10) -> List[Dict]:
        """Fetch general SET market news."""
        try:
            # Use SET index as proxy for market news
            ticker = yf.Ticker("^SET.BK")
            news = ticker.news or []
            results = []
            for item in news[:max_items]:
                content = item.get("content", {})
                results.append({
                    "title": content.get("title", ""),
                    "summary": content.get("summary", ""),
                    "published": content.get("pubDate", ""),
                })
            return results
        except Exception as e:
            logger.error(f"Failed to fetch market news: {e}")
            return []
