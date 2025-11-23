"""Binance API client for fetching AggTrades data."""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, AsyncIterator
import logging

import httpx
import pandas as pd

from .models import AggTrade

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Client for fetching Binance Futures AggTrades data.

    Uses the public API endpoint (no authentication required).
    Handles pagination, rate limiting, and data storage.
    """

    BASE_URL = "https://fapi.binance.com"
    AGGTRADES_ENDPOINT = "/fapi/v1/aggTrades"
    MAX_TRADES_PER_REQUEST = 1000
    DEFAULT_RATE_LIMIT_DELAY = 0.1  # seconds

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        testnet: bool = False,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit_delay = rate_limit_delay

        if testnet:
            self.BASE_URL = "https://testnet.binancefuture.com"

        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch_agg_trades(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        from_id: Optional[int] = None,
        limit: int = MAX_TRADES_PER_REQUEST,
    ) -> List[AggTrade]:
        """
        Fetch aggregated trades from Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            from_id: Start from this aggregate trade ID
            limit: Maximum number of trades (max 1000)

        Returns:
            List of AggTrade objects
        """
        client = await self._get_client()

        params = {
            "symbol": symbol,
            "limit": min(limit, self.MAX_TRADES_PER_REQUEST),
        }

        if from_id is not None:
            params["fromId"] = from_id
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        url = f"{self.BASE_URL}{self.AGGTRADES_ENDPOINT}"

        try:
            logger.info(f"Fetching from {url} with params: {params}")
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning(f"Empty response from Binance for {symbol}")
                return []

            trades = [AggTrade.from_binance(item) for item in data]
            logger.info(f"Fetched {len(trades)} trades for {symbol}")

            return trades

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching trades: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            raise

    async def fetch_agg_trades_range(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        progress_callback: Optional[callable] = None,
    ) -> AsyncIterator[List[AggTrade]]:
        """
        Fetch all aggregated trades in a time range.

        Handles pagination automatically.

        Args:
            symbol: Trading pair
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            progress_callback: Optional callback(current_time, end_time)

        Yields:
            Batches of AggTrade objects
        """
        current_time = start_time
        last_trade_id = None

        while current_time < end_time:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            try:
                if last_trade_id is not None:
                    # Fetch from last trade ID to avoid duplicates
                    trades = await self.fetch_agg_trades(
                        symbol=symbol,
                        from_id=last_trade_id + 1,
                        limit=self.MAX_TRADES_PER_REQUEST,
                    )
                else:
                    # First request - use start time
                    trades = await self.fetch_agg_trades(
                        symbol=symbol,
                        start_time=current_time,
                        end_time=end_time,
                        limit=self.MAX_TRADES_PER_REQUEST,
                    )

                if not trades:
                    logger.info(f"No more trades after {current_time}")
                    break

                # Filter trades within our time range
                trades = [t for t in trades if t.timestamp <= end_time]

                if not trades:
                    break

                # Update position
                last_trade = trades[-1]
                last_trade_id = last_trade.agg_trade_id
                current_time = last_trade.timestamp

                if progress_callback:
                    progress_callback(current_time, end_time)

                yield trades

                # If we got fewer than limit, we're at the end
                if len(trades) < self.MAX_TRADES_PER_REQUEST:
                    break

            except Exception as e:
                logger.error(f"Error in fetch loop: {e}")
                # Wait and retry
                await asyncio.sleep(2.0)
                continue

    async def download_to_dataframe(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Download all trades in date range to DataFrame.

        Args:
            symbol: Trading pair
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            progress_callback: Optional progress callback

        Returns:
            DataFrame with all trades
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        start_time = int(start_dt.timestamp() * 1000)
        end_time = int(end_dt.timestamp() * 1000)

        logger.info(
            f"Downloading {symbol} trades from {start_date} to {end_date}"
        )

        all_trades = []

        async for batch in self.fetch_agg_trades_range(
            symbol, start_time, end_time, progress_callback
        ):
            all_trades.extend(batch)
            logger.info(f"Downloaded {len(all_trades)} trades so far...")

        if not all_trades:
            logger.warning(f"No trades found for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "agg_trade_id": t.agg_trade_id,
                "price": float(t.price),
                "quantity": float(t.quantity),
                "first_trade_id": t.first_trade_id,
                "last_trade_id": t.last_trade_id,
                "timestamp": t.timestamp,
                "is_buyer_maker": t.is_buyer_maker,
            }
            for t in all_trades
        ])

        logger.info(f"Downloaded {len(df)} total trades for {symbol}")
        return df

    async def download_to_file(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw",
        file_format: str = "parquet",
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """
        Download trades to file.

        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory
            file_format: "parquet" or "csv"
            progress_callback: Optional progress callback

        Returns:
            Path to saved file
        """
        df = await self.download_to_dataframe(
            symbol, start_date, end_date, progress_callback
        )

        if df.empty:
            raise ValueError(f"No data downloaded for {symbol}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{symbol}_{start_date}_{end_date}_aggtrades"

        if file_format == "parquet":
            filepath = output_path / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        else:
            filepath = output_path / f"{filename}.csv"
            df.to_csv(filepath, index=False)

        logger.info(f"Saved {len(df)} trades to {filepath}")
        return filepath


# Synchronous wrapper for convenience
def download_trades(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    file_format: str = "parquet",
) -> Path:
    """
    Synchronous wrapper to download trades.

    Args:
        symbol: Trading pair
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        file_format: "parquet" or "csv"

    Returns:
        Path to saved file
    """
    async def _download():
        async with BinanceClient() as client:
            return await client.download_to_file(
                symbol, start_date, end_date, output_dir, file_format
            )

    return asyncio.run(_download())
