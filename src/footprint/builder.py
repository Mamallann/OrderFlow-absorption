"""Footprint builder - converts tick data to footprint candles."""

from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from typing import List, Optional, Dict
import logging

import pandas as pd
import numpy as np

from .structures import FootprintCandle, FootprintLevel
from ..data.models import AggTrade

logger = logging.getLogger(__name__)


class FootprintBuilder:
    """
    Build footprint candles from aggregated trades.

    Converts raw tick data into footprint candles with bid/ask
    volume per price level, delta, and other metrics.
    """

    # Timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "10m": 10 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    def __init__(
        self,
        timeframe: str = "10m",
        tick_size: float = 100.0,
    ):
        """
        Initialize footprint builder.

        Args:
            timeframe: Candle timeframe (e.g., "10m", "1h")
            tick_size: Price quantization in USD (e.g., 100 = $100 per level)
        """
        self.timeframe = timeframe
        self.tick_size = Decimal(str(tick_size))

        if timeframe not in self.TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        self.timeframe_ms = self.TIMEFRAME_MS[timeframe]

    def quantize_price(self, price: Decimal) -> Decimal:
        """
        Quantize price to tick size.

        Args:
            price: Raw price

        Returns:
            Quantized price (rounded down to nearest tick)
        """
        return (price / self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.tick_size

    def get_candle_timestamp(self, trade_timestamp: int) -> int:
        """
        Get candle open timestamp for a trade.

        Args:
            trade_timestamp: Trade timestamp in milliseconds

        Returns:
            Candle open timestamp
        """
        return (trade_timestamp // self.timeframe_ms) * self.timeframe_ms

    def build_from_trades(
        self,
        trades: List[AggTrade],
        compute_cum_delta: bool = True,
    ) -> List[FootprintCandle]:
        """
        Build footprint candles from trade list.

        Args:
            trades: List of AggTrade objects
            compute_cum_delta: Whether to compute cumulative delta

        Returns:
            List of FootprintCandle objects
        """
        if not trades:
            return []

        # Group trades by candle
        candle_trades: Dict[int, List[AggTrade]] = defaultdict(list)

        for trade in trades:
            candle_ts = self.get_candle_timestamp(trade.timestamp)
            candle_trades[candle_ts].append(trade)

        # Build candles
        candles = []
        cum_delta = Decimal("0")

        for candle_ts in sorted(candle_trades.keys()):
            candle = self._build_single_candle(
                candle_ts,
                candle_trades[candle_ts],
            )

            if compute_cum_delta:
                cum_delta += candle.delta
                candle.cum_delta = cum_delta

            candles.append(candle)

        logger.info(f"Built {len(candles)} footprint candles")
        return candles

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        compute_cum_delta: bool = True,
    ) -> List[FootprintCandle]:
        """
        Build footprint candles from DataFrame.

        Args:
            df: DataFrame with columns: price, quantity, timestamp, is_buyer_maker
            compute_cum_delta: Whether to compute cumulative delta

        Returns:
            List of FootprintCandle objects
        """
        if df.empty:
            return []

        # Ensure proper types
        df = df.copy()
        df["price"] = df["price"].astype(float)
        df["quantity"] = df["quantity"].astype(float)
        df["timestamp"] = df["timestamp"].astype(int)
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

        # Calculate candle timestamps
        df["candle_ts"] = (df["timestamp"] // self.timeframe_ms) * self.timeframe_ms

        # Quantize prices
        tick_size = float(self.tick_size)
        df["price_level"] = (df["price"] // tick_size) * tick_size

        # Build candles
        candles = []
        cum_delta = Decimal("0")

        for candle_ts, group in df.groupby("candle_ts", sort=True):
            candle = self._build_candle_from_group(int(candle_ts), group)

            if compute_cum_delta:
                cum_delta += candle.delta
                candle.cum_delta = cum_delta

            candles.append(candle)

        logger.info(f"Built {len(candles)} footprint candles from DataFrame")
        return candles

    def _build_single_candle(
        self,
        timestamp: int,
        trades: List[AggTrade],
    ) -> FootprintCandle:
        """Build a single footprint candle from trades."""
        if not trades:
            raise ValueError("Cannot build candle from empty trades list")

        # OHLCV
        prices = [trade.price for trade in trades]
        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]
        volume = sum(trade.quantity for trade in trades)

        # Build levels
        levels: Dict[Decimal, FootprintLevel] = {}

        for trade in trades:
            price_level = self.quantize_price(trade.price)

            if price_level not in levels:
                levels[price_level] = FootprintLevel(price=price_level)

            level = levels[price_level]
            level.trade_count += 1

            if trade.is_buyer_maker:
                # Seller is aggressor (hitting bids)
                level.bid_volume += trade.quantity
            else:
                # Buyer is aggressor (lifting asks)
                level.ask_volume += trade.quantity

        # Calculate delta
        delta = sum(level.delta for level in levels.values())

        # Find POC (Point of Control)
        poc_price = max(levels.values(), key=lambda x: x.total_volume).price

        # Trade counts
        buy_trades = sum(1 for t in trades if not t.is_buyer_maker)
        sell_trades = sum(1 for t in trades if t.is_buyer_maker)

        return FootprintCandle(
            timestamp=timestamp,
            timeframe=self.timeframe,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            levels=levels,
            delta=delta,
            poc_price=poc_price,
            total_trades=len(trades),
            buy_trades=buy_trades,
            sell_trades=sell_trades,
        )

    def _build_candle_from_group(
        self,
        timestamp: int,
        group: pd.DataFrame,
    ) -> FootprintCandle:
        """Build a single footprint candle from DataFrame group."""
        # Sort by timestamp to get proper OHLC
        group = group.sort_values("timestamp")

        # OHLCV
        open_price = Decimal(str(group.iloc[0]["price"]))
        high_price = Decimal(str(group["price"].max()))
        low_price = Decimal(str(group["price"].min()))
        close_price = Decimal(str(group.iloc[-1]["price"]))
        volume = Decimal(str(group["quantity"].sum()))

        # Build levels
        levels: Dict[Decimal, FootprintLevel] = {}

        for _, row in group.iterrows():
            price_level = Decimal(str(row["price_level"]))
            qty = Decimal(str(row["quantity"]))

            if price_level not in levels:
                levels[price_level] = FootprintLevel(price=price_level)

            level = levels[price_level]
            level.trade_count += 1

            if row["is_buyer_maker"]:
                level.bid_volume += qty
            else:
                level.ask_volume += qty

        # Calculate delta
        delta = sum(level.delta for level in levels.values())

        # Find POC
        poc_price = max(levels.values(), key=lambda x: x.total_volume).price if levels else open_price

        # Trade counts
        buy_trades = int((~group["is_buyer_maker"]).sum())
        sell_trades = int(group["is_buyer_maker"].sum())

        return FootprintCandle(
            timestamp=timestamp,
            timeframe=self.timeframe,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            levels=levels,
            delta=delta,
            poc_price=poc_price,
            total_trades=len(group),
            buy_trades=buy_trades,
            sell_trades=sell_trades,
        )


def build_footprint(
    df: pd.DataFrame,
    timeframe: str = "10m",
    tick_size: float = 100.0,
) -> List[FootprintCandle]:
    """
    Convenience function to build footprint candles.

    Args:
        df: DataFrame with trade data
        timeframe: Candle timeframe
        tick_size: Price quantization

    Returns:
        List of FootprintCandle objects
    """
    builder = FootprintBuilder(timeframe=timeframe, tick_size=tick_size)
    return builder.build_from_dataframe(df)
