"""Footprint data structures."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Literal
from enum import Enum


class ImbalanceType(Enum):
    """Type of diagonal imbalance."""
    BID = "BID"   # Bid imbalance = buying absorption (strong buyers at this level)
    ASK = "ASK"   # Ask imbalance = selling absorption (strong sellers at this level)


@dataclass
class FootprintLevel:
    """
    Single price level in a footprint candle.

    Contains bid (sell aggressor) and ask (buy aggressor) volumes.
    """
    price: Decimal
    bid_volume: Decimal = Decimal("0")  # Sell aggressor volume (hitting bids)
    ask_volume: Decimal = Decimal("0")  # Buy aggressor volume (lifting asks)
    trade_count: int = 0

    @property
    def delta(self) -> Decimal:
        """Level delta = ask - bid (positive = net buying)."""
        return self.ask_volume - self.bid_volume

    @property
    def total_volume(self) -> Decimal:
        """Total volume at this level."""
        return self.bid_volume + self.ask_volume

    @property
    def bid_ask_ratio(self) -> float:
        """Ratio of bid to ask volume."""
        if self.ask_volume == 0:
            return float("inf") if self.bid_volume > 0 else 0
        return float(self.bid_volume / self.ask_volume)

    @property
    def ask_bid_ratio(self) -> float:
        """Ratio of ask to bid volume."""
        if self.bid_volume == 0:
            return float("inf") if self.ask_volume > 0 else 0
        return float(self.ask_volume / self.bid_volume)

    def __repr__(self) -> str:
        return f"Level({self.price}: {self.bid_volume}×{self.ask_volume}, Δ={self.delta})"


@dataclass
class DiagonalImbalance:
    """
    Diagonal imbalance between adjacent price levels.

    BID imbalance: bid_volume at price P >> ask_volume at price P+1 (diagonal)
                   Indicates strong buying absorption at P

    ASK imbalance: ask_volume at price P >> bid_volume at price P-1 (diagonal)
                   Indicates strong selling absorption at P
    """
    price: Decimal                          # Price level of the imbalance
    type: ImbalanceType                     # BID or ASK
    ratio: float                            # Imbalance ratio (e.g., 3.5 = 350%)
    volume: Decimal                         # Volume that caused imbalance
    diagonal_volume: Decimal                # Volume at diagonal level
    consecutive_count: int = 1              # Number of consecutive imbalances

    @property
    def strength(self) -> float:
        """Imbalance strength = ratio × consecutive_count."""
        return self.ratio * self.consecutive_count

    def __repr__(self) -> str:
        return (
            f"Imbalance({self.type.value} @ {self.price}, "
            f"ratio={self.ratio:.1f}x, consec={self.consecutive_count})"
        )


@dataclass
class FootprintCandle:
    """
    Complete footprint candle with all order flow data.

    Contains OHLCV data plus bid/ask volume per price level,
    delta calculations, and detected imbalances.
    """
    # Time
    timestamp: int  # Unix milliseconds (candle open time)
    timeframe: str  # e.g., "10m"

    # OHLCV
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    # Footprint data - price -> level
    levels: Dict[Decimal, FootprintLevel] = field(default_factory=dict)

    # Computed delta values
    delta: Decimal = Decimal("0")           # Candle delta (sum of all level deltas)
    cum_delta: Decimal = Decimal("0")       # Cumulative delta from start
    max_delta: Decimal = Decimal("0")       # Maximum intra-candle delta
    min_delta: Decimal = Decimal("0")       # Minimum intra-candle delta

    # Volume analysis
    poc_price: Optional[Decimal] = None     # Point of Control (highest volume level)
    value_area_high: Optional[Decimal] = None
    value_area_low: Optional[Decimal] = None

    # Imbalances detected in this candle
    bid_imbalances: List[DiagonalImbalance] = field(default_factory=list)
    ask_imbalances: List[DiagonalImbalance] = field(default_factory=list)

    # Flags
    has_absorption: bool = False
    absorption_type: Optional[Literal["BUY", "SELL"]] = None

    # Trade counts
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0

    @property
    def total_bid_volume(self) -> Decimal:
        """Total bid (sell aggressor) volume."""
        return sum(level.bid_volume for level in self.levels.values())

    @property
    def total_ask_volume(self) -> Decimal:
        """Total ask (buy aggressor) volume."""
        return sum(level.ask_volume for level in self.levels.values())

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_size(self) -> Decimal:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> Decimal:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> Decimal:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def has_bid_imbalance(self) -> bool:
        return len(self.bid_imbalances) > 0

    @property
    def has_ask_imbalance(self) -> bool:
        return len(self.ask_imbalances) > 0

    @property
    def max_consecutive_bid_imbalance(self) -> int:
        """Maximum consecutive bid imbalance rows."""
        if not self.bid_imbalances:
            return 0
        return max(imb.consecutive_count for imb in self.bid_imbalances)

    @property
    def max_consecutive_ask_imbalance(self) -> int:
        """Maximum consecutive ask imbalance rows."""
        if not self.ask_imbalances:
            return 0
        return max(imb.consecutive_count for imb in self.ask_imbalances)

    @property
    def strongest_bid_imbalance(self) -> Optional[DiagonalImbalance]:
        """Get the strongest bid imbalance by ratio."""
        if not self.bid_imbalances:
            return None
        return max(self.bid_imbalances, key=lambda x: x.ratio)

    @property
    def strongest_ask_imbalance(self) -> Optional[DiagonalImbalance]:
        """Get the strongest ask imbalance by ratio."""
        if not self.ask_imbalances:
            return None
        return max(self.ask_imbalances, key=lambda x: x.ratio)

    def get_level(self, price: Decimal) -> Optional[FootprintLevel]:
        """Get footprint level at specific price."""
        return self.levels.get(price)

    def get_sorted_levels(self, ascending: bool = True) -> List[FootprintLevel]:
        """Get levels sorted by price."""
        return sorted(
            self.levels.values(),
            key=lambda x: x.price,
            reverse=not ascending
        )

    def __repr__(self) -> str:
        return (
            f"FootprintCandle({self.timestamp}, "
            f"O={self.open} H={self.high} L={self.low} C={self.close}, "
            f"Δ={self.delta}, levels={len(self.levels)})"
        )
