"""Core data models for the OrderFlow backtest system."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
from enum import Enum
import pandas as pd


class Side(Enum):
    """Trade side - BUY or SELL aggressor."""
    BUY = "BUY"    # Buyer is aggressor (lifting asks)
    SELL = "SELL"  # Seller is aggressor (hitting bids)


@dataclass
class AggTrade:
    """
    Aggregated trade from Binance.

    Represents a single trade or aggregated trades at the same price/time.
    is_buyer_maker: True means SELL aggressor (buyer was maker, seller took)
                   False means BUY aggressor (seller was maker, buyer took)
    """
    agg_trade_id: int
    price: Decimal
    quantity: Decimal
    first_trade_id: int
    last_trade_id: int
    timestamp: int  # Unix milliseconds
    is_buyer_maker: bool

    @property
    def side(self) -> Side:
        """Get the aggressor side."""
        # is_buyer_maker=True means the SELL side was aggressor (hitting bids)
        # is_buyer_maker=False means the BUY side was aggressor (lifting asks)
        return Side.SELL if self.is_buyer_maker else Side.BUY

    @classmethod
    def from_binance(cls, data: dict) -> "AggTrade":
        """Create from Binance API response."""
        return cls(
            agg_trade_id=data["a"],
            price=Decimal(str(data["p"])),
            quantity=Decimal(str(data["q"])),
            first_trade_id=data["f"],
            last_trade_id=data["l"],
            timestamp=data["T"],
            is_buyer_maker=data["m"],
        )

    @classmethod
    def from_row(cls, row: pd.Series) -> "AggTrade":
        """Create from DataFrame row."""
        return cls(
            agg_trade_id=int(row["agg_trade_id"]),
            price=Decimal(str(row["price"])),
            quantity=Decimal(str(row["quantity"])),
            first_trade_id=int(row["first_trade_id"]),
            last_trade_id=int(row["last_trade_id"]),
            timestamp=int(row["timestamp"]),
            is_buyer_maker=bool(row["is_buyer_maker"]),
        )


@dataclass
class Candle:
    """Basic OHLCV candle."""
    timestamp: int  # Unix milliseconds (candle open time)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades: int = 0

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


@dataclass
class TradeRecord:
    """
    Record of a completed backtest trade.
    """
    trade_id: int
    symbol: str
    side: Side  # LONG or SHORT

    # Entry
    entry_time: int
    entry_price: Decimal
    entry_reason: str

    # Exit
    exit_time: int
    exit_price: Decimal
    exit_reason: str  # "SL", "TP", "TP1", "TP2", "TRAIL", "SIGNAL"

    # Position
    quantity: Decimal

    # P&L
    pnl: Decimal
    pnl_percent: float
    r_multiple: float

    # Risk metrics
    stop_loss: Decimal
    take_profit: Optional[Decimal]
    risk_amount: Decimal

    # Excursion
    max_favorable_excursion: Decimal  # Best unrealized P&L
    max_adverse_excursion: Decimal    # Worst unrealized P&L

    # Context
    absorption_candle_idx: int
    shift_candle_idx: int

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def duration_ms(self) -> int:
        return self.exit_time - self.entry_time
