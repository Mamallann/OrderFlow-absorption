"""Position and trade management for backtesting."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Literal, List
from enum import Enum
import logging

from ..strategy.signals import Signal

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status."""
    PENDING = "PENDING"      # Waiting for entry
    OPEN = "OPEN"            # Position is open
    CLOSED = "CLOSED"        # Position is closed


class ExitReason(Enum):
    """Reason for position exit."""
    STOP_LOSS = "SL"
    TAKE_PROFIT = "TP"
    TAKE_PROFIT_1 = "TP1"
    TAKE_PROFIT_2 = "TP2"
    TAKE_PROFIT_3 = "TP3"
    TRAILING_STOP = "TRAIL"
    SIGNAL = "SIGNAL"        # Opposite signal
    MANUAL = "MANUAL"
    END_OF_DATA = "EOD"


@dataclass
class Position:
    """
    Represents an open or closed trading position.
    """
    position_id: int
    signal: Signal
    symbol: str
    side: Literal["LONG", "SHORT"]

    # Entry
    entry_price: Decimal
    entry_time: int
    quantity: Decimal

    # Current stop loss (may be updated for trailing)
    stop_loss: Decimal
    initial_stop_loss: Decimal

    # Take profit levels
    take_profit: Optional[Decimal] = None
    tp_targets: List[dict] = field(default_factory=list)  # [{price, size, hit}]

    # Status
    status: PositionStatus = PositionStatus.OPEN

    # Exit
    exit_price: Optional[Decimal] = None
    exit_time: Optional[int] = None
    exit_reason: Optional[ExitReason] = None

    # Tracking
    highest_price: Decimal = Decimal("0")  # For trailing (LONG)
    lowest_price: Decimal = Decimal("inf")  # For trailing (SHORT)
    max_favorable_excursion: Decimal = Decimal("0")  # Best unrealized P&L
    max_adverse_excursion: Decimal = Decimal("0")   # Worst unrealized P&L

    # Partial closes
    remaining_quantity: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    # Trailing stop state
    trailing_activated: bool = False
    trailing_stop_price: Optional[Decimal] = None

    def __post_init__(self):
        self.remaining_quantity = self.quantity
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.initial_stop_loss = self.stop_loss

    @property
    def risk_per_unit(self) -> Decimal:
        """Risk per unit (entry - initial SL)."""
        return abs(self.entry_price - self.initial_stop_loss)

    @property
    def current_r(self) -> float:
        """Current R multiple based on exit or current price."""
        if self.exit_price and self.risk_per_unit > 0:
            if self.side == "LONG":
                pnl = self.exit_price - self.entry_price
            else:
                pnl = self.entry_price - self.exit_price
            return float(pnl / self.risk_per_unit)
        return 0.0

    @property
    def unrealized_pnl(self) -> Decimal:
        """Current unrealized P&L (use highest/lowest as estimate)."""
        if self.status == PositionStatus.CLOSED:
            return Decimal("0")

        if self.side == "LONG":
            return (self.highest_price - self.entry_price) * self.remaining_quantity
        else:
            return (self.entry_price - self.lowest_price) * self.remaining_quantity

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)."""
        if self.status == PositionStatus.CLOSED and self.exit_price:
            if self.side == "LONG":
                return (self.exit_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - self.exit_price) * self.quantity
        return self.realized_pnl + self.unrealized_pnl

    def update_price(self, high: Decimal, low: Decimal, close: Decimal):
        """
        Update position with new price data.

        Args:
            high: Candle high
            low: Candle low
            close: Candle close
        """
        if self.status != PositionStatus.OPEN:
            return

        # Update highest/lowest
        if high > self.highest_price:
            self.highest_price = high
        if low < self.lowest_price:
            self.lowest_price = low

        # Update MFE/MAE
        if self.side == "LONG":
            favorable = (self.highest_price - self.entry_price) * self.remaining_quantity
            adverse = (self.entry_price - self.lowest_price) * self.remaining_quantity
        else:
            favorable = (self.entry_price - self.lowest_price) * self.remaining_quantity
            adverse = (self.highest_price - self.entry_price) * self.remaining_quantity

        if favorable > self.max_favorable_excursion:
            self.max_favorable_excursion = favorable
        if adverse > self.max_adverse_excursion:
            self.max_adverse_excursion = adverse

    def close(
        self,
        exit_price: Decimal,
        exit_time: int,
        reason: ExitReason,
        partial_size: Optional[Decimal] = None,
    ):
        """
        Close position (fully or partially).

        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Exit reason
            partial_size: If partial close, the quantity to close
        """
        if partial_size and partial_size < self.remaining_quantity:
            # Partial close
            close_qty = partial_size
            self.remaining_quantity -= close_qty

            if self.side == "LONG":
                pnl = (exit_price - self.entry_price) * close_qty
            else:
                pnl = (self.entry_price - exit_price) * close_qty

            self.realized_pnl += pnl
            logger.debug(
                f"Partial close: {close_qty} @ {exit_price}, "
                f"PnL: {pnl}, Remaining: {self.remaining_quantity}"
            )
        else:
            # Full close
            self.exit_price = exit_price
            self.exit_time = exit_time
            self.exit_reason = reason
            self.status = PositionStatus.CLOSED

            if self.side == "LONG":
                pnl = (exit_price - self.entry_price) * self.remaining_quantity
            else:
                pnl = (self.entry_price - exit_price) * self.remaining_quantity

            self.realized_pnl += pnl
            self.remaining_quantity = Decimal("0")

            logger.debug(
                f"Position closed: {reason.value} @ {exit_price}, "
                f"Total PnL: {self.realized_pnl}"
            )


@dataclass
class Trade:
    """
    Completed trade record for reporting.
    """
    trade_id: int
    position_id: int
    symbol: str
    side: Literal["LONG", "SHORT"]

    # Entry
    entry_time: int
    entry_price: Decimal

    # Exit
    exit_time: int
    exit_price: Decimal
    exit_reason: str

    # Size
    quantity: Decimal

    # P&L
    pnl: Decimal
    pnl_percent: float
    r_multiple: float

    # Risk
    initial_risk: Decimal
    stop_loss: Decimal
    take_profit: Optional[Decimal]

    # Excursion
    max_favorable_excursion: Decimal
    max_adverse_excursion: Decimal

    # Duration
    duration_ms: int

    # Confidence
    signal_confidence: float

    @classmethod
    def from_position(cls, position: Position, trade_id: int) -> "Trade":
        """Create Trade from closed Position."""
        if position.status != PositionStatus.CLOSED:
            raise ValueError("Cannot create Trade from open Position")

        pnl = position.total_pnl
        pnl_pct = float(pnl / (position.entry_price * position.quantity)) if position.quantity > 0 else 0
        r_mult = position.current_r

        return cls(
            trade_id=trade_id,
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=position.exit_time,
            exit_price=position.exit_price,
            exit_reason=position.exit_reason.value if position.exit_reason else "UNKNOWN",
            quantity=position.quantity,
            pnl=pnl,
            pnl_percent=pnl_pct,
            r_multiple=r_mult,
            initial_risk=position.risk_per_unit * position.quantity,
            stop_loss=position.initial_stop_loss,
            take_profit=position.take_profit,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            duration_ms=position.exit_time - position.entry_time if position.exit_time else 0,
            signal_confidence=position.signal.combined_confidence,
        )

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def duration_hours(self) -> float:
        return self.duration_ms / (1000 * 60 * 60)
