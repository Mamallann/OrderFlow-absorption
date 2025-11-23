"""Risk management for backtesting - SL/TP/Trailing logic."""

from decimal import Decimal
from typing import Optional, Tuple, List
import logging

from ..footprint.structures import FootprintCandle
from ..config import TakeProfitConfig, StopLossConfig, TrailingConfig, RRTarget
from .position import Position, ExitReason

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manage stop loss, take profit, and trailing stops.

    Supports:
    - Fixed R:R take profit
    - Multiple TP targets with partial closes
    - Trailing stop after TP1
    - ATR-based or percentage-based trailing
    """

    def __init__(
        self,
        tp_config: Optional[TakeProfitConfig] = None,
        sl_config: Optional[StopLossConfig] = None,
    ):
        """
        Initialize risk manager.

        Args:
            tp_config: Take profit configuration
            sl_config: Stop loss configuration
        """
        self.tp_config = tp_config or TakeProfitConfig()
        self.sl_config = sl_config or StopLossConfig()

    def check_exit(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Optional[Decimal], Optional[ExitReason]]:
        """
        Check if position should exit on this candle.

        Returns:
            Tuple of (should_exit, exit_price, exit_reason)
        """
        if position.status.value != "OPEN":
            return False, None, None

        high = candle.high
        low = candle.low

        # Update position tracking
        position.update_price(high, low, candle.close)

        # Check stop loss first (priority)
        sl_hit, sl_price = self._check_stop_loss(position, candle)
        if sl_hit:
            return True, sl_price, ExitReason.STOP_LOSS

        # Check trailing stop
        if position.trailing_activated and position.trailing_stop_price:
            trail_hit, trail_price = self._check_trailing_stop(position, candle)
            if trail_hit:
                return True, trail_price, ExitReason.TRAILING_STOP

        # Check take profit
        tp_hit, tp_price, tp_reason = self._check_take_profit(position, candle)
        if tp_hit:
            return True, tp_price, tp_reason

        return False, None, None

    def _check_stop_loss(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Optional[Decimal]]:
        """Check if stop loss is hit."""
        if position.side == "LONG":
            # SL is below entry
            if candle.low <= position.stop_loss:
                # Use SL price (assuming we get filled at SL)
                return True, position.stop_loss
        else:
            # SHORT - SL is above entry
            if candle.high >= position.stop_loss:
                return True, position.stop_loss

        return False, None

    def _check_take_profit(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Optional[Decimal], Optional[ExitReason]]:
        """Check if take profit is hit."""
        mode = self.tp_config.mode

        if mode == "rr":
            # Simple fixed R:R
            return self._check_simple_tp(position, candle)

        elif mode == "rr_then_trail":
            # Check TP first, then activate trailing
            tp_hit, tp_price = self._check_simple_tp_level(position, candle)
            if tp_hit:
                # Activate trailing instead of closing
                self._activate_trailing(position, candle)
                return False, None, None  # Don't close yet
            return False, None, None

        elif mode == "targets":
            # Multiple TP targets
            return self._check_tp_targets(position, candle)

        elif mode == "trail_only":
            # Only trailing, check activation
            self._maybe_activate_trailing(position, candle)
            return False, None, None

        return False, None, None

    def _check_simple_tp(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Optional[Decimal], Optional[ExitReason]]:
        """Check simple fixed TP."""
        if not position.take_profit:
            return False, None, None

        if position.side == "LONG":
            if candle.high >= position.take_profit:
                return True, position.take_profit, ExitReason.TAKE_PROFIT
        else:
            if candle.low <= position.take_profit:
                return True, position.take_profit, ExitReason.TAKE_PROFIT

        return False, None, None

    def _check_simple_tp_level(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Decimal]:
        """Check if TP level is reached (for rr_then_trail mode)."""
        if not position.take_profit:
            return False, Decimal("0")

        if position.side == "LONG":
            if candle.high >= position.take_profit:
                return True, position.take_profit
        else:
            if candle.low <= position.take_profit:
                return True, position.take_profit

        return False, Decimal("0")

    def _check_tp_targets(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Optional[Decimal], Optional[ExitReason]]:
        """Check multiple TP targets."""
        if not position.tp_targets:
            # Initialize targets
            position.tp_targets = self._create_tp_targets(position)

        for i, target in enumerate(position.tp_targets):
            if target.get("hit", False):
                continue

            target_price = Decimal(str(target["price"]))

            hit = False
            if position.side == "LONG":
                hit = candle.high >= target_price
            else:
                hit = candle.low <= target_price

            if hit:
                target["hit"] = True
                size = Decimal(str(target["size"]))

                # Partial close
                close_qty = position.quantity * size

                # Determine exit reason
                if i == 0:
                    reason = ExitReason.TAKE_PROFIT_1
                elif i == 1:
                    reason = ExitReason.TAKE_PROFIT_2
                else:
                    reason = ExitReason.TAKE_PROFIT_3

                # Check if this is the last target
                remaining_targets = [t for t in position.tp_targets if not t.get("hit", False)]

                if not remaining_targets:
                    # Close full remaining position
                    return True, target_price, reason
                else:
                    # Partial close
                    position.close(
                        exit_price=target_price,
                        exit_time=candle.timestamp,
                        reason=reason,
                        partial_size=close_qty,
                    )
                    # Activate trailing after first TP
                    if i == 0:
                        self._activate_trailing(position, candle)
                    return False, None, None

        return False, None, None

    def _create_tp_targets(self, position: Position) -> List[dict]:
        """Create TP target list from config."""
        targets = []
        risk = position.risk_per_unit

        for rr_target in self.tp_config.rr_targets:
            reward = risk * Decimal(str(rr_target.ratio))

            if position.side == "LONG":
                price = position.entry_price + reward
            else:
                price = position.entry_price - reward

            targets.append({
                "price": float(price),
                "size": rr_target.size,
                "ratio": rr_target.ratio,
                "hit": False,
            })

        return targets

    def _check_trailing_stop(
        self,
        position: Position,
        candle: FootprintCandle,
    ) -> Tuple[bool, Optional[Decimal]]:
        """Check if trailing stop is hit."""
        if not position.trailing_stop_price:
            return False, None

        if position.side == "LONG":
            # Update trailing stop (moves up with price)
            self._update_trailing_stop(position, candle)

            # Check if hit
            if candle.low <= position.trailing_stop_price:
                return True, position.trailing_stop_price
        else:
            # SHORT
            self._update_trailing_stop(position, candle)

            if candle.high >= position.trailing_stop_price:
                return True, position.trailing_stop_price

        return False, None

    def _activate_trailing(self, position: Position, candle: FootprintCandle):
        """Activate trailing stop."""
        if position.trailing_activated:
            return

        position.trailing_activated = True
        trailing_config = self.tp_config.trailing

        # Set initial trailing stop
        if trailing_config.use_atr:
            # Would need ATR calculation - using percent for now
            trail_amount = candle.close * Decimal(str(trailing_config.trail_percent))
        else:
            trail_amount = candle.close * Decimal(str(trailing_config.trail_percent))

        if position.side == "LONG":
            position.trailing_stop_price = candle.close - trail_amount
            # Don't let trailing be worse than original SL
            position.trailing_stop_price = max(
                position.trailing_stop_price,
                position.stop_loss
            )
        else:
            position.trailing_stop_price = candle.close + trail_amount
            position.trailing_stop_price = min(
                position.trailing_stop_price,
                position.stop_loss
            )

        logger.debug(
            f"Trailing activated for position {position.position_id} "
            f"at {position.trailing_stop_price}"
        )

    def _maybe_activate_trailing(self, position: Position, candle: FootprintCandle):
        """Activate trailing if conditions met (for trail_only mode)."""
        if position.trailing_activated:
            return

        trailing_config = self.tp_config.trailing
        activation_rr = trailing_config.activation_rr
        risk = position.risk_per_unit

        if risk == 0:
            return

        # Calculate current R
        if position.side == "LONG":
            current_profit = candle.high - position.entry_price
        else:
            current_profit = position.entry_price - candle.low

        current_r = float(current_profit / risk)

        if current_r >= activation_rr:
            self._activate_trailing(position, candle)

    def _update_trailing_stop(self, position: Position, candle: FootprintCandle):
        """Update trailing stop price."""
        if not position.trailing_activated or not position.trailing_stop_price:
            return

        trailing_config = self.tp_config.trailing
        trail_amount = candle.close * Decimal(str(trailing_config.trail_percent))

        if position.side == "LONG":
            # Trail up with price
            new_trail = candle.high - trail_amount
            if new_trail > position.trailing_stop_price:
                position.trailing_stop_price = new_trail
                # Move SL up too
                position.stop_loss = max(position.stop_loss, new_trail)
        else:
            # Trail down with price
            new_trail = candle.low + trail_amount
            if new_trail < position.trailing_stop_price:
                position.trailing_stop_price = new_trail
                position.stop_loss = min(position.stop_loss, new_trail)

    def calculate_position_size(
        self,
        capital: Decimal,
        risk_percent: float,
        entry_price: Decimal,
        stop_loss: Decimal,
        mode: str = "risk",
        fixed_size: float = 1000.0,
        position_percent: float = 0.1,
    ) -> Decimal:
        """
        Calculate position size.

        Args:
            capital: Available capital
            risk_percent: Risk per trade (e.g., 0.02 = 2%)
            entry_price: Entry price
            stop_loss: Stop loss price
            mode: "risk", "fixed", or "percent"
            fixed_size: Fixed position size in USD
            position_percent: Percentage of capital

        Returns:
            Position size (quantity)
        """
        if mode == "fixed":
            return Decimal(str(fixed_size)) / entry_price

        elif mode == "percent":
            position_value = capital * Decimal(str(position_percent))
            return position_value / entry_price

        else:  # risk-based
            risk_amount = capital * Decimal(str(risk_percent))
            risk_per_unit = abs(entry_price - stop_loss)

            if risk_per_unit == 0:
                return Decimal("0")

            # Quantity that risks the specified amount
            quantity = risk_amount / risk_per_unit
            return quantity
