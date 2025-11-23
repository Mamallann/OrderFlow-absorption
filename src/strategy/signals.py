"""Signal generation for order flow strategy."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Literal
import logging

from ..footprint.structures import FootprintCandle
from ..config import Config, EntryConfig, StopLossConfig, TakeProfitConfig
from .absorption import AbsorptionDetector, AbsorptionEvent
from .shift import ShiftDetector, ShiftEvent

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    Trading signal generated from absorption + shift detection.
    """
    signal_id: int
    timestamp: int
    symbol: str
    side: Literal["LONG", "SHORT"]

    # Entry
    entry_price: Decimal
    entry_candle_idx: int

    # Risk management
    stop_loss: Decimal

    # Context (required fields - no defaults)
    absorption_event: AbsorptionEvent = field(repr=False)
    shift_event: ShiftEvent = field(repr=False)

    # Optional/default fields must come after required fields
    take_profit: Optional[Decimal] = None  # Primary TP (if fixed R:R)

    # Metrics
    risk_amount: Decimal = Decimal("0")  # Entry - SL
    reward_amount: Decimal = Decimal("0")  # TP - Entry
    rr_ratio: float = 0.0

    # Confidence
    absorption_confidence: float = 0.0
    shift_confidence: float = 0.0
    combined_confidence: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Signal({self.side} @ {self.entry_price}, "
            f"SL={self.stop_loss}, TP={self.take_profit}, "
            f"conf={self.combined_confidence:.2f})"
        )


class SignalGenerator:
    """
    Generate trading signals from footprint analysis.

    Combines absorption detection and shift detection
    to generate entry signals with stop loss and take profit.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        absorption_detector: Optional[AbsorptionDetector] = None,
        shift_detector: Optional[ShiftDetector] = None,
    ):
        """
        Initialize signal generator.

        Args:
            config: Full configuration
            absorption_detector: Absorption detector instance
            shift_detector: Shift detector instance
        """
        self.config = config or Config()
        self.absorption_detector = absorption_detector or AbsorptionDetector(
            config=self.config.absorption
        )
        self.shift_detector = shift_detector or ShiftDetector(
            config=self.config.shift
        )

        self._signal_counter = 0

    def generate_signals(
        self,
        candles: List[FootprintCandle],
        symbol: str = "BTCUSDT",
        direction: Literal["LONG", "SHORT", "BOTH"] = "BOTH",
    ) -> List[Signal]:
        """
        Generate all trading signals from candles.

        Args:
            candles: List of footprint candles
            symbol: Trading symbol
            direction: Which direction to scan

        Returns:
            List of trading signals
        """
        # Step 1: Detect absorptions
        absorptions = self.absorption_detector.scan_for_absorption(
            candles, direction=direction
        )

        if not absorptions:
            logger.info("No absorption events detected")
            return []

        # Step 2: Find shifts for each absorption
        shifts = self.shift_detector.find_shifts_for_absorptions(
            candles, absorptions
        )

        if not shifts:
            logger.info("No shift events detected")
            return []

        # Step 3: Generate signals from shifts
        signals = []
        for shift in shifts:
            signal = self._create_signal(candles, shift, symbol)
            if signal:
                signals.append(signal)

        # Sort by timestamp
        signals.sort(key=lambda x: x.timestamp)

        logger.info(f"Generated {len(signals)} signals")
        return signals

    def _create_signal(
        self,
        candles: List[FootprintCandle],
        shift: ShiftEvent,
        symbol: str,
    ) -> Optional[Signal]:
        """
        Create a trading signal from a shift event.

        Args:
            candles: List of candles
            shift: Shift event
            symbol: Trading symbol

        Returns:
            Signal if valid, None otherwise
        """
        absorption = shift.absorption_event
        shift_candle = shift.candle
        entry_config = self.config.entry
        sl_config = self.config.stop_loss
        tp_config = self.config.take_profit

        # Determine side
        side: Literal["LONG", "SHORT"]
        if shift.shift_type == "BULLISH":
            side = "LONG"
        else:
            side = "SHORT"

        # Calculate entry price
        entry_price = self._calculate_entry_price(
            candles, shift, entry_config
        )

        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            candles, shift, side, sl_config
        )

        # Validate SL distance
        if side == "LONG" and stop_loss >= entry_price:
            logger.warning(f"Invalid SL for LONG: SL={stop_loss} >= Entry={entry_price}")
            return None
        if side == "SHORT" and stop_loss <= entry_price:
            logger.warning(f"Invalid SL for SHORT: SL={stop_loss} <= Entry={entry_price}")
            return None

        # Calculate risk
        risk_amount = abs(entry_price - stop_loss)

        # Check max SL percent
        max_sl_pct = sl_config.max_sl_percent
        sl_pct = float(risk_amount / entry_price)
        if sl_pct > max_sl_pct:
            logger.warning(f"SL too wide: {sl_pct:.2%} > {max_sl_pct:.2%}")
            return None

        # Calculate take profit
        take_profit = self._calculate_take_profit(
            entry_price, stop_loss, side, tp_config
        )

        # Calculate reward
        reward_amount = Decimal("0")
        rr_ratio = 0.0
        if take_profit:
            reward_amount = abs(take_profit - entry_price)
            if risk_amount > 0:
                rr_ratio = float(reward_amount / risk_amount)

        # Calculate combined confidence
        combined_confidence = (
            absorption.confidence * 0.5 + shift.confidence * 0.5
        )

        # Generate signal ID
        self._signal_counter += 1

        return Signal(
            signal_id=self._signal_counter,
            timestamp=shift_candle.timestamp,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_candle_idx=shift.candle_idx,
            stop_loss=stop_loss,
            take_profit=take_profit,
            absorption_event=absorption,
            shift_event=shift,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            rr_ratio=rr_ratio,
            absorption_confidence=absorption.confidence,
            shift_confidence=shift.confidence,
            combined_confidence=combined_confidence,
        )

    def _calculate_entry_price(
        self,
        candles: List[FootprintCandle],
        shift: ShiftEvent,
        config: EntryConfig,
    ) -> Decimal:
        """Calculate entry price based on configuration."""
        shift_candle = shift.candle
        mode = config.mode

        if mode == "shift_close":
            # Enter at shift candle close
            return shift_candle.close

        elif mode == "next_candle_open":
            # Enter at next candle open
            next_idx = shift.candle_idx + 1
            if next_idx < len(candles):
                return candles[next_idx].open
            return shift_candle.close

        elif mode == "breakout":
            # Enter on breakout of shift candle
            buffer = Decimal(str(config.breakout_buffer))
            if shift.shift_type == "BULLISH":
                return shift_candle.high * (1 + buffer)
            else:
                return shift_candle.low * (1 - buffer)

        return shift_candle.close

    def _calculate_stop_loss(
        self,
        candles: List[FootprintCandle],
        shift: ShiftEvent,
        side: str,
        config: StopLossConfig,
    ) -> Decimal:
        """Calculate stop loss based on configuration."""
        absorption = shift.absorption_event
        absorption_candle = absorption.candle
        shift_candle = shift.candle

        mode = config.mode
        buffer = Decimal(str(config.buffer))

        if side == "LONG":
            # SL below absorption/shift low
            if mode == "absorption_low":
                base_sl = absorption_candle.low
            elif mode == "shift_low":
                base_sl = shift_candle.low
            else:
                base_sl = min(absorption_candle.low, shift_candle.low)

            # Apply buffer (below)
            return base_sl * (1 - buffer)

        else:  # SHORT
            # SL above absorption/shift high
            if mode == "absorption_low":  # Actually absorption_high for shorts
                base_sl = absorption_candle.high
            elif mode == "shift_low":  # Actually shift_high for shorts
                base_sl = shift_candle.high
            else:
                base_sl = max(absorption_candle.high, shift_candle.high)

            # Apply buffer (above)
            return base_sl * (1 + buffer)

    def _calculate_take_profit(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
        side: str,
        config: TakeProfitConfig,
    ) -> Optional[Decimal]:
        """Calculate take profit based on configuration."""
        mode = config.mode
        risk = abs(entry_price - stop_loss)

        if mode in ("rr", "rr_then_trail"):
            # Fixed R:R
            rr_ratio = Decimal(str(config.rr_ratio))
            reward = risk * rr_ratio

            if side == "LONG":
                return entry_price + reward
            else:
                return entry_price - reward

        elif mode == "trail_only":
            # No fixed TP, will use trailing
            return None

        elif mode == "targets":
            # Use first target as primary TP
            if config.rr_targets:
                first_target = config.rr_targets[0]
                reward = risk * Decimal(str(first_target.ratio))

                if side == "LONG":
                    return entry_price + reward
                else:
                    return entry_price - reward

        return None

    def filter_signals(
        self,
        signals: List[Signal],
        min_confidence: float = 0.5,
        min_rr_ratio: float = 1.5,
    ) -> List[Signal]:
        """
        Filter signals based on criteria.

        Args:
            signals: List of signals
            min_confidence: Minimum combined confidence
            min_rr_ratio: Minimum risk:reward ratio

        Returns:
            Filtered signals
        """
        filtered = []

        for signal in signals:
            if signal.combined_confidence < min_confidence:
                continue
            if signal.rr_ratio < min_rr_ratio:
                continue
            filtered.append(signal)

        logger.info(f"Filtered {len(signals)} → {len(filtered)} signals")
        return filtered
