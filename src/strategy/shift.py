"""Shift detection for order flow strategy."""

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Literal
import logging

from ..footprint.structures import FootprintCandle
from ..config import ShiftConfig
from .absorption import AbsorptionEvent

logger = logging.getLogger(__name__)


@dataclass
class ShiftEvent:
    """
    Detected shift event.

    Represents a candle where control shifted from
    absorption to the opposite side (entry signal).
    """
    candle_idx: int
    candle: FootprintCandle
    shift_type: Literal["BULLISH", "BEARISH"]  # BULLISH = long entry, BEARISH = short entry

    # Related absorption
    absorption_event: AbsorptionEvent
    candles_after_absorption: int

    # Shift indicators
    has_opposite_imbalance: bool
    opposite_imbalance_count: int
    delta_flipped: bool
    has_aggression: bool

    # Confidence score
    confidence: float

    def __repr__(self) -> str:
        return (
            f"Shift({self.shift_type} @ idx={self.candle_idx}, "
            f"after_absorption={self.candles_after_absorption}, conf={self.confidence:.2f})"
        )


class ShiftDetector:
    """
    Detect shift phases following absorption.

    BULLISH SHIFT (after sell absorption, for LONG):
    - ASK imbalance appears (buyers taking control)
    - Delta flips positive (optional)
    - Buy aggression visible

    BEARISH SHIFT (after buy absorption, for SHORT):
    - BID imbalance appears (sellers taking control)
    - Delta flips negative (optional)
    - Sell aggression visible
    """

    def __init__(self, config: Optional[ShiftConfig] = None):
        """
        Initialize shift detector.

        Args:
            config: Shift configuration
        """
        self.config = config or ShiftConfig()

    def detect_bullish_shift(
        self,
        candles: List[FootprintCandle],
        absorption: AbsorptionEvent,
    ) -> Optional[ShiftEvent]:
        """
        Detect BULLISH shift after sell absorption.

        This is the entry signal for LONG positions.

        Args:
            candles: List of footprint candles
            absorption: The absorption event to check after

        Returns:
            ShiftEvent if detected, None otherwise
        """
        if absorption.absorption_type != "SELL":
            return None

        abs_idx = absorption.candle_idx
        max_candles = self.config.max_candles_after_absorption

        # Look at candles after absorption
        for offset in range(1, max_candles + 1):
            shift_idx = abs_idx + offset

            if shift_idx >= len(candles):
                break

            shift_candle = candles[shift_idx]

            # Check shift conditions
            shift_event = self._check_bullish_shift(
                candle=shift_candle,
                candle_idx=shift_idx,
                absorption=absorption,
                offset=offset,
            )

            if shift_event:
                return shift_event

        return None

    def detect_bearish_shift(
        self,
        candles: List[FootprintCandle],
        absorption: AbsorptionEvent,
    ) -> Optional[ShiftEvent]:
        """
        Detect BEARISH shift after buy absorption.

        This is the entry signal for SHORT positions.

        Args:
            candles: List of footprint candles
            absorption: The absorption event to check after

        Returns:
            ShiftEvent if detected, None otherwise
        """
        if absorption.absorption_type != "BUY":
            return None

        abs_idx = absorption.candle_idx
        max_candles = self.config.max_candles_after_absorption

        for offset in range(1, max_candles + 1):
            shift_idx = abs_idx + offset

            if shift_idx >= len(candles):
                break

            shift_candle = candles[shift_idx]

            shift_event = self._check_bearish_shift(
                candle=shift_candle,
                candle_idx=shift_idx,
                absorption=absorption,
                offset=offset,
            )

            if shift_event:
                return shift_event

        return None

    def _check_bullish_shift(
        self,
        candle: FootprintCandle,
        candle_idx: int,
        absorption: AbsorptionEvent,
        offset: int,
    ) -> Optional[ShiftEvent]:
        """Check for bullish shift conditions."""
        conditions_met = 0
        total_conditions = 0

        # === CONDITION 1: ASK imbalance (opposite to absorption) ===
        has_opposite_imbalance = False
        opposite_count = 0

        if self.config.require_opposite_imbalance:
            total_conditions += 1
            if candle.ask_imbalances:
                opposite_count = len(candle.ask_imbalances)
                if opposite_count >= self.config.min_opposite_imbalances:
                    has_opposite_imbalance = True
                    conditions_met += 1

        # === CONDITION 2: Delta flip (optional) ===
        delta_flipped = False

        if self.config.require_delta_flip:
            total_conditions += 1
            # For bullish shift, delta should be positive
            if candle.delta > 0:
                delta_flipped = True
                conditions_met += 1
        else:
            # Still track it for confidence
            delta_flipped = candle.delta > 0

        # === CONDITION 3: Buy aggression ===
        has_aggression = False

        if self.config.require_buy_aggression:
            total_conditions += 1
            # Check if any level has ask > bid
            for level in candle.levels.values():
                if level.ask_volume > level.bid_volume:
                    has_aggression = True
                    break
            if has_aggression:
                conditions_met += 1
        else:
            # Still track it
            has_aggression = any(
                level.ask_volume > level.bid_volume
                for level in candle.levels.values()
            )

        # === Evaluate ===
        # Must meet all required conditions
        if total_conditions > 0 and conditions_met < total_conditions:
            return None

        # Calculate confidence
        confidence = self._calculate_shift_confidence(
            has_opposite_imbalance=has_opposite_imbalance,
            opposite_count=opposite_count,
            delta_flipped=delta_flipped,
            has_aggression=has_aggression,
            candles_after=offset,
            is_bullish_candle=candle.is_bullish,
        )

        # Require minimum confidence
        if confidence < 0.4:
            return None

        return ShiftEvent(
            candle_idx=candle_idx,
            candle=candle,
            shift_type="BULLISH",
            absorption_event=absorption,
            candles_after_absorption=offset,
            has_opposite_imbalance=has_opposite_imbalance,
            opposite_imbalance_count=opposite_count,
            delta_flipped=delta_flipped,
            has_aggression=has_aggression,
            confidence=confidence,
        )

    def _check_bearish_shift(
        self,
        candle: FootprintCandle,
        candle_idx: int,
        absorption: AbsorptionEvent,
        offset: int,
    ) -> Optional[ShiftEvent]:
        """Check for bearish shift conditions."""
        conditions_met = 0
        total_conditions = 0

        # === CONDITION 1: BID imbalance (opposite to absorption) ===
        has_opposite_imbalance = False
        opposite_count = 0

        if self.config.require_opposite_imbalance:
            total_conditions += 1
            if candle.bid_imbalances:
                opposite_count = len(candle.bid_imbalances)
                if opposite_count >= self.config.min_opposite_imbalances:
                    has_opposite_imbalance = True
                    conditions_met += 1

        # === CONDITION 2: Delta flip (optional) ===
        delta_flipped = False

        if self.config.require_delta_flip:
            total_conditions += 1
            # For bearish shift, delta should be negative
            if candle.delta < 0:
                delta_flipped = True
                conditions_met += 1
        else:
            delta_flipped = candle.delta < 0

        # === CONDITION 3: Sell aggression ===
        has_aggression = False

        if self.config.require_buy_aggression:  # Same config key, opposite check
            total_conditions += 1
            # Check if any level has bid > ask
            for level in candle.levels.values():
                if level.bid_volume > level.ask_volume:
                    has_aggression = True
                    break
            if has_aggression:
                conditions_met += 1
        else:
            has_aggression = any(
                level.bid_volume > level.ask_volume
                for level in candle.levels.values()
            )

        # === Evaluate ===
        if total_conditions > 0 and conditions_met < total_conditions:
            return None

        confidence = self._calculate_shift_confidence(
            has_opposite_imbalance=has_opposite_imbalance,
            opposite_count=opposite_count,
            delta_flipped=delta_flipped,
            has_aggression=has_aggression,
            candles_after=offset,
            is_bullish_candle=not candle.is_bearish,  # Bearish is good for short
        )

        if confidence < 0.4:
            return None

        return ShiftEvent(
            candle_idx=candle_idx,
            candle=candle,
            shift_type="BEARISH",
            absorption_event=absorption,
            candles_after_absorption=offset,
            has_opposite_imbalance=has_opposite_imbalance,
            opposite_imbalance_count=opposite_count,
            delta_flipped=delta_flipped,
            has_aggression=has_aggression,
            confidence=confidence,
        )

    def _calculate_shift_confidence(
        self,
        has_opposite_imbalance: bool,
        opposite_count: int,
        delta_flipped: bool,
        has_aggression: bool,
        candles_after: int,
        is_bullish_candle: bool,
    ) -> float:
        """Calculate confidence score for shift."""
        score = 0.0

        # Opposite imbalance (0.35)
        if has_opposite_imbalance:
            score += 0.25
            if opposite_count >= 2:
                score += 0.05
            if opposite_count >= 3:
                score += 0.05

        # Delta flip (0.25)
        if delta_flipped:
            score += 0.25

        # Aggression (0.2)
        if has_aggression:
            score += 0.2

        # Timing bonus (0.1) - earlier shift is better
        if candles_after == 1:
            score += 0.1
        elif candles_after == 2:
            score += 0.05

        # Candle direction bonus (0.1)
        if is_bullish_candle:
            score += 0.1

        return min(score, 1.0)

    def find_shifts_for_absorptions(
        self,
        candles: List[FootprintCandle],
        absorptions: List[AbsorptionEvent],
    ) -> List[ShiftEvent]:
        """
        Find shift events for all absorption events.

        Args:
            candles: List of footprint candles
            absorptions: List of detected absorptions

        Returns:
            List of shift events
        """
        shifts = []

        for absorption in absorptions:
            if absorption.absorption_type == "SELL":
                shift = self.detect_bullish_shift(candles, absorption)
            else:
                shift = self.detect_bearish_shift(candles, absorption)

            if shift:
                shifts.append(shift)

        logger.info(f"Found {len(shifts)} shift events from {len(absorptions)} absorptions")
        return shifts
