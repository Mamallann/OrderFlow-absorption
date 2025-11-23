"""Absorption detection for order flow strategy."""

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Literal
import logging

from ..footprint.structures import FootprintCandle
from ..footprint.imbalance import ImbalanceDetector
from ..footprint.delta import DeltaCalculator
from ..config import AbsorptionConfig

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionEvent:
    """
    Detected absorption event.

    Represents a candle where absorption was detected,
    along with all relevant metrics and context.
    """
    candle_idx: int
    candle: FootprintCandle
    absorption_type: Literal["BUY", "SELL"]  # BUY = buying absorption, SELL = selling absorption

    # Imbalance data
    imbalance_count: int
    max_consecutive: int
    max_ratio: float

    # Delta data
    delta: Decimal
    delta_stronger_than_previous: bool

    # Price action
    price_held: bool
    wick_rejection: bool

    # Confidence score (0-1)
    confidence: float

    def __repr__(self) -> str:
        return (
            f"Absorption({self.absorption_type} @ idx={self.candle_idx}, "
            f"consec={self.max_consecutive}, conf={self.confidence:.2f})"
        )


class AbsorptionDetector:
    """
    Detect absorption phases in footprint data.

    SELL ABSORPTION (for LONG setup):
    - Huge BID imbalances (buyers absorbing sellers)
    - Large sell delta (selling pressure)
    - Price does NOT break lower
    - Wick rejection at lows

    BUY ABSORPTION (for SHORT setup):
    - Huge ASK imbalances (sellers absorbing buyers)
    - Large buy delta (buying pressure)
    - Price does NOT break higher
    - Wick rejection at highs
    """

    def __init__(
        self,
        config: Optional[AbsorptionConfig] = None,
        imbalance_detector: Optional[ImbalanceDetector] = None,
        delta_calculator: Optional[DeltaCalculator] = None,
    ):
        """
        Initialize absorption detector.

        Args:
            config: Absorption configuration
            imbalance_detector: Imbalance detector instance
            delta_calculator: Delta calculator instance
        """
        self.config = config or AbsorptionConfig()
        self.imbalance_detector = imbalance_detector or ImbalanceDetector()
        self.delta_calculator = delta_calculator or DeltaCalculator()

    def detect_sell_absorption(
        self,
        candles: List[FootprintCandle],
        idx: int,
    ) -> Optional[AbsorptionEvent]:
        """
        Detect SELL absorption at a candle (setup for LONG).

        Conditions:
        1. Huge BID imbalances (3+ consecutive rows)
        2. Delta shows selling pressure (larger than previous)
        3. Price does NOT break lower (or wick rejection)

        Args:
            candles: List of footprint candles
            idx: Index of candle to check

        Returns:
            AbsorptionEvent if detected, None otherwise
        """
        if idx < self.config.delta_lookback:
            return None

        candle = candles[idx]

        # === RULE 1: Check for strong BID imbalances ===
        has_bid_imbalance = self._check_bid_imbalance(candle)
        if not has_bid_imbalance:
            return None

        # Get imbalance metrics
        max_consecutive = candle.max_consecutive_bid_imbalance
        max_ratio = (
            candle.strongest_bid_imbalance.ratio
            if candle.strongest_bid_imbalance
            else 0
        )

        # === RULE 2: Check delta condition ===
        # For sell absorption, we want negative delta (selling pressure)
        # that is larger in magnitude than previous candles
        delta_condition = self._check_delta_condition(
            candles, idx, check_negative=True
        )

        # === RULE 3: Check price holds ===
        price_held, wick_rejection = self._check_price_holds_low(candles, idx)

        # Must have price hold OR wick rejection
        price_condition = price_held or wick_rejection

        if not price_condition:
            return None

        # === Calculate confidence ===
        confidence = self._calculate_confidence(
            has_imbalance=True,
            max_consecutive=max_consecutive,
            max_ratio=max_ratio,
            delta_condition=delta_condition,
            price_held=price_held,
            wick_rejection=wick_rejection,
        )

        # Require minimum confidence
        if confidence < 0.5:
            return None

        return AbsorptionEvent(
            candle_idx=idx,
            candle=candle,
            absorption_type="SELL",
            imbalance_count=len(candle.bid_imbalances),
            max_consecutive=max_consecutive,
            max_ratio=max_ratio,
            delta=candle.delta,
            delta_stronger_than_previous=delta_condition,
            price_held=price_held,
            wick_rejection=wick_rejection,
            confidence=confidence,
        )

    def detect_buy_absorption(
        self,
        candles: List[FootprintCandle],
        idx: int,
    ) -> Optional[AbsorptionEvent]:
        """
        Detect BUY absorption at a candle (setup for SHORT).

        Conditions:
        1. Huge ASK imbalances (3+ consecutive rows)
        2. Delta shows buying pressure (larger than previous)
        3. Price does NOT break higher (or wick rejection)

        Args:
            candles: List of footprint candles
            idx: Index of candle to check

        Returns:
            AbsorptionEvent if detected, None otherwise
        """
        if idx < self.config.delta_lookback:
            return None

        candle = candles[idx]

        # === RULE 1: Check for strong ASK imbalances ===
        has_ask_imbalance = self._check_ask_imbalance(candle)
        if not has_ask_imbalance:
            return None

        # Get imbalance metrics
        max_consecutive = candle.max_consecutive_ask_imbalance
        max_ratio = (
            candle.strongest_ask_imbalance.ratio
            if candle.strongest_ask_imbalance
            else 0
        )

        # === RULE 2: Check delta condition ===
        # For buy absorption, we want positive delta (buying pressure)
        # that is larger in magnitude than previous candles
        delta_condition = self._check_delta_condition(
            candles, idx, check_negative=False
        )

        # === RULE 3: Check price holds ===
        price_held, wick_rejection = self._check_price_holds_high(candles, idx)

        # Must have price hold OR wick rejection
        price_condition = price_held or wick_rejection

        if not price_condition:
            return None

        # === Calculate confidence ===
        confidence = self._calculate_confidence(
            has_imbalance=True,
            max_consecutive=max_consecutive,
            max_ratio=max_ratio,
            delta_condition=delta_condition,
            price_held=price_held,
            wick_rejection=wick_rejection,
        )

        if confidence < 0.5:
            return None

        return AbsorptionEvent(
            candle_idx=idx,
            candle=candle,
            absorption_type="BUY",
            imbalance_count=len(candle.ask_imbalances),
            max_consecutive=max_consecutive,
            max_ratio=max_ratio,
            delta=candle.delta,
            delta_stronger_than_previous=delta_condition,
            price_held=price_held,
            wick_rejection=wick_rejection,
            confidence=confidence,
        )

    def _check_bid_imbalance(self, candle: FootprintCandle) -> bool:
        """Check if candle has required bid imbalance."""
        if not candle.bid_imbalances:
            return False

        # Check for consecutive rows meeting threshold
        for imb in candle.bid_imbalances:
            if (
                imb.consecutive_count >= self.config.min_imbalance_rows
                and imb.ratio >= self.config.min_imbalance_ratio
            ):
                return True
        return False

    def _check_ask_imbalance(self, candle: FootprintCandle) -> bool:
        """Check if candle has required ask imbalance."""
        if not candle.ask_imbalances:
            return False

        for imb in candle.ask_imbalances:
            if (
                imb.consecutive_count >= self.config.min_imbalance_rows
                and imb.ratio >= self.config.min_imbalance_ratio
            ):
                return True
        return False

    def _check_delta_condition(
        self,
        candles: List[FootprintCandle],
        idx: int,
        check_negative: bool,
    ) -> bool:
        """
        Check if delta meets absorption criteria.

        Args:
            candles: List of candles
            idx: Current index
            check_negative: True for sell absorption (negative delta)

        Returns:
            True if delta condition is met
        """
        current = candles[idx]
        lookback = self.config.delta_lookback
        comparison = self.config.delta_comparison

        # Get previous deltas
        prev_deltas = [
            candles[idx - i].delta
            for i in range(1, lookback + 1)
            if idx - i >= 0
        ]

        if not prev_deltas:
            return True  # No previous data to compare

        current_delta = current.delta

        # For sell absorption, delta should be negative (selling pressure)
        # For buy absorption, delta should be positive (buying pressure)
        if check_negative and current_delta >= 0:
            # Allow slightly positive delta if it's still stronger
            pass  # Don't return False immediately
        elif not check_negative and current_delta <= 0:
            pass

        # Compare absolute values
        abs_current = abs(current_delta)

        if comparison == "both":
            # Must be larger than ALL previous
            return all(abs_current > abs(d) for d in prev_deltas)

        elif comparison == "any":
            # Must be larger than at least one
            return any(abs_current > abs(d) for d in prev_deltas)

        elif comparison == "average":
            avg = sum(abs(d) for d in prev_deltas) / len(prev_deltas)
            return abs_current > avg * self.config.delta_multiplier

        return False

    def _check_price_holds_low(
        self,
        candles: List[FootprintCandle],
        idx: int,
    ) -> tuple[bool, bool]:
        """
        Check if price holds at lows (for sell absorption).

        Returns:
            Tuple of (price_held, wick_rejection)
        """
        current = candles[idx]
        prev = candles[idx - 1] if idx > 0 else None

        price_held = False
        wick_rejection = False

        mode = self.config.price_hold_mode

        if prev:
            # Price hold: current low >= previous low
            if self.config.allow_equal_lows:
                price_held = current.low >= prev.low
            else:
                price_held = current.low > prev.low

        # Wick rejection: lower wick is small relative to body
        if current.range > 0:
            lower_wick_ratio = float(current.lower_wick / current.range)
            wick_rejection = lower_wick_ratio <= self.config.wick_rejection_ratio

            # Also check close is above midpoint (rejection)
            midpoint = (current.high + current.low) / 2
            if current.close > midpoint:
                wick_rejection = True

        if mode == "no_break":
            return price_held, False
        elif mode == "wick_rejection":
            return False, wick_rejection
        else:  # both
            return price_held, wick_rejection

    def _check_price_holds_high(
        self,
        candles: List[FootprintCandle],
        idx: int,
    ) -> tuple[bool, bool]:
        """
        Check if price holds at highs (for buy absorption).

        Returns:
            Tuple of (price_held, wick_rejection)
        """
        current = candles[idx]
        prev = candles[idx - 1] if idx > 0 else None

        price_held = False
        wick_rejection = False

        mode = self.config.price_hold_mode

        if prev:
            # Price hold: current high <= previous high
            if self.config.allow_equal_lows:
                price_held = current.high <= prev.high
            else:
                price_held = current.high < prev.high

        # Wick rejection: upper wick is small relative to body
        if current.range > 0:
            upper_wick_ratio = float(current.upper_wick / current.range)
            wick_rejection = upper_wick_ratio <= self.config.wick_rejection_ratio

            # Also check close is below midpoint (rejection)
            midpoint = (current.high + current.low) / 2
            if current.close < midpoint:
                wick_rejection = True

        if mode == "no_break":
            return price_held, False
        elif mode == "wick_rejection":
            return False, wick_rejection
        else:
            return price_held, wick_rejection

    def _calculate_confidence(
        self,
        has_imbalance: bool,
        max_consecutive: int,
        max_ratio: float,
        delta_condition: bool,
        price_held: bool,
        wick_rejection: bool,
    ) -> float:
        """Calculate confidence score for absorption."""
        score = 0.0

        # Imbalance quality (up to 0.4)
        if has_imbalance:
            score += 0.2
            # Bonus for more consecutive rows
            if max_consecutive >= 4:
                score += 0.1
            if max_consecutive >= 5:
                score += 0.1
            # Bonus for high ratio
            if max_ratio >= 5.0:
                score += 0.05
            if max_ratio >= 10.0:
                score += 0.05

        # Delta condition (0.3)
        if delta_condition:
            score += 0.3

        # Price action (0.3)
        if price_held:
            score += 0.15
        if wick_rejection:
            score += 0.15

        return min(score, 1.0)

    def scan_for_absorption(
        self,
        candles: List[FootprintCandle],
        direction: Literal["LONG", "SHORT", "BOTH"] = "BOTH",
    ) -> List[AbsorptionEvent]:
        """
        Scan all candles for absorption events.

        Args:
            candles: List of footprint candles
            direction: "LONG" (sell absorption), "SHORT" (buy absorption), or "BOTH"

        Returns:
            List of detected absorption events
        """
        events = []

        for idx in range(self.config.delta_lookback, len(candles)):
            if direction in ("LONG", "BOTH"):
                event = self.detect_sell_absorption(candles, idx)
                if event:
                    events.append(event)

            if direction in ("SHORT", "BOTH"):
                event = self.detect_buy_absorption(candles, idx)
                if event:
                    events.append(event)

        logger.info(f"Found {len(events)} absorption events")
        return events
