"""Diagonal imbalance detection for footprint analysis."""

from decimal import Decimal
from typing import List, Optional, Tuple
import logging

from .structures import FootprintCandle, FootprintLevel, DiagonalImbalance, ImbalanceType

logger = logging.getLogger(__name__)


class ImbalanceDetector:
    """
    Detect diagonal imbalances in footprint candles.

    Diagonal imbalance detection (ATAS-style):
    - BID imbalance: bid_at_P > ratio × ask_at_(P + tick)
      Strong buying at P absorbing sells coming from above

    - ASK imbalance: ask_at_P > ratio × bid_at_(P - tick)
      Strong selling at P absorbing buys coming from below
    """

    def __init__(
        self,
        ratio_threshold: float = 3.0,
        min_volume: float = 1.0,
        consecutive_rows: int = 3,
        tick_size: float = 100.0,
    ):
        """
        Initialize imbalance detector.

        Args:
            ratio_threshold: Minimum ratio for imbalance (e.g., 3.0 = 300%)
            min_volume: Minimum volume to consider
            consecutive_rows: Required consecutive rows for valid imbalance
            tick_size: Price tick size for diagonal calculation
        """
        self.ratio_threshold = ratio_threshold
        self.min_volume = Decimal(str(min_volume))
        self.consecutive_rows = consecutive_rows
        self.tick_size = Decimal(str(tick_size))

    def detect_imbalances(
        self,
        candle: FootprintCandle,
    ) -> Tuple[List[DiagonalImbalance], List[DiagonalImbalance]]:
        """
        Detect all diagonal imbalances in a candle.

        Args:
            candle: FootprintCandle to analyze

        Returns:
            Tuple of (bid_imbalances, ask_imbalances)
        """
        bid_imbalances = self._detect_bid_imbalances(candle)
        ask_imbalances = self._detect_ask_imbalances(candle)

        # Group consecutive imbalances
        bid_imbalances = self._group_consecutive(bid_imbalances, ImbalanceType.BID)
        ask_imbalances = self._group_consecutive(ask_imbalances, ImbalanceType.ASK)

        return bid_imbalances, ask_imbalances

    def _detect_bid_imbalances(
        self,
        candle: FootprintCandle,
    ) -> List[DiagonalImbalance]:
        """
        Detect BID imbalances (buying absorption).

        BID imbalance at price P:
        bid_volume at P > ratio × ask_volume at (P + tick_size)

        This means buyers at P are absorbing sellers coming from above.
        """
        imbalances = []
        sorted_prices = sorted(candle.levels.keys())

        for i, price in enumerate(sorted_prices):
            level = candle.levels[price]

            # Skip if bid volume too low
            if level.bid_volume < self.min_volume:
                continue

            # Get diagonal level (one tick above)
            diagonal_price = price + self.tick_size

            if diagonal_price in candle.levels:
                diagonal_level = candle.levels[diagonal_price]
                diagonal_ask = diagonal_level.ask_volume
            else:
                diagonal_ask = Decimal("0")

            # Calculate ratio
            if diagonal_ask > 0:
                ratio = float(level.bid_volume / diagonal_ask)
            else:
                # If no diagonal ask volume, it's an infinite ratio
                # Only count if there's significant bid volume
                ratio = float("inf") if level.bid_volume >= self.min_volume else 0

            # Check threshold
            if ratio >= self.ratio_threshold:
                imbalance = DiagonalImbalance(
                    price=price,
                    type=ImbalanceType.BID,
                    ratio=ratio if ratio != float("inf") else 999.0,
                    volume=level.bid_volume,
                    diagonal_volume=diagonal_ask,
                    consecutive_count=1,
                )
                imbalances.append(imbalance)

        return imbalances

    def _detect_ask_imbalances(
        self,
        candle: FootprintCandle,
    ) -> List[DiagonalImbalance]:
        """
        Detect ASK imbalances (selling absorption).

        ASK imbalance at price P:
        ask_volume at P > ratio × bid_volume at (P - tick_size)

        This means sellers at P are absorbing buyers coming from below.
        """
        imbalances = []
        sorted_prices = sorted(candle.levels.keys())

        for i, price in enumerate(sorted_prices):
            level = candle.levels[price]

            # Skip if ask volume too low
            if level.ask_volume < self.min_volume:
                continue

            # Get diagonal level (one tick below)
            diagonal_price = price - self.tick_size

            if diagonal_price in candle.levels:
                diagonal_level = candle.levels[diagonal_price]
                diagonal_bid = diagonal_level.bid_volume
            else:
                diagonal_bid = Decimal("0")

            # Calculate ratio
            if diagonal_bid > 0:
                ratio = float(level.ask_volume / diagonal_bid)
            else:
                ratio = float("inf") if level.ask_volume >= self.min_volume else 0

            # Check threshold
            if ratio >= self.ratio_threshold:
                imbalance = DiagonalImbalance(
                    price=price,
                    type=ImbalanceType.ASK,
                    ratio=ratio if ratio != float("inf") else 999.0,
                    volume=level.ask_volume,
                    diagonal_volume=diagonal_bid,
                    consecutive_count=1,
                )
                imbalances.append(imbalance)

        return imbalances

    def _group_consecutive(
        self,
        imbalances: List[DiagonalImbalance],
        imb_type: ImbalanceType,
    ) -> List[DiagonalImbalance]:
        """
        Group consecutive imbalances and count them.

        Args:
            imbalances: List of individual imbalances
            imb_type: Type of imbalance

        Returns:
            List with consecutive_count updated
        """
        if not imbalances:
            return []

        # Sort by price
        sorted_imbs = sorted(imbalances, key=lambda x: x.price)

        # Group consecutive
        groups = []
        current_group = [sorted_imbs[0]]

        for i in range(1, len(sorted_imbs)):
            prev_price = sorted_imbs[i - 1].price
            curr_price = sorted_imbs[i].price

            # Check if consecutive (within one tick)
            if curr_price - prev_price == self.tick_size:
                current_group.append(sorted_imbs[i])
            else:
                groups.append(current_group)
                current_group = [sorted_imbs[i]]

        groups.append(current_group)

        # Update consecutive counts and filter
        result = []
        for group in groups:
            count = len(group)

            # Update all imbalances in group with consecutive count
            for imb in group:
                imb.consecutive_count = count

            # Only include groups that meet minimum consecutive requirement
            if count >= self.consecutive_rows:
                result.extend(group)

        return result

    def process_candles(
        self,
        candles: List[FootprintCandle],
    ) -> List[FootprintCandle]:
        """
        Process all candles and detect imbalances.

        Args:
            candles: List of footprint candles

        Returns:
            Same candles with imbalances populated
        """
        for candle in candles:
            bid_imbs, ask_imbs = self.detect_imbalances(candle)
            candle.bid_imbalances = bid_imbs
            candle.ask_imbalances = ask_imbs

        logger.info(f"Processed imbalances for {len(candles)} candles")
        return candles

    def has_strong_bid_imbalance(
        self,
        candle: FootprintCandle,
        min_consecutive: int = None,
        min_ratio: float = None,
    ) -> bool:
        """
        Check if candle has strong bid imbalance.

        Args:
            candle: Footprint candle
            min_consecutive: Override minimum consecutive rows
            min_ratio: Override minimum ratio

        Returns:
            True if strong bid imbalance exists
        """
        min_consecutive = min_consecutive or self.consecutive_rows
        min_ratio = min_ratio or self.ratio_threshold

        for imb in candle.bid_imbalances:
            if imb.consecutive_count >= min_consecutive and imb.ratio >= min_ratio:
                return True
        return False

    def has_strong_ask_imbalance(
        self,
        candle: FootprintCandle,
        min_consecutive: int = None,
        min_ratio: float = None,
    ) -> bool:
        """
        Check if candle has strong ask imbalance.

        Args:
            candle: Footprint candle
            min_consecutive: Override minimum consecutive rows
            min_ratio: Override minimum ratio

        Returns:
            True if strong ask imbalance exists
        """
        min_consecutive = min_consecutive or self.consecutive_rows
        min_ratio = min_ratio or self.ratio_threshold

        for imb in candle.ask_imbalances:
            if imb.consecutive_count >= min_consecutive and imb.ratio >= min_ratio:
                return True
        return False

    def get_imbalance_summary(
        self,
        candle: FootprintCandle,
    ) -> dict:
        """
        Get summary of imbalances in a candle.

        Args:
            candle: Footprint candle

        Returns:
            Dictionary with imbalance summary
        """
        return {
            "bid_imbalance_count": len(candle.bid_imbalances),
            "ask_imbalance_count": len(candle.ask_imbalances),
            "max_bid_consecutive": candle.max_consecutive_bid_imbalance,
            "max_ask_consecutive": candle.max_consecutive_ask_imbalance,
            "strongest_bid_ratio": (
                candle.strongest_bid_imbalance.ratio
                if candle.strongest_bid_imbalance
                else 0
            ),
            "strongest_ask_ratio": (
                candle.strongest_ask_imbalance.ratio
                if candle.strongest_ask_imbalance
                else 0
            ),
        }


def detect_imbalances(
    candles: List[FootprintCandle],
    ratio_threshold: float = 3.0,
    min_volume: float = 1.0,
    consecutive_rows: int = 3,
    tick_size: float = 100.0,
) -> List[FootprintCandle]:
    """
    Convenience function to detect imbalances in candles.

    Args:
        candles: List of footprint candles
        ratio_threshold: Minimum ratio for imbalance
        min_volume: Minimum volume to consider
        consecutive_rows: Required consecutive rows
        tick_size: Price tick size

    Returns:
        Candles with imbalances populated
    """
    detector = ImbalanceDetector(
        ratio_threshold=ratio_threshold,
        min_volume=min_volume,
        consecutive_rows=consecutive_rows,
        tick_size=tick_size,
    )
    return detector.process_candles(candles)
