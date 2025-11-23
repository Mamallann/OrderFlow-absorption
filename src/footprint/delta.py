"""Delta calculations for footprint analysis."""

from decimal import Decimal
from typing import List, Optional, Tuple
import logging

from .structures import FootprintCandle

logger = logging.getLogger(__name__)


class DeltaCalculator:
    """
    Calculate and analyze delta metrics for footprint candles.

    Delta = Ask Volume - Bid Volume
    Positive delta = net buying pressure
    Negative delta = net selling pressure
    """

    def __init__(self):
        pass

    def compute_cumulative_delta(
        self,
        candles: List[FootprintCandle],
        reset_daily: bool = False,
    ) -> List[FootprintCandle]:
        """
        Compute cumulative delta for candles.

        Args:
            candles: List of footprint candles
            reset_daily: Reset cumulative delta at start of each day

        Returns:
            Same candles with cum_delta updated
        """
        cum_delta = Decimal("0")
        current_day = None

        for candle in candles:
            # Check for day reset
            if reset_daily:
                # Extract day from timestamp (ms -> day)
                day = candle.timestamp // (24 * 60 * 60 * 1000)
                if current_day is not None and day != current_day:
                    cum_delta = Decimal("0")
                current_day = day

            cum_delta += candle.delta
            candle.cum_delta = cum_delta

        return candles

    def compute_delta_divergence(
        self,
        candles: List[FootprintCandle],
        idx: int,
        lookback: int = 2,
    ) -> Tuple[bool, float]:
        """
        Check for delta divergence at a candle.

        For bearish candle with positive or increasing delta = bullish divergence
        For bullish candle with negative or decreasing delta = bearish divergence

        Args:
            candles: List of footprint candles
            idx: Index of candle to check
            lookback: Number of previous candles to compare

        Returns:
            Tuple of (has_divergence, divergence_strength)
        """
        if idx < lookback:
            return False, 0.0

        current = candles[idx]
        prev_deltas = [candles[idx - i - 1].delta for i in range(lookback)]

        # Check price direction vs delta direction
        price_bearish = current.close < current.open
        price_bullish = current.close > current.open

        # Current delta is higher than all previous (potential bullish divergence)
        delta_increasing = all(current.delta > d for d in prev_deltas)

        # Current delta is lower than all previous (potential bearish divergence)
        delta_decreasing = all(current.delta < d for d in prev_deltas)

        # Bullish divergence: price down but delta up/positive
        if price_bearish and (delta_increasing or current.delta > 0):
            avg_prev = sum(prev_deltas) / len(prev_deltas) if prev_deltas else Decimal("1")
            if avg_prev != 0:
                strength = float(abs(current.delta - avg_prev) / abs(avg_prev))
            else:
                strength = float(abs(current.delta))
            return True, strength

        # Bearish divergence: price up but delta down/negative
        if price_bullish and (delta_decreasing or current.delta < 0):
            avg_prev = sum(prev_deltas) / len(prev_deltas) if prev_deltas else Decimal("1")
            if avg_prev != 0:
                strength = float(abs(current.delta - avg_prev) / abs(avg_prev))
            else:
                strength = float(abs(current.delta))
            return True, strength

        return False, 0.0

    def is_delta_larger_than_previous(
        self,
        candles: List[FootprintCandle],
        idx: int,
        lookback: int = 2,
        comparison: str = "both",
        use_absolute: bool = True,
    ) -> bool:
        """
        Check if current candle's delta is larger than previous candles.

        This is used for absorption detection where we look for
        strong selling (negative delta) or buying (positive delta).

        Args:
            candles: List of footprint candles
            idx: Index of candle to check
            lookback: Number of previous candles to compare
            comparison: "both" (all), "any" (at least one), "average"
            use_absolute: Compare absolute values

        Returns:
            True if delta condition is met
        """
        if idx < lookback:
            return False

        current = candles[idx]
        current_delta = abs(current.delta) if use_absolute else current.delta

        if comparison == "both":
            # Must be larger than ALL previous
            for i in range(1, lookback + 1):
                prev_delta = candles[idx - i].delta
                if use_absolute:
                    prev_delta = abs(prev_delta)
                if current_delta <= prev_delta:
                    return False
            return True

        elif comparison == "any":
            # Must be larger than AT LEAST ONE previous
            for i in range(1, lookback + 1):
                prev_delta = candles[idx - i].delta
                if use_absolute:
                    prev_delta = abs(prev_delta)
                if current_delta > prev_delta:
                    return True
            return False

        elif comparison == "average":
            # Must be larger than average of previous
            prev_deltas = [candles[idx - i].delta for i in range(1, lookback + 1)]
            if use_absolute:
                prev_deltas = [abs(d) for d in prev_deltas]
            avg_prev = sum(prev_deltas) / len(prev_deltas)
            return current_delta > avg_prev

        else:
            raise ValueError(f"Unknown comparison mode: {comparison}")

    def get_delta_stats(
        self,
        candles: List[FootprintCandle],
        window: int = 20,
    ) -> dict:
        """
        Get delta statistics over a window.

        Args:
            candles: List of footprint candles
            window: Lookback window

        Returns:
            Dictionary with delta statistics
        """
        if len(candles) < window:
            window = len(candles)

        recent = candles[-window:]
        deltas = [float(c.delta) for c in recent]

        import numpy as np
        deltas_arr = np.array(deltas)

        return {
            "mean": float(np.mean(deltas_arr)),
            "std": float(np.std(deltas_arr)),
            "min": float(np.min(deltas_arr)),
            "max": float(np.max(deltas_arr)),
            "positive_count": int(np.sum(deltas_arr > 0)),
            "negative_count": int(np.sum(deltas_arr < 0)),
            "sum": float(np.sum(deltas_arr)),
        }
