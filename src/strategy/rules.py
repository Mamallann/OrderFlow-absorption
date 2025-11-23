"""Strategy rules and validation."""

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional
import logging

from ..footprint.structures import FootprintCandle
from ..config import Config
from .signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class RuleValidation:
    """Result of rule validation."""
    passed: bool
    rule_name: str
    message: str
    details: dict = None


class StrategyRules:
    """
    Additional strategy rules and filters.

    These can be applied after signal generation
    to filter out low-quality setups.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def validate_signal(
        self,
        signal: Signal,
        candles: List[FootprintCandle],
    ) -> List[RuleValidation]:
        """
        Validate a signal against all rules.

        Args:
            signal: Signal to validate
            candles: List of candles for context

        Returns:
            List of validation results
        """
        validations = []

        # Check confidence
        validations.append(self._check_confidence(signal))

        # Check R:R
        validations.append(self._check_rr_ratio(signal))

        # Check SL distance
        validations.append(self._check_sl_distance(signal))

        # Check volume
        validations.append(self._check_volume(signal, candles))

        # Check trend alignment (optional)
        validations.append(self._check_trend(signal, candles))

        return validations

    def is_valid_signal(
        self,
        signal: Signal,
        candles: List[FootprintCandle],
    ) -> bool:
        """
        Check if signal passes all required rules.

        Args:
            signal: Signal to check
            candles: List of candles

        Returns:
            True if signal is valid
        """
        validations = self.validate_signal(signal, candles)
        return all(v.passed for v in validations)

    def _check_confidence(self, signal: Signal) -> RuleValidation:
        """Check confidence threshold."""
        min_conf = 0.5  # Could be configurable
        passed = signal.combined_confidence >= min_conf

        return RuleValidation(
            passed=passed,
            rule_name="confidence",
            message=f"Confidence {signal.combined_confidence:.2f} {'≥' if passed else '<'} {min_conf}",
            details={"value": signal.combined_confidence, "threshold": min_conf},
        )

    def _check_rr_ratio(self, signal: Signal) -> RuleValidation:
        """Check risk:reward ratio."""
        min_rr = 1.5  # Could be configurable
        passed = signal.rr_ratio >= min_rr

        return RuleValidation(
            passed=passed,
            rule_name="rr_ratio",
            message=f"R:R {signal.rr_ratio:.2f} {'≥' if passed else '<'} {min_rr}",
            details={"value": signal.rr_ratio, "threshold": min_rr},
        )

    def _check_sl_distance(self, signal: Signal) -> RuleValidation:
        """Check stop loss distance is reasonable."""
        max_sl_pct = self.config.stop_loss.max_sl_percent
        sl_pct = float(signal.risk_amount / signal.entry_price)
        passed = sl_pct <= max_sl_pct

        return RuleValidation(
            passed=passed,
            rule_name="sl_distance",
            message=f"SL distance {sl_pct:.2%} {'≤' if passed else '>'} {max_sl_pct:.2%}",
            details={"value": sl_pct, "threshold": max_sl_pct},
        )

    def _check_volume(
        self,
        signal: Signal,
        candles: List[FootprintCandle],
    ) -> RuleValidation:
        """Check volume is adequate."""
        # Get recent average volume
        idx = signal.entry_candle_idx
        lookback = min(20, idx)

        if lookback < 5:
            return RuleValidation(
                passed=True,
                rule_name="volume",
                message="Insufficient history for volume check",
            )

        recent_volumes = [
            float(candles[idx - i].volume)
            for i in range(1, lookback + 1)
        ]
        avg_volume = sum(recent_volumes) / len(recent_volumes)

        signal_volume = float(signal.shift_event.candle.volume)
        volume_ratio = signal_volume / avg_volume if avg_volume > 0 else 1

        # Volume should be at least 50% of average
        passed = volume_ratio >= 0.5

        return RuleValidation(
            passed=passed,
            rule_name="volume",
            message=f"Volume ratio {volume_ratio:.2f} {'≥' if passed else '<'} 0.5",
            details={"signal_volume": signal_volume, "avg_volume": avg_volume},
        )

    def _check_trend(
        self,
        signal: Signal,
        candles: List[FootprintCandle],
    ) -> RuleValidation:
        """
        Optional trend filter.

        For counter-trend trading (absorption), we actually
        want to trade against the immediate trend, so this
        always passes by default.
        """
        # Absorption trading is inherently counter-trend
        # Could add higher timeframe trend filter here
        return RuleValidation(
            passed=True,
            rule_name="trend",
            message="Trend filter disabled for absorption strategy",
        )

    def apply_filters(
        self,
        signals: List[Signal],
        candles: List[FootprintCandle],
    ) -> List[Signal]:
        """
        Apply all filters to signals.

        Args:
            signals: List of signals
            candles: List of candles

        Returns:
            Filtered signals
        """
        valid_signals = []

        for signal in signals:
            if self.is_valid_signal(signal, candles):
                valid_signals.append(signal)
            else:
                logger.debug(f"Signal {signal.signal_id} filtered out")

        logger.info(f"Rules filtered {len(signals)} → {len(valid_signals)} signals")
        return valid_signals
