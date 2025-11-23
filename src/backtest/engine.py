"""Main backtesting engine."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Dict
from datetime import datetime
import logging

from ..footprint.structures import FootprintCandle
from ..strategy.signals import Signal, SignalGenerator
from ..config import Config, BacktestConfig
from .position import Position, Trade, PositionStatus, ExitReason
from .risk import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestState:
    """Current state of the backtest."""
    capital: Decimal
    equity: Decimal
    open_positions: List[Position] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    # Tracking
    equity_curve: List[dict] = field(default_factory=list)
    peak_equity: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0

    # Counters
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    trades_today: int = 0
    current_day: Optional[int] = None

    # Position ID counter
    _position_id: int = 0
    _trade_id: int = 0

    def next_position_id(self) -> int:
        self._position_id += 1
        return self._position_id

    def next_trade_id(self) -> int:
        self._trade_id += 1
        return self._trade_id


class BacktestEngine:
    """
    Main backtesting engine.

    Simulates trading based on signals with:
    - Position management
    - Risk management (SL/TP/trailing)
    - Commission and slippage
    - Trade limits
    - Equity tracking
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        signal_generator: Optional[SignalGenerator] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Full configuration
            signal_generator: Signal generator instance
            risk_manager: Risk manager instance
        """
        self.config = config or Config()
        self.backtest_config = self.config.backtest

        self.signal_generator = signal_generator or SignalGenerator(config=self.config)
        self.risk_manager = risk_manager or RiskManager(
            tp_config=self.config.take_profit,
            sl_config=self.config.stop_loss,
        )

        self.state: Optional[BacktestState] = None

    def run(
        self,
        candles: List[FootprintCandle],
        signals: Optional[List[Signal]] = None,
        symbol: str = "BTCUSDT",
    ) -> BacktestState:
        """
        Run backtest on candles.

        Args:
            candles: List of footprint candles
            signals: Pre-generated signals (or generate from candles)
            symbol: Trading symbol

        Returns:
            BacktestState with results
        """
        # Initialize state
        initial_capital = Decimal(str(self.backtest_config.initial_capital))
        self.state = BacktestState(
            capital=initial_capital,
            equity=initial_capital,
            peak_equity=initial_capital,
        )

        # Generate signals if not provided
        if signals is None:
            logger.info("Generating signals from candles...")
            signals = self.signal_generator.generate_signals(candles, symbol)

        if not signals:
            logger.warning("No signals to backtest")
            return self.state

        logger.info(f"Running backtest with {len(signals)} signals on {len(candles)} candles")

        # Create signal lookup by candle index
        signal_map: Dict[int, Signal] = {}
        for signal in signals:
            idx = signal.entry_candle_idx
            if idx not in signal_map:
                signal_map[idx] = signal

        # Main backtest loop
        for idx, candle in enumerate(candles):
            # Update day tracking
            self._update_day_tracking(candle)

            # Update existing positions
            self._update_positions(candle)

            # Check for new signals
            if idx in signal_map:
                signal = signal_map[idx]
                self._process_signal(signal, candle)

            # Update equity curve
            self._update_equity(candle)

        # Close any remaining positions
        if candles:
            self._close_all_positions(candles[-1], ExitReason.END_OF_DATA)

        # Generate trade records
        self._generate_trade_records()

        logger.info(
            f"Backtest complete: {self.state.total_trades} trades, "
            f"{self.state.winning_trades} winners, "
            f"{self.state.losing_trades} losers"
        )

        return self.state

    def _update_day_tracking(self, candle: FootprintCandle):
        """Update daily trade counter."""
        day = candle.timestamp // (24 * 60 * 60 * 1000)

        if self.state.current_day != day:
            self.state.current_day = day
            self.state.trades_today = 0

    def _update_positions(self, candle: FootprintCandle):
        """Update all open positions with new candle."""
        positions_to_close = []

        for position in self.state.open_positions:
            # Check for exit
            should_exit, exit_price, exit_reason = self.risk_manager.check_exit(
                position, candle
            )

            if should_exit and exit_price and exit_reason:
                # Apply slippage
                exit_price = self._apply_slippage(exit_price, position.side, is_exit=True)

                # Close position
                position.close(exit_price, candle.timestamp, exit_reason)
                positions_to_close.append(position)

        # Move closed positions
        for position in positions_to_close:
            self.state.open_positions.remove(position)
            self.state.closed_positions.append(position)

            # Apply commission
            commission = self._calculate_commission(position)
            position.realized_pnl -= commission

            # Update counters
            self.state.total_trades += 1
            if position.total_pnl > 0:
                self.state.winning_trades += 1
            else:
                self.state.losing_trades += 1

    def _process_signal(self, signal: Signal, candle: FootprintCandle):
        """Process a trading signal."""
        # Check trade limits
        if not self._can_open_trade(signal):
            logger.debug(f"Cannot open trade: limits reached")
            return

        # Check for conflicting positions
        if self._has_conflicting_position(signal):
            logger.debug(f"Conflicting position exists")
            return

        # Calculate position size
        entry_price = self._apply_slippage(signal.entry_price, signal.side, is_exit=False)

        quantity = self.risk_manager.calculate_position_size(
            capital=self.state.equity,
            risk_percent=self.backtest_config.risk_per_trade,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            mode=self.backtest_config.position_size_mode,
            fixed_size=self.backtest_config.fixed_size,
            position_percent=self.backtest_config.position_percent,
        )

        if quantity <= 0:
            logger.warning(f"Invalid position size: {quantity}")
            return

        # Create position
        position = Position(
            position_id=self.state.next_position_id(),
            signal=signal,
            symbol=signal.symbol,
            side=signal.side,
            entry_price=entry_price,
            entry_time=candle.timestamp,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            initial_stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        # Apply entry commission
        entry_commission = self._calculate_entry_commission(position)
        position.realized_pnl -= entry_commission

        self.state.open_positions.append(position)
        self.state.trades_today += 1

        logger.info(
            f"Opened {signal.side} position #{position.position_id} "
            f"@ {entry_price}, SL={signal.stop_loss}, TP={signal.take_profit}"
        )

    def _can_open_trade(self, signal: Signal) -> bool:
        """Check if we can open a new trade."""
        # Check daily limit
        if self.state.trades_today >= self.backtest_config.max_trades_per_day:
            return False

        # Check concurrent positions
        if len(self.state.open_positions) >= self.backtest_config.max_concurrent_trades:
            return False

        return True

    def _has_conflicting_position(self, signal: Signal) -> bool:
        """Check if there's a conflicting open position."""
        for position in self.state.open_positions:
            if position.symbol == signal.symbol:
                # Already have a position in this symbol
                return True
        return False

    def _apply_slippage(
        self,
        price: Decimal,
        side: str,
        is_exit: bool,
    ) -> Decimal:
        """Apply slippage to price."""
        slippage = Decimal(str(self.backtest_config.slippage))

        if side == "LONG":
            if is_exit:
                # Selling - slip down
                return price * (1 - slippage)
            else:
                # Buying - slip up
                return price * (1 + slippage)
        else:
            if is_exit:
                # Buying to cover - slip up
                return price * (1 + slippage)
            else:
                # Selling short - slip down
                return price * (1 - slippage)

    def _calculate_commission(self, position: Position) -> Decimal:
        """Calculate round-trip commission."""
        commission_rate = Decimal(str(self.backtest_config.commission))
        notional = position.entry_price * position.quantity

        # Entry + Exit
        entry_comm = notional * commission_rate
        exit_comm = (position.exit_price or position.entry_price) * position.quantity * commission_rate

        return entry_comm + exit_comm

    def _calculate_entry_commission(self, position: Position) -> Decimal:
        """Calculate entry commission only."""
        commission_rate = Decimal(str(self.backtest_config.commission))
        notional = position.entry_price * position.quantity
        return notional * commission_rate

    def _update_equity(self, candle: FootprintCandle):
        """Update equity curve and drawdown."""
        # Calculate current equity
        unrealized_pnl = sum(
            p.unrealized_pnl for p in self.state.open_positions
        )
        realized_pnl = sum(
            p.realized_pnl for p in self.state.closed_positions
        )

        self.state.equity = self.state.capital + realized_pnl + unrealized_pnl

        # Update peak and drawdown
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        drawdown = self.state.peak_equity - self.state.equity
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown
            if self.state.peak_equity > 0:
                self.state.max_drawdown_pct = float(drawdown / self.state.peak_equity)

        # Record equity curve point
        self.state.equity_curve.append({
            "timestamp": candle.timestamp,
            "equity": float(self.state.equity),
            "drawdown": float(drawdown),
            "open_positions": len(self.state.open_positions),
        })

    def _close_all_positions(self, candle: FootprintCandle, reason: ExitReason):
        """Close all open positions."""
        for position in list(self.state.open_positions):
            exit_price = self._apply_slippage(candle.close, position.side, is_exit=True)
            position.close(exit_price, candle.timestamp, reason)
            self.state.open_positions.remove(position)
            self.state.closed_positions.append(position)

            commission = self._calculate_commission(position)
            position.realized_pnl -= commission

            self.state.total_trades += 1
            if position.total_pnl > 0:
                self.state.winning_trades += 1
            else:
                self.state.losing_trades += 1

    def _generate_trade_records(self):
        """Generate trade records from closed positions."""
        for position in self.state.closed_positions:
            trade = Trade.from_position(position, self.state.next_trade_id())
            self.state.trades.append(trade)


def run_backtest(
    candles: List[FootprintCandle],
    config: Optional[Config] = None,
    signals: Optional[List[Signal]] = None,
    symbol: str = "BTCUSDT",
) -> BacktestState:
    """
    Convenience function to run backtest.

    Args:
        candles: List of footprint candles
        config: Configuration
        signals: Pre-generated signals (optional)
        symbol: Trading symbol

    Returns:
        BacktestState with results
    """
    engine = BacktestEngine(config=config)
    return engine.run(candles, signals, symbol)
