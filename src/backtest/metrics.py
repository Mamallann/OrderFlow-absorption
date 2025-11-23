"""Performance metrics for backtest results."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Dict
import math
import logging

from .position import Trade
from .engine import BacktestState

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for backtest results.
    """
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_r_multiple: float = 0.0
    expectancy: float = 0.0

    # Best/Worst
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # R-multiple analysis
    total_r: float = 0.0
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0

    # Duration
    avg_trade_duration_hours: float = 0.0
    avg_winner_duration_hours: float = 0.0
    avg_loser_duration_hours: float = 0.0

    # MFE/MAE
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_mae_ratio: float = 0.0

    # By side
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    # Exit analysis
    exit_reasons: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_backtest_state(cls, state: BacktestState) -> "PerformanceMetrics":
        """Calculate metrics from backtest state."""
        metrics = cls()

        if not state.trades:
            return metrics

        trades = state.trades

        # Basic counts
        metrics.total_trades = len(trades)
        metrics.winning_trades = sum(1 for t in trades if t.is_winner)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades

        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # P&L
        pnls = [float(t.pnl) for t in trades]
        metrics.total_pnl = sum(pnls)

        initial_capital = float(state.capital)
        if initial_capital > 0:
            metrics.total_pnl_percent = metrics.total_pnl / initial_capital

        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p <= 0]

        metrics.gross_profit = sum(winning_pnls) if winning_pnls else 0
        metrics.gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0

        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        elif metrics.gross_profit > 0:
            metrics.profit_factor = float("inf")

        # Averages
        if winning_pnls:
            metrics.avg_win = sum(winning_pnls) / len(winning_pnls)
        if losing_pnls:
            metrics.avg_loss = abs(sum(losing_pnls) / len(losing_pnls))
        if pnls:
            metrics.avg_trade = sum(pnls) / len(pnls)

        # R-multiples
        r_multiples = [t.r_multiple for t in trades]
        if r_multiples:
            metrics.total_r = sum(r_multiples)
            metrics.avg_r_multiple = sum(r_multiples) / len(r_multiples)

        winning_r = [t.r_multiple for t in trades if t.is_winner]
        losing_r = [t.r_multiple for t in trades if not t.is_winner]

        if winning_r:
            metrics.avg_winner_r = sum(winning_r) / len(winning_r)
        if losing_r:
            metrics.avg_loser_r = sum(losing_r) / len(losing_r)

        # Expectancy
        if metrics.win_rate > 0 or metrics.avg_loss > 0:
            metrics.expectancy = (
                metrics.win_rate * metrics.avg_win -
                (1 - metrics.win_rate) * metrics.avg_loss
            )

        # Best/Worst
        if pnls:
            metrics.largest_win = max(pnls)
            metrics.largest_loss = min(pnls)

        # Consecutive wins/losses
        metrics.max_consecutive_wins = _max_consecutive(trades, True)
        metrics.max_consecutive_losses = _max_consecutive(trades, False)

        # Drawdown
        metrics.max_drawdown = float(state.max_drawdown)
        metrics.max_drawdown_pct = state.max_drawdown_pct

        # Risk-adjusted returns
        if state.equity_curve:
            returns = _calculate_returns(state.equity_curve)
            metrics.sharpe_ratio = _calculate_sharpe(returns)
            metrics.sortino_ratio = _calculate_sortino(returns)

            if metrics.max_drawdown_pct > 0:
                annual_return = metrics.total_pnl_percent  # Simplified
                metrics.calmar_ratio = annual_return / metrics.max_drawdown_pct

        # Duration
        durations = [t.duration_hours for t in trades]
        if durations:
            metrics.avg_trade_duration_hours = sum(durations) / len(durations)

        winner_durations = [t.duration_hours for t in trades if t.is_winner]
        loser_durations = [t.duration_hours for t in trades if not t.is_winner]

        if winner_durations:
            metrics.avg_winner_duration_hours = sum(winner_durations) / len(winner_durations)
        if loser_durations:
            metrics.avg_loser_duration_hours = sum(loser_durations) / len(loser_durations)

        # MFE/MAE
        mfes = [float(t.max_favorable_excursion) for t in trades]
        maes = [float(t.max_adverse_excursion) for t in trades]

        if mfes:
            metrics.avg_mfe = sum(mfes) / len(mfes)
        if maes:
            metrics.avg_mae = sum(maes) / len(maes)

        if metrics.avg_mae > 0:
            metrics.mfe_mae_ratio = metrics.avg_mfe / metrics.avg_mae

        # By side
        long_trades = [t for t in trades if t.side == "LONG"]
        short_trades = [t for t in trades if t.side == "SHORT"]

        metrics.long_trades = len(long_trades)
        metrics.short_trades = len(short_trades)

        if long_trades:
            long_winners = sum(1 for t in long_trades if t.is_winner)
            metrics.long_win_rate = long_winners / len(long_trades)

        if short_trades:
            short_winners = sum(1 for t in short_trades if t.is_winner)
            metrics.short_win_rate = short_winners / len(short_trades)

        # Exit reasons
        for trade in trades:
            reason = trade.exit_reason
            metrics.exit_reasons[reason] = metrics.exit_reasons.get(reason, 0) + 1

        return metrics

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_percent": round(self.total_pnl_percent * 100, 2),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float("inf") else "∞",
            "avg_r_multiple": round(self.avg_r_multiple, 2),
            "expectancy": round(self.expectancy, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "avg_winner_r": round(self.avg_winner_r, 2),
            "avg_loser_r": round(self.avg_loser_r, 2),
            "long_win_rate": round(self.long_win_rate * 100, 2),
            "short_win_rate": round(self.short_win_rate * 100, 2),
            "exit_reasons": self.exit_reasons,
        }

    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 60)
        print("         ORDERFLOW ABSORPTION BACKTEST REPORT")
        print("=" * 60)

        print("\nPERFORMANCE METRICS")
        print("-" * 40)
        print(f"Total Trades:        {self.total_trades}")
        print(f"Win Rate:            {self.win_rate:.1%}")
        print(f"Profit Factor:       {self.profit_factor:.2f}")
        print(f"Avg R Multiple:      {self.avg_r_multiple:.2f}R")
        print(f"Max Drawdown:        {self.max_drawdown_pct:.1%}")
        print(f"Sharpe Ratio:        {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {self.sortino_ratio:.2f}")

        print("\nTRADE BREAKDOWN")
        print("-" * 40)
        print(f"Long Trades:         {self.long_trades} ({self.long_win_rate:.1%} win)")
        print(f"Short Trades:        {self.short_trades} ({self.short_win_rate:.1%} win)")
        print(f"Avg Winner:          +{self.avg_winner_r:.2f}R")
        print(f"Avg Loser:           {self.avg_loser_r:.2f}R")
        print(f"Largest Win:         ${self.largest_win:.2f}")
        print(f"Largest Loss:        ${self.largest_loss:.2f}")

        print("\nEXIT ANALYSIS")
        print("-" * 40)
        for reason, count in sorted(self.exit_reasons.items()):
            pct = count / self.total_trades * 100 if self.total_trades > 0 else 0
            print(f"{reason}:".ljust(20) + f"{count} ({pct:.1f}%)")

        print("\n" + "=" * 60)


def _max_consecutive(trades: List[Trade], winners: bool) -> int:
    """Calculate max consecutive wins or losses."""
    max_streak = 0
    current_streak = 0

    for trade in trades:
        if trade.is_winner == winners:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def _calculate_returns(equity_curve: List[dict]) -> List[float]:
    """Calculate returns from equity curve."""
    returns = []
    for i in range(1, len(equity_curve)):
        prev_equity = equity_curve[i - 1]["equity"]
        curr_equity = equity_curve[i]["equity"]
        if prev_equity > 0:
            ret = (curr_equity - prev_equity) / prev_equity
            returns.append(ret)
    return returns


def _calculate_sharpe(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if not returns or len(returns) < 2:
        return 0.0

    avg_return = sum(returns) / len(returns)
    std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))

    if std_dev == 0:
        return 0.0

    # Annualize (assuming daily returns)
    sharpe = (avg_return - risk_free_rate) / std_dev * math.sqrt(252)
    return sharpe


def _calculate_sortino(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation only)."""
    if not returns or len(returns) < 2:
        return 0.0

    avg_return = sum(returns) / len(returns)
    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        return float("inf") if avg_return > 0 else 0.0

    downside_dev = math.sqrt(sum(r ** 2 for r in negative_returns) / len(negative_returns))

    if downside_dev == 0:
        return 0.0

    sortino = (avg_return - risk_free_rate) / downside_dev * math.sqrt(252)
    return sortino


def calculate_metrics(state: BacktestState) -> PerformanceMetrics:
    """Convenience function to calculate metrics."""
    return PerformanceMetrics.from_backtest_state(state)
