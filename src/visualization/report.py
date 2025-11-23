"""Report generation for backtest results."""

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import json
import logging

import pandas as pd

from ..backtest.engine import BacktestState
from ..backtest.metrics import PerformanceMetrics
from ..config import ReportingConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate backtest reports in various formats.
    """

    def __init__(self, config: Optional[ReportingConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Reporting configuration
        """
        self.config = config or ReportingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        state: BacktestState,
        metrics: PerformanceMetrics,
        symbol: str = "BTCUSDT",
        timeframe: str = "10m",
    ) -> Path:
        """
        Generate complete backtest report.

        Args:
            state: Backtest state
            metrics: Performance metrics
            symbol: Trading symbol
            timeframe: Timeframe used

        Returns:
            Path to report directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{symbol}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save trades CSV
        if self.config.save_trades and state.trades:
            self._save_trades_csv(state.trades, report_dir)

        # Save equity curve
        if self.config.save_equity_curve and state.equity_curve:
            self._save_equity_curve(state.equity_curve, report_dir)

        # Save metrics JSON
        self._save_metrics_json(metrics, symbol, timeframe, report_dir)

        # Generate text report
        self._save_text_report(metrics, symbol, timeframe, report_dir)

        # Generate HTML report (if enabled)
        if self.config.generate_html:
            self._save_html_report(state, metrics, symbol, timeframe, report_dir)

        logger.info(f"Report saved to {report_dir}")
        return report_dir

    def _save_trades_csv(self, trades: list, report_dir: Path):
        """Save trades to CSV."""
        records = []
        for trade in trades:
            records.append({
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_time": datetime.fromtimestamp(trade.entry_time / 1000).isoformat(),
                "entry_price": float(trade.entry_price),
                "exit_time": datetime.fromtimestamp(trade.exit_time / 1000).isoformat(),
                "exit_price": float(trade.exit_price),
                "exit_reason": trade.exit_reason,
                "quantity": float(trade.quantity),
                "pnl": float(trade.pnl),
                "pnl_percent": trade.pnl_percent,
                "r_multiple": trade.r_multiple,
                "stop_loss": float(trade.stop_loss),
                "take_profit": float(trade.take_profit) if trade.take_profit else None,
                "max_favorable_excursion": float(trade.max_favorable_excursion),
                "max_adverse_excursion": float(trade.max_adverse_excursion),
                "duration_hours": trade.duration_hours,
                "confidence": trade.signal_confidence,
            })

        df = pd.DataFrame(records)
        filepath = report_dir / "trades.csv"
        df.to_csv(filepath, index=False)
        logger.debug(f"Saved trades to {filepath}")

    def _save_equity_curve(self, equity_curve: list, report_dir: Path):
        """Save equity curve to CSV."""
        records = []
        for point in equity_curve:
            records.append({
                "timestamp": datetime.fromtimestamp(point["timestamp"] / 1000).isoformat(),
                "equity": point["equity"],
                "drawdown": point["drawdown"],
                "open_positions": point["open_positions"],
            })

        df = pd.DataFrame(records)
        filepath = report_dir / "equity_curve.csv"
        df.to_csv(filepath, index=False)
        logger.debug(f"Saved equity curve to {filepath}")

    def _save_metrics_json(
        self,
        metrics: PerformanceMetrics,
        symbol: str,
        timeframe: str,
        report_dir: Path,
    ):
        """Save metrics to JSON."""
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
        }

        filepath = report_dir / "metrics.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved metrics to {filepath}")

    def _save_text_report(
        self,
        metrics: PerformanceMetrics,
        symbol: str,
        timeframe: str,
        report_dir: Path,
    ):
        """Save plain text report."""
        lines = [
            "=" * 60,
            "         ORDERFLOW ABSORPTION BACKTEST REPORT",
            "=" * 60,
            "",
            f"Symbol:          {symbol}",
            f"Timeframe:       {timeframe}",
            f"Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PERFORMANCE METRICS",
            "-" * 40,
            f"Total Trades:        {metrics.total_trades}",
            f"Win Rate:            {metrics.win_rate:.1%}",
            f"Profit Factor:       {metrics.profit_factor:.2f}",
            f"Avg R Multiple:      {metrics.avg_r_multiple:.2f}R",
            f"Max Drawdown:        {metrics.max_drawdown_pct:.1%}",
            f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}",
            f"Sortino Ratio:       {metrics.sortino_ratio:.2f}",
            "",
            "TRADE BREAKDOWN",
            "-" * 40,
            f"Long Trades:         {metrics.long_trades} ({metrics.long_win_rate:.1%} win)",
            f"Short Trades:        {metrics.short_trades} ({metrics.short_win_rate:.1%} win)",
            f"Avg Winner:          +{metrics.avg_winner_r:.2f}R",
            f"Avg Loser:           {metrics.avg_loser_r:.2f}R",
            f"Largest Win:         ${metrics.largest_win:.2f}",
            f"Largest Loss:        ${metrics.largest_loss:.2f}",
            f"Max Consec Wins:     {metrics.max_consecutive_wins}",
            f"Max Consec Losses:   {metrics.max_consecutive_losses}",
            "",
            "EXIT ANALYSIS",
            "-" * 40,
        ]

        for reason, count in sorted(metrics.exit_reasons.items()):
            pct = count / metrics.total_trades * 100 if metrics.total_trades > 0 else 0
            lines.append(f"{reason}:".ljust(20) + f"{count} ({pct:.1f}%)")

        lines.extend([
            "",
            "=" * 60,
        ])

        filepath = report_dir / "report.txt"
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        logger.debug(f"Saved text report to {filepath}")

    def _save_html_report(
        self,
        state: BacktestState,
        metrics: PerformanceMetrics,
        symbol: str,
        timeframe: str,
        report_dir: Path,
    ):
        """Save HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {symbol}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1, h2 {{
            color: #00d4ff;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #00d4ff;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
        }}
        .metric-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #16213e;
            color: #00d4ff;
        }}
        tr:hover {{
            background: #16213e;
        }}
        .section {{
            margin-bottom: 40px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OrderFlow Absorption Backtest</h1>
        <p>{symbol} | {timeframe} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.avg_r_multiple:.2f}R</div>
                <div class="metric-label">Avg R Multiple</div>
            </div>
            <div class="metric-card">
                <div class="metric-value class="negative">{metrics.max_drawdown_pct:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value class="positive">${metrics.total_pnl:.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.expectancy:.2f}</div>
                <div class="metric-label">Expectancy</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Trade Breakdown</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Long</th>
                <th>Short</th>
                <th>Total</th>
            </tr>
            <tr>
                <td>Trades</td>
                <td>{metrics.long_trades}</td>
                <td>{metrics.short_trades}</td>
                <td>{metrics.total_trades}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{metrics.long_win_rate:.1%}</td>
                <td>{metrics.short_win_rate:.1%}</td>
                <td>{metrics.win_rate:.1%}</td>
            </tr>
            <tr>
                <td>Avg Winner</td>
                <td colspan="3">+{metrics.avg_winner_r:.2f}R</td>
            </tr>
            <tr>
                <td>Avg Loser</td>
                <td colspan="3">{metrics.avg_loser_r:.2f}R</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Exit Analysis</h2>
        <table>
            <tr>
                <th>Exit Reason</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {"".join(f'''<tr>
                <td>{reason}</td>
                <td>{count}</td>
                <td>{count/metrics.total_trades*100:.1f}%</td>
            </tr>''' for reason, count in sorted(metrics.exit_reasons.items()))}
        </table>
    </div>

    <div class="section">
        <h2>Risk Metrics</h2>
        <table>
            <tr>
                <td>Max Consecutive Wins</td>
                <td>{metrics.max_consecutive_wins}</td>
            </tr>
            <tr>
                <td>Max Consecutive Losses</td>
                <td>{metrics.max_consecutive_losses}</td>
            </tr>
            <tr>
                <td>Largest Win</td>
                <td class="positive">${metrics.largest_win:.2f}</td>
            </tr>
            <tr>
                <td>Largest Loss</td>
                <td class="negative">${metrics.largest_loss:.2f}</td>
            </tr>
            <tr>
                <td>Avg MFE</td>
                <td>${metrics.avg_mfe:.2f}</td>
            </tr>
            <tr>
                <td>Avg MAE</td>
                <td>${metrics.avg_mae:.2f}</td>
            </tr>
        </table>
    </div>

</body>
</html>
"""

        filepath = report_dir / "report.html"
        with open(filepath, "w") as f:
            f.write(html)

        logger.debug(f"Saved HTML report to {filepath}")

    def print_summary(self, metrics: PerformanceMetrics):
        """Print summary to console."""
        metrics.print_report()
