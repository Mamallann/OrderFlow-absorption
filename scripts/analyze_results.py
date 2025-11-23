#!/usr/bin/env python3
"""
Analyze backtest results.

Usage:
    python scripts/analyze_results.py --report data/results/report_BTCUSDT_20240101_120000
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze backtest results")

    parser.add_argument(
        "--report", "-r",
        type=str,
        required=True,
        help="Path to report directory"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )

    return parser.parse_args()


def load_trades(report_dir: Path) -> pd.DataFrame:
    """Load trades CSV."""
    trades_file = report_dir / "trades.csv"
    if not trades_file.exists():
        print(f"Trades file not found: {trades_file}")
        return pd.DataFrame()
    return pd.read_csv(trades_file)


def load_metrics(report_dir: Path) -> dict:
    """Load metrics JSON."""
    metrics_file = report_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return {}
    with open(metrics_file) as f:
        return json.load(f)


def load_equity_curve(report_dir: Path) -> pd.DataFrame:
    """Load equity curve CSV."""
    equity_file = report_dir / "equity_curve.csv"
    if not equity_file.exists():
        return pd.DataFrame()
    return pd.read_csv(equity_file)


def analyze_trades(trades: pd.DataFrame) -> dict:
    """Detailed trade analysis."""
    if trades.empty:
        return {}

    # R-multiple distribution
    r_mults = trades["r_multiple"]
    r_dist = {
        "< -1R": len(r_mults[r_mults < -1]),
        "-1R to 0": len(r_mults[(r_mults >= -1) & (r_mults < 0)]),
        "0 to 1R": len(r_mults[(r_mults >= 0) & (r_mults < 1)]),
        "1R to 2R": len(r_mults[(r_mults >= 1) & (r_mults < 2)]),
        "2R to 3R": len(r_mults[(r_mults >= 2) & (r_mults < 3)]),
        "> 3R": len(r_mults[r_mults >= 3]),
    }

    # Duration analysis
    durations = trades["duration_hours"]
    duration_stats = {
        "avg": durations.mean(),
        "min": durations.min(),
        "max": durations.max(),
        "median": durations.median(),
    }

    # Exit reason breakdown
    exit_reasons = trades["exit_reason"].value_counts().to_dict()

    # Long vs Short
    long_trades = trades[trades["side"] == "LONG"]
    short_trades = trades[trades["side"] == "SHORT"]

    side_stats = {
        "long": {
            "count": len(long_trades),
            "win_rate": (long_trades["pnl"] > 0).mean() if len(long_trades) > 0 else 0,
            "avg_r": long_trades["r_multiple"].mean() if len(long_trades) > 0 else 0,
        },
        "short": {
            "count": len(short_trades),
            "win_rate": (short_trades["pnl"] > 0).mean() if len(short_trades) > 0 else 0,
            "avg_r": short_trades["r_multiple"].mean() if len(short_trades) > 0 else 0,
        },
    }

    # Confidence analysis
    conf = trades["confidence"]
    confidence_stats = {
        "avg": conf.mean(),
        "correlation_with_pnl": trades["confidence"].corr(trades["pnl"]),
    }

    return {
        "r_distribution": r_dist,
        "duration_stats": duration_stats,
        "exit_reasons": exit_reasons,
        "side_stats": side_stats,
        "confidence_stats": confidence_stats,
    }


def print_analysis(metrics: dict, trade_analysis: dict):
    """Print formatted analysis."""
    print("\n" + "=" * 60)
    print("         DETAILED BACKTEST ANALYSIS")
    print("=" * 60)

    if metrics:
        m = metrics.get("metrics", {})
        print(f"\nSymbol: {metrics.get('symbol', 'N/A')}")
        print(f"Timeframe: {metrics.get('timeframe', 'N/A')}")

        print("\nKEY METRICS")
        print("-" * 40)
        print(f"Total Trades:    {m.get('total_trades', 0)}")
        print(f"Win Rate:        {m.get('win_rate', 0):.1f}%")
        print(f"Profit Factor:   {m.get('profit_factor', 0)}")
        print(f"Sharpe Ratio:    {m.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:    {m.get('max_drawdown_pct', 0):.1f}%")

    if trade_analysis:
        print("\nR-MULTIPLE DISTRIBUTION")
        print("-" * 40)
        for bucket, count in trade_analysis.get("r_distribution", {}).items():
            print(f"{bucket}: {count}")

        print("\nDURATION STATISTICS (hours)")
        print("-" * 40)
        dur = trade_analysis.get("duration_stats", {})
        print(f"Average:  {dur.get('avg', 0):.1f}")
        print(f"Median:   {dur.get('median', 0):.1f}")
        print(f"Min:      {dur.get('min', 0):.1f}")
        print(f"Max:      {dur.get('max', 0):.1f}")

        print("\nEXIT REASONS")
        print("-" * 40)
        for reason, count in trade_analysis.get("exit_reasons", {}).items():
            print(f"{reason}: {count}")

        print("\nLONG vs SHORT")
        print("-" * 40)
        side = trade_analysis.get("side_stats", {})
        if "long" in side:
            l = side["long"]
            print(f"Long:  {l['count']} trades, {l['win_rate']*100:.1f}% win, {l['avg_r']:.2f}R avg")
        if "short" in side:
            s = side["short"]
            print(f"Short: {s['count']} trades, {s['win_rate']*100:.1f}% win, {s['avg_r']:.2f}R avg")

        print("\nCONFIDENCE ANALYSIS")
        print("-" * 40)
        conf = trade_analysis.get("confidence_stats", {})
        print(f"Avg Confidence:           {conf.get('avg', 0):.2f}")
        print(f"Correlation with P&L:     {conf.get('correlation_with_pnl', 0):.3f}")

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    report_dir = Path(args.report)
    if not report_dir.exists():
        print(f"Report directory not found: {report_dir}")
        sys.exit(1)

    # Load data
    trades = load_trades(report_dir)
    metrics = load_metrics(report_dir)
    equity = load_equity_curve(report_dir)

    # Analyze
    trade_analysis = analyze_trades(trades)

    if args.format == "json":
        output = {
            "metrics": metrics,
            "trade_analysis": trade_analysis,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_analysis(metrics, trade_analysis)


if __name__ == "__main__":
    main()
