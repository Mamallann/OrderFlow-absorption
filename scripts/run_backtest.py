#!/usr/bin/env python3
"""
Run OrderFlow Absorption backtest.

Usage:
    python scripts/run_backtest.py --symbol BTCUSDT
    python scripts/run_backtest.py --config config.yaml --all-symbols
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.data_loader import DataLoader
from src.footprint.builder import FootprintBuilder
from src.footprint.imbalance import ImbalanceDetector
from src.strategy.signals import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics
from src.visualization.report import ReportGenerator


def setup_logging(level: str = "INFO"):
    """Set up logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run OrderFlow Absorption Backtest")

    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Trading symbol (e.g., BTCUSDT)"
    )
    parser.add_argument(
        "--start", "-S",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-E",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Config file path (default: config.yaml)"
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Run backtest on all symbols from config"
    )
    parser.add_argument(
        "--direction",
        choices=["LONG", "SHORT", "BOTH"],
        default="BOTH",
        help="Trading direction (default: BOTH)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )

    return parser.parse_args()


def run_single_backtest(
    symbol: str,
    config,
    start_date: str,
    end_date: str,
    direction: str,
    generate_report: bool = True,
) -> dict:
    """Run backtest for a single symbol."""
    logger = logging.getLogger(__name__)

    print(f"\n{'='*60}")
    print(f"BACKTESTING {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Direction: {direction}")
    print(f"{'='*60}\n")

    # Step 1: Load data
    print("Loading data...")
    loader = DataLoader(data_dir=config.data.data_dir)
    df = loader.load_trades_df(symbol, start_date, end_date)

    if df.empty:
        print(f"No data found for {symbol}")
        return {"symbol": symbol, "error": "No data found"}

    print(f"Loaded {len(df):,} trades")

    # Step 2: Build footprint
    print("\nBuilding footprint candles...")
    builder = FootprintBuilder(
        timeframe=config.data.timeframe,
        tick_size=config.data.tick_size,
    )
    candles = builder.build_from_dataframe(df)
    print(f"Built {len(candles):,} candles")

    # Step 3: Detect imbalances
    print("\nDetecting imbalances...")
    imbalance_detector = ImbalanceDetector(
        ratio_threshold=config.imbalance.ratio_threshold,
        min_volume=config.imbalance.min_volume,
        consecutive_rows=config.imbalance.consecutive_rows,
        tick_size=config.data.tick_size,
    )
    candles = imbalance_detector.process_candles(candles)

    total_bid_imb = sum(len(c.bid_imbalances) for c in candles)
    total_ask_imb = sum(len(c.ask_imbalances) for c in candles)
    print(f"Found {total_bid_imb:,} bid imbalances, {total_ask_imb:,} ask imbalances")

    # Step 4: Generate signals
    print("\nGenerating signals...")
    signal_generator = SignalGenerator(config=config)
    signals = signal_generator.generate_signals(candles, symbol, direction)
    print(f"Generated {len(signals)} signals")

    if not signals:
        print("No signals generated - check your parameters")
        return {"symbol": symbol, "signals": 0, "trades": 0}

    # Step 5: Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(config=config)
    state = engine.run(candles, signals, symbol)
    print(f"Executed {state.total_trades} trades")

    # Step 6: Calculate metrics
    print("\nCalculating metrics...")
    metrics = PerformanceMetrics.from_backtest_state(state)

    # Print summary
    metrics.print_report()

    # Step 7: Generate report
    if generate_report:
        print("\nGenerating report...")
        reporter = ReportGenerator(config=config.reporting)
        report_path = reporter.generate_report(
            state=state,
            metrics=metrics,
            symbol=symbol,
            timeframe=config.data.timeframe,
        )
        print(f"Report saved to: {report_path}")

    return {
        "symbol": symbol,
        "signals": len(signals),
        "trades": state.total_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_pnl": metrics.total_pnl,
        "max_drawdown": metrics.max_drawdown_pct,
        "sharpe": metrics.sharpe_ratio,
    }


def main():
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Get symbols
    if args.all_symbols:
        symbols = config.data.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = config.data.symbols[:1]

    # Get date range
    start_date = args.start or config.date_range.start
    end_date = args.end or config.date_range.end

    # Run backtests
    results = []
    for symbol in symbols:
        result = run_single_backtest(
            symbol=symbol,
            config=config,
            start_date=start_date,
            end_date=end_date,
            direction=args.direction,
            generate_report=not args.no_report,
        )
        results.append(result)

    # Print summary if multiple symbols
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"{'Symbol':<12} {'Signals':>8} {'Trades':>8} {'WinRate':>8} {'PF':>8} {'Sharpe':>8}")
        print("-" * 60)
        for r in results:
            if "error" in r:
                print(f"{r['symbol']:<12} ERROR: {r['error']}")
            else:
                print(
                    f"{r['symbol']:<12} "
                    f"{r['signals']:>8} "
                    f"{r['trades']:>8} "
                    f"{r['win_rate']*100:>7.1f}% "
                    f"{r['profit_factor']:>8.2f} "
                    f"{r['sharpe']:>8.2f}"
                )
        print("=" * 60)


if __name__ == "__main__":
    main()
