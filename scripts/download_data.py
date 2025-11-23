#!/usr/bin/env python3
"""
Download historical AggTrades data from Binance.

Usage:
    python scripts/download_data.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31
    python scripts/download_data.py --config config.yaml
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.binance_client import BinanceClient
from src.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Binance AggTrades data")

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
        "--output", "-o",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)"
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
        help="Download all symbols from config"
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=0,
        help="Maximum trades to download (0 = unlimited). Use 500000 for ~1 hour of BTC"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=0,
        help="Download only N hours of data (overrides --end)"
    )

    return parser.parse_args()


async def download_symbol(
    client: BinanceClient,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    file_format: str,
    max_trades: int = 0,
):
    """Download data for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Downloading {symbol}")
    print(f"Period: {start_date} to {end_date}")
    if max_trades > 0:
        print(f"Max trades: {max_trades:,}")
    print(f"{'='*60}\n")

    start_time = datetime.now()
    trade_count = [0]  # Use list to allow modification in nested function

    def progress(current_ts, end_ts):
        current_dt = datetime.fromtimestamp(current_ts / 1000)
        progress_pct = (current_ts - int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)) / \
                      (end_ts - int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)) * 100
        print(f"\rProgress: {progress_pct:.1f}% - Current: {current_dt.strftime('%Y-%m-%d %H:%M')}", end="")

    try:
        filepath = await client.download_to_file(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            file_format=file_format,
            progress_callback=progress,
            max_trades=max_trades,
        )

        elapsed = datetime.now() - start_time
        print(f"\n\nComplete! Saved to: {filepath}")
        print(f"Time elapsed: {elapsed}")

        return filepath

    except Exception as e:
        print(f"\nError downloading {symbol}: {e}")
        return None


async def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Get symbols
    if args.all_symbols:
        symbols = config.data.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = config.data.symbols[:1]  # Default to first symbol

    # Get date range
    start_date = args.start or config.date_range.start
    end_date = args.end or config.date_range.end

    # Handle --hours option (overrides end_date)
    if args.hours > 0:
        from datetime import timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(hours=args.hours)
        end_date = end_dt.strftime("%Y-%m-%d")
        print(f"Using {args.hours} hour(s) of data")

    # Output settings
    output_dir = args.output or config.data.data_dir + "/raw"
    file_format = args.format
    max_trades = args.max_trades

    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    if max_trades > 0:
        print(f"Max trades: {max_trades:,}")
    print(f"Output: {output_dir} ({file_format})")

    # Create client and download
    async with BinanceClient(
        rate_limit_delay=config.binance.rate_limit_delay
    ) as client:
        results = []
        for symbol in symbols:
            result = await download_symbol(
                client=client,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                file_format=file_format,
                max_trades=max_trades,
            )
            results.append((symbol, result))

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for symbol, filepath in results:
        status = "✓" if filepath else "✗"
        print(f"{status} {symbol}: {filepath or 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
