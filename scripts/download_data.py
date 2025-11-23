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

    return parser.parse_args()


async def download_symbol(
    client: BinanceClient,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    file_format: str,
):
    """Download data for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Downloading {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    def progress(current_ts, end_ts):
        current_dt = datetime.fromtimestamp(current_ts / 1000)
        end_dt = datetime.fromtimestamp(end_ts / 1000)
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

    # Output settings
    output_dir = args.output or config.data.data_dir + "/raw"
    file_format = args.format

    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
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
