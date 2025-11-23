#!/usr/bin/env python3
"""
Quick test script to verify Binance API connectivity.
"""

import asyncio
import httpx
from datetime import datetime, timedelta

async def test_binance_api():
    """Test Binance Futures API connectivity."""

    # Test endpoints
    spot_url = "https://api.binance.com/api/v3/aggTrades"
    futures_url = "https://fapi.binance.com/fapi/v1/aggTrades"

    # Calculate timestamps for last hour
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (60 * 60 * 1000)  # 1 hour ago

    params = {
        "symbol": "BTCUSDT",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 10
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test Futures API
        print("=" * 60)
        print("Testing Binance FUTURES API (fapi.binance.com)")
        print("=" * 60)
        print(f"URL: {futures_url}")
        print(f"Params: {params}")

        try:
            response = await client.get(futures_url, params=params)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Trades received: {len(data)}")
                if data:
                    print(f"First trade: {data[0]}")
                else:
                    print("WARNING: Empty response!")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

        print()

        # Test Spot API for comparison
        print("=" * 60)
        print("Testing Binance SPOT API (api.binance.com)")
        print("=" * 60)
        print(f"URL: {spot_url}")

        try:
            response = await client.get(spot_url, params=params)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Trades received: {len(data)}")
                if data:
                    print(f"First trade: {data[0]}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

        print()

        # Test with a specific date in the past (Jan 2024)
        print("=" * 60)
        print("Testing with January 2024 date range")
        print("=" * 60)

        jan_start = int(datetime(2024, 1, 1).timestamp() * 1000)
        jan_end = int(datetime(2024, 1, 1, 1, 0, 0).timestamp() * 1000)  # 1 hour

        params_jan = {
            "symbol": "BTCUSDT",
            "startTime": jan_start,
            "endTime": jan_end,
            "limit": 10
        }

        print(f"Start: {datetime.fromtimestamp(jan_start/1000)}")
        print(f"End: {datetime.fromtimestamp(jan_end/1000)}")

        try:
            response = await client.get(futures_url, params=params_jan)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Trades received: {len(data)}")
                if data:
                    print(f"First trade timestamp: {datetime.fromtimestamp(data[0]['T']/1000)}")
                else:
                    print("WARNING: No trades for Jan 2024!")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_binance_api())
