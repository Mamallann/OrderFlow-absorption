"""Data loading and management utilities."""

from pathlib import Path
from typing import List, Optional, Iterator
from decimal import Decimal
import logging

import pandas as pd

from .models import AggTrade

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and manage tick data from various sources.

    Supports CSV and Parquet files, with lazy loading for large datasets.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if needed
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def find_data_files(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Path]:
        """
        Find data files for a symbol.

        Args:
            symbol: Trading pair
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of matching file paths
        """
        files = []

        # Search in raw directory
        for pattern in [f"{symbol}*.parquet", f"{symbol}*.csv"]:
            files.extend(self.raw_dir.glob(pattern))

        # Sort by filename (which includes dates)
        files.sort(key=lambda x: x.name)

        logger.info(f"Found {len(files)} data files for {symbol}")
        return files

    def load_trades_df(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load trades as DataFrame.

        Args:
            symbol: Trading pair
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with trade data
        """
        files = self.find_data_files(symbol, start_date, end_date)

        if not files:
            logger.warning(f"No data files found for {symbol}")
            return pd.DataFrame()

        # Load all files
        dfs = []
        for file in files:
            if file.suffix == ".parquet":
                df = pd.read_parquet(file)
            else:
                df = pd.read_csv(file)
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {file.name}")

        # Combine
        df = pd.concat(dfs, ignore_index=True)

        # Filter by date range if specified
        if start_date:
            start_ts = pd.Timestamp(start_date).timestamp() * 1000
            df = df[df["timestamp"] >= start_ts]

        if end_date:
            end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp() * 1000
            df = df[df["timestamp"] < end_ts]

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=["agg_trade_id"])

        logger.info(f"Loaded {len(df)} trades for {symbol}")
        return df

    def load_trades(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[AggTrade]:
        """
        Load trades as AggTrade objects.

        Args:
            symbol: Trading pair
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of AggTrade objects
        """
        df = self.load_trades_df(symbol, start_date, end_date)

        if df.empty:
            return []

        trades = []
        for _, row in df.iterrows():
            trade = AggTrade(
                agg_trade_id=int(row["agg_trade_id"]),
                price=Decimal(str(row["price"])),
                quantity=Decimal(str(row["quantity"])),
                first_trade_id=int(row["first_trade_id"]),
                last_trade_id=int(row["last_trade_id"]),
                timestamp=int(row["timestamp"]),
                is_buyer_maker=bool(row["is_buyer_maker"]),
            )
            trades.append(trade)

        return trades

    def iterate_trades(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        chunk_size: int = 100000,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over trades in chunks (memory efficient).

        Args:
            symbol: Trading pair
            start_date: Optional start date
            end_date: Optional end date
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        files = self.find_data_files(symbol, start_date, end_date)

        for file in files:
            if file.suffix == ".parquet":
                # Parquet doesn't support chunked reading easily
                df = pd.read_parquet(file)
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i + chunk_size]
            else:
                # CSV supports chunked reading
                for chunk in pd.read_csv(file, chunksize=chunk_size):
                    yield chunk

    def load_from_file(self, filepath: str | Path) -> pd.DataFrame:
        """
        Load trades from a specific file.

        Args:
            filepath: Path to data file

        Returns:
            DataFrame with trades
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Loaded {len(df)} trades from {filepath}")
        return df

    def save_trades_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        file_format: str = "parquet",
        subdir: str = "raw",
    ) -> Path:
        """
        Save trades DataFrame to file.

        Args:
            df: DataFrame to save
            symbol: Trading pair
            start_date: Start date for filename
            end_date: End date for filename
            file_format: "parquet" or "csv"
            subdir: Subdirectory within data_dir

        Returns:
            Path to saved file
        """
        output_dir = self.data_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol}_{start_date}_{end_date}_aggtrades"

        if file_format == "parquet":
            filepath = output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        else:
            filepath = output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)

        logger.info(f"Saved {len(df)} trades to {filepath}")
        return filepath

    def get_data_summary(self, symbol: str) -> dict:
        """
        Get summary of available data for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with data summary
        """
        files = self.find_data_files(symbol)

        if not files:
            return {
                "symbol": symbol,
                "files": 0,
                "total_trades": 0,
                "date_range": None,
            }

        # Load first and last timestamps
        total_trades = 0
        min_timestamp = float("inf")
        max_timestamp = 0

        for file in files:
            if file.suffix == ".parquet":
                df = pd.read_parquet(file, columns=["timestamp"])
            else:
                df = pd.read_csv(file, usecols=["timestamp"])

            total_trades += len(df)
            min_timestamp = min(min_timestamp, df["timestamp"].min())
            max_timestamp = max(max_timestamp, df["timestamp"].max())

        return {
            "symbol": symbol,
            "files": len(files),
            "total_trades": total_trades,
            "date_range": {
                "start": pd.Timestamp(min_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
                "end": pd.Timestamp(max_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
            },
        }
