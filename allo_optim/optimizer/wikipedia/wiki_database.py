"""
Simple SQL-based data storage for Wikipedia page views, stock prices, and volumes.

This module provides a clean, simple interface for managing market data in SQLite databases
with manual control over data downloading and loading.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from allooptim.config.stock_dataclasses import StockUniverse
from allooptim.optimizer.wikipedia.patched_pageview_api import pageviewapi

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_DIR = Path("generated_output/databases")
DATABASE_PATH = DATABASE_DIR / "market_data.db"


def _get_date_range(
    table_name: str, database_path: Optional[Path] = None
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Get the min and max dates from a table."""
    db_path = database_path or DATABASE_PATH
    db_path_str = str(db_path)
    try:
        with sqlite3.connect(db_path_str) as conn:
            cursor = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table_name}")
            result = cursor.fetchone()

            if result[0] is None:
                return None, None

            min_date = datetime.strptime(result[0], "%Y-%m-%d")
            max_date = datetime.strptime(result[1], "%Y-%m-%d")
            return min_date, max_date
    except Exception:
        return None, None


def _get_existing_symbols(table_name: str, database_path: Optional[Path] = None) -> list[str]:
    """Get existing symbol columns from a table."""
    db_path = database_path or DATABASE_PATH
    db_path_str = str(db_path)
    with sqlite3.connect(db_path_str) as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        # Remove 'date' column, keep only symbol columns
        return [col for col in columns if col != "date"]


def _add_symbol_columns(table_name: str, new_symbols: list[str], database_path: Optional[Path] = None):
    """Add new symbol columns to a table. Creates table if it doesn't exist."""
    db_path = database_path or DATABASE_PATH
    db_path_str = str(db_path)
    with sqlite3.connect(db_path_str) as conn:
        # Create table if it doesn't exist
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT PRIMARY KEY
            )
        """)

        existing_symbols = set(_get_existing_symbols(table_name, db_path))
        symbols_to_add = [s for s in new_symbols if s not in existing_symbols]

        if symbols_to_add:
            for symbol in symbols_to_add:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN `{symbol}` REAL")
            conn.commit()
        logger.info(f"Added {len(symbols_to_add)} new symbol columns to {table_name}")


def _get_table_report(table_name: str, database_path: Optional[Path] = None) -> dict[str, Any]:
    """Get statistics for a table."""
    db_path = database_path or DATABASE_PATH
    db_path_str = str(db_path)
    try:
        with sqlite3.connect(db_path_str) as conn:
            # Get row count
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get symbol count
            symbols = _get_existing_symbols(table_name, db_path)
            symbol_count = len(symbols)

            # Get date range
            date_min, date_max = _get_date_range(table_name, db_path)

            return {
                "table_name": table_name,
                "row_count": row_count,
                "symbol_count": symbol_count,
                "symbols": sorted(symbols),
                "date_range": (date_min, date_max),
                "date_min": date_min,
                "date_max": date_max,
            }
    except Exception as e:
        logger.error(f"Error getting stats for {table_name}: {e}")
        return {
            "table_name": table_name,
            "row_count": 0,
            "symbol_count": 0,
            "symbols": [],
            "date_range": (None, None),
            "date_min": None,
            "date_max": None,
        }


def _fetch_wikipedia_data(
    stocks: list[StockUniverse],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Fetch Wikipedia page view data for stocks in date range."""
    data_list = []

    def _get_time_string(time: datetime) -> str:
        """Format datetime for Wikipedia API."""
        return f"{time.year}{time.month:02d}{time.day:02d}"

    for index, stock in enumerate(stocks):
        logger.debug(f"Fetching Wikipedia data for {stock.symbol} ({index+1}/{len(stocks)})")

        try:
            response = pageviewapi.per_article(
                "en.wikipedia",
                stock.wikipedia_name,
                _get_time_string(start_date),
                _get_time_string(end_date),
                access="all-access",
                agent="user",
            )

            # Process response
            for item in response.get("items", []):
                item_date = datetime.strptime(item["timestamp"], "%Y%m%d00")
                data_list.append(
                    {
                        "date": item_date,
                        "symbol": stock.symbol,
                        "views": item["views"],
                    }
                )

        except Exception as error:
            logger.warning(f"Failed to fetch Wikipedia data for {stock.symbol}: {error}")
            # Add NaN entries for missing data
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            for date in date_range:
                data_list.append(
                    {
                        "date": date.to_pydatetime(),
                        "symbol": stock.symbol,
                        "views": np.nan,
                    }
                )

    if len(data_list) == 0:
        raise ValueError("No Wikipedia data fetched")

    df = pd.DataFrame(data_list)
    df_pivot = df.pivot(index="date", columns="symbol", values="views")

    return df_pivot


def _fetch_stock_data(
    stocks: list[StockUniverse],
    start_date: datetime,
    end_date: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch stock price and volume data for stocks in date range."""
    symbols = [stock.symbol for stock in stocks]

    try:
        # Download all stocks at once for efficiency
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date + timedelta(days=1),  # Include end date
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            raise ValueError("No data fetched from yfinance")

        # Handle single vs multiple symbols
        if len(symbols) == 1:
            prices_df = pd.DataFrame(index=data.index, columns=symbols)
            volumes_df = pd.DataFrame(index=data.index, columns=symbols)
            prices_df[symbols[0]] = data.get("Close")
            volumes_df[symbols[0]] = data.get("Volume")

        else:
            prices_df = data.get("Close")
            volumes_df = data.get("Volume")

            # Ensure we have all requested symbols as columns
            for symbol in symbols:
                if symbol not in prices_df.columns:
                    prices_df[symbol] = np.nan
                if symbol not in volumes_df.columns:
                    volumes_df[symbol] = np.nan

        return prices_df, volumes_df

    except Exception as e:
        logger.error(f"Failed to fetch stock data: {e}")
        empty_index = pd.DatetimeIndex([], name="date")
        empty_df = pd.DataFrame(index=empty_index, columns=symbols)
        return empty_df, empty_df


def _save_to_database(
    table_name: str,
    df: pd.DataFrame,
    database_path: Optional[Path] = None,
) -> None:
    """Save DataFrame to database table."""
    db_path = database_path or DATABASE_PATH
    db_path_str = str(db_path)
    if df.empty:
        logger.warning(f"No data to save to {table_name}")
        return

    # Ensure symbol columns exist in database
    symbols = [col for col in df.columns if col != "date"]
    _add_symbol_columns(table_name, symbols, db_path)

    # Prepare data for saving
    if isinstance(df.index, pd.DatetimeIndex):
        df_to_save = df.reset_index()
        # The reset index will create a column with the index name or default to 0
        # Find the datetime column and rename it to 'date'
        if df.index.name is not None:
            # If index has a name, it will be used as column name
            if df.index.name in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={df.index.name: "date"})
        else:
            # If index has no name, pandas uses 0, 1, 2... or the first available
            # Find the datetime column
            datetime_cols = df_to_save.select_dtypes(include=["datetime64"]).columns
            if len(datetime_cols) > 0:
                df_to_save = df_to_save.rename(columns={datetime_cols[0]: "date"})
            else:
                # Assume the first column is the date
                first_col = df_to_save.columns[0]
                df_to_save = df_to_save.rename(columns={first_col: "date"})
    else:
        df_to_save = df.copy()
        # If no datetime index, assume 'date' column already exists
        if "date" not in df_to_save.columns:
            raise ValueError(
                f"DataFrame must have a 'date' column or DatetimeIndex. Columns: {df_to_save.columns.tolist()}"
            )

    # Ensure date column is properly formatted
    if "date" in df_to_save.columns:
        df_to_save["date"] = pd.to_datetime(df_to_save["date"]).dt.strftime("%Y-%m-%d")

    # Calculate chunk size to avoid SQLite variable limit
    # SQLite default limit is 999 variables, but can be up to 32,766
    # We'll use conservative chunk size based on number of columns
    num_columns = len(df_to_save.columns)
    max_vars_per_chunk = 2000  # Allow larger chunks
    chunk_size = max(1, max_vars_per_chunk // num_columns)

    logger.info(f"Saving {len(df_to_save)} rows with {num_columns} columns in chunks of {chunk_size}")

    # Save to database incrementally using INSERT OR REPLACE for each row
    with sqlite3.connect(db_path_str) as conn:
        # Create a list of all columns (date + symbols)
        all_columns = ["date"] + symbols
        placeholders = ", ".join(["?" for _ in all_columns])
        columns_str = ", ".join([f"`{col}`" for col in all_columns])

        # Create INSERT OR REPLACE statement
        insert_sql = f"""
            INSERT OR REPLACE INTO {table_name} ({columns_str})
            VALUES ({placeholders})
        """

        # Save data in chunks to avoid SQLite variable limits
        rows_inserted = 0
        for i in range(0, len(df_to_save), chunk_size):
            chunk = df_to_save.iloc[i : i + chunk_size]

            # Convert chunk to list of tuples for batch insert
            chunk_data = []
            for _, row in chunk.iterrows():
                row_data = [row["date"]] + [row.get(sym, None) for sym in symbols]
                chunk_data.append(tuple(row_data))

            # Execute batch insert
            conn.executemany(insert_sql, chunk_data)
            rows_inserted += len(chunk_data)

        conn.commit()

    logger.info(f"Incrementally saved/updated {rows_inserted} rows to {table_name}")


def _load_from_database(
    table_name: str,
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    database_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load data from database table."""
    db_path = database_path or DATABASE_PATH
    db_path_str = str(db_path)
    logger.info(f"Loading from database path: {db_path_str}")
    try:
        # Ensure symbols exist in database
        _add_symbol_columns(table_name, symbols, db_path)

        # Build query for requested symbols
        symbol_columns = ", ".join([f"`{sym}`" for sym in symbols])
        query = f"""
            SELECT date, {symbol_columns}
            FROM {table_name}
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """

        with sqlite3.connect(db_path_str) as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
                parse_dates=["date"],
                index_col="date",
            )

        # Ensure we have all requested symbols as columns (fill with NaN if missing)
        for symbol in symbols:
            if symbol not in df.columns:
                df[symbol] = np.nan

        # Reorder columns to match requested order
        df = df[symbols]

        # Convert all columns to numeric, replacing None with NaN
        # This fixes the issue where SQL NULL values become Python None and create object dtypes
        for col in df.columns.tolist():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as error:
        logger.error(f"Error loading from {table_name}: {error}")
        logger.error(f"Database path: {db_path_str}")
        import os

        logger.error(f"Database file exists: {os.path.exists(db_path_str)}")
        if os.path.exists(db_path_str):
            logger.error(f"Database file size: {os.path.getsize(db_path_str)} bytes")

        raise ValueError(f"Error loading from {table_name}: {str(error)}") from error


def status_report_databases() -> dict[str, Any]:
    """
    Read-only status report of all three databases.

    Returns:
        Dictionary with statistics for wiki_views, stock_prices, and stock_volumes tables
    """
    logger.info("Generating database status report...")

    tables = ["wiki_views", "stock_prices", "stock_volumes"]
    report = {}

    for table in tables:
        stats = _get_table_report(table)
        report[table] = stats

        # Log summary
        if stats["date_min"] and stats["date_max"]:
            logger.info(
                f"{table}: {stats['row_count']} rows, {stats['symbol_count']} symbols, "
                f"dates {stats['date_min'].date()} to {stats['date_max'].date()}"
            )
        else:
            logger.info(f"{table}: {stats['row_count']} rows, {stats['symbol_count']} symbols, no data")

    return report


def download_data(start_date: datetime, end_date: datetime, stocks: list[StockUniverse]) -> None:
    """
    B) Manual data download/update function (incremental).

    Args:
        start_date: Start date for data download
        end_date: End date for data download
        stocks: list of StockUniverse objects to download
    """
    logger.info(f"Incremental download for {len(stocks)} stocks from {start_date.date()} to {end_date.date()}")

    # Check existing data ranges to optimize downloads
    # wiki_min, wiki_max = sql_cache.get_date_range("wiki_views")
    # stock_min, stock_max = sql_cache.get_date_range("stock_prices")

    # Download Wikipedia data (always download requested range as it's date-specific)
    logger.info("Downloading Wikipedia page views...")
    wiki_data = _fetch_wikipedia_data(stocks, start_date, end_date)
    if not wiki_data.empty:
        _save_to_database("wiki_views", wiki_data)
    else:
        logger.info("No Wikipedia data to save")

    # Download stock data (always download requested range for consistency)
    logger.info("Downloading stock prices and volumes...")
    prices_data, volumes_data = _fetch_stock_data(stocks, start_date, end_date)

    if not prices_data.empty:
        _save_to_database("stock_prices", prices_data)
    else:
        logger.info("No stock price data to save")

    if not volumes_data.empty:
        _save_to_database("stock_volumes", volumes_data)
    else:
        logger.info("No stock volume data to save")

    logger.info("Incremental data download completed")


def load_data(
    start_date: datetime,
    end_date: datetime,
    stocks: list[StockUniverse],
    use_wiki_database: bool,
    database_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from SQL database or fetch fresh from APIs.

    Args:
        start_date: Start date for data
        end_date: End date for data
        stocks: list of StockUniverse objects
        use_wiki_database: If True, load from SQL database. If False, fetch fresh data.

    Returns:
        tuple of (wiki_views_df, stock_prices_df, stock_volumes_df)
    """
    symbols = [stock.symbol for stock in stocks]

    if use_wiki_database:
        logger.debug(f"Loading data from SQL database for {len(stocks)} stocks")
        wiki_df = _load_from_database("wiki_views", symbols, start_date, end_date, database_path)
        prices_df = _load_from_database("stock_prices", symbols, start_date, end_date, database_path)
        volumes_df = _load_from_database("stock_volumes", symbols, start_date, end_date, database_path)

        logger.debug(f"Loaded from SQL: wiki {wiki_df.shape}, prices {prices_df.shape}, volumes {volumes_df.shape}")

    else:
        logger.debug(f"Fetching fresh data from APIs for {len(stocks)} stocks")
        wiki_df = _fetch_wikipedia_data(stocks, start_date, end_date)
        prices_df, volumes_df = _fetch_stock_data(stocks, start_date, end_date)

        logger.debug(f"Fetched fresh: wiki {wiki_df.shape}, prices {prices_df.shape}, volumes {volumes_df.shape}")

    # Ensure proper datetime index and symbol ordering
    wiki_df.index.name = "date"
    prices_df.index.name = "date"
    volumes_df.index.name = "date"

    # Create complete date range from start_date to end_date
    # Use naive datetime (no timezone) for pure DatetimeIndex matching database format
    start_naive = start_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    end_naive = end_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    complete_date_range = pd.date_range(start=start_naive, end=end_naive, freq="D")

    # Ensure all DataFrames have complete date index and all symbols
    for df_name, df in [
        ("wiki", wiki_df),
        ("prices", prices_df),
        ("volumes", volumes_df),
    ]:
        # Remove duplicate dates by taking the last value for each date
        if df.index.has_duplicates:
            logger.warning(f"Found duplicate dates in {df_name} data, keeping last value for each date")
            df_cleaned = df[~df.index.duplicated(keep="last")]
        else:
            df_cleaned = df

        # Reindex to include all dates in range, filling missing dates with NaN
        df_cleaned = df_cleaned.reindex(complete_date_range, fill_value=np.nan)

        for col in df_cleaned.columns.tolist():
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

        # Ensure all requested symbols are present as columns in correct order
        for symbol in symbols:
            if symbol not in df_cleaned.columns:
                df_cleaned[symbol] = np.nan

        # Reorder columns to match the requested symbol order
        df_cleaned = df_cleaned[symbols]

        # Update the reference
        if df_name == "wiki":
            wiki_df = df_cleaned
        elif df_name == "prices":
            prices_df = df_cleaned
        else:  # volumes
            volumes_df = df_cleaned

    return wiki_df, prices_df, volumes_df
