#!/usr/bin/env python3
"""
Download script for Wikipedia optimizer test resources.

This script downloads a small dataset for testing the Wikipedia optimizer
with a few popular stocks over a short time period.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import allo_optim.optimizer.wikipedia.wiki_database as sql_db
from allo_optim.config.stock_universe import get_stocks_by_symbols
from allo_optim.optimizer.wikipedia.wiki_database import download_data

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Override the database path for test resources
test_db_path = Path(__file__).parent / "test_wikipedia.db"
sql_db.DATABASE_PATH = test_db_path
sql_db.DATABASE_DIR = test_db_path.parent


def main():
    """Download test data for Wikipedia optimizer."""

    # Define test stocks (popular ones with good Wikipedia data)
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE"]

    # Get stock universe objects
    stocks = get_stocks_by_symbols(test_symbols)
    print(f"Found {len(stocks)} stocks: {[s.symbol for s in stocks]}")

    # Define date range (last 60 days to keep it small)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    print(f"Downloading data from {start_date.date()} to {end_date.date()}")

    # Download the data
    download_data(start_date, end_date, stocks)

    print("Download completed!")


if __name__ == "__main__":
    main()
