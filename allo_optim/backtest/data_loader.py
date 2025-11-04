import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from allo_optim.backtest.data_cleaning import clean_price_data
from allo_optim.config.stock_universe import get_stocks_by_symbols

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of historical price data."""

    def __init__(self, benchmark: str, symbols: list[str],) -> None:
        self.stock_universe = get_stocks_by_symbols(symbols)
        self.symbols = symbols

        # Add SPY for benchmark
        if benchmark not in self.symbols:
            self.symbols.append(benchmark)

        logger.info(f"Loaded universe with {len(self.symbols)} symbols")

    def load_price_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load historical price data for the entire universe.

        Returns:
            DataFrame with dates as index and symbols as columns
        """
        logger.info(f"Loading price data from {start_date} to {end_date}")

        # Download data in batches to avoid API limits
        batch_size = 50
        all_data = []

        for i in range(0, len(self.symbols), batch_size):
            batch_symbols = self.symbols[i : i + batch_size]
            logger.info(f"Loading batch {i//batch_size + 1}: {len(batch_symbols)} symbols")

            try:
                batch_data = yf.download(
                    batch_symbols, start=start_date, end=end_date, progress=False, group_by="ticker", auto_adjust=False
                )

                # Handle single vs multiple symbols
                if len(batch_symbols) == 1:
                    symbol = batch_symbols[0]
                    if "Adj Close" in batch_data.columns:
                        prices = batch_data["Adj Close"].dropna().to_frame(symbol)
                    else:
                        prices = pd.DataFrame()
                else:
                    # Extract adjusted close prices
                    prices = pd.DataFrame()

                    # Check if we have multi-level columns
                    if hasattr(batch_data.columns, "levels") and len(batch_data.columns.levels) > 1:
                        # Multi-level columns (symbol, field)
                        for symbol in batch_symbols:
                            try:
                                if (symbol, "Adj Close") in batch_data.columns:
                                    adj_close = batch_data[(symbol, "Adj Close")].dropna()
                                    if not adj_close.empty:
                                        prices[symbol] = adj_close
                            except Exception as e:
                                logger.warning(f"Could not extract {symbol}: {e}")
                    elif "Adj Close" in batch_data.columns and len(batch_symbols) == 1:
                        symbol = batch_symbols[0]
                        prices[symbol] = batch_data["Adj Close"].dropna()

                if not prices.empty:
                    all_data.append(prices)

            except Exception as e:
                logger.error(f"Failed to load batch {i//batch_size + 1}: {e}")

        if not all_data:
            raise ValueError("No price data could be loaded")

        # Combine all batches
        combined_data = pd.concat(all_data, axis=1)

        # Remove duplicated columns if any
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

        # Apply robust data cleaning to handle NaN values and market closures
        combined_data = clean_price_data(combined_data)

        logger.info(f"Loaded data for {len(combined_data.columns)} assets over {len(combined_data)} days")

        return combined_data
