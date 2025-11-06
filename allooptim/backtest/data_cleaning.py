"""Data Cleaning Utilities.

Provides comprehensive data cleaning functions for financial price data.
Handles NaN values, market closures, insufficient data, and data quality issues.

Author: AI Assistant
Date: 2024-11-02
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_price_data(
    df_prices: pd.DataFrame,
    trading_day_threshold: float = 0.50,
    min_data_threshold: float = 0.80,
    max_individual_gaps_pct: float = 0.05,
    min_symbols_required: int = 10,
) -> pd.DataFrame:
    """Clean price data by removing market closures, insufficient data, and handling NaN values.

    This function performs comprehensive data cleaning in multiple steps:
    1. Remove market closure days (weekends/holidays where majority of stocks have NaN)
    2. Remove stocks with insufficient trading data (< 80% of trading days)
    3. Handle remaining individual stock gaps (halts, suspensions)
    4. Final validation to ensure no NaN values remain

    Args:
        df_prices: DataFrame with dates as index and symbols as columns
        trading_day_threshold: Minimum fraction of stocks that must have data for a day to be considered a trading day
        (default: 0.50)
        min_data_threshold: Minimum fraction of trading days a stock must have data for (default: 0.80)
        max_individual_gaps_pct: Maximum fraction of trading days allowed for individual stock gaps (default: 0.05)
        min_symbols_required: Minimum number of symbols required after cleaning (default: 10)

    Returns:
        Cleaned DataFrame with no NaN values, or empty DataFrame if cleaning fails
    """
    try:
        # Ensure we have data and clean it
        if df_prices.empty:
            logger.error("Price data DataFrame is empty")
            return pd.DataFrame()

        # Clean the data by removing market closures and insufficient data
        initial_shape = df_prices.shape
        logger.info(f"Raw price data shape: {initial_shape}")

        # Step 1: Remove market closure days (rows where majority of stocks have NaN)
        # Market closures affect most/all stocks simultaneously (weekends, holidays)
        initial_rows = len(df_prices)

        # Calculate how many stocks have data each day
        daily_data_availability = df_prices.count(axis=1) / len(df_prices.columns)

        # Keep only trading days where at least threshold fraction of stocks have data
        trading_days_mask = daily_data_availability >= trading_day_threshold
        df_prices = df_prices[trading_days_mask]

        removed_days = initial_rows - len(df_prices)
        if removed_days > 0:
            logger.info(f"Removed {removed_days} market closure days (weekends/holidays)")

        # Step 2: Remove stocks with insufficient trading data
        # After removing market closures, require stocks to have data for min_data_threshold of trading days
        actual_trading_days = len(df_prices)
        min_required_days = int(actual_trading_days * min_data_threshold)

        valid_columns = df_prices.columns[df_prices.count() >= min_required_days]
        df_prices = df_prices[valid_columns]

        if len(valid_columns) < len(df_prices.columns):
            dropped_stocks = len(df_prices.columns) - len(valid_columns)
            logger.info(
                f"Dropped {dropped_stocks} stocks with insufficient data "
                f"(< {min_required_days}/{actual_trading_days} trading days)"
            )

        # Step 3: Handle remaining individual stock gaps (halts, suspensions)
        if df_prices.isna().any().any():
            # Count remaining NaN values per stock
            remaining_nans = df_prices.isna().sum()

            # Set threshold: allow up to max_individual_gaps_pct missing data for individual stock issues
            max_individual_gaps = int(
                actual_trading_days * max_individual_gaps_pct
            )  # max_individual_gaps_pct of trading days

            # Remove stocks with too many individual gaps
            problematic_stocks = remaining_nans[remaining_nans > max_individual_gaps].index.tolist()

            if problematic_stocks:
                logger.warning(f"Removing {len(problematic_stocks)} stocks with excessive individual gaps")
                df_prices = df_prices.drop(columns=problematic_stocks)

            # For remaining small gaps (1-2 days), remove the rows to avoid data duplication
            if df_prices.isna().any().any():
                # Remove any remaining rows with NaN values (individual trading halts)
                clean_rows_mask = ~df_prices.isna().any(axis=1)
                df_prices = df_prices[clean_rows_mask]

                removed_gap_days = (~clean_rows_mask).sum()
                if removed_gap_days > 0:
                    logger.info(f"Removed {removed_gap_days} days with individual stock gaps (halts/suspensions)")

        # Final validation - ensure no NaN values remain
        if df_prices.isna().any().any():
            logger.error("DataFrame still contains NaN values after cleaning - this should not happen")
            # Fallback: remove any columns still containing NaN
            df_prices = df_prices.dropna(axis=1, how="any")

        # Step 4: Ensure we have enough data after cleaning
        if df_prices.empty:
            logger.error("No clean price data remaining after NaN removal")
            return pd.DataFrame()

        if df_prices.shape[1] < min_symbols_required:
            logger.warning(
                f"Very few symbols remain after cleaning: {df_prices.shape[1]} (required: {min_symbols_required})"
            )

        # Final validation
        if df_prices.isna().any().any():
            raise ValueError("DataFrame still contains NaN values after cleaning")

        logger.info(
            f"Clean price data shape: {df_prices.shape} (dropped {initial_shape[1] - df_prices.shape[1]} symbols)"
        )
        return df_prices

    except Exception as e:
        logger.error(f"Error cleaning price data: {e}")
        return pd.DataFrame()
