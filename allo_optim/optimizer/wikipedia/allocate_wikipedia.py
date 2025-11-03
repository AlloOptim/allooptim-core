import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz
from scipy import stats

from allo_optim.config.allocation_dataclasses import (
    AllocationResult,
    WikipediaStatistics,
    validate_asset_weights_length,
)

# Old complex caching imports removed - now using simple_sql_cache
from allo_optim.config.stock_dataclasses import StockUniverse
from allo_optim.config.stock_universe import list_of_dax_stocks
from allo_optim.optimizer.wikipedia.sql_database import load_data
from common.debug.performance_profiler import (
    log_performance_report,
    profile_function,
)

logger = logging.getLogger(__name__)

ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(days=7)


def _exponential_decay_weights(
    n: int,
    decay_factor: float,
) -> np.ndarray:
    """
    Create exponential decay weights where the most recent point has 'factor' times
    higher weight than the oldest point.

    Args:
        n: Number of weights to generate
        factor: Ratio of newest weight to oldest weight (default 3.0)

    Returns:
        Array of weights with exponential decay from oldest to newest
    """
    if n <= 1:
        return np.ones(n)

    # Calculate decay rate: r^(n-1) = 1/factor
    # So r = (1/factor)^(1/(n-1))
    decay_rate = (1.0 / decay_factor) ** (1.0 / (n - 1))

    # Generate weights: weight[i] = decay_rate^(n-1-i)
    # This gives oldest (i=0) the smallest weight, newest (i=n-1) the largest
    weights = np.array([decay_rate ** (n - 1 - i) for i in range(n)])

    return weights


def _get_failed_result(
    end_date: datetime,
    all_stocks: List[StockUniverse],
) -> AllocationResult:
    statistics = WikipediaStatistics(
        end_date=end_date.strftime("%Y-%m-%d"),
        r_squared=0.0,
        p_value=1.0,
        std_err=0.0,
        slope=0.0,
        intercept=0.0,
        all_symbols=[stock.symbol for stock in all_stocks],
        valid_data_symbols=[],
        significant_positive_stocks=[],
        top_n_symbols=[],
    )

    return AllocationResult(
        asset_weights={stock.symbol: 0.0 for stock in all_stocks},
        success=False,
        statistics=statistics,
        error_message="Failed to allocate Wikipedia-based weights",
    )


@profile_function()
def _remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    iqr_factor: float,
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the IQR method.
    k is the IQR multiplier, typically 1.5 for outliers or 3 for extreme outliers.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR

        # Create a mask for rows to keep
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        df = df[mask]

    return df


@profile_function()
def _filter_by_correlations(
    df_merged: pd.DataFrame,
    p_threshold_significance: float,
) -> tuple[bool, list[str]]:
    """Filter for stocks with positive correlations using vectorized groupby."""

    def compute_regression(group):
        """Vectorized regression for a single symbol group."""
        x = group["stock_price"].values.astype(np.float32)  # Use float32 for speed
        y = group["wiki_views"].values.astype(np.float32)
        n = len(x)
        if n < 2:
            return pd.Series({"slope": 0.0, "p_value": 1.0})

        # Check for zero variance in x (independent variable)
        # If all x values are the same, regression is undefined
        x_variance = np.var(x, ddof=1)
        if x_variance <= 1e-10:  # Very small threshold for numerical stability
            return pd.Series({"slope": 0.0, "p_value": 1.0})

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        cov_xy = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

        # Safe division: slope = cov_xy / var_x
        slope = cov_xy / x_variance
        intercept = mean_y - slope * mean_x

        # Calculate standard error safely
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)

        # Degrees of freedom for residual
        df_res = max(n - 2, 1)  # Ensure at least 1 df

        # Standard error of the slope
        se_slope_squared = ss_res / (df_res * max(np.sum((x - mean_x) ** 2), 1e-10))
        std_err = np.sqrt(max(se_slope_squared, 1e-10))  # Ensure positive std_err

        # t-statistic and p-value
        if std_err > 1e-10:
            t_stat = slope / std_err
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_res))
        else:
            # If std_err is essentially zero, slope is infinitely significant
            p_value = 0.0 if abs(slope) > 1e-10 else 1.0

        return pd.Series({"slope": slope, "p_value": p_value})

    results = df_merged.groupby("symbol").apply(compute_regression, include_groups=False)

    valid_symbols = results[(results["slope"] > 0) & (results["p_value"] < p_threshold_significance)].index.tolist()

    logger.debug(f"Stocks with significant positive correlation: {valid_symbols}")

    if not valid_symbols:
        logger.warning(f"No stocks with significant positive correlation found. Total stocks analyzed: {len(results)}")
        return False, None

    return True, valid_symbols


@profile_function()
def _select_top_n_stocks(
    df_significant: pd.DataFrame,
    n_max_stocks: int,
) -> tuple[bool, pd.DataFrame]:
    """Select top N stocks by total Wikipedia views."""
    if len(df_significant["symbol"].unique()) <= n_max_stocks:
        return True, df_significant

    top_symbols = df_significant.groupby("symbol")["wiki_views"].sum().nlargest(n_max_stocks).index.tolist()

    df_top = df_significant[df_significant["symbol"].isin(top_symbols)]

    logger.debug(f"Top {n_max_stocks} stocks by wiki views: {top_symbols}")

    if df_top.empty:
        logger.warning("No top stocks remaining.")
        return False, None

    return True, df_top


@profile_function()
def _create_asset_weights(
    df_top: pd.DataFrame,
    all_stocks: list[StockUniverse],
) -> tuple[bool, dict]:
    """Create asset weights - equal weight for top stocks, zero for others (including validation)."""
    top_symbols = df_top["symbol"].unique().tolist()
    asset_weights = {stock.symbol: 0.0 for stock in all_stocks}
    for symbol in top_symbols:
        asset_weights[symbol] = 1.0 / len(top_symbols)

    try:
        validate_asset_weights_length(asset_weights, len(all_stocks))
        return True, asset_weights

    except Exception as error:
        logger.error(f"Failed to validate asset weights: {error}")
        return False, None


@profile_function()
def _estimate_statistics(
    df_top: pd.DataFrame,
    end_date: datetime,
    all_symbols: list[str],
    valid_data_symbols: list[str],
    significant_positive_stocks: list[str],
    top_stocks: list[str],
) -> tuple[bool, WikipediaStatistics]:
    """Estimate statistics."""
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_top["stock_price"].values, df_top["wiki_views"].values
        )

        wiki_statistics = WikipediaStatistics(
            end_date=end_date.strftime("%Y-%m-%d"),
            r_squared=r_value**2,
            p_value=p_value,
            std_err=std_err,
            slope=slope,
            intercept=intercept,
            all_symbols=all_symbols,
            valid_data_symbols=valid_data_symbols,
            significant_positive_stocks=significant_positive_stocks,
            top_n_symbols=top_stocks,
        )
        return True, wiki_statistics

    except Exception as error:
        logger.error(f"Failed to estimate statistics: {error}")
        return False, None


@profile_function()
def _process_and_filter_data_combined(
    df_wiki_views: pd.DataFrame,
    df_stock_prices: pd.DataFrame,
    df_stock_volumes: Optional[pd.DataFrame],
    decay_factor: float,
    n_historical_days: int,
    n_lag_days: int,
    n_weeks_difference: int,
    min_trading_volume: int,
    iqr_factor: float,
) -> tuple[bool, pd.DataFrame, list[str]]:
    """Combined function: Vectorized data quality filtering and processing with wide-form DataFrames."""

    # Step 1: Filter symbols with >90% NaN in any DataFrame
    wiki_nan_pct = df_wiki_views.isna().mean()
    price_nan_pct = df_stock_prices.isna().mean()

    if df_stock_volumes is not None:
        volume_nan_pct = df_stock_volumes.isna().mean()
        valid_volume = volume_nan_pct > 0.9
    else:
        valid_volume = pd.Series(True, index=df_wiki_views.columns)

    # Symbol is invalid if >90% NaN in any column of the DataFrame
    invalid_symbols = (wiki_nan_pct > 0.9) | (price_nan_pct > 0.9) | (valid_volume)
    valid_symbols = invalid_symbols[~invalid_symbols].index.tolist()

    if len(valid_symbols) < 1:
        logger.warning("No valid symbols remaining after NaN filtering")
        return False, None, []

    logger.debug(f"Valid symbols after NaN filtering: {len(valid_symbols)}")

    # Step 3: Vectorized processing using shift operations
    shift_price_prev = n_weeks_difference * 7
    shift_wiki_current = n_lag_days
    shift_wiki_prev = n_lag_days + n_weeks_difference * 7

    # Create shifted DataFrames
    df_current_prices = df_stock_prices[valid_symbols].iloc[-n_historical_days:, :]
    df_prev_prices = df_stock_prices[valid_symbols].shift(shift_price_prev).iloc[-n_historical_days:, :]
    df_current_views = df_wiki_views[valid_symbols].shift(shift_wiki_current).iloc[-n_historical_days:, :]
    df_prev_views = df_wiki_views[valid_symbols].shift(shift_wiki_prev).iloc[-n_historical_days:, :]
    df_volumes = df_stock_volumes[valid_symbols].iloc[-n_historical_days:, :]

    # Vectorized condition checks
    low_volume = df_volumes < min_trading_volume

    has_negative = (df_current_prices < 0) | (df_prev_prices < 0) | (df_current_views < 0) | (df_prev_views < 0)

    # Calculate ratios and logs where conditions are met
    valid_mask = ~(low_volume | has_negative)

    wiki_ratios = df_current_views / df_prev_views
    price_ratios = df_current_prices / df_prev_prices

    # Apply exponential decay weights to give higher weight to more recent data
    decay_weights = _exponential_decay_weights(
        len(wiki_ratios),
        decay_factor=decay_factor,
    )
    wiki_ratios = wiki_ratios.mul(decay_weights, axis=0)
    price_ratios = price_ratios.mul(decay_weights, axis=0)

    wiki_logs = np.log(wiki_ratios.where((wiki_ratios > 0) & valid_mask))
    price_logs = np.log(price_ratios.where((price_ratios > 0) & valid_mask))

    # Create data list from valid entries using vectorized operations
    # Stack the logs to create a MultiIndex Series
    wiki_logs_stacked = wiki_logs.stack()
    price_logs_stacked = price_logs.stack()

    # Combine into DataFrame
    df_merged = pd.DataFrame({"wiki_views": wiki_logs_stacked, "stock_price": price_logs_stacked})

    # Drop rows with NaN in either column
    df_merged = df_merged.dropna(how="any")

    # Reset index to get symbol and date columns
    df_merged = df_merged.reset_index()
    df_merged = df_merged.drop("level_0", axis=1)
    df_merged.columns = ["symbol", "wiki_views", "stock_price"]

    if df_merged.empty:
        logger.warning("No valid data found for analysis")
        return False, None, []

    # Step 4: Apply IQR outlier removal
    df_merged = _remove_outliers_iqr(df_merged, ["wiki_views", "stock_price"], iqr_factor)

    logger.debug(
        f"After processing and outlier removal: {len(df_merged)} data points, {len(df_merged['symbol'].unique())} unique stocks"
    )
    return True, df_merged, valid_symbols


def allocate_wikipedia(
    all_stocks: list[StockUniverse],
    time_today: datetime,
    n_historical_days: int = 30,
    n_lag_days: int = 0,
    n_weeks_difference: int = 2,
    min_trading_volume: int = 10_000,
    p_threshold_significance: float = 0.5,
    n_max_final_stocks: int = 200,
    iqr_factor: float = 1.5,
    decay_factor: float = 1.0,
    use_sql_database: bool = True,
    use_volume_filter: bool = True,
    df_wiki_views: Optional[pd.DataFrame] = None,
    df_stock_prices: Optional[pd.DataFrame] = None,
    df_stock_volumes: Optional[pd.DataFrame] = None,
) -> AllocationResult:
    """
    Production-ready Wikipedia-based stock allocation.

    Analyzes historical correlation between Wikipedia page views and stock price movements
    to allocate weights to stocks with significant positive correlations.

    Parameters:
    - all_stocks: List of stocks to consider for allocation
    - time_today: The current date for analysis
    - n_historical_days: Number of historical days to analyze (walking backwards from time_today)
    - n_lag_days: Days lag between Wikipedia views and stock prices
    - n_weeks_difference: Weeks difference for price change comparison
    - min_trading_volume: Minimum trading volume threshold
    - p_threshold_significance: P-value threshold for statistical significance
    - n_max_final_stocks: Maximum number of stocks to select
    - iqr_factor: Interquartile range factor for outlier removal
    - decay_factor: Factor for exponential decay weighting (newest data gets this times higher weight than oldest)
    - use_sql_database: If True, load from SQL database. If False, fetch fresh data from APIs.
    - df_wiki_views: Optional pre-loaded Wikipedia views DataFrame (for testing)
    - df_stock_prices: Optional pre-loaded stock prices DataFrame (for testing)
    - df_stock_volumes: Optional pre-loaded stock volumes DataFrame (for testing)
    """

    end_date = time_today.astimezone(pytz.UTC)
    start_date = end_date - n_lag_days * ONE_DAY - n_historical_days * ONE_DAY - n_weeks_difference * ONE_WEEK

    # Use provided dataframes if available, otherwise load from database/API
    if df_wiki_views is not None and df_stock_prices is not None and df_stock_volumes is not None:
        logger.debug("Using provided dataframes for testing")
    else:
        try:
            df_wiki_views, df_stock_prices, df_stock_volumes = load_data(
                start_date, end_date, all_stocks, use_sql_database
            )
            logger.debug(
                f"Loaded {len(df_wiki_views)} wiki view records, {len(df_stock_prices)} price records, {len(df_stock_volumes)} volume records"
            )

        except Exception as error:
            logger.error(f"Failed to fetch data: {error}")
            return _get_failed_result(end_date, all_stocks)

    # success = _verify_data(
    #     df_wiki_views,
    #     df_stock_prices,
    #     df_stock_volumes,
    #     start_date,
    #     end_date,
    #     all_stocks,
    # )
    # if not success:
    #     logger.error("Data verification failed.")
    #     return _get_failed_result(end_date, all_stocks)

    success, df_merged, valid_symbols = _process_and_filter_data_combined(
        df_wiki_views,
        df_stock_prices,
        df_stock_volumes,
        decay_factor,
        n_historical_days,
        n_lag_days,
        n_weeks_difference,
        min_trading_volume,
        iqr_factor,
    )
    if not success:
        return _get_failed_result(end_date, all_stocks)

    success, significant_positive_stocks = _filter_by_correlations(df_merged, p_threshold_significance)
    if not success:
        return _get_failed_result(end_date, all_stocks)

    df_significant = df_merged[df_merged["symbol"].isin(significant_positive_stocks)]

    success, df_top = _select_top_n_stocks(
        df_significant,
        n_max_final_stocks,
    )
    if not success:
        return _get_failed_result(end_date, all_stocks)

    success, asset_weights = _create_asset_weights(df_top, all_stocks)
    if not success:
        return _get_failed_result(end_date, all_stocks)

    success, wiki_statistics = _estimate_statistics(
        df_top,
        end_date,
        df_wiki_views.columns.tolist(),
        valid_symbols,
        significant_positive_stocks,
        df_top["symbol"].unique().tolist(),
    )
    if not success:
        return _get_failed_result(end_date, all_stocks)

    return AllocationResult(
        asset_weights=asset_weights,
        success=True,
        statistics=wiki_statistics,
    )


if __name__ == "__main__":
    stocks = list_of_dax_stocks()
    time_today = datetime(2024, 6, 1)

    result = allocate_wikipedia(stocks, time_today)
    print(f"Allocation result: {result}")

    # Log performance report
    print("Logging performance report...")
    log_performance_report()
    print("Done.")
