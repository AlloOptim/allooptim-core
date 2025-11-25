"""Fundamental data providers for portfolio optimization.

This module provides interfaces and implementations for fetching fundamental
company data from various sources. It supports both live trading (yfinance)
and backtesting (SimFin) modes.

Key features:
- Abstract interface for fundamental data providers
- Yahoo Finance implementation for live trading
- SimFin implementation for backtesting
- Data caching and storage for backtesting efficiency
- Unified data format across providers
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from allooptim.data.fundamental_data import FundamentalData

logger = logging.getLogger(__name__)

# Constants for fundamental data processing
DEBT_TO_EQUITY_PERCENTAGE_THRESHOLD = 50  # Above this, likely expressed as percentage


class FundamentalDataProvider(ABC):
    """Abstract base class for fundamental data providers."""

    @abstractmethod
    def get_fundamental_data(self, tickers: List[str], date: Optional[datetime] = None) -> List[FundamentalData]:
        """Get fundamental data for a list of tickers.

        Args:
            tickers: List of ticker symbols
            date: Date for historical data (None for latest)

        Returns:
            List of FundamentalData objects
        """
        pass

    @abstractmethod
    def supports_historical_data(self) -> bool:
        """Return True if provider supports historical fundamental data."""
        pass


class YahooFinanceProvider(FundamentalDataProvider):
    """Yahoo Finance implementation for live trading fundamental data."""

    def supports_historical_data(self) -> bool:
        """Yahoo Finance has limited historical fundamental data."""
        return False

    def get_fundamental_data(self, tickers: List[str], date: Optional[datetime] = None) -> List[FundamentalData]:
        """Get fundamental data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols
            date: Ignored for Yahoo Finance (always latest)

        Returns:
            List of FundamentalData objects
        """
        all_results = []
        batch_size = 100  # Yahoo Finance batch limit

        logger.debug(f"Fetching fundamental data from Yahoo Finance for {len(tickers)} tickers")

        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i : i + batch_size]

            try:
                tickers_data = yf.Tickers(tickers=batch_tickers)

                # Process each ticker in the batch
                for ticker in batch_tickers:
                    try:
                        info = tickers_data.tickers[ticker].info

                        # Extract fundamental metrics
                        market_cap = info.get("marketCap")
                        roe = info.get("returnOnEquity")
                        debt_to_equity = info.get("debtToEquity")
                        pb_ratio = info.get("priceToBook")
                        current_ratio = info.get("currentRatio")

                        # Handle debt_to_equity format variations
                        if (
                            debt_to_equity is not None and debt_to_equity > DEBT_TO_EQUITY_PERCENTAGE_THRESHOLD
                        ):  # Above this, likely percentage
                            debt_to_equity = debt_to_equity / 100.0

                        # Create FundamentalData object
                        fund_data = FundamentalData(
                            ticker=ticker,
                            market_cap=market_cap,
                            roe=roe,
                            debt_to_equity=debt_to_equity,
                            pb_ratio=pb_ratio,
                            current_ratio=current_ratio,
                        )

                        all_results.append(fund_data)

                        if fund_data.is_valid:
                            logger.debug(f"  ✓ {ticker}")
                        else:
                            logger.debug(f"  ✗ {ticker} (no valid data)")

                    except Exception as e:
                        logger.warning(f"  ✗ {ticker}: {e}")
                        # Create empty FundamentalData for failed ticker
                        all_results.append(FundamentalData(ticker=ticker))

            except Exception as e:
                logger.error(f"Yahoo Finance batch failed: {e}")
                # Create empty FundamentalData for all tickers in failed batch
                for ticker in batch_tickers:
                    all_results.append(FundamentalData(ticker=ticker))

        valid_count = sum(1 for data in all_results if data.is_valid)
        logger.debug(f"Yahoo Finance: {valid_count}/{len(tickers)} tickers successful")

        return all_results


class SimFinProvider(FundamentalDataProvider):
    """SimFin implementation for backtesting fundamental data."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize SimFin provider.

        Args:
            api_key: SimFin API key. If None, uses environment variable.
        """
        try:
            import os

            import simfin as sf

            # Set data directory for SimFin (use a cache directory)
            data_dir = os.path.join(os.path.expanduser("~"), ".simfin_cache")
            os.makedirs(data_dir, exist_ok=True)
            sf.set_data_dir(data_dir)

            self.sf = sf
            self.api_key = api_key or os.getenv("SIMFIN_API_KEY")
            sf.set_api_key(self.api_key)
            logger.info("SimFin provider initialized with API key")
        except ImportError as err:
            raise ImportError("simfin package required for SimFin provider. Install with: pip install simfin") from err

    def supports_historical_data(self) -> bool:
        """SimFin supports historical fundamental data."""
        return True

    def get_fundamental_data(self, tickers: List[str], date: Optional[datetime] = None) -> List[FundamentalData]:
        """Get fundamental data from SimFin.

        Args:
            tickers: List of ticker symbols
            date: Date for historical data (uses latest available if None)

        Returns:
            List of FundamentalData objects
        """
        all_results = []

        logger.debug(f"Fetching fundamental data from SimFin for {len(tickers)} tickers")

        try:
            import simfin as sf
            from simfin.names import SIMFIN_ID

            # Load companies dataset to map tickers to SimFin IDs
            companies = sf.load_companies(market="us")
            logger.debug(f"Companies dataset loaded with columns: {list(companies.columns)}")
            logger.debug(f"First few rows of companies data:\n{companies.head()}")

            # The ticker is the index in the companies DataFrame
            # Create ticker to SimFin ID mapping using the index
            ticker_to_simfin = companies[SIMFIN_ID].to_dict()
            logger.debug(f"Loaded {len(ticker_to_simfin)} ticker to SimFin ID mappings")

            # Load all annual datasets once (more complete data than quarterly)
            logger.debug("Loading annual datasets...")
            bs_full = sf.load_balance(variant="annual", market="us")
            pl_full = sf.load_income(variant="annual", market="us")
            cf_full = sf.load_cashflow(variant="annual", market="us")
            logger.debug("Annual datasets loaded successfully")

            # Debug: Print column names
            logger.info(f"Balance Sheet columns: {list(bs_full.columns)[:10]}...")
            logger.info(f"Income Statement columns: {list(pl_full.columns)[:10]}...")
            logger.info(f"Cash Flow columns: {list(cf_full.columns)[:10]}...")

            for ticker in tickers:
                try:
                    # Check if ticker exists in companies data
                    if ticker not in ticker_to_simfin:
                        logger.warning(f"  ✗ {ticker}: Not found in SimFin database")
                        all_results.append(FundamentalData(ticker=ticker))
                        continue

                    # Determine the period to fetch (use the provided date or latest)
                    if date:
                        # Find the most recent year end before or on the given date
                        target_date = date
                        # SimFin typically has annual data, so find the year end
                        year_end = pd.Timestamp(target_date).to_period("YE").end_time
                        if year_end > target_date:
                            # If year end is after target date, use previous year
                            year_end = (pd.Timestamp(target_date) - pd.DateOffset(years=1)).to_period("YE").end_time
                        period_end = year_end
                    else:
                        # Use latest available data
                        period_end = None

                    # Fetch balance sheet data
                    try:
                        if ticker in bs_full.index.get_level_values("Ticker"):
                            company_bs = bs_full.xs(ticker, level="Ticker")
                            logger.debug(f"Balance sheet columns for {ticker}: {list(company_bs.columns)}")
                            if period_end:
                                # Find the most recent period before or equal to period_end
                                available_periods = company_bs.index
                                valid_periods = available_periods[available_periods <= period_end]
                                if len(valid_periods) > 0:
                                    latest_period = valid_periods.max()
                                    bs_data = company_bs.loc[latest_period]
                                    logger.debug(
                                        f"{ticker}: Using balance sheet data for {latest_period}: {bs_data.to_dict()}"
                                    )
                                else:
                                    bs_data = None
                            else:
                                bs_data = company_bs.iloc[-1]  # Latest available
                        else:
                            bs_data = None
                    except Exception as e:
                        logger.warning(f"  Failed to load balance sheet for {ticker}: {e}")
                        bs_data = None

                    # Fetch income statement data
                    try:
                        if ticker in pl_full.index.get_level_values("Ticker"):
                            company_pl = pl_full.xs(ticker, level="Ticker")
                            if period_end:
                                available_periods = company_pl.index
                                valid_periods = available_periods[available_periods <= period_end]
                                if len(valid_periods) > 0:
                                    latest_period = valid_periods.max()
                                    pl_data = company_pl.loc[latest_period]
                                    logger.debug(
                                        f"{ticker}: Using income statement data for {latest_period}: {pl_data.to_dict()}"
                                    )
                                else:
                                    pl_data = None
                            else:
                                pl_data = company_pl.iloc[-1]  # Latest available
                        else:
                            pl_data = None
                    except Exception as e:
                        logger.warning(f"  Failed to load income statement for {ticker}: {e}")
                        pl_data = None

                    # Fetch cash flow data
                    try:
                        if ticker in cf_full.index.get_level_values("Ticker"):
                            company_cf = cf_full.xs(ticker, level="Ticker")
                            if period_end:
                                available_periods = company_cf.index
                                valid_periods = available_periods[available_periods <= period_end]
                                if len(valid_periods) > 0:
                                    latest_period = valid_periods.max()
                                    cf_data = company_cf.loc[latest_period]
                                else:
                                    cf_data = None
                            else:
                                cf_data = company_cf.iloc[-1]  # Latest available
                        else:
                            cf_data = None
                    except Exception as e:
                        logger.warning(f"  Failed to load cash flow for {ticker}: {e}")
                        cf_data = None

                    # Extract fundamental metrics
                    market_cap = None
                    roe = None
                    debt_to_equity = None
                    pb_ratio = None
                    current_ratio = None

                    # ROE (Return on Equity) = Net Income / Shareholder Equity
                    if pl_data is not None and bs_data is not None:
                        # Try different possible column names for net income
                        net_income = None
                        for col_name in [
                            "Net Income",
                            "Net Income/Starting Line",
                            "Net Income Available for Common Shareholders",
                        ]:
                            if col_name in cf_data.index:
                                net_income = cf_data[col_name]
                                break

                        # Try different possible column names for equity
                        shareholder_equity = None
                        for col_name in ["Total Equity", "Shareholders Equity", "Equity"]:
                            if col_name in bs_data.index:
                                shareholder_equity = bs_data[col_name]
                                break

                        logger.debug(f"{ticker}: Net Income = {net_income}, Total Equity = {shareholder_equity}")

                        if net_income is not None and shareholder_equity is not None and shareholder_equity != 0:
                            roe = net_income / shareholder_equity
                            logger.debug(f"{ticker}: Calculated ROE = {roe}")

                    # Debt-to-Equity Ratio = Total Debt / Total Equity
                    if bs_data is not None:
                        # Try different possible column names for liabilities and equity
                        total_liabilities = None
                        for col_name in ["Total Liabilities", "Total Liab"]:
                            if col_name in bs_data.index:
                                total_liabilities = bs_data[col_name]
                                break

                        total_equity = None
                        for col_name in ["Total Equity", "Shareholders Equity", "Equity"]:
                            if col_name in bs_data.index:
                                total_equity = bs_data[col_name]
                                break

                        logger.debug(
                            f"{ticker}: Total Liabilities = {total_liabilities}, Total Equity = {total_equity}"
                        )

                        if total_liabilities is not None and total_equity is not None and total_equity != 0:
                            debt_to_equity = total_liabilities / total_equity
                            logger.debug(f"{ticker}: Calculated Debt/Equity = {debt_to_equity}")

                    # Current Ratio = Current Assets / Current Liabilities
                    if bs_data is not None:
                        # Try different possible column names for current assets and liabilities
                        current_assets = None
                        for col_name in ["Total Current Assets", "Current Assets"]:
                            if col_name in bs_data.index:
                                current_assets = bs_data[col_name]
                                break

                        current_liabilities = None
                        for col_name in ["Total Current Liabilities", "Current Liabilities"]:
                            if col_name in bs_data.index:
                                current_liabilities = bs_data[col_name]
                                break

                        logger.debug(
                            f"{ticker}: Current Assets = {current_assets}, Current Liabilities = {current_liabilities}"
                        )

                        if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
                            current_ratio = current_assets / current_liabilities
                            logger.debug(f"{ticker}: Calculated Current Ratio = {current_ratio}")

                    logger.debug(
                        f"{ticker}: Final data - ROE: {roe}, D/E: {debt_to_equity}, Current: {current_ratio}, is_valid: {any([roe is not None, debt_to_equity is not None, current_ratio is not None])}"
                    )

                    # Create FundamentalData object
                    fund_data = FundamentalData(
                        ticker=ticker,
                        market_cap=market_cap,
                        roe=roe,
                        debt_to_equity=debt_to_equity,
                        pb_ratio=pb_ratio,
                        current_ratio=current_ratio,
                    )

                    all_results.append(fund_data)
                    logger.debug(f"  ✓ {ticker} (SimFin data fetched)")

                except Exception as e:
                    logger.warning(f"  ✗ {ticker}: {e}")
                    all_results.append(FundamentalData(ticker=ticker))

        except Exception as e:
            logger.error(f"SimFin data fetch failed: {e}")
            # Return empty data for all tickers
            for ticker in tickers:
                all_results.append(FundamentalData(ticker=ticker))

        valid_count = sum(1 for data in all_results if data.is_valid)
        logger.debug(f"SimFin: {valid_count}/{len(tickers)} tickers successful")

        return all_results


class FundamentalDataStore:
    """Storage and caching for fundamental data during backtesting."""

    def __init__(self):
        """Initialize the fundamental data store."""
        self._data_cache: Dict[str, FundamentalData] = {}
        self._last_update: Optional[datetime] = None

    def store_data(self, data: List[FundamentalData], date: datetime) -> None:
        """Store fundamental data for a specific date.

        Args:
            data: List of FundamentalData objects
            date: Date the data corresponds to
        """
        date_str = date.strftime("%Y-%m-%d")
        logger.debug(f"Storing {len(data)} fundamental data points for {date_str}")

        for fund_data in data:
            key = f"{fund_data.ticker}_{date_str}"
            self._data_cache[key] = fund_data
            logger.debug(
                f"  Stored {key}: valid={fund_data.is_valid}, roe={fund_data.roe}, debt_to_equity={fund_data.debt_to_equity}, current_ratio={fund_data.current_ratio}"
            )

        self._last_update = date
        logger.debug(f"Stored {len(data)} fundamental data points for {date_str}. Cache size: {len(self._data_cache)}")

    def get_data(self, tickers: List[str], date: datetime) -> List[FundamentalData]:
        """Retrieve fundamental data for tickers on a specific date.

        Args:
            tickers: List of ticker symbols
            date: Date to retrieve data for

        Returns:
            List of FundamentalData objects (empty objects for missing data)
        """
        results = []
        date_str = date.strftime("%Y-%m-%d")

        logger.info(f"Retrieving fundamental data for {len(tickers)} tickers on {date_str}")
        logger.info(f"Cache size: {len(self._data_cache)}")
        logger.info(f"Cache keys sample: {list(self._data_cache.keys())[:10]}...")

        for ticker in tickers:
            key = f"{ticker}_{date_str}"
            logger.debug(f"Looking for key: {key}")
            if key in self._data_cache:
                data = self._data_cache[key]
                logger.debug(
                    f"  Found cached data for {ticker}: valid={data.is_valid}, roe={data.roe}, debt_to_equity={data.debt_to_equity}, current_ratio={data.current_ratio}"
                )
                results.append(data)
            else:
                logger.debug(f"  No cached data for {ticker} with key {key}")
                results.append(FundamentalData(ticker=ticker))

        return results

    def has_data_for_date(self, date: datetime) -> bool:
        """Check if we have data for a specific date.

        Args:
            date: Date to check

        Returns:
            True if data exists for this date
        """
        date_str = date.strftime("%Y-%m-%d")
        return any(key.endswith(f"_{date_str}") for key in self._data_cache)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._data_cache.clear()
        self._last_update = None
        logger.debug("Fundamental data cache cleared")

    @property
    def cache_size(self) -> int:
        """Get the number of cached data points."""
        return len(self._data_cache)


class FundamentalDataManager:
    """Manager for fundamental data providers and storage.

    DEPRECATED: Use UnifiedFundamentalProvider directly instead.
    This class is maintained for backward compatibility only.
    """

    def __init__(self, mode: str = "live"):
        """Initialize the fundamental data manager.

        Args:
            mode: DEPRECATED - ignored, kept for backward compatibility
        """
        import warnings

        warnings.warn(
            "FundamentalDataManager is deprecated. Use UnifiedFundamentalProvider directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from allooptim.data.provider_factory import FundamentalDataProviderFactory

        self.provider = FundamentalDataProviderFactory.create_provider()

        logger.info("FundamentalDataManager initialized (deprecated - use UnifiedFundamentalProvider)")

    def get_fundamental_data(self, tickers: List[str], date: Optional[datetime] = None) -> List[FundamentalData]:
        """Get fundamental data using the unified provider.

        Args:
            tickers: List of ticker symbols
            date: Date for data

        Returns:
            List of FundamentalData objects
        """
        return self.provider.get_fundamental_data(tickers, date)
