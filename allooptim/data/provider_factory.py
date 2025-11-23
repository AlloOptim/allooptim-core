"""Factory and unified provider for fundamental data sources.

This module provides a clean factory pattern for creating fundamental data providers
with automatic fallback and smart provider selection.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

from allooptim.data.fundamental_data import FundamentalData
from allooptim.data.fundamental_providers import (
    FundamentalDataProvider,
    FundamentalDataStore,
    SimFinProvider,
    YahooFinanceProvider,
)

logger = logging.getLogger(__name__)


class FundamentalDataProviderFactory:
    """Factory for creating fundamental data providers with smart selection.

    This factory automatically selects the best available provider based on:
    - API key availability (SimFin preferred if key present)
    - Requirements (historical vs current data)
    - Configuration preferences

    Examples:
        # Default: auto-detect SimFin, fallback to Yahoo
        provider = FundamentalDataProviderFactory.create_provider()

        # Force Yahoo Finance only
        provider = FundamentalDataProviderFactory.create_provider(prefer_simfin=False)

        # With caching for backtests
        provider = FundamentalDataProviderFactory.create_provider(enable_caching=True)
    """

    @staticmethod
    def create_provider(
        prefer_simfin: bool = True, simfin_api_key: Optional[str] = None, enable_caching: bool = False
    ) -> "UnifiedFundamentalProvider":
        """Create a provider with fallback chain.

        Args:
            prefer_simfin: Try SimFin first if API key available
            simfin_api_key: API key for SimFin (checks env if None)
            enable_caching: Enable data caching for backtests

        Returns:
            UnifiedFundamentalProvider with configured fallback chain
        """
        providers: List[FundamentalDataProvider] = []

        # Try SimFin first if preferred and available
        if prefer_simfin:
            api_key = simfin_api_key or os.getenv("SIMFIN_API_KEY")
            if api_key:
                try:
                    simfin = SimFinProvider(api_key=api_key)
                    providers.append(simfin)
                    logger.debug("SimFin provider initialized")
                except (ImportError, ValueError) as e:
                    logger.info(f"SimFin unavailable: {e}")
            else:
                logger.debug("SimFin API key not found, skipping")

        # Always include Yahoo Finance as fallback
        providers.append(YahooFinanceProvider())
        logger.debug("Yahoo Finance provider initialized")

        return UnifiedFundamentalProvider(providers=providers, enable_caching=enable_caching)


class UnifiedFundamentalProvider:
    """Unified provider that tries multiple sources with fallback.

    This provider attempts to get data from providers in order, falling back
    to the next provider if the current one fails or doesn't support the request.
    """

    def __init__(self, providers: List[FundamentalDataProvider], enable_caching: bool = False):
        """Initialize unified provider.

        Args:
            providers: List of providers to try in order (first = highest priority)
            enable_caching: Whether to enable data caching
        """
        if not providers:
            raise ValueError("At least one fundamental data provider must be provided")

        self.providers = providers
        self.cache = FundamentalDataStore() if enable_caching else None

        # Validate provider chain
        self._validate_provider_chain()

    def get_fundamental_data(self, tickers: List[str], date: Optional[datetime] = None) -> List[FundamentalData]:
        """Get data from first available provider.

        Args:
            tickers: List of stock tickers
            date: Date for historical data (None for current)

        Returns:
            List of FundamentalData objects, one per ticker
        """
        # Check cache first (backtest mode)
        if self.cache and date and self.cache.has_data_for_date(date):
            logger.debug(f"Using cached data for {date}")
            return self.cache.get_data(tickers, date)

        # Try providers in order
        for provider in self.providers:
            # Skip if provider doesn't support historical and date is required
            if date and not provider.supports_historical_data():
                logger.debug(f"Skipping {provider.__class__.__name__} (no historical support)")
                continue

            try:
                data = provider.get_fundamental_data(tickers, date)

                # Cache if enabled
                if self.cache and date:
                    self.cache.store_data(data, date)

                logger.debug(f"Data fetched from {provider.__class__.__name__}")
                return data

            except Exception as e:
                logger.warning(f"{provider.__class__.__name__} failed: {e}")
                continue

        # All providers failed - return empty data
        logger.error("All providers failed, returning empty data")
        return [FundamentalData(ticker=t) for t in tickers]

    def preload_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> None:
        """Preload data for backtest period.

        Args:
            tickers: List of stock tickers
            start_date: Start date for preloading
            end_date: End date for preloading
        """
        if not self.cache:
            logger.warning("Caching not enabled, skipping preload")
            return

        import pandas as pd

        dates = pd.date_range(start=start_date, end=end_date, freq="YE")
        for date in dates:
            logger.info(f"Preloading data for {date}")
            self.get_fundamental_data(tickers, date)

    def supports_historical_data(self) -> bool:
        """Check if any provider supports historical data."""
        return any(provider.supports_historical_data() for provider in self.providers)

    def _validate_provider_chain(self) -> None:
        """Validate that the provider chain is properly configured."""
        if not self.providers:
            raise ValueError("Provider chain cannot be empty")

        # Check that we have at least one provider that can handle current data
        current_providers = [p for p in self.providers if not p.supports_historical_data()]
        if not current_providers:
            logger.warning("No providers configured for current data - historical-only backtests may fail")

        # Check for duplicate provider types (not necessarily an error, but warn)
        provider_types = [type(p).__name__ for p in self.providers]
        if len(provider_types) != len(set(provider_types)):
            logger.info(f"Provider chain contains duplicate types: {provider_types}")

        logger.debug(f"Validated provider chain with {len(self.providers)} providers: {provider_types}")
