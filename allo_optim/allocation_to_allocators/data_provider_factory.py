"""
Data Provider Factory

Factory for creating data providers with time-step alignment abstraction.
Provides clean interface for creating observation simulators for different contexts.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd
from allooptim.allocation_to_allocators.observation_simulator import (
    MuCovPartialObservationSimulator,
)
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)


class AbstractDataProviderFactory(ABC):
    """
    Abstract factory for creating data providers with time-step alignment.

    This factory pattern abstracts the creation of observation simulators,
    ensuring proper time-step alignment and context-specific configuration.
    """

    @abstractmethod
    def create_data_provider(
        self,
        df_prices: pd.DataFrame,
        time_current: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
    ) -> AbstractObservationSimulator:
        """
        Create a data provider for the given context.

        Args:
            df_prices: Historical price data
            time_current: Current time point (for backtest context)
            lookback_days: Number of days to look back (for estimation window)

        Returns:
            Configured data provider
        """
        pass


class BacktestDataProviderFactory(AbstractDataProviderFactory):
    """
    Factory for creating data providers in backtest context.

    Creates data providers that use the full available historical data
    for ground truth estimation, suitable for backtesting scenarios.
    """

    def create_data_provider(
        self,
        df_prices: pd.DataFrame,
        time_current: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
    ) -> AbstractObservationSimulator:
        """
        Create a backtest data provider.

        For backtesting, we use a simple wrapper that provides ground truth
        from the available historical data.

        Args:
            df_prices: Historical price data for the current period
            time_current: Current time point (unused in backtest context)
            lookback_days: Lookback window (unused in backtest context)

        Returns:
            Data provider configured for backtest use
        """
        # Import here to avoid circular imports
        from allooptim.backtest.backtest_engine import _PriceDataProvider

        return _PriceDataProvider(df_prices)


class MCOSDataProviderFactory(AbstractDataProviderFactory):
    """
    Factory for creating data providers in MCOS simulation context.

    Creates data providers that generate synthetic observations for
    Monte Carlo simulation and error estimation.
    """

    def __init__(self, n_simulations: int = 100):
        """
        Initialize MCOS data provider factory.

        Args:
            n_simulations: Number of simulation observations to generate
        """
        self.n_simulations = n_simulations

    def create_data_provider(
        self,
        df_prices: pd.DataFrame,
        time_current: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
    ) -> AbstractObservationSimulator:
        """
        Create an MCOS data provider.

        For MCOS simulations, we use partial observation simulation
        to generate synthetic data for error estimation.

        Args:
            df_prices: Historical price data for simulation
            time_current: Current time point (unused in MCOS context)
            lookback_days: Lookback window (unused in MCOS context)

        Returns:
            Data provider configured for MCOS simulation
        """
        return MuCovPartialObservationSimulator(df_prices, self.n_simulations)


# Default factory instances
backtest_factory = BacktestDataProviderFactory()
mcos_factory = MCOSDataProviderFactory()


def get_data_provider_factory(context: str = "backtest") -> AbstractDataProviderFactory:
    """
    Get the appropriate data provider factory for the given context.

    Args:
        context: Context type ("backtest" or "mcos")

    Returns:
        Configured data provider factory

    Raises:
        ValueError: If context is not recognized
    """
    if context == "backtest":
        return backtest_factory
    elif context == "mcos":
        return mcos_factory
    else:
        raise ValueError(f"Unknown context: {context}. Must be 'backtest' or 'mcos'")
