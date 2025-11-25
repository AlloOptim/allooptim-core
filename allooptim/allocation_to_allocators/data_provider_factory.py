"""Data Provider Factory.

Factory for creating data providers with time-step alignment abstraction.
Provides clean interface for creating observation simulators for different contexts.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)


class AbstractDataProviderFactory(ABC):
    """Abstract factory for creating data providers with time-step alignment.

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
        """Create a data provider for the given context.

        Args:
            df_prices: Historical price data
            time_current: Current time point (for backtest context)
            lookback_days: Number of days to look back (for estimation window)

        Returns:
            Configured data provider
        """
        pass
