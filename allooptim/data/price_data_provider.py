"""Price data provider for backtesting and orchestrator compatibility."""

import logging

import numpy as np
import pandas as pd

from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.optimizer.allocation_metric import (
    MIN_OBSERVATIONS,
    LMoments,
    estimate_linear_moments,
)

logger = logging.getLogger(__name__)


class PriceDataProvider(AbstractObservationSimulator):
    """Simple data provider that wraps price DataFrame for orchestrator compatibility."""

    def __init__(self, price_data: pd.DataFrame):
        """Initialize the price data provider.

        Args:
            price_data: Historical price data with datetime index and asset columns.
        """
        self.price_data = price_data
        # Calculate basic statistics (simplified) - return as pandas objects
        returns = price_data.pct_change().dropna()
        self._mu = returns.mean()
        self._cov = returns.cov()

        # Compute L-moments from returns data
        if len(returns) >= MIN_OBSERVATIONS:
            self._l_moments = estimate_linear_moments(returns.values)
        else:
            # Fallback: create zero L-moments with correct structure
            n_assets = len(price_data.columns)
            self._l_moments = LMoments(
                lt_comoment_1=np.zeros((n_assets, n_assets)),
                lt_comoment_2=np.zeros((n_assets, n_assets)),
                lt_comoment_3=np.zeros((n_assets, n_assets)),
                lt_comoment_4=np.zeros((n_assets, n_assets)),
            )

    @property
    def mu(self):
        """Get expected returns as pandas Series."""
        return self._mu

    @property
    def cov(self):
        """Get covariance matrix as pandas DataFrame."""
        return self._cov

    @property
    def historical_prices(self):
        """Get historical price data as pandas DataFrame."""
        return self.price_data

    @property
    def n_observations(self):
        """Get number of observations in the price data."""
        return len(self.price_data)

    def get_sample(self):
        """Get sample market parameters (same as ground truth for backtest)."""
        return self.get_ground_truth()

    def get_ground_truth(self):
        """Get ground truth market parameters from historical data."""
        time_end = self.price_data.index[-1]  # Last timestamp in the data
        return self._mu, self._cov, self.price_data, time_end, self._l_moments

    @property
    def name(self) -> str:
        """Get the data provider name identifier."""
        return "BacktestDataProvider"
