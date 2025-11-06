"""Base class for machine learning-based portfolio optimizers.

This module provides common functionality for ML optimizers to eliminate code duplication
while maintaining clear separation between lightweight and heavyweight implementations.

Architecture:
    - BaseMLOptimizer: Abstract base class with common wrapper logic
    - Subclasses implement _create_engine() to instantiate their specific optimizer
    - Shared: fit(), allocate(), data augmentation, validation
    - Specialized: Model architecture, training time, data requirements

Usage:
    Lightweight optimizers (LightGBM):
        - Training time: Seconds
        - Min data: 61 periods
        - Suitable for: Production, daily rebalancing, quick experiments

    Heavyweight optimizers (LSTM/MAMBA/TCN):
        - Training time: Minutes to hours
        - Min data: 500+ periods recommended
        - Suitable for: Research, offline optimization, monthly rebalancing
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import LMoments
from allooptim.optimizer.asset_name_utils import (
    create_weights_series,
    validate_asset_names,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)

# Constants for data validation
MIN_PRICE_OBSERVATIONS_FOR_RETURNS = 2


class BaseMLOptimizerConfig(BaseModel):
    """Configuration parameters for ML-based optimizers."""

    model_config = DEFAULT_PYDANTIC_CONFIG
    update_timedelta: timedelta = timedelta(days=1)
    use_data_augmentation: bool = False
    n_lookback: int = 60
    n_augmentation: int = 10


class BaseMLOptimizer(AbstractOptimizer, ABC):
    """Abstract base class for machine learning-based portfolio optimizers.

    This class handles all common wrapper logic including:
    - Data validation and preprocessing
    - Training data management
    - Incremental updates
    - Data augmentation
    - Weight prediction and normalization

    Subclasses must implement:
    - _create_engine(): Factory method to instantiate the specific optimizer
    - name: Property returning the optimizer name
    """

    def __init__(self, config: Optional[BaseMLOptimizerConfig] = None) -> None:
        """Initialize the base ML optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
        self.config = config or BaseMLOptimizerConfig()
        self._optimizer = None
        self._df_prices: Optional[pd.DataFrame] = None
        self._last_update: Optional[datetime] = None

    @abstractmethod
    def _create_engine(self, n_assets: int, n_lookback: int) -> None:
        """Factory method to create the optimization engine.

        Args:
            n_assets: Number of assets in the portfolio
            n_lookback: Number of lookback periods

        Returns:
            Optimizer instance (e.g., FastPortfolioOptimizer or DeepLearningOptimizer)
        """
        pass

    def fit(self, df_prices: pd.DataFrame) -> None:
        """Fit the optimizer with historical price data.

        Args:
            df_prices: Historical price data with dates as index and assets as columns

        Raises:
            ValueError: If price data is invalid
        """
        # Validation
        if df_prices is None or df_prices.empty:
            raise ValueError("Price data cannot be None or empty")

        if len(df_prices) < MIN_PRICE_OBSERVATIONS_FOR_RETURNS:
            raise ValueError("Need at least 2 price observations to calculate returns")

        if any(df_prices.isnull().values.flatten()):
            raise ValueError("Price data contains NaN values")

        # Store price data
        self._df_prices = df_prices.copy()

        logger.debug(
            f"Stored optimizer with price data: {df_prices.shape[0]} observations, "
            f"{df_prices.shape[1]} assets, date range: {df_prices.index[0]} to {df_prices.index[-1]}"
        )

        # Create optimizer engine
        n_assets = df_prices.shape[1]
        self._optimizer = self._create_engine(
            n_assets,
            n_lookback=self.config.n_lookback,
        )

        # Prepare training data with optional augmentation
        historical_prices = df_prices.values
        augmented_prices, augmented_returns = self._artificial_data_augmentation(historical_prices)

        # Train the engine
        self._optimizer.train(augmented_prices, augmented_returns)

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights based on expected returns and covariance.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Optional price data (not used by ML optimizers)
            df_allocations: Optional previous allocations (not used by ML optimizers)
            time: Current timestamp for incremental updates
            l_moments: Optional L-moments (not used by ML optimizers)

        Returns:
            Portfolio weights as pandas Series with asset names as index
        """
        # Validation
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()
        n_assets = len(asset_names)

        assert self._df_prices is not None, "Optimizer has not been fitted with price data."
        assert self._optimizer is not None, "Optimizer has not been initialized."
        assert n_assets == self._df_prices.shape[1], "Number of assets does not match fitted data."

        # Check if optimizer was successfully trained
        if not self._optimizer.trained:
            logger.warning(f"{self.name} not trained due to insufficient data, returning equal weights")
            equal_weights = np.ones(n_assets) / n_assets
            return create_weights_series(equal_weights, asset_names)

        current_prices = self._df_prices.values[-self.config.n_lookback :]

        # Incremental update if enough time has passed
        if self._last_update is None or (time is not None and time - self._last_update > self.config.update_timedelta):
            self._last_update = time

            # Get augmented data for update
            augmented_prices, augmented_returns = self._artificial_data_augmentation(current_prices)
            self._optimizer.incremental_update(augmented_prices, augmented_returns)

        # Predict optimal weights
        weights = self._optimizer.predict(current_prices)

        return create_weights_series(weights, asset_names)

    def _artificial_data_augmentation(self, current_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation by adding Gaussian noise to returns.

        This technique helps prevent overfitting by creating synthetic training samples
        that maintain the statistical properties of the original data.

        Args:
            current_prices: Price data array (T, n_assets)

        Returns:
            tuple of (augmented_prices, augmented_returns)
        """
        # Calculate returns from prices
        returns_array = np.diff(np.log(current_prices), axis=0)

        # If augmentation is disabled, return original data
        if not self.config.use_data_augmentation:
            return current_prices, returns_array

        # Create augmented samples with noise
        augmented_returns_list = [returns_array]

        for _ in range(self.config.n_augmentation):
            # Add Gaussian noise to returns (1% std)
            noise = np.random.normal(0, 0.01, returns_array.shape)
            augmented_returns = returns_array + noise
            augmented_returns_list.append(augmented_returns)

        # Concatenate all augmented returns
        augmented_returns = np.concatenate(augmented_returns_list, axis=0)

        # Reconstruct prices from augmented returns
        starting_prices = np.tile(current_prices[0, :], (len(augmented_returns) + 1, 1))
        all_returns = np.vstack([np.zeros((1, current_prices.shape[1])), augmented_returns])
        augmented_prices = starting_prices * np.exp(np.cumsum(all_returns, axis=0))

        return augmented_prices, augmented_returns
