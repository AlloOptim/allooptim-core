"""Base optimizer implementation with common functionality.

This module provides the BaseOptimizer class that implements common functionality
shared across all portfolio optimizers, such as initialization, fitting, and
configuration management.

The hierarchy is:
- AbstractOptimizer (pure interface)
- BaseOptimizer (common implementation)
- ConcreteOptimizer (specific algorithms)
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class BaseOptimizer(AbstractOptimizer):
    """Base implementation for portfolio optimization algorithms.

    This class provides common functionality that all optimizers share:
    - Initialization with display names
    - Fitting interface for historical data
    - Reset functionality
    - Cash and leverage configuration
    - Display name management
    - Graceful failure handling with fallback to equal weights

    All concrete optimizers should inherit from this class rather than
    AbstractOptimizer directly.

    Examples:
        >>> class MyOptimizer(BaseOptimizer):
        ...     def allocate(self, ds_mu, df_cov, **kwargs):
        ...         return pd.Series(1 / len(ds_mu), index=ds_mu.index)
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyOptimizer"
    """

    def __init__(self, display_name: Optional[str] = None):
        """Initialize the optimizer.

        Args:
            display_name: Optional display name for this optimizer instance.
                         If None, defaults to the optimizer's name property.
        """
        self._display_name = display_name
        self.allow_cash = False  # Default to False for backward compatibility
        self.max_leverage = None  # Default to None (no leverage limit)

    def fit(
        self,
        df_prices: Optional[pd.DataFrame] = None,
    ) -> None:
        """Optional setup method to prepare the optimizer with historical data.

        This method can be overridden by optimizers that need to learn from
        historical data (e.g., machine learning models, momentum-based strategies).

        Args:
            df_prices: Historical price data for fitting the optimizer.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state of the optimizer.

        This method should restore the optimizer to its initial state,
        clearing any learned parameters or cached computations.
        """
        self.__init__(self._display_name)

    def set_allow_cash(self, allow_cash: bool) -> None:
        """Set whether this optimizer is allowed to use cash (partial investment).

        Args:
            allow_cash: Whether to allow cash positions (sum of weights < 1.0)
        """
        self.allow_cash = allow_cash

    def set_max_leverage(self, max_leverage: Optional[float]) -> None:
        """Set the maximum leverage allowed for this optimizer.

        Args:
            max_leverage: Maximum leverage factor (sum of weights <= max_leverage).
                         None means no leverage limit.
        """
        self.max_leverage = max_leverage

    def allocate_safe(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Allocate portfolio weights with graceful failure handling.

        This method wraps the allocate() call with error handling. If the
        optimization fails for any reason, it falls back to equal weights
        across all assets.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Optional historical price data
            time: Optional timestamp for time-dependent optimizers
            l_moments: Optional L-moments for advanced risk modeling

        Returns:
            Portfolio weights as pandas Series. On failure, returns equal weights.
        """
        try:
            # Attempt normal allocation
            return self.allocate(ds_mu, df_cov, df_prices, time, l_moments)
        except Exception as e:
            # Log the error
            logger.warning(
                f"{self.display_name} failed with {type(e).__name__}: {e}. " f"Falling back to equal weights."
            )

            # Fallback to equal weights
            asset_names = ds_mu.index.tolist()
            n_assets = len(asset_names)
            equal_weight = 1.0 / n_assets
            weights = np.full(n_assets, equal_weight)

            return pd.Series(weights, index=asset_names, name=self.display_name)

    @property
    def display_name(self) -> str:
        """Display name of this optimizer instance.

        Returns:
            The display name if set, otherwise the optimizer's name property.
        """
        return self._display_name if self._display_name is not None else self.name
