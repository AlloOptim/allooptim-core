"""Base ensemble optimizer implementation with common functionality.

This module provides the EnsembleBaseOptimizer class that implements common
functionality shared across all ensemble portfolio optimizers.

The hierarchy is:
- AbstractEnsembleOptimizer (pure interface)
- EnsembleBaseOptimizer (common implementation)
- ConcreteEnsembleOptimizer (specific ensemble algorithms)
"""

from typing import Optional

import pandas as pd

from allooptim.optimizer.optimizer_interface import AbstractEnsembleOptimizer


class EnsembleBaseOptimizer(AbstractEnsembleOptimizer):
    """Base implementation for ensemble portfolio optimization algorithms.

    This class provides common functionality that all ensemble optimizers share:
    - Initialization with display names
    - Fitting interface for historical data
    - Reset functionality
    - Cash and leverage configuration
    - Display name management

    Ensemble optimizers typically combine results from multiple individual
    optimizers, such as A2A (Allocation-to-Allocators) methods.

    All concrete ensemble optimizers should inherit from this class rather than
    AbstractEnsembleOptimizer directly.

    Examples:
        >>> class MyEnsembleOptimizer(EnsembleBaseOptimizer):
        ...     def allocate(self, ds_mu, df_cov, df_allocations=None, **kwargs):
        ...         if df_allocations is not None:
        ...             return df_allocations.mean(axis=0)
        ...         return pd.Series(1 / len(ds_mu), index=ds_mu.index)
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyEnsemble"
    """

    def __init__(self, display_name: Optional[str] = None):
        """Initialize the ensemble optimizer.

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

        This method can be overridden by ensemble optimizers that need to learn
        from historical data or optimizer performance patterns.

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

    @property
    def display_name(self) -> str:
        """Display name of this optimizer instance.

        Returns:
            The display name if set, otherwise the optimizer's name property.
        """
        return self._display_name if self._display_name is not None else self.name
