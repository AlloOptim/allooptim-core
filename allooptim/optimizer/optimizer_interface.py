"""Portfolio optimizer interfaces and abstract base classes.

This module defines the core interfaces for portfolio optimization algorithms
in AlloOptim. It provides abstract base classes that ensure consistent APIs
across different optimization strategies including traditional mean-variance
optimization, risk parity, hierarchical risk parity, and machine learning-based
approaches.

The interfaces support:
- Standard pandas DataFrame inputs for price/return data
- Flexible risk metric specifications (variance, CVaR, drawdown, L-moments)
- Ensemble optimization combining multiple strategies
- Warm-start capabilities for iterative optimization
- Consistent error handling and validation
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from allooptim.optimizer.allocation_metric import LMoments


class AbstractOptimizer(ABC):
    """Abstract base class for all portfolio optimization algorithms.

    All optimizers in AlloOptim inherit from this class and implement the
    `allocate()` method to compute portfolio weights. This ensures a consistent
    interface across different optimization strategies.

    The optimizer interface is designed for flexibility and composability:
    - Supports various risk metrics (variance, CVaR, max drawdown, L-moments)
    - Handles missing data and edge cases gracefully
    - Maintains asset name consistency throughout pipeline
    - Enables warm-start optimization for performance

    Subclassing Guide:
        1. Inherit from AbstractOptimizer
        2. Implement allocate() method
        3. Implement name property
        4. Add configuration via Pydantic BaseModel
        5. Register config in optimizer registry

    Examples:
        Creating a simple optimizer:

        >>> class MyOptimizer(AbstractOptimizer):
        ...     def allocate(self, ds_mu, df_cov, **kwargs):
        ...         # Equal-weight allocation
        ...         n = len(ds_mu)
        ...         return pd.Series(1 / n, index=ds_mu.index)
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyOptimizer"

        Using an existing optimizer:

        >>> from allooptim.optimizer.efficient_frontier import EfficientFrontierOptimizer
        >>> optimizer = EfficientFrontierOptimizer(risk_aversion=2.0)
        >>> weights = optimizer.allocate(expected_returns, covariance_matrix)
        >>> print(weights.sum())  # Should be 1.0
        1.0

    See Also:
        - :class:`AbstractEnsembleOptimizer`: Base class for ensemble methods
        - :mod:`allooptim.optimizer.optimizer_list`: Available optimizer catalog
        - :mod:`allooptim.optimizer.optimizer_factory`: Optimizer creation utilities
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
        """Optional setup method to prepare the optimizer with historical data."""
        pass

    def reset(self) -> None:
        """Optional method to reset any internal state of the optimizer."""
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
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Create an optimal portfolio allocation with graceful failure handling.

        This method wraps the allocate() method and provides fallback behavior
        when the optimizer fails. For standalone optimizer usage (not in A2A),
        this always uses EQUAL_WEIGHTS fallback on failure.

        Args:
            ds_mu: Expected return vector as pandas Series with asset names as index
            df_cov: Expected covariance matrix as pandas DataFrame with asset names as both index and columns
            df_prices: Optional historical prices DataFrame
            time: Optional timestamp for time-dependent optimizers
            l_moments: Optional L-moments for advanced risk modeling

        Returns:
            Portfolio weights as pandas Series with asset names as index
        """
        try:
            return self.allocate(ds_mu, df_cov, df_prices, time, l_moments)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Optimizer {self.name} failed with {type(e).__name__}: {e}")

            # For standalone usage, always use EQUAL_WEIGHTS fallback
            n_assets = len(ds_mu)
            equal_weight = 1.0 / n_assets
            return pd.Series(equal_weight, index=ds_mu.index, name=self.name)
        """Create an optimal portfolio allocation given the expected returns vector and covariance matrix.

        Args:
            ds_mu: Expected return vector as pandas Series with asset names as index
            df_cov: Expected covariance matrix as pandas DataFrame with asset names as both index and columns
            df_prices: Optional historical prices DataFrame
            time: Optional timestamp for time-dependent optimizers
            l_moments: Optional L-moments for advanced risk modeling

        Returns:
            Portfolio weights as pandas Series with asset names as index

        Note:
            - Asset names in mu.index must match cov.index and cov.columns
            - Returned weights should sum to 1.0 for long-only portfolios
            - Optimizer implementations can access asset names via cov.columns or mu.index
            - Individual optimizers typically ignore df_allocations; ensemble optimizers use it for efficiency
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this optimizer class."""
        pass

    @property
    def display_name(self) -> str:
        """Name of this optimizer. This can depend on the optimizer configuration, if multiple instances exist."""
        return self._display_name if self._display_name is not None else self.name


class AbstractEnsembleOptimizer(ABC):
    """Abstract base class for ensemble portfolio optimization algorithms with pandas interface."""

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
        """Optional setup method to prepare the optimizer with historical data."""
        pass

    def reset(self) -> None:
        """Optional method to reset any internal state of the optimizer."""
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

    @abstractmethod
    def allocate(  # noqa: PLR0913
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        df_allocations: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Create an optimal portfolio allocation given the expected returns vector and covariance matrix.

        Args:
            ds_mu: Expected return vector as pandas Series with asset names as index
            df_cov: Expected covariance matrix as pandas DataFrame with asset names as both index and columns
            df_prices: Optional historical prices DataFrame
            df_allocations: Optional DataFrame with previous optimizer allocations.
                           Rows are optimizer names, columns are asset names, values are allocation weights.
                           Used by ensemble optimizers (e.g., A2A) to avoid re-computation.
            time: Optional timestamp for time-dependent optimizers
            l_moments: Optional L-moments for advanced risk modeling

        Returns:
            Portfolio weights as pandas Series with asset names as index

        Note:
            - Asset names in mu.index must match cov.index and cov.columns
            - Returned weights should sum to 1.0 for long-only portfolios
            - Optimizer implementations can access asset names via cov.columns or mu.index
            - Individual optimizers typically ignore df_allocations; ensemble optimizers use it for efficiency
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this optimizer. The name will be displayed in the MCOS results DataFrame."""
        pass

    @property
    def display_name(self) -> str:
        """Name of this optimizer. This can depend on the optimizer configuration, if multiple instances exist."""
        return self._display_name if self._display_name is not None else self.name
