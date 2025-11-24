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
    """Abstract interface for all portfolio optimization algorithms.

    All optimizers in AlloOptim implement this interface to ensure a consistent
    API across different optimization strategies. This is a pure interface with
    no implementation details.

    The optimizer interface is designed for flexibility and composability:
    - Supports various risk metrics (variance, CVaR, max drawdown, L-moments)
    - Handles missing data and edge cases gracefully
    - Maintains asset name consistency throughout pipeline
    - Enables warm-start optimization for performance

    Subclassing Guide:
        1. Inherit from BaseOptimizer (not AbstractOptimizer directly)
        2. Implement allocate() method
        3. Implement name property
        4. Add configuration via Pydantic BaseModel
        5. Register config in optimizer registry

    Examples:
        Creating a simple optimizer:

        >>> class MyOptimizer(BaseOptimizer):
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
        - :class:`BaseOptimizer`: Base implementation with common functionality
        - :class:`AbstractEnsembleOptimizer`: Interface for ensemble methods
        - :mod:`allooptim.optimizer.optimizer_list`: Available optimizer catalog
        - :mod:`allooptim.optimizer.optimizer_factory`: Optimizer creation utilities
    """

    is_wiki_optimizer: bool
    is_fundamental_optimizer: bool

    @abstractmethod
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

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state of the optimizer.

        This method should restore the optimizer to its initial state,
        clearing any learned parameters or cached computations.
        """
        pass

    @abstractmethod
    def set_allow_cash(self, allow_cash: bool) -> None:
        """Set whether this optimizer is allowed to use cash (partial investment).

        Args:
            allow_cash: Whether to allow cash positions (sum of weights < 1.0)
        """
        pass

    @abstractmethod
    def set_max_leverage(self, max_leverage: Optional[float]) -> None:
        """Set the maximum leverage allowed for this optimizer.

        Args:
            max_leverage: Maximum leverage factor (sum of weights <= max_leverage).
                         None means no leverage limit.
        """
        pass

    @abstractmethod
    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Create an optimal portfolio allocation.

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
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this optimizer class."""
        pass

    @property
    def display_name(self) -> str:
        """Display name of this optimizer instance."""
        pass


class AbstractEnsembleOptimizer(ABC):
    """Abstract interface for ensemble portfolio optimization algorithms."""

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
        """Create an optimal ensemble portfolio allocation.

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
    @abstractmethod
    def display_name(self) -> str:
        """Display name of this optimizer instance."""
        pass
