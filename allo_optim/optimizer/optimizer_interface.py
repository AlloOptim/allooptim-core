from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from allo_optim.optimizer.allocation_metric import LMoments


class AbstractOptimizer(ABC):
	"""
	Abstract base class for portfolio optimization algorithms with pandas interface.

	This interface defines the standard contract for all portfolio optimizers in the system.
	All optimizers work with pandas Series/DataFrame inputs to provide direct access to asset
	names and maintain data integrity throughout the optimization process.

	Key Features:
	    - Asset name accessibility via mu.index and cov.columns
	    - Automatic asset name validation and consistency checks
	    - Pandas-based interface for seamless data manipulation
	    - Consistent return format with preserved asset identities

	Examples:
	    Basic optimizer usage with asset name access:

	    >>> optimizer = MaxSharpeOptimizer()
	    >>> weights = optimizer.allocate(mu, cov)
	    >>> asset_names = weights.index.tolist()  # Direct access to asset names
	    >>> for asset, weight in weights.items():
	    ...     print(f"{asset}: {weight:.3f}")
	    AAPL: 0.250
	    GOOGL: 0.300
	    ...

	    Accessing asset names from inputs:

	    >>> available_assets = mu.index.tolist()  # From expected returns
	    >>> covariance_assets = cov.columns.tolist()  # From covariance matrix
	    >>> # Both should be identical for valid inputs
	"""

	def fit(
		self,
		df_prices: Optional[pd.DataFrame] = None,
	) -> None:
		"""Optional setup method to prepare the optimizer with historical data"""
		pass

	def reset(self) -> None:
		"""Optional method to reset any internal state of the optimizer"""
		self.__init__()

	@abstractmethod
	def allocate(
		self,
		ds_mu: pd.Series,
		df_cov: pd.DataFrame,
		df_prices: Optional[pd.DataFrame] = None,
		time: Optional[datetime] = None,
		l_moments: Optional[LMoments] = None,
	) -> pd.Series:
		"""
		Create an optimal portfolio allocation given the expected returns vector and covariance matrix.

		Args:
		    mu: Expected return vector as pandas Series with asset names as index
		    cov: Expected covariance matrix as pandas DataFrame with asset names as both index and columns
		    time: Optional timestamp for time-dependent optimizers
		    l_moments: Optional L-moments for advanced risk modeling
		    df_allocations: Optional DataFrame with previous optimizer allocations.
		                   Rows are optimizer names, columns are asset names, values are allocation weights.
		                   Used by ensemble optimizers (e.g., A2A) to avoid re-computation.

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
		"""
		Name of this optimizer. The name will be displayed in the MCOS results DataFrame.
		"""
		pass


class AbstractEnsembleOptimizer(ABC):
	"""
	Abstract base class for ensemble portfolio optimization algorithms with pandas interface.
	"""

	def fit(
		self,
		df_prices: Optional[pd.DataFrame] = None,
	) -> None:
		"""Optional setup method to prepare the optimizer with historical data"""
		pass

	def reset(self) -> None:
		"""Optional method to reset any internal state of the optimizer"""
		self.__init__()

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
		"""
		Create an optimal portfolio allocation given the expected returns vector and covariance matrix.

		Args:
		    df_allocations: Optional DataFrame with previous optimizer allocations.
		                   Rows are optimizer names, columns are asset names, values are allocation weights.
		                   Used by ensemble optimizers (e.g., A2A) to avoid re-computation.

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
		"""
		Name of this optimizer. The name will be displayed in the MCOS results DataFrame.
		"""
		pass
