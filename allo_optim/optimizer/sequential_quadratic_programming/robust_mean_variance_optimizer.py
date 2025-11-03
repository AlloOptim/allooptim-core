"""
Robust Mean-Variance Portfolio Optimizer

This module implements robust portfolio optimization with ellipsoidal uncertainty sets
for both expected returns and covariance matrix. Unlike standard mean-variance optimization
which assumes perfect knowledge of parameters, this optimizer explicitly accounts for
estimation error by solving a worst-case optimization problem.

Key Features:
    - Handles uncertainty in expected returns (μ) and covariance matrix (Σ)
    - Uses ellipsoidal uncertainty sets for tractable robust counterpart
    - Allows cash holding (0 ≤ sum(weights) ≤ 1) when conditions are unfavorable
    - Conservative by design - protects against parameter estimation errors
    - Based on Ben-Tal & Nemirovski robust optimization framework

Theoretical Background:
    Standard MVO: max_w (w'μ - λw'Σw)

    Robust MVO: max_w min_{μ̃,Σ̃ ∈ U} (w'μ̃ - λw'Σ̃w)

    Where uncertainty sets:
        μ̃ = μ + δμ  with ||δμ|| ≤ ε_μ  (ellipsoidal uncertainty in mean)
        Σ̃ = Σ + ΔΣ  with ||ΔΣ||_F ≤ ε_Σ (Frobenius norm uncertainty in covariance)

    Robust counterpart (conservative approximation):
        max_w (w'μ - ε_μ||w|| - λw'(Σ + ε_ΣI)w)
        subject to: w ≥ 0, sum(w) ≤ 1

    The ε terms represent uncertainty levels:
        - Larger ε → more conservative (protects against larger errors)
        - Smaller ε → less conservative (closer to standard MVO)

References:
    - Ben-Tal, A. & Nemirovski, A. (1998). "Robust Convex Optimization"
    - Ben-Tal, A., El Ghaoui, L., Nemirovski, A. (2009). "Robust Optimization"
    - Fabozzi, F. et al. (2007). "Robust Portfolio Optimization and Management"
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import LMoments
from allo_optim.optimizer.asset_name_utils import (
	convert_pandas_to_numpy,
	create_weights_series,
	get_asset_names,
	validate_asset_names,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.sequential_quadratic_programming.sqp_multistart import minimize_with_multistart

logger = logging.getLogger(__name__)


class RobustMeanVarianceOptimizerConfig(BaseModel):
	"""Configuration for Robust Mean-Variance optimizer."""

	model_config = DEFAULT_PYDANTIC_CONFIG

	risk_aversion: float = Field(
		default=1.0,
		ge=0.0,
		description="Risk aversion parameter λ (higher = more risk averse)",
	)

	mu_uncertainty_level: float = Field(
		default=0.1,
		ge=0.0,
		le=1.0,
		description="Uncertainty level ε_μ for expected returns (0 = no uncertainty, 1 = max uncertainty)",
	)

	cov_uncertainty_level: float = Field(
		default=0.1,
		ge=0.0,
		le=1.0,
		description="Uncertainty level ε_Σ for covariance matrix (0 = no uncertainty, 1 = max uncertainty)",
	)

	uncertainty_estimation_method: str = Field(
		default="bootstrap",
		description="Method to estimate uncertainty: 'fixed', 'bootstrap', 'historical_std'",
	)

	allow_cash: bool = Field(
		default=True,
		description="Whether to allow cash holding (sum(w) < 1) when conditions are unfavorable",
	)

	min_total_weight: float = Field(
		default=0.0,
		ge=0.0,
		le=1.0,
		description="Minimum total weight to invest (0 = can hold all cash, 1 = must be fully invested)",
	)

	bootstrap_samples: int = Field(
		default=100,
		ge=10,
		description="Number of bootstrap samples for uncertainty estimation",
	)


class RobustMeanVarianceOptimizer(AbstractOptimizer):
	"""
	Robust Mean-Variance Portfolio Optimizer with Ellipsoidal Uncertainty.

	This optimizer solves a worst-case optimization problem that protects against
	parameter estimation errors in both expected returns and covariance matrix.
	Instead of assuming perfect knowledge of μ and Σ, it optimizes for the worst-case
	scenario within specified uncertainty sets.

	The optimizer is more conservative than standard mean-variance optimization,
	making it suitable for uncertain market conditions or when parameter estimates
	are unreliable. It can choose to hold cash (not fully invest) if the worst-case
	scenarios are too pessimistic.

	Key Differentiators from Other Optimizers:
	    - Standard MVO: Assumes perfect knowledge of μ and Σ
	    - Kelly: Uses win probabilities, maximizes geometric mean
	    - RiskParity: Ignores returns, equalizes risk contributions
	    - Robust MVO: Explicitly models parameter uncertainty, worst-case optimization

	Implementation:
	    The robust counterpart is solved using scipy.optimize with a conservative
	    approximation that accounts for uncertainty through:
	    1. Mean uncertainty: Subtract ε_μ * ||w|| from expected return
	    2. Covariance uncertainty: Add ε_Σ * I to covariance matrix

	    This results in a tractable quadratic program that can be solved efficiently.

	Examples:
	    >>> config = RobustMeanVarianceOptimizerConfig(mu_uncertainty_level=0.2,
	    ...                                           cov_uncertainty_level=0.1, allow_cash=True)
	    >>> optimizer = RobustMeanVarianceOptimizer(config)
	    >>> optimizer.fit(df_prices)  # Estimate uncertainty from data
	    >>> weights = optimizer.allocate(mu, cov)
	    >>> print(f"Total investment: {weights.sum():.2%}")  # May be < 100% if conservative

	Args:
	    config: Configuration object with robust optimization parameters

	Attributes:
	    estimated_mu_uncertainty: Estimated uncertainty level for μ from historical data
	    estimated_cov_uncertainty: Estimated uncertainty level for Σ from historical data
	"""

	def __init__(self, config: Optional[RobustMeanVarianceOptimizerConfig] = None) -> None:
		self.config = config or RobustMeanVarianceOptimizerConfig()
		self.estimated_mu_uncertainty: Optional[float] = None
		self.estimated_cov_uncertainty: Optional[float] = None
		self._previous_best_weights: Optional[np.ndarray] = None

	def fit(self, df_prices: Optional[pd.DataFrame] = None) -> None:
		"""
		Estimate uncertainty levels from historical price data.

		Uses bootstrap resampling or historical standard deviation to estimate
		the magnitude of parameter estimation errors. These estimates can override
		the fixed uncertainty levels in the config.

		Args:
		    df_prices: DataFrame with datetime index and asset columns containing prices

		Methods:
		    - 'fixed': Use config values directly (no estimation)
		    - 'bootstrap': Bootstrap resampling to estimate parameter variability
		    - 'historical_std': Use rolling window standard deviation of estimates
		"""
		if df_prices is None or len(df_prices) < 30:
			logger.warning("Not enough price history for uncertainty estimation, using fixed levels")
			self.estimated_mu_uncertainty = self.config.mu_uncertainty_level
			self.estimated_cov_uncertainty = self.config.cov_uncertainty_level
			return

		if self.config.uncertainty_estimation_method == "fixed":
			self.estimated_mu_uncertainty = self.config.mu_uncertainty_level
			self.estimated_cov_uncertainty = self.config.cov_uncertainty_level
			logger.debug("Using fixed uncertainty levels from config")
			return

		# Calculate returns
		returns = df_prices.pct_change().dropna()

		if self.config.uncertainty_estimation_method == "bootstrap":
			# Bootstrap estimation of parameter uncertainty
			n_samples = self.config.bootstrap_samples
			n_obs = len(returns)

			# Store bootstrap estimates
			bootstrap_mus = []
			bootstrap_covs = []

			for _ in range(n_samples):
				# Resample with replacement
				sample_indices = np.random.choice(n_obs, size=n_obs, replace=True)
				sample_returns = returns.iloc[sample_indices]

				# Estimate parameters
				bootstrap_mus.append(sample_returns.mean().values)
				bootstrap_covs.append(sample_returns.cov().values)

			bootstrap_mus = np.array(bootstrap_mus)
			bootstrap_covs = np.array(bootstrap_covs)

			# Estimate uncertainty as standard deviation of estimates
			# For μ: use norm of standard deviation vector
			mu_std = np.std(bootstrap_mus, axis=0)
			self.estimated_mu_uncertainty = np.linalg.norm(mu_std)

			# For Σ: use Frobenius norm of standard deviation
			cov_std = np.std(bootstrap_covs, axis=0)
			self.estimated_cov_uncertainty = np.linalg.norm(cov_std, "fro")

			logger.debug(
				f"Bootstrap uncertainty estimation: ε_μ={self.estimated_mu_uncertainty:.4f}, "
				f"ε_Σ={self.estimated_cov_uncertainty:.4f} (from {n_samples} bootstrap samples)"
			)

		elif self.config.uncertainty_estimation_method == "historical_std":
			# Use rolling window to estimate parameter stability
			window = min(60, len(returns) // 2)  # Use half the data or 60 days

			rolling_mu = returns.rolling(window).mean()
			rolling_cov_diag = returns.rolling(window).var()  # Diagonal elements

			# Estimate uncertainty from variation in rolling estimates
			mu_variation = rolling_mu.std().values
			self.estimated_mu_uncertainty = np.linalg.norm(mu_variation)

			cov_variation = rolling_cov_diag.std().values
			self.estimated_cov_uncertainty = np.linalg.norm(cov_variation)

			logger.debug(
				f"Historical std uncertainty estimation: ε_μ={self.estimated_mu_uncertainty:.4f}, "
				f"ε_Σ={self.estimated_cov_uncertainty:.4f} (window={window})"
			)

		else:
			logger.warning(f"Unknown uncertainty estimation method: {self.config.uncertainty_estimation_method}")
			self.estimated_mu_uncertainty = self.config.mu_uncertainty_level
			self.estimated_cov_uncertainty = self.config.cov_uncertainty_level

	def allocate(
		self,
		ds_mu: pd.Series,
		df_cov: pd.DataFrame,
		df_prices: Optional[pd.DataFrame] = None,
		df_allocations: Optional[pd.DataFrame] = None,
		time: Optional[datetime] = None,
		l_moments: Optional[LMoments] = None,
	) -> pd.Series:
		"""
		Calculate robust portfolio weights using worst-case optimization.

		Solves the robust counterpart:
		    max_w (w'μ - ε_μ||w|| - λw'(Σ + ε_ΣI)w)
		    subject to: w ≥ 0, sum(w) ≤ 1, sum(w) ≥ min_total_weight

		The worst-case formulation penalizes the expected return by the uncertainty
		in mean (ε_μ||w||) and inflates the covariance matrix (Σ + ε_ΣI) to account
		for estimation error. This makes the optimization conservative.

		Args:
		    ds_mu: Expected returns as pandas Series with asset names
		    df_cov: Covariance matrix as pandas DataFrame
		    df_prices: Optional price history (triggers fit if not done yet)
		    df_allocations: Optional previous allocations (not used)
		    time: Optional timestamp (not used)
		    l_moments: Optional L-moments (not used)

		Returns:
		    Portfolio weights as pandas Series with asset names
		    - sum(weights) may be < 1.0 if allow_cash=True and conditions are unfavorable
		    - sum(weights) ≥ min_total_weight always satisfied
		"""
		# Validate inputs
		validate_asset_names(ds_mu, df_cov)
		asset_names = get_asset_names(mu=ds_mu)

		# Convert to numpy
		mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)

		# Fit if needed
		if df_prices is not None and self.estimated_mu_uncertainty is None:
			self.fit(df_prices)

		# Use estimated uncertainty if available, otherwise use config
		eps_mu = (
			self.estimated_mu_uncertainty
			if self.estimated_mu_uncertainty is not None
			else self.config.mu_uncertainty_level
		)
		eps_cov = (
			self.estimated_cov_uncertainty
			if self.estimated_cov_uncertainty is not None
			else self.config.cov_uncertainty_level
		)

		# Handle edge cases
		if len(asset_names) == 0:
			logger.warning("Robust: No assets provided")
			return create_weights_series(np.array([]), [])

		if np.any(np.isnan(mu_array)) or np.any(np.isinf(mu_array)):
			logger.warning("Robust: NaN or inf in expected returns, returning zero weights")
			return create_weights_series(np.zeros(len(asset_names)), asset_names)

		if np.any(np.isnan(cov_array)) or np.any(np.isinf(cov_array)):
			logger.warning("Robust: NaN or inf in covariance matrix, returning zero weights")
			return create_weights_series(np.zeros(len(asset_names)), asset_names)

		n_assets = len(asset_names)

		# Create robust covariance matrix: Σ_robust = Σ + ε_Σ * I
		# This inflates the variance to account for estimation error
		cov_robust = cov_array + eps_cov * np.eye(n_assets)

		# Store parameters for objective function
		self._mu_array = mu_array
		self._cov_robust = cov_robust
		self._eps_mu = eps_mu

		# Run optimization with multi-start to avoid local minima
		weights = minimize_with_multistart(
			objective_function=self._objective_function,
			n_assets=n_assets,
			allow_cash=self.config.allow_cash,
			previous_best_weights=self._previous_best_weights,
		)

		# Store best weights for next optimization warm start
		self._previous_best_weights = weights.copy()

		# Clip very small weights to zero
		weights[weights < 1e-6] = 0.0

		# Only normalize if sum > 1.0 (constraint violation for safety)
		total_weight = np.sum(weights)
		if total_weight > 1.0:
			logger.warning(f"Robust: Total weight {total_weight:.6f} > 1.0, renormalizing")
			weights = weights / total_weight
		elif not self.config.allow_cash and abs(total_weight - 1.0) > 1e-6:
			logger.warning(f"Robust: Total weight {total_weight:.6f} != 1.0 (fully invested mode), renormalizing")
			weights = weights / total_weight

		logger.debug(
			f"Robust optimization: total_weight={np.sum(weights):.4f}, " f"eps_mu={eps_mu:.4f}, eps_cov={eps_cov:.4f}"
		)

		# Log if holding significant cash
		if self.config.allow_cash:
			cash_position = 1.0 - np.sum(weights)
			if cash_position > 0.1:  # More than 10% cash
				logger.debug(
					f"Robust optimizer holding {cash_position:.2%} cash due to uncertain conditions "
					f"(ε_μ={eps_mu:.4f}, ε_Σ={eps_cov:.4f})"
				)

		return create_weights_series(weights, asset_names)

	def _objective_function(self, w: np.ndarray) -> float:
		"""
		Objective function for robust optimization: -[w'μ - ε_μ||w|| - λw'Σ_robust w]
		We minimize the negative to maximize.

		Args:
		    w: Portfolio weights

		Returns:
		    Negative of robust utility (to minimize)
		"""
		portfolio_return = np.dot(w, self._mu_array)
		# Worst-case adjustment for mean uncertainty: subtract ε_μ * ||w||_2
		uncertainty_penalty = self._eps_mu * np.linalg.norm(w)
		portfolio_variance = w @ self._cov_robust @ w
		# Maximize return - uncertainty_penalty - risk_aversion * variance
		# Minimize negative of this
		return -(portfolio_return - uncertainty_penalty - self.config.risk_aversion * portfolio_variance)

	@property
	def name(self) -> str:
		"""Return optimizer name."""
		return "RobustMVO"
