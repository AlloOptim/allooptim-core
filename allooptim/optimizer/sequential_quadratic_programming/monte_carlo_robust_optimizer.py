"""Robust Portfolio Optimizer with Monte Carlo Sampling.

This module implements robust portfolio optimization using Monte Carlo sampling
to account for parameter uncertainty in covariance estimation. Uses:
    - Bootstrap sampling (standard or block-based)
    - Wishart distribution sampling
    - Log returns (better statistical properties)
    - Median aggregation (robust against outliers)
    - Randomized initial weights (avoids local minima)
    - Multiple objective functions (Variance, Sortino, CVaR, Max Drawdown, Max Diversification)

Key Features:
    - Handles uncertainty through Monte Carlo sampling
    - Block bootstrap preserves temporal structure (important for autocorrelated returns)
    - Wishart sampling guarantees positive semi-definite covariance matrices
    - Median aggregation provides robust estimates
    - Random initialization in each sample avoids getting stuck in local minima
    - Flexible objective functions for different risk preferences

Theoretical Background:
    Instead of assuming perfect knowledge of Σ and μ, we sample from distributions
    of plausible parameters and optimize for each sample with randomized starting points.
    The final weights are aggregated via median across all samples.

    This provides robustness against:
        - Estimation error in covariance matrix and returns
        - Non-stationarity in market conditions
        - Outliers and extreme events
        - Local minima in non-convex objectives
        - Small sample sizes

References:
    - Efron, B. (1979). "Bootstrap Methods"
    - Politis, D. & Romano, J. (1994). "The Stationary Bootstrap"
    - Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"
    - Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of CVaR"
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy.stats import wishart

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import LMoments
from allooptim.optimizer.asset_name_utils import (
    convert_pandas_to_numpy,
    create_weights_series,
    get_asset_names,
    validate_asset_names,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.sequential_quadratic_programming.sqp_multistart import (
    minimize_given_initial,
)

logger = logging.getLogger(__name__)

# Constants
MIN_VOLATILITY_THRESHOLD = 1e-10
MIN_DOWNSIDE_DEVIATION_THRESHOLD = 1e-10
MIN_OBSERVATIONS_REQUIRED = 10
WEIGHT_CLIPPING_THRESHOLD = 1e-6
WEIGHT_SUM_TOLERANCE = 1e-6
DEFAULT_BLOCK_SIZE = 5
REGULARIZATION_EPSILON = 1e-8
MIN_BLOCK_SIZE = 2
MAX_BLOCK_SIZE = 20
HALF_CHANCE_VALUE = 0.5


@dataclass
class SamplingResult:
    """Result of a clustering operation for NCO algorithm.

    Stores the outcome of attempting to cluster assets with K-means,
    including quality metrics and success status.
    """

    age: int
    weights: np.ndarray


class SamplingMethod(str, Enum):
    """Methods for sampling covariance matrices."""

    BOOTSTRAP = "bootstrap"
    BLOCK_BOOTSTRAP = "block_bootstrap"
    WISHART = "wishart"


class ObjectiveFunction(str, Enum):
    """Objective functions for portfolio optimization."""

    MIN_VARIANCE = "min_variance"
    MIN_CVAR = "min_cvar"
    MAX_SORTINO = "max_sortino"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_MAX_DRAWDOWN = "min_max_drawdown"


class MonteCarloRobustOptimizerConfig(BaseModel):
    """Configuration for Monte Carlo Robust Portfolio Optimizer."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    n_monte_carlo_samples: int = Field(
        default=50,
        ge=1,
        description="Number of Monte Carlo samples",
    )

    max_result_age : int = Field(
        default=10,
        ge=1,
        description="Maximum age of sampling results to keep",
    )

    random_prune_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of samples to randomly prune",
    )

    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.BLOCK_BOOTSTRAP,
        description="Method for sampling covariance matrices",
    )

    block_size: int = Field(
        default=DEFAULT_BLOCK_SIZE,
        ge=MIN_BLOCK_SIZE,
        le=MAX_BLOCK_SIZE,
        description="Block size for block bootstrap",
    )

    allow_cash_by_optimizer: bool = Field(
        default=True,
        description="Allow partial portfolio allocation (sum(weights) < 1)",
    )

    allow_cash_by_variance: bool = Field(
        default=True,
        description="Allow partial portfolio allocation (sum(weights) < 1)",
    )

    min_total_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum total weight to invest (0 = can hold all cash, 1 = must be fully invested)",
    )

    wishart_df_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Multiplier for Wishart degrees of freedom (df = multiplier * n_days)",
    )

    cvar_alpha: float = Field(
        default=0.05,
        ge=0.01,
        le=0.10,
        description="Confidence level for CVaR (e.g., 0.05 for 95% CVaR)",
    )

    target_return: Optional[float] = Field(
        default=None,
        description="Target return for Sortino ratio (if None, uses mean return)",
    )

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: int, info) -> int:
        """Ensure block size is reasonable."""
        if v < MIN_BLOCK_SIZE:
            logger.warning(f"Block size {v} too small, using {MIN_BLOCK_SIZE}")
            return MIN_BLOCK_SIZE
        if v > MAX_BLOCK_SIZE:
            logger.warning(f"Block size {v} too large, using {MAX_BLOCK_SIZE}")
            return MAX_BLOCK_SIZE
        return v
    
    @property
    def allow_cash(self) -> bool:
        """Determine if cash is allowed based on both settings."""
        return self.allow_cash_by_optimizer or self.allow_cash_by_variance


class MonteCarloMinVarianceOptimizer(AbstractOptimizer):
    """Monte Carlo Min Variance Portfolio Optimizer.

    This optimizer uses Monte Carlo sampling to handle uncertainty in covariance
    estimation. Instead of relying on a single point estimate, it samples many
    plausible covariance matrices and optimizes for each, then aggregates the results.

    Key Advantages:
        - Robust to estimation error in covariance matrix
        - No assumptions about uncertainty sets (unlike ellipsoidal robust optimization)
        - Flexible: supports multiple sampling and aggregation methods
        - Handles temporal structure through block bootstrap
        - Natural way to incorporate parameter uncertainty

    Compared to Other Approaches:
        - Standard MVO: Single point estimate, prone to estimation error
        - Ellipsoidal Robust: Assumes specific uncertainty structure
        - Monte Carlo Robust: Data-driven, flexible, no parametric assumptions

    Examples:
        >>> config = MonteCarloRobustOptimizerConfig(
        ...     n_monte_carlo_samples=2000,
        ...     sampling_method=SamplingMethod.BLOCK_BOOTSTRAP,
        ...     aggregation_method=AggregationMethod.MEDIAN,
        ...     returns_type=ReturnsType.LOG,
        ... )
        >>> optimizer = MonteCarloRobustOptimizer(config)
        >>> weights = optimizer.allocate(mu, cov, df_prices)

    Args:
        config: Configuration object with Monte Carlo parameters
    """

    objective_function: ObjectiveFunction = ObjectiveFunction.MIN_VARIANCE

    def __init__(self, config: Optional[MonteCarloRobustOptimizerConfig] = None) -> None:
        """Initialize the Monte Carlo robust optimizer.

        Args:
            config: Configuration parameters. If None, uses default config.
        """
        self.config = config or MonteCarloRobustOptimizerConfig()
        self._all_sample_results: Optional[list[SamplingResult]] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        df_allocations: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Calculate robust portfolio weights using Monte Carlo sampling.

        Samples multiple covariance matrices, optimizes the selected objective
        function for each with randomized initial weights, then aggregates
        using median (robust against outliers).

        Always uses log returns for better statistical properties.

        Args:
            ds_mu: Expected returns
            df_cov: Covariance matrix estimate (used as baseline)
            df_prices: Price history for sampling (REQUIRED for this optimizer)
            df_allocations: Previous allocations (not used)
            time: Timestamp (not used)
            l_moments: L-moments (not used)

        Returns:
            Portfolio weights as pandas Series
            - sum(weights) may be < 1.0 if allow_cash=True
            - Robust to estimation error through Monte Carlo aggregation

        Raises:
            ValueError: If df_prices is None or insufficient data
        """
        # Validate inputs
        validate_asset_names(ds_mu, df_cov)
        asset_names = get_asset_names(mu=ds_mu)

        if df_prices is None:
            raise ValueError("df_prices is required for Monte Carlo robust optimization")

        if len(df_prices) < MIN_OBSERVATIONS_REQUIRED:
            raise ValueError(
                f"Insufficient price history: {len(df_prices)} days provided, " f"{MIN_OBSERVATIONS_REQUIRED} required"
            )

        # Always use log returns (better statistical properties)
        returns = np.diff(np.log(df_prices.values), axis=0)

        # Convert mu to numpy
        mu_array, _, _ = convert_pandas_to_numpy(ds_mu, df_cov)
        _, n_assets = returns.shape

        self._prune_too_old_results()
        self._prune_random_samples()

        n_existing_samples = 0 if self._all_sample_results is None else len(self._all_sample_results)

        n_new_samples = self.config.n_monte_carlo_samples - n_existing_samples

        # Sample covariance matrices and optimize
        if self.config.sampling_method == SamplingMethod.BOOTSTRAP:
            w_samples = self._monte_carlo_bootstrap(returns, mu_array, n_assets, n_new_samples)
        elif self.config.sampling_method == SamplingMethod.BLOCK_BOOTSTRAP:
            w_samples = self._monte_carlo_block_bootstrap(returns, mu_array, n_assets, n_new_samples)
        else:  # WISHART
            w_samples = self._monte_carlo_wishart(returns, mu_array, n_assets, n_new_samples)

        # Store for analysis
        if self._all_sample_results is None:
            self._all_sample_results = [SamplingResult(age=0, weights=w) for w in w_samples]

        else:
            self._all_sample_results = [SamplingResult(age=res.age + 1, weights=res.weights) for res in self._all_sample_results]
            self._all_sample_results.extend([SamplingResult(age=0, weights=w) for w in w_samples])

        # Aggregate results
        w_matrix = np.array([res.weights for res in self._all_sample_results])

        w_matrix = self._scale_weights_by_variance(w_matrix)

        # Always aggregate with median (robust against outliers)
        w_final = np.median(w_matrix, axis=0)

        # Post-processing
        w_final = self._postprocess_weights(w_final)

        return create_weights_series(w_final, asset_names)

    def _prune_too_old_results(self) -> None:
        """Prune old sampling results to save memory."""
        if self._all_sample_results is None:
            return None

        self._all_sample_results = [res for res in self._all_sample_results if res.age < self.config.max_result_age]

    def _prune_random_samples(self) -> None:
        """Prune random samples to save memory."""
        if self._all_sample_results is None:
            return None

        current_size = len(self._all_sample_results)
        keep_size = int((1 - self.config.random_prune_fraction) * current_size)
        if keep_size == 0:
            return None

        keep_index = np.random.choice(
            current_size,
            size=keep_size,
            replace=False,
        )

        self._all_sample_results = [self._all_sample_results[i] for i in keep_index]

    def _monte_carlo_bootstrap(self, returns: np.ndarray, mu: np.ndarray, n_assets: int, n_new_samples: int,) -> list[np.ndarray]:
        """Standard bootstrap sampling (i.i.d. resampling with replacement).

        Args:
            returns: Return matrix (n_days, n_assets)
            mu: Expected returns
            n_assets: Number of assets
            n_new_samples: Number of bootstrap samples to generate

        Returns:
            Tuple of (sample_weights, sample_metrics)
        """
        n_days = len(returns)
        w_samples = []

        for _ in range(n_new_samples):
            # Resample with replacement
            idx = np.random.choice(n_days, size=n_days, replace=True)
            bootstrap_returns = returns[idx, :]

            # Estimate covariance
            sampled_cov = np.cov(bootstrap_returns.T)
            sampled_cov += np.eye(n_assets) * REGULARIZATION_EPSILON

            # Estimate mean from bootstrap sample
            sampled_mu = np.mean(bootstrap_returns, axis=0)

            # Random initial weights
            w0 = self._random_initial_weights(n_assets)

            # Optimize with selected objective
            w_opt = self._optimize_objective(sampled_cov, sampled_mu, bootstrap_returns, w0)

            w_samples.append(w_opt)

        return w_samples

    def _monte_carlo_block_bootstrap(self, returns: np.ndarray, mu: np.ndarray, n_assets: int, n_new_samples: int,) -> list[np.ndarray]:
        """Block bootstrap sampling (preserves temporal structure).

        Recommended method when returns exhibit autocorrelation or volatility clustering.

        Args:
            returns: Return matrix (n_days, n_assets)
            mu: Expected returns
            n_assets: Number of assets
            n_new_samples: Number of bootstrap samples to generate

        Returns:
            sample_weights
        """
        n_days = len(returns)
        block_size = self.config.block_size
        n_blocks = n_days // block_size

        w_samples = []

        for _ in range(n_new_samples):
            # Sample entire blocks to preserve temporal structure
            bootstrap_returns = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n_days - block_size + 1)
                block = returns[start_idx : start_idx + block_size, :]
                bootstrap_returns.append(block)

            bootstrap_returns = np.vstack(bootstrap_returns)

            # Estimate covariance
            sampled_cov = np.cov(bootstrap_returns.T)
            sampled_cov += np.eye(n_assets) * REGULARIZATION_EPSILON

            # Estimate mean from bootstrap sample
            sampled_mu = np.mean(bootstrap_returns, axis=0)

            # Random initial weights
            w0 = self._random_initial_weights(n_assets)

            # Optimize with selected objective
            w_opt = self._optimize_objective(sampled_cov, sampled_mu, bootstrap_returns, w0)

            w_samples.append(w_opt)

        return w_samples

    def _monte_carlo_wishart(self, returns: np.ndarray, mu: np.ndarray, n_assets: int, n_new_samples: int,) -> list[np.ndarray]:
        """Wishart distribution sampling (guarantees positive semi-definite matrices).

        The Wishart distribution is the natural distribution for covariance matrices,
        similar to how chi-squared is for variances.

        Args:
            returns: Return matrix (n_days, n_assets)
            mu: Expected returns
            n_assets: Number of assets
            n_new_samples: Number of samples to generate from Wishart distribution

        Returns:
            sample_weights
        """
        n_days = len(returns)
        emp_cov = np.cov(returns.T)
        emp_mu = np.mean(returns, axis=0)

        # Degrees of freedom should be >= n_assets for valid Wishart
        df = max(n_assets, int(n_days * self.config.wishart_df_multiplier))

        w_samples = []

        for _ in range(n_new_samples):
            # Sample from Wishart distribution
            # scale parameter is normalized by df to keep mean = emp_cov
            sampled_cov = wishart.rvs(df=df, scale=emp_cov / df)

            # Use empirical mean (Wishart only samples covariance)
            sampled_mu = emp_mu

            # Random initial weights
            w0 = self._random_initial_weights(n_assets)

            # Optimize with selected objective
            w_opt = self._optimize_objective(sampled_cov, sampled_mu, returns, w0)

            w_samples.append(w_opt)

        return w_samples

    def _random_initial_weights(self, n_assets: int) -> np.ndarray:
        """Generate random initial weights for optimization.

        Uses Dirichlet distribution to ensure they sum to <= 1 and are non-negative.

        Args:
            n_assets: Number of assets

        Returns:
            Random initial weights
        """
        # Dirichlet with alpha=1 gives uniform distribution on simplex
        w = np.random.dirichlet(np.ones(n_assets))

        # Randomly scale down to allow for cash position
        if self.config.allow_cash and np.random.rand() > HALF_CHANCE_VALUE:
            scale = np.random.uniform(self.config.min_total_weight, 1.0)
            w = w * scale

        return w

    def _optimize_objective(
        self,
        cov_matrix: np.ndarray,
        mu: np.ndarray,
        returns: np.ndarray,
        w0: np.ndarray,
    ) -> np.ndarray:
        """Optimize the selected objective function.

        Args:
            cov_matrix: Covariance matrix
            mu: Expected returns
            returns: Historical returns
            w0: Initial weights

        Returns:
            Optimal weights
        """
        allow_cash = self.config.allow_cash_by_optimizer

        # Select objective function
        if self.objective_function == ObjectiveFunction.MIN_VARIANCE:

            def objective(w):
                return w.T @ cov_matrix @ w

            allow_cash = False

        elif self.objective_function == ObjectiveFunction.MIN_CVAR:

            def objective(w):
                return self._cvar_objective(w, returns)

        elif self.objective_function == ObjectiveFunction.MAX_SORTINO:

            def objective(w):
                return -self._sortino_objective(w, returns, mu)

        elif self.objective_function == ObjectiveFunction.MAX_DIVERSIFICATION:

            def objective(w):
                return -self._diversification_objective(w, cov_matrix)

        elif self.objective_function == ObjectiveFunction.MIN_MAX_DRAWDOWN:

            def objective(w):
                return self._max_drawdown_objective(w, returns)

            allow_cash = False

        else:
            raise ValueError(f"Unsupported objective function: {self.objective_function}")

        # Optimize
        result = minimize_given_initial(
            objective_function=objective,
            allow_cash=allow_cash,
            x0=w0,
        )

        if not result.success:
            logger.debug(f"Optimization failed: {result.message}, using initial weights")
            return w0

        return result.x

    def _cvar_objective(self, w: np.ndarray, returns: np.ndarray) -> float:
        """Calculate CVaR (Conditional Value at Risk) for portfolio.

        CVaR_α = VaR_α + (1/α) * E[max(0, -R - VaR_α)]

        Args:
            w: Portfolio weights
            returns: Historical returns

        Returns:
            CVaR (to minimize)
        """
        portfolio_returns = returns @ w
        sorted_returns = np.sort(portfolio_returns)

        # Calculate VaR
        var_idx = int(len(sorted_returns) * self.config.cvar_alpha)
        var = -sorted_returns[var_idx]

        # Calculate CVaR (expected loss beyond VaR)
        losses_beyond_var = sorted_returns[:var_idx]
        cvar = var + np.mean(np.maximum(0, -losses_beyond_var - var)) / self.config.cvar_alpha

        return cvar

    def _sortino_objective(self, w: np.ndarray, returns: np.ndarray, mu: np.ndarray) -> float:
        """Calculate Sortino ratio for portfolio.

        Sortino = (E[R] - target) / downside_deviation

        Args:
            w: Portfolio weights
            returns: Historical returns
            mu: Expected returns

        Returns:
            Negative Sortino ratio (to minimize for maximization)
        """
        portfolio_returns = returns @ w
        expected_return = np.dot(w, mu)

        # Target return (MAR - Minimum Acceptable Return)
        target = self.config.target_return if self.config.target_return is not None else 0.0

        # Downside deviation (only negative deviations from target)
        downside_returns = portfolio_returns - target
        downside_deviation = np.sqrt(np.mean(np.minimum(downside_returns, 0) ** 2))

        if downside_deviation < MIN_DOWNSIDE_DEVIATION_THRESHOLD:
            return -1e10  # Very high Sortino if no downside

        sortino = (expected_return - target) / downside_deviation
        return sortino  # Return positive (we negate in objective)

    def _diversification_objective(self, w: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio for portfolio.

        DR = (weighted avg of volatilities) / (portfolio volatility)

        Args:
            w: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Negative diversification ratio (to minimize for maximization)
        """
        # Individual asset volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix))

        # Weighted average of individual volatilities
        weighted_vol_sum = np.dot(w, individual_vols)

        # Portfolio volatility
        portfolio_vol = np.sqrt(w.T @ cov_matrix @ w)

        if portfolio_vol < MIN_VOLATILITY_THRESHOLD:
            return -1e10  # Maximum diversification if no volatility

        diversification_ratio = weighted_vol_sum / portfolio_vol
        return diversification_ratio  # Return positive (we negate in objective)

    def _max_drawdown_objective(self, w: np.ndarray, returns: np.ndarray) -> float:
        """Calculate maximum drawdown for portfolio.

        Max Drawdown = max(peak - trough) / peak

        Args:
            w: Portfolio weights
            returns: Historical returns

        Returns:
            Maximum drawdown (to minimize)
        """
        portfolio_returns = returns @ w

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + portfolio_returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown at each point
        drawdown = (running_max - cumulative) / running_max

        # Maximum drawdown
        max_dd = np.max(drawdown)

        return max_dd

    def _scale_weights_by_variance(self, w_matrix: np.ndarray) -> np.ndarray:
        """Post-process weights: clip small values and normalize if needed.

        Args:
            w_matrix: Raw optimized weights

        Returns:
            Processed weights
        """
        if not self.config.allow_cash_by_variance:
            return w_matrix

        std_per_asset_squared = np.std(w_matrix) ** 2

        # each weight per asset must be between 0 and 1
        # if std_per_asset_squared == 1, asset weight should be 0
        # if std_per_asset_squared == 0, asset weight should be previous value
        # if std_per_asset_squared is between 0 and 1, asset weight should be scaled down accordingly
        w_matrix = w_matrix * (1 - std_per_asset_squared)

        return w_matrix

    def _postprocess_weights(self, weights: np.ndarray) -> np.ndarray:
        """Post-process weights: clip small values and normalize if needed.

        Args:
            weights: Raw optimized weights

        Returns:
            Processed weights
        """
        # Clip very small weights to zero
        weights[weights < WEIGHT_CLIPPING_THRESHOLD] = 0.0

        # Ensure constraints are satisfied
        total_weight = np.sum(weights)

        if total_weight > 1.0 + WEIGHT_SUM_TOLERANCE:
            logger.warning(f"Total weight {total_weight:.6f} > 1.0, normalizing")
            weights = weights / total_weight
        elif not self.config.allow_cash and abs(total_weight - 1.0) > WEIGHT_SUM_TOLERANCE:
            logger.warning(f"Total weight {total_weight:.6f} != 1.0 (fully invested), normalizing")
            weights = weights / total_weight
        elif total_weight < self.config.min_total_weight:
            logger.warning(f"Total weight {total_weight:.6f} < min {self.config.min_total_weight:.6f}, " f"scaling up")
            weights = weights * (self.config.min_total_weight / total_weight)

        return weights

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "MonteCarloMinVarianceOptimizer"


class MonteCarloMaxDrawdownOptimizer(MonteCarloMinVarianceOptimizer):
    """Monte Carlo Max Drawdown Portfolio Optimizer.

    Inherits from MonteCarloMinVarianceOptimizer but changes the objective function
    to minimize maximum drawdown.
    """

    objective_function: ObjectiveFunction = ObjectiveFunction.MIN_MAX_DRAWDOWN

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "MonteCarloMaxDrawdownOptimizer"


class MonteCarloMaxDiversificationOptimizer(MonteCarloMinVarianceOptimizer):
    """Monte Carlo Max Diversification Portfolio Optimizer.

    Inherits from MonteCarloMinVarianceOptimizer but changes the objective function
    to maximize diversification ratio.
    """

    objective_function: ObjectiveFunction = ObjectiveFunction.MAX_DIVERSIFICATION

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "MonteCarloMaxDiversificationOptimizer"


class MonteCarloMaxSortinoOptimizer(MonteCarloMinVarianceOptimizer):
    """Monte Carlo Max Sortino Portfolio Optimizer.

    Inherits from MonteCarloMinVarianceOptimizer but changes the objective function
    to maximize Sortino ratio.
    """

    objective_function: ObjectiveFunction = ObjectiveFunction.MAX_SORTINO

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "MonteCarloMaxSortinoOptimizer"


class MonteCarloMinCVAROptimizer(MonteCarloMinVarianceOptimizer):
    """Monte Carlo Min CVaR Portfolio Optimizer.

    Inherits from MonteCarloMinVarianceOptimizer but changes the objective function
    to minimize Conditional Value at Risk (CVaR).
    """

    objective_function: ObjectiveFunction = ObjectiveFunction.MIN_CVAR

    @property
    def name(self) -> str:
        """Return optimizer name."""
        return "MonteCarloMinCVAROptimizer"
