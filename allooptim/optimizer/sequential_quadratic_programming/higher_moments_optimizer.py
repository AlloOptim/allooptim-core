"""Higher moments portfolio optimization using sequential quadratic programming.

This module implements portfolio optimization strategies that incorporate higher-order moments
(such as skewness and kurtosis) beyond traditional mean-variance optimization. The optimizer
uses sequential quadratic programming with multi-start capabilities to find optimal portfolio
weights that account for non-normal return distributions and investor preferences for
asymmetric risk measures.

Key features:
- Skewness and kurtosis optimization
- Multi-start SQP for global optimization
- Higher moment risk-adjusted returns
- Portfolio rebalancing with moment constraints
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import (
    LMoments,
)
from allooptim.optimizer.asset_name_utils import (
    convert_pandas_to_numpy,
    create_weights_series,
    validate_asset_names,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.sequential_quadratic_programming.minimize_multistart import minimize_with_multistart

logger = logging.getLogger(__name__)

# Constants for numerical stability
DIVISION_BY_ZERO_TOLERANCE = 1e-10


class HigherMomentsOptimizerConfig(BaseModel):
    """Configuration for higher moments optimizer.

    This config holds parameters for the higher moments optimizer including
    weights for skewness and kurtosis terms, risk aversion, and L-moments usage.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    alpha_skew: float = Field(default=1.0, ge=0.0, description="Weight for L-skewness (positive skew is good)")
    beta_kurt: float = Field(
        default=1.0, ge=0.0, description="Weight for L-kurtosis (high kurtosis is bad, penalize fat tails)"
    )
    risk_aversion: float = Field(default=1.0, ge=0.0, description="Risk aversion for mean-variance component")
    use_l_moments: bool = Field(default=True, description="Use L-moments if available, else classical moments")

    maxiter: int = Field(default=100, ge=1, description="Maximum number of iterations for the optimizer")
    ftol: float = Field(default=1e-6, ge=0.0, description="Function tolerance for termination")
    optimizer_name: str = Field(default="SLSQP", description="Name of the optimizer to use")


class HigherMomentOptimizer(AbstractOptimizer):
    """Higher-Moment Portfolio Optimizer (Skewness-Kurtosis).

    Goes beyond mean-variance by incorporating third and fourth moments:
    - L-skewness (τ3): Measures asymmetry - positive values indicate upside potential
    - L-kurtosis (τ4): Measures tail risk - higher values indicate fat tails

    Objective: maximize E[R] - λ*Var + α*L-Skew - β*L-Kurt

    L-moments are more robust than classical moments:
    - Less sensitive to outliers
    - Exist even for distributions with infinite variance
    - Better for non-Gaussian distributions

    Suitable for investors seeking asymmetric upside while managing tail risk.
    """

    def __init__(
        self, config: Optional[HigherMomentsOptimizerConfig] = None, display_name: Optional[str] = None
    ) -> None:
        """Initialize the higher moments optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or HigherMomentsOptimizerConfig()

        # Store moment data for optimization
        self._mu: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._l_moments: Optional[LMoments] = None
        self._skew_vec: Optional[np.ndarray] = None
        self._kurt_vec: Optional[np.ndarray] = None
        self._previous_weights: Optional[np.ndarray] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights using higher moment optimization.

        Args:
            ds_mu: Expected returns for each asset
            df_cov: Covariance matrix
            df_prices: Historical prices (used if l_moments is None)
            df_allocations: Previous allocations (unused)
            time: Current time (unused)
            l_moments: L-moment co-moments (preferred if available)

        Returns:
            Series of optimal portfolio weights
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()
        n_assets = len(asset_names)

        # Convert to numpy for optimization
        mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)
        self._mu = mu_array
        self._cov = cov_array

        # Use L-moments if available and configured
        if self.config.use_l_moments and l_moments is not None:
            self._l_moments = l_moments
            self._skew_vec = None
            self._kurt_vec = None
        else:
            # Fall back to classical moments
            if df_prices is None:
                logger.warning("No df_prices provided for classical moments, using zero skew/kurtosis")
                self._skew_vec = np.zeros(n_assets)
                self._kurt_vec = np.zeros(n_assets)
            else:
                returns = df_prices.pct_change().dropna()
                self._skew_vec = returns.skew().values
                self._kurt_vec = returns.kurtosis().values
            self._l_moments = None

        # Run optimization with multi-start to avoid local minima
        optimal_weights = minimize_with_multistart(
            objective_function=self._objective,
            jacobian=self._objective_jacobian,
            n_assets=n_assets,
            allow_cash=True,
            previous_best_weights=self._previous_weights,
            maxiter=self.config.maxiter,
            ftol=self.config.ftol,
            optimizer_name=self.config.optimizer_name,
        )

        # Store best weights for next optimization warm start
        self._previous_weights = optimal_weights.copy()

        return create_weights_series(optimal_weights, asset_names)

    def _objective(self, x: np.ndarray) -> float:
        """Objective function: minimize -(E[R] - λ*Var + α*Skew - β*Kurt).

        We negate because scipy.optimize.minimize minimizes the objective.
        """
        x = np.array(x)

        # Mean-variance component
        portfolio_return = np.dot(x, self._mu)
        portfolio_variance = x @ self._cov @ x
        utility = portfolio_return - self.config.risk_aversion * portfolio_variance

        # Higher moment components
        if self._l_moments is not None:
            # Use L-moments: calculate portfolio L-skewness and L-kurtosis
            # Portfolio L-moments: λ_r = w' L_r w (where L_r is the r-th L-comoment matrix)

            # Calculate portfolio L-scale (λ2)
            lambda_2 = x @ self._l_moments.lt_comoment_2 @ x

            if np.abs(lambda_2) > DIVISION_BY_ZERO_TOLERANCE:  # Avoid division by zero
                # Calculate portfolio L-skewness (τ3 = λ3 / λ2)
                lambda_3 = x @ self._l_moments.lt_comoment_3 @ x
                l_skewness = lambda_3 / lambda_2

                # Calculate portfolio L-kurtosis (τ4 = λ4 / λ2)
                lambda_4 = x @ self._l_moments.lt_comoment_4 @ x
                l_kurtosis = lambda_4 / lambda_2
            else:
                l_skewness = 0.0
                l_kurtosis = 0.0

            # Add L-moment components to utility
            # Positive L-skewness is good (upside asymmetry)
            # High L-kurtosis is bad (fat tails)
            utility += self.config.alpha_skew * l_skewness
            utility -= self.config.beta_kurt * np.abs(l_kurtosis)  # Penalize extreme kurtosis

        else:
            # Use classical moments: simple weighted average (approximation)
            # More sophisticated: would need co-skewness and co-kurtosis tensors
            portfolio_skew = np.dot(x, self._skew_vec)
            portfolio_kurt = np.dot(x, self._kurt_vec)

            utility += self.config.alpha_skew * portfolio_skew
            utility -= self.config.beta_kurt * portfolio_kurt  # Excess kurtosis (>3 is bad)

        # Negate for minimization
        return -utility

    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytical gradient for mean-variance component and classical moments.

        For L-moments case, returns partial gradient (MV component only).
        For classical moments, includes skew and kurtosis derivatives.

        Args:
            x: Portfolio weights

        Returns:
            Gradient of objective function
        """
        x = np.array(x)

        # Mean-variance gradient (always computable)
        # d/dx[mu'x - lambda*x'Cx] = mu - 2*lambda*Cx
        grad_mv = self._mu - 2 * self.config.risk_aversion * self._cov @ x

        if self._l_moments is not None:
            # L-moments case: only MV component implemented
            # TODO: Implement quotient rule for L-skewness (lambda_3/lambda_2)
            # TODO: Implement quotient rule for L-kurtosis (lambda_4/lambda_2)
            logger.debug("L-moments gradient not implemented, returning MV component only")
            full_grad = grad_mv
        else:
            # Classical moments case: include skew and kurtosis derivatives
            # d/dx[alpha_skew * x'skew_vec] = alpha_skew * skew_vec
            # d/dx[-beta_kurt * x'kurt_vec] = -beta_kurt * kurt_vec
            grad_skew = self.config.alpha_skew * self._skew_vec
            grad_kurt = -self.config.beta_kurt * self._kurt_vec
            full_grad = grad_mv + grad_skew + grad_kurt

        # Negate for minimization (objective = -utility)
        return -full_grad

    @property
    def name(self) -> str:
        """Get the name of the higher moments optimizer.

        Returns:
            Optimizer name string
        """
        return "HigherMomentOptimizer"
