"""Risk parity portfolio optimization using sequential quadratic programming.

This module implements risk parity optimization strategies that allocate portfolio weights
such that each asset contributes equally to the overall portfolio risk. The optimizer uses
sequential quadratic programming with multi-start capabilities to achieve risk parity across
assets, ensuring balanced risk contribution regardless of asset volatility.

Key features:
- Equal risk contribution across assets
- Sequential quadratic programming optimization
- Multi-start global optimization
- Risk-based portfolio construction
- Volatility normalization for fair comparison
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from allooptim.config.cash_config import normalize_weights_optimizers
from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import (
    LMoments,
)
from allooptim.optimizer.asset_name_utils import (
    convert_pandas_to_numpy,
    create_weights_series,
    validate_asset_names,
)
from allooptim.optimizer.base_optimizer import BaseOptimizer
from allooptim.optimizer.sequential_quadratic_programming.minimize_multistart import minimize_with_multistart

logger = logging.getLogger(__name__)


class RiskParityOptimizerConfig(BaseModel):
    """Configuration for risk parity optimizer.

    This config holds parameters for the risk parity optimizer. Currently minimal
    as risk parity typically uses equal risk contribution by default, but structured
    for future extensibility.
    """

    maxiter: int = 100
    ftol: float = 1e-6
    optimizer_name: str = "SLSQP"

    model_config = DEFAULT_PYDANTIC_CONFIG


class RiskParityOptimizer(BaseOptimizer):
    """Risk Parity Optimizer."""

    def __init__(self, config: Optional[RiskParityOptimizerConfig] = None, display_name: Optional[str] = None) -> None:
        """Initialize the risk parity optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or RiskParityOptimizerConfig()
        self._previous_weights: Optional[np.ndarray] = None
        self._target_risk: Optional[np.ndarray] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Gets position weights according to the risk parity method.

        :param cov: covariance matrix
        :param mu: vector of expected returns
        :return: list of position weights.
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()
        n_assets = len(asset_names)

        # Convert to numpy for optimization
        _, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)

        if self._previous_weights is None:
            self._previous_weights = np.ones(n_assets) / n_assets
        if self._target_risk is None:
            self._target_risk = np.ones(n_assets) / n_assets

        weights = self._solve_optimization(cov_array)

        # Apply normalization constraints based on allow_cash and max_leverage
        weights = normalize_weights_optimizers(weights, self.allow_cash, self.max_leverage)

        return create_weights_series(weights, asset_names)

    # risk budgeting optimization
    def _calculate_portfolio_var(self, w: np.array, cov: np.array) -> float:
        # function that calculates portfolio risk
        w = np.array(w, ndmin=2)
        return (w @ cov @ w.T)[0, 0]

    def _calculate_risk_contribution(self, weights: np.array, cov: np.array) -> np.ndarray:
        # function that calculates asset contribution to total risk
        weights = np.array(weights, ndmin=2)

        sigma = np.sqrt(self._calculate_portfolio_var(weights, cov))

        marginal_risk_contribution = cov @ weights.T

        risk_contribution = np.multiply(marginal_risk_contribution, weights.T) / sigma
        return risk_contribution

    def _risk_budget_objective(self, x: np.ndarray) -> float:
        # calculate portfolio risk
        sigma_portfolio = np.sqrt(self._calculate_portfolio_var(x, self._cov))
        risk_target = np.array(np.multiply(sigma_portfolio, self._target_risk), ndmin=2)
        asset_risk_contribution = self._calculate_risk_contribution(x, self._cov)
        cost = sum(np.square(asset_risk_contribution - risk_target.T))[0]  # sum of squared error
        return cost

    def _risk_budget_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytical gradient of risk budget objective."""
        n = len(x)
        sigma_p = np.sqrt(self._calculate_portfolio_var(x, self._cov))  # Portfolio volatility

        # Marginal risk contributions
        mrc = self._cov @ x  # Shape: (n,)

        # Risk contributions: RC_i = w_i * MRC_i / sigma_p
        rc = (x * mrc) / sigma_p

        # Target risk contributions
        target_rc = sigma_p * self._target_risk

        # Residuals: e_i = RC_i - target_RC_i
        residuals = rc - target_rc

        # Compute gradient using chain rule
        # d(sigma_p)/dw = mrc / sigma_p
        d_sigma_p = mrc / sigma_p

        # d(RC_i)/dw_j - complex due to ratio and product
        # For each asset i, gradient w.r.t. all weights w_j:
        grad = np.zeros(n)
        for j in range(n):
            # Derivative of RC_i w.r.t. w_j
            d_rc_j = np.zeros(n)
            for i in range(n):
                if i == j:
                    # Self-derivative: product rule + quotient rule
                    d_rc_j[i] = (mrc[i] + x[i] * self._cov[i, j]) / sigma_p - (
                        x[i] * mrc[i] * d_sigma_p[j]
                    ) / sigma_p**2
                else:
                    # Cross-derivative: only quotient rule
                    d_rc_j[i] = (x[i] * self._cov[i, j]) / sigma_p - (x[i] * mrc[i] * d_sigma_p[j]) / sigma_p**2

            # d(target_RC_i)/dw_j = target_risk_i * d_sigma_p[j]
            d_target_rc_j = self._target_risk * d_sigma_p[j]

            # Sum over all residuals: 2 * sum_i (e_i * de_i/dw_j)
            grad[j] = 2 * np.sum(residuals * (d_rc_j - d_target_rc_j))

        return grad

    def _solve_optimization(self, cov: np.array) -> np.ndarray:
        # Store covariance for objective function
        self._cov = cov

        n_assets = len(cov)

        # Run optimization with multi-start to avoid local minima
        weights = minimize_with_multistart(
            objective_function=self._risk_budget_objective,
            n_assets=n_assets,
            allow_cash=self.allow_cash,
            previous_best_weights=self._previous_weights,
            jacobian=self._risk_budget_jacobian,
            maxiter=self.config.maxiter,
            ftol=self.config.ftol,
            optimizer_name=self.config.optimizer_name,
        )

        # Store best weights for next optimization warm start
        self._previous_weights = weights.copy()

        return self._previous_weights

    @property
    def name(self) -> str:
        """Get the name of the risk parity optimizer.

        Returns:
            Optimizer name string
        """
        return "RiskParityOptimizer"
