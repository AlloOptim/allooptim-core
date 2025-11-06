import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

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
from allooptim.optimizer.sequential_quadratic_programming.sqp_multistart import minimize_with_multistart

logger = logging.getLogger(__name__)


class RiskParityOptimizerConfig(BaseModel):
    """Configuration for risk parity optimizer.

    This config holds parameters for the risk parity optimizer. Currently minimal
    as risk parity typically uses equal risk contribution by default, but structured
    for future extensibility.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    # Risk parity uses equal risk contribution by default, but can be customized
    # target_risk will be None by default and set to equal weights in __init__
    pass


class RiskParityOptimizer(AbstractOptimizer):
    """Risk Parity Optimizer"""

    def __init__(self, config: Optional[RiskParityOptimizerConfig] = None) -> None:
        """Initialize the risk parity optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
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
        """Gets position weights according to the risk parity method
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

        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

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

    def _total_weight_constraint(self, x: np.array) -> float:
        # sum of weights less than 1.0
        return 1.0 - np.sum(x)

    def _solve_optimization(self, cov: np.array) -> np.ndarray:
        # Store covariance for objective function
        self._cov = cov

        n_assets = len(cov)

        # Run optimization with multi-start to avoid local minima
        weights = minimize_with_multistart(
            objective_function=self._risk_budget_objective,
            n_assets=n_assets,
            allow_cash=True,
            previous_best_weights=self._previous_weights,
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
