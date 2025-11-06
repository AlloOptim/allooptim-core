"""Adjusted return portfolio optimization using sequential quadratic programming.

This module implements portfolio optimization with return adjustments using
Sequential Quadratic Programming (SQP). It optimizes portfolios with custom
return objectives while maintaining risk constraints.

Key features:
- Return-adjusted portfolio optimization
- Sequential quadratic programming solver
- Custom objective functions
- Risk-constrained optimization
- Higher-order moment considerations
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import (
    LMoments,
    expected_return_classical,
    expected_return_moments,
    make_positive_definite,
)
from allooptim.optimizer.asset_name_utils import (
    convert_pandas_to_numpy,
    create_weights_series,
    validate_asset_names,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.sequential_quadratic_programming.estimate_robust_ema_moments import (
    calculate_robust_ema_moments,
)
from allooptim.optimizer.sequential_quadratic_programming.sqp_multistart import minimize_with_multistart

logger = logging.getLogger(__name__)


class MeanVarianceAdjustedReturnsOptimizerConfig(BaseModel):
    """Configuration for mean-variance adjusted returns optimizer.

    This config holds parameters for the adjusted returns optimizer including
    risk aversion, L-moments reduction settings, and EMA parameters.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    risk_aversion: float = 4.0
    reduce_l_moments_to_diagonal: bool = True
    ema_span: int = 90
    target_return: float = 0.0  # Target return for semivariance (downside threshold)


class MeanVarianceAdjustedReturnsOptimizer(AbstractOptimizer):
    """Adjusted Returns Optimizer."""

    enable_l_moments: bool = False
    enable_ema: bool = False
    enable_semi_variance: bool = False

    def __init__(self, config: Optional[MeanVarianceAdjustedReturnsOptimizerConfig] = None) -> None:
        """Initialize the mean-variance adjusted returns optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
        self.config = config or MeanVarianceAdjustedReturnsOptimizerConfig()

        self._mu: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._l_moments: Optional[LMoments] = None
        self._df_prices: Optional[pd.DataFrame] = None
        self._previous_best_weights: Optional[np.ndarray] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Gets position weights according to the adjusted returns method.

        :param cov: covariance matrix
        :param mu: vector of expected returns
        :return: Series of position weights with asset names as index.
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()
        n_assets = len(asset_names)

        # Store df_prices if needed for semivariance calculation
        if self.enable_semi_variance:
            assert df_prices is not None, "df_prices must be provided when enable_semi_variance is True"
            self._df_prices = df_prices

        if self.enable_ema:
            assert df_prices is not None, "df_prices must be provided when enable_ema is True"

            # Use df_prices directly to calculate EMA moments (independent of mu/cov parameters)
            # This avoids log/no-log return inconsistencies

            # Calculate simple returns from prices
            returns = df_prices.pct_change().dropna()

            # Use custom robust EMA calculation instead of PyPortfolioOpt's unstable implementation
            mu_ema, cov_ema = calculate_robust_ema_moments(returns, span=self.config.ema_span)

            self._mu = mu_ema
            # Ensure covariance matrix is positive definite
            self._cov = make_positive_definite(cov_ema)

            # Validate reasonable values
            if np.any(np.isnan(self._mu)) or np.any(np.isnan(self._cov)):
                logger.warning("EMA calculation produced NaN values, using simple historical moments")
                self._mu = returns.mean().values
                self._cov = make_positive_definite(returns.cov().values)

            elif np.any(np.abs(self._mu) > 1.0):  # Returns > 100% are suspicious
                logger.warning(
                    f"EMA produced extreme returns (max abs: {np.max(np.abs(self._mu)):.2e}), "
                    f"using simple historical moments"
                )
                self._mu = returns.mean().values
                self._cov = make_positive_definite(returns.cov().values)

        elif self.enable_semi_variance:
            # Use standard mean returns but downside covariance
            mu_array, _, _ = convert_pandas_to_numpy(ds_mu, df_cov)
            semi_cov = self._calculate_semivariance_matrix(df_prices, self.config.target_return)

            self._mu = mu_array
            self._cov = semi_cov

        else:
            mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)

            self._mu = mu_array
            self._cov = cov_array

        if self.config.reduce_l_moments_to_diagonal and l_moments is not None:
            l_moments = LMoments(
                lt_comoment_1=np.diag(np.diag(l_moments.lt_comoment_1)),
                lt_comoment_2=np.diag(np.diag(l_moments.lt_comoment_2)),
                lt_comoment_3=np.diag(np.diag(l_moments.lt_comoment_3)),
                lt_comoment_4=0 * np.diag(np.diag(l_moments.lt_comoment_4)),
            )

        self._l_moments = l_moments

        # Run optimization with multi-start to avoid local minima
        optimal_weights = minimize_with_multistart(
            objective_function=self._objective_function,
            n_assets=n_assets,
            allow_cash=True,
            previous_best_weights=self._previous_best_weights,
        )

        # Store best weights for next optimization warm start
        self._previous_best_weights = optimal_weights.copy()

        return create_weights_series(optimal_weights, asset_names)

    def _calculate_semivariance_matrix(self, df_prices: pd.DataFrame, target_return: float = 0.0) -> np.ndarray:
        """Calculate downside covariance matrix (semivariance) from price data.

        Only considers returns below the target return threshold, making it asymmetric
        and suitable for loss-averse investors who care more about downside risk.

        Args:
            df_prices: DataFrame of asset prices (n_observations, n_assets)
            target_return: Minimum acceptable return threshold (default 0.0)

        Returns:
            Downside covariance matrix (n_assets, n_assets)
        """
        # Calculate returns
        returns = df_prices.pct_change().dropna()

        # Calculate excess returns relative to target
        excess_returns = returns - target_return

        # Only keep negative excess returns (downside)
        # Replace positive values with 0
        downside_returns = excess_returns.where(excess_returns < 0, 0)

        # Calculate covariance of downside returns
        semi_cov = downside_returns.cov().values

        # Ensure positive definite (add small regularization if needed)
        epsilon = 1e-8
        semi_cov += epsilon * np.eye(len(semi_cov))

        return semi_cov

    def _total_weight_constraint(self, x):
        # sum of weights less than 1.0
        return 1.0 - np.sum(x)

    def _objective_function(self, x: np.ndarray) -> float:
        x = x[np.newaxis, :]

        if self.enable_l_moments:
            cost = -1 * expected_return_moments(
                weights=x,
                l_moments=self._l_moments,
                risk_aversion=self.config.risk_aversion,
                normalize_weights=False,
            )
        else:
            cost = -1 * expected_return_classical(
                weights=x,
                mu=self._mu,
                cov=self._cov,
                risk_aversion=self.config.risk_aversion,
                normalize_weights=False,
            )

        return cost.item() if hasattr(cost, "item") else float(cost)

    @property
    def name(self) -> str:
        """Get the name of the mean-variance adjusted returns optimizer.

        Returns:
            Optimizer name string
        """
        return "AdjustedReturnsMeanVariance"


class LMomentsAdjustedReturnsOptimizer(MeanVarianceAdjustedReturnsOptimizer):
    """Adjusted Returns Optimizer with L-Moments."""

    enable_l_moments: bool = True
    enable_ema: bool = False

    @property
    def name(self) -> str:
        """Get the name of the L-moments adjusted returns optimizer.

        Returns:
            Optimizer name string
        """
        return "AdjustedReturnsLMoments"


class EMAAdjustedReturnsOptimizer(MeanVarianceAdjustedReturnsOptimizer):
    """Adjusted Returns Optimizer with EMA."""

    enable_l_moments: bool = False
    enable_ema: bool = True

    @property
    def name(self) -> str:
        """Get the name of the EMA adjusted returns optimizer.

        Returns:
            Optimizer name string
        """
        return "AdjustedReturnsEMA"


class SemiVarianceAdjustedReturnsOptimizer(MeanVarianceAdjustedReturnsOptimizer):
    """Mean-Semivariance Optimizer.

    Optimizes return-risk tradeoff using only downside deviations (semivariance)
    instead of total variance. This asymmetric risk measure is more suitable for
    loss-averse investors who care more about downside risk than upside volatility.

    Objective: maximize E[R] - λ * Semivariance
    where Semivariance = E[min(R - target, 0)²]
    """

    enable_l_moments: bool = False
    enable_ema: bool = False
    enable_semi_variance: bool = True

    @property
    def name(self) -> str:
        """Get the name of the semi-variance adjusted returns optimizer.

        Returns:
            Optimizer name string
        """
        return "AdjustedReturnsSemiVariance"
