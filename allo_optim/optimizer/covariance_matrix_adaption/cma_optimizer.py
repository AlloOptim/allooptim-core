import logging
from datetime import datetime
from enum import Enum
from typing import Optional

import cma
import numpy as np
import pandas as pd
from pydantic import BaseModel

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import (
    LMoments,
    expected_return_classical,
    expected_return_moments,
)
from allo_optim.optimizer.asset_name_utils import create_weights_series, validate_asset_names
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.particle_swarm.pso_objective import (
    conditional_value_at_risk_objective,
    maximum_drawdown_objective,
    robust_sharpe_objective,
    sortino_ratio_objective,
)

logger = logging.getLogger(__name__)


# Constants for numerical stability and thresholds
EPSILON_WEIGHTS_SUM = 1e-10
PENALTY_INVALID_FITNESS = 1e10
MIN_OBSERVATIONS_RISK_METRICS = 30


class RiskMetric(Enum):
    MEAN_VARIANCE = "mean_variance"
    SORTINO = "sortino"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    ROBUST_SHARPE = "robust_sharpe"
    L_MOMENTS = "l_moments"


DISTRIBUTION_FREE_METRICS = {
    RiskMetric.SORTINO,
    RiskMetric.CVAR,
    RiskMetric.MAX_DRAWDOWN,
    RiskMetric.ROBUST_SHARPE,
    RiskMetric.L_MOMENTS,
}


class CMAState(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    sigma: float
    mean: np.ndarray
    C: Optional[np.ndarray] = None


class CMAOptimizerConfig(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    enable_simple_warm_start: bool = True
    enable_full_warm_start: bool = False
    budget: int = 4000
    budget_warm: int = 1000
    n_popsize: int = 1000
    risk_aversion: float = 4.0
    sigma: float = 0.2
    patience: int = 100
    min_improvement_threshold: float = 1e-6
    alpha_cvar: float = 0.05
    target_return_sortino: float = 0.0
    risk_penalty_max_drawdown: float = 1.0


class MeanVarianceCMAOptimizer(AbstractOptimizer):
    """CMA-ES optimizer with explicit scaling and interpolation"""

    risk_metric: RiskMetric = RiskMetric.MEAN_VARIANCE

    def __init__(
        self,
        config: Optional[CMAOptimizerConfig] = None,
    ) -> None:
        self.config = config or CMAOptimizerConfig()

        self._previous_solution: Optional[np.ndarray] = None
        self._previous_cma_state: Optional[CMAState] = None
        self._n_assets: Optional[int] = None
        self._mu: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._l_moments: Optional[LMoments] = None
        self._df_prices: Optional[pd.DataFrame] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        mu_array = np.asarray(ds_mu.values).flatten()
        cov_array = np.asarray(df_cov.values)

        assert df_prices is not None, "Price data must be fitted before allocation."

        assert df_prices.shape[1] == len(mu_array), "Fitted price data asset count does not match mu/cov asset count"

        assert len(mu_array) == cov_array.shape[0] == cov_array.shape[1], "Inconsistent asset counts between mu and cov"

        if self._n_assets is not None and self._n_assets != len(mu_array):
            logger.warning(
                f"Asset count changed from {self._n_assets} to {len(mu_array)}; "
                f"resetting previous solution and CMA state"
            )
            self._previous_solution = None
            self._previous_cma_state = None

        self._n_assets = len(mu_array)
        self._mu = mu_array
        self._cov = cov_array
        self._df_prices = df_prices
        self._l_moments = l_moments

        if self.risk_metric in DISTRIBUTION_FREE_METRICS and df_prices is not None:
            missing_assets = set(asset_names) - set(df_prices.columns)
            if missing_assets:
                raise ValueError(f"Fitted price data missing assets: {missing_assets}")

        if self.risk_metric == RiskMetric.L_MOMENTS and l_moments is None:
            raise ValueError("L-moments must be provided when using L_MOMENTS risk metric")

        optimal_solution = self._optimize_with_ask_tell()

        if self.config.enable_simple_warm_start:
            self._previous_solution = np.clip(optimal_solution, 0, 1)

        optimal_scale = optimal_solution[0]
        optimal_raw_weights = optimal_solution[1:]

        if np.sum(optimal_raw_weights) > EPSILON_WEIGHTS_SUM:
            normalized_weights = optimal_raw_weights / np.sum(optimal_raw_weights)
            final_weights = optimal_scale * normalized_weights
        else:
            logger.error("Sum of optimal raw weights is zero or too small; defaulting to equal weights")
            final_weights = np.ones(self._n_assets) / self._n_assets

        logger.debug(
            f"CMA-ES optimal scale: {optimal_scale:.4f}, total exposure: {np.sum(final_weights):.4f} "
            f"using {self.risk_metric} metric"
        )

        return create_weights_series(final_weights, asset_names)

    def _optimize_with_ask_tell(self) -> np.ndarray:
        dimensions = self._n_assets + 1

        lower_bounds = [0.0] * dimensions
        upper_bounds = [1.0] * dimensions

        options = {
            "bounds": [lower_bounds, upper_bounds],
            "seed": 42,
            "verbose": -9,
            "popsize": self.config.n_popsize,
        }

        if self._previous_solution is not None and self.config.enable_simple_warm_start:
            initial_point = self._previous_solution
            budget = self.config.budget_warm
        else:
            initial_point = np.ones(dimensions) / self._n_assets
            initial_point[0] = 0.8
            budget = self.config.budget

        cma_optimizer = cma.CMAEvolutionStrategy(initial_point, self.config.sigma, options)

        if self._previous_cma_state is not None and self.config.enable_full_warm_start:
            if len(initial_point) == len(self._previous_cma_state.mean):
                cma_optimizer.sigma = self._previous_cma_state.sigma
                cma_optimizer.mean = self._previous_cma_state.mean.copy()
                if self._previous_cma_state.C is not None:
                    cma_optimizer.C = self._previous_cma_state.C.copy()

        eval_count = 0
        best_solution = None
        best_fitness = float("inf")
        iterations_without_improvement = 0
        last_improvement_fitness = float("inf")

        while not cma_optimizer.stop() and eval_count < budget:
            candidates = cma_optimizer.ask()
            n_candidates = len(candidates)

            X = np.array(candidates)
            fitness_values = self._objective_matrix(X)

            eval_count += n_candidates

            # Check for invalid fitness values
            if not np.all(np.isfinite(fitness_values)):
                logger.warning(f"Invalid fitness values detected: {fitness_values}")
                # Replace invalid values with a large penalty
                fitness_values = np.where(np.isfinite(fitness_values), fitness_values, PENALTY_INVALID_FITNESS)
                # If all values are invalid, skip this iteration
                if not np.any(np.isfinite(fitness_values)):
                    logger.error("All fitness values are invalid, skipping iteration")
                    iterations_without_improvement += 1
                    continue

            min_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[min_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = candidates[min_idx].copy()

                improvement = last_improvement_fitness - current_best_fitness
                if improvement >= self.config.min_improvement_threshold:
                    iterations_without_improvement = 0
                    last_improvement_fitness = current_best_fitness
                    logger.debug(f"Significant improvement: {improvement:.8f}, resetting patience counter")
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            cma_optimizer.tell(candidates, fitness_values)

            logger.debug(
                f"Evaluated {eval_count}/{budget}, best fitness: {best_fitness:.6f}, "
                f"no improvement: {iterations_without_improvement}/{self.config.patience}"
            )

            # Early stopping check
            if iterations_without_improvement >= self.config.patience:
                logger.info(
                    f"Early stopping triggered after {iterations_without_improvement} iterations "
                    f"without improvement >= {self.config.min_improvement_threshold}"
                )
                break

        if self.config.enable_full_warm_start:
            try:
                self._previous_cma_state = CMAState(
                    sigma=cma_optimizer.sigma,
                    mean=cma_optimizer.mean.copy() if hasattr(cma_optimizer, "mean") else initial_point.copy(),
                    C=cma_optimizer.C.copy() if hasattr(cma_optimizer, "C") else None,
                )
            except Exception:
                logger.warning("Failed to store CMA state for warm start")
                self._previous_cma_state = None

        if hasattr(cma_optimizer.result, "xbest") and cma_optimizer.result.fbest < best_fitness:
            final_solution = cma_optimizer.result.xbest

        elif best_solution is not None:
            final_solution = best_solution

        else:
            logger.error("No valid solution found; returning initial point")
            final_solution = initial_point

        return final_solution

    def _objective_matrix(self, X: np.ndarray) -> np.ndarray:
        # Check for sufficient price data for distribution-free metrics
        if self.risk_metric in DISTRIBUTION_FREE_METRICS:
            if self._df_prices is None:
                logger.warning(f"No price data available for {self.risk_metric} metric")

            if self._df_prices.shape[0] < MIN_OBSERVATIONS_RISK_METRICS:
                # Minimum 30 observations for meaningful risk metrics
                logger.warning(
                    f"Insufficient price data ({self._df_prices.shape[0]} observations) for {self.risk_metric} metric"
                )

        scales = X[:, 0:1]
        raw_weights = X[:, 1:]

        scales = self._adjust_scaling_matrix(scales)

        raw_weight_sums = np.sum(raw_weights, axis=1, keepdims=True)
        raw_weight_sums = np.maximum(raw_weight_sums, EPSILON_WEIGHTS_SUM)
        normalized_weights = raw_weights / raw_weight_sums
        final_weights = scales * normalized_weights

        if self.risk_metric == RiskMetric.MEAN_VARIANCE:
            return -1 * expected_return_classical(
                weights=final_weights,
                mu=self._mu,
                cov=self._cov,
                risk_aversion=self.config.risk_aversion,
                normalize_weights=False,
            )

        if self.risk_metric == RiskMetric.L_MOMENTS:
            return -1 * expected_return_moments(
                weights=final_weights,
                l_moments=self._l_moments,
                risk_aversion=self.config.risk_aversion,
                normalize_weights=False,
            )

        if self.risk_metric == RiskMetric.SORTINO:
            return sortino_ratio_objective(
                final_weights,
                self._df_prices.values,
                target_return=self.config.target_return_sortino,
            )

        if self.risk_metric == RiskMetric.CVAR:
            return conditional_value_at_risk_objective(
                final_weights,
                self._df_prices.values,
                alpha=self.config.alpha_cvar,
            )

        if self.risk_metric == RiskMetric.MAX_DRAWDOWN:
            return maximum_drawdown_objective(
                final_weights,
                self._df_prices.values,
                return_penalty=self.config.risk_penalty_max_drawdown,
            )

        if self.risk_metric == RiskMetric.ROBUST_SHARPE:
            return robust_sharpe_objective(final_weights, self._df_prices.values)

        raise ValueError(f"Unknown risk metric: {self.risk_metric}")

    @staticmethod
    def _adjust_scaling_matrix(scales: np.ndarray) -> np.ndarray:
        scales = np.clip(scales, 0.0, 1.0)
        x_points = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
        y_points = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        adjusted_scales = np.interp(scales.flatten(), x_points, y_points)
        return adjusted_scales.reshape(scales.shape)

    @property
    def name(self) -> str:
        return f"CMA_{self.risk_metric.name}"


class CVARCMAOptimizer(MeanVarianceCMAOptimizer):
    """CMA-ES optimizer with explicit scaling and interpolation"""

    risk_metric: RiskMetric = RiskMetric.CVAR


class LMomentsCMAOptimizer(MeanVarianceCMAOptimizer):
    """CMA-ES optimizer with explicit scaling and interpolation"""

    risk_metric: RiskMetric = RiskMetric.L_MOMENTS


class RobustSharpeCMAOptimizer(MeanVarianceCMAOptimizer):
    """CMA-ES optimizer with explicit scaling and interpolation"""

    risk_metric: RiskMetric = RiskMetric.ROBUST_SHARPE


class SortinoCMAOptimizer(MeanVarianceCMAOptimizer):
    """CMA-ES optimizer with explicit scaling and interpolation"""

    risk_metric: RiskMetric = RiskMetric.SORTINO


class MaxDrawdownCMAOptimizer(MeanVarianceCMAOptimizer):
    """CMA-ES optimizer with explicit scaling and interpolation"""

    risk_metric: RiskMetric = RiskMetric.MAX_DRAWDOWN
