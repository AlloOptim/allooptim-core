import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pyswarms as ps
from pydantic import BaseModel

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import (
    LMoments,
)
from allo_optim.optimizer.asset_name_utils import create_weights_series, validate_asset_names
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.particle_swarm.early_stopping import EarlyStopObjective
from allo_optim.optimizer.particle_swarm.pso_objective import risk_adjusted_returns_objective

logger = logging.getLogger(__name__)

# Constants for numerical thresholds
WEIGHT_SUM_MINIMUM_THRESHOLD = 1e-10


class PSOOptimizerConfig(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    enable_warm_start: bool = True
    c1: float = 1.7  # Cognitive parameter
    c2: float = 1.7  # Social parameter
    w: float = 0.7  # Inertia weight
    n_particles: int = 2000
    n_iters: int = 500
    n_iters_warm: int = 100
    risk_aversion: float = 4.0
    ftol: float = 1e-5
    ftol_iter: int = 20


class MeanVarianceParticleSwarmOptimizer(AbstractOptimizer):
    """Optimizer based on the naive momentum"""

    enable_l_moments: bool = False

    def __init__(self, config: Optional[PSOOptimizerConfig] = None) -> None:
        self.config = config or PSOOptimizerConfig()
        self._previous_positions = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        # Ensure mu is 1D (handle both 1D and 2D cases from different simulators)
        mu_array = np.asarray(ds_mu.values).flatten()
        cov_array = np.asarray(df_cov.values)

        n_assets = len(mu_array)
        self.mu = mu_array
        self.cov = cov_array
        self.l_moments = l_moments

        if self.enable_l_moments and l_moments is None:
            logger.error("L-moments must be provided when enable_l_moments is True")
            weights_array = np.ones(n_assets) / n_assets
            return pd.Series(weights_array, index=asset_names)

        # Dimensions: [scale] + [weight1, weight2, ..., weightn]
        dimensions = n_assets + 1

        # Bounds: scale ∈ [0,1], weights ∈ [0,1]
        lower_bounds = np.zeros(dimensions)
        upper_bounds = np.ones(dimensions)

        if self._previous_positions is not None and self._previous_positions.shape != (
            self.config.n_particles,
            dimensions,
        ):
            logger.warning("Previous positions shape does not match current dimensions, resetting warm start.")
            self._previous_positions = None

        options = {"c1": self.config.c1, "c2": self.config.c2, "w": self.config.w}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.config.n_particles,
            dimensions=dimensions,
            options=options,
            bounds=(lower_bounds, upper_bounds),
            ftol=self.config.ftol,
            ftol_iter=self.config.ftol_iter,
            init_pos=self._previous_positions,
        )

        def objective_function(x):
            return risk_adjusted_returns_objective(
                x,
                enable_l_moments=self.enable_l_moments,
                l_moments=l_moments,
                risk_aversion=self.config.risk_aversion,
                mu=mu_array,
                cov=cov_array,
            )

        objective_with_early_stopping = EarlyStopObjective(
            objective_function=objective_function,
        )

        n_iters = self.config.n_iters if self._previous_positions is None else self.config.n_iters_warm

        _, optimal_solution = optimizer.optimize(
            objective_with_early_stopping,
            iters=n_iters,
            verbose=False,
        )

        if self.config.enable_warm_start:
            self._previous_positions = np.clip(optimizer.swarm.position, 0, 1)

        # Extract scale and raw weights from optimal solution
        optimal_scale = optimal_solution[0]
        optimal_raw_weights = optimal_solution[1:]

        # Convert to final portfolio weights
        if np.sum(optimal_raw_weights) > WEIGHT_SUM_MINIMUM_THRESHOLD:
            normalized_weights = optimal_raw_weights / np.sum(optimal_raw_weights)
            final_weights = optimal_scale * normalized_weights

        else:
            logger.error("PSO returned degenerate weights, using equal portfolio")
            final_weights = np.ones(n_assets) / n_assets

        logger.debug(f"PSO optimal scale: {optimal_scale:.4f}, total exposure: {np.sum(final_weights):.4f}")

        return create_weights_series(final_weights, asset_names)

    @property
    def name(self) -> str:
        return "PSO_MeanVariance"


class LMomentsParticleSwarmOptimizer(MeanVarianceParticleSwarmOptimizer):
    """Optimizer based on the naive momentum"""

    enable_l_moments: bool = True

    @property
    def name(self) -> str:
        return "PSO_LMoments"
