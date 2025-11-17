"""Optimizer Simulation Module.

Enhanced MCOS simulation that always returns allocation statistics.
Clean implementation with explicit error handling and no legacy fallbacks.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from allooptim.config.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.observation_simulator import (
    MuCovPartialObservationSimulator,
)
from allooptim.allocation_to_allocators.optimizer_convergence import (
    simulate_optimizers_with_convergence,
)
from allooptim.covariance_transformer.covariance_transformer import (
    DeNoiserCovarianceTransformer,
)
from allooptim.optimizer.allocation_metric import validate_no_nan
from allooptim.optimizer.optimizer_interface import AbstractOptimizer


@dataclass
class MCOSAllocationResult:
    """Complete MCOS result with error estimates and allocation statistics."""

    expected_return_means: np.ndarray  # (n_optimizers, 1)
    expected_returns_covariance: np.ndarray  # (n_optimizers, n_optimizers)


def simulate_optimizers_with_allocation_statistics(
    df_assets: pd.DataFrame,
    optimizer_list: List[AbstractOptimizer],
    config: A2AConfig,
) -> MCOSAllocationResult:
    """Simulate optimizers with enhanced allocation statistics tracking.

    Args:
        df_assets: Asset price DataFrame
        optimizer_list: List of optimizers to simulate
        config: A2A configuration containing simulation settings

    Returns:
        MCOSAllocationResult with error estimates and allocation statistics

    Raises:
        ValueError: If data contains NaN or optimizers fail validation
    """
    # Input validation
    if df_assets.isna().any().any():
        raise ValueError("df_assets contains NaN values")

    if len(optimizer_list) == 0:
        raise ValueError("optimizer_list cannot be empty")

    if len(optimizer_list) == 0:
        raise ValueError("optimizer_list cannot be empty")

    # Initialize components
    covariance_transformer = DeNoiserCovarianceTransformer(n_observations=config.n_data_observations)
    obs_simulator = MuCovPartialObservationSimulator(df_assets, config.n_data_observations)

    # Run simulations
    estimated_returns = simulate_optimizers_with_convergence(
        optimizer_list=optimizer_list,
        obs_simulator=obs_simulator,
        covariance_transformers=[covariance_transformer],
        n_observations=config.n_data_observations,
        n_optimizers=len(optimizer_list),
        n_assets=df_assets.shape[1],
        n_time_steps=config.n_data_observations,
        config=config,
    )

    estimates_mean = np.mean(estimated_returns, axis=0)
    estimated_cov = np.cov(estimated_returns, rowvar=False)

    validate_no_nan(estimates_mean, "mean error estimates")
    validate_no_nan(estimated_cov, "covariance of error estimates")

    # Ensure covariance matrix is 2D for consistency
    if estimated_cov.ndim == 0:
        estimated_cov = np.array([[estimated_cov.item()]])

    # Ensure mean is 2D column vector for consistency with optimize_allocator_weights
    if estimates_mean.ndim == 1:
        estimates_mean = estimates_mean.reshape(-1, 1)

    return MCOSAllocationResult(expected_return_means=estimates_mean, expected_returns_covariance=estimated_cov)
