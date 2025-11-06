"""Optimizer convergence analysis and simulation utilities.

This module provides tools for analyzing optimizer convergence behavior and
stability across different data conditions. It simulates optimization processes
with varying observation windows and data quality to assess robustness.

Key features:
- Convergence analysis across multiple simulation runs
- Optimizer stability assessment under data uncertainty
- Performance comparison with varying data availability
- Statistical analysis of optimization outcomes
- Integration with MCOS simulation framework
"""

from typing import List

import numpy as np
import pandas as pd

from allooptim.allocation_to_allocators.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allooptim.optimizer.allocation_metric import (
    expected_return_classical,
    validate_no_nan,
)
from allooptim.optimizer.optimizer_interface import (
    AbstractOptimizer,
)


def _simulate_optimization(
    optimizer_list: List[AbstractOptimizer],
    obs_simulator: AbstractObservationSimulator,
    covariance_transformers: List[AbstractCovarianceTransformer],
    n_observations: int,
    n_optimizers: int,
    n_assets: int,
    use_optimal_observation: bool,
    allow_partial_investment: bool,
) -> np.ndarray:
    """Run single optimization simulation with allocation tracking."""
    try:
        # Generate observation
        if use_optimal_observation:
            mu_hat, cov_hat, prices_hat, time_hat, l_moments_hat = obs_simulator.get_ground_truth()

        else:
            mu_hat, cov_hat, prices_hat, time_hat, l_moments_hat = obs_simulator.get_sample()

        # mu_hat should be a pd.Series (don't convert to numpy array)
        validate_no_nan(mu_hat, "simulated mu")
        validate_no_nan(cov_hat, "simulated cov")

        # Apply covariance transformations
        for transformer in covariance_transformers:
            cov_hat = transformer.transform(cov_hat, n_observations)
            validate_no_nan(cov_hat, f"transformed cov by {transformer.__class__.__name__}")

        allocation = np.zeros((n_optimizers, n_assets))

        # Generate allocation
        for i, optimizer in enumerate(optimizer_list):
            alloc_result = optimizer.allocate(mu_hat, cov_hat, prices_hat, time_hat, l_moments_hat)
            # Ensure result is 1D array
            weights = np.array(alloc_result).flatten()

            # Normalize weights only if partial investment is allowed and sum > 0
            weight_sum = np.sum(weights)
            if allow_partial_investment and weight_sum > 0:
                weights = weights / weight_sum

            allocation[i, :] = weights
            validate_no_nan(allocation, f"{optimizer.name} allocation")

        # Convert mu_hat to numpy array for expected_return_classical
        mu_hat_array = np.array(mu_hat).flatten() if isinstance(mu_hat, pd.Series) else mu_hat
        cov_hat_array = cov_hat.values if isinstance(cov_hat, pd.DataFrame) else cov_hat
        expected_returns = expected_return_classical(allocation, mu_hat_array, cov_hat_array)

        assert expected_returns.shape == (
            n_optimizers,
        ), f"Expected shape ({n_optimizers},), got {expected_returns.shape}"

        validate_no_nan(expected_returns, "error estimate for expected returns.")

        expected_returns = expected_returns.squeeze()

        return expected_returns

    except Exception as e:
        optimizer_names = [opt.name for opt in optimizer_list]
        raise RuntimeError(f"Simulation failed for optimizers {optimizer_names}: {str(e)}") from e


def simulate_optimizers_with_convergence(
    optimizer_list: List[AbstractOptimizer],
    obs_simulator: AbstractObservationSimulator,
    covariance_transformers: List[AbstractCovarianceTransformer],
    n_observations: int,
    n_optimizers: int,
    n_assets: int,
    n_time_steps: int,
    config: A2AConfig,
) -> np.ndarray:
    """Estimate expected returns with simple convergence detection.

    Args:
        optimizer_list: List of optimizers to simulate
        obs_simulator: Observation simulator for generating market scenarios
        covariance_transformers: List of covariance transformers
        n_observations: Number of observations per simulation
        n_optimizers: Total number of optimizers
        n_assets: Number of assets
        n_time_steps: Number of time steps to simulate
        config: A2A configuration containing convergence and simulation settings

    Returns:
        Array of expected returns for all time steps (n_time_steps, n_optimizers)
    """
    expected_return_all_time_steps = np.zeros((n_time_steps, n_optimizers))

    if config.run_all_steps:
        for k in range(n_time_steps):
            expected_returns = _simulate_optimization(
                optimizer_list=optimizer_list,
                obs_simulator=obs_simulator,
                covariance_transformers=covariance_transformers,
                n_observations=n_observations,
                n_optimizers=len(optimizer_list),
                n_assets=n_assets,
                use_optimal_observation=False,
                allow_partial_investment=config.allow_partial_investment,
            )

            expected_return_all_time_steps[k, :] = expected_returns

        return expected_return_all_time_steps

    # Track error differences for each optimizer
    error_diffs = [[] for _ in range(n_optimizers)]  # Store |expected - theoretical| for each optimizer
    converged_optimizers = set()  # Indices of converged optimizers
    optimizer_distributions = {}  # {optimizer_idx: {'mean': float, 'std': float}}

    # Create mapping from optimizer to index
    remaining_optimizer_indices = list(range(n_optimizers))

    theoretical_returns = _simulate_optimization(
        optimizer_list=optimizer_list,
        obs_simulator=obs_simulator,
        covariance_transformers=covariance_transformers,
        n_observations=n_observations,
        n_optimizers=len(optimizer_list),
        n_assets=n_assets,
        use_optimal_observation=True,
        allow_partial_investment=config.allow_partial_investment,
    )

    steps_completed = 0
    for k in range(n_time_steps):
        # Get indices of non-converged optimizers
        active_indices = [i for i in remaining_optimizer_indices if i not in converged_optimizers]

        if not active_indices:
            # All optimizers converged, break before processing this step
            break

        # Create list of active optimizers
        active_optimizers = [optimizer_list[i] for i in active_indices]

        # 1. Simulate only non-converged optimizers
        expected_returns = _simulate_optimization(
            optimizer_list=active_optimizers,
            obs_simulator=obs_simulator,
            covariance_transformers=covariance_transformers,
            n_observations=n_observations,
            n_optimizers=len(active_optimizers),
            n_assets=n_assets,
            use_optimal_observation=False,
            allow_partial_investment=config.allow_partial_investment,
        )

        # 2. Ensure arrays are proper shape
        if expected_returns.ndim == 0:
            expected_returns = np.array([expected_returns])
        if theoretical_returns.ndim == 0:
            theoretical_returns = np.array([theoretical_returns])

        # 2. Calculate absolute differences and store
        current_diffs = np.abs(expected_returns - theoretical_returns)

        # Store errors for each active optimizer
        for i, optimizer_idx in enumerate(active_indices):
            error_diffs[optimizer_idx].append(current_diffs[i])

        # 5 & 6. Fill expected_return_all_time_steps for this time step FIRST
        for optimizer_idx in range(n_optimizers):
            if optimizer_idx in converged_optimizers:
                # Draw sample from existing distribution
                dist = optimizer_distributions[optimizer_idx]

                # Use last known expected return as baseline
                baseline = dist["last_expected_return"]

                # Add some controlled variation based on error distribution
                variation = np.random.normal(0, dist["std"])
                sampled_return = baseline + variation

                expected_return_all_time_steps[k, optimizer_idx] = sampled_return

            elif optimizer_idx in active_indices:
                # Use actual computed return
                active_idx = active_indices.index(optimizer_idx)
                expected_return_all_time_steps[k, optimizer_idx] = expected_returns[active_idx]
            else:
                raise ValueError(f"Optimizer index {optimizer_idx} not in converged or active sets")

        # 3 & 4. Check convergence for optimizers with enough data AFTER filling results
        newly_converged = []
        min_collected_points = max(1, int(config.min_points_fraction * n_time_steps))

        for i, optimizer_idx in enumerate(active_indices):
            if len(error_diffs[optimizer_idx]) >= min_collected_points:
                errors = np.array(error_diffs[optimizer_idx])
                mean_error = np.mean(errors)
                std_error = np.std(errors) + 1e-6

                current_error = current_diffs[i]
                z_score = abs(current_error - mean_error) / std_error

                if z_score <= config.convergence_threshold:
                    newly_converged.append(optimizer_idx)
                    converged_optimizers.add(optimizer_idx)

                    # Store distribution for sampling
                    optimizer_distributions[optimizer_idx] = {
                        "mean": mean_error,
                        "std": std_error,
                        "last_expected_return": expected_returns[i],
                    }

        # Mark this step as completed
        steps_completed += 1

    # 8. Reduce to actual number of time steps simulated
    expected_return_all_time_steps = expected_return_all_time_steps[:steps_completed, :]

    return expected_return_all_time_steps
