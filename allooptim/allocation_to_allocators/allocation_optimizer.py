"""Allocation Optimizer Module

Enhanced allocation optimization using direct inter-optimizer covariance matrices.
Clean implementation with analytical solutions and explicit error handling.
"""

from functools import partial

import numpy as np
import pyswarms as ps

from allooptim.allocation_to_allocators.optimizer_simulator import (
    MCOSAllocationResult,
)
from allooptim.optimizer.allocation_metric import (
    expected_return_classical,
    validate_no_nan,
)


def optimize_allocator_weights(
    mcos_result: MCOSAllocationResult,
    n_particle_swarm_iterations: int = 1_000,
    n_particles: int = 200,
) -> np.ndarray:
    """Optimize allocator weights using expected returns mean and covariance.

    Args:
        mcos_result: MCOS result with expected returns mean and covariance
        n_particle_swarm_iterations: Number of PSO optimization iterations
        n_particles: Number of PSO particles

    Returns:
        Optimal weights for allocators

    Raises:
        ValueError: If inputs contain NaN or optimization fails
    """
    # Input validation - use the simplified structure
    if mcos_result.expected_returns_covariance is None:
        raise ValueError("MCOS result must contain expected returns covariance matrix")

    if mcos_result.expected_return_means is None:
        raise ValueError("MCOS result must contain expected return means")

    # Use the direct mean and covariance from simulations
    allocation_cov = mcos_result.expected_returns_covariance
    error_means = mcos_result.expected_return_means.flatten()  # Convert to 1D for compatibility

    validate_no_nan(allocation_cov, "inter-optimizer covariance")
    validate_no_nan(error_means, "error means")

    n_allocators = len(error_means)

    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_allocators,
        bounds=(np.zeros(n_allocators), np.ones(n_allocators)),
        options=options,
    )

    # Optimization objective function
    opt_fun = partial(
        expected_return_classical,
        mu=error_means,
        cov=allocation_cov,
    )

    # Run optimization
    try:
        _, optimal_weights = optimizer.optimize(
            opt_fun,
            iters=n_particle_swarm_iterations,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"PSO optimization failed: {str(e)}") from e

    # Validate and normalize result
    optimal_weights = np.array(optimal_weights)
    validate_no_nan(optimal_weights, "optimal weights")

    # Normalize to sum to 1
    weight_sum = np.sum(optimal_weights)
    if weight_sum <= 1e-10:
        raise ValueError(f"Optimal weights sum to near zero: {weight_sum}")

    optimal_weights = optimal_weights / weight_sum

    # Final validation
    validate_no_nan(optimal_weights, "normalized optimal weights")

    if not np.isclose(np.sum(optimal_weights), 1.0, rtol=1e-6):
        raise ValueError(f"Optimal weights sum {np.sum(optimal_weights)} != 1.0")

    if np.any(optimal_weights < -1e-6):
        raise ValueError(f"Optimal weights contain negative values: {optimal_weights}")

    return optimal_weights
