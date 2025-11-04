"""
Optimized A2A Orchestrator

Monte Carlo Optimization Selection (MCOS) orchestrator with PSO optimization of optimizer weights.
"""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from allo_optim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allo_optim.allocation_to_allocators.allocation_optimizer import (
    optimize_allocator_weights,
)
from allo_optim.allocation_to_allocators.optimizer_simulator import (
    simulate_optimizers_with_allocation_statistics,
)
from allo_optim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allo_optim.config.allocation_dataclasses import (
    A2AStatistics,
    AllocationResult,
)
from allo_optim.config.stock_dataclasses import StockUniverse
from allo_optim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class OptimizedOrchestrator(BaseOrchestrator):
    """
    Optimized Allocation-to-Allocators orchestrator using Monte Carlo + PSO.

    Process:
    1. Run Monte Carlo simulation to get optimizer performance statistics
    2. Use PSO to optimize weights for combining optimizer allocations
    3. Apply optimal weights to get final portfolio allocation

    This implements the full MCOS (Monte Carlo Optimization Selection) approach.
    """

    def __init__(
        self,
        optimizers: List[AbstractOptimizer],
        covariance_transformers: List[AbstractCovarianceTransformer],
        n_data_observations: int = 50,
        n_particle_swarm_iterations: int = 1000,
        n_particles: int = 1000,
    ):
        super().__init__(optimizers, covariance_transformers)
        self.n_data_observations = n_data_observations
        self.n_particle_swarm_iterations = n_particle_swarm_iterations
        self.n_particles = n_particles

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> AllocationResult:
        """
        Run optimized allocation orchestration with MCOS + PSO.

        Args:
            data_provider: Provides sampling capability for Monte Carlo
            time_today: Current time step (optional)

        Returns:
            AllocationResult with PSO-optimized optimizer combination
        """
        # Step 1: Run enhanced MCOS simulation
        mcos_result = simulate_optimizers_with_allocation_statistics(
            df_assets=data_provider.historical_prices,
            optimizer_list=self.optimizers,
            n_data_observations=self.n_data_observations,
        )

        # Validate MCOS result
        if mcos_result.expected_return_means is None:
            raise ValueError("MCOS simulation failed to generate expected return means")

        if mcos_result.expected_returns_covariance is None:
            raise ValueError("MCOS simulation failed to generate expected returns covariance")

        if len(mcos_result.expected_return_means) != len(self.optimizers):
            raise ValueError(
                f"MCOS generated {len(mcos_result.expected_return_means)} optimizer means "
                f"but expected {len(self.optimizers)}"
            )

        # Step 2: Optimize allocator weights using PSO
        allocator_weights = optimize_allocator_weights(
            mcos_result=mcos_result,
            n_particle_swarm_iterations=self.n_particle_swarm_iterations,
            n_particles=self.n_particles,
        )

        # Step 3: Compute final asset weights and statistics
        asset_weights_dict, statistics_dict, df_allocation, memory_usage, computation_time = (
            self._compute_final_asset_weights(
                data_provider=data_provider,
                time_today=time_today,
                allocator_weights=allocator_weights,
            )
        )

        # Build result
        statistics = A2AStatistics(
            asset_returns=statistics_dict["returns"],
            asset_volatilities=statistics_dict["volatilities"],
            algo_runtime=statistics_dict["runtime"],
            algo_weights=statistics_dict["algo_weights"],
            algo_memory_usage=memory_usage,
            algo_computation_time=computation_time,
        )

        result = AllocationResult(
            asset_weights=asset_weights_dict,
            success=True,
            statistics=statistics,
            df_allocation=df_allocation,
            optimizer_memory_usage=memory_usage,
            optimizer_computation_time=computation_time,
        )

        return result

    def _compute_final_asset_weights(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: datetime,
        allocator_weights: np.ndarray,
    ) -> tuple:
        """
        Compute final asset weights and performance statistics.

        Args:
            data_provider: Provides ground truth parameters
            time_today: Current time step
            allocator_weights: Optimal weights for each allocator

        Returns:
            Tuple of (asset_weights_dict, statistics_dict, df_allocation, memory_usage, computation_time)
        """
        import tracemalloc
        from timeit import default_timer as timer

        if len(self.optimizers) != len(allocator_weights):
            raise ValueError(f"Optimizer count {len(self.optimizers)} != weight count {len(allocator_weights)}")

        # Get ground truth parameters
        mu, cov, prices, time, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, len(prices))

        # Initialize tracking
        asset_weights = np.zeros(len(mu))
        statistic_returns = {}
        statistic_volatilities = {}
        statistic_runtime = {"A2A": np.nan}

        # Track individual optimizer allocations for df_allocation
        optimizer_allocations = {}

        # Track memory and computation time per optimizer
        memory_usage = {}
        computation_time = {}

        # Compute weighted asset allocation
        for k, optimizer in enumerate(self.optimizers):
            # Track memory and time
            tracemalloc.start()
            start = timer()

            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(prices)

                weights = optimizer.allocate(mu, cov_transformed, prices, time, l_moments)
                weights = np.array(weights).flatten()
                weights = weights / np.sum(weights)  # Normalize

                optimizer_allocations[optimizer.name] = pd.Series(weights, index=mu.index)

            except Exception as error:
                optimizer.reset()
                raise RuntimeError(f"Allocation failed for {optimizer.name}: {str(error)}") from error

            end = timer()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Store memory and computation time
            memory_usage[optimizer.name] = peak / 1024 / 1024  # Convert to MB
            computation_time[optimizer.name] = end - start

            # Compute optimizer statistics
            expected_return = (weights * mu).sum()
            expected_volatility = np.sqrt((weights * (cov_transformed @ weights)).sum())

            # Accumulate weighted asset allocation
            asset_weights += allocator_weights[k] * weights

            # Store statistics
            statistic_returns[optimizer.name] = expected_return
            statistic_volatilities[optimizer.name] = expected_volatility
            statistic_runtime[optimizer.name] = end - start

        # Build df_allocation DataFrame (rows=optimizers, cols=assets)
        df_allocation = pd.DataFrame(optimizer_allocations).T if optimizer_allocations else None

        # Final validation of asset weights
        asset_weights_dict = {asset: weight for asset, weight in zip(mu.index, asset_weights)}

        # Compute A2A portfolio statistics
        expected_return_a2a = np.dot(asset_weights, mu)
        expected_volatility_a2a = np.sqrt(np.dot(asset_weights, np.dot(cov_transformed, asset_weights)))

        statistic_returns["A2A"] = expected_return_a2a
        statistic_volatilities["A2A"] = expected_volatility_a2a

        # Build result dictionaries
        statistic_algo_weights = {
            optimizer.name: weight for optimizer, weight in zip(self.optimizers, allocator_weights)
        }

        statistics_dict = {
            "returns": statistic_returns,
            "volatilities": statistic_volatilities,
            "runtime": statistic_runtime,
            "algo_weights": statistic_algo_weights,
            "asset_weights": asset_weights_dict,
        }

        return asset_weights_dict, statistics_dict, df_allocation, memory_usage, computation_time

    @property
    def name(self) -> str:
        return "Optimized_A2A"
