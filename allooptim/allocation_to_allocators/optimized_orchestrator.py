"""Optimized A2A Orchestrator.

Monte Carlo Optimization Selection (MCOS) orchestrator with PSO optimization of optimizer weights.
"""

import logging
import time
import tracemalloc
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from allooptim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allooptim.allocation_to_allocators.a2a_result import (
    A2AResult,
    OptimizerAllocation,
    OptimizerError,
    OptimizerWeight,
    PerformanceMetrics,
)
from allooptim.allocation_to_allocators.allocation_constraints import AllocationConstraints
from allooptim.allocation_to_allocators.allocation_optimizer import optimize_allocator_weights
from allooptim.allocation_to_allocators.optimizer_simulator import (
    simulate_optimizers_with_allocation_statistics,
)
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.config.a2a_config import A2AConfig
from allooptim.config.cash_config import normalize_weights_a2a
from allooptim.config.stock_dataclasses import StockUniverse
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class OptimizedOrchestrator(BaseOrchestrator):
    """Optimized Allocation-to-Allocators orchestrator using Monte Carlo + PSO.

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
        a2a_config: A2AConfig,
    ):
        """Initialize the Optimized Orchestrator.

        Args:
            optimizers: List of portfolio optimization algorithms to orchestrate.
            covariance_transformers: List of covariance matrix transformations to apply.
            a2a_config: Configuration object with A2A orchestration parameters including
                Monte Carlo simulation count and PSO optimization settings.
        """
        super().__init__(optimizers, covariance_transformers, a2a_config)

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> A2AResult:
        """Run optimized allocation orchestration with MCOS + PSO.

        Args:
            data_provider: Provides sampling capability for Monte Carlo
            time_today: Current time step (optional)
            all_stocks: List of all available stocks (optional)

        Returns:
            A2AResult with PSO-optimized optimizer combination
        """
        start_time = time.time()

        if time_today is None:
            logger.debug("time_today not provided, using current datetime.")
            time_today = datetime.now()

        # Special case: single optimizer doesn't need optimization
        if len(self.optimizers) == 1:
            logger.info("Single optimizer detected, skipping MCOS + PSO optimization")
            return self._allocate_single_optimizer(data_provider, time_today)

        # Step 1: Run enhanced MCOS simulation
        mcos_result = simulate_optimizers_with_allocation_statistics(
            df_assets=data_provider.historical_prices,
            optimizer_list=self.optimizers,
            config=self.config,
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
            n_particle_swarm_iterations=self.config.n_pso_iterations,  # Use config
            n_particles=self.config.n_particles,  # Use config
        )

        # Step 3: Compute final asset weights and structured results
        final_allocation, optimizer_allocations_list, optimizer_weights_list, metrics, optimizer_errors = (
            self._compute_final_asset_weights_and_metrics(
                data_provider=data_provider,
                time_today=time_today,
                allocator_weights=allocator_weights,
            )
        )

        # Apply allocation constraints
        final_allocation = AllocationConstraints.apply_all_constraints(
            weights=final_allocation,
            n_max_active_assets=self.config.n_max_active_assets,
            max_asset_concentration_pct=self.config.max_asset_concentration_pct,
            n_min_active_assets=self.config.n_min_active_assets,
            min_weight_threshold=self.config.min_weight_threshold,
        )

        runtime_seconds = time.time() - start_time

        # Create A2AResult
        result = A2AResult(
            final_allocation=final_allocation,
            optimizer_allocations=optimizer_allocations_list,
            optimizer_weights=optimizer_weights_list,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            n_simulations=self.config.n_simulations,
            optimizer_errors=optimizer_errors,
            orchestrator_name=self.name,
            timestamp=time_today,
            config=self.config,
        )

        return result

    def _allocate_single_optimizer(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: datetime,
    ) -> A2AResult:
        """Handle allocation for single optimizer case (no optimization needed)."""
        start_time = time.time()

        # Get ground truth parameters
        mu, cov, prices, current_time, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, data_provider.n_observations)

        optimizer = self.optimizers[0]

        # Track memory and time
        tracemalloc.start()
        runtime_start = time.time()

        try:
            logger.info(f"Computing allocation for {optimizer.name}...")

            optimizer.fit(prices)

            weights = optimizer.allocate(mu, cov_transformed, prices, current_time, l_moments)
            if isinstance(weights, np.ndarray):
                weights = weights.flatten()

            weights = np.array(weights)

            # Track memory and time
            runtime_end = time.time()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            runtime_seconds = runtime_end - runtime_start
            memory_usage_mb = peak / (1024 * 1024)  # Convert bytes to MB

            # Store optimizer allocation
            weights_series = pd.Series(weights, index=mu.index)
            optimizer_allocations_list = [
                OptimizerAllocation(
                    instance_id=optimizer.display_name,
                    weights=weights_series,
                    runtime_seconds=runtime_seconds,
                    memory_usage_mb=memory_usage_mb,
                )
            ]

            # Store optimizer weight (1.0 for single optimizer)
            optimizer_weights_list = [OptimizerWeight(instance_id=optimizer.display_name, weight=1.0)]

        except Exception as error:
            tracemalloc.stop()
            optimizer.reset()
            # For standalone usage (single optimizer), always use EQUAL_WEIGHTS fallback
            n_assets = len(mu)
            equal_weight = 1.0 / n_assets
            fallback_weights = pd.Series(equal_weight, index=mu.index, name=optimizer.name)

            # Store fallback allocation
            weights_series = fallback_weights
            optimizer_allocations_list = [
                OptimizerAllocation(
                    instance_id=optimizer.display_name,
                    weights=weights_series,
                    runtime_seconds=None,
                    memory_usage_mb=None,
                )
            ]

            # Store optimizer weight (1.0 for single optimizer)
            optimizer_weights_list = [OptimizerWeight(instance_id=optimizer.display_name, weight=1.0)]

        # Create optimizer errors
        optimizer_errors = [
            OptimizerError(
                instance_id=optimizer.display_name,
                error=0.0,
                error_components=[],
            )
        ]

        # Final allocation is just the optimizer's weights
        final_allocation = weights_series

        # Apply allocation constraints
        final_allocation = AllocationConstraints.apply_all_constraints(
            weights=final_allocation,
            n_max_active_assets=self.config.n_max_active_assets,
            max_asset_concentration_pct=self.config.max_asset_concentration_pct,
            n_min_active_assets=self.config.n_min_active_assets,
            min_weight_threshold=self.config.min_weight_threshold,
        )

        # Compute performance metrics
        portfolio_return = (final_allocation * mu).sum()
        portfolio_variance = final_allocation.values @ cov_transformed.values @ final_allocation.values
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Diversity score is 0 for single optimizer
        diversity_score = 0.0

        metrics = PerformanceMetrics(
            expected_return=float(portfolio_return),
            volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            diversity_score=float(diversity_score),
        )

        runtime_seconds = time.time() - start_time

        # Create A2AResult
        result = A2AResult(
            final_allocation=final_allocation,
            optimizer_allocations=optimizer_allocations_list,
            optimizer_weights=optimizer_weights_list,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            n_simulations=1,  # Single optimizer uses ground truth only
            optimizer_errors=optimizer_errors,
            orchestrator_name=self.name,
            timestamp=time_today,
            config=self.config,
        )

        return result

    def _compute_final_asset_weights_and_metrics(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: datetime,
        allocator_weights: np.ndarray,
    ) -> tuple:
        """Compute final asset weights and performance metrics using structured models.

        Args:
            data_provider: Provides ground truth parameters
            time_today: Current time step
            allocator_weights: Optimal weights for each allocator

        Returns:
            Tuple of (final_allocation, optimizer_allocations_list, optimizer_weights_list, metrics, optimizer_errors)
        """
        if len(self.optimizers) != len(allocator_weights):
            raise ValueError(f"Optimizer count {len(self.optimizers)} != weight count {len(allocator_weights)}")

        # Get ground truth parameters
        mu, cov, prices, current_time, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, data_provider.n_observations)

        # Initialize tracking
        asset_weights = np.zeros(len(mu))
        optimizer_allocations_list = []
        optimizer_weights_list = []
        optimizer_errors = []

        # Compute weighted asset allocation
        for k, optimizer in enumerate(self.optimizers):
            # Track memory and time
            tracemalloc.start()
            runtime_start = time.time()

            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(prices)

                weights = optimizer.allocate(mu, cov_transformed, prices, current_time, l_moments)
                if isinstance(weights, np.ndarray):
                    weights = weights.flatten()

                weights = np.array(weights)

                # Track memory and time
                runtime_end = time.time()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                runtime_seconds = runtime_end - runtime_start
                memory_usage_mb = peak / (1024 * 1024)  # Convert bytes to MB

                # Add weighted contribution to final allocation
                asset_weights += allocator_weights[k] * weights

                # Store optimizer allocation
                weights_series = pd.Series(weights, index=mu.index)
                optimizer_allocations_list.append(
                    OptimizerAllocation(
                        instance_id=optimizer.display_name,
                        weights=weights_series,
                        runtime_seconds=runtime_seconds,
                        memory_usage_mb=memory_usage_mb,
                    )
                )

                # Store optimizer weight
                optimizer_weights_list.append(
                    OptimizerWeight(instance_id=optimizer.display_name, weight=float(allocator_weights[k]))
                )

            except Exception as error:
                tracemalloc.stop()
                optimizer.reset()
                # Use configured failure handling strategy for A2A usage
                fallback_weights = self._handle_optimizer_failure(
                    optimizer=optimizer,
                    exception=error,
                    n_assets=len(mu),
                    asset_names=mu.index.tolist(),
                )

                if fallback_weights is not None:
                    # Add weighted contribution to final allocation (even for fallback)
                    asset_weights += allocator_weights[k] * fallback_weights.values

                    # Store fallback allocation
                    optimizer_allocations_list.append(
                        OptimizerAllocation(
                            instance_id=optimizer.display_name,
                            weights=fallback_weights,
                            runtime_seconds=None,
                            memory_usage_mb=None,
                        )
                    )

                    # Store optimizer weight (still use allocated weight even for fallback)
                    optimizer_weights_list.append(
                        OptimizerWeight(instance_id=optimizer.display_name, weight=float(allocator_weights[k]))
                    )
                # If fallback_weights is None (IGNORE_OPTIMIZER), skip this optimizer entirely
                # Note: This breaks the correspondence between allocator_weights and optimizer lists

            # Create optimizer error (simplified - in future this should use error estimators)
            optimizer_errors.append(
                OptimizerError(
                    instance_id=optimizer.display_name,
                    error=0.0,  # Placeholder - should compute actual error
                    error_components=[],
                )
            )

        # Check if all optimizers failed
        if len(optimizer_allocations_list) == 0:
            if self.config.failure_handling.raise_on_all_failed:
                raise RuntimeError(
                    "All optimizers failed. No valid allocations available. "
                    "Check optimizer configurations and input data quality."
                )
            else:
                # Graceful degradation: add equal-weight fallback allocation
                logger.error(
                    "All optimizers failed, returning equal-weight fallback allocation"
                )
                equal_weight = 1.0 / len(mu)
                fallback_weights = pd.Series(equal_weight, index=mu.index)
                
                # Add to asset_weights with weight 1.0 (since this is the only allocation)
                asset_weights += 1.0 * fallback_weights.values
                
                fallback_allocation = OptimizerAllocation(
                    instance_id="EMERGENCY_FALLBACK",
                    weights=fallback_weights,
                    runtime_seconds=None,
                    memory_usage_mb=None,
                )
                optimizer_allocations_list.append(fallback_allocation)
                optimizer_weights_list.append(
                    OptimizerWeight(instance_id="EMERGENCY_FALLBACK", weight=1.0)
                )

        final_allocation_values = normalize_weights_a2a(asset_weights, self.config.cash_config)
        final_allocation = pd.Series(final_allocation_values, index=mu.index)

        # Compute performance metrics
        portfolio_return = (final_allocation * mu).sum()
        portfolio_variance = final_allocation.values @ cov_transformed.values @ final_allocation.values
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Compute diversity score (1 - mean correlation)
        optimizer_alloc_df = pd.DataFrame({alloc.instance_id: alloc.weights for alloc in optimizer_allocations_list})
        corr_matrix = optimizer_alloc_df.corr()
        n = len(corr_matrix)
        if n <= 1:
            diversity_score = 0.0
        else:
            avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))
            diversity_score = 1 - avg_corr

        metrics = PerformanceMetrics(
            expected_return=float(portfolio_return),
            volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            diversity_score=float(diversity_score),
        )

        return final_allocation, optimizer_allocations_list, optimizer_weights_list, metrics, optimizer_errors

    @property
    def name(self) -> str:
        """Get the orchestrator name identifier.

        Returns:
            String identifier for this orchestrator type.
        """
        return "Optimized_A2A"
