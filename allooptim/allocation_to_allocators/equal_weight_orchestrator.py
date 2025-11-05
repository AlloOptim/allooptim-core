"""
Equal Weight A2A Orchestrator

Simplest orchestrator that calls each optimizer once and combines results with equal weights.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from allooptim.allocation_to_allocators.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allooptim.allocation_to_allocators.a2a_result import (
    A2AResult,
    OptimizerAllocation,
    OptimizerError,
    OptimizerWeight,
    PerformanceMetrics,
)
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.config.stock_dataclasses import StockUniverse
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class EqualWeightOrchestrator(BaseOrchestrator):
    """
    Equal Weight Allocation-to-Allocators orchestrator.

    Process:
    1. Call each optimizer once with ground truth parameters
    2. Combine optimizer allocations using equal weights
    3. Return final portfolio allocation

    This is the simplest orchestrator - no Monte Carlo sampling,
    no optimization of optimizer weights.
    """

    def __init__(
        self,
        optimizers: List[AbstractOptimizer],
        covariance_transformers: List[AbstractCovarianceTransformer],
        config: A2AConfig,
    ):
        super().__init__(optimizers, covariance_transformers, config)

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> A2AResult:
        """
        Run equal weight allocation orchestration.

        Args:
            data_provider: Provides ground truth parameters
            time_today: Current time step (optional)

        Returns:
            A2AResult with equal-weighted optimizer combination
        """
        start_time = time.time()

        # Get ground truth parameters (no sampling for equal weight)
        mu, cov, prices, current_time, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, data_provider.n_observations)

        # Initialize tracking
        asset_weights = np.zeros(len(mu))
        optimizer_allocations_list = []
        optimizer_weights_list = []

        # Equal weights for all optimizers
        equal_weight = 1.0 / len(self.optimizers)

        # Call each optimizer once
        for optimizer in self.optimizers:
            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(prices)

                weights = optimizer.allocate(mu, cov_transformed, prices, current_time, l_moments)
                if isinstance(weights, np.ndarray):
                    weights = weights.flatten()
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize

                # Store optimizer allocation
                weights_series = pd.Series(weights, index=mu.index)
                optimizer_allocations_list.append(
                    OptimizerAllocation(optimizer_name=optimizer.name, weights=weights_series)
                )

                # Store optimizer weight
                optimizer_weights_list.append(OptimizerWeight(optimizer_name=optimizer.name, weight=equal_weight))

                asset_weights += equal_weight * weights

            except Exception as error:
                logger.warning(f"Allocation failed for {optimizer.name}: {str(error)}")
                # Use equal weights fallback
                equal_weights = np.ones(len(mu)) / len(mu)
                weights_series = pd.Series(equal_weights, index=mu.index)

                optimizer_allocations_list.append(
                    OptimizerAllocation(optimizer_name=optimizer.name, weights=weights_series)
                )

                optimizer_weights_list.append(OptimizerWeight(optimizer_name=optimizer.name, weight=equal_weight))

                asset_weights += equal_weight * equal_weights

        # Normalize final asset weights
        final_allocation = pd.Series(asset_weights, index=mu.index)
        final_allocation = final_allocation / final_allocation.sum()

        # Compute performance metrics
        portfolio_return = (final_allocation * mu).sum()
        portfolio_variance = final_allocation.values @ cov_transformed.values @ final_allocation.values
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Compute diversity score (1 - mean correlation)
        optimizer_alloc_df = pd.DataFrame({alloc.optimizer_name: alloc.weights for alloc in optimizer_allocations_list})
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

        # Create optimizer errors (empty for equal weight)
        optimizer_errors = [
            OptimizerError(
                optimizer_name=opt.optimizer_name,
                error=0.0,  # No error estimation for equal weight
                error_components=[],
            )
            for opt in optimizer_allocations_list
        ]

        runtime_seconds = time.time() - start_time

        # Create A2AResult
        result = A2AResult(
            final_allocation=final_allocation,
            optimizer_allocations=optimizer_allocations_list,
            optimizer_weights=optimizer_weights_list,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            n_simulations=1,  # Equal weight uses ground truth only
            optimizer_errors=optimizer_errors,
            orchestrator_name=self.name,
            timestamp=current_time or datetime.now(),
            config=self.config,
        )

        return result

    @property
    def name(self) -> str:
        return "EqualWeight_A2A"
