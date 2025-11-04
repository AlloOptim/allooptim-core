"""
Equal Weight A2A Orchestrator

Simplest orchestrator that calls each optimizer once and combines results with equal weights.
"""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from allo_optim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allo_optim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allo_optim.config.allocation_dataclasses import AllocationResult, NoStatistics
from allo_optim.config.stock_dataclasses import StockUniverse
from allo_optim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

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
    ):
        super().__init__(optimizers, covariance_transformers)

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> AllocationResult:
        """
        Run equal weight allocation orchestration.

        Args:
            data_provider: Provides ground truth parameters
            time_today: Current time step (optional)

        Returns:
            AllocationResult with equal-weighted optimizer combination
        """
        # Get ground truth parameters (no sampling for equal weight)
        mu, cov, prices, time, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, len(prices))

        # Initialize tracking
        asset_weights = np.zeros(len(mu))
        optimizer_allocations = {}

        # Call each optimizer once
        for optimizer in self.optimizers:
            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(prices)

                weights = optimizer.allocate(mu, cov_transformed, prices, time, l_moments)
                weights = np.array(weights).flatten()
                weights = weights / np.sum(weights)  # Normalize

                optimizer_allocations[optimizer.name] = pd.Series(weights, index=mu.index)
                asset_weights += weights

            except Exception as error:
                logger.warning(f"Allocation failed for {optimizer.name}: {str(error)}")
                # Use equal weights fallback
                equal_weights = np.ones(len(mu)) / len(mu)
                optimizer_allocations[optimizer.name] = pd.Series(
                    equal_weights, index=mu.index if hasattr(mu, "index") else range(len(mu))
                )
                asset_weights += equal_weights

        # Normalize final asset weights
        asset_weights = asset_weights / len(self.optimizers)
        asset_weights_dict = {asset: weight for asset, weight in zip(mu.index, asset_weights)}

        # Create result
        result = AllocationResult(
            asset_weights=asset_weights_dict,
            success=True,
            statistics=NoStatistics(),  # No statistics for equal weight
            df_allocation=pd.DataFrame(optimizer_allocations).T,
        )

        return result

    @property
    def name(self) -> str:
        return "EqualWeight_A2A"
