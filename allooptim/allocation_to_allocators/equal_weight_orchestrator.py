"""Equal Weight A2A Orchestrator.

Simplest orchestrator that calls each optimizer once and combines results with equal weights.
"""

import logging
import time
from datetime import datetime
from enum import Enum
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


class CombinedWeightType(str, Enum):
    """Enumeration of combined weight types."""

    EQUAL = "equal"
    CUSTOM = "custom"
    MEDIAN = "median"


class EqualWeightOrchestrator(BaseOrchestrator):
    """Configurable Weight Allocation-to-Allocators orchestrator.

    Supports multiple weight combination methods:
    - EQUAL: Equal weights for all optimizers
    - MEDIAN: Take median allocation across optimizers for each asset
    - CUSTOM: Use custom weights specified in config

    Process:
    1. Call each optimizer once with ground truth parameters
    2. Combine optimizer allocations using specified method
    3. Return final portfolio allocation
    """

    combined_weight_type = CombinedWeightType.EQUAL

    def __init__(
        self,
        optimizers: List[AbstractOptimizer],
        covariance_transformers: List[AbstractCovarianceTransformer],
        config: A2AConfig,
    ) -> None:
        """Initialize the Equal Weight Orchestrator.

        Args:
            optimizers: List of portfolio optimization algorithms to orchestrate.
            covariance_transformers: List of covariance matrix transformations to apply.
            config: Configuration object with A2A orchestration parameters.
        """
        super().__init__(optimizers, covariance_transformers, config)

        if self.combined_weight_type == CombinedWeightType.CUSTOM:
            if not self.config.custom_a2a_weights:
                raise ValueError("Custom A2A weights must be provided in config for CUSTOM weight type.")

            if set(self.config.custom_a2a_weights.keys()) != set(opt.name for opt in self.optimizers):
                raise ValueError("Custom A2A weights keys must match optimizer names.")

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> A2AResult:
        """Run allocation orchestration with configurable weight combination.

        Supports EQUAL, MEDIAN, and CUSTOM weight combination methods
        as specified by self.combined_weight_type.

        Args:
            data_provider: Provides ground truth parameters
            time_today: Current time step (optional)
            all_stocks: List of all available stocks (optional)

        Returns:
            A2AResult with combined optimizer allocations
        """
        start_time = time.time()

        # Get ground truth parameters (no sampling for equal weight)
        mu, cov, prices, current_time, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, data_provider.n_observations)

        # Initialize tracking
        optimizer_allocations_list: List[OptimizerAllocation] = []
        optimizer_weights_list: List[OptimizerWeight] = []

        # First pass: collect all optimizer allocations
        for optimizer in self.optimizers:
            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(prices)

                weights = optimizer.allocate(mu, cov_transformed, prices, current_time, l_moments)
                if isinstance(weights, np.ndarray):
                    weights = weights.flatten()

                weights = np.array(weights)
                weights_sum = np.sum(weights)
                if weights_sum > 0 and (not self.config.allow_partial_investment or weights_sum > 1.0):
                    weights = weights / weights_sum

                # Store optimizer allocation
                weights_series = pd.Series(weights, index=mu.index)
                optimizer_allocations_list.append(
                    OptimizerAllocation(optimizer_name=optimizer.name, weights=weights_series)
                )

            except Exception as error:
                logger.warning(f"Allocation failed for {optimizer.name}: {str(error)}")
                # Use equal weights fallback
                equal_weights = np.ones(len(mu)) / len(mu)
                weights_series = pd.Series(equal_weights, index=mu.index)

                optimizer_allocations_list.append(
                    OptimizerAllocation(optimizer_name=optimizer.name, weights=weights_series)
                )

        # Second pass: determine A2A weights based on combination method
        match self.combined_weight_type:
            case CombinedWeightType.EQUAL | CombinedWeightType.MEDIAN:
                # Equal weights for all optimizers
                # For median, this is the best approximation
                a2a_weights = {
                    opt.optimizer_name: 1.0 / len(optimizer_allocations_list) for opt in optimizer_allocations_list
                }

            case CombinedWeightType.CUSTOM:
                a2a_weights = self.config.custom_a2a_weights.copy()

            case _:
                raise ValueError(f"Unknown combined weight type: {self.combined_weight_type}")

        # Third pass: combine allocations based on method
        match self.combined_weight_type:
            case CombinedWeightType.EQUAL | CombinedWeightType.CUSTOM:
                # Weighted combination of optimizer allocations
                asset_weights = np.zeros(len(mu))
                for opt_alloc in optimizer_allocations_list:
                    weight = a2a_weights[opt_alloc.optimizer_name]
                    asset_weights += weight * opt_alloc.weights.values

            case CombinedWeightType.MEDIAN:
                # Take median across optimizer allocations for each asset
                alloc_df = pd.DataFrame({opt.optimizer_name: opt.weights for opt in optimizer_allocations_list})
                asset_weights = alloc_df.median(axis=1).values

            case _:
                raise ValueError(f"Unknown combined weight type: {self.combined_weight_type}")

        sum_asset_weights = np.sum(asset_weights)
        if sum_asset_weights > 0 and (not self.config.allow_partial_investment or sum_asset_weights > 1.0):
            asset_weights = asset_weights / sum_asset_weights

        # Store optimizer weights
        for opt_alloc in optimizer_allocations_list:
            optimizer_weights_list.append(
                OptimizerWeight(
                    optimizer_name=opt_alloc.optimizer_name,
                    weight=a2a_weights[opt_alloc.optimizer_name],
                )
            )

        # Normalize final asset weights
        final_allocation = pd.Series(asset_weights, index=mu.index)
        final_allocation_sum = final_allocation.sum()
        if final_allocation_sum > 0:
            if not self.config.allow_partial_investment or final_allocation_sum > 1.0:
                final_allocation = final_allocation / final_allocation_sum

        else:
            logger.warning("Final allocation sums to zero; returning zero weights.")
            final_allocation = pd.Series(0.0, index=mu.index)

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
        """Get the orchestrator name identifier.

        Returns:
            String identifier for this orchestrator type.
        """
        return "EqualWeight_A2A"


class MedianWeightOrchestrator(EqualWeightOrchestrator):
    """Median Weight Allocation-to-Allocators orchestrator.

    Takes the median allocation across all optimizers for each asset.
    """

    combined_weight_type = CombinedWeightType.MEDIAN

    @property
    def name(self) -> str:
        """Get the orchestrator name identifier.

        Returns:
            String identifier for this orchestrator type.
        """
        return "MedianWeight_A2A"


class CustomWeightOrchestrator(EqualWeightOrchestrator):
    """Custom Weight Allocation-to-Allocators orchestrator.

    Uses custom weights specified in config to combine optimizer allocations.
    """

    combined_weight_type = CombinedWeightType.CUSTOM

    @property
    def name(self) -> str:
        """Get the orchestrator name identifier.

        Returns:
            String identifier for this orchestrator type.
        """
        return "CustomWeight_A2A"
