"""Equal Weight A2A Orchestrator.

Simplest orchestrator that calls each optimizer once and combines results with equal weights.
"""

import logging
import time
import tracemalloc
from datetime import datetime
from enum import Enum
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


class CombinedWeightType(str, Enum):
    """Enumeration of combined weight types."""

    EQUAL = "equal"
    CUSTOM = "custom"
    MEDIAN = "median"
    VOLATILITY = "volatility"


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
        a2a_config: A2AConfig,
    ) -> None:
        """Initialize the Equal Weight Orchestrator.

        Args:
            optimizers: List of portfolio optimization algorithms to orchestrate.
            covariance_transformers: List of covariance matrix transformations to apply.
            a2a_config: Configuration object with A2A orchestration parameters.
        """
        super().__init__(optimizers, covariance_transformers, a2a_config)

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
        if time_today is None:
            time_today = datetime.now()

        runtime_start = time.time()

        # Get ground truth parameters (no sampling for equal weight)
        mu, cov, prices, _, l_moments = data_provider.get_ground_truth()

        # Apply covariance transformations
        cov_transformed = self._apply_covariance_transformers(cov, data_provider.n_observations)

        # Initialize tracking
        optimizer_allocations_list: List[OptimizerAllocation] = []
        optimizer_weights_list: List[OptimizerWeight] = []

        # First pass: collect all optimizer allocations
        for optimizer in self.optimizers:
            tracemalloc.start()
            runtime_start = time.time()

            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(prices)

                weights = optimizer.allocate(mu, cov_transformed, prices, time_today, l_moments)
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
                optimizer_allocations_list.append(
                    OptimizerAllocation(
                        instance_id=optimizer.display_name,
                        weights=weights_series,
                        runtime_seconds=runtime_seconds,
                        memory_usage_mb=memory_usage_mb,
                    )
                )

            except Exception as error:
                tracemalloc.stop()
                logger.warning(f"Allocation failed for {optimizer.name}: {str(error)}")
                # Use configured failure handling strategy
                fallback_weights = self._handle_optimizer_failure(
                    optimizer=optimizer,
                    exception=error,
                    n_assets=len(mu),
                    asset_names=mu.index.tolist(),
                )

                if fallback_weights is not None:
                    # Store fallback allocation
                    optimizer_allocations_list.append(
                        OptimizerAllocation(
                            instance_id=optimizer.display_name,
                            weights=fallback_weights,
                            runtime_seconds=None,
                            memory_usage_mb=None,
                        )
                    )
                # If fallback_weights is None (IGNORE_OPTIMIZER), skip this optimizer entirely

        # Check if all optimizers failed
        if len(optimizer_allocations_list) == 0:
            if self.config.failure_handling.raise_on_all_failed:
                raise RuntimeError(
                    "All optimizers failed. No valid allocations available. "
                    "Check optimizer configurations and input data quality."
                )
            else:
                # Graceful degradation: add equal-weight fallback allocation
                logger.error("All optimizers failed, returning equal-weight fallback allocation")
                equal_weight = 1.0 / len(mu)
                fallback_allocation = OptimizerAllocation(
                    instance_id="EMERGENCY_FALLBACK",
                    weights=pd.Series(equal_weight, index=mu.index),
                    runtime_seconds=None,
                    memory_usage_mb=None,
                )
                optimizer_allocations_list.append(fallback_allocation)

        # Determine A2A weights and combine allocations based on combination method
        match self.combined_weight_type:
            case CombinedWeightType.EQUAL:
                # Equal weights for all optimizers
                a2a_weights = {
                    opt_alloc.instance_id: 1.0 / len(optimizer_allocations_list)
                    for opt_alloc in optimizer_allocations_list
                }
                # Weighted combination of optimizer allocations
                asset_weights = np.zeros(len(mu))
                for opt_alloc in optimizer_allocations_list:
                    weight = a2a_weights[opt_alloc.instance_id]
                    asset_weights += weight * opt_alloc.weights.values

            case CombinedWeightType.MEDIAN:
                # Equal weights for all optimizers (best approximation for median)
                a2a_weights = {
                    opt_alloc.instance_id: 1.0 / len(optimizer_allocations_list)
                    for opt_alloc in optimizer_allocations_list
                }
                # Take median across optimizer allocations for each asset
                alloc_df = pd.DataFrame(
                    {opt_alloc.instance_id: opt_alloc.weights for opt_alloc in optimizer_allocations_list}
                )
                asset_weights = alloc_df.median(axis=1).values

            case CombinedWeightType.VOLATILITY:
                # Equal weights for all optimizers (best approximation for volatility)
                a2a_weights = {
                    opt_alloc.instance_id: 1.0 / len(optimizer_allocations_list)
                    for opt_alloc in optimizer_allocations_list
                }
                # Volatility-adjusted weighting
                alloc_df = pd.DataFrame(
                    {opt_alloc.instance_id: opt_alloc.weights for opt_alloc in optimizer_allocations_list}
                )

                # Handle single optimizer case
                if len(optimizer_allocations_list) == 1:
                    asset_weights = alloc_df.iloc[:, 0].values
                else:
                    mean_asset = alloc_df.mean(axis=1)
                    variance_asset = np.clip(alloc_df.std(axis=1) ** 2, a_min=0.0, a_max=1.0)
                    variance_overall = variance_asset.mean()
                    asset_weights = (
                        1.0 - self.config.volatility_adjustment
                    ) * mean_asset + self.config.volatility_adjustment * (variance_overall - variance_asset)
                    asset_weights = np.clip(asset_weights, a_min=0.0, a_max=1.0)

            case CombinedWeightType.CUSTOM:
                a2a_weights = self.config.custom_a2a_weights.copy()
                # Weighted combination of optimizer allocations
                asset_weights = np.zeros(len(mu))
                for opt_alloc in optimizer_allocations_list:
                    weight = a2a_weights[opt_alloc.instance_id]
                    asset_weights += weight * opt_alloc.weights.values

            case _:
                raise ValueError(f"Unknown combined weight type: {self.combined_weight_type}")

        # Convert asset_weights to pandas Series for constraint application
        asset_weights = pd.Series(asset_weights, index=mu.index)

        # Store optimizer weights
        for opt_alloc in optimizer_allocations_list:
            optimizer_weights_list.append(
                OptimizerWeight(
                    instance_id=opt_alloc.instance_id,
                    weight=a2a_weights[opt_alloc.instance_id],
                )
            )

        # Apply allocation constraints
        asset_weights = AllocationConstraints.apply_all_constraints(
            weights=asset_weights,
            n_max_active_assets=self.config.n_max_active_assets,
            max_asset_concentration_pct=self.config.max_asset_concentration_pct,
            n_min_active_assets=self.config.n_min_active_assets,
            min_weight_threshold=self.config.min_weight_threshold,
        )

        # Normalize final asset weights according to global cash settings
        asset_weights_values = normalize_weights_a2a(
            asset_weights, self.config.cash_config
        )
        asset_weights = pd.Series(asset_weights_values, index=mu.index)

        # Compute performance metrics
        portfolio_return = (asset_weights * mu).sum()
        portfolio_variance = asset_weights.values @ cov_transformed.values @ asset_weights.values
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

        # Create optimizer errors (empty for equal weight)
        optimizer_errors = [
            OptimizerError(
                instance_id=opt.instance_id,
                error=0.0,  # No error estimation for equal weight
                error_components=[],
            )
            for opt in optimizer_allocations_list
        ]

        runtime_seconds = time.time() - runtime_start

        # Create A2AResult
        result = A2AResult(
            final_allocation=asset_weights,
            optimizer_allocations=optimizer_allocations_list,
            optimizer_weights=optimizer_weights_list,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            n_simulations=1,  # Equal weight uses ground truth only
            optimizer_errors=optimizer_errors,
            orchestrator_name=self.name,
            timestamp=time_today,
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


class VolatilityOrchestrator(EqualWeightOrchestrator):
    """Volatility-based Allocation-to-Allocators orchestrator.

    Combines optimizer allocations using a volatility-adjusted weighting scheme:
    - Computes mean (mu_asset) and std (std_asset) of weights across optimizers for each asset
    - Calculates overall std as mean of asset stds
    - Weights assets as: mu_asset + alpha * (std_overall - std_asset)
    - Clips negative weights to zero and normalizes
    """

    combined_weight_type = CombinedWeightType.VOLATILITY

    @property
    def name(self) -> str:
        """Get the orchestrator name identifier.

        Returns:
            String identifier for this orchestrator type.
        """
        return f"Volatility_A2A_alpha_{self.config.volatility_adjustment}"
