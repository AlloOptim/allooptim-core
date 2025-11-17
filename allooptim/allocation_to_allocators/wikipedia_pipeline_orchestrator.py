"""Wikipedia Pipeline A2A Orchestrator.

Orchestrator that combines Wikipedia-based stock pre-selection with portfolio optimization.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from allooptim.config.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allooptim.allocation_to_allocators.a2a_result import (
    A2AResult,
    OptimizerAllocation,
    PerformanceMetrics,
)
from allooptim.allocation_to_allocators.equal_weight_orchestrator import (
    EqualWeightOrchestrator,
)
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.config.stock_dataclasses import StockUniverse
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.wikipedia.allocate_wikipedia import allocate_wikipedia

logger = logging.getLogger(__name__)


class WikipediaPipelineOrchestrator(BaseOrchestrator):
    """Wikipedia Pipeline Allocation-to-Allocators orchestrator.

    Process:
    1. Use allocate_wikipedia for pre-selection of stocks with significant Wikipedia correlation
    2. Filter data provider to only include pre-selected stocks
    3. Run equal weight optimization on the filtered dataset
    4. Return the optimized portfolio allocation with padding for non-selected assets

    This orchestrator combines stock selection with portfolio optimization.
    """

    def __init__(
        self,
        optimizers: List[AbstractOptimizer],
        covariance_transformers: List[AbstractCovarianceTransformer],
        config: A2AConfig,
        n_historical_days: int = 60,
        use_wiki_database: bool = False,
        wiki_database_path: Optional[Path] = None,
    ):
        """Initialize the Wikipedia Pipeline Orchestrator.

        Args:
            optimizers: List of portfolio optimization algorithms to orchestrate.
            covariance_transformers: List of covariance matrix transformations to apply.
            config: Configuration object with A2A orchestration parameters.
            n_historical_days: Number of historical days to analyze for Wikipedia correlations.
            use_wiki_database: Whether to use local Wikipedia database instead of API calls.
            wiki_database_path: Path to local Wikipedia database file (if use_wiki_database=True).
        """
        super().__init__(optimizers, covariance_transformers, config)
        self.n_historical_days = n_historical_days
        self.use_wiki_database = use_wiki_database
        self.wiki_database_path = wiki_database_path

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> A2AResult:
        """Run Wikipedia pipeline allocation orchestration.

        Args:
            data_provider: Provides full dataset parameters
            time_today: Current time step
            all_stocks: List of all available stocks for Wikipedia analysis

        Returns:
            AllocationResult with Wikipedia-filtered optimization
        """
        if time_today is None:
            raise ValueError("time_today is required for Wikipedia pipeline orchestration")

        if all_stocks is None:
            raise ValueError("all_stocks is required for Wikipedia pipeline orchestration")

        start_time = time.time()

        try:
            # Localize time_today if it's timezone-naive
            if time_today.tzinfo is None:
                time_today = time_today.tz_localize("UTC")

            # Step 1: Get pre-selection from Wikipedia allocation
            wikipedia_result = allocate_wikipedia(
                all_stocks=all_stocks,
                time_today=time_today,
                n_historical_days=self.n_historical_days,
                use_wiki_database=self.use_wiki_database,
                wiki_database_path=self.wiki_database_path,
            )

            if not wikipedia_result.success:
                # Return failed result if wikipedia allocation failed
                runtime_seconds = time.time() - start_time
                # Create minimal A2AResult for failure case
                empty_allocation = pd.Series(dtype=float)
                return A2AResult(
                    final_allocation=empty_allocation,
                    optimizer_allocations=[],
                    optimizer_weights=[],
                    metrics=PerformanceMetrics(
                        expected_return=0.0, volatility=0.0, sharpe_ratio=0.0, diversity_score=0.0
                    ),
                    runtime_seconds=runtime_seconds,
                    n_simulations=0,
                    optimizer_errors=[],
                    orchestrator_name=self.name,
                    timestamp=time_today,
                    config=self.config,
                )

            # Step 2: Extract pre-selected stocks (those with non-zero weights)
            preselected_stocks = [symbol for symbol, weight in wikipedia_result.asset_weights.items() if weight > 0.0]

            logger.info(f"Pre-selected {len(preselected_stocks)} / {len(all_stocks)} stocks from Wikipedia allocation")

            if not preselected_stocks:
                # Return failed result if no stocks were pre-selected
                runtime_seconds = time.time() - start_time
                empty_allocation = pd.Series(dtype=float)
                return A2AResult(
                    final_allocation=empty_allocation,
                    optimizer_allocations=[],
                    optimizer_weights=[],
                    metrics=PerformanceMetrics(
                        expected_return=0.0, volatility=0.0, sharpe_ratio=0.0, diversity_score=0.0
                    ),
                    runtime_seconds=runtime_seconds,
                    n_simulations=0,
                    optimizer_errors=[],
                    orchestrator_name=self.name,
                    timestamp=time_today,
                    config=self.config,
                )

            # Step 3: Filter data provider to only include pre-selected stocks
            # Get ground truth data
            mu_full, cov_full, prices_full, time_full, l_moments_full = data_provider.get_ground_truth()

            # Filter to pre-selected stocks
            available_columns = [col for col in preselected_stocks if col in prices_full.columns]
            if not available_columns:
                # Return failed result if none of the pre-selected stocks are in price data
                runtime_seconds = time.time() - start_time
                empty_allocation = pd.Series(dtype=float)
                return A2AResult(
                    final_allocation=empty_allocation,
                    optimizer_allocations=[],
                    optimizer_weights=[],
                    metrics=PerformanceMetrics(
                        expected_return=0.0, volatility=0.0, sharpe_ratio=0.0, diversity_score=0.0
                    ),
                    runtime_seconds=runtime_seconds,
                    n_simulations=0,
                    optimizer_errors=[],
                    orchestrator_name=self.name,
                    timestamp=time_today,
                    config=self.config,
                )

            prices_filtered = prices_full[available_columns]
            mu_filtered = mu_full[available_columns]
            cov_filtered = cov_full.loc[available_columns, available_columns]

            # Create filtered data provider (simplified approach - just use filtered data)
            # In a more complete implementation, we'd create a FilteredDataProvider
            filtered_data_provider = _FilteredDataProvider(
                mu_filtered, cov_filtered, prices_filtered, time_full, l_moments_full
            )

            # Step 4: Run equal weight optimization on filtered data
            equal_orchestrator = EqualWeightOrchestrator(self.optimizers, self.covariance_transformers, self.config)
            optimization_result = equal_orchestrator.allocate(filtered_data_provider, time_today)

            # Step 5: Pad weights with zeros for non-selected assets
            # Create full weight dict with all original assets
            all_assets = prices_full.columns.tolist()
            final_allocation = pd.Series(0.0, index=all_assets)

            # Update with actual weights from optimization
            if optimization_result.final_allocation is not None:
                final_allocation.update(optimization_result.final_allocation)

            # Ensure weights sum to 1
            total_weight = final_allocation.sum()
            if self.config.allow_partial_investment and total_weight > 0:
                final_allocation = final_allocation / total_weight

            # Step 6: Pad optimizer allocations with zeros for non-selected assets
            padded_optimizer_allocations = []
            for opt_alloc in optimization_result.optimizer_allocations:
                # Create full weights series with zeros
                full_weights = pd.Series(0.0, index=all_assets)
                # Update with actual weights for selected assets
                full_weights.update(opt_alloc.weights)
                padded_optimizer_allocations.append(
                    OptimizerAllocation(instance_id=opt_alloc.instance_id, weights=full_weights)
                )

            runtime_seconds = time.time() - start_time

            # Return the A2AResult from optimization (with padded allocation)
            return A2AResult(
                final_allocation=final_allocation,
                optimizer_allocations=padded_optimizer_allocations,
                optimizer_weights=optimization_result.optimizer_weights,
                metrics=optimization_result.metrics,
                runtime_seconds=runtime_seconds,
                n_simulations=optimization_result.n_simulations,
                optimizer_errors=optimization_result.optimizer_errors,
                orchestrator_name=self.name,
                timestamp=time_today,
                config=self.config,
            )

        except Exception as e:
            runtime_seconds = time.time() - start_time
            logger.error(f"Wikipedia pipeline allocation failed: {str(e)}")
            empty_allocation = pd.Series(dtype=float)
            return A2AResult(
                final_allocation=empty_allocation,
                optimizer_allocations=[],
                optimizer_weights=[],
                metrics=PerformanceMetrics(expected_return=0.0, volatility=0.0, sharpe_ratio=0.0, diversity_score=0.0),
                runtime_seconds=runtime_seconds,
                n_simulations=0,
                optimizer_errors=[],
                orchestrator_name=self.name,
                timestamp=time_today or datetime.now(),
                config=self.config,
            )

    @property
    def name(self) -> str:
        """Get the orchestrator name identifier."""
        return "WikipediaPipeline_A2A"


class _FilteredDataProvider(AbstractObservationSimulator):
    """Simple filtered data provider for Wikipedia pipeline.

    This is a temporary implementation - in the future, this should be a proper DataProvider.
    """

    def __init__(self, mu, cov, prices, time, l_moments):
        """Initialize the filtered data provider.

        Args:
            mu: Expected returns as pandas Series.
            cov: Covariance matrix as pandas DataFrame.
            prices: Historical price data as pandas DataFrame.
            time: Timestamp for the data.
            l_moments: L-moments for higher-order statistics.
        """
        self._mu = mu
        self._cov = cov
        self._prices = prices
        self._time = time
        self._l_moments = l_moments

    @property
    def mu(self):
        """Get expected returns as numpy array."""
        return self._mu.values

    @property
    def cov(self):
        """Get covariance matrix as numpy array."""
        return self._cov.values

    @property
    def historical_prices(self):
        """Get historical price data as pandas DataFrame."""
        return self._prices

    @property
    def n_observations(self):
        """Get number of observations in the historical price data."""
        return len(self._prices)

    def get_sample(self):
        """Get sample market parameters (same as ground truth for filtered provider)."""
        return self.get_ground_truth()

    def get_ground_truth(self):
        """Get ground truth market parameters from filtered dataset."""
        return self._mu, self._cov, self._prices, self._time, self._l_moments

    @property
    def name(self) -> str:
        """Get the data provider name identifier."""
        return "FilteredDataProvider"
