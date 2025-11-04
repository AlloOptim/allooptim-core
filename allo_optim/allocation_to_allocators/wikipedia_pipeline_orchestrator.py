"""
Wikipedia Pipeline A2A Orchestrator

Orchestrator that combines Wikipedia-based stock pre-selection with portfolio optimization.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from allo_optim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allo_optim.allocation_to_allocators.equal_weight_orchestrator import (
    EqualWeightOrchestrator,
)
from allo_optim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allo_optim.config.allocation_dataclasses import AllocationResult, NoStatistics
from allo_optim.config.stock_dataclasses import StockUniverse
from allo_optim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.wikipedia.allocate_wikipedia import allocate_wikipedia

logger = logging.getLogger(__name__)


class WikipediaPipelineOrchestrator(BaseOrchestrator):
    """
    Wikipedia Pipeline Allocation-to-Allocators orchestrator.

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
        n_historical_days: int = 60,
        use_wiki_database: bool = False,
        wiki_database_path: Optional[Path] = None,
    ):
        super().__init__(optimizers, covariance_transformers)
        self.n_historical_days = n_historical_days
        self.use_wiki_database = use_wiki_database
        self.wiki_database_path = wiki_database_path

    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> AllocationResult:
        """
        Run Wikipedia pipeline allocation orchestration.

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

        start_time = datetime.now()

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
                end_time = datetime.now()
                return AllocationResult(
                    asset_weights=wikipedia_result.asset_weights,
                    success=False,
                    statistics=wikipedia_result.statistics or NoStatistics(),
                    computation_time=(end_time - start_time).total_seconds(),
                    error_message="Wikipedia allocation failed",
                )

            # Step 2: Extract pre-selected stocks (those with non-zero weights)
            preselected_stocks = [symbol for symbol, weight in wikipedia_result.asset_weights.items() if weight > 0.0]

            logger.info(f"Pre-selected {len(preselected_stocks)} / {len(all_stocks)} stocks from Wikipedia allocation")

            if not preselected_stocks:
                # Return failed result if no stocks were pre-selected
                end_time = datetime.now()
                return AllocationResult(
                    asset_weights={},
                    success=False,
                    statistics=NoStatistics(),
                    computation_time=(end_time - start_time).total_seconds(),
                    error_message="No stocks pre-selected from Wikipedia",
                )

            # Step 3: Filter data provider to only include pre-selected stocks
            # Get ground truth data
            mu_full, cov_full, prices_full, time_full, l_moments_full = data_provider.get_ground_truth()

            # Filter to pre-selected stocks
            available_columns = [col for col in preselected_stocks if col in prices_full.columns]
            if not available_columns:
                # Return failed result if none of the pre-selected stocks are in price data
                end_time = datetime.now()
                return AllocationResult(
                    asset_weights={},
                    success=False,
                    statistics=NoStatistics(),
                    computation_time=(end_time - start_time).total_seconds(),
                    error_message="No pre-selected stocks available in price data",
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
            equal_orchestrator = EqualWeightOrchestrator(self.optimizers, self.covariance_transformers)
            optimization_result = equal_orchestrator.allocate(filtered_data_provider, time_today)

            # Step 5: Pad weights with zeros for non-selected assets
            # Create full weight dict with all original assets
            all_assets = prices_full.columns.tolist()
            asset_weights_padded = {asset: 0.0 for asset in all_assets}

            # Update with actual weights from optimization
            if optimization_result.asset_weights is not None:
                asset_weights_padded.update(optimization_result.asset_weights)

            # Pad df_allocation with zeros for non-selected assets
            df_allocation_padded = None
            if optimization_result.df_allocation is not None:
                df_allocation_padded = optimization_result.df_allocation.copy()
                # Add missing assets with zero weights
                for asset in all_assets:
                    if asset not in df_allocation_padded.columns:
                        df_allocation_padded[asset] = 0.0
                # Reorder to match original asset order
                df_allocation_padded = df_allocation_padded[all_assets]

            end_time = datetime.now()
            return AllocationResult(
                asset_weights=asset_weights_padded,
                success=optimization_result.success,
                statistics=wikipedia_result.statistics,  # Use Wikipedia statistics from pre-selection
                computation_time=(end_time - start_time).total_seconds(),
                df_allocation=df_allocation_padded,
            )

        except Exception as e:
            end_time = datetime.now()
            logger.error(f"Wikipedia pipeline allocation failed: {str(e)}")
            return AllocationResult(
                asset_weights={},
                success=False,
                statistics=NoStatistics(),
                computation_time=(end_time - start_time).total_seconds(),
                error_message=str(e),
            )

    @property
    def name(self) -> str:
        return "WikipediaPipeline_A2A"


class _FilteredDataProvider(AbstractObservationSimulator):
    """
    Simple filtered data provider for Wikipedia pipeline.
    This is a temporary implementation - in the future, this should be a proper DataProvider.
    """

    def __init__(self, mu, cov, prices, time, l_moments):
        self._mu = mu
        self._cov = cov
        self._prices = prices
        self._time = time
        self._l_moments = l_moments

    @property
    def mu(self):
        return self._mu.values

    @property
    def cov(self):
        return self._cov.values

    @property
    def historical_prices(self):
        return self._prices

    @property
    def n_observations(self):
        return len(self._prices)

    def get_sample(self):
        # For filtered provider, sample and ground truth are the same
        return self.get_ground_truth()

    def get_ground_truth(self):
        return self._mu, self._cov, self._prices, self._time, self._l_moments

    @property
    def name(self) -> str:
        return "FilteredDataProvider"
