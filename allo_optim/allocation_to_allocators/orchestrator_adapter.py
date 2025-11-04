"""
Adapter to wrap AllocationOrchestrator as an AbstractOptimizer for backtest integration.

This adapter allows AllocationOrchestrator to be used in backtests alongside
regular optimizers, enabling clean integration without modifying existing code.
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from allo_optim.allocation_to_allocators.allocation_orchestrator import (
    AllocationOrchestrator,
    AllocationOrchestratorConfig,
    OrchestrationType,
)
from allo_optim.config.stock_universe import get_stocks_by_symbols
from allo_optim.optimizer.allocation_metric import LMoments
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class AllocationOrchestratorAdapter(AbstractOptimizer):
    """
    Adapter that wraps AllocationOrchestrator to match AbstractOptimizer interface.
    
    This enables AllocationOrchestrator to be used in backtests and standalone runs
    alongside individual optimizers.
    
    Args:
        optimizer_names: List of optimizer names to use
        transformer_names: List of transformer names to apply
        orchestration_type: Type of orchestration (equal, optimized, wikipedia_pipeline)
        config: Optional AllocationOrchestratorConfig for fine-tuning
    """

    def __init__(
        self,
        optimizer_names: List[str],
        transformer_names: List[str],
        orchestration_type: OrchestrationType = OrchestrationType.EQUAL,
        config: Optional[AllocationOrchestratorConfig] = None,
    ) -> None:
        if config is None:
            config = AllocationOrchestratorConfig(orchestration_type=orchestration_type)
        else:
            config.orchestration_type = orchestration_type

        self._optimizer_names = optimizer_names
        self._transformer_names = transformer_names
        self._orchestration_type = orchestration_type

        self.orchestrator = AllocationOrchestrator(
            optimizer_names=optimizer_names,
            transformer_names=transformer_names,
            config=config,
        )

        self._df_prices_cache = None  # Cache for fit() data

    @property
    def name(self) -> str:
        """Return name indicating orchestration type."""
        return f"A2A_{self._orchestration_type.value.title()}"

    def fit(self, df_prices: pd.DataFrame) -> None:
        """
        Cache price data for later use in allocate().
        
        AllocationOrchestrator doesn't have a separate fit step,
        so we just cache the data here.
        """
        
        self.orchestrator.fit_optimizers(df_prices=df_prices)
        
        self._df_prices_cache = df_prices
        logger.debug(f"{self.name}: Cached {len(df_prices)} price observations")

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """
        Run allocation using AllocationOrchestrator and return weights as Series.
        
        Args:
            ds_mu: Expected returns (not used, orchestrator computes internally)
            df_cov: Covariance matrix (not used, orchestrator computes internally)
            df_prices: Price data (used if provided, otherwise uses cached from fit())
            time: Current time for allocation
            l_moments: L-moments (not used by orchestrator)
            df_allocations: Previous allocations (not used by orchestrator)
            
        Returns:
            pd.Series with asset weights
        """
        # Use provided prices or fall back to cached
        if df_prices is None:
            if self._df_prices_cache is None:
                raise ValueError(f"{self.name}: No price data available. Call fit() first or provide df_prices")
            df_prices = self._df_prices_cache

        if time is None:
            time = pd.Timestamp.now()

        # Get stock universe (for wikipedia pipeline)
        asset_names = df_prices.columns.tolist()
        all_stocks = get_stocks_by_symbols(asset_names)

        # Run orchestrator
        result = self.orchestrator.run_allocation(
            all_stocks=all_stocks,
            time_today=time,
            df_prices=df_prices,
        )

        if not result.success:
            logger.error(f"{self.name}: Allocation failed: {result.error_message}")
            # Return equal weights as fallback
            return pd.Series(1.0 / len(df_prices.columns), index=df_prices.columns)

        # Convert dict to Series
        weights = pd.Series(result.asset_weights)

        # Ensure weights match the asset order from df_prices
        weights = weights.reindex(df_prices.columns, fill_value=0.0)

        return weights

    def reset(self) -> None:
        """Reset the orchestrator state."""
        self._df_prices_cache = None
        self.orchestrator = AllocationOrchestrator(
            optimizer_names=self._optimizer_names,
            transformer_names=self._transformer_names,
            config=self.orchestrator.config,
        )

        logger.debug(f"{self.name}: Reset state")
