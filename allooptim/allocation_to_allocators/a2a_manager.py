"""A2A Manager
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from allooptim.allocation_to_allocators.a2a_result import A2AResult
from allooptim.allocation_to_allocators.orchestrator_factory import (
    create_orchestrator,
)
from allooptim.allocation_to_allocators.rebalancer import PortfolioRebalancer
from allooptim.backtest.data_loader import DataLoader
from allooptim.config.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.a2a_manager_config import A2AManagerConfig
from allooptim.covariance_transformer.transformer_list import get_transformer_by_names
from allooptim.data.price_data_provider import PriceDataProvider
from allooptim.data.provider_factory import FundamentalDataProviderFactory
from allooptim.optimizer.wikipedia.wiki_database import download_data

logger = logging.getLogger(__name__)


class A2AManager:
    """Main A2A Manager."""

    def __init__(
        self,
        a2a_manager_config: Optional[A2AManagerConfig] = None,
        **orchestrator_kwargs,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            a2a_manager_config: Configuration for backtest parameters including date ranges,
                optimizers, and result storage settings.
            **orchestrator_kwargs: Additional keyword arguments passed to orchestrator creation.
        """
        self.a2a_manager_config = a2a_manager_config or A2AManagerConfig()
        self.data_loader = DataLoader(
            benchmark=self.a2a_manager_config.benchmark,
            symbols=self.a2a_manager_config.symbols,
            interval=self.a2a_manager_config.data_interval,
        )

        # Create shared fundamental data provider with caching for backtests
        self.fundamental_provider = FundamentalDataProviderFactory.create_provider(enable_caching=True)

        self.orchestrator = create_orchestrator(
            orchestrator_type=self.a2a_manager_config.orchestration_type,
            optimizer_configs=self.a2a_manager_config.optimizer_configs,
            transformer_names=self.a2a_manager_config.transformer_names,
            a2a_config=self.a2a_manager_config.a2a_config,
            **orchestrator_kwargs,
        )

        for optimizer in self.orchestrator.optimizers:
            if optimizer.is_fundamental_optimizer:
                optimizer.data_provider = self.fundamental_provider

        self.transformers = get_transformer_by_names(self.a2a_manager_config.transformer_names)
        self.rebalancer = PortfolioRebalancer()

    def load_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logger.debug("Loading data")

        has_wikipedia_optimizer = any(optimizer.is_wiki_optimizer for optimizer in self.orchestrator.optimizers)

        if has_wikipedia_optimizer:
            logger.info("Wikipedia optimizer detected, downloading Wikipedia data...")
            try:
                download_data(start_date, end_date, self.data_loader.stock_universe)
                logger.info("Wikipedia data download completed")
            except Exception as e:
                logger.warning(f"Failed to download Wikipedia data: {e}")

        has_fundamental_optimizer = any(
            optimizer.is_fundamental_optimizer for optimizer in self.orchestrator.optimizers
        )

        if has_fundamental_optimizer:
            logger.info("Fundamental optimizers detected, preloading fundamental data...")
            try:
                self.fundamental_provider.preload_data(
                    tickers=self.a2a_manager_config.symbols,
                    start_date=start_date,
                    end_date=end_date,
                )
                logger.info("Fundamental data preload completed")
            except Exception as e:
                logger.warning(f"Failed to preload fundamental data: {e}")

        # Load price data
        clean_data = self.data_loader.load_price_data(start_date, end_date)

        return clean_data

    def run_allocation(self, time_today: datetime, clean_data: pd.DataFrame) -> A2AResult:
        """Run the comprehensive backtest.

        Returns:
            Dictionary containing all results and metrics
        """
        logger.debug("Starting allocation")

        data_provider = PriceDataProvider(clean_data)

        allocation_result = self.orchestrator.allocate(
            data_provider=data_provider,
            time_today=time_today,
            all_stocks=self.data_loader.stock_universe,
        )

        allocation_result.final_allocation = self.rebalancer.rebalance(allocation_result.final_allocation)

        return allocation_result
