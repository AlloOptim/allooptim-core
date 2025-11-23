"""Fundamental analysis-based portfolio allocation optimizer.

This module provides portfolio optimization strategies based on fundamental
company data and financial metrics. It implements various fundamental investing
approaches including value investing, quality growth, and market cap weighting.

Key features:
- Value investing strategies
- Quality and growth factor investing
- Market capitalization weighting
- Fundamental data integration
- Long-term investment focus
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from allooptim.data.fundamental_providers import FundamentalDataManager
from allooptim.data.provider_factory import FundamentalDataProviderFactory, UnifiedFundamentalProvider
from allooptim.optimizer.allocation_metric import (
    LMoments,
)
from allooptim.optimizer.asset_name_utils import (
    create_weights_series,
    validate_asset_names,
)
from allooptim.optimizer.base_optimizer import BaseOptimizer
from allooptim.optimizer.fundamental.fundamental_methods import (
    BalancedFundamentalConfig,
    OnlyMarketCapFundamentalConfig,
    QualityGrowthFundamentalConfig,
    ValueInvestingFundamentalConfig,
    allocate,
)

logger = logging.getLogger(__name__)


class BalancedFundamentalOptimizer(BaseOptimizer):
    """Balanced fundamental optimizer using fundamental analysis for portfolio allocation."""

    def __init__(
        self,
        config: Optional[BalancedFundamentalConfig] = None,
        display_name: Optional[str] = None,
        data_provider: Optional[UnifiedFundamentalProvider] = None,
        data_manager: Optional[FundamentalDataManager] = None,  # DEPRECATED
    ) -> None:
        """Initialize balanced fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access. If None, creates from factory.
            data_manager: DEPRECATED - Use data_provider instead. Fundamental data manager for data access.
        """
        super().__init__(display_name)
        self.config = config or BalancedFundamentalConfig()
        
        # Handle data provider selection with backward compatibility
        if data_manager is not None:
            import warnings
            warnings.warn(
                "data_manager parameter is deprecated, use data_provider instead",
                DeprecationWarning,
                stacklevel=2
            )
            # Adapt old manager to new interface (temporary compatibility)
            self.data_provider = _adapt_manager_to_provider(data_manager)
        elif data_provider is not None:
            self.data_provider = data_provider
        else:
            # Default: create from factory
            self.data_provider = FundamentalDataProviderFactory.create_provider()

        self._weights_today: Optional[np.ndarray] = None
        self._last_calculation_time: Optional[datetime] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights using fundamental analysis.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Historical price data (unused)
            time: Current timestamp for weight estimation
            l_moments: L-moments (unused)

        Returns:
            Portfolio weights as pandas Series
        """
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        # For backtesting, always recalculate weights for each date
        # The data_manager handles caching of fundamental data appropriately
        logger.info(f"FundamentalOptimizer.allocate called for {self.name} at time {time}")
        if self._weights_today is None or time != self._last_calculation_time:
            estimate_new = True
            logger.info(f"  -> Estimating NEW weights (time changed from {self._last_calculation_time} to {time})")
        else:
            estimate_new = False
            logger.info(f"  -> Using CACHED weights (time {time} same as last {self._last_calculation_time})")

        if estimate_new:
            try:
                logger.info("Estimating new weights using FundamentalOptimizer.")
                # Handle case where time might be None
                today = time or datetime.now()
                weights = allocate(
                    asset_names=asset_names,
                    today=today,
                    config=self.config,
                    data_provider=self.data_provider,
                )

                self._weights_today = weights
                self._last_calculation_time = time

            except Exception as error:
                logger.error(f"Exception in FundamentalOptimizer.allocate: {error}")
                n_assets = len(asset_names)
                weights = np.ones(n_assets) / n_assets

        else:
            # If we get here, _weights_today should not be None
            assert self._weights_today is not None, "Cached weights should not be None"
            weights = self._weights_today

        return create_weights_series(weights, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the balanced fundamental optimizer.

        Returns:
            Optimizer name string
        """
        return "BalancedFundamentalOptimizer"


class QualityGrowthFundamentalOptimizer(BalancedFundamentalOptimizer):
    """Quality and growth focused fundamental optimizer."""

    def __init__(
        self,
        config: Optional[QualityGrowthFundamentalConfig] = None,
        display_name: Optional[str] = None,
        data_provider: Optional[UnifiedFundamentalProvider] = None,
        data_manager: Optional[FundamentalDataManager] = None,  # DEPRECATED
    ) -> None:
        """Initialize quality growth fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access.
            data_manager: DEPRECATED - Use data_provider instead.
        """
        super().__init__(config, display_name, data_provider, data_manager)
        self.config = config or QualityGrowthFundamentalConfig()

    @property
    def name(self) -> str:
        """Get the name of the quality growth fundamental optimizer.

        Returns:
            Optimizer name string
        """
        return "QualityGrowthFundamentalOptimizer"


class ValueInvestingFundamentalOptimizer(BalancedFundamentalOptimizer):
    """Value investing focused fundamental optimizer."""

    def __init__(
        self,
        config: Optional[ValueInvestingFundamentalConfig] = None,
        display_name: Optional[str] = None,
        data_provider: Optional[UnifiedFundamentalProvider] = None,
        data_manager: Optional[FundamentalDataManager] = None,  # DEPRECATED
    ) -> None:
        """Initialize value investing fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access.
            data_manager: DEPRECATED - Use data_provider instead.
        """
        super().__init__(config, display_name, data_provider, data_manager)
        self.config = config or ValueInvestingFundamentalConfig()

    @property
    def name(self) -> str:
        """Get the name of the value investing fundamental optimizer.

        Returns:
            Optimizer name string
        """
        return "ValueInvestingFundamentalOptimizer"


class MarketCapFundamentalOptimizer(BalancedFundamentalOptimizer):
    """Market capitalization based fundamental optimizer."""

    def __init__(
        self,
        config: Optional[OnlyMarketCapFundamentalConfig] = None,
        display_name: Optional[str] = None,
        data_provider: Optional[UnifiedFundamentalProvider] = None,
        data_manager: Optional[FundamentalDataManager] = None,  # DEPRECATED
    ) -> None:
        """Initialize market cap fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access.
            data_manager: DEPRECATED - Use data_provider instead.
        """
        super().__init__(config, display_name, data_provider, data_manager)
        self.config = config or OnlyMarketCapFundamentalConfig()

    @property
    def name(self) -> str:
        """Get the name of the market cap fundamental optimizer.

        Returns:
            Optimizer name string
        """
        return "MarketCapFundamentalOptimizer"


def _adapt_manager_to_provider(data_manager: FundamentalDataManager) -> UnifiedFundamentalProvider:
    """Temporary adapter to convert old FundamentalDataManager to new UnifiedFundamentalProvider.
    
    This is for backward compatibility during migration.
    """
    from allooptim.data.provider_factory import UnifiedFundamentalProvider
    from allooptim.data.fundamental_providers import FundamentalDataProvider
    
    # Create a simple adapter that wraps the manager
    class ManagerAdapter(FundamentalDataProvider):
        def __init__(self, manager):
            self.manager = manager
            
        def get_fundamental_data(self, tickers, date=None):
            return self.manager.get_fundamental_data(tickers, date)
            
        def supports_historical_data(self):
            # Check if manager has provider (new implementation) or mode (old)
            if hasattr(self.manager, 'provider'):
                return self.manager.provider.supports_historical_data()
            elif hasattr(self.manager, 'mode'):
                return self.manager.mode == "backtest"
            else:
                # Default to False for safety
                return False
    
    return UnifiedFundamentalProvider([ManagerAdapter(data_manager)])
