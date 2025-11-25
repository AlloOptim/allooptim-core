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
    ) -> None:
        """Initialize balanced fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access. If None, creates from factory.
        """
        super().__init__(display_name)
        self.config = config or BalancedFundamentalConfig()

        self.data_provider = data_provider or FundamentalDataProviderFactory.create_provider()

        self._weights_today: Optional[np.ndarray] = None
        self._last_calculation_time: Optional[datetime] = None

        self.is_fundamental_optimizer = True

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
            assert self._weights_today is not None, "Cached weights should not be None"  # nosec B101 - Assert for internal consistency checks
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
    ) -> None:
        """Initialize quality growth fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access.
        """
        super().__init__(config, display_name, data_provider)
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
    ) -> None:
        """Initialize value investing fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access.
        """
        super().__init__(config, display_name, data_provider)
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
    ) -> None:
        """Initialize market cap fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
            data_provider: Fundamental data provider for data access.
        """
        super().__init__(config, display_name, data_provider)
        self.config = config or OnlyMarketCapFundamentalConfig()

    @property
    def name(self) -> str:
        """Get the name of the market cap fundamental optimizer.

        Returns:
            Optimizer name string
        """
        return "MarketCapFundamentalOptimizer"
