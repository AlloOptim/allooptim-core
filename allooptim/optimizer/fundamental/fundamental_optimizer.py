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
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

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

    def __init__(self, config: Optional[BalancedFundamentalConfig] = None, display_name: Optional[str] = None) -> None:
        """Initialize balanced fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or BalancedFundamentalConfig()

        self._weights_today: Optional[np.ndarray] = None

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

        if self._weights_today is None:
            estimate_new = True

        elif time - datetime.now() < timedelta(days=1):
            logger.debug("Data fetching is only supported for the current day.")
            estimate_new = False

        else:
            estimate_new = True

        if estimate_new:
            try:
                logger.info("Estimating new weights using FundamentalOptimizer.")
                weights = allocate(
                    asset_names=asset_names,
                    today=time,
                    config=self.config,
                )

                self._weights_today = weights

            except Exception as error:
                logger.error(f"Exception in FundamentalOptimizer.allocate: {error}")
                n_assets = len(asset_names)
                weights = np.ones(n_assets) / n_assets

        else:
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
        self, config: Optional[QualityGrowthFundamentalConfig] = None, display_name: Optional[str] = None
    ) -> None:
        """Initialize quality growth fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(config, display_name)
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
        self, config: Optional[ValueInvestingFundamentalConfig] = None, display_name: Optional[str] = None
    ) -> None:
        """Initialize value investing fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(config, display_name)
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
        self, config: Optional[OnlyMarketCapFundamentalConfig] = None, display_name: Optional[str] = None
    ) -> None:
        """Initialize market cap fundamental optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(config, display_name)
        self.config = config or OnlyMarketCapFundamentalConfig()

    @property
    def name(self) -> str:
        """Get the name of the market cap fundamental optimizer.

        Returns:
            Optimizer name string
        """
        return "MarketCapFundamentalOptimizer"
