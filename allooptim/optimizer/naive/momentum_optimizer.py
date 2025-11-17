"""Momentum-based portfolio allocation strategies.

This module implements momentum-based portfolio allocation strategies that
allocate capital to assets showing positive price momentum. These strategies
capitalize on the tendency of assets with recent positive performance to
continue performing well.

Key features:
- Momentum-based asset selection and weighting
- Lookback period configuration for momentum calculation
- Risk-adjusted momentum strategies
- Trend-following allocation approaches
- Integration with price data for momentum signals
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import (
    LMoments,
)
from allooptim.optimizer.asset_name_utils import (
    create_weights_series,
    get_asset_names,
    validate_asset_names,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class MomentumOptimizerConfig(BaseModel):
    """Configuration for momentum-based optimizer.

    This config holds parameters for momentum optimization including
    the minimum percentage of assets that must have positive momentum
    to trigger investment.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    min_positive_percentage: float = 0.05


class MomentumOptimizer(AbstractOptimizer):
    """Optimizer based on the naive momentum."""

    def __init__(self, config: Optional[MomentumOptimizerConfig] = None, display_name: Optional[str] = None) -> None:
        """Initialize the momentum optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or MomentumOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio using momentum strategy.

        Invests only in assets with positive expected returns, normalized to sum to 1.
        Requires minimum percentage of assets to have positive momentum to invest.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame (unused)
            df_prices: Historical price data (unused)
            time: Current timestamp (unused)
            l_moments: L-moments (unused)

        Returns:
            Portfolio weights as pandas Series, zero weights if insufficient positive momentum
        """
        # Validate inputs
        validate_asset_names(ds_mu, df_cov)
        asset_names = get_asset_names(mu=ds_mu)

        # Handle NaN or infinite values in mu
        if ds_mu.isna().any() or np.isinf(ds_mu.values).any():
            return create_weights_series(np.zeros(len(asset_names)), asset_names)

        # Create momentum weights (only positive expected returns)
        mu_weight = ds_mu.copy()
        mu_weight[mu_weight < 0] = 0.0

        # If only a small percentage of assets have positive momentum, do not invest
        positive_assets = (mu_weight > 0).sum()
        min_required = self.config.min_positive_percentage * len(asset_names)

        if positive_assets < min_required:
            return create_weights_series(np.zeros(len(asset_names)), asset_names)

        # Normalize positive weights
        if mu_weight.sum() > 0:
            mu_weight = mu_weight / mu_weight.sum()
        else:
            logger.error("Momentum weights sum to zero after filtering, returning zero weights.")
            mu_weight = pd.Series(0.0, index=asset_names)

        return mu_weight

    @property
    def name(self) -> str:
        """Get the name of the momentum optimizer.

        Returns:
            Optimizer name string
        """
        return "MomentumOptimizer"


class EMAMomentumOptimizer(MomentumOptimizer):
    """Optimizer based on the naive momentum with EMA."""

    def __init__(self, config: Optional[MomentumOptimizerConfig] = None, display_name: Optional[str] = None) -> None:
        """Initialize the EMA momentum optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(config, display_name)

    @property
    def name(self) -> str:
        """Get the name of the EMA momentum optimizer.

        Returns:
            Optimizer name string
        """
        return "MomentumEMAOptimizer"
        