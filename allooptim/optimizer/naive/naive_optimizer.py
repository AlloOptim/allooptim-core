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


class NaiveOptimizerConfig(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    # No parameters needed for naive equal weight, but keeping for consistency
    pass


class NaiveOptimizer(AbstractOptimizer):
    """
    Equal-weight portfolio optimizer with pandas interface.

    Implements the simplest possible allocation strategy by assigning equal weights
    to all available assets (1/N strategy). Despite its simplicity, this approach
    often serves as a strong benchmark against more sophisticated optimization methods.

    Key Features:
        - Assigns equal weights to all assets: weight_i = 1/N for all i
        - No optimization required - deterministic allocation
        - Robust to estimation errors in expected returns and covariance
        - Fast execution with O(1) complexity
        - Asset names preserved from input to output

    Examples:
        Basic usage with asset name preservation:

        >>> optimizer = NaiveOptimizer()
        >>> weights = optimizer.allocate(mu, cov)
        >>> print(f"Equal weight per asset: {1/len(weights):.4f}")
        >>> for asset, weight in weights.items():
        ...     print(f"{asset}: {weight:.4f}")
        AAPL: 0.2500
        GOOGL: 0.2500
        MSFT: 0.2500
        TSLA: 0.2500

        Access asset information:

        >>> asset_names = weights.index.tolist()
        >>> n_assets = len(weights)
        >>> print(f"Portfolio contains {n_assets} assets: {asset_names}")
    """

    def __init__(self, config: Optional[NaiveOptimizerConfig] = None) -> None:
        self.config = config or NaiveOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        # Validate inputs
        validate_asset_names(ds_mu, df_cov)

        # Get asset names
        asset_names = get_asset_names(mu=ds_mu)
        n_assets = len(asset_names)

        # Create equal weights
        equal_weight = 1.0 / n_assets
        weights = np.ones(n_assets) * equal_weight

        # Return as pandas Series with asset names
        return create_weights_series(weights, asset_names)

    @property
    def name(self) -> str:
        return "NaiveOptimizer"
