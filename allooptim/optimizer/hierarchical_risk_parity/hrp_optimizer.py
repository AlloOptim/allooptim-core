import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pypfopt.hierarchical_portfolio import HRPOpt

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import LMoments
from allooptim.optimizer.asset_name_utils import create_weights_series, validate_asset_names
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)

# Constants for portfolio weight validation
PORTFOLIO_WEIGHT_SUM_UPPER_TOLERANCE = 1.001
PORTFOLIO_WEIGHT_SUM_LOWER_TOLERANCE = 0.999


class HRPOptimizerConfig(BaseModel):
    """Configuration for Hierarchical Risk Parity optimizer.

    This config class holds parameters for the HRP optimizer. Currently minimal
    as HRP typically doesn't require extensive configuration, but structured
    for future extensibility.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    # HRP typically doesn't need many parameters, but adding for consistency
    # Could add linkage method, distance metric parameters here if needed
    pass


class HRPOptimizer(AbstractOptimizer):
    """Hierarchical Risk Parity optimizer for portfolio allocation.

    This optimizer uses the Hierarchical Risk Parity (HRP) approach, which
    constructs portfolios by considering the hierarchical structure of asset
    correlations. It aims to provide better diversification by allocating
    weights based on risk clustering rather than traditional mean-variance
    optimization.
    """

    def __init__(self, config: Optional[HRPOptimizerConfig] = None) -> None:
        """Initialize the Hierarchical Risk Parity optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
        self.config = config or HRPOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights using Hierarchical Risk Parity optimization.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Historical price data (unused by HRP)
            time: Current timestamp (unused by HRP)
            l_moments: L-moments (unused by HRP)

        Returns:
            Portfolio weights as pandas Series with asset names as index
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        hrp = HRPOpt(cov_matrix=df_cov)

        weights_dict = hrp.optimize()
        weights_array = np.array([weights_dict[key] for key in asset_names])

        if (
            weights_array.sum() > PORTFOLIO_WEIGHT_SUM_UPPER_TOLERANCE
            or weights_array.sum() < PORTFOLIO_WEIGHT_SUM_LOWER_TOLERANCE
        ):
            logger.error("Portfolio allocations don't sum to 1.")
            return create_weights_series(np.zeros(len(asset_names)), asset_names)

        return create_weights_series(weights_array, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the Hierarchical Risk Parity optimizer.

        Returns:
            Optimizer name string
        """
        return "HRPOptimizer"
