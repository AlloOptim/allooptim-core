"""Black-Litterman portfolio optimization model.

This module implements the Black-Litterman model for portfolio optimization,
which combines market equilibrium returns with investor views to produce
posterior expected returns. This approach allows incorporating subjective
investment opinions into the optimization process.

Key features:
- Integration of investor views with market equilibrium
- Bayesian updating of expected returns
- Confidence intervals for investor views
- Risk-adjusted portfolio optimization
- PyPortfolioOpt integration
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import LMoments
from allooptim.optimizer.asset_name_utils import create_weights_series, validate_asset_names
from allooptim.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class BLOptimizerConfig(BaseModel):
    """Configuration parameters for Black-Litterman optimizer."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    view_dict: Optional[dict] = None
    use_implied_market: bool = False


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman portfolio optimizer combining prior beliefs with market views."""

    def __init__(
        self,
        config: Optional[BLOptimizerConfig] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """Initialize Black-Litterman optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or BLOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights using Black-Litterman model.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Historical price data (required if use_implied_market=True)
            time: Current timestamp (unused)
            l_moments: L-moments (unused)

        Returns:
            Optimal portfolio weights as pandas Series
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        if self.config.view_dict is None:
            logger.debug("No views provided, using HRP without Black-Litterman adjustment")
            view_dict = {name: 0.0 for name in asset_names}
        else:
            if len(self.config.view_dict) != len(asset_names):
                raise ValueError("View dictionary length must match number of assets")
            if not all(name in asset_names for name in self.config.view_dict):
                raise ValueError("All view keys must match asset names")

            view_dict = self.config.view_dict

        cov_matrix = df_cov  # Keep as DataFrame for pypfopt tickers

        bl = BlackLittermanModel(cov_matrix, pi=ds_mu, absolute_views=view_dict)

        if self.config.use_implied_market:
            if df_prices is None:
                raise ValueError("Price data must be fitted before allocation")

            delta = black_litterman.market_implied_risk_aversion(df_prices)
            bl.bl_weights(delta)
            weights_dict = bl.clean_weights()
        else:
            rets = bl.bl_returns()
            ef = EfficientFrontier(rets, df_cov)  # Use DataFrame for EfficientFrontier
            ef.min_volatility()
            weights_dict = ef.clean_weights()

        weights_array = np.array(list(weights_dict.values()))

        return create_weights_series(weights_array, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the Black-Litterman optimizer.

        Returns:
            Optimizer name string
        """
        return "BlackLittermanOptimizer"
