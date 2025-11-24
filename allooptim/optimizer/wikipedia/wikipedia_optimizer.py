"""Wikipedia-based portfolio optimization strategies.

This module implements portfolio optimization strategies inspired by or derived from
Wikipedia articles on portfolio theory and asset allocation. The optimizer provides
implementations of various allocation methods described in financial literature and
online resources, adapted for practical portfolio management.

Key features:
- Wikipedia-inspired allocation methods
- Historical portfolio theory implementations
- Educational allocation strategies
- Reference implementations for comparison
- Integration with stock universe data
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.config.stock_universe import get_stocks_by_symbols
from allooptim.optimizer.allocation_metric import (
    LMoments,
)
from allooptim.optimizer.asset_name_utils import (
    create_weights_series,
    get_asset_names,
    validate_asset_names,
)
from allooptim.optimizer.base_optimizer import BaseOptimizer
from allooptim.optimizer.wikipedia.allocate_wikipedia import allocate_wikipedia

logger = logging.getLogger(__name__)


class WikipediaOptimizerConfig(BaseModel):
    """Configuration for Wikipedia-based optimizer.

    This config holds parameters for the Wikipedia optimizer. Currently minimal
    as the Wikipedia strategy doesn't require extensive configuration, but structured
    for future extensibility.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    # Wikipedia optimizer doesn't need specific parameters currently


class WikipediaOptimizer(BaseOptimizer):
    """Wikipedia-based portfolio optimizer.

    This optimizer uses Wikipedia page view data and other web-based signals
    to determine portfolio allocations. It leverages the allocate_wikipedia
    function to compute weights based on online attention metrics.
    """

    def __init__(self, config: Optional[WikipediaOptimizerConfig] = None, display_name: Optional[str] = None) -> None:
        """Initialize the Wikipedia optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or WikipediaOptimizerConfig()
        
        self.is_wiki_optimizer = True

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio using Wikipedia-based signals.

        Uses Wikipedia page view data and web-based metrics to determine
        portfolio weights. Falls back to equal weights if Wikipedia data
        is unavailable or allocation fails.

        Args:
            ds_mu: Expected returns series with asset names as index (unused)
            df_cov: Covariance matrix DataFrame (unused)
            df_prices: Historical price data (unused)
            time: Current timestamp (required for Wikipedia data lookup)
            l_moments: L-moments (unused)

        Returns:
            Portfolio weights as pandas Series based on Wikipedia signals
        """
        # Validate inputs
        validate_asset_names(ds_mu, df_cov)
        if time is None:
            raise ValueError("Time parameter must be provided")

        # Ensure time is timezone-aware (required by allocate_wikipedia)
        time = time.replace(tzinfo=pytz.UTC) if time.tzinfo is None else time.astimezone(pytz.UTC)

        # Get asset names
        asset_names = get_asset_names(mu=ds_mu)
        n_assets = len(asset_names)

        try:
            all_stocks = get_stocks_by_symbols(asset_names)

            # Filter asset_names to only include stocks available in the universe
            available_symbols = {stock.symbol for stock in all_stocks}
            filtered_asset_names = [name for name in asset_names if name in available_symbols]

            if not filtered_asset_names:
                logger.warning("No assets available in stock universe for Wikipedia allocation")
                equal_weight = 1.0 / n_assets
                weights = np.ones(n_assets) * equal_weight
            else:
                allocation_result = allocate_wikipedia(
                    all_stocks=all_stocks,
                    time_today=time,
                    use_wiki_database=True,
                )

                if allocation_result.success:
                    weights_dict = allocation_result.asset_weights
                    # Create weights array for filtered assets
                    filtered_weights = np.array([weights_dict[key] for key in filtered_asset_names])

                    # Create full weights array with zeros for unavailable assets
                    weights = np.zeros(n_assets)
                    for i, asset_name in enumerate(asset_names):
                        if asset_name in filtered_asset_names:
                            idx = filtered_asset_names.index(asset_name)
                            weights[i] = filtered_weights[idx]
                        # Unavailable assets get 0 weight
                else:
                    equal_weight = 1.0 / n_assets
                    weights = np.ones(n_assets) * equal_weight

        except Exception as e:
            logger.error(f"Error in Wikipedia allocation: {e}")
            equal_weight = 1.0 / n_assets
            weights = np.ones(n_assets) * equal_weight

        # Return as pandas Series with asset names
        return create_weights_series(weights, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the Wikipedia optimizer.

        Returns:
            Optimizer name string
        """
        return "WikipediaOptimizer"
