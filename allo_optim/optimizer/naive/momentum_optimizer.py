import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import (
    LMoments,
)
from allo_optim.optimizer.asset_name_utils import (
    create_weights_series,
    get_asset_names,
    validate_asset_names,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class MomentumOptimizerConfig(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    min_positive_percentage: float = 0.05


class MomentumOptimizer(AbstractOptimizer):
    """Optimizer based on the naive momentum"""

    def __init__(self, config: Optional[MomentumOptimizerConfig] = None) -> None:
        self.config = config or MomentumOptimizerConfig()

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
        return "MomentumOptimizer"


class EMAMomentumOptimizer(MomentumOptimizer):
    """Optimizer based on the naive momentum with EMA"""

    def __init__(self, config: Optional[MomentumOptimizerConfig] = None) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "MomentumEMAOptimizer"
