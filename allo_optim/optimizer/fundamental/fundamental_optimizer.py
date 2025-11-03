import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from allo_optim.optimizer.allocation_metric import (
    LMoments,
)
from allo_optim.optimizer.asset_name_utils import (
    create_weights_series,
    validate_asset_names,
)
from allo_optim.optimizer.fundamental.fundamental_methods import (
    BalancedFundamentalConfig,
    OnlyMarketCapFundamentalConfig,
    QualityGrowthFundamentalConfig,
    ValueInvestingFundamentalConfig,
    allocate,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class BalancedFundamentalOptimizer(AbstractOptimizer):
    def __init__(self) -> None:
        self.config = BalancedFundamentalConfig()

        self._weights_today: Optional[np.ndarray] = None

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        df_allocations: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
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
        return "BalancedFundamentalOptimizer"


class QualityGrowthFundamentalOptimizer(BalancedFundamentalOptimizer):
    def __init__(self) -> None:
        self.config = QualityGrowthFundamentalConfig()

    @property
    def name(self) -> str:
        return "QualityGrowthFundamentalOptimizer"


class ValueInvestingFundamentalOptimizer(BalancedFundamentalOptimizer):
    def __init__(self) -> None:
        self.config = ValueInvestingFundamentalConfig()

    @property
    def name(self) -> str:
        return "ValueInvestingFundamentalOptimizer"


class MarketCapFundamentalOptimizer(BalancedFundamentalOptimizer):
    def __init__(self) -> None:
        self.config = OnlyMarketCapFundamentalConfig()

    @property
    def name(self) -> str:
        return "MarketCapFundamentalOptimizer"
