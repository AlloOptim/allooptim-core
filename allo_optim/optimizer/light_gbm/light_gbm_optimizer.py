"""
LIGHTWEIGHT OPTIMIZER
Training time: Seconds
Min data: 61 periods
Suitable for: Production, daily rebalancing, quick experiments
"""

import logging

from allo_optim.optimizer.base_ml_optimizer import BaseMLOptimizer, BaseMLOptimizerConfig
from allo_optim.optimizer.light_gbm.light_gbm_base import FastPortfolioOptimizer

logger = logging.getLogger(__name__)


class LightGBMOptimizer(BaseMLOptimizer):
    """Lightweight optimizer using LightGBM for portfolio optimization."""

    def _create_engine(self, n_assets: int):
        """Create the LightGBM-based optimization engine."""
        return FastPortfolioOptimizer(n_assets=n_assets)

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "LightGBMOptimizer"


class AugmentedLightGBMOptimizer(LightGBMOptimizer):
    """LightGBM optimizer with data augmentation enabled."""

    def __init__(self) -> None:
        super().__init__()
        # Replace config with augmented version

        self.config = BaseMLOptimizerConfig(use_data_augmentation=True)

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "AugmentedLightGBMOptimizer"
