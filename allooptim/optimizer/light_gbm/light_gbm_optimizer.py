"""LIGHTWEIGHT OPTIMIZER
Training time: Seconds
Min data: 61 periods
Suitable for: Production, daily rebalancing, quick experiments.
"""

import logging
from typing import Optional

from allooptim.optimizer.base_ml_optimizer import BaseMLOptimizer, BaseMLOptimizerConfig
from allooptim.optimizer.light_gbm.light_gbm_base import LightGBMOptimizerEngine

logger = logging.getLogger(__name__)


class LightGBMOptimizer(BaseMLOptimizer):
    """Lightweight optimizer using LightGBM for portfolio optimization."""

    def __init__(self, config: Optional[BaseMLOptimizerConfig] = None) -> None:
        """Initialize the LightGBM optimizer.

        Args:
            config: Configuration for the optimizer. If None, uses default config.
        """
        super().__init__(config)
        self.config.use_data_augmentation = False

    def _create_engine(self, n_assets: int, n_lookback: int) -> None:
        """Create the LightGBM-based optimization engine."""
        return LightGBMOptimizerEngine(n_assets=n_assets, n_lookback=n_lookback)

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "LightGBMOptimizer"


class AugmentedLightGBMOptimizer(LightGBMOptimizer):
    """LightGBM optimizer with data augmentation enabled."""

    def __init__(self, config: Optional[BaseMLOptimizerConfig] = None) -> None:
        """Initialize the augmented LightGBM optimizer.

        Args:
            config: Configuration for the optimizer. If None, uses default config.
        """
        super().__init__(config)
        self.config.use_data_augmentation = True

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "AugmentedLightGBMOptimizer"
