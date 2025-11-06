"""HEAVYWEIGHT OPTIMIZER.

Training time: Minutes to hours
Min data: 500+ periods (recommended)
Suitable for: Research, long-term strategies, complex patterns
Architectures: LSTM+Transformer, MAMBA (SSM), TCN.
"""

import logging

from allooptim.optimizer.base_ml_optimizer import BaseMLOptimizer
from allooptim.optimizer.deep_learning.deep_learning_base import (
    DeepLearningOptimizerEngine,
    ModelType,
)

logger = logging.getLogger(__name__)


class LSTMOptimizer(BaseMLOptimizer):
    """Deep learning optimizer using LSTM + Transformer architecture."""

    model_type = ModelType.LSTM

    def _create_engine(self, n_assets: int, n_lookback: int) -> None:
        """Create the LSTM-based deep learning optimization engine."""
        engine = DeepLearningOptimizerEngine(n_assets=n_assets, n_lookback=n_lookback)
        engine.model_type = self.model_type
        return engine

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "LSTMOptimizer"


class MAMBAOptimizer(LSTMOptimizer):
    """Deep learning optimizer using MAMBA (Selective State Space Model) architecture."""

    model_type = ModelType.MAMBA

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "MAMBAOptimizer"


class TCNOptimizer(LSTMOptimizer):
    """Deep learning optimizer using TCN (Temporal Convolutional Network) architecture."""

    model_type = ModelType.TCN

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return "TCNOptimizer"
