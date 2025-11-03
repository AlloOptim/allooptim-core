"""
HEAVYWEIGHT OPTIMIZER
Training time: Minutes to hours
Min data: 500+ periods (recommended)
Suitable for: Research, long-term strategies, complex patterns
Architectures: LSTM+Transformer, MAMBA (SSM), TCN
"""

import logging

from allo_optim.optimizer.base_ml_optimizer import BaseMLOptimizer
from allo_optim.optimizer.deep_learning.deep_learning_base import DeepLearningOptimizer

logger = logging.getLogger(__name__)


class LSTMOptimizer(BaseMLOptimizer):
	"""Deep learning optimizer using LSTM + Transformer architecture."""

	model_type = "lstm"

	def _create_engine(self, n_assets: int):
		"""Create the LSTM-based deep learning optimization engine."""
		engine = DeepLearningOptimizer(n_assets=n_assets)
		engine.model_type = self.model_type
		return engine

	@property
	def name(self) -> str:
		"""Return the name of this optimizer."""
		return "LSTMOptimizer"


class MAMBAOptimizer(LSTMOptimizer):
	"""Deep learning optimizer using MAMBA (Selective State Space Model) architecture."""

	model_type = "mamba"

	@property
	def name(self) -> str:
		"""Return the name of this optimizer."""
		return "MAMBAOptimizer"


class TCNOptimizer(LSTMOptimizer):
	"""Deep learning optimizer using TCN (Temporal Convolutional Network) architecture."""

	model_type = "tcn"

	@property
	def name(self) -> str:
		"""Return the name of this optimizer."""
		return "TCNOptimizer"
