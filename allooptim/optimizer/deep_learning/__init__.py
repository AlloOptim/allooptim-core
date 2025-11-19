"""Deep learning-based portfolio optimization algorithms.

This package contains neural network-based portfolio optimizers using
various architectures including LSTM, Mamba, and Temporal Convolutional
Networks (TCN). These models learn patterns in historical price data
to make allocation decisions.

Key components:
- DeepLearningOptimizerEngine: Base class for neural network optimizers
- LSTM, Mamba, TCN architectures for time series prediction
- Online learning capabilities for adapting to new data
- Integration with TinyGrad for efficient computation
- Portfolio weight prediction from historical returns
"""

import logging

logger = logging.getLogger(__name__)
