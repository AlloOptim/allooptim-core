"""Portfolio Optimization Algorithms.

This module provides a comprehensive suite of 35+ portfolio optimization
algorithms spanning traditional mean-variance, risk parity, machine learning,
and ensemble methods.

Optimizer Categories:
    - **Mean-Variance**: Classical Markowitz optimization variants
    - **CMA-ES**: Evolutionary optimization with covariance adaptation
    - **Particle Swarm**: Population-based metaheuristic optimization
    - **HRP**: Hierarchical Risk Parity for diversification
    - **ML-based**: LightGBM, LSTM, MAMBA, TCN neural networks
    - **Fundamental**: Factor-based allocation using fundamentals
    - **Wikipedia**: Alternative data-driven allocation
    - **Ensemble**: Meta-optimizers combining multiple strategies

Quick Start:
    >>> from allooptim.optimizer import get_optimizer_by_names
    >>> optimizer = get_optimizer_by_names(["MeanVariance"])[0]
    >>> weights = optimizer.allocate(expected_returns, cov_matrix)

See Also:
    - :mod:`allooptim.optimizer.optimizer_list`: Available optimizers
    - :mod:`allooptim.optimizer.optimizer_interface`: Base interfaces
"""

from allooptim.optimizer.optimizer_factory import get_optimizer_by_names
from allooptim.optimizer.optimizer_list import get_all_optimizers

__all__ = ["get_optimizer_by_names", "get_all_optimizers"]
