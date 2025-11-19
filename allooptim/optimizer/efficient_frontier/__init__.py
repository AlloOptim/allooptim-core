"""Efficient frontier and modern portfolio theory optimizers.

This package implements classical and modern portfolio optimization
algorithms based on mean-variance theory and efficient frontier concepts.

Key optimizers:
- EfficientFrontierOptimizer: Classical Markowitz mean-variance optimization
- BlackLittermanOptimizer: Bayesian views integration with market equilibrium
- Risk-return optimization with various constraints
- Integration with PyPortfolioOpt library
- Support for custom risk measures and constraints
"""

import logging

logger = logging.getLogger(__name__)
