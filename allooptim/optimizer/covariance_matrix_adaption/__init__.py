"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization.

This package implements CMA-ES, a evolutionary optimization algorithm
that adapts the covariance matrix of a multivariate normal distribution
to efficiently search for optimal portfolio weights.

Key features:
- CMA-ES optimization for portfolio weight selection
- Adaptive covariance matrix learning
- Population-based search with elitism
- Robust convergence properties
- Integration with cma library
"""

import logging

logger = logging.getLogger(__name__)
