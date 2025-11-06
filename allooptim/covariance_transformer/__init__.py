"""Covariance Matrix Transformations.

This module provides statistical techniques to improve covariance matrix estimates
for better-conditioned portfolio optimization. Transformations include shrinkage,
denoising, regularization, and dimensionality reduction methods.

Transformer Categories:
    - **Shrinkage**: Reduce estimation error by blending sample covariance with targets
    - **Denoising**: Remove noise from covariance matrices using random matrix theory
    - **Regularization**: Add stability through ridge regression or similar techniques
    - **Autoencoder**: Neural network-based covariance matrix compression

Quick Start:
    >>> from allooptim.covariance_transformer import SimpleShrinkageCovarianceTransformer
    >>> transformer = SimpleShrinkageCovarianceTransformer(shrinkage=0.2)
    >>> clean_cov = transformer.transform(sample_cov, n_observations=252)

See Also:
    - :mod:`allooptim.covariance_transformer.transformer_list`: Available transformers
    - :mod:`allooptim.covariance_transformer.transformer_interface`: Base interfaces
"""

from allooptim.covariance_transformer.transformer_list import get_all_transformers

__all__ = ["get_all_transformers"]
