import logging

import numpy as np
import pandas as pd

from allo_optim.optimizer.allocation_metric import (
    make_positive_definite,
)

logger = logging.getLogger(__name__)


def calculate_robust_ema_moments(
    returns: pd.DataFrame,
    span: int,
    n_min_observations: int = 5,
    max_reasonable_return: float = 0.5,  # annually
    min_reasonable_return: float = -0.5,  # annually
    blend_factor: float = 0.3,  # Blend 30% simple with 70% EMA
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate robust exponentially weighted moments from returns data.

    This implementation is more stable than PyPortfolioOpt's version and includes:
    - Proper exponential weighting of returns
    - Multiple time horizon averaging for robustness
    - Bounds checking and regularization
    - Handling of edge cases

    Args:
        returns: DataFrame of asset returns (n_observations, n_assets)
        span: EMA span parameter (higher = more smoothing)

    Returns:
        Tuple of (expected_returns, covariance_matrix)
    """
    n_obs, n_assets = returns.shape

    # Ensure we have enough data
    if n_obs < n_min_observations:
        logger.warning(f"Insufficient data for EMA ({n_obs} observations), using simple moments")
        return returns.mean().values, returns.cov().values

    # Calculate exponential weights (more recent observations have higher weight)
    # Use pandas ewm with com=span-1 for consistency with PyPortfolioOpt
    alpha = 2.0 / (span + 1.0)  # Convert span to alpha

    # Calculate EMA of returns (this gives the expected return)
    ema_returns = returns.ewm(alpha=alpha, adjust=False).mean()

    # Use the most recent EMA values as expected returns
    expected_returns = ema_returns.iloc[-1].values

    # For covariance, we need exponentially weighted covariance matrix
    # This is more complex - we need to weight the outer products

    # Calculate exponentially weighted covariance
    # Method 1: Use pandas ewm on the covariance matrix (simpler but less accurate)
    # Method 2: Calculate weighted covariance directly (more accurate but complex)

    # Use Method 1 for simplicity and robustness
    cov_matrix = returns.ewm(alpha=alpha, adjust=False).cov()

    # Extract the most recent covariance matrix
    cov_values = cov_matrix.iloc[-n_assets:].values  # Last n_assets rows contain the full matrix

    # Ensure covariance matrix is properly formed
    if cov_values.shape != (n_assets, n_assets):
        # Fallback: reshape if needed
        cov_values = cov_values.reshape((n_assets, n_assets))

    # Apply bounds checking on expected returns
    # Clip extreme values that might indicate numerical issues
    expected_returns = np.clip(expected_returns, min_reasonable_return, max_reasonable_return)

    # Additional robustness: if returns are still extreme, use a blended approach
    simple_returns = returns.mean().values

    # Blend EMA with simple historical returns for stability
    expected_returns = (1 - blend_factor) * expected_returns + blend_factor * simple_returns

    # For covariance, also blend with simple covariance
    simple_cov = returns.cov().values
    cov_values = (1 - blend_factor) * cov_values + blend_factor * simple_cov
    cov_values = make_positive_definite(cov_values)

    return expected_returns, cov_values
