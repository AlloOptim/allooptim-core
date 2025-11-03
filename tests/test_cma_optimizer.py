#!/usr/bin/env python3
"""
Test script for the new CMA-ES optimizer integration
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

from allo_optim.optimizer.covariance_matrix_adaption.cma_optimizer import MeanVarianceCMAOptimizer


def create_test_data():
    """Create sample test data for optimization"""
    # Create sample expected returns
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    mu = pd.Series([0.12, 0.15, 0.10, 0.14, 0.18], index=assets)

    # Create sample covariance matrix
    np.random.seed(42)  # For reproducible results
    cov_data = np.random.randn(5, 5) * 0.1
    cov_data = cov_data @ cov_data.T  # Make positive semi-definite
    cov = pd.DataFrame(cov_data, index=assets, columns=assets)

    return mu, cov


def test_cma_optimizer():
    """Test the CMA-ES optimizer"""
    print("Testing CMA-ES Mean Variance Optimizer...")

    mu, cov = create_test_data()

    # Test CMA-ES optimizer
    try:
        cma_optimizer = MeanVarianceCMAOptimizer()
        cma_optimizer.risk_aversion = 4.0  # Set risk aversion after initialization
        print(f"Created optimizer: {cma_optimizer.name}")

        # Create sample price data for fitting (required for some risk metrics)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        price_data = pd.DataFrame(np.random.randn(100, 5).cumsum(axis=0) + 100, index=dates, columns=mu.index)
        cma_optimizer.fit(price_data)

        # Run optimization
        weights = cma_optimizer.allocate(mu, cov, price_data)

        print("CMA-ES Optimization Results:")
        print(f"Portfolio weights:")
        for asset, weight in weights.items():
            print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")

        print(f"Total weight: {weights.sum():.6f}")

        # Calculate portfolio metrics
        portfolio_return = (weights * mu).sum()
        portfolio_risk = np.sqrt((weights * (cov @ weights)).sum())
        sharpe_ratio = portfolio_return / portfolio_risk

        print(f"Expected return: {portfolio_return:.4f}")
        print(f"Portfolio risk: {portfolio_risk:.4f}")
        print(f"Sharpe ratio: {sharpe_ratio:.4f}")

        # Validate results
        assert isinstance(weights, pd.Series), "Weights should be a pandas Series"
        assert len(weights) == len(mu), "Weight dimensions should match input"
        assert abs(weights.sum() - 1.0) < 0.01, f"Weights should sum to ~1.0, got {weights.sum()}"
        assert (weights >= 0).all(), "All weights should be non-negative"

        print("✓ All assertions passed!")

    except Exception as e:
        print(f"CMA-ES optimizer failed: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise the exception for pytest to catch


if __name__ == "__main__":
    print("Testing new CMA-ES optimizer integration...")
    test_cma_optimizer()
    print("CMA-ES optimizer test passed successfully!")
