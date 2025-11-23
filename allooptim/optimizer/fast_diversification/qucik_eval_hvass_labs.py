"""
Magnus Hvass Portfolio Optimization Algorithms Collection

This module implements the main portfolio optimization algorithms developed by
Magnus Erik Hvass Pedersen, as presented in his research papers:

1. Hvass Diversification - Fast Portfolio Diversification
2. Simple Portfolio Optimization - Filter + Diversify approach
3. Signal-Based Portfolio Optimization - Using predictive signals
4. Group Constraints Portfolio - Portfolio with asset group constraints

All optimizers inherit from the AbstractOptimizer base class and can be used
interchangeably in portfolio optimization workflows.

References:
- Pedersen, Magnus Erik Hvass (2021): "Simple Portfolio Optimization That Works!"
- Pedersen, Magnus Erik Hvass (2022): "Fast Portfolio Diversification"
- Pedersen, Magnus Erik Hvass (2022): "Portfolio Group Constraints"
- GitHub: https://github.com/Hvass-Labs/FinanceOps
"""

from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize as scipy_minimize


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_assets = 10
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Expected returns
    ds_mu = pd.Series(
        np.random.randn(n_assets) * 0.1 + 0.08,
        index=asset_names
    )
    
    # Covariance matrix
    volatilities = np.random.uniform(0.1, 0.3, n_assets)
    corr_matrix = np.random.uniform(-0.2, 0.6, (n_assets, n_assets))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Make positive semi-definite
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    D = np.diag(volatilities)
    cov_matrix = D @ corr_matrix @ D
    df_cov = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
    
    print("=" * 70)
    print("MAGNUS HVASS PORTFOLIO OPTIMIZATION ALGORITHMS")
    print("=" * 70)
    
    # Test all optimizers
    optimizers = [
        HvassDiversificationOptimizer(),
        SimplePortfolioOptimizer(return_threshold=0.05),
        SignalBasedOptimizer(signal_type='linear', apply_diversification=True),
        MinimumVarianceOptimizer(),
    ]
    
    for optimizer in optimizers:
        print(f"\n{optimizer.name} Optimizer:")
        print("-" * 70)
        weights = optimizer.allocate(ds_mu, df_cov)
        print(f"Weights:\n{weights.sort_values(ascending=False)}")
        print(f"Sum: {weights.sum():.6f}")
        print(f"Non-zero assets: {(weights > 1e-6).sum()}")
        
        # Calculate portfolio statistics
        portfolio_return = weights @ ds_mu
        portfolio_variance = weights @ df_cov @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        print(f"\nPortfolio Statistics:")
        print(f"  Expected Return: {portfolio_return:.4f}")
        print(f"  Volatility: {portfolio_volatility:.4f}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
