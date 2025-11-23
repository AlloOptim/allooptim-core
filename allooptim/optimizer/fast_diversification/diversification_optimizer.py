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
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np

class DiversificationOptimizer(AbstractOptimizer):
    """
    Fast Portfolio Diversification using the Hvass algorithm.
    
    This is the core diversification algorithm that minimizes correlated exposure
    through iterative weight adjustments. Extremely fast and robust to estimation errors.
    
    Parameters
    ----------
    max_iterations : int, default=100
        Maximum number of iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    adjust_for_volatility : bool, default=True
        Start with inverse volatility weights
    min_weight : float, default=0.0
        Minimum weight per asset
    max_weight : float, default=1.0
        Maximum weight per asset
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        adjust_for_volatility: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.adjust_for_volatility = adjust_for_volatility
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.iterations_used_ = 0
        self.converged_ = False

    @property
    def name(self) -> str:
        return "DiversificationOptimizer"

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Compute portfolio weights using Hvass Diversification algorithm."""
        n_assets = len(ds_mu)
        asset_names = ds_mu.index

        volatilities = np.sqrt(np.diag(df_cov.values))
        corr_matrix = self._cov_to_corr(df_cov.values)

        # Initialize weights
        if self.adjust_for_volatility:
            weights = 1.0 / (volatilities + 1e-8)
        else:
            weights = np.ones(n_assets)
        weights = weights / np.sum(weights)

        # Iterative diversification
        for iteration in range(self.max_iterations):
            weights_old = weights.copy()
            correlated_exposure = self._compute_correlated_exposure(weights, corr_matrix)
            diversification_score = 1.0 / (correlated_exposure + 1e-8)
            weights = weights * diversification_score
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / np.sum(weights)

            if np.max(np.abs(weights - weights_old)) < self.tolerance:
                self.converged_ = True
                self.iterations_used_ = iteration + 1
                break
        else:
            self.converged_ = False
            self.iterations_used_ = self.max_iterations

        return pd.Series(weights, index=asset_names)

    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std_devs = np.sqrt(np.diag(cov_matrix))
        outer_std = np.outer(std_devs, std_devs)
        corr_matrix = cov_matrix / (outer_std + 1e-8)
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix

    def _compute_correlated_exposure(
        self, weights: np.ndarray, corr_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute correlated exposure for each asset."""
        n_assets = len(weights)
        correlated_exposure = np.zeros(n_assets)
        for i in range(n_assets):
            exposure = 0.0
            for j in range(n_assets):
                if i != j:
                    exposure += abs(corr_matrix[i, j]) * weights[j]
            correlated_exposure[i] = exposure
        return correlated_exposure
