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

class FilterAndDiversifyOptimizer(AbstractOptimizer):
    """
    Simple Portfolio Optimization That Works!
    
    Two-step approach:
    1. Filter: Keep only assets with expected return above threshold
    2. Diversify: Apply Hvass Diversification to filtered assets
    
    This method is extremely robust and outperforms traditional mean-variance
    optimization in real-world applications.
    
    Parameters
    ----------
    return_threshold : float, default=0.0
        Minimum expected return for inclusion
    use_percentile_filter : bool, default=False
        Use percentile-based filtering instead of absolute threshold
    percentile : float, default=0.5
        If use_percentile_filter=True, keep top X percentile
    max_iterations : int, default=100
        Max iterations for diversification step
    adjust_for_volatility : bool, default=True
        Adjust for volatility in diversification
    """

    def __init__(
        self,
        return_threshold: float = 0.0,
        use_percentile_filter: bool = False,
        percentile: float = 0.5,
        max_iterations: int = 100,
        adjust_for_volatility: bool = True,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.return_threshold = return_threshold
        self.use_percentile_filter = use_percentile_filter
        self.percentile = percentile
        self.max_iterations = max_iterations
        self.adjust_for_volatility = adjust_for_volatility
        
        # Internal diversifier
        self.diversifier = HvassDiversificationOptimizer(
            max_iterations=max_iterations,
            adjust_for_volatility=adjust_for_volatility,
        )

    @property
    def name(self) -> str:
        return "SimplePortfolio"

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Compute portfolio weights using filter + diversify approach."""
        n_assets = len(ds_mu)
        asset_names = ds_mu.index
        
        # Step 1: Filter assets based on expected returns
        if self.use_percentile_filter:
            threshold = ds_mu.quantile(self.percentile)
            mask = ds_mu >= threshold
        else:
            mask = ds_mu >= self.return_threshold
        
        if not mask.any():
            # No assets pass filter, return equal weight
            return pd.Series(1.0 / n_assets, index=asset_names)
        
        # Get filtered assets
        filtered_mu = ds_mu[mask]
        filtered_cov = df_cov.loc[mask, mask]
        
        # Step 2: Diversify the filtered portfolio
        if len(filtered_mu) == 1:
            # Only one asset, give it all weight
            weights = pd.Series(0.0, index=asset_names)
            weights[filtered_mu.index[0]] = 1.0
        else:
            # Apply diversification
            filtered_weights = self.diversifier.allocate(filtered_mu, filtered_cov)
            
            # Map back to full asset universe
            weights = pd.Series(0.0, index=asset_names)
            weights[filtered_weights.index] = filtered_weights.values
        
        return weights
