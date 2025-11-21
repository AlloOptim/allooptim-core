import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from pydantic import BaseModel

class FastDiversificationConfig(BaseModel):
    """Configuration for Fast Portfolio Diversification Optimizer."""
    
    max_iter: int = 50
    """Maximum number of iterations for convergence."""
    
    tol: float = 1e-6
    """Tolerance level for convergence."""

class FastPortfolioDiversificationOptimizer(AbstractOptimizer):
    """Fast Portfolio Diversification optimizer implementation.
    
    This optimizer implements the fast diversification algorithm by Magnus Erik Hvass Pedersen (2022).
    The algorithm iteratively adjusts weights to minimize portfolio concentration while respecting covariance structure.
    """

    def __init__(
        self,
        config: Optional[FastDiversificationConfig] = None,     
        display_name: Optional[str] = None,
    ):
        """
        Initialize the optimizer.
        
        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
            display_name: Optional display name for this optimizer instance.
        """
        super().__init__(display_name)
        self.config = config or FastDiversificationConfig()

    @property
    def name(self) -> str:
        return "FastPortfolioDiversification"

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """
        Compute portfolio weights using the Fast Portfolio Diversification algorithm.
        
        Args:
            ds_mu: Expected returns vector (not directly used in this algorithm).
            df_cov: Covariance matrix of asset returns.
            df_prices: Optional historical prices (not used here).
            time: Optional timestamp, not used.
            l_moments: Optional advanced risk measures, not used here.
        
        Returns:
            Portfolio weights as pandas Series summing to 1.0.
        """
        n = len(ds_mu)

        # Initialize weights equally
        w = np.ones(n) / n

        # Convert covariance matrix to numpy array for speed
        cov = df_cov.values

        for iteration in range(self.config.max_iter):
            w_old = w.copy()

            # Contribution to portfolio variance by each asset: (cov @ w) element-wise divided by sqrt of quadratic form
            contrib = cov @ w

            # The key step: update weights inversely proportional to the contribution to risk (fast diversification idea)
            w = 1 / contrib

            # Normalize weights to sum to 1
            w = w / np.sum(w)

            # Check convergence by L1 norm (sum of absolute weight changes)
            if np.sum(np.abs(w - w_old)) < self.config.tol:
                break

        # Return weights as pandas Series indexed by asset names
        return pd.Series(w, index=ds_mu.index)
