"""
Hvass Fast Portfolio Diversification Optimizer

Implementation of Magnus Hvass Pedersen's Fast Portfolio Diversification algorithm.
Based on the research paper "Fast Portfolio Diversification" (2022).

The algorithm minimizes correlated exposure in the portfolio through iterative
weight adjustments, converging to optimal diversification in just a few iterations.

References:
- Pedersen, Magnus Erik Hvass (2022): "Fast Portfolio Diversification"
- Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4009041
"""

from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.optimizer_interface import AbstractOptimizer


class DiversificationOptimizerConfig(BaseModel):
    """Configuration for Hvass Diversification optimizer.

    This config holds parameters for the fast portfolio diversification algorithm
    including convergence settings and risk contribution preferences.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    max_iterations: int = 100
    tolerance: float = 1e-6
    equal_risk_contribution: bool = True
    adjust_for_volatility: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0


class DiversificationOptimizer(AbstractOptimizer):
    """
    Fast Portfolio Diversification using the Hvass algorithm.
    
    This optimizer minimizes the correlated exposure in a portfolio through
    an iterative algorithm that converges quickly to optimal diversification.
    
    The algorithm is extremely fast (typically 6-7 iterations) and very robust
    to estimation errors in the correlation matrix.
    
    Parameters
    ----------
    max_iterations : int, default=100
        Maximum number of iterations for the algorithm
    tolerance : float, default=1e-6
        Convergence tolerance for weight changes
    equal_risk_contribution : bool, default=True
        If True, aims for equal risk contribution across assets
    adjust_for_volatility : bool, default=True
        If True, adjusts weights by inverse volatility before diversification
    min_weight : float, default=0.0
        Minimum weight for any asset
    max_weight : float, default=1.0
        Maximum weight for any asset
        
    Attributes
    ----------
    iterations_used_ : int
        Number of iterations used in the last optimization
    converged_ : bool
        Whether the algorithm converged in the last optimization
    """

    def __init__(
        self,
        config: Optional[DiversificationOptimizerConfig] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.config = config or DiversificationOptimizerConfig()

        # State variables
        self._iterations_used = 0
        self._converged = False

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """
        Compute portfolio weights using Hvass Diversification algorithm.
        
        Parameters
        ----------
        ds_mu : pd.Series
            Expected returns (not used in pure diversification)
        df_cov : pd.DataFrame
            Covariance matrix
        df_prices : pd.DataFrame, optional
            Historical prices (not used)
        time : datetime, optional
            Current time (not used)
        l_moments : object, optional
            L-moments (not used)
            
        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        n_assets = len(ds_mu)
        asset_names = ds_mu.index

        # Extract correlation matrix and volatilities from covariance matrix
        volatilities = np.sqrt(np.diag(df_cov.values))
        corr_matrix = self._cov_to_corr(df_cov.values)

        # Initialize weights
        if self.config.adjust_for_volatility:
            # Start with inverse volatility weighting
            weights = 1.0 / (volatilities + 1e-8)
        else:
            # Start with equal weights
            weights = np.ones(n_assets)

        # Normalize initial weights
        weights = weights / np.sum(weights)

        # Iterative diversification algorithm
        for iteration in range(self.config.max_iterations):
            weights_old = weights.copy()

            # Compute full exposure for each asset
            full_exposure = weights.copy()

            # Compute correlated exposure for each asset
            correlated_exposure = self._compute_correlated_exposure(
                weights, corr_matrix
            )

            # Adjust weights to reduce correlated exposure
            if self.config.equal_risk_contribution:
                # Equal Risk Contribution variant
                weights = self._adjust_weights_erc(
                    weights, full_exposure, correlated_exposure, corr_matrix
                )
            else:
                # Standard diversification
                weights = self._adjust_weights_standard(
                    weights, full_exposure, correlated_exposure
                )

            # Apply constraints
            weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)

            # Check convergence
            weight_change = np.max(np.abs(weights - weights_old))
            if weight_change < self.config.tolerance:
                self._converged = True
                self._iterations_used = iteration + 1
                break
        else:
            self._converged = False
            self._iterations_used = self.config.max_iterations

        # Create result Series with asset names
        result = pd.Series(weights, index=asset_names)

        # Handle cash constraint if needed
        if self.allow_cash:
            # Portfolio can hold less than 100%
            pass  # Weights already normalized to 1.0, but could be adjusted

        # Handle leverage constraint if needed
        if self.max_leverage is not None:
            total_weight = result.sum()
            if total_weight > self.max_leverage:
                result = result * (self.max_leverage / total_weight)

        return result

    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std_devs = np.sqrt(np.diag(cov_matrix))
        outer_std = np.outer(std_devs, std_devs)
        corr_matrix = cov_matrix / (outer_std + 1e-8)
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix

    def _compute_correlated_exposure(
        self, weights: np.ndarray, corr_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute the correlated exposure for each asset.
        
        The correlated exposure measures how much an asset's risk overlaps
        with other assets in the portfolio due to correlations.
        """

        # Matrix formulation: abs(corr_matrix) @ weights - weights
        # This computes sum_{j≠i} |corr[i,j]| * weights[j] for each i
        correlated_exposure = np.abs(corr_matrix) @ weights - weights

        return correlated_exposure

    def _adjust_weights_standard(
        self,
        weights: np.ndarray,
        full_exposure: np.ndarray,
        correlated_exposure: np.ndarray,
    ) -> np.ndarray:
        """
        Standard weight adjustment to minimize correlated exposure.
        
        Assets with higher correlated exposure get lower weights.
        """
        # Compute diversification score (inverse of correlated exposure)
        diversification_score = 1.0 / (correlated_exposure + 1e-8)

        # Adjust weights proportional to diversification score
        adjusted_weights = weights * diversification_score

        return adjusted_weights

    def _adjust_weights_erc(
        self,
        weights: np.ndarray,
        full_exposure: np.ndarray,
        correlated_exposure: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Equal Risk Contribution variant of weight adjustment.
        
        Aims to equalize the marginal risk contribution of each asset.
        """
        n_assets = len(weights)

        # Compute marginal risk contribution for each asset
        portfolio_variance = weights @ corr_matrix @ weights
        marginal_risk = corr_matrix @ weights / (np.sqrt(portfolio_variance) + 1e-8)

        # Asset risk contribution
        risk_contribution = weights * marginal_risk

        # Target equal risk contribution
        target_risk_contrib = np.mean(risk_contribution)

        # Adjust weights inversely proportional to current risk contribution
        adjustment_factor = target_risk_contrib / (risk_contribution + 1e-8)
        adjusted_weights = weights * adjustment_factor

        return adjusted_weights

    @property
    def name(self) -> str:
        return "DiversificationOptimizer"

    @property
    def iterations_used_(self) -> int:
        """Number of iterations used in the last optimization."""
        return self._iterations_used

    @property
    def converged_(self) -> bool:
        """Whether the algorithm converged in the last optimization."""
        return self._converged
