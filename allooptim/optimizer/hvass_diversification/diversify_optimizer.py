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
from allooptim.optimizer.optimizer_interface import AbstractOptimizer


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
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        equal_risk_contribution: bool = True,
        adjust_for_volatility: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.equal_risk_contribution = equal_risk_contribution
        self.adjust_for_volatility = adjust_for_volatility
        self.min_weight = min_weight
        self.max_weight = max_weight

        # State variables
        self.iterations_used_ = 0
        self.converged_ = False

    @property
    def name(self) -> str:
        return "HvassDiversification"

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
        if self.adjust_for_volatility:
            # Start with inverse volatility weighting
            weights = 1.0 / (volatilities + 1e-8)
        else:
            # Start with equal weights
            weights = np.ones(n_assets)

        # Normalize initial weights
        weights = weights / np.sum(weights)

        # Iterative diversification algorithm
        for iteration in range(self.max_iterations):
            weights_old = weights.copy()

            # Compute full exposure for each asset
            full_exposure = weights.copy()

            # Compute correlated exposure for each asset
            correlated_exposure = self._compute_correlated_exposure(
                weights, corr_matrix
            )

            # Adjust weights to reduce correlated exposure
            if self.equal_risk_contribution:
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
            weights = np.clip(weights, self.min_weight, self.max_weight)

            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)

            # Check convergence
            weight_change = np.max(np.abs(weights - weights_old))
            if weight_change < self.tolerance:
                self.converged_ = True
                self.iterations_used_ = iteration + 1
                break
        else:
            self.converged_ = False
            self.iterations_used_ = self.max_iterations

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
        
        if not self.use_matrix:
            n_assets = len(weights)
            correlated_exposure = np.zeros(n_assets)

            for i in range(n_assets):
                # Sum of weighted correlations with other assets
                exposure = 0.0
                for j in range(n_assets):
                    if i != j:
                        # Correlation weighted by the other asset's weight
                        exposure += abs(corr_matrix[i, j]) * weights[j]

                correlated_exposure[i] = exposure

        else:
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

    def get_diversification_metrics(
        self, weights: pd.Series, df_cov: pd.DataFrame
    ) -> dict:
        """
        Compute diversification metrics for a given portfolio.
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        df_cov : pd.DataFrame
            Covariance matrix
            
        Returns
        -------
        dict
            Dictionary containing diversification metrics
        """
        w = weights.values
        cov = df_cov.values

        # Portfolio variance
        portfolio_variance = w @ cov @ w
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Individual asset volatilities
        volatilities = np.sqrt(np.diag(cov))

        # Diversification ratio (weighted sum of volatilities / portfolio volatility)
        diversification_ratio = (w @ volatilities) / portfolio_volatility

        # Effective number of assets (inverse Herfindahl index)
        effective_n_assets = 1.0 / np.sum(w ** 2)

        # Correlation matrix
        corr = self._cov_to_corr(cov)

        # Average pairwise correlation (weighted)
        avg_correlation = 0.0
        total_weight_pairs = 0.0
        n = len(w)
        for i in range(n):
            for j in range(i + 1, n):
                weight_pair = w[i] * w[j]
                avg_correlation += corr[i, j] * weight_pair
                total_weight_pairs += weight_pair

        if total_weight_pairs > 0:
            avg_correlation /= total_weight_pairs

        return {
            "portfolio_volatility": portfolio_volatility,
            "diversification_ratio": diversification_ratio,
            "effective_n_assets": effective_n_assets,
            "avg_correlation": avg_correlation,
            "iterations_used": self.iterations_used_,
            "converged": self.converged_,
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_assets = 10
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
    # Initialize optimizer
    optimizer = DiversificationOptimizer(
        max_iterations=100,
        tolerance=1e-6,
        equal_risk_contribution=True,
        adjust_for_volatility=True,
    )
    
    for _ in range(1000):
        # Expected returns (not used in pure diversification, but required by interface)
        ds_mu = pd.Series(
            np.random.randn(n_assets) * 0.1 + 0.05,
            index=asset_names
        )
        
        # Generate a realistic covariance matrix
        volatilities = np.random.uniform(0.1, 0.3, n_assets)
        corr_matrix = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Create covariance matrix
        D = np.diag(volatilities)
        cov_matrix = D @ corr_matrix @ D
        df_cov = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)

        optimizer.use_matrix = True
        a1 = optimizer._compute_correlated_exposure(ds_mu.values, df_cov.values)
        optimizer.use_matrix = False
        a2 = optimizer._compute_correlated_exposure(ds_mu.values, df_cov.values)
    
        assert np.allclose(a1, a2)
    
    # # Compute optimal weights
    # weights = optimizer.allocate(ds_mu, df_cov)
    
    # print("Hvass Diversification Optimizer Results")
    # print("=" * 50)
    # print("\nPortfolio Weights:")
    # print(weights.sort_values(ascending=False))
    # print(f"\nSum of weights: {weights.sum():.4f}")
    
    # # Get diversification metrics
    # metrics = optimizer.get_diversification_metrics(weights, df_cov)
    # print("\nDiversification Metrics:")
    # for key, value in metrics.items():
    #     print(f"  {key}: {value}")
