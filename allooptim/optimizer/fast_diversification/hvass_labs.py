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

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize as scipy_minimize


class AbstractOptimizer(ABC):
    """Abstract base class for all portfolio optimization algorithms."""

    def __init__(self, display_name: Optional[str] = None):
        self._display_name = display_name
        self.allow_cash = False
        self.max_leverage = None

    def fit(self, df_prices: Optional[pd.DataFrame] = None) -> None:
        """Optional setup method to prepare the optimizer with historical data."""
        pass

    def reset(self) -> None:
        """Optional method to reset any internal state of the optimizer."""
        self.__init__(self._display_name)

    def set_allow_cash(self, allow_cash: bool) -> None:
        """Set whether this optimizer is allowed to use cash."""
        self.allow_cash = allow_cash

    def set_max_leverage(self, max_leverage: Optional[float]) -> None:
        """Set the maximum leverage allowed for this optimizer."""
        self.max_leverage = max_leverage

    @abstractmethod
    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Create an optimal portfolio allocation."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this optimizer class."""
        pass

    @property
    def display_name(self) -> str:
        """Display name of this optimizer instance."""
        return self._display_name if self._display_name is not None else self.name


# ============================================================================
# 1. HVASS DIVERSIFICATION OPTIMIZER
# ============================================================================

class HvassDiversificationOptimizer(AbstractOptimizer):
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
        return "HvassDiversification"

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


# ============================================================================
# 2. SIMPLE PORTFOLIO OPTIMIZER (Filter + Diversify)
# ============================================================================

class SimplePortfolioOptimizer(AbstractOptimizer):
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


# ============================================================================
# 3. SIGNAL-BASED PORTFOLIO OPTIMIZER
# ============================================================================

class SignalBasedOptimizer(AbstractOptimizer):
    """
    Portfolio Optimization Using Signals.
    
    Uses predictive signals (e.g., P/Sales, P/E ratios, momentum) to determine
    portfolio weights. The signals are transformed into weights using a flexible
    mapping function.
    
    Parameters
    ----------
    signal_type : str, default='linear'
        Type of signal transformation: 'linear', 'sigmoid', 'exponential'
    scale_by_returns : bool, default=True
        Scale weights by expected returns
    apply_diversification : bool, default=True
        Apply Hvass diversification after signal-based weighting
    signal_power : float, default=1.0
        Power to raise signals to (higher = more aggressive)
    min_signal : float, default=0.0
        Minimum signal value to include asset
    """

    def __init__(
        self,
        signal_type: str = 'linear',
        scale_by_returns: bool = True,
        apply_diversification: bool = True,
        signal_power: float = 1.0,
        min_signal: float = 0.0,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.signal_type = signal_type
        self.scale_by_returns = scale_by_returns
        self.apply_diversification = apply_diversification
        self.signal_power = signal_power
        self.min_signal = min_signal
        
        if apply_diversification:
            self.diversifier = HvassDiversificationOptimizer()

    @property
    def name(self) -> str:
        return "SignalBased"

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Compute portfolio weights using predictive signals."""
        asset_names = ds_mu.index
        
        # Use expected returns as signals (in practice, you'd use external signals)
        signals = ds_mu.copy()
        
        # Filter by minimum signal
        mask = signals >= self.min_signal
        if not mask.any():
            return pd.Series(1.0 / len(signals), index=asset_names)
        
        signals = signals[mask]
        
        # Transform signals to weights
        weights = self._transform_signals(signals)
        
        # Optionally scale by expected returns
        if self.scale_by_returns:
            filtered_mu = ds_mu[mask]
            weights = weights * np.maximum(filtered_mu, 0)
        
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        # Apply diversification if requested
        if self.apply_diversification and len(weights) > 1:
            filtered_cov = df_cov.loc[mask, mask]
            weights = self.diversifier.allocate(weights, filtered_cov)
        
        # Map back to full universe
        full_weights = pd.Series(0.0, index=asset_names)
        full_weights[weights.index] = weights.values
        
        return full_weights

    def _transform_signals(self, signals: pd.Series) -> pd.Series:
        """Transform signals to weights using specified method."""
        if self.signal_type == 'linear':
            # Linear transformation with power
            weights = np.power(np.maximum(signals, 0), self.signal_power)
        elif self.signal_type == 'sigmoid':
            # Sigmoid transformation
            weights = 1.0 / (1.0 + np.exp(-self.signal_power * signals))
        elif self.signal_type == 'exponential':
            # Exponential transformation
            weights = np.exp(self.signal_power * signals)
        else:
            weights = signals.copy()
        
        return weights


# ============================================================================
# 4. GROUP CONSTRAINTS OPTIMIZER
# ============================================================================

class GroupConstraintsOptimizer(AbstractOptimizer):
    """
    Portfolio Optimization with Group Constraints.
    
    Enforces constraints on groups of assets (e.g., sector limits, geography limits).
    Uses optimization to find weights that satisfy group constraints while
    maximizing diversification.
    
    Parameters
    ----------
    group_constraints : Dict[str, Tuple[float, float]]
        Dictionary mapping group names to (min_weight, max_weight) tuples
    asset_to_group : Dict[str, str]
        Dictionary mapping asset names to group names
    optimize_within_groups : bool, default=True
        Whether to optimize diversification within each group
    """

    def __init__(
        self,
        group_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        asset_to_group: Optional[Dict[str, str]] = None,
        optimize_within_groups: bool = True,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.group_constraints = group_constraints or {}
        self.asset_to_group = asset_to_group or {}
        self.optimize_within_groups = optimize_within_groups
        
        if optimize_within_groups:
            self.diversifier = HvassDiversificationOptimizer()

    @property
    def name(self) -> str:
        return "GroupConstraints"

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Compute portfolio weights with group constraints."""
        asset_names = ds_mu.index
        n_assets = len(ds_mu)
        
        # If no constraints, just diversify normally
        if not self.group_constraints or not self.asset_to_group:
            return self.diversifier.allocate(ds_mu, df_cov)
        
        # Group assets
        groups = {}
        for asset in asset_names:
            group = self.asset_to_group.get(asset, 'default')
            if group not in groups:
                groups[group] = []
            groups[group].append(asset)
        
        # Allocate weights respecting group constraints
        weights = pd.Series(0.0, index=asset_names)
        
        for group_name, group_assets in groups.items():
            min_weight, max_weight = self.group_constraints.get(
                group_name, (0.0, 1.0)
            )
            
            # Get expected returns and covariance for this group
            group_mu = ds_mu[group_assets]
            group_cov = df_cov.loc[group_assets, group_assets]
            
            # Optimize within group
            if self.optimize_within_groups and len(group_assets) > 1:
                group_weights = self.diversifier.allocate(group_mu, group_cov)
            else:
                # Equal weight within group
                group_weights = pd.Series(
                    1.0 / len(group_assets), index=group_assets
                )
            
            # Scale to group constraint
            target_group_weight = np.clip(
                1.0 / len(groups), min_weight, max_weight
            )
            weights[group_assets] = group_weights * target_group_weight
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights


# ============================================================================
# 5. MINIMUM VARIANCE OPTIMIZER (for comparison)
# ============================================================================

class MinimumVarianceOptimizer(AbstractOptimizer):
    """
    Classic Minimum Variance Portfolio.
    
    Finds the portfolio with minimum variance subject to constraints.
    Included for comparison with Hvass methods.
    
    Parameters
    ----------
    allow_short : bool, default=False
        Allow short positions (negative weights)
    """

    def __init__(
        self,
        allow_short: bool = False,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.allow_short = allow_short

    @property
    def name(self) -> str:
        return "MinimumVariance"

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[object] = None,
    ) -> pd.Series:
        """Compute minimum variance portfolio weights."""
        n_assets = len(ds_mu)
        asset_names = ds_mu.index
        cov_matrix = df_cov.values
        
        # Objective: minimize portfolio variance
        def objective(w):
            return w @ cov_matrix @ w
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Bounds
        if self.allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = scipy_minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = result.x
        return pd.Series(weights, index=asset_names)


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