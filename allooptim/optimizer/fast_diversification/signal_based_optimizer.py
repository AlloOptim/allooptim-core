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

