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
from pydantic import BaseModel, field_validator
from enum import Enum

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.hvass_diversification.diversify_optimizer import (
    DiversificationOptimizer,
)


class SignalType(str, Enum):
    """Enumeration of supported signal transformation types."""
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"


class SignalBasedOptimizerConfig(BaseModel):
    """Configuration for Signal-Based optimizer.

    This config holds parameters for signal-based portfolio optimization
    including signal transformation and diversification settings.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    signal_type: SignalType = SignalType.LINEAR
    scale_by_returns: bool = True
    apply_diversification: bool = True
    signal_power: float = 1.0
    min_signal: float = 0.0

    @field_validator("signal_type", mode="before")
    @classmethod
    def validate_signal_type(cls, v: str) -> SignalType:
        """Validate that signal type is one of the allowed values."""
        return SignalType(v)


class SignalBasedOptimizer(AbstractOptimizer):
    """
    Portfolio Optimization Using Signals.
    
    Uses predictive signals (e.g., P/Sales, P/E ratios, momentum) to determine
    portfolio weights. The signals are transformed into weights using a flexible
    mapping function.
    
    Parameters
    ----------
    signal_type : SignalType, default=SignalType.LINEAR
        Type of signal transformation: LINEAR, SIGMOID, EXPONENTIAL
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
        config: Optional[SignalBasedOptimizerConfig] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.config = config or SignalBasedOptimizerConfig()

        if self.config.apply_diversification:
            self.diversifier = DiversificationOptimizer()

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
        mask = signals >= self.config.min_signal
        if not mask.any():
            return pd.Series(1.0 / len(signals), index=asset_names)

        signals = signals[mask]

        # Transform signals to weights
        weights = self._transform_signals(signals)

        # Optionally scale by expected returns
        if self.config.scale_by_returns:
            filtered_mu = ds_mu[mask]
            weights = weights * np.maximum(filtered_mu, 0)

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(weights), index=weights.index)

        # Apply diversification if requested
        if self.config.apply_diversification and len(weights) > 1:
            filtered_cov = df_cov.loc[mask, mask]
            weights = self.diversifier.allocate(weights, filtered_cov)

        # Map back to full universe
        full_weights = pd.Series(0.0, index=asset_names)
        full_weights[weights.index] = weights.values

        return full_weights

    def _transform_signals(self, signals: pd.Series) -> pd.Series:
        """Transform signals to weights using specified method."""
        match self.config.signal_type:
            case SignalType.LINEAR:
                # Linear transformation with power
                weights = np.power(np.maximum(signals, 0), self.config.signal_power)
            case SignalType.SIGMOID:
                # Sigmoid transformation
                weights = 1.0 / (1.0 + np.exp(-self.config.signal_power * signals))
            case SignalType.EXPONENTIAL:
                # Exponential transformation
                weights = np.exp(self.config.signal_power * signals)

            case _:
                raise NotImplementedError(f"Unsupported signal type: {self.config.signal_type}")

        return weights

    @property
    def name(self) -> str:
        return "SignalBasedOptimizer"