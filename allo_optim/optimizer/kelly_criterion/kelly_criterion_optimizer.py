"""
Kelly Criterion Portfolio Optimizer

This module implements the Kelly Criterion for portfolio optimization, which maximizes
the expected logarithm of wealth (geometric mean return) rather than the arithmetic mean.

Key Features:
    - Maximizes long-term growth rate through log-wealth optimization
    - Uses historical win rates to assess probability of positive returns
    - Implements fractional Kelly (e.g., half-Kelly) for practical risk control
    - Information-theoretic foundation based on Kelly's original 1956 paper
    - Differentiates from mean-variance by focusing on geometric mean

Theoretical Background:
    For a single asset: f* = (μ - r_f) / σ²
    For multi-asset portfolio: f = (1 + r_f) * Σ⁻¹ * (μ - r_f)

    Where:
        f* = optimal fraction to invest
        μ = expected return
        r_f = risk-free rate
        σ² = variance
        Σ = covariance matrix

    Fractional Kelly: f_final = kelly_fraction * f_optimal
    - Full Kelly (1.0): Maximum growth, highest volatility
    - Half Kelly (0.5): 75% of growth, 50% of volatility (recommended)
    - Quarter Kelly (0.25): Conservative, low volatility

References:
    - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
    - Thorp, E.O. (1997). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import LMoments
from allo_optim.optimizer.asset_name_utils import (
    convert_pandas_to_numpy,
    create_weights_series,
    get_asset_names,
    validate_asset_names,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class KellyCriterionOptimizerConfig(BaseModel):
    """Configuration for Kelly Criterion optimizer."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    allow_sell_all: bool = Field(
        default=True,
        description="Whether to enable selling all assets. If true, sum of weights is less than 1.0.",
    )

    lookback_window: int = Field(
        default=252,
        description="Number of periods to look back for win rate calculation (e.g., 252 trading days = 1 year)",
    )

    kelly_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Fraction of full Kelly to use (0.5 = half-Kelly, recommended for practical use)",
    )

    risk_free_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=0.1,
        description="Risk-free rate for Kelly formula (annualized)",
    )

    min_win_rate: float = Field(
        default=0.51,
        ge=0.0,
        le=1.0,
        description="Minimum win rate required to invest (must have positive edge)",
    )

    use_win_rate_adjustment: bool = Field(
        default=True,
        description="Whether to adjust Kelly weights by historical win rate",
    )

    min_sample_size: int = Field(
        default=30,
        ge=10,
        description="Minimum number of observations required for win rate calculation",
    )


class KellyCriterionOptimizer(AbstractOptimizer):
    """
    Kelly Criterion Portfolio Optimizer.

    Implements the Kelly Criterion for optimal portfolio sizing, maximizing the expected
    logarithm of wealth (geometric mean return). Unlike mean-variance optimization which
    maximizes the Sharpe ratio (arithmetic mean), Kelly Criterion explicitly accounts for
    the compounding nature of returns through log-wealth maximization.

    The optimizer uses historical win rates to estimate the probability of positive returns,
    providing an information-theoretic approach to position sizing. Fractional Kelly (e.g.,
    half-Kelly with kelly_fraction=0.5) is recommended for practical use to reduce volatility
    while maintaining most of the growth rate.

    Key Differentiators from Other Optimizers:
        - MaxSharpe: Kelly maximizes geometric mean, MaxSharpe maximizes arithmetic mean / std
        - MeanVariance: Kelly uses log-wealth, MV uses quadratic utility
        - RiskParity: Kelly sizes by growth rate, RP sizes by risk contribution
        - Momentum: Kelly uses probability theory, Momentum uses trend signals

    Examples:
        >>> config = KellyCriterionOptimizerConfig(kelly_fraction=0.5, lookback_window=252)
        >>> optimizer = KellyCriterionOptimizer(config)
        >>> optimizer.fit(df_prices)  # Calculate historical win rates
        >>> weights = optimizer.allocate(mu, cov)
        >>> print(f"Kelly weights: {weights}")
        AAPL: 0.25, GOOGL: 0.30, MSFT: 0.20, AMZN: 0.25

    Args:
        config: Configuration object with Kelly-specific parameters

    Attributes:
        win_rates: Historical win rate for each asset (fraction of positive returns)
        avg_gains: Average gain when return is positive (as fraction)
        avg_losses: Average loss when return is negative (as fraction)
    """

    def __init__(self, config: Optional[KellyCriterionOptimizerConfig] = None) -> None:
        self.config = config or KellyCriterionOptimizerConfig()
        self.win_rates: Optional[pd.Series] = None
        self.avg_gains: Optional[pd.Series] = None
        self.avg_losses: Optional[pd.Series] = None

    def fit(self, df_prices: Optional[pd.DataFrame] = None) -> None:
        """
        Calculate historical win rates and average gains/losses from price data.

        This method computes key statistics needed for Kelly optimization:
            - Win rate: fraction of periods with positive returns
            - Average gain: mean return when return > 0
            - Average loss: mean absolute return when return < 0

        Args:
            df_prices: DataFrame with datetime index and asset columns containing prices

        Note:
            If df_prices has fewer observations than min_sample_size, win rates will
            be set to None and the allocate() method will use only mu and cov.
        """
        if df_prices is None or len(df_prices) < self.config.min_sample_size:
            logger.warning(
                f"Not enough price history for Kelly win rate calculation "
                f"(need {self.config.min_sample_size}, got {len(df_prices) if df_prices is not None else 0})"
            )
            self.win_rates = None
            self.avg_gains = None
            self.avg_losses = None
            return

        # Use only the lookback window
        lookback = self.config.lookback_window
        df_prices_window = df_prices.iloc[-lookback:] if len(df_prices) > lookback else df_prices

        # Calculate returns
        returns = df_prices_window.pct_change().dropna()

        if len(returns) < self.config.min_sample_size:
            logger.warning(
                f"Not enough returns for Kelly calculation after dropna "
                f"(need {self.config.min_sample_size}, got {len(returns)})"
            )
            self.win_rates = None
            self.avg_gains = None
            self.avg_losses = None
            return

        # Calculate win rates (fraction of positive returns)
        self.win_rates = (returns > 0).sum() / len(returns)

        # Calculate average gains (only positive returns)
        self.avg_gains = returns[returns > 0].mean()
        # Replace NaN with small positive value if no gains exist
        self.avg_gains = self.avg_gains.fillna(0.01)

        # Calculate average losses (absolute value of negative returns)
        self.avg_losses = returns[returns < 0].abs().mean()
        # Replace NaN with small positive value if no losses exist
        self.avg_losses = self.avg_losses.fillna(0.01)

        logger.debug(
            f"Kelly Criterion fitted with {len(returns)} observations. "
            f"Win rates range: [{self.win_rates.min():.3f}, {self.win_rates.max():.3f}]"
        )

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        df_allocations: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """
        Calculate Kelly-optimal portfolio weights.

        The allocation process follows these steps:
        1. Validate inputs and extract asset names
        2. Calculate base Kelly weights: f = (1 + r_f) * Σ⁻¹ * (μ - r_f)
        3. If win rates available and use_win_rate_adjustment=True, adjust by win probability
        4. Apply fractional Kelly: f_final = kelly_fraction * f
        5. Handle negative weights and edge cases
        6. Normalize weights to sum to 1.0

        Args:
            ds_mu: Expected returns as pandas Series with asset names
            df_cov: Covariance matrix as pandas DataFrame
            df_prices: Optional price history (can trigger re-fit if needed)
            df_allocations: Optional previous allocations (not used by Kelly)
            time: Optional timestamp (not used by Kelly)
            l_moments: Optional L-moments (not used by Kelly)

        Returns:
            Portfolio weights as pandas Series with asset names, summing to 1.0

        Note:
            If an asset has win_rate < min_win_rate, its weight is set to 0 (no edge).
            Negative weights are clipped to 0 (long-only constraint).
        """
        # Validate inputs
        validate_asset_names(ds_mu, df_cov)
        asset_names = get_asset_names(mu=ds_mu)

        # Convert to numpy for calculation
        mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)

        # If we have price data but haven't fitted, fit now
        if df_prices is not None and self.win_rates is None:
            self.fit(df_prices)

        # Handle edge cases
        if len(asset_names) == 0:
            logger.warning("Kelly: No assets provided, returning empty weights")
            return create_weights_series(np.array([]), [])

        # Handle NaN or infinite values
        if np.any(np.isnan(mu_array)) or np.any(np.isinf(mu_array)):
            logger.warning("Kelly: NaN or inf in expected returns, returning zero weights")
            return create_weights_series(np.zeros(len(asset_names)), asset_names)

        if np.any(np.isnan(cov_array)) or np.any(np.isinf(cov_array)):
            logger.warning("Kelly: NaN or inf in covariance matrix, returning zero weights")
            return create_weights_series(np.zeros(len(asset_names)), asset_names)

        # Calculate excess returns (mu - risk_free_rate)
        excess_returns = mu_array - self.config.risk_free_rate

        # Check if covariance matrix is positive definite
        try:
            # Try Cholesky decomposition to check positive definiteness
            np.linalg.cholesky(cov_array)
        except np.linalg.LinAlgError:
            logger.warning("Kelly: Covariance matrix not positive definite, using pseudo-inverse")
            # Use Moore-Penrose pseudo-inverse for non-positive-definite matrices
            try:
                cov_inv = np.linalg.pinv(cov_array)
            except np.linalg.LinAlgError:
                logger.error("Kelly: Cannot invert covariance matrix, returning equal weights")
                equal_weights = np.ones(len(asset_names)) / len(asset_names)
                return create_weights_series(equal_weights, asset_names)
        else:
            # Matrix is positive definite, use regular inverse
            try:
                cov_inv = np.linalg.inv(cov_array)
            except np.linalg.LinAlgError:
                logger.error("Kelly: Cannot invert covariance matrix, returning equal weights")
                equal_weights = np.ones(len(asset_names)) / len(asset_names)
                return create_weights_series(equal_weights, asset_names)

        # Calculate base Kelly weights: f = (1 + r_f) * Σ⁻¹ * (μ - r_f)
        # This is the continuous-time Kelly formula for multiple assets
        kelly_weights = (1.0 + self.config.risk_free_rate) * cov_inv @ excess_returns

        # Apply win rate adjustment if available and enabled
        if self.config.use_win_rate_adjustment and self.win_rates is not None:
            # Ensure win_rates align with asset names
            win_rates_aligned = self.win_rates.reindex(asset_names, fill_value=0.5)
            avg_gains_aligned = self.avg_gains.reindex(asset_names, fill_value=0.01)
            avg_losses_aligned = self.avg_losses.reindex(asset_names, fill_value=0.01)

            # Calculate Win-Loss Ratio (WLR) = average_gain / average_loss
            wlr = avg_gains_aligned.values / avg_losses_aligned.values

            # Kelly adjustment factor: (p * WLR - q) / WLR where q = 1 - p
            # This comes from the binary Kelly formula: f* = (p*b - q) / b
            # For investments: f* = p/l - q/g, reformulated as (p*WLR - q) / WLR
            p = win_rates_aligned.values
            q = 1.0 - p
            kelly_adjustment = (p * wlr - q) / wlr

            # Apply minimum win rate filter
            kelly_adjustment = np.where(p >= self.config.min_win_rate, kelly_adjustment, 0.0)

            # Clip negative adjustments to 0 (no edge)
            kelly_adjustment = np.maximum(kelly_adjustment, 0.0)

            # Apply adjustment
            kelly_weights = kelly_weights * kelly_adjustment

            logger.debug(f"Kelly: Applied win rate adjustment. " f"Avg adjustment: {np.mean(kelly_adjustment):.3f}")

        # Apply fractional Kelly
        kelly_weights = self.config.kelly_fraction * kelly_weights

        # Clip negative weights to 0 (long-only constraint)
        kelly_weights = np.clip(kelly_weights, min=0.0, max=1.0)

        if np.sum(kelly_weights) > 1.0 or not self.config.allow_sell_all:
            kelly_weights = kelly_weights / np.sum(kelly_weights)

        return create_weights_series(kelly_weights, asset_names)

    @property
    def name(self) -> str:
        """Return optimizer name."""

        return "KellyCriterion"
