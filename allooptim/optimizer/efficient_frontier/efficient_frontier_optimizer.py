import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import sharpe_ratio

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import (
    LMoments,
    make_positive_definite,
)
from allooptim.optimizer.asset_name_utils import (
    convert_pandas_to_numpy,
    create_weights_series,
    get_asset_names,
    validate_asset_names,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class MaxSharpeOptimizerConfig(BaseModel):
    """Configuration parameters for maximum Sharpe ratio optimizer."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    # Configuration for fallback bounds when nonconvex optimization is needed
    min_weight_multiplier: float = 0.1  # min_weight = max(min_weight_multiplier / n_assets, 0.0)
    max_weight_multiplier: float = 10.0  # max_weight = min(max_weight_multiplier / n_assets, 1.0)
    min_positive_percentage: float = 0.05  # Minimum percentage of assets with positive expected returns


class MaxSharpeOptimizer(AbstractOptimizer):
    """Maximum Sharpe ratio optimizer using Modern Portfolio Theory with pandas interface.

    Implements the classic mean-variance optimization to find the portfolio with the highest
    Sharpe ratio (return per unit of risk). Based on Harry Markowitz's Nobel Prize-winning
    "Portfolio Selection" theory, this optimizer finds the tangency portfolio on the
    efficient frontier.

    Mathematical Formulation:
        Maximize: (μᵀw - r_f) / √(wᵀΣw)
        Subject to: Σwᵢ = 1, wᵢ ≥ 0

    Where:
        - μ: Expected returns vector (accessible via mu.index for asset names)
        - Σ: Covariance matrix (accessible via cov.columns for asset names)
        - w: Portfolio weights (returned with asset names as index)
        - r_f: Risk-free rate (assumed to be 0)

    Key Features:
        - Maximizes risk-adjusted returns (Sharpe ratio)
        - Uses pypfopt library for robust numerical optimization
        - Automatic asset name preservation throughout optimization
        - Long-only constraints (no short selling)
        - Handles numerical instabilities in covariance matrices

    Examples:
        Basic usage with asset name access:

        >>> optimizer = MaxSharpeOptimizer()
        >>> weights = optimizer.allocate(mu, cov)
        >>> # Access high-weight assets
        >>> top_holdings = weights.nlargest(3)
        >>> for asset, weight in top_holdings.items():
        ...     print(f"{asset}: {weight:.4f} ({weight*100:.1f}%)")
        AAPL: 0.4520 (45.2%)
        GOOGL: 0.3210 (32.1%)
        MSFT: 0.1850 (18.5%)

        Calculate portfolio risk-return metrics:

        >>> expected_return = (weights * mu).sum()
        >>> portfolio_vol = np.sqrt((weights * (cov @ weights)).sum())
        >>> sharpe_ratio = expected_return / portfolio_vol
        >>> print(f"Optimized Sharpe Ratio: {sharpe_ratio:.4f}")
    """

    def __init__(
        self,
        config: Optional[MaxSharpeOptimizerConfig] = None,
    ) -> None:
        """Initialize maximum Sharpe ratio optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
        self.config = config or MaxSharpeOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights to maximize Sharpe ratio.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Historical price data (unused)
            time: Current timestamp (unused)
            l_moments: L-moments (unused)

        Returns:
            Optimal portfolio weights as pandas Series
        """
        # Validate inputs
        validate_asset_names(ds_mu, df_cov)
        asset_names = get_asset_names(mu=ds_mu)
        n_assets = len(asset_names)

        # Convert to numpy for pypfopt
        mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)
        cov_array = make_positive_definite(cov_array)

        # If only a small percentage of assets have positive momentum, do not invest
        positive_assets = (mu_array > 0).mean()

        if positive_assets < self.config.min_positive_percentage:
            return create_weights_series(np.zeros(len(asset_names)), asset_names)

        try:
            ef = EfficientFrontier(mu_array, cov_array)
            ef.max_sharpe()
            weights_dict = ef.clean_weights()

            # Convert pypfopt output (dict with integer keys) back to array
            weights_array = np.array(list(weights_dict.values()))
            apply_nonconvex_fallback = False

        except Exception as e:
            logger.debug(f"MaxSharpe optimization failed: {e}. Proceed to non-convex fallback.")
            apply_nonconvex_fallback = True

        if apply_nonconvex_fallback:
            try:
                ef = EfficientFrontier(mu_array, cov_array)
                min_weight = max(self.config.min_weight_multiplier / n_assets, 0.0)
                max_weight = min(self.config.max_weight_multiplier / n_assets, 1.0)
                weights_dict = ef.nonconvex_objective(
                    sharpe_ratio,
                    objective_args=(ef.expected_returns, ef.cov_matrix),
                    constraints=[
                        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # sum to 1
                        {"type": "ineq", "fun": lambda w: w - min_weight},  # greater than min_weight
                        {"type": "ineq", "fun": lambda w: max_weight - w},  # less than max_weight
                    ],
                )
                weights_array = np.array(list(weights_dict.values()))

            except Exception as e:
                logger.error(f"Non-convex MaxSharpe optimization failed, too: {e}. Proceed using equal weights.")
                weights_array = np.ones(n_assets) / n_assets

        # Handle zero weights case
        if np.sum(weights_array) == 0:
            logger.error("MaxSharpe optimization resulted in zero weights, using equal weights instead.")
            weights_array = np.ones(n_assets) / n_assets
        else:
            weights_array = weights_array / np.sum(weights_array)

        return create_weights_series(weights_array, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the maximum Sharpe ratio optimizer.

        Returns:
            Optimizer name string
        """
        return "MaxSharpe"


class EfficientRiskOptimizerConfig(BaseModel):
    """Configuration parameters for efficient risk optimizer."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    min_target_risk: float = 0.15  # Minimum target risk
    risk_multiplier: float = 1.0  # Multiplier for diagonal risk calculation


class EfficientRiskOptimizer(AbstractOptimizer):
    """Optimizer based on the Modern Portfolio Theory pioneered by Harry Markowitz's paper 'Portfolio Selection'"""

    def __init__(self, config: Optional[EfficientRiskOptimizerConfig] = None) -> None:
        """Initialize efficient risk optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
        self.config = config or EfficientRiskOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights for efficient risk optimization.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Historical price data (unused)
            time: Current timestamp (unused)
            l_moments: L-moments (unused)

        Returns:
            Optimal portfolio weights as pandas Series
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        # Convert to numpy for pypfopt
        mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)
        cov_array = make_positive_definite(cov_array)

        ef = EfficientFrontier(mu_array, cov_array)

        diag_risk = max(
            self.config.min_target_risk, self.config.risk_multiplier * np.mean(np.diag(cov_array))
        )  # Use configurable target risk

        ef.efficient_risk(diag_risk)
        weights_dict = ef.clean_weights()

        weights_array = np.array(list(weights_dict.values()))
        if np.sum(weights_array) == 0:
            logger.error("EfficientRisk optimization resulted in zero weights, using equal weights instead.")
            n_assets = len(asset_names)
            weights_array = 1.0 / n_assets * np.ones(n_assets)
        else:
            weights_array = weights_array / np.sum(weights_array)

        return create_weights_series(weights_array, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the efficient risk optimizer.

        Returns:
            Optimizer name string
        """
        return "EfficientRisk"


class EfficientReturnOptimizerConfig(BaseModel):
    """Configuration parameters for efficient return optimizer."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    min_target_return: float = 0.0  # Minimum target return
    return_multiplier: float = 1.0  # Multiplier for mean return calculation


class EfficientReturnOptimizer(AbstractOptimizer):
    """Optimizer based on the Modern Portfolio Theory pioneered by Harry Markowitz's paper 'Portfolio Selection'"""

    def __init__(self, config: Optional[EfficientReturnOptimizerConfig] = None) -> None:
        """Initialize efficient return optimizer.

        Args:
            config: Configuration parameters for the optimizer. If None, uses default config.
        """
        self.config = config or EfficientReturnOptimizerConfig()

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        """Allocate portfolio weights for efficient return optimization.

        Args:
            ds_mu: Expected returns series with asset names as index
            df_cov: Covariance matrix DataFrame
            df_prices: Historical price data (unused)
            time: Current timestamp (unused)
            l_moments: L-moments (unused)

        Returns:
            Optimal portfolio weights as pandas Series
        """
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        # Convert to numpy for pypfopt
        mu_array, cov_array, _ = convert_pandas_to_numpy(ds_mu, df_cov)
        cov_array = make_positive_definite(cov_array)

        ef = EfficientFrontier(mu_array, cov_array)

        mean_return = max(self.config.min_target_return, self.config.return_multiplier * np.mean(mu_array))

        ef.efficient_return(mean_return)
        weights_dict = ef.clean_weights()

        weights_array = np.array(list(weights_dict.values()))
        if np.sum(weights_array) == 0:
            logger.error("EfficientReturn optimization resulted in zero weights, using equal weights instead.")
            n_assets = len(asset_names)
            weights_array = 1.0 / n_assets * np.ones(n_assets)
        else:
            weights_array = weights_array / np.sum(weights_array)

        return create_weights_series(weights_array, asset_names)

    @property
    def name(self) -> str:
        """Get the name of the efficient return optimizer.

        Returns:
            Optimizer name string
        """
        return "EfficientReturn"
