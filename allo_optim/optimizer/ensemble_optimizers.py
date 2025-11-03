"""
Ensemble Optimizers Module

Contains optimizers that combine or aggregate results from multiple individual optimizers:
- A2AEnsembleOptimizer: Efficiently computes ensemble allocation from pre-computed allocations
- SPY500Benchmark: S&P 500 benchmark allocation

These optimizers are designed to work with the enhanced allocation framework and leverage
the df_allocations parameter for efficiency improvements.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import LMoments
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)


class EnsembleOptimizerConfig(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    # Ensemble optimizers don't need specific parameters currently


class A2AEnsembleOptimizer(AbstractOptimizer):
    def __init__(self) -> None:
        self.config = EnsembleOptimizerConfig()

    """
    Efficient Allocation-to-Allocators (A2A) ensemble optimizer.

    Instead of re-running all individual optimizers, this optimizer uses the df_allocations
    DataFrame containing pre-computed allocations from all individual optimizers and computes
    a simple mean average for the ensemble allocation.

    Key Features:
        - Efficient computation using pre-computed allocations
        - No re-computation of individual optimizer results
        - Simple equal-weight averaging of all optimizer allocations
        - Automatic fallback handling for failed optimizers
        - Asset name preservation throughout ensemble process

    Performance Benefits:
        - Eliminates redundant optimizer computations in A2A
        - Significantly faster execution for large optimizer lists
        - Maintains same ensemble logic with improved efficiency

    Example Usage:
        >>> # Individual optimizers run first, results collected in df_allocations
        >>> df_allocations = pd.DataFrame(
        ...     {
        ...         "AAPL": [0.25, 0.30, 0.20],  # Allocations from 3 optimizers
        ...         "GOOGL": [0.35, 0.25, 0.40],
        ...         "MSFT": [0.40, 0.45, 0.40],
        ...     },
        ...     index=["MaxSharpe", "RiskParity", "Momentum"],
        ... )
        >>> # A2A computes ensemble as mean of all allocations
        >>> a2a = A2AEnsembleOptimizer()
        >>> ensemble_weights = a2a.allocate(mu, cov, time, l_moments, df_allocations)
        >>> print(ensemble_weights)
        AAPL     0.250  # (0.25 + 0.30 + 0.20) / 3
        GOOGL    0.333  # (0.35 + 0.25 + 0.40) / 3
        MSFT     0.417  # (0.40 + 0.45 + 0.40) / 3
    """

    @property
    def name(self) -> str:
        return "A2A_Ensemble"

    def fit(self, df_prices: Optional[pd.DataFrame] = None) -> None:
        """No fitting needed for ensemble optimizer."""
        pass

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
        Compute efficient ensemble allocation from pre-computed optimizer allocations.

        Args:
            mu: Expected returns (used for asset names and fallback)
            cov: Covariance matrix (used for asset names)
            time: Current timestamp
            l_moments: L-moments (not used by ensemble)
            df_allocations: DataFrame with optimizer allocations (rows=optimizers, cols=assets)

        Returns:
            Ensemble weights as pandas Series with asset names as index
        """
        asset_names = ds_mu.index.tolist()
        n_assets = len(asset_names)

        # Check if we have pre-computed allocations
        if df_allocations is None or df_allocations.empty:
            logger.warning("No pre-computed allocations provided to A2A, using equal weights")
            return pd.Series(np.ones(n_assets) / n_assets, index=asset_names)

        # Validate allocations DataFrame structure
        if not all(asset in df_allocations.columns for asset in asset_names):
            logger.warning("Asset mismatch in df_allocations, using equal weights fallback")
            return pd.Series(np.ones(n_assets) / n_assets, index=asset_names)

        try:
            # Select only the assets we need (in case df_allocations has extra columns)
            allocations_subset = df_allocations[asset_names]

            # Compute mean allocation across all optimizers
            ensemble_weights = allocations_subset.mean(axis=0)  # Mean across optimizer rows

            # Handle NaN values (replace with 0)
            ensemble_weights = ensemble_weights.fillna(0.0)

            # Normalize weights to sum to 1
            weight_sum = ensemble_weights.sum()
            if weight_sum > 0:
                ensemble_weights = ensemble_weights / weight_sum
            else:
                logger.warning("All ensemble weights are zero, using equal weights fallback")
                ensemble_weights = pd.Series(np.ones(n_assets) / n_assets, index=asset_names)

            logger.info(f"A2A ensemble computed from {len(df_allocations)} optimizers")

            return ensemble_weights

        except Exception as e:
            logger.error(f"Error computing A2A ensemble: {e}, using equal weights fallback")
            return pd.Series(np.ones(n_assets) / n_assets, index=asset_names)


class SPY500Benchmark(AbstractOptimizer):
    def __init__(self) -> None:
        self.config = EnsembleOptimizerConfig()

    """
    S&P 500 benchmark optimizer that allocates 100% to SPY.

    Provides a simple benchmark allocation strategy for comparison with active
    optimization strategies. Always allocates 100% to SPY if available,
    otherwise falls back to equal weight allocation.

    Key Features:
        - Single asset allocation to SPY (S&P 500 ETF)
        - Automatic fallback to equal weights if SPY unavailable
        - No fitting or optimization required
        - Deterministic allocation independent of market conditions

    Example Usage:
        >>> spy_bench = SPY500Benchmark()
        >>> weights = spy_bench.allocate(mu, cov)
        >>> print(weights["SPY"])  # Should be 1.0 if SPY is available
        1.0
    """

    @property
    def name(self) -> str:
        return "SPY_Benchmark"

    def fit(self, df_prices: Optional[pd.DataFrame] = None) -> None:
        """No fitting needed for benchmark."""
        pass

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
        Allocate 100% to SPY if available, otherwise equal weights.

        Args:
            mu: Expected returns (used for asset names)
            cov: Covariance matrix (used for asset names)
            time: Current timestamp (not used)
            l_moments: L-moments (not used)
            df_allocations: Pre-computed allocations (not used by benchmark)

        Returns:
            Benchmark weights as pandas Series with asset names as index
        """
        asset_names = ds_mu.index.tolist()
        weights = pd.Series(np.zeros(len(asset_names)), index=asset_names)

        # Try to allocate 100% to SPY
        if "SPY" in asset_names:
            weights["SPY"] = 1.0
            logger.info("SPY benchmark: 100% allocation to SPY")
        else:
            # Fallback to equal weights if SPY not available
            weights = pd.Series(np.ones(len(asset_names)) / len(asset_names), index=asset_names)
            logger.warning("SPY not available, using equal weights fallback")

        return weights
