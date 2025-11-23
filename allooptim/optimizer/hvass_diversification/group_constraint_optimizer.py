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
from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.hvass_diversification.diversify_optimizer import (
    DiversificationOptimizer,
)


class GroupConstraintsOptimizerConfig(BaseModel):
    """Configuration for Group Constraints optimizer.

    This config holds parameters for portfolio optimization with group constraints
    including group weight limits and diversification settings.
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    group_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    asset_to_group: Optional[Dict[str, str]] = None
    optimize_within_groups: bool = True


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
        config: Optional[GroupConstraintsOptimizerConfig] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(display_name)
        self.config = config or GroupConstraintsOptimizerConfig()

        if self.config.optimize_within_groups:
            self.diversifier = DiversificationOptimizer()

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
        if not self.config.group_constraints or not self.config.asset_to_group:
            return self.diversifier.allocate(ds_mu, df_cov)

        # Group assets
        groups = {}
        for asset in asset_names:
            group = self.config.asset_to_group.get(asset, 'default')
            if group not in groups:
                groups[group] = []
            groups[group].append(asset)

        # Allocate weights respecting group constraints
        weights = pd.Series(0.0, index=asset_names)

        for group_name, group_assets in groups.items():
            min_weight, max_weight = self.config.group_constraints.get(
                group_name, (0.0, 1.0)
            )

            # Get expected returns and covariance for this group
            group_mu = ds_mu[group_assets]
            group_cov = df_cov.loc[group_assets, group_assets]

            # Optimize within group
            if self.config.optimize_within_groups and len(group_assets) > 1:
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

    @property
    def name(self) -> str:
        return "GroupConstraintsOptimizer"