"""Allocation post-processing utilities for A2A orchestrators.

This module provides shared logic for applying allocation constraints
such as maximum active assets, concentration limits, and minimum active assets.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationConstraints:
    """Utility class for applying allocation constraints to portfolio weights."""

    @staticmethod
    def apply_max_active_assets(
        weights: pd.Series,
        n_max_active_assets: Optional[int],
        min_weight_threshold: float = 1e-6,
    ) -> pd.Series:
        """Apply maximum active assets constraint.

        Args:
            weights: Portfolio weights as pandas Series
            n_max_active_assets: Maximum number of assets with weights > 0.
                                Disabled if None.
            min_weight_threshold: Minimum weight threshold to consider an asset "active"

        Returns:
            Modified weights with constraint applied
        """
        if n_max_active_assets is None:
            return weights

        # Find assets with weights above threshold
        active_assets = weights[weights > min_weight_threshold]

        if len(active_assets) <= n_max_active_assets:
            return weights

        # Sort by weight (descending) and keep only top n_max_active_assets
        sorted_assets = active_assets.sort_values(ascending=False)
        top_assets = sorted_assets.head(n_max_active_assets)

        # Create new weights series with only top assets
        new_weights = pd.Series(0.0, index=weights.index)
        new_weights[top_assets.index] = top_assets.values

        # Renormalize to maintain total weight
        total_weight = new_weights.sum()
        if total_weight > 0:
            new_weights = new_weights / total_weight

        logger.debug(
            f"Applied max_active_assets constraint: {len(active_assets)} -> {len(top_assets)} assets"
        )

        return new_weights

    @staticmethod
    def apply_max_concentration(
        weights: pd.Series,
        max_asset_concentration_pct: Optional[float],
    ) -> pd.Series:
        """Apply maximum concentration constraint by clipping individual weights.

        Args:
            weights: Portfolio weights as pandas Series
            max_asset_concentration_pct: Maximum concentration for any single asset.
                                       Weights above this threshold get clipped.
                                       Disabled if None.

        Returns:
            Modified weights with constraint applied
        """
        if max_asset_concentration_pct is None:
            return weights

        # Clip weights above the threshold
        clipped_weights = weights.clip(upper=max_asset_concentration_pct)

        # Check if any weights were clipped
        if not clipped_weights.equals(weights):
            clipped_count = (weights > max_asset_concentration_pct).sum()
            logger.debug(
                f"Applied max_concentration constraint ({max_asset_concentration_pct:.1%}): "
                f"{clipped_count} assets clipped"
            )

            # Renormalize to maintain total weight
            total_weight = clipped_weights.sum()
            if total_weight > 0:
                clipped_weights = clipped_weights / total_weight

        return clipped_weights

    @staticmethod
    def apply_min_active_assets(
        weights: pd.Series,
        n_min_active_assets: Optional[int],
        min_weight_threshold: float = 1e-6,
    ) -> pd.Series:
        """Apply minimum active assets constraint.

        Args:
            weights: Portfolio weights as pandas Series
            n_min_active_assets: Minimum number of assets with weights > 0.
                                Disabled if None.
            min_weight_threshold: Minimum weight threshold to consider an asset "active"

        Returns:
            Modified weights with constraint applied
        """
        if n_min_active_assets is None:
            return weights

        # Find assets with weights above threshold
        active_assets = weights[weights > min_weight_threshold]

        if len(active_assets) >= n_min_active_assets:
            return weights

        # Need to activate more assets
        n_to_activate = n_min_active_assets - len(active_assets)

        # Find inactive assets (weights <= threshold)
        inactive_assets = weights[weights <= min_weight_threshold]

        if len(inactive_assets) == 0:
            logger.warning(
                f"Cannot satisfy min_active_assets constraint: only {len(active_assets)} "
                f"assets available, need {n_min_active_assets}"
            )
            return weights

        # Select assets to activate (prefer those with smallest non-zero weights)
        # Sort inactive assets by their current weight (ascending)
        sorted_inactive = inactive_assets.sort_values()
        assets_to_activate = sorted_inactive.head(min(n_to_activate, len(sorted_inactive)))

        # Distribute weight equally among assets to activate
        activation_weight = 0.01  # Small weight to activate assets
        total_activation_weight = activation_weight * len(assets_to_activate)

        # Reduce existing active weights proportionally to make room
        if total_activation_weight > 0:
            active_weight_sum = active_assets.sum()
            if active_weight_sum > 0:
                # Scale down active weights to make room for new assets
                scale_factor = (active_weight_sum - total_activation_weight) / active_weight_sum
                weights = weights.copy()
                weights[active_assets.index] *= scale_factor

                # Add activation weights
                weights[assets_to_activate.index] = activation_weight

                logger.debug(
                    f"Applied min_active_assets constraint: activated {len(assets_to_activate)} "
                    f"additional assets"
                )

        return weights

    @staticmethod
    def apply_all_constraints(
        weights: pd.Series,
        n_max_active_assets: Optional[int] = None,
        max_asset_concentration_pct: Optional[float] = None,
        n_min_active_assets: Optional[int] = None,
        min_weight_threshold: float = 1e-6,
    ) -> pd.Series:
        """Apply all allocation constraints in the correct order.

        Order of application:
        1. Max concentration (clipping)
        2. Max active assets (reduce by removing smallest)
        3. Min active assets (activate additional assets)

        Args:
            weights: Portfolio weights as pandas Series
            n_max_active_assets: Maximum number of active assets
            max_asset_concentration_pct: Maximum concentration per asset
            n_min_active_assets: Minimum number of active assets
            min_weight_threshold: Threshold for considering assets "active"

        Returns:
            Modified weights with all constraints applied
        """
        # Start with a copy to avoid modifying the original
        constrained_weights = weights.copy()

        # 1. Apply max concentration first (clipping individual weights)
        constrained_weights = AllocationConstraints.apply_max_concentration(
            constrained_weights, max_asset_concentration_pct
        )

        # 2. Apply max active assets (reduce number of active assets)
        constrained_weights = AllocationConstraints.apply_max_active_assets(
            constrained_weights, n_max_active_assets, min_weight_threshold
        )

        # 3. Apply min active assets (ensure minimum number of active assets)
        constrained_weights = AllocationConstraints.apply_min_active_assets(
            constrained_weights, n_min_active_assets, min_weight_threshold
        )

        # Final normalization to ensure weights sum to 1 (or less if partial investment allowed)
        total_weight = constrained_weights.sum()
        if total_weight > 0:
            constrained_weights = constrained_weights / total_weight

        return constrained_weights