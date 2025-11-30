"""Allocation post-processing utilities for A2A orchestrators.

This module provides shared logic for applying allocation constraints
such as maximum active assets, concentration limits, and minimum active assets.
"""

import logging
import numpy as np
import pandas as pd

from allooptim.config.postprocessing_config import PostProcessingConfig

logger = logging.getLogger(__name__)


def apply_postprocessing_constraints(
    weights: pd.Series,
    postprocessing_config: PostProcessingConfig,
) -> pd.Series:
    """Apply configured allocation constraints to the given weights.

    Args:
        weights: Portfolio weights as pandas Series
        postprocessing_config: Configuration for post-processing constraints

    Returns:
        Modified weights with constraints applied
    """

    previous_total_weight = weights.sum()

    if not postprocessing_config.allow_leverage and previous_total_weight > 1.0:
        logger.warning(
            f"Total weight {previous_total_weight:.4f} exceeds 1.0 but leverage is not allowed. "
            f"Scaling down to 1.0."
        )
        weights = weights * (1.0 / previous_total_weight)
        previous_total_weight = 1.0

    if postprocessing_config.full_investment and previous_total_weight < 1.0:
        logger.warning(
            f"Total weight {previous_total_weight:.4f} is less than 1.0 but full investment is required. "
            f"Scaling up to 1.0."
        )
        weights = weights * (1.0 / previous_total_weight)
        previous_total_weight = 1.0

    weights = np.clip(
        weights, 0.0, postprocessing_config.max_asset_concentration_pct or 1.0
    )

    effective_n_min_assets = postprocessing_config.n_min_assets or 1
    effective_n_max_assets = postprocessing_config.n_max_assets or len(weights)

    n_quantile_assets = postprocessing_config.max_quantile_assets

    # given the cumulative weights, estimate the number of assets to reach max_quantile_pct_assets of previous_total_weight
    if n_quantile_assets is not None:
        sorted_weights = weights.sort_values(ascending=False)
        cumulative_weights = sorted_weights.cumsum()
        threshold_weight = weights.sum() * n_quantile_assets
        n_assets_to_keep = (cumulative_weights <= threshold_weight).sum()

        n_limit = min(effective_n_max_assets, n_assets_to_keep)

    n_limit = max(effective_n_min_assets, n_limit)
    logger.info(f"Keeping top {n_limit} assets.")

    normalized_keeped_weights = weights.iloc[:n_limit]
    if normalized_keeped_weights.sum() > 0:
        normalized_keeped_weights = (
            normalized_keeped_weights
            * previous_total_weight
            / normalized_keeped_weights.sum()
        )
    else:
        logger.warning(
            "All weights are zero after applying constraints, cannot normalize. "
        )

    weights.iloc[:n_limit] = normalized_keeped_weights
    weights.iloc[n_limit:] = 0.0

    return weights
