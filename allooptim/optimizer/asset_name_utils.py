"""Asset Name Utilities for Pandas-based Optimizer Interface.

Provides validation and conversion functions for ensuring asset name consistency
between mu (pd.Series) and cov (pd.DataFrame) inputs in the optimizer interface.
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd


def validate_asset_names(mu: pd.Series, cov: pd.DataFrame) -> None:
    """Validate that asset names are consistent between mu and cov.

    Args:
        mu: Expected returns as pandas Series with asset names as index
        cov: Covariance matrix as pandas DataFrame with asset names as index/columns

    Raises:
        ValueError: If asset names are inconsistent or missing
        TypeError: If inputs are not pandas Series/DataFrame
    """
    if not isinstance(mu, pd.Series):
        raise TypeError(f"mu must be pd.Series, got {type(mu)}")

    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be pd.DataFrame, got {type(cov)}")

    # Check that cov is square
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square matrix, got shape {cov.shape}")

    # Check that cov index and columns are identical
    if not cov.index.equals(cov.columns):
        raise ValueError("cov.index must equal cov.columns (asset names)")

    # Check that mu.index matches cov.index
    if not mu.index.equals(cov.index):
        mu_assets = set(mu.index)
        cov_assets = set(cov.index)
        missing_in_cov = mu_assets - cov_assets
        missing_in_mu = cov_assets - mu_assets

        error_msg = []
        if missing_in_cov:
            error_msg.append(f"Assets in mu but not in cov: {sorted(missing_in_cov)}")
        if missing_in_mu:
            error_msg.append(f"Assets in cov but not in mu: {sorted(missing_in_mu)}")

        raise ValueError("Asset names mismatch between mu and cov. " + "; ".join(error_msg))

    # Check for duplicate asset names
    if mu.index.has_duplicates:
        raise ValueError(f"mu.index has duplicate asset names: {mu.index.duplicated().sum()} duplicates")

    if cov.index.has_duplicates:
        raise ValueError(f"cov.index has duplicate asset names: {cov.index.duplicated().sum()} duplicates")


def get_asset_names(mu: pd.Series = None, cov: pd.DataFrame = None) -> list[str]:
    """Extract asset names from mu or cov.

    Args:
        mu: Expected returns series (optional)
        cov: Covariance matrix DataFrame (optional)

    Returns:
        list of asset names

    Raises:
        ValueError: If neither mu nor cov is provided
    """
    if mu is not None:
        return mu.index.tolist()
    elif cov is not None:
        return cov.columns.tolist()
    else:
        raise ValueError("Either mu or cov must be provided")


def align_assets(
    mu: pd.Series, cov: pd.DataFrame, reorder: bool = True, fill_missing: bool = False
) -> tuple[pd.Series, pd.DataFrame]:
    """Align asset names between mu and cov, optionally reordering or filling missing assets.

    Args:
        mu: Expected returns series
        cov: Covariance matrix DataFrame
        reorder: If True, reorder to match asset order in cov
        fill_missing: If True, fill missing assets with default values (mu=0, cov=diagonal with small variance)

    Returns:
        tuple of (aligned_mu, aligned_cov) with consistent asset ordering

    Raises:
        ValueError: If assets cannot be aligned and fill_missing=False
    """
    mu_assets = set(mu.index)
    cov_assets = set(cov.index)

    if mu_assets == cov_assets:
        # Perfect match - just reorder if requested
        if reorder:
            asset_order = cov.index
            return mu.reindex(asset_order), cov
        else:
            return mu, cov

    # Find common assets
    common_assets = mu_assets & cov_assets

    if len(common_assets) == 0:
        raise ValueError("No common assets between mu and cov")

    if not fill_missing:
        # Use only common assets
        if len(common_assets) < len(mu_assets) or len(common_assets) < len(cov_assets):
            missing_mu = mu_assets - common_assets
            missing_cov = cov_assets - common_assets
            warnings.warn(
                f"Dropping assets - Missing in mu: {sorted(missing_mu)}, " f"Missing in cov: {sorted(missing_cov)}",
                stacklevel=2,
            )

        # Keep only common assets
        common_assets_list = sorted(common_assets)
        aligned_mu = mu.reindex(common_assets_list)
        aligned_cov = cov.reindex(index=common_assets_list, columns=common_assets_list)

        return aligned_mu, aligned_cov

    else:
        # Fill missing assets
        all_assets = sorted(mu_assets | cov_assets)

        # Extend mu with zeros for missing assets
        aligned_mu = mu.reindex(all_assets, fill_value=0.0)

        # Extend cov with small diagonal variance for missing assets
        aligned_cov = cov.reindex(index=all_assets, columns=all_assets, fill_value=0.0)

        # Fill diagonal elements for new assets with small variance
        default_variance = 0.01  # 1% variance for missing assets
        for asset in all_assets:
            if asset not in cov.index:
                aligned_cov.loc[asset, asset] = default_variance

        return aligned_mu, aligned_cov


def convert_numpy_to_pandas(
    mu: Union[np.ndarray, pd.Series], cov: Union[np.ndarray, pd.DataFrame], asset_names: list[str] = None
) -> tuple[pd.Series, pd.DataFrame]:
    """Convert numpy arrays to pandas Series/DataFrame with asset names.

    Args:
        mu: Expected returns as numpy array or pandas Series
        cov: Covariance matrix as numpy array or pandas DataFrame
        asset_names: list of asset names (required if inputs are numpy arrays)

    Returns:
        tuple of (pd.Series, pd.DataFrame) with asset names

    Raises:
        ValueError: If asset_names is required but not provided
    """
    # Handle mu conversion
    if isinstance(mu, pd.Series):
        mu_series = mu
        if asset_names is not None and not mu.index.equals(pd.Index(asset_names)):
            warnings.warn("Provided asset_names differ from mu.index, using mu.index", stacklevel=2)
    else:
        if asset_names is None:
            raise ValueError("asset_names required when mu is numpy array")
        mu_array = np.asarray(mu).flatten()
        if len(mu_array) != len(asset_names):
            raise ValueError(f"mu length ({len(mu_array)}) must match asset_names length ({len(asset_names)})")
        mu_series = pd.Series(mu_array, index=asset_names)

    # Handle cov conversion
    if isinstance(cov, pd.DataFrame):
        cov_df = cov
        if asset_names is not None:
            expected_index = pd.Index(asset_names)
            if not cov.index.equals(expected_index) or not cov.columns.equals(expected_index):
                warnings.warn(
                    "Provided asset_names differ from cov.index/columns, using cov index/columns", stacklevel=2
                )
    else:
        if asset_names is None:
            asset_names = mu_series.index.tolist()
        cov_array = np.asarray(cov)
        if cov_array.shape != (len(asset_names), len(asset_names)):
            raise ValueError(f"cov shape {cov_array.shape} must match ({len(asset_names)}, {len(asset_names)})")
        cov_df = pd.DataFrame(cov_array, index=asset_names, columns=asset_names)

    return mu_series, cov_df


def convert_pandas_to_numpy(mu: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert pandas Series/DataFrame to numpy arrays while preserving asset name order.

    Args:
        mu: Expected returns as pandas Series
        cov: Covariance matrix as pandas DataFrame

    Returns:
        tuple of (mu_array, cov_array, asset_names)
    """
    validate_asset_names(mu, cov)

    asset_names = mu.index.tolist()
    mu_array = mu.values
    cov_array = cov.values

    return mu_array, cov_array, asset_names


def create_weights_series(weights: np.ndarray, asset_names: list[str]) -> pd.Series:
    """Convert weight array to pandas Series with asset names.

    Args:
        weights: Portfolio weights as numpy array
        asset_names: list of asset names corresponding to weights

    Returns:
        Portfolio weights as pandas Series with asset names as index
    """
    weights_array = np.asarray(weights).flatten()
    if len(weights_array) != len(asset_names):
        raise ValueError(f"weights length ({len(weights_array)}) must match asset_names length ({len(asset_names)})")

    return pd.Series(weights_array, index=asset_names)


def validate_weights_series(weights: pd.Series, tolerance: float = 1e-6) -> None:
    """Validate that weights series is properly formatted for portfolio optimization.

    Args:
        weights: Portfolio weights as pandas Series
        tolerance: Numerical tolerance for sum-to-one constraint

    Raises:
        ValueError: If weights are invalid
    """
    if not isinstance(weights, pd.Series):
        raise TypeError(f"weights must be pd.Series, got {type(weights)}")

    if weights.isna().any():
        raise ValueError("weights contains NaN values")

    if not np.isfinite(weights).all():
        raise ValueError("weights contains infinite values")

    weights_sum = weights.sum()
    if abs(weights_sum - 1.0) > tolerance:
        raise ValueError(f"weights must sum to 1.0, got {weights_sum:.6f}")

    if weights.index.has_duplicates:
        raise ValueError("weights.index has duplicate asset names")


# Backward compatibility wrapper for numpy-based optimizers
class NumpyOptimizerAdapter:
    """Adapter to wrap numpy-based optimizers for use with pandas interface.

    This allows gradual migration of optimizers from numpy to pandas interface.
    """

    def __init__(self, numpy_optimizer):
        """Args.

        numpy_optimizer: Optimizer with numpy-based allocate method.
        """
        self.numpy_optimizer = numpy_optimizer

    def allocate(self, mu: pd.Series, cov: pd.DataFrame, **kwargs) -> pd.Series:
        """Adapt pandas inputs to numpy, call numpy optimizer, convert back to pandas."""
        # Convert to numpy
        mu_array, cov_array, asset_names = convert_pandas_to_numpy(mu, cov)

        # Call numpy-based optimizer
        weights_array = self.numpy_optimizer.allocate(mu_array, cov_array, **kwargs)

        # Convert back to pandas
        weights_series = create_weights_series(weights_array, asset_names)

        return weights_series

    @property
    def name(self) -> str:
        """Get the name of the wrapped numpy optimizer.

        Returns:
            Name of the optimizer, defaults to "NumpyAdapter" if not available
        """
        return getattr(self.numpy_optimizer, "name", "NumpyAdapter")
