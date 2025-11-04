"""
Allocation Optimizers Testing

Tests for portfolio optimization algorithms weight constraints.
All optimizers must ensure 0 <= sum(weights) <= 1.02
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from allo_optim.optimizer.allocation_metric import LMoments
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.optimizer_list import OPTIMIZER_LIST

# Constants for test tolerances
MAX_PORTFOLIO_WEIGHT_SUM_TOLERANCE = 1.02


def get_all_optimizers() -> list[type[AbstractOptimizer]]:
    """Get all optimizer classes for testing"""
    return OPTIMIZER_LIST


@pytest.fixture
def sample_mu() -> pd.Series:
    asset_names = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    return pd.Series([0.001, 0.0012, 0.0008, 0.0009, 0.0011], index=asset_names)


@pytest.fixture
def sample_cov() -> pd.DataFrame:
    asset_names = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    cov_data = np.array(
        [
            [0.04, 0.01, 0.005, 0.002, 0.003],
            [0.01, 0.05, 0.002, 0.001, 0.004],
            [0.005, 0.002, 0.03, 0.001, 0.002],
            [0.002, 0.001, 0.001, 0.02, 0.003],
            [0.003, 0.004, 0.002, 0.003, 0.05],
        ]
    )
    return pd.DataFrame(cov_data, index=asset_names, columns=asset_names)


@pytest.fixture
def sample_l_moments() -> LMoments:
    # Create proper L-comoment matrices (n_assets x n_assets)
    n_assets = 5
    np.random.seed(42)

    # Generate symmetric positive definite matrices for L-comoments
    base_matrix = np.random.randn(n_assets, n_assets)
    lt_comoment_1 = (base_matrix + base_matrix.T) / 2 + n_assets * np.eye(n_assets)
    lt_comoment_2 = (base_matrix + base_matrix.T) / 2 + n_assets * np.eye(n_assets)
    lt_comoment_3 = (base_matrix + base_matrix.T) / 2 + n_assets * np.eye(n_assets)
    lt_comoment_4 = (base_matrix + base_matrix.T) / 2 + n_assets * np.eye(n_assets)

    return LMoments(
        lt_comoment_1=lt_comoment_1,
        lt_comoment_2=lt_comoment_2,
        lt_comoment_3=lt_comoment_3,
        lt_comoment_4=lt_comoment_4,
    )


@pytest.fixture
def sample_time() -> datetime:
    return datetime(2024, 1, 1)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Generate sample historical price data for optimizers that need it."""
    asset_names = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Generate some realistic price movements
    np.random.seed(42)
    initial_prices = [150, 2800, 400, 800, 3200]  # Starting prices
    price_data = {}

    for i, asset in enumerate(asset_names):
        prices = [initial_prices[i]]
        for _ in range(99):  # 99 more days
            # Random walk with small daily changes
            change = np.random.normal(0.001, 0.02)  # 0.1% avg daily return, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure prices don't go below $1
        price_data[asset] = prices

    return pd.DataFrame(price_data, index=dates)


@pytest.mark.parametrize("optimizer_class", get_all_optimizers())
def test_optimizer_weight_constraints(  # noqa: PLR0913
    optimizer_class: type[AbstractOptimizer],
    sample_mu: pd.Series,
    sample_cov: pd.DataFrame,
    sample_l_moments: LMoments,
    sample_time: datetime,
    sample_price_data: pd.DataFrame,
) -> None:
    """Test that all optimizers produce weights with sum between 0 and 1.02"""
    optimizer = optimizer_class()

    # Some optimizers require historical price data to be fitted first
    if hasattr(optimizer, "fit"):
        try:
            optimizer.fit(sample_price_data)
        except Exception:
            # If fit fails, some optimizers might not need it
            pass

    weights = optimizer.allocate(sample_mu, sample_cov, sample_price_data, sample_time, sample_l_moments)

    # Ensure weights is a pandas Series
    assert isinstance(weights, pd.Series)

    # Ensure weights have correct shape and asset names
    assert len(weights) == len(sample_mu)
    assert list(weights.index) == list(sample_mu.index)

    # Ensure all weights are non-negative
    assert (weights >= 0).all()

    # Ensure sum of weights is between 0 and 1.02 (inclusive)
    total_weight = weights.sum()
    assert 0 <= total_weight <= MAX_PORTFOLIO_WEIGHT_SUM_TOLERANCE
