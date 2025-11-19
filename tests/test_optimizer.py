from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

import allooptim.optimizer.wikipedia.wiki_database as wiki_db
from allooptim.optimizer.allocation_metric import estimate_linear_moments
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer
from allooptim.optimizer.hierarchical_risk_parity.hrp_optimizer import HRPOptimizer
from allooptim.optimizer.nested_cluster.nco_optimizer import NCOSharpeOptimizer
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.optimizer_list import OPTIMIZER_LIST
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer

# Constants for test tolerances
WEIGHT_SUM_TEST_TOLERANCE = 0.01
MAX_PORTFOLIO_WEIGHT_SUM_TOLERANCE = 1.02
MIN_WEIGHT_TOLERANCE = -0.01


class TestMarkowitzOptimizer:
    def test_allocate(self, prices_df):
        mu = mean_historical_return(prices_df)  # Returns pandas Series
        cov = sample_cov(prices_df)  # Returns pandas DataFrame

        weights = MaxSharpeOptimizer().allocate(mu, cov)

        assert_array_almost_equal(
            weights.values,  # Extract numpy array for comparison
            np.array([0.326587, 0.052549, 0.0, 0.620864]),
        )

    def test_name(self):
        assert MaxSharpeOptimizer().name == "MaxSharpe"


class TestNCOOptimizer:
    def test_allocate_max_sharpe(self, prices_df):
        mu = mean_historical_return(prices_df)  # Returns pandas Series
        cov = sample_cov(prices_df)  # Returns pandas DataFrame

        weights = NCOSharpeOptimizer().allocate(mu, cov)
        assert_array_almost_equal(
            weights.values,  # Extract numpy array for comparison
            [0.330264, 0.049734, 0.0, 0.620002],
        )

    def test_allocate_min_variance(self, prices_df):
        cov = sample_cov(prices_df)  # Returns pandas DataFrame
        # For min variance, pass None for mu (NCO optimizer handles this)
        # or create matching Series with zeros
        mu = pd.Series([0.0] * len(cov.index), index=cov.index)

        weights = NCOSharpeOptimizer().allocate(ds_mu=mu, df_cov=cov)
        # NCO min variance should give reasonable weights (not exactly [0, 0.5, 0, 0.5])
        assert len(weights) == len(cov.index)
        assert abs(weights.sum() - 1.0) < WEIGHT_SUM_TEST_TOLERANCE

    def test_name(self):
        assert NCOSharpeOptimizer().name == "NCOSharpeOptimizer"


class TestHRPOptimizer:
    def test_allocate(self, prices_df):
        mu = mean_historical_return(prices_df)  # Returns pandas Series
        cov = sample_cov(prices_df)  # Returns pandas DataFrame

        weights = HRPOptimizer().allocate(mu, cov)
        # Check that weights are reasonable (sum to 1, positive)
        assert abs(weights.sum() - 1.0) < WEIGHT_SUM_TEST_TOLERANCE
        assert all(w >= 0 for w in weights.values)
        assert len(weights) == len(mu)

    def test_name(self):
        assert HRPOptimizer().name == "HRPOptimizer"


class TestRiskParityOptimizer:
    # Create pandas Series and DataFrame with asset names
    assets = ["Asset_0", "Asset_1", "Asset_2", "Asset_3"]
    mu = pd.Series([0.14, 0.12, 0.15, 0.07], index=assets)

    cov = pd.DataFrame(
        [
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52],
        ],
        index=assets,
        columns=assets,
    )

    def test_allocate(self):
        weights = RiskParityOptimizer().allocate(self.mu, self.cov)

        assert_almost_equal(weights.values, np.array([0.1954397, 0.2152156, 0.1626095, 0.4267352]))

    def test_allocate_custom_risk_budget(self):
        target_risk = np.array(
            [0.30, 0.30, 0.10, 0.30]
        )  # your risk budget percent of total portfolio risk (equal risk)

        optimizer = RiskParityOptimizer()
        optimizer.target_risk = target_risk
        weights = optimizer.allocate(self.mu, self.cov)
        assert_almost_equal(
            weights.values,
            np.array([0.195, 0.215, 0.163, 0.427]),
            decimal=3,
        )

    def test_name(self):
        assert RiskParityOptimizer().name == "RiskParityOptimizer"


@pytest.mark.parametrize("optimizer_class", OPTIMIZER_LIST)
def test_optimizers(optimizer_class, wikipedia_test_db_path):
    """Test that all optimizers in OPTIMIZER_LIST work correctly."""
    assert issubclass(optimizer_class, AbstractOptimizer)

    # Create sample data for testing
    assets = ["A", "B", "C"]
    mu = pd.Series([0.1, 0.12, 0.08], index=assets)
    cov = pd.DataFrame(
        [[0.001, 0.0001, 0.00005], [0.0001, 0.002, 0.00008], [0.00005, 0.00008, 0.003]], index=assets, columns=assets
    )

    # Create sample price data for optimizers that need it
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    price_data = np.random.randn(100, 3).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)

    # Calculate historical returns for L-moments estimation
    returns = prices.pct_change().dropna().values  # Shape: (n_observations, n_assets)
    l_moments = estimate_linear_moments(returns)

    # all optimizers should be instantiable and take the default config if the passed one is None
    optimizer = optimizer_class(config=None)

    # Special handling for WikipediaOptimizer - use test database
    if optimizer.name == "WikipediaOptimizer":
        original_db_path = wiki_db.DATABASE_PATH
        wiki_db.DATABASE_PATH = wikipedia_test_db_path

    # Skip optimizers that require special setup or are known to be broken
    skip_optimizers = [
        "LSTMOptimizer",  # Requires training data
        "MAMBAOptimizer",  # Requires training data
        "TCNOptimizer",  # Requires training data
    ]

    if optimizer.name in skip_optimizers:
        pytest.skip(f"Skipping {optimizer.name} as it requires special setup")

    # Check that fit and allocate don't raise warnings
    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        optimizer.fit(df_prices=prices)

        weights = optimizer.allocate(
            ds_mu=mu,
            df_cov=cov,
            df_prices=prices,
            time=datetime.now(),
            l_moments=l_moments,
        )

    # Filter out expected warnings (SLSQP doesn't use Hessian)
    filtered_warnings = [
        w for w in warning_list if "Method SLSQP does not use Hessian information" not in str(w.message)
    ]

    # Fail the test if any unexpected warnings were raised
    if filtered_warnings:
        warning_messages = [str(w.message) for w in filtered_warnings]
        pytest.fail(f"Optimizer {optimizer.name} raised unexpected warnings during fit/allocate: {warning_messages}")

    # Restore original database path for WikipediaOptimizer
    if optimizer.name == "WikipediaOptimizer":
        wiki_db.DATABASE_PATH = original_db_path

    # Check return type and basic properties
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(assets)
    assert weights.index.equals(pd.Index(assets))

    # Check that weights are reasonable (sum close to 1, but be lenient)
    weight_sum = weights.sum()
    assert weight_sum >= 0.0, f"Weights sum to {weight_sum}, expected non-negative sum"
    assert (
        weight_sum <= MAX_PORTFOLIO_WEIGHT_SUM_TOLERANCE
    ), f"Weights sum to {weight_sum}, expected sum <= {MAX_PORTFOLIO_WEIGHT_SUM_TOLERANCE}"
    assert all(w >= MIN_WEIGHT_TOLERANCE for w in weights.values), f"Negative weights found: {weights.values}"
    assert all(w <= 1.0 for w in weights.values), f"Weights greater than 1 found: {weights.values}"

    # Check that name is a string
    assert isinstance(optimizer.name, str)
    assert len(optimizer.name) > 0
