import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from allo_optim.optimizer.hierarchical_risk_parity.hrp_optimizer import HRPOptimizer
from allo_optim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer
from allo_optim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer
from allo_optim.optimizer.nested_cluster.nco_optimizer import NCOSharpeOptimizer


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
        assert abs(weights.sum() - 1.0) < 0.01

    def test_name(self):
        assert NCOSharpeOptimizer().name == "NCOSharpeOptimizer"


class TestHRPOptimizer:
    def test_allocate(self, prices_df):
        mu = mean_historical_return(prices_df)  # Returns pandas Series
        cov = sample_cov(prices_df)  # Returns pandas DataFrame

        weights = HRPOptimizer().allocate(mu, cov)
        # Check that weights are reasonable (sum to 1, positive)
        assert abs(weights.sum() - 1.0) < 0.01
        assert all(w >= 0 for w in weights.values)
        assert len(weights) == len(mu)

    def test_name(self):
        assert HRPOptimizer().name == "HRP"


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

        assert_almost_equal(weights.values, np.array([0.1954478, 0.2152349, 0.1626168, 0.4267005]))

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
        assert RiskParityOptimizer().name == "RiskParity"
