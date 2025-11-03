from typing import Tuple, Union

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

from allo_optim.simulator.simulator_interface import (
    AbstractObservationSimulator,
)


class MuCovRVSObservationSimulator(AbstractObservationSimulator):
    """
    Simulator for generating Monte Carlo observations of mean and covariance matrices.

    This is used for robust out-of-sample testing by simulating different market
    conditions and observation noise.
    """

    def __init__(self, mu: Union[np.ndarray, pd.Series], cov: Union[np.ndarray, pd.DataFrame], n_observations=10):
        """
        Initialize the simulator.

        Parameters:
        - mu: Expected returns vector (numpy array or pandas Series)
        - cov: Covariance matrix (numpy array or pandas DataFrame)
        - n_observations: Number of observations to use for estimation
        """
        # Handle pandas inputs and preserve asset names
        if isinstance(mu, pd.Series):
            self.asset_names = mu.index.tolist()
            self.mu = mu.values.astype(float)
        else:
            self.mu = np.array(mu, dtype=float)
            self.asset_names = [f"Asset_{i}" for i in range(len(self.mu))]

        if isinstance(cov, pd.DataFrame):
            if not hasattr(self, "asset_names"):
                self.asset_names = cov.index.tolist()
            self.cov = cov.values.astype(float)
        else:
            self.cov = np.array(cov, dtype=float)
            if not hasattr(self, "asset_names"):
                self.asset_names = [f"Asset_{i}" for i in range(self.cov.shape[0])]

        self.n_observations = n_observations
        self.n_assets = len(self.mu)

        # Validate inputs
        if self.cov.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"Covariance matrix shape {self.cov.shape} doesn't match mu length {self.n_assets}")

    def simulate(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate a simulated observation of mean and covariance.

        Returns:
        - mu_sim: Simulated mean Series with asset names
        - cov_sim: Simulated covariance DataFrame with asset names
        """
        # Generate simulated returns data
        # Use multivariate normal with the true parameters
        simulated_returns = multivariate_normal.rvs(mean=self.mu, cov=self.cov, size=self.n_observations)

        # Estimate mean and covariance from simulated data
        mu_sim = np.mean(simulated_returns, axis=0)
        cov_sim = np.cov(simulated_returns, rowvar=False)

        # Ensure positive definiteness by adding small regularization if needed
        eigenvals = np.linalg.eigvals(cov_sim)
        if np.any(eigenvals <= 0):
            # Add small regularization to make positive definite
            min_eigenval = np.min(eigenvals)
            regularization = max(1e-8, -min_eigenval + 1e-8)
            cov_sim += np.eye(self.n_assets) * regularization

        # Return as pandas objects with asset names
        mu_sim_series = pd.Series(mu_sim, index=self.asset_names)
        cov_sim_df = pd.DataFrame(cov_sim, index=self.asset_names, columns=self.asset_names)

        return mu_sim_series, cov_sim_df


class MuCovPartialObservationSimulator(AbstractObservationSimulator):
    def __init__(
        self,
        prices_df: pd.DataFrame,
        n_observations: int,
    ) -> None:
        self.historical_prices_all = prices_df.copy()
        self.historical_prices = prices_df.copy()

        self.n_observations = n_observations
        self.n_time_steps, self.n_assets = prices_df.shape
        self.asset_names = prices_df.columns.tolist()  # Preserve asset names

        self.n_min_length = 10
        self.historical_prices_all.bfill(inplace=True)

        self.mu = mean_historical_return(self.historical_prices_all).values
        self.cov = sample_cov(self.historical_prices_all).values

        assert not self.historical_prices_all.isna().any().any()

    def simulate(self) -> Tuple[pd.Series, pd.DataFrame]:
        x_all = np.zeros((self.n_observations, self.n_assets))
        mean_start = 0
        mean_end = 0

        for k in range(self.n_observations):
            start_end_index = np.random.choice(self.n_time_steps, 2, replace=False)
            start_end_index.sort()
            start = start_end_index[0]
            end = start_end_index[1]

            if end - start < self.n_min_length:
                start = max(0, start - self.n_min_length)
                end = min(self.n_time_steps - 1, end + self.n_min_length)

            mean_start += start / self.n_observations
            mean_end += end / self.n_observations

            partial_df = self.historical_prices_all.iloc[start:end, :]

            mu = mean_historical_return(partial_df).values
            cov = sample_cov(partial_df).values

            x_all[k, :] = np.random.multivariate_normal(mu, cov)

        self.historical_prices = self.historical_prices_all.iloc[int(mean_start) : int(mean_end), :]

        # Convert to pandas with asset names
        mu_sim = x_all.mean(axis=0).flatten()
        cov_sim = np.cov(x_all, rowvar=False)

        mu_sim_series = pd.Series(mu_sim, index=self.asset_names)
        cov_sim_df = pd.DataFrame(cov_sim, index=self.asset_names, columns=self.asset_names)

        return mu_sim_series, cov_sim_df


class MuCovLedoitWolfObservationSimulator(AbstractObservationSimulator):
    def __init__(self, mu: Union[np.ndarray, pd.Series], cov: Union[np.ndarray, pd.DataFrame], n_observations: int):
        # Handle pandas inputs and preserve asset names
        if isinstance(mu, pd.Series):
            self.asset_names = mu.index.tolist()
            self.mu = mu.values.astype(float)
        else:
            self.mu = np.array(mu, dtype=float)
            self.asset_names = [f"Asset_{i}" for i in range(len(self.mu))]

        if isinstance(cov, pd.DataFrame):
            if not hasattr(self, "asset_names"):
                self.asset_names = cov.index.tolist()
            self.cov = cov.values.astype(float)
        else:
            self.cov = np.array(cov, dtype=float)
            if not hasattr(self, "asset_names"):
                self.asset_names = [f"Asset_{i}" for i in range(self.cov.shape[0])]

        self.n_observations = n_observations

    @property
    def name(self) -> str:
        return "MuCovLedoitWolf"

    def simulate(self) -> Tuple[pd.Series, pd.DataFrame]:
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)

        mu_sim = x.mean(axis=0)
        cov_sim = LedoitWolf().fit(x).covariance_

        # Return as pandas objects with asset names
        mu_sim_series = pd.Series(mu_sim, index=self.asset_names)
        cov_sim_df = pd.DataFrame(cov_sim, index=self.asset_names, columns=self.asset_names)

        return mu_sim_series, cov_sim_df


class MuCovObservationSimulator(AbstractObservationSimulator):
    def __init__(self, mu: Union[np.ndarray, pd.Series], cov: Union[np.ndarray, pd.DataFrame], n_observations: int):
        # Handle pandas inputs and preserve asset names
        if isinstance(mu, pd.Series):
            self.asset_names = mu.index.tolist()
            self.mu = mu.values.astype(float)
        else:
            self.mu = np.array(mu, dtype=float)
            self.asset_names = [f"Asset_{i}" for i in range(len(self.mu))]

        if isinstance(cov, pd.DataFrame):
            if not hasattr(self, "asset_names"):
                self.asset_names = cov.index.tolist()
            self.cov = cov.values.astype(float)
        else:
            self.cov = np.array(cov, dtype=float)
            if not hasattr(self, "asset_names"):
                self.asset_names = [f"Asset_{i}" for i in range(self.cov.shape[0])]

        self.n_observations = n_observations

    @property
    def name(self) -> str:
        return "MuCov"

    def simulate(self) -> Tuple[pd.Series, pd.DataFrame]:
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)

        mu_sim = x.mean(axis=0)
        cov_sim = np.cov(x, rowvar=False)

        # Return as pandas objects with asset names
        mu_sim_series = pd.Series(mu_sim, index=self.asset_names)
        cov_sim_df = pd.DataFrame(cov_sim, index=self.asset_names, columns=self.asset_names)

        return mu_sim_series, cov_sim_df


class MuCovJackknifeObservationSimulator(AbstractObservationSimulator):
    def __init__(self, mu: Union[np.ndarray, pd.Series], cov: Union[np.ndarray, pd.DataFrame], n_observations: int):
        # Handle pandas inputs and preserve asset names
        if isinstance(mu, pd.Series):
            self.asset_names = mu.index.tolist()
            self.mu = mu.values.astype(float)
        else:
            self.mu = np.array(mu, dtype=float)
            self.asset_names = [f"Asset_{i}" for i in range(len(self.mu))]

        if isinstance(cov, pd.DataFrame):
            if not hasattr(self, "asset_names"):
                self.asset_names = cov.index.tolist()
            self.cov = cov.values.astype(float)
        else:
            self.cov = np.array(cov, dtype=float)
            if not hasattr(self, "asset_names"):
                self.asset_names = [f"Asset_{i}" for i in range(self.cov.shape[0])]

        self.n_observations = n_observations

    @property
    def name(self) -> str:
        return "MuCovJackknife"

    def simulate(self) -> Tuple[pd.Series, pd.DataFrame]:
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)

        idx = np.arange(len(x))
        # Use list comprehension and sum() builtin to avoid deprecated np.sum(generator)
        cov_matrices = [np.cov(x[idx != i], rowvar=False) for i in range(len(x))]
        cov_hat = sum(cov_matrices) / float(len(x))

        x_subsets = [x[idx != i] for i in range(len(x))]
        x_prime = sum(x_subsets) / float(len(x))

        mu_sim = x_prime.mean(axis=0).flatten()

        # Return as pandas objects with asset names
        mu_sim_series = pd.Series(mu_sim, index=self.asset_names)
        cov_sim_df = pd.DataFrame(cov_hat, index=self.asset_names, columns=self.asset_names)

        return mu_sim_series, cov_sim_df
