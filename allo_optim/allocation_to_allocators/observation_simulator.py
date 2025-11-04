from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from allo_optim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allo_optim.optimizer.allocation_metric import LMoments, estimate_linear_moments


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

        # Pre-compute ground truth L-moments from full dataset
        returns_all = self.historical_prices_all.pct_change().dropna().values
        self.ground_truth_l_moments = estimate_linear_moments(returns_all)

    @property
    def name(self) -> str:
        """Name of this observation simulator."""
        return "MuCovPartialObservationSimulator"

    def get_sample(self) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, datetime, LMoments]:
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

        # Compute L-moments from the simulated returns
        # Only compute L-moments if we have sufficient observations
        if x_all.shape[0] >= 10:  # MIN_OBSERVATIONS from allocation_metric.py
            l_moments_sim = estimate_linear_moments(x_all)
        else:
            l_moments_sim = None

        # Use the end date of the partial period as the simulation time
        time_sim = self.historical_prices_all.index[int(mean_end)]

        return mu_sim_series, cov_sim_df, self.historical_prices, time_sim, l_moments_sim

    def get_ground_truth(self) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, datetime, LMoments]:
        """
        Return ground truth parameters computed from the full historical dataset.
        """
        mu_gt_series = pd.Series(self.mu, index=self.asset_names)
        cov_gt_df = pd.DataFrame(self.cov, index=self.asset_names, columns=self.asset_names)

        # Use the last date of the full historical period
        time_gt = self.historical_prices_all.index[-1]

        return mu_gt_series, cov_gt_df, self.historical_prices_all, time_gt, self.ground_truth_l_moments
