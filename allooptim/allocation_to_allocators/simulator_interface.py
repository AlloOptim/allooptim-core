from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from allooptim.optimizer.allocation_metric import LMoments


class AbstractObservationSimulator(ABC):
    """
    Abstract base class for generating synthetic observations with asset name preservation.

    Observation simulators are used in Monte Carlo Cross-Simulation (MCOS) to generate
    multiple realizations of expected returns and covariance matrices for robust
    backtesting and uncertainty quantification.

    All simulators maintain asset name consistency between input and output, enabling
    seamless integration with pandas-based optimizers and workflows.

    Examples:
        Basic simulation with asset name preservation:

        >>> simulator = MuCovObservationSimulator(mu, cov, n_observations=252)
        >>> mu_sim, cov_sim, prices_sim, time_sim, l_moments_sim = simulator.get_sample()
        >>> print("Original assets:", mu.index.tolist())
        >>> print("Simulated assets:", mu_sim.index.tolist())
        >>> # Asset names are preserved across simulations

        Use in MCOS workflow:

        >>> for i in range(n_simulations):
        ...     mu_sim, cov_sim, prices_sim, time_sim, l_moments_sim = simulator.get_sample()
        ...     weights = optimizer.allocate(mu_sim, cov_sim, prices_sim, time_sim, l_moments_sim)
        ...     # All operations preserve asset names
    """

    mu: np.ndarray
    cov: np.ndarray
    historical_prices: pd.DataFrame
    n_observations: int

    @abstractmethod
    def get_sample(self) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, datetime, LMoments]:
        """
        Generate synthetic observations of expected returns, covariance matrix, prices, time, and L-moments.

        Implements various statistical techniques to simulate realistic market scenarios
        while preserving asset name information throughout the process.

        Returns:
            Tuple containing:
            - mu_sim: Expected returns as pandas Series with asset names as index
            - cov_sim: Covariance matrix as pandas DataFrame with asset names as index/columns
            - prices_sim: Historical prices as pandas DataFrame with asset names as columns
            - time_sim: Timestamp for the simulation
            - l_moments_sim: L-moments for higher-order risk modeling

        Note:
            Asset names in the returned objects match those from the original input data.
            This enables seamless chaining with optimizers and other pandas-aware components.
        """
        pass

    @abstractmethod
    def get_ground_truth(self) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, datetime, LMoments]:
        """
        Return the ground truth parameters from the full historical dataset.

        This provides the baseline parameters computed from all available historical data,
        useful for error estimation and comparison with simulated samples.

        Returns:
            Tuple containing:
            - mu_gt: Ground truth expected returns as pandas Series with asset names as index
            - cov_gt: Ground truth covariance matrix as pandas DataFrame with asset names as index/columns
            - prices_gt: Full historical prices as pandas DataFrame with asset names as columns
            - time_gt: Timestamp representing the end of the historical period
            - l_moments_gt: Ground truth L-moments for higher-order risk modeling
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of this optimizer. The name will be displayed in the MCOS results DataFrame.
        """
        pass
