from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


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
        >>> mu_sim, cov_sim = simulator.simulate()
        >>> print("Original assets:", mu.index.tolist())
        >>> print("Simulated assets:", mu_sim.index.tolist())
        >>> # Asset names are preserved across simulations

        Use in MCOS workflow:

        >>> for i in range(n_simulations):
        ...     mu_sim, cov_sim = simulator.simulate()
        ...     weights = optimizer.allocate(mu_sim, cov_sim)
        ...     # All operations preserve asset names
    """

    mu: np.ndarray
    cov: np.ndarray
    historical_prices: pd.DataFrame
    n_observations: int

    @abstractmethod
    def simulate(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate synthetic observations of expected returns and covariance matrix.

        Implements various statistical techniques to simulate realistic market scenarios
        while preserving asset name information throughout the process.

        Returns:
            Tuple containing:
            - mu_sim: Expected returns as pandas Series with asset names as index
            - cov_sim: Covariance matrix as pandas DataFrame with asset names as index/columns

        Note:
            Asset names in the returned objects match those from the original input data.
            This enables seamless chaining with optimizers and other pandas-aware components.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of this optimizer. The name will be displayed in the MCOS results DataFrame.
        """
        pass
